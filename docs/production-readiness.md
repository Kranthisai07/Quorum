# Production readiness

What it would take to run Quorum for real, what already holds, and what is
honestly still missing. Items marked **planned** name the phase that delivers
them; nothing here is claimed as done that is not done.

---

## 1. Failure modes, and what happens

### An agent dies mid-claim

The single most important failure, because it is the one that deadlocks naive
systems.

A claim is a **lease**, not a lock: `claimed_by` plus `claim_expires_at`, kept
alive by a heartbeat on `agent_sessions.heartbeat_at`. A dead agent stops
heartbeating; its lease expires; the reaper flips the unit back to `pending` and
another agent claims it. The `version` bump means a zombie agent that wakes up
and tries to write its result is rejected rather than silently overwriting
newer work.

The expiry scan is served by a partial index —
`INDEX (claim_expires_at) WHERE status = 'claimed'` — so reaping does not scan
the workspace.

*State: **implemented** (phase 2). Proven by `test_dead_agent_loses_no_work_and_blocks_nobody`,
which SIGKILLs a real worker process holding a lease, asserts the unit is *not*
stolen while the lease is still live, then asserts it is reclaimed and finished
by another agent once the lease lapses — with the reclaim recorded in
`conflict_log`.*

### A CockroachDB node dies

Nothing is lost and nothing stops. Ranges are replicated; a surviving replica
takes over leadership; in-flight transactions on the failed node abort with a
retryable error, and `run_serializable` replays them. Agents see a brief spike in
`40001` retries, which `TxnStats` reports, and continue.

This is demonstrated rather than asserted: phase 9 kills a node with `ccloud`
mid-run and shows the run completing with zero lost work.

*State: client-side retry with exponential backoff and full jitter is
implemented (`db.run_serializable`); the demo kill is phase 9.*

### Two agents claim the same unit

Exactly one wins. The loser either aborts with SQLSTATE `40001` and replays, or
— more often — blocks on the `FOR UPDATE` lock and finds the row taken when it
is let through. Both shapes are detected and recorded in `conflict_log`.

*State: **implemented** (phase 2). 200 iterations x 8 agents x 16 units: 3,200
claims, 0 double-claims, 4,216 recorded conflicts, one unit lost by seven agents
at once. The identical harness in `naive` mode double-claims 1,596 times in 25
iterations and logs nothing.*

### Two agents reach contradictory conclusions

The second write is caught by an ANN pre-check before it commits, classified by
Bedrock, and — if contradictory — reconciled, with the loser marked `superseded`
and dependent work invalidated.

*State: phase 4.*

### A cascade fails half-way

It cannot commit half-way. The transitive walk and every `stale` flip are one
serializable transaction: it commits whole or aborts whole.

*State: phase 5.*

### The database is unreachable

Connections carry `connect_timeout=10`, so an unreachable cluster fails fast
with a clear error instead of hanging. (This was found the hard way: `localhost`
resolves to `::1` first on Windows, and every connection stalled on IPv6 before
falling back. The local profile now pins the IPv4 loopback.)

*State: implemented.*

### A migration is edited after being applied

Rejected. Each applied migration is recorded with a checksum; a changed file
raises rather than silently diverging from the deployed schema.

*State: implemented.*

### A seed fails part-way

Impossible to observe. The workspace row, all work units, and all dependency
edges commit in one transaction. A workspace with units but no edges would give
the cascade an incomplete graph to walk — it would report success while leaving
stale work in place — so this has a test from phase 1
(`test_failed_seed_leaves_nothing_behind`).

*State: implemented and tested.*

---

### Amazon Bedrock is unreachable or misconfigured

Two endpoints, two failure modes, and they are reported separately because they
fail independently: Claude on the Messages API (`bedrock-mantle`) and Titan on
`bedrock-runtime` `InvokeModel`. `quorum bedrock check` probes each and names
which one broke, including the case that matters most before Phase 4 — an
embedding width that disagrees with the `VECTOR(1024)` column. That is caught at
the health check and again on every finding write, rather than surfacing as an
opaque INSERT failure in the middle of contradiction detection.

A missing credential is not a crash: the `stub` backend is a first-class path,
so the entire engine (including the 200-iteration stress suite) runs offline and
deterministically with `QUORUM_LLM_BACKEND=stub`.

*State: **implemented, but not verified against a live AWS account.** The
machine this was built on has no AWS credentials, so both Bedrock paths have
been exercised only up to the auth boundary — clients construct, requests are
shaped, and both fail with credential errors rather than import or API-shape
errors. `quorum bedrock check` exists precisely so that verification is one
command rather than a debugging session, and it must be run before Phase 4
depends on embeddings.*

### A model returns something unusable

A response with no `<migrated>` block fails the unit rather than storing
garbage: `MigrationError` marks the work unit `failed` with the reason recorded
on its spec, and the agent moves on. A safety refusal (`stop_reason: "refusal"`,
which arrives as HTTP 200) is checked before `content` is read, so a decline
cannot be mistaken for an empty migration.

*State: implemented (phase 3).*

## 2. Access control

**Now (local dev).** Insecure single node bound to `127.0.0.1`, no
authentication. Appropriate for a laptop, disqualifying anywhere else.

**Cloud profile.**

- CockroachDB Cloud with `sslmode=verify-full`; credentials only via
  environment, never in the repo. `.env` is gitignored; only `.env.example` is
  committed, and it contains no real values.
- The application connects as a role restricted to the `quorum` database with
  `SELECT`/`INSERT`/`UPDATE` on its tables — no `DROP`, no cluster settings. DDL
  runs as a separate migration role.
- Lambda workers use an execution role granting exactly: `bedrock:InvokeModel`
  on the two model IDs in use, `s3:PutObject`/`GetObject` on the artifact bucket
  prefix, and Secrets Manager read for the database URL. No wildcards.
- Agents reach shared memory for **reads** through the CockroachDB Cloud Managed
  MCP Server, which is audited and safe by default. Write paths that need
  isolation stay in the application layer, where the transaction boundary and
  retry policy are explicit and reviewable.
- Connection strings are redacted in every CLI output path (`_redact`) so a
  password never reaches a terminal or a screenshot.

*State: local implemented; cloud roles are phase 8.*

---

## 3. Observability

**Structured logging.** JSON on stderr from the first commit, one object per
event, with `event`, `logger`, `level`, `pid`, and typed fields. Ships to
CloudWatch unchanged in the Lambda profile. A `log_context` block tags every
record inside an agent's lifecycle with its workspace and session id, so a
single run can be reconstructed by filtering on one field.

A logging call must never take down a coordination run, so `extra` keys that
collide with `LogRecord` attributes (`name`, `module`, `args` — all natural
domain fields) are folded into the payload instead of raising `KeyError`.

**Transaction metrics.** `TxnStats` counts attempts, retries, commits, and
failures per transaction label. Retry rate under contention is the headline
number: it is the visible price of serializability, and it is reported rather
than suppressed.

**The conflict log is the product's own telemetry.** Every claim conflict,
semantic contradiction, and invalidation cascade is a row with the agents
involved, a JSONB detail payload, a resolution, and detection/resolution
timestamps. The dashboard reads this table directly.

**MCP audit log.** Agent reads through the Managed MCP Server are audited, and
that audit stream is surfaced in the dashboard (phase 6/7).

**Cluster observability.** The CockroachDB DB Console is available at
`127.0.0.1:8080` locally and in CockroachDB Cloud for the cloud profile:
transaction retries, contention hotspots, and hot ranges come for free.

### Two documented failure modes, hit and fixed

Both are recorded here rather than quietly patched, because they are the same
mistake wearing different clothes: **a check that is too narrow to see the case
that actually happens.**

**Blocked-then-stale contention.** The first contention detector watched only
for SQLSTATE `40001`, on the assumption that a loser aborts. Under
`SELECT ... FOR UPDATE` it usually does not — it *blocks* on the lock and finds
the row already taken when CockroachDB hands it over, committing cleanly with no
error at all. The detector therefore reported **zero conflicts** on a workspace
where eight agents were fighting over every row. Nothing was broken; the
evidence was simply invisible. Both shapes are now detected, and
`test_contention_is_real_and_recorded` explicitly does *not* assert on retry
count, because doing so is what hid the problem.

**Instrumentation that manufactured its own findings.** The fix for the above
was an unlocked peek at the intended unit — placed, at first, inside the claim
transaction. That left a read at a timestamp the winner's commit invalidated, so
the transaction could not refresh and aborted. It generated ~1.9 retries per
claim *that existed only because it was watching*, and then reported them as
evidence of contention. Moved outside the transaction, identical detection costs
**zero** aborts (1,848 → 0 on the same workload) and 1.22× throughput instead of
2.63×. `test_detection_does_not_manufacture_aborts` guards the regression.

The same shape as the swallowed `feature.vector_index.enabled` exception: a
narrow check that passes for the wrong reason. It is worth assuming there is a
third one not yet found.

**Bugs this discipline has already caught**, listed because they are the
argument for the discipline: a `SELECT ... FOR UPDATE` contention detector that
only watched for 40001 aborts and therefore logged *zero* conflicts under heavy
contention; a process-global log context that attributed every concurrent
agent's work to the last one registered; a `timeout_seconds or default` that
turned an explicit `0` into 30 seconds; and a crashed worker that reported a
clean run with zero claims instead of an error.

**Missing, and known to be missing:** metrics export (Prometheus/OTel),
distributed tracing across agent → MCP → database, and alerting thresholds.

---

## 4. Correctness and testing

Concurrency bugs found by a judge are fatal; concurrency bugs found by our own
stress test are a slide. Policy: **every conflict path gets a deterministic
test.**

Current suite: 167 tests, split into pure tests (no database) and `integration`
tests, which skip rather than fail when no cluster is running. Stress depth is
tunable with `QUORUM_STRESS_ITERATIONS` (default 200). Integration tests
run against a separate `quorum_test` database, so running `pytest` never
destroys a seeded demo workspace.

| Property | Test | Phase |
|---|---|---|
| Seed is atomic; a failed seed leaves nothing | `test_failed_seed_leaves_nothing_behind` | 1 ✅ |
| Dependency edges resolve to real units | `test_dependency_edges_point_at_real_units` | 1 ✅ |
| Decomposition rejects dangling edges, duplicates, self-deps | `TestDecompositionValidation` | 1 ✅ |
| Vector index exists, is cosine, is workspace-scoped | `test_decisions_has_a_cosine_vector_index` | 1 ✅ |
| Embedding width matches the configured model | `test_embedding_width_matches_configured_model` | 1 ✅ |
| Non-retryable errors are not swallowed by the retry loop | `test_non_retryable_errors_propagate` | 1 ✅ |
| N agents, M units: zero double-claims (200 iterations) | `test_never_double_claims` | 2 ✅ |
| Contention was real, not accidental serialization | `test_contention_is_real_and_recorded` | 2 ✅ |
| The same harness *fails* under `naive` mode | `test_the_same_harness_produces_opposite_verdicts` | 2 ✅ |
| `naive` races unaided, so the barrier is not the cause | `test_double_claims_without_any_help` | 2 ✅ |
| `naive` logs zero conflicts while corrupting the workspace | `test_never_notices_its_own_conflicts` | 2 ✅ |
| A SIGKILLed agent loses no work and blocks nobody | `test_dead_agent_loses_no_work_and_blocks_nobody` | 2 ✅ |
| A heartbeating agent keeps its lease | `test_a_live_agent_keeps_its_lease` | 2 ✅ |
| An expired lease is reclaimed, re-versioned, and logged | `TestExpiredLeases`, `TestReaper` | 2 ✅ |
| A zombie agent cannot overwrite the takeover | `test_the_original_agent_cannot_overwrite_the_takeover` | 2 ✅ |
| A contradiction is detected and supersedes exactly one decision | semantic tests | 4 |
| A cascade re-queues every transitive dependent, or none | cascade tests | 5 |
| A malformed model response fails the unit, not the workspace | `TestResponseParsing` | 3 ✅ |
| Findings are embedded and searchable through the vector index | `test_findings_are_embedded_on_write` | 3 ✅ |
| An embedding-width mismatch is caught before the INSERT | `test_a_dimension_mismatch_is_caught_before_the_insert` | 3 ✅ |
| Agents never write into the vendored fixture | `test_the_vendored_fixture_is_never_modified` | 3 ✅ |
| A result ref is versioned, so a redo cannot overwrite the original | `test_version_is_part_of_the_key` | 3 ✅ |
| The stub's similarity is lexical, and is documented as such | `test_stub_similarity_is_lexical_not_semantic` | 3 ✅ |

Static checks: `ruff` with `E, F, I, UP, B, SIM, ANN, RUF, S, ARG, PL` — including
the security rules — clean across the repo. Type hints throughout.

---

## 5. Operational surface

| Concern | Position |
|---|---|
| Schema changes | Versioned SQL migrations with checksums; a migration may opt out of a transaction (`-- quorum:no-transaction`) for index backfills |
| Rollback | Forward-only. Failed deployments are fixed by a new migration, not by mutating an applied one |
| Setup reproducibility | One command from clone: the CockroachDB binary is downloaded and pinned to a known version (`v26.2.5`), not assumed present |
| Configuration | Environment-driven, resolved once, one variable per axis; local defaults are a complete working setup |
| Secrets | Never in the repo. `.env` gitignored, `.env.example` committed with placeholders only |
| Fixture isolation | Agents never write into `fixtures/`; results go to the artifact store, so repeated runs leave the tree pristine |
| Backpressure | Bedrock throttling needs a token-bucket limiter shared across workers — **not built** |
| Cost control | Per-workspace caps on model calls — **not built**. Token usage per unit is recorded on the worker report, so the input exists |
| Model failure isolation | A refusal or malformed response fails one work unit, never the run |
| Observability cost | Conflict detection is 1.22x throughput and switchable (`QUORUM_CONFLICT_DETECTION`). Sampling it under load is a config change, not a code change |
| Multi-tenancy | Every query is already scoped by `workspace_id`, including the ANN search; row-level authorisation is **not built** |

---

## 6. Scale, honestly

The claim is correctness under contention, not throughput. Quorum is designed
for tens of agents against thousands of work units — the scale at which
coordination is hard and the failures are interesting.

Known limits, unmeasured at the time of writing:

- Every claim is a serializable transaction against a small hot set of pending
  units, and deterministic `ORDER BY target` deliberately points every agent at
  the same row. Measured with 8 agents over 16 units: 293 claims/sec unsafe,
  49.7 safe (5.9x, the cost of correctness), 40.7 with the conflict feed on
  (a further 1.22x, the cost of the evidence, and switchable). Losers block on
  the lock rather than aborting, so the safe path runs at **zero** retries --
  the bottleneck is lock queueing on the hot row, not transaction restarts.
  Randomised or sharded claim ordering would trade contention visibility for
  throughput.
- The semantic pre-check adds an ANN query plus a possible model call to every
  decision write. Decision volume, not unit volume, is what would bite first.
- The cascade walks dependencies in one transaction. A pathological graph would
  make that transaction long-running, which increases its own abort risk.

Each of these has a shape and a fix. None of them is measured yet, and this
document will say so until they are.
