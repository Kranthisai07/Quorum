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

*State: schema and index in place (phase 1); reaper and heartbeat loop in phase 2.*

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

Exactly one wins. The loser gets SQLSTATE `40001`, retries the whole
transaction, and takes a different unit. Both sides are recorded in
`conflict_log` so the contention is visible.

*State: phase 2.*

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

**Missing, and known to be missing:** metrics export (Prometheus/OTel),
distributed tracing across agent → MCP → database, and alerting thresholds.

---

## 4. Correctness and testing

Concurrency bugs found by a judge are fatal; concurrency bugs found by our own
stress test are a slide. Policy: **every conflict path gets a deterministic
test.**

Current suite: 72 tests, split into pure tests (no database) and `integration`
tests, which skip rather than fail when no cluster is running. Integration tests
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
| N agents, M units: zero double-claims | stress test | 2 |
| The same stress test *fails* under `naive` mode | stress test | 2 |
| An expired lease is reclaimed exactly once | claim tests | 2 |
| A contradiction is detected and supersedes exactly one decision | semantic tests | 4 |
| A cascade re-queues every transitive dependent, or none | cascade tests | 5 |

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
| Cost control | Per-workspace caps on model calls — **not built** |
| Multi-tenancy | Every query is already scoped by `workspace_id`, including the ANN search; row-level authorisation is **not built** |

---

## 6. Scale, honestly

The claim is correctness under contention, not throughput. Quorum is designed
for tens of agents against thousands of work units — the scale at which
coordination is hard and the failures are interesting.

Known limits, unmeasured at the time of writing:

- Every claim is a serializable transaction against a small hot set of pending
  units. Beyond some agent count, contention on that hot set becomes the
  bottleneck; sharded claim ordering would be the fix.
- The semantic pre-check adds an ANN query plus a possible model call to every
  decision write. Decision volume, not unit volume, is what would bite first.
- The cascade walks dependencies in one transaction. A pathological graph would
  make that transaction long-running, which increases its own abort risk.

Each of these has a shape and a fix. None of them is measured yet, and this
document will say so until they are.
