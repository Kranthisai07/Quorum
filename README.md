# Quorum

**A shared-memory coordination layer for concurrent AI agents, built on CockroachDB.**

Multiple agents work one large task at the same time against a single shared
memory. The hard part is not the agents — it is that they **contend**:

1. They grab the same work.
2. They reach contradictory conclusions independently, neither aware of the other.
3. One agent's late discovery invalidates work another agent already finished.

Quorum resolves all three using CockroachDB's serializable transactions and
distributed vector index as the coordination primitive.

> ## An empty conflict feed is not evidence of a calm system.
>
> That is the whole problem. When concurrent agents corrupt shared memory, the
> corruption does not announce itself — there is no exception, no failed
> request, no red line in a dashboard. Two agents migrate the same file and one
> result silently wins. Two agents adopt contradictory conventions and both
> decisions sit there marked `active`. Nothing is logged, because nothing
> noticed.
>
> That is why it survives in production. Quorum's job is to make the failure
> *visible* first, and prevented second.
>
> The measurement below makes the point twice over. Run the unsafe path and it
> double-claims 1,596 times while logging **zero** conflicts. Run the safe path
> with its instrumentation switched off and it logs **zero** conflicts too —
> while eight agents fight over every single row. Same empty feed. Completely
> different systems.

> **The thesis:** remove CockroachDB and this system does not degrade — it
> produces corrupt, contradictory output. The database is not storage here.
> It is the concurrency control.
>
> That claim is testable, and Quorum ships the test: every workspace runs in
> `safe` or `naive` mode, and `quorum compare` runs the same task, same agents,
> same seed, in both.

---

## Status

Built in phases. This is what currently runs end to end:

| Phase | Scope | State |
|---|---|---|
| 1 | Foundation: schema, migrations, decomposition, seeding, local cluster | **done** |
| 2 | Claim engine: leases, heartbeats, expiry reclaim, contention log | **done** |
| 3 | Real agents on Amazon Bedrock | **done** (unverified against a live AWS account) |
| 4 | Semantic conflict: embeddings, vector search, reconciliation | next |
| 5 | Invalidation cascade | planned |
| 6 | CockroachDB Cloud Managed MCP Server for agent reads | planned |
| 7 | Dashboard | planned |
| 8 | Cloud profile: CockroachDB Cloud, Lambda, S3 | planned |
| 9 | Demo harness, including a `ccloud` node kill | planned |

Phase 2 proves conflict #1 under contention. Conflicts #2 and #3 land in
phases 4 and 5, and this table will say `planned` until they do.

---

## The three conflicts

### 1. Claim conflict — two agents grab the same work unit

Resolved by claiming inside a **serializable transaction** with `SELECT … FOR
UPDATE`, a lease (`claimed_by`, `claim_expires_at`), and a heartbeat. An expired
lease is reclaimable, so an agent that dies mid-claim cannot deadlock the
workspace. Every contended claim is written to `conflict_log`, because
contention that is merely *handled* is invisible, and invisible contention
convinces nobody.

**Proven, not asserted.** `quorum stress` runs this scenario hundreds of times
in both modes. 8 agents, 16 units, deterministic ordering so every agent goes
for the *same* row:

| | `safe` | `naive` | `naive` (no barrier) |
|---|---|---|---|
| Iterations | 200 | 25 | 50 |
| Claims | 3,200 | 1,996 | 3,988 |
| **Double-claims** | **0** | **1,596** | **3,188** |
| Rounds with duplicates | 0 / 200 | 25 / 25 | 50 / 50 |
| Most agents on one unit | 1 | 8 | 8 |
| Most agents losing one race | 7 | — | — |
| Conflicts logged | 4,216 | **0** | **0** |
| Claims/sec | 44.8 | 178.9 | 165.7 |

Naive mode double-claims 1,596 times and logs nothing, because it never finds
out. Safe mode hands every unit to exactly one agent and records all 4,216
races it resolved — including one row that seven agents lost simultaneously.

The barrier column holds every naive agent at its select-then-update window so
the race is deterministic on demand, which is useful for a live demo. The third
column is the same run *without* it: the bug is entirely naive mode's own, and
the barrier only removes the luck.

#### What safety actually costs

A single "safe is 7× slower" number invites the obvious question — slower
because of *what*? Two separable things, with very different standing
([docs/stress-cost-breakdown.json](docs/stress-cost-breakdown.json)):

| | claims/sec | vs. naive |
|---|---|---|
| `naive` — unsafe | 293.3 | 1.0× |
| `safe` — correctness only | 49.7 | **5.9× slower** |
| `safe` + conflict logging | 40.7 | 7.2× slower |

**5.9× buys correctness** and is not optional. The remaining **1.22× buys the
conflict feed** and is: `QUORUM_CONFLICT_DETECTION=false` turns it off, or it
could be sampled under load. Correctness is identical either way — only the
evidence disappears.

That last row is worth dwelling on, because it is the thesis in miniature. With
detection off, safe mode is *just as contended* — eight agents still fight over
every row — and its conflict feed is empty. Silence is the default state of a
system under contention, whether or not that contention is being handled
correctly. Only deliberate instrumentation tells the two apart.

Reproduce with
`quorum stress <workspace> --agents 8 --iterations 200 --mode safe|naive`
and `python scripts/benchmark_claim_cost.py`. Raw reports:
[docs/stress-safe.json](docs/stress-safe.json),
[docs/stress-naive.json](docs/stress-naive.json).

### 2. Semantic conflict — two agents reach contradictory conclusions

Agent A decides "standardise the transport layer on `httpx`". Agent B, three
files away, decides "keep the `requests` adapter for the unix socket". Neither
knows about the other, and no keyword match connects those two sentences.

Every decision is embedded on write. Before it commits, an **ANN query over the
distributed vector index** finds near-neighbours in the same workspace and
scope; anything above threshold is classified by Bedrock as `agrees` /
`contradicts` / `unrelated`. On `contradicts`, one decision wins, the loser is
marked `superseded`, and the work built on it is invalidated.

This is the non-negotiable justification for a vector index: `LIKE` cannot tell
that those two sentences are the same argument.

### 3. Invalidation cascade — a late finding invalidates finished work

`unit_deps` is walked transitively from the invalidated node; dependents flip to
`stale`, are re-queued, and get a `version` bump. The **entire cascade commits
in one serializable transaction**, because a half-applied cascade — some
dependents re-queued, others still marked `done` — is exactly the corruption
being claimed against.

---

## Architecture

```mermaid
flowchart TB
    subgraph agents["Agent workers (local process / AWS Lambda)"]
        A1["agent 1"]
        A2["agent 2"]
        A3["agent N"]
    end

    subgraph crdb["CockroachDB — the coordination primitive"]
        WU["work_units<br/><i>lease + version</i>"]
        DEP["unit_deps<br/><i>cascade edges</i>"]
        DEC["decisions<br/><i>VECTOR(1024) + cosine index</i>"]
        FIN["findings"]
        CL["conflict_log<br/><b>the headline artifact</b>"]
    end

    subgraph aws["AWS"]
        BR["Bedrock<br/>reasoning + embeddings"]
        S3["S3 / local fs<br/>result artifacts"]
    end

    MCP["CockroachDB Cloud<br/>Managed MCP Server"]
    DASH["Dashboard<br/><i>polls the DB</i>"]

    A1 & A2 & A3 -->|"claim: SERIALIZABLE + FOR UPDATE"| WU
    A1 & A2 & A3 -->|"read context (audited)"| MCP
    MCP --> crdb
    A1 & A2 & A3 -->|"embed + classify"| BR
    A1 & A2 & A3 -->|"write result"| S3
    A1 & A2 & A3 -->|"decision pre-check: ANN query"| DEC
    DEC -->|"contradiction"| CL
    WU --> DEP
    DEP -->|"transactional cascade"| WU
    FIN -->|"invalidates"| DEP
    crdb --> DASH
```

Full detail in [docs/architecture.md](docs/architecture.md); operational
behaviour in [docs/production-readiness.md](docs/production-readiness.md).

---

## CockroachDB: which features, and what the agents do with them

| CockroachDB capability | What Quorum does with it |
|---|---|
| **`SERIALIZABLE` isolation** (the default) | Claims, seeds, and invalidation cascades run as one transaction each. Anomalies abort with SQLSTATE `40001` rather than committing quietly. |
| **Client-side retry on `40001`** | `run_serializable` replays the whole unit of work with exponential backoff and full jitter, and counts every retry (`TxnStats`) as an observable cost of correctness, not an error. |
| **`SELECT … FOR UPDATE`** | Serialises two agents racing for the same work unit, so exactly one wins the lease and the loser is logged as a contended claim. |
| **`VECTOR(1024)` column type** | Stores Bedrock Titan embeddings for every decision and finding. |
| **Distributed vector index (C-SPANN)** | `CREATE VECTOR INDEX … (workspace_id, embedding vector_cosine_ops)` — an ANN search over prior decisions runs before every decision commits. The `workspace_id` prefix scopes the search to one workspace instead of filtering after the fact. |
| **Partial index** | `INDEX (claim_expires_at) WHERE status = 'claimed'` lets the lease reaper find expired claims without scanning every unit. |
| **`JSONB`** | Work unit specs and conflict details stay schemaless where the domain plugs in, indexed where the engine needs them. |
| **`UUID[]`** | `conflict_log.agents` records exactly who was involved in each conflict. |
| **Lock-wait vs. abort** | Under `FOR UPDATE`, losers *block on the lock* rather than aborting with 40001 — a correct, heavily contended run has **zero** retries. Quorum detects both shapes, because a detector that only watches for aborts reports zero conflicts under heavy contention: correct behaviour, invisible evidence. |
| **Autocommit path (deliberately)** | `naive` mode uses no transaction and no retry, giving the demo a control group that fails the way ordinary agent memory fails. |
| **CockroachDB Cloud Managed MCP Server** | Agent *read* paths go through MCP (audited, safe by default). Writes that need isolation stay in the application layer — see the honesty note below. |
| **`ccloud`** | Kills a node mid-run in the demo, to show agents continuing with zero lost work. |

### The MCP split, stated plainly

Agent **reads** go through the CockroachDB Cloud Managed MCP Server: audited,
safe by default, and the right way for an agent to query shared memory.

Agent **writes** that require transactional guarantees do **not**. A claim is
`BEGIN; SELECT … FOR UPDATE; UPDATE; COMMIT` with an explicit retry loop on
`40001` — an isolation contract, not a query. Routing it through a generic tool
layer would mean giving up the control that makes it correct.

Read via MCP, write via controlled transactions. That is the honest split, and
it is a better engineering answer than pretending everything goes through MCP.

---

## AWS services, and how they are used

| Service | Role |
|---|---|
| **Amazon Bedrock — Claude** | Agent reasoning (the migration itself), and in Phase 4 the judge that classifies two near-neighbour decisions as `agrees` / `contradicts` / `unrelated`. Reached through the **Messages API** endpoint (`bedrock-mantle.{region}.api.aws`) via `AnthropicBedrockMantle`. |
| **Amazon Bedrock — Titan** | Embeddings (Titan Text Embeddings V2, 1024-d, matching the `VECTOR(1024)` column). Reached through **`bedrock-runtime` `InvokeModel`** via boto3 — a different endpoint with different auth. |
| **AWS Lambda** | Agent workers in the cloud profile. The worker entrypoint is Lambda-shaped from the start; the local runner invokes the same handler in a thread or a subprocess. |
| **Amazon S3** | Migration artifacts (unified diffs). `work_units.result_ref` holds the key, versioned so a redo after a lease expiry cannot overwrite the original. Locally this is a directory. |

**Two Bedrock endpoints, not one.** Claude and Titan do not share a client, an
endpoint, or an auth path, and their model-ID conventions differ —
`anthropic.claude-opus-5` has no revision suffix, `amazon.titan-embed-text-v2:0`
does. `quorum bedrock check` probes each independently and reports them
separately, because a green reasoning check tells you nothing about whether
embeddings work.

```
$ quorum bedrock check
backend  bedrock
  region        us-east-1
  reasoning     ok      anthropic.claude-opus-5
                via Messages API (bedrock-mantle)
  embeddings    ok      amazon.titan-embed-text-v2:0
                via boto3 bedrock-runtime InvokeModel
  dimensions    1024 vs VECTOR(1024) in schema  match
```

That last line is the one that matters before Phase 4: an embedding width that
disagrees with the schema is caught here, not in the middle of contradiction
detection.

Local development requires none of them: `QUORUM_LLM_BACKEND=stub`,
`QUORUM_ARTIFACT_BACKEND=local`, `QUORUM_RUNNER=local`.

---

## Setup

Requirements: **Python 3.11+**. That is all — CockroachDB is downloaded on
demand into `.tools/` (gitignored), so there is no Docker requirement and
nothing to install by hand.

```bash
git clone <this repo> && cd Quorum

# Windows
powershell -ExecutionPolicy Bypass -File scripts/bootstrap.ps1

# macOS / Linux
./scripts/bootstrap.sh
```

The bootstrap script creates a virtualenv, installs the package, starts a local
single-node cluster, applies migrations, and seeds a workspace from the vendored
`docker-py` repository. Doing it by hand is four commands:

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
quorum db up          # download + start local CockroachDB
quorum migrate        # apply versioned schema migrations
quorum seed           # decompose fixtures/docker-py, seed a workspace
```

Configuration is environment-driven. Copy `.env.example` to `.env` to change
anything; the defaults are a complete working local setup. **No secrets belong
in this repo.**

---

## Run

```bash
quorum version                       # resolved profile and backends
quorum db status                     # is the local cluster up?

quorum decompose --units             # what the task decomposes into, no writes
quorum seed --name demo --mode safe  # seed a workspace
quorum seed --name demo-naive --mode naive

quorum workspaces                    # every workspace and its unit counts
quorum status demo                   # units, decisions, conflicts

quorum run demo --agents 8           # run stub agents until the workspace drains
quorum agents demo                   # live sessions and what they hold
quorum conflicts demo                # the conflict feed
quorum reap demo                     # return expired leases to the pool

quorum bedrock check                 # probe both Bedrock endpoints separately
quorum findings demo --invalidating  # discoveries that affect other work

quorum stress demo --agents 8 --iterations 200 --mode safe
quorum stress demo-naive --agents 8 --iterations 25 --mode naive --barrier

quorum migrate --status              # which migrations are applied
quorum reset --yes                   # drop and rebuild the database
quorum db down                       # stop the local cluster
```

The DB Console is at <http://127.0.0.1:8080> while the local cluster is running.

### Tests and lint

```bash
pytest                    # 167 tests; integration tests skip if no cluster is up
pytest -m "not integration"
ruff check .
```

Integration tests run against `quorum_test` on the same cluster, so running the
suite never destroys a seeded demo workspace.

---

## The task being coordinated

The reference task migrates [docker-py](https://github.com/docker/docker-py)
(vendored under `fixtures/`, Apache-2.0) off `requests` onto `httpx`. It
decomposes into **16 work units** across **20 real import-graph dependency
edges** and **8 decision scopes**, several shared by four or more files — which
is exactly the overlap that produces semantic conflict.

The engine is domain-agnostic. Decomposition is a plugin
(`quorum.decompose.Decomposer`); code migration is the reference
implementation, not a hardcoded assumption. See
[fixtures/README.md](fixtures/README.md).

---

## Repository layout

```
src/quorum/
  llm.py               Bedrock (two clients) and the deterministic stub backend
  migration.py         the work itself: prompt, parse, diff, finding
  artifacts.py         result storage: local filesystem or S3
  findings.py          agent discoveries, embedded on write
  claims.py            the claim engine: leases, contention, reclaim
  sessions.py          agent registration, heartbeats, liveness
  conflicts.py         the conflict log
  worker.py            Lambda-shaped agent entrypoint (stub agent for now)
  runner.py            local execution: threads, or killable processes
  stress.py            the contention harness behind the numbers above
  config.py            environment-driven settings, local vs cloud
  logging.py           structured JSON logging
  db.py                pooling, serializable + autocommit paths, retry accounting
  localdb.py           local single-node CockroachDB lifecycle
  migrations/          versioned .sql migrations and their runner
  decompose/           pluggable task decomposition (code migration reference impl)
  workspace.py         atomic seeding, lookup, summaries
  cli.py               the `quorum` command
tasks/                 task specs
fixtures/              vendored real repositories to migrate
docs/                  architecture, production readiness
tests/                 pure tests + integration tests
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).

The vendored fixture is also Apache-2.0 and keeps its own upstream license
notice at `fixtures/docker-py/LICENSE`, unmodified.
