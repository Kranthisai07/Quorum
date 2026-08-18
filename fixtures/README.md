# Fixtures

Real code, vendored so the demo shows real files changing. Quorum is a
coordination engine, not a migration tool — but a coordination engine with a toy
task proves nothing, so the reference task operates on an actual open-source
repository.

## `docker-py/`

| | |
|---|---|
| Upstream | https://github.com/docker/docker-py |
| Commit | `afc6d1ee308e78b908b96a94298c37fa8c465588` (2026-07-10) |
| License | Apache License 2.0 — see `docker-py/LICENSE`, retained unmodified (the same license Quorum itself uses) |
| Vendored | Source tree unmodified; `.git` and `tests/` removed (see below) |


Upstream's `tests/` directory is **not** vendored. It ships SSH private keys and
TLS certificates as test fixtures (`tests/ssh/config/`,
`tests/unit/testdata/certs/`) — harmless upstream, but this repository does not
carry private keys of any kind, and GitHub push protection correctly rejects
them. Quorum only ever scans `package_roots: ["docker"]`, so nothing in the test
tree was used. Every file under `docker/` is byte-for-byte upstream.

Chosen for the reference task (`tasks/requests-to-httpx.json`) because it has
the shape the coordination problem needs:

- **Enough files to contend over.** 24 source files carry `requests`-specific
  idioms, so N agents against M units is a real race rather than a staged one.
- **Genuine cross-file decisions.** `docker/errors.py` maps `requests`
  exceptions for the entire codebase, and `docker/transport/*.py` subclasses
  `HTTPAdapter` four different ways. Whether to keep an adapter shim or move to
  an `httpx` transport is a single decision that twenty files depend on — and
  two agents working different files will reach it independently. That is the
  semantic conflict, occurring naturally.
- **A real import graph.** Dependency edges come from parsing actual imports, so
  the invalidation cascade follows how the code is genuinely coupled.

Agents never write into `fixtures/`. Migration output is written to the artifact
store (`artifacts/` locally, S3 in the cloud profile), so the fixture stays
pristine across runs and `git status` stays clean.

## Adding another fixture

Decomposition is pluggable (`quorum.decompose`). A new fixture needs a vendored
tree here, a signal table in `quorum/decompose/code_migration.py` if the source
library differs, and a task spec in `tasks/`. Nothing in the coordination engine
changes.
