"""The agent worker entrypoint.

Lambda-shaped from the start: :func:`handler` takes an event dict and returns a
JSON-serialisable dict, so the local runner, a subprocess, and AWS Lambda all
execute the same code path. Phase 8 changes where it runs, not what it is.

The agent loop is: claim -> read context -> migrate -> write finding + artifact
-> complete. Two work modes share it:

* `migrate` -- the real thing. Reads the file, calls the configured backend,
  writes a unified diff to the artifact store, records a finding.
* `sleep` -- Phase 2's stub, kept because the stress harness needs work that
  costs a known amount of time and nothing else.

The claim engine was proven correct against `sleep` before a model was allowed
anywhere near it, so that a failed run is never ambiguous between a coordination
bug and a bad model response.
"""

from __future__ import annotations

import json
import random
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from quorum import claims, findings, sessions
from quorum.claims import ClaimedUnit, StaleClaimError
from quorum.config import Settings, get_settings
from quorum.llm import LLMBackend, get_backend
from quorum.logging import configure_logging, get_logger, log_context
from quorum.migration import MigrationError, migrate_unit, store_result
from quorum.workspace import resolve_workspace

log = get_logger(__name__)


@dataclass
class WorkerReport:
    """What one agent did. Returned from the handler and aggregated by the runner."""

    agent: str
    session_id: str
    workspace_id: str
    mode: str
    claimed: list[str] = field(default_factory=list)
    completed: int = 0
    failed: int = 0
    stale_writes: int = 0
    txn_retries: int = 0
    max_retries_single_claim: int = 0
    contended: int = 0
    findings_recorded: int = 0
    invalidating_findings: int = 0
    changed_lines: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    duration_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def handler(event: dict[str, Any], context: object | None = None) -> dict[str, Any]:  # noqa: ARG001
    """Run one agent until the workspace is drained or its quota is reached.

    Event fields:
        workspace_id   required, uuid or workspace name
        agent_name     defaults to a generated name
        mode           safe | naive; defaults to the workspace setting
        max_units      stop after this many claims (default: unlimited)
        work_seconds   stub work duration per unit (default 0.01)
        seed           makes stub work durations reproducible
        claim_only     claim without completing -- used by the stress harness
        lease_seconds  override the claim lease
        work_mode      migrate (default) | sleep
    """
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)

    workspace = resolve_workspace(str(event["workspace_id"]), settings)
    workspace_id: uuid.UUID = workspace["id"]
    mode = str(event.get("mode") or workspace["mode"])
    agent_name = str(event.get("agent_name") or f"agent-{uuid.uuid4().hex[:6]}")

    repo_root = _repo_root(workspace, settings)
    session = sessions.register(workspace_id, agent_name, settings)
    report = WorkerReport(
        agent=agent_name,
        session_id=str(session.id),
        workspace_id=str(workspace_id),
        mode=mode,
    )

    started = time.monotonic()
    with log_context(workspace_id=workspace_id, session_id=session.id, agent=agent_name):
        try:
            _run_loop(event, session.id, workspace_id, mode, report, settings, repo_root)
        except Exception as exc:
            report.error = f"{type(exc).__name__}: {exc}"
            log.exception("worker.crashed", extra={"agent": agent_name})
        finally:
            report.duration_s = time.monotonic() - started
            sessions.close(session.id, settings=settings)

    log.info(
        "worker.finished",
        extra={
            "agent": agent_name,
            "claimed": len(report.claimed),
            "completed": report.completed,
            "findings": report.findings_recorded,
            "stale_writes": report.stale_writes,
            "txn_retries": report.txn_retries,
        },
    )
    return report.to_dict()


def _repo_root(workspace: dict[str, Any], settings: Settings) -> Path:
    """Where the source tree lives, resolved relative to the repo if needed."""
    raw = str(workspace["task_spec"].get("repo", ""))
    path = Path(raw)
    if not path.is_absolute():
        path = settings.repo_root / path
    return path.resolve()


def _run_loop(
    event: dict[str, Any],
    session_id: uuid.UUID,
    workspace_id: uuid.UUID,
    mode: str,
    report: WorkerReport,
    settings: Settings,
    repo_root: Path,
) -> None:
    max_units = event.get("max_units")
    limit = float("inf") if max_units is None else int(max_units)
    claim_only = bool(event.get("claim_only", False))
    work_seconds = float(event.get("work_seconds", 0.01))
    lease_seconds = event.get("lease_seconds")
    work_mode = str(event.get("work_mode", "migrate"))
    # Seeded for reproducible stub timings; not a security context.
    rng = random.Random(event.get("seed"))  # noqa: S311
    # Resolved once per agent, not per unit: building a Bedrock client is not
    # free, and an agent works many units.
    backend: LLMBackend | None = None if claim_only else get_backend(settings)

    heartbeat: sessions.Heartbeater | None = None
    if not claim_only:
        # A claim-only run never holds a lease long enough to need renewing, and
        # the extra thread would only add noise to the stress numbers.
        heartbeat = sessions.Heartbeater(session_id, settings=settings).start()

    try:
        while len(report.claimed) < limit:
            outcome = claims.claim_next(
                workspace_id,
                session_id,
                mode=mode,  # type: ignore[arg-type]
                lease_seconds=lease_seconds,
                settings=settings,
            )
            report.txn_retries += outcome.retries
            report.max_retries_single_claim = max(
                report.max_retries_single_claim, outcome.retries
            )
            report.contended += len(outcome.contended)

            if outcome.unit is None:
                break

            report.claimed.append(str(outcome.unit.id))
            if claim_only:
                continue

            try:
                result_ref = _do_work(
                    outcome.unit,
                    workspace_id=workspace_id,
                    work_mode=work_mode,
                    work_seconds=work_seconds,
                    rng=rng,
                    repo_root=repo_root,
                    backend=backend,
                    report=report,
                    settings=settings,
                )
                claims.complete(
                    outcome.unit,
                    session_id,
                    result_ref=result_ref,
                    mode=mode,  # type: ignore[arg-type]
                    settings=settings,
                )
                report.completed += 1
            except MigrationError as exc:
                claims.fail(outcome.unit, session_id, reason=str(exc), settings=settings)
                report.failed += 1
            except StaleClaimError:
                # The lease expired mid-work and someone else took over. The
                # write was refused, which is the correct outcome -- count it
                # and move on to the next unit.
                report.stale_writes += 1
            except Exception as exc:
                claims.fail(outcome.unit, session_id, reason=str(exc), settings=settings)
                report.failed += 1
    finally:
        if heartbeat is not None:
            heartbeat.stop()


def _do_work(
    unit: ClaimedUnit,
    *,
    workspace_id: uuid.UUID,
    work_mode: str,
    work_seconds: float,
    rng: random.Random,
    repo_root: Path,
    backend: LLMBackend | None,
    report: WorkerReport,
    settings: Settings,
) -> str:
    """Do the unit's work and return the `result_ref` to record against it."""
    if work_mode == "sleep":
        # Phase 2 behaviour, kept for the stress harness: costs time, nothing else.
        jitter = work_seconds * rng.uniform(0.5, 1.5) if work_seconds else 0.0
        if jitter:
            time.sleep(jitter)
        return f"sleep://{unit.target}@v{unit.version}"

    result = migrate_unit(
        unit.spec, repo_root=repo_root, backend=backend, settings=settings
    )
    result_ref = store_result(workspace_id, result, unit.version, settings=settings)

    # The finding is written before the unit is completed. If the agent dies in
    # between, the workspace keeps what it learned and loses only the claim --
    # the opposite ordering would discard the discovery along with the lease.
    findings.record(
        workspace_id,
        result.finding,
        unit_id=unit.id,
        invalidates=result.invalidates,
        backend=backend,
        settings=settings,
    )
    report.findings_recorded += 1
    report.invalidating_findings += int(result.invalidates)
    report.changed_lines += result.changed_lines
    report.input_tokens += result.input_tokens
    report.output_tokens += result.output_tokens

    log.info(
        "unit.migrated",
        extra={
            "target": result.target,
            "changed_lines": result.changed_lines,
            "invalidates": result.invalidates,
            "model": result.model,
        },
    )
    return result_ref


def main(argv: list[str] | None = None) -> int:
    """Run the handler from the command line, so a worker can be a real process.

    The local runner uses this form when a test needs an agent it can actually
    kill -- you cannot SIGKILL a thread, and "agent dies mid-claim" is the
    failure everyone will ask about.
    """
    args = sys.argv[1:] if argv is None else argv
    payload = args[0] if args else sys.stdin.read()
    result = handler(json.loads(payload))
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()
    return 1 if result.get("error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
