"""Local execution of agent workers.

Two transports, because they answer different questions:

* **threads** -- fast, used when the question is "does the database keep N
  concurrent agents honest?". Isolation is enforced by CockroachDB, not by the
  process boundary, so threads are a perfectly valid way to contend.
* **processes** -- slower, used when the question is "what happens when an agent
  *dies*?". You cannot SIGKILL a thread, and an agent that dies mid-claim is the
  failure a reviewer will ask about first.

Phase 8 adds a Lambda transport alongside these. It invokes the same
:func:`quorum.worker.handler` with the same event, which is why the worker was
written Lambda-shaped from the start.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

from quorum import worker
from quorum.config import Settings, get_settings
from quorum.logging import get_logger
from quorum.workspace import resolve_workspace

log = get_logger(__name__)

Transport = Literal["thread", "process"]


@dataclass
class RunReport:
    """Aggregate of one multi-agent run."""

    workspace_id: str
    mode: str
    agents: int
    transport: str
    workers: list[dict[str, Any]] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def total_claimed(self) -> int:
        return sum(len(w.get("claimed", [])) for w in self.workers)

    @property
    def total_completed(self) -> int:
        return sum(int(w.get("completed", 0)) for w in self.workers)

    @property
    def total_retries(self) -> int:
        return sum(int(w.get("txn_retries", 0)) for w in self.workers)

    @property
    def total_contended(self) -> int:
        return sum(int(w.get("contended", 0)) for w in self.workers)

    @property
    def total_stale_writes(self) -> int:
        return sum(int(w.get("stale_writes", 0)) for w in self.workers)

    @property
    def errors(self) -> list[dict[str, str]]:
        """Workers that crashed.

        Without this a worker that died on its first statement looked exactly
        like a healthy run that found no work: zero claims, no complaint. A
        coordination run must never fail quietly.
        """
        return [
            {"agent": str(w.get("agent", "?")), "error": str(w["error"])}
            for w in self.workers
            if w.get("error")
        ]

    @property
    def duplicate_claims(self) -> dict[str, list[str]]:
        """Units claimed by more than one agent. Must be empty in safe mode."""
        holders: dict[str, list[str]] = {}
        for report in self.workers:
            for unit_id in report.get("claimed", []):
                holders.setdefault(unit_id, []).append(report["agent"])
        return {unit: agents for unit, agents in holders.items() if len(agents) > 1}

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "mode": self.mode,
            "agents": self.agents,
            "transport": self.transport,
            "duration_s": round(self.duration_s, 3),
            "total_claimed": self.total_claimed,
            "total_completed": self.total_completed,
            "total_retries": self.total_retries,
            "total_contended": self.total_contended,
            "total_stale_writes": self.total_stale_writes,
            "duplicate_claims": self.duplicate_claims,
            "errors": self.errors,
            "workers": self.workers,
        }


def run(
    workspace_ref: str,
    *,
    agents: int = 4,
    mode: str | None = None,
    max_units: int | None = None,
    work_seconds: float = 0.05,
    seed: int | None = None,
    transport: Transport = "thread",
    settings: Settings | None = None,
) -> RunReport:
    """Run `agents` stub workers against one workspace until it is drained."""
    settings = settings or get_settings()
    workspace = resolve_workspace(workspace_ref, settings)
    workspace_id: uuid.UUID = workspace["id"]
    resolved_mode = mode or str(workspace["mode"])

    events = [
        {
            "workspace_id": str(workspace_id),
            "agent_name": f"agent-{index}",
            "mode": resolved_mode,
            "max_units": max_units,
            "work_seconds": work_seconds,
            "seed": None if seed is None else seed + index,
        }
        for index in range(agents)
    ]

    log.info(
        "run.start",
        extra={
            "workspace_id": workspace_id,
            "agents": agents,
            "run_mode": resolved_mode,
            "transport": transport,
        },
    )

    started = time.monotonic()
    if transport == "process":
        results = _run_processes(events)
    else:
        with ThreadPoolExecutor(max_workers=agents, thread_name_prefix="agent") as pool:
            results = list(pool.map(worker.handler, events))
    duration = time.monotonic() - started

    report = RunReport(
        workspace_id=str(workspace_id),
        mode=resolved_mode,
        agents=agents,
        transport=transport,
        workers=results,
        duration_s=duration,
    )
    if report.errors:
        log.error(
            "run.workers_failed",
            extra={"workspace_id": workspace_id, "errors": report.errors},
        )
    log.info(
        "run.finished",
        extra={
            "workspace_id": workspace_id,
            "claimed": report.total_claimed,
            "completed": report.total_completed,
            "retries": report.total_retries,
            "duplicates": len(report.duplicate_claims),
            "duration_s": round(duration, 3),
        },
    )
    return report


def _run_processes(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    procs = [(event, spawn(event)) for event in events]
    results: list[dict[str, Any]] = []
    for event, proc in procs:
        stdout, _ = proc.communicate()
        results.append(_parse_worker_output(event, stdout))
    return results


def spawn(event: dict[str, Any]) -> subprocess.Popen[str]:
    """Start one agent as a real, killable OS process."""
    return subprocess.Popen(  # noqa: S603 -- fixed argv, no shell
        [sys.executable, "-m", "quorum.worker", json.dumps(event)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def _parse_worker_output(event: dict[str, Any], stdout: str) -> dict[str, Any]:
    """Last line of stdout is the worker report; earlier lines may be logs."""
    for line in reversed((stdout or "").strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return {
        "agent": event.get("agent_name", "unknown"),
        "claimed": [],
        "error": "worker produced no parseable report",
    }
