"""The claim stress harness.

Concurrency bugs are probabilistic, so a single green run proves nothing. This
harness runs the same contention scenario hundreds of times and reports what
actually happened: every claim, every duplicate, every serialization retry, and
the worst contention observed on any single unit.

It runs in both modes, and the difference is the argument:

* **safe** -- N agents, M units, deterministic ordering so every agent goes for
  the *same* row. Zero duplicate claims, across every iteration.
* **naive** -- the identical scenario with the identical agents, where the
  select and the update are separate autocommit statements. Duplicates.

On the barrier: naive mode has a real time-of-check-to-time-of-use window
between its two statements, and on a fast local cluster that window is about a
millisecond wide. `barrier=True` holds every agent at exactly that point so the
race is observed every time instead of occasionally. It does not create the bug
-- `barrier=False` finds the same duplicates by luck, which
`naive_natural_rate` measures on purpose, precisely so the barrier cannot be
accused of manufacturing the result. Safe mode ignores the barrier entirely,
because its read and write are one transaction and there is no window to stand
in.
"""

from __future__ import annotations

import contextlib
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from psycopg import Cursor

from quorum import claims, conflicts, sessions
from quorum.config import Settings, get_settings
from quorum.db import connection, run_serializable
from quorum.logging import get_logger

log = get_logger(__name__)

# Long enough that no lease expires mid-iteration: any duplicate claim is then
# a genuine coordination failure rather than a legitimate expiry takeover.
STRESS_LEASE_SECONDS = 600
BARRIER_TIMEOUT_SECONDS = 10.0


@dataclass
class StressReport:
    """Everything one stress run observed."""

    mode: str
    agents: int
    units: int
    iterations: int
    barrier: bool
    total_claims: int = 0
    duplicate_claims: int = 0
    iterations_with_duplicates: int = 0
    unclaimed_units: int = 0
    txn_retries: int = 0
    max_retries_single_claim: int = 0
    max_agents_on_one_unit: int = 1
    conflicts_logged: int = 0
    max_losers_on_one_unit: int = 0
    conflicts_on_hottest_unit: int = 0
    duration_s: float = 0.0
    samples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True only if every unit went to exactly one agent, every iteration."""
        return self.duplicate_claims == 0 and self.unclaimed_units == 0

    @property
    def claims_per_second(self) -> float:
        return self.total_claims / self.duration_s if self.duration_s else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "agents": self.agents,
            "units": self.units,
            "iterations": self.iterations,
            "barrier": self.barrier,
            "ok": self.ok,
            "total_claims": self.total_claims,
            "duplicate_claims": self.duplicate_claims,
            "iterations_with_duplicates": self.iterations_with_duplicates,
            "unclaimed_units": self.unclaimed_units,
            "txn_retries": self.txn_retries,
            "max_retries_single_claim": self.max_retries_single_claim,
            "max_agents_on_one_unit": self.max_agents_on_one_unit,
            "conflicts_logged": self.conflicts_logged,
            "max_losers_on_one_unit": self.max_losers_on_one_unit,
            "conflicts_on_hottest_unit": self.conflicts_on_hottest_unit,
            "duration_s": round(self.duration_s, 2),
            "claims_per_second": round(self.claims_per_second, 1),
            "samples": self.samples[:5],
        }

    def summary(self) -> str:
        verdict = "PASS" if self.ok else "FAIL"
        return (
            f"[{verdict}] mode={self.mode} agents={self.agents} units={self.units} "
            f"iterations={self.iterations} claims={self.total_claims} "
            f"duplicates={self.duplicate_claims} "
            f"retries={self.txn_retries} (max {self.max_retries_single_claim} on one claim) "
            f"conflicts_logged={self.conflicts_logged}"
        )


class _OneShotGate:
    """Holds every agent at its first claim, then gets out of the way.

    Only the first claim is synchronised: after that the agents drain at their
    own pace, so the harness measures a real race rather than a lockstep march.
    """

    def __init__(self, barrier: threading.Barrier) -> None:
        self._barrier = barrier
        self._used = False

    def __call__(self, _unit_id: uuid.UUID) -> None:
        if self._used:
            return
        self._used = True
        # A broken barrier means an agent found nothing to claim and never
        # arrived. Not a problem: the rest proceed without synchronisation.
        with contextlib.suppress(threading.BrokenBarrierError):
            self._barrier.wait(timeout=BARRIER_TIMEOUT_SECONDS)


def reset_units(
    workspace_id: uuid.UUID, settings: Settings | None = None
) -> int:
    """Return every unit to `pending`, version 1, unheld. Between iterations."""

    def _reset(cur: Cursor) -> int:
        cur.execute(
            """
            UPDATE work_units
               SET status = 'pending',
                   claimed_by = NULL,
                   claim_expires_at = NULL,
                   version = 1,
                   result_ref = NULL,
                   updated_at = now()
             WHERE workspace_id = %s
            """,
            (workspace_id,),
        )
        return cur.rowcount

    return run_serializable(_reset, label="stress.reset", settings=settings).value


def run_stress(
    workspace_id: uuid.UUID,
    *,
    agents: int = 8,
    iterations: int = 200,
    mode: str = "safe",
    barrier: bool = False,
    settings: Settings | None = None,
) -> StressReport:
    """Contend `agents` agents over one workspace, `iterations` times."""
    settings = settings or get_settings()
    unit_count = len(claims.unit_states(workspace_id, settings))
    if unit_count == 0:
        raise ValueError("workspace has no work units to contend over")

    report = StressReport(
        mode=mode, agents=agents, units=unit_count, iterations=iterations, barrier=barrier
    )
    conflicts_before = conflicts.counts(workspace_id, settings).get("total", 0)

    # Sessions are registered once and reused: an agent is a long-lived worker,
    # and re-registering every iteration would measure INSERT throughput rather
    # than claim contention.
    agent_sessions = [
        sessions.register(workspace_id, f"stress-agent-{index}", settings)
        for index in range(agents)
    ]

    started = time.monotonic()
    try:
        for iteration in range(iterations):
            reset_units(workspace_id, settings)
            observed = _drain(
                workspace_id, agent_sessions, mode=mode, barrier=barrier, settings=settings
            )
            _fold(report, iteration, observed, unit_count)
    finally:
        report.duration_s = time.monotonic() - started
        for session in agent_sessions:
            sessions.close(session.id, settings=settings)

    conflicts_after = conflicts.counts(workspace_id, settings).get("total", 0)
    report.conflicts_logged = conflicts_after - conflicts_before
    report.conflicts_on_hottest_unit = _hottest_unit(workspace_id, settings)

    log.info("stress.complete", extra=report.to_dict())
    return report


@dataclass
class _Observed:
    """One iteration, from the agents' point of view."""

    claims: list[tuple[str, str]] = field(default_factory=list)
    contended: list[tuple[str, str]] = field(default_factory=list)  # (unit_id, loser)
    retries: int = 0
    max_retries: int = 0


def _drain(
    workspace_id: uuid.UUID,
    agent_sessions: list[sessions.AgentSession],
    *,
    mode: str,
    barrier: bool,
    settings: Settings,
) -> _Observed:
    """All agents claim until nothing is left. Claims only -- no completion.

    Not completing is deliberate: a claimed unit stays claimed, so any unit that
    shows up twice in the results was handed to two agents at once. With
    completion in the loop the evidence would be muddier.
    """
    observed = _Observed()
    lock = threading.Lock()
    gate_barrier = threading.Barrier(len(agent_sessions)) if barrier else None
    threads: list[threading.Thread] = []

    def agent_loop(session: sessions.AgentSession) -> None:
        gate = _OneShotGate(gate_barrier) if gate_barrier is not None else None
        while True:
            outcome = claims.claim_next(
                workspace_id,
                session.id,
                mode=mode,  # type: ignore[arg-type]
                lease_seconds=STRESS_LEASE_SECONDS,
                settings=settings,
                on_selected=gate,
            )
            with lock:
                observed.retries += outcome.retries
                observed.max_retries = max(observed.max_retries, outcome.retries)
                for event in outcome.contended:
                    observed.contended.append((event["unit_id"], session.name))
                if outcome.unit is not None:
                    observed.claims.append((str(outcome.unit.id), session.name))
            if outcome.unit is None:
                break

    for session in agent_sessions:
        thread = threading.Thread(target=agent_loop, args=(session,), name=session.name)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    if gate_barrier is not None:
        gate_barrier.abort()  # release anyone still waiting
    return observed


def _fold(report: StressReport, iteration: int, observed: _Observed, unit_count: int) -> None:
    holders: dict[str, list[str]] = {}
    for unit_id, agent in observed.claims:
        holders.setdefault(unit_id, []).append(agent)

    duplicates = {unit: agents for unit, agents in holders.items() if len(agents) > 1}
    extra_claims = sum(len(agents) - 1 for agents in duplicates.values())

    report.total_claims += len(observed.claims)
    report.txn_retries += observed.retries
    report.max_retries_single_claim = max(report.max_retries_single_claim, observed.max_retries)
    report.duplicate_claims += extra_claims
    report.unclaimed_units += max(0, unit_count - len(holders))

    # How many distinct agents lost a race for the same unit, in this iteration.
    # Counting conflict_log rows across the whole run instead would just report
    # how long the run was.
    losers: dict[str, set[str]] = {}
    for unit_id, agent in observed.contended:
        losers.setdefault(unit_id, set()).add(agent)
    if losers:
        report.max_losers_on_one_unit = max(
            report.max_losers_on_one_unit, *(len(a) for a in losers.values())
        )

    if duplicates:
        report.iterations_with_duplicates += 1
        report.max_agents_on_one_unit = max(
            report.max_agents_on_one_unit, *(len(a) for a in duplicates.values())
        )
        report.samples.append(
            {
                "iteration": iteration,
                "unit_id": next(iter(duplicates)),
                "agents": duplicates[next(iter(duplicates))],
            }
        )


def _hottest_unit(workspace_id: uuid.UUID, settings: Settings) -> int:
    """Conflict rows recorded against the single most-contended unit."""
    with connection(settings) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT detail->>'unit_id' AS unit_id, count(*) AS n
              FROM conflict_log
             WHERE workspace_id = %s
               AND kind = 'claim'
               AND detail->>'reason' = 'concurrent_claim'
             GROUP BY detail->>'unit_id'
             ORDER BY count(*) DESC
             LIMIT 1
            """,
            (workspace_id,),
        )
        row = cur.fetchone()
        return 0 if row is None else int(row["n"])


def naive_natural_rate(
    workspace_id: uuid.UUID,
    *,
    agents: int = 8,
    iterations: int = 50,
    settings: Settings | None = None,
) -> StressReport:
    """Naive mode with no barrier: how often the race bites on its own.

    Exists to answer the obvious objection -- that the barrier manufactures the
    failure. It does not. This finds the same duplicates unaided; the barrier
    only makes them reproducible on demand.
    """
    return run_stress(
        workspace_id,
        agents=agents,
        iterations=iterations,
        mode="naive",
        barrier=False,
        settings=settings,
    )


def contention_histogram(
    workspace_id: uuid.UUID, settings: Settings | None = None
) -> Counter[str]:
    """How conflicts were resolved, by resolution string."""
    return Counter(conflicts.resolution_counts(workspace_id, settings))
