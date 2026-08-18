"""The claim engine: leases, contention, and reclaim.

This is conflict #1, and the place where the whole thesis is either true or not.

**Safe mode** claims inside one `SERIALIZABLE` transaction that takes an explicit
`FOR UPDATE` lock on the candidate row. Two agents racing for the same unit
produce exactly one winner; the loser aborts with SQLSTATE 40001, replays the
whole transaction, observes that the unit is now held by someone else, records
that contention in `conflict_log`, and takes a different unit.

**Naive mode** does the same work as two separate autocommit statements with no
lock, no isolation, and no retry -- a textbook time-of-check-to-time-of-use
race. It is not a strawman: it is what an agent memory layer does when the store
is a document database, and it is the control group the demo needs.

Claims are **leases**, not locks. A lock held by a crashed agent deadlocks the
workspace forever. A lease stops being renewed, expires, and is reclaimed --
with a `version` bump, so the crashed agent cannot come back and overwrite
newer work.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from psycopg import Cursor
from psycopg.types.json import Jsonb

from quorum import conflicts
from quorum.config import Settings, get_settings
from quorum.db import connection, run_autocommit, run_serializable
from quorum.logging import get_logger

log = get_logger(__name__)

Mode = Literal["safe", "naive"]

# A unit is available if nobody holds a live lease on it. `stale` is included so
# that units re-queued by the phase 5 invalidation cascade are picked straight
# back up.
CANDIDATE_SQL = """
    SELECT id, target, spec, version, status, claimed_by
      FROM work_units
     WHERE workspace_id = %s
       AND (
             status IN ('pending', 'stale')
             OR (status = 'claimed' AND claim_expires_at < now())
           )
     ORDER BY target
     LIMIT 1
"""


class StaleClaimError(RuntimeError):
    """Raised when an agent tries to write a result it no longer has the right to.

    Means its lease expired and the unit was reclaimed (and re-versioned) while
    it was still working. The write is rejected rather than allowed to clobber
    whatever the replacement agent produced.
    """


@dataclass(frozen=True)
class ClaimedUnit:
    """A work unit this agent now holds a lease on."""

    id: uuid.UUID
    workspace_id: uuid.UUID
    target: str
    spec: dict[str, Any]
    version: int
    reclaimed: bool  # True if taken from an agent whose lease had expired


@dataclass
class ClaimOutcome:
    """What a claim attempt produced, and what it cost."""

    unit: ClaimedUnit | None
    retries: int = 0
    contended: list[dict[str, Any]] = field(default_factory=list)
    duration_s: float = 0.0

    @property
    def claimed(self) -> bool:
        return self.unit is not None


@dataclass
class _ClaimState:
    """Carried across transaction retries.

    `run_serializable` replays the callable from scratch on a 40001, so anything
    the loser learned about who beat it has to live outside the transaction.
    Deduplicated by (unit, winner) so a claim that retries several times against
    the same opponent logs one conflict, not one per attempt.
    """

    last_target: uuid.UUID | None = None
    contended: list[dict[str, Any]] = field(default_factory=list)
    seen: set[tuple[str, str]] = field(default_factory=set)

    def note(self, unit_id: uuid.UUID, target: str, winner: uuid.UUID, version: int) -> None:
        key = (str(unit_id), str(winner))
        if key in self.seen:
            return
        self.seen.add(key)
        self.contended.append(
            {
                "unit_id": str(unit_id),
                "target": target,
                "winner": str(winner),
                "unit_version": version,
            }
        )


def claim_next(
    workspace_id: uuid.UUID,
    session_id: uuid.UUID,
    *,
    mode: Mode = "safe",
    lease_seconds: int | None = None,
    settings: Settings | None = None,
    on_selected: Callable[[uuid.UUID], None] | None = None,
    detect_contention: bool | None = None,
) -> ClaimOutcome:
    """Claim one available work unit, or return an empty outcome if none remain.

    `on_selected` is invoked between selecting a candidate and writing the claim.
    It exists so tests can hold every naive-mode agent at exactly that point with
    a barrier, turning a probabilistic race into a deterministic one. It does not
    create the race -- it only removes the luck from observing it -- and it is
    ignored in safe mode, where the window is inside a transaction and therefore
    is not a window at all.
    """
    settings = settings or get_settings()
    lease = settings.claim_lease_seconds if lease_seconds is None else lease_seconds

    if mode == "naive":
        return _claim_naive(
            workspace_id, session_id, lease=lease, settings=settings, on_selected=on_selected
        )
    detect = settings.conflict_detection if detect_contention is None else detect_contention
    return _claim_safe(
        workspace_id, session_id, lease=lease, settings=settings, detect=detect
    )


def _claim_safe(
    workspace_id: uuid.UUID,
    session_id: uuid.UUID,
    *,
    lease: int,
    settings: Settings,
    detect: bool = True,
) -> ClaimOutcome:
    state = _ClaimState()

    # Contention shows up in two shapes, and both must be caught or the conflict
    # log under-reports what happened:
    #
    #   1. The transaction aborted with 40001 and is being replayed. The unit it
    #      was going for last time now belongs to someone else.
    #   2. Far more common under `FOR UPDATE`: the transaction never aborted at
    #      all. It *blocked* on the lock, and by the time CockroachDB handed it
    #      over the winner had already taken the row. No retry, no error -- but
    #      a race was lost all the same.
    #
    # The peek that catches shape 2 runs **outside** the transaction, and that
    # placement is the whole design. Inside, it put a read at a timestamp the
    # winner's commit then invalidated, so the transaction could not refresh and
    # aborted -- the instrumentation manufactured the very retries it was
    # reporting. Measured: 1,848 aborts with the peek inside, 0 with it outside,
    # on identical work. Diagnostics must not perturb what they measure.
    intended_id: uuid.UUID | None = None
    if detect:
        intended_id = _peek_candidate(workspace_id, settings)
        state.last_target = intended_id

    def _attempt(cur: Cursor) -> ClaimedUnit | None:
        cur.execute(CANDIDATE_SQL + " FOR UPDATE", (workspace_id,))
        candidate = cur.fetchone()

        # Every read below happens *after* the lock is held, so the transaction
        # timestamp is already past the winner's commit and nothing here can
        # cause a refresh failure.
        if detect:
            if state.last_target is not None and (
                candidate is None or candidate["id"] != state.last_target
            ):
                _note_if_taken(cur, state, state.last_target, session_id)
            state.last_target = None if candidate is None else candidate["id"]

        if candidate is None:
            _flush_conflicts(cur, workspace_id, session_id, state)
            return None

        # Reclaiming an expired lease re-versions the unit, which is what makes
        # the previous holder's late write detectable instead of destructive.
        reclaimed = candidate["status"] == "claimed"
        version = int(candidate["version"]) + (1 if reclaimed else 0)

        cur.execute(
            """
            UPDATE work_units
               SET status = 'claimed',
                   claimed_by = %s,
                   claim_expires_at = now() + (%s::INT * INTERVAL '1 second'),
                   version = %s,
                   updated_at = now()
             WHERE id = %s
            """,
            (session_id, lease, version, candidate["id"]),
        )

        if reclaimed:
            conflicts.record(
                cur,
                workspace_id=workspace_id,
                kind="claim",
                agents=[session_id, candidate["claimed_by"]],
                detail={
                    "reason": "expired_lease_taken_over",
                    "unit_id": str(candidate["id"]),
                    "target": str(candidate["target"]),
                    "previous_holder": str(candidate["claimed_by"]),
                    "new_version": version,
                },
                resolution="reclaimed_by_new_agent",
            )

        _flush_conflicts(cur, workspace_id, session_id, state)

        return ClaimedUnit(
            id=candidate["id"],
            workspace_id=workspace_id,
            target=str(candidate["target"]),
            spec=dict(candidate["spec"]),
            version=version,
            reclaimed=reclaimed,
        )

    result = run_serializable(_attempt, label="claim.next", settings=settings)

    outcome = ClaimOutcome(
        unit=result.value,
        retries=result.retries,
        contended=list(state.contended),
        duration_s=result.duration_s,
    )
    if outcome.unit is not None:
        log.info(
            "claim.acquired",
            extra={
                "session_id": session_id,
                "unit_id": outcome.unit.id,
                "target": outcome.unit.target,
                "unit_version": outcome.unit.version,
                "reclaimed": outcome.unit.reclaimed,
                "txn_retries": outcome.retries,
                "contended_with": len(outcome.contended),
            },
        )
    return outcome


def _peek_candidate(workspace_id: uuid.UUID, settings: Settings) -> uuid.UUID | None:
    """Which unit this agent would go for, read outside any transaction.

    Its own tiny autocommit read, deliberately. Anything this observes is
    advisory: it feeds the conflict log and nothing else, so a stale answer
    costs an unlogged conflict at worst and never a wrong claim.
    """

    def _read(cur: Cursor) -> uuid.UUID | None:
        cur.execute(CANDIDATE_SQL, (workspace_id,))
        row = cur.fetchone()
        return None if row is None else row["id"]

    return run_autocommit(_read, label="claim.peek", settings=settings).value


def _note_if_taken(
    cur: Cursor, state: _ClaimState, unit_id: uuid.UUID, session_id: uuid.UUID
) -> None:
    """Record a lost race, if this unit is now held by somebody else.

    Only the loser of a race can observe that it lost -- the winner's claim
    looks identical to an uncontended one -- so contention is always logged
    from the losing side.
    """
    cur.execute(
        "SELECT id, target, status, claimed_by, version FROM work_units WHERE id = %s",
        (unit_id,),
    )
    row = cur.fetchone()
    if (
        row is not None
        and row["claimed_by"] is not None
        and row["claimed_by"] != session_id
        and row["status"] in ("claimed", "done")
    ):
        state.note(row["id"], str(row["target"]), row["claimed_by"], int(row["version"]))


def _flush_conflicts(
    cur: Cursor, workspace_id: uuid.UUID, session_id: uuid.UUID, state: _ClaimState
) -> None:
    """Write every contention this agent observed, in the committing transaction.

    Re-written on each attempt because an aborted attempt rolls its inserts back
    along with everything else. Only the transaction that commits leaves rows.
    """
    for event in state.contended:
        conflicts.record(
            cur,
            workspace_id=workspace_id,
            kind="claim",
            agents=[session_id, uuid.UUID(event["winner"])],
            detail={
                "reason": "concurrent_claim",
                "unit_id": event["unit_id"],
                "target": event["target"],
                "winner": event["winner"],
                "loser": str(session_id),
                "unit_version": event["unit_version"],
            },
            resolution="serialized_by_cockroachdb",
        )


def _claim_naive(
    workspace_id: uuid.UUID,
    session_id: uuid.UUID,
    *,
    lease: int,
    settings: Settings,
    on_selected: Callable[[uuid.UUID], None] | None,
) -> ClaimOutcome:
    """Select, then update, with nothing in between holding the two together.

    Every statement is its own implicit transaction. Between the read and the
    write, any number of other agents can read the same row and reach the same
    conclusion. No lock, no version check, no retry, and -- notably -- nothing
    written to `conflict_log`, because naive mode never finds out.
    """
    started = time.monotonic()

    def _select(cur: Cursor) -> Any:
        cur.execute(CANDIDATE_SQL, (workspace_id,))
        return cur.fetchone()

    candidate = run_autocommit(_select, label="claim.next.naive", settings=settings).value
    if candidate is None:
        return ClaimOutcome(unit=None, duration_s=time.monotonic() - started)

    if on_selected is not None:
        on_selected(candidate["id"])

    reclaimed = candidate["status"] == "claimed"
    version = int(candidate["version"]) + (1 if reclaimed else 0)

    def _update(cur: Cursor) -> None:
        cur.execute(
            """
            UPDATE work_units
               SET status = 'claimed',
                   claimed_by = %s,
                   claim_expires_at = now() + (%s::INT * INTERVAL '1 second'),
                   version = %s,
                   updated_at = now()
             WHERE id = %s
            """,
            (session_id, lease, version, candidate["id"]),
        )

    run_autocommit(_update, label="claim.write.naive", settings=settings)

    unit = ClaimedUnit(
        id=candidate["id"],
        workspace_id=workspace_id,
        target=str(candidate["target"]),
        spec=dict(candidate["spec"]),
        version=version,
        reclaimed=reclaimed,
    )
    log.info(
        "claim.acquired",
        extra={
            "session_id": session_id,
            "unit_id": unit.id,
            "target": unit.target,
            "claim_mode": "naive",
        },
    )
    return ClaimOutcome(unit=unit, duration_s=time.monotonic() - started)


def complete(
    unit: ClaimedUnit,
    session_id: uuid.UUID,
    *,
    result_ref: str | None = None,
    mode: Mode = "safe",
    settings: Settings | None = None,
) -> None:
    """Mark a claimed unit done.

    In safe mode the write is guarded on holder, version, and status together.
    If the lease expired and someone else took over, the guard fails, the attempt
    is logged, and `StaleClaimError` is raised -- the whole point of versioning
    a reclaim.
    """
    settings = settings or get_settings()

    if mode == "naive":
        run_autocommit(
            lambda cur: cur.execute(
                """
                UPDATE work_units
                   SET status = 'done', result_ref = %s, updated_at = now()
                 WHERE id = %s
                """,
                (result_ref, unit.id),
            ),
            label="claim.complete.naive",
            settings=settings,
        )
        return

    def _update(cur: Cursor) -> int:
        cur.execute(
            """
            UPDATE work_units
               SET status = 'done', result_ref = %s, updated_at = now()
             WHERE id = %s
               AND claimed_by = %s
               AND version = %s
               AND status = 'claimed'
            """,
            (result_ref, unit.id, session_id, unit.version),
        )
        if cur.rowcount == 1:
            return 1

        cur.execute(
            "SELECT status, claimed_by, version FROM work_units WHERE id = %s", (unit.id,)
        )
        current = cur.fetchone()
        conflicts.record(
            cur,
            workspace_id=unit.workspace_id,
            kind="claim",
            agents=[session_id, current["claimed_by"] if current else None],
            detail={
                "reason": "stale_write_rejected",
                "unit_id": str(unit.id),
                "target": unit.target,
                "held_version": unit.version,
                "current_version": int(current["version"]) if current else None,
                "current_holder": str(current["claimed_by"]) if current else None,
                "current_status": str(current["status"]) if current else None,
            },
            resolution="rejected_lost_lease",
        )
        return 0

    result = run_serializable(_update, label="claim.complete", settings=settings)
    if result.value == 0:
        log.warning(
            "claim.stale_write_rejected",
            extra={"session_id": session_id, "unit_id": unit.id, "held_version": unit.version},
        )
        raise StaleClaimError(
            f"session {session_id} no longer holds unit {unit.id} at version {unit.version}"
        )

    log.info(
        "claim.completed",
        extra={"session_id": session_id, "unit_id": unit.id, "target": unit.target},
    )


def release(
    unit: ClaimedUnit,
    session_id: uuid.UUID,
    *,
    status: str = "pending",
    settings: Settings | None = None,
) -> bool:
    """Give a claimed unit back without completing it. Returns True if released."""

    def _update(cur: Cursor) -> bool:
        cur.execute(
            """
            UPDATE work_units
               SET status = %s, claim_expires_at = NULL, updated_at = now()
             WHERE id = %s AND claimed_by = %s AND version = %s AND status = 'claimed'
            """,
            (status, unit.id, session_id, unit.version),
        )
        return cur.rowcount == 1

    result = run_serializable(_update, label="claim.release", settings=settings)
    log.info(
        "claim.released",
        extra={"session_id": session_id, "unit_id": unit.id, "released": result.value},
    )
    return result.value


def fail(
    unit: ClaimedUnit,
    session_id: uuid.UUID,
    *,
    reason: str,
    settings: Settings | None = None,
) -> None:
    """Mark a unit failed, recording why on the unit spec."""

    def _update(cur: Cursor) -> None:
        cur.execute(
            """
            UPDATE work_units
               SET status = 'failed',
                   spec = spec || %s,
                   updated_at = now()
             WHERE id = %s AND claimed_by = %s
            """,
            (Jsonb({"failure_reason": reason}), unit.id, session_id),
        )

    run_serializable(_update, label="claim.fail", settings=settings)
    log.warning(
        "claim.failed",
        extra={"session_id": session_id, "unit_id": unit.id, "reason": reason},
    )


def reap_expired(
    workspace_id: uuid.UUID,
    *,
    mode: Mode = "safe",
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Return every expired lease to the pool. Returns the units reclaimed.

    Naive mode has no reaper at all: a unit claimed by a dead agent stays
    claimed forever, which is precisely the deadlock that leases exist to
    prevent, and one of the differences `quorum compare` reports.
    """
    settings = settings or get_settings()
    if mode == "naive":
        return []

    def _reap(cur: Cursor) -> list[dict[str, Any]]:
        cur.execute(
            """
            UPDATE work_units
               SET status = 'pending',
                   claim_expires_at = NULL,
                   version = version + 1,
                   updated_at = now()
             WHERE workspace_id = %s
               AND status = 'claimed'
               AND claim_expires_at < now()
            RETURNING id, target, claimed_by, version
            """,
            (workspace_id,),
        )
        reclaimed = [dict(row) for row in cur.fetchall()]

        for unit in reclaimed:
            conflicts.record(
                cur,
                workspace_id=workspace_id,
                kind="claim",
                agents=[unit["claimed_by"]],
                detail={
                    "reason": "lease_expired",
                    "unit_id": str(unit["id"]),
                    "target": str(unit["target"]),
                    "previous_holder": str(unit["claimed_by"]),
                    "new_version": int(unit["version"]),
                },
                resolution="requeued_by_reaper",
            )
        return reclaimed

    result = run_serializable(_reap, label="claim.reap", settings=settings)
    if result.value:
        log.warning(
            "claim.reaped",
            extra={
                "workspace_id": workspace_id,
                "reclaimed": len(result.value),
                "targets": [str(u["target"]) for u in result.value],
            },
        )
    return result.value


def unit_states(
    workspace_id: uuid.UUID, settings: Settings | None = None
) -> list[dict[str, Any]]:
    """Every unit with its current holder. Used by tests and the dashboard."""
    with connection(settings) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, target, status, claimed_by, claim_expires_at, version, result_ref
              FROM work_units
             WHERE workspace_id = %s
             ORDER BY target
            """,
            (workspace_id,),
        )
        return [dict(row) for row in cur.fetchall()]


def expire_lease_now(
    unit_id: uuid.UUID, settings: Settings | None = None
) -> datetime | None:
    """Force a lease to be expired. Test and demo hook for agent death.

    Kills the *lease*, not the agent -- used when a test needs the state a dead
    agent leaves behind without waiting out a real lease interval.
    """

    def _update(cur: Cursor) -> datetime | None:
        cur.execute(
            """
            UPDATE work_units
               SET claim_expires_at = now() - INTERVAL '1 second'
             WHERE id = %s AND status = 'claimed'
            RETURNING claim_expires_at
            """,
            (unit_id,),
        )
        row = cur.fetchone()
        return None if row is None else row["claim_expires_at"]

    return run_serializable(_update, label="claim.expire_lease", settings=settings).value
