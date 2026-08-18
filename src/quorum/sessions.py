"""Agent sessions: registration, heartbeats, and liveness.

A session is how the workspace knows an agent is still alive. It matters because
Quorum hands out *leases*, not locks: a lock held by a crashed agent deadlocks
the workspace forever, whereas a lease held by a crashed agent simply stops
being renewed and expires.

Heartbeating therefore does two things at once -- it advances
`agent_sessions.heartbeat_at`, and it extends the expiry on every unit the
session currently holds. An agent that stops running stops doing both, and the
reaper reclaims its work.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Any

from psycopg import Cursor

from quorum.config import Settings, get_settings
from quorum.db import connection, run_serializable
from quorum.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class AgentSession:
    """A registered agent, and the workspace it is working in."""

    id: uuid.UUID
    workspace_id: uuid.UUID
    name: str


def _default_timeout(timeout_seconds: int | None, settings: Settings) -> int:
    """Liveness window, defaulting to three missed heartbeats."""
    if timeout_seconds is None:
        return settings.heartbeat_seconds * 3
    return timeout_seconds


def register(
    workspace_id: uuid.UUID, name: str, settings: Settings | None = None
) -> AgentSession:
    """Create a running session for an agent."""

    def _insert(cur: Cursor) -> uuid.UUID:
        cur.execute(
            """
            INSERT INTO agent_sessions (workspace_id, name, status)
            VALUES (%s, %s, 'running')
            RETURNING id
            """,
            (workspace_id, name),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("agent_sessions INSERT did not return an id")
        return row["id"]

    result = run_serializable(_insert, label="session.register", settings=settings)
    session = AgentSession(id=result.value, workspace_id=workspace_id, name=name)
    log.info(
        "session.registered",
        extra={"session_id": session.id, "workspace_id": workspace_id, "agent": name},
    )
    return session


def heartbeat(
    session_id: uuid.UUID,
    *,
    lease_seconds: int | None = None,
    renew_leases: bool = True,
    settings: Settings | None = None,
) -> int:
    """Mark the session alive and extend the leases it holds.

    Returns the number of leases renewed. Runs as a single autocommit statement
    pair on purpose: a heartbeat that contends with claim traffic and retries
    would make liveness *depend* on contention, which is backwards.
    """
    settings = settings or get_settings()
    lease = settings.claim_lease_seconds if lease_seconds is None else lease_seconds

    with connection(settings) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE agent_sessions SET heartbeat_at = now() WHERE id = %s",
                (session_id,),
            )
            if not renew_leases:
                return 0
            cur.execute(
                """
                UPDATE work_units
                   SET claim_expires_at = now() + (%s::INT * INTERVAL '1 second'),
                       updated_at = now()
                 WHERE claimed_by = %s AND status = 'claimed'
                """,
                (lease, session_id),
            )
            return cur.rowcount


def close(
    session_id: uuid.UUID,
    *,
    status: str = "stopped",
    settings: Settings | None = None,
) -> None:
    """End a session. Units it still holds are left to expire naturally."""

    def _update(cur: Cursor) -> None:
        cur.execute(
            "UPDATE agent_sessions SET status = %s WHERE id = %s", (status, session_id)
        )

    run_serializable(_update, label="session.close", settings=settings)
    log.info("session.closed", extra={"session_id": session_id, "session_status": status})


def live(
    workspace_id: uuid.UUID,
    *,
    timeout_seconds: int | None = None,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Sessions that are running and have heartbeated recently enough."""
    settings = settings or get_settings()
    # Three missed heartbeats before a session is considered gone: one missed
    # beat is a slow query, three is a dead process.
    # `is None`, not `or`: timeout_seconds=0 means "everything not beating right
    # now", and `or` would silently turn that into the 30-second default.
    timeout = _default_timeout(timeout_seconds, settings)

    with connection(settings) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT * FROM agent_sessions
            WHERE workspace_id = %s
              AND status = 'running'
              AND heartbeat_at > now() - (%s::INT * INTERVAL '1 second')
            ORDER BY started_at
            """,
            (workspace_id, timeout),
        )
        return [dict(row) for row in cur.fetchall()]


def mark_stale_dead(
    workspace_id: uuid.UUID,
    *,
    timeout_seconds: int | None = None,
    settings: Settings | None = None,
) -> list[uuid.UUID]:
    """Flip silent sessions to `dead`. Returns the ids that were flipped.

    This is bookkeeping for the dashboard, not the safety mechanism -- unit
    recovery keys off `claim_expires_at`, so a unit is reclaimable whether or
    not anyone has got around to marking its holder dead.
    """
    settings = settings or get_settings()
    timeout = _default_timeout(timeout_seconds, settings)

    def _update(cur: Cursor) -> list[uuid.UUID]:
        cur.execute(
            """
            UPDATE agent_sessions
               SET status = 'dead'
             WHERE workspace_id = %s
               AND status = 'running'
               AND heartbeat_at < now() - (%s::INT * INTERVAL '1 second')
            RETURNING id
            """,
            (workspace_id, timeout),
        )
        return [row["id"] for row in cur.fetchall()]

    result = run_serializable(_update, label="session.reap", settings=settings)
    if result.value:
        log.warning(
            "session.marked_dead",
            extra={"workspace_id": workspace_id, "sessions": [str(s) for s in result.value]},
        )
    return result.value


class Heartbeater:
    """Background thread that heartbeats a session until stopped.

    Used by the local runner. The Lambda runner heartbeats inline instead, since
    a background thread does not survive a frozen execution environment.
    """

    def __init__(
        self,
        session_id: uuid.UUID,
        *,
        interval_seconds: float | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.session_id = session_id
        self.interval = interval_seconds or float(self.settings.heartbeat_seconds)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.beats = 0

    def start(self) -> Heartbeater:
        self._thread = threading.Thread(
            target=self._loop, name=f"heartbeat-{self.session_id}", daemon=True
        )
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                heartbeat(self.session_id, settings=self.settings)
                self.beats += 1
            except Exception as exc:  # a failed beat must not kill the agent
                log.warning(
                    "heartbeat.failed",
                    extra={"session_id": self.session_id, "reason": str(exc)},
                )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 1.0)

    def __enter__(self) -> Heartbeater:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()
