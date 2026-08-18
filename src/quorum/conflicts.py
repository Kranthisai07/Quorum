"""The conflict log: proof that contention happened.

This is the headline artifact, not telemetry. A coordination layer that resolves
conflicts silently is indistinguishable from one that never had any, so every
contended claim, contradiction, and cascade is written here with the agents
involved, what was contended, and how it was resolved.

Writers take a cursor rather than opening their own transaction. A conflict row
must commit with the resolution that produced it -- a conflict logged but not
resolved, or resolved but not logged, is a lie in either direction.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any, Literal

from psycopg import Cursor
from psycopg.types.json import Jsonb

from quorum.config import Settings
from quorum.db import connection
from quorum.logging import get_logger

log = get_logger(__name__)

ConflictKind = Literal["claim", "semantic", "invalidation"]


def record(
    cur: Cursor,
    *,
    workspace_id: uuid.UUID,
    kind: ConflictKind,
    agents: Sequence[uuid.UUID | None],
    detail: dict[str, Any],
    resolution: str,
    resolved: bool = True,
) -> uuid.UUID:
    """Write one conflict row inside the caller's transaction."""
    cur.execute(
        """
        INSERT INTO conflict_log
            (workspace_id, kind, agents, detail, resolution, resolved_at)
        VALUES (%s, %s, %s, %s, %s, CASE WHEN %s THEN now() ELSE NULL END)
        RETURNING id
        """,
        (
            workspace_id,
            kind,
            [a for a in agents if a is not None],
            Jsonb(detail),
            resolution,
            resolved,
        ),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("conflict_log INSERT did not return an id")

    conflict_id: uuid.UUID = row["id"]
    log.info(
        "conflict.recorded",
        extra={
            "conflict_id": conflict_id,
            "workspace_id": workspace_id,
            "kind": kind,
            "resolution": resolution,
            "detail": detail,
        },
    )
    return conflict_id


def listing(
    workspace_id: uuid.UUID,
    *,
    kind: ConflictKind | None = None,
    limit: int = 50,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Most recent conflicts first. Backs `quorum conflicts` and the dashboard."""
    with connection(settings) as conn, conn.cursor() as cur:
        if kind is None:
            cur.execute(
                """
                SELECT * FROM conflict_log
                WHERE workspace_id = %s
                ORDER BY detected_at DESC
                LIMIT %s
                """,
                (workspace_id, limit),
            )
        else:
            cur.execute(
                """
                SELECT * FROM conflict_log
                WHERE workspace_id = %s AND kind = %s
                ORDER BY detected_at DESC
                LIMIT %s
                """,
                (workspace_id, kind, limit),
            )
        return [dict(row) for row in cur.fetchall()]


def counts(
    workspace_id: uuid.UUID, settings: Settings | None = None
) -> dict[str, int]:
    """Conflict tally by kind, plus a total."""
    with connection(settings) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT kind, count(*) AS n
            FROM conflict_log
            WHERE workspace_id = %s
            GROUP BY kind
            """,
            (workspace_id,),
        )
        by_kind = {str(row["kind"]): int(row["n"]) for row in cur.fetchall()}
    by_kind["total"] = sum(by_kind.values())
    return by_kind


def resolution_counts(
    workspace_id: uuid.UUID, settings: Settings | None = None
) -> dict[str, int]:
    """Tally by resolution string -- how contention was actually settled."""
    with connection(settings) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT resolution, count(*) AS n
            FROM conflict_log
            WHERE workspace_id = %s
            GROUP BY resolution
            ORDER BY count(*) DESC
            """,
            (workspace_id,),
        )
        return {str(row["resolution"]): int(row["n"]) for row in cur.fetchall()}
