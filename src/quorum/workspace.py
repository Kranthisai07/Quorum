"""Workspace lifecycle: seeding, lookup, and summary reporting.

Seeding is the first place the serializable guarantee earns its keep. A
workspace is only meaningful if its work units and dependency edges exist
together: a half-written dependency graph would make the Phase 5 invalidation
cascade silently incomplete, which is precisely the corruption Quorum claims to
prevent. So the whole decomposition commits in one transaction or none of it
does.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from psycopg import Cursor
from psycopg.types.json import Jsonb

from quorum.config import Settings, get_settings
from quorum.db import connection, run_serializable
from quorum.decompose import Decomposition, get_decomposer
from quorum.logging import get_logger

log = get_logger(__name__)

Mode = Literal["safe", "naive"]


@dataclass(frozen=True)
class SeedResult:
    """What a seed produced, for the CLI and for tests to assert on."""

    workspace_id: uuid.UUID
    name: str
    mode: Mode
    unit_count: int
    dep_count: int
    scopes: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    txn_retries: int = 0


def seed_workspace(
    *,
    name: str,
    task_spec: Mapping[str, Any],
    mode: Mode = "safe",
    decomposer: str | None = None,
    settings: Settings | None = None,
) -> SeedResult:
    """Decompose `task_spec` and write the resulting workspace atomically."""
    settings = settings or get_settings()
    decomposer_name = decomposer or str(task_spec.get("kind", "code_migration"))
    decomposition = get_decomposer(decomposer_name).decompose(task_spec)

    if not decomposition.units:
        raise ValueError(
            "decomposition produced no work units; check the repo path and "
            "the from_library signal table"
        )

    spec_payload = dict(task_spec)
    spec_payload["decomposer"] = decomposer_name
    spec_payload["decomposition"] = decomposition.metadata

    def _write(cur: Cursor) -> uuid.UUID:
        cur.execute(
            """
            INSERT INTO workspaces (name, task_spec, mode)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (name, Jsonb(spec_payload), mode),
        )
        row = _require(cur.fetchone(), "workspaces INSERT did not return an id")
        workspace_id: uuid.UUID = row["id"]

        target_to_id: dict[str, uuid.UUID] = {}
        for unit in decomposition.units:
            cur.execute(
                """
                INSERT INTO work_units (workspace_id, target, spec, status)
                VALUES (%s, %s, %s, 'pending')
                RETURNING id
                """,
                (workspace_id, unit.target, Jsonb(unit.spec)),
            )
            unit_row = _require(
                cur.fetchone(), f"work_units INSERT did not return an id for {unit.target}"
            )
            target_to_id[unit.target] = unit_row["id"]

        for unit_target, depends_on in decomposition.deps:
            cur.execute(
                """
                INSERT INTO unit_deps (unit_id, depends_on_unit_id)
                VALUES (%s, %s)
                """,
                (target_to_id[unit_target], target_to_id[depends_on]),
            )

        return workspace_id

    result = run_serializable(_write, label="workspace.seed", settings=settings)

    log.info(
        "workspace.seeded",
        extra={
            "workspace_id": result.value,
            "name": name,
            "mode": mode,
            "units": len(decomposition.units),
            "deps": len(decomposition.deps),
            "txn_retries": result.retries,
        },
    )
    return SeedResult(
        workspace_id=result.value,
        name=name,
        mode=mode,
        unit_count=len(decomposition.units),
        dep_count=len(decomposition.deps),
        scopes=decomposition.scopes(),
        metadata=decomposition.metadata,
        txn_retries=result.retries,
    )


def _require(row: Any, message: str) -> Any:
    """Assert a RETURNING clause produced a row, without using `assert`.

    Python runs with -O in some deployments, which strips `assert` entirely --
    and a silently missing id here would corrupt the dependency graph.
    """
    if row is None:
        raise RuntimeError(message)
    return row


def resolve_workspace(ref: str, settings: Settings | None = None) -> dict[str, Any]:
    """Look a workspace up by id or by name (most recent wins on ties)."""
    with connection(settings) as conn, conn.cursor() as cur:
        try:
            workspace_id = uuid.UUID(ref)
        except ValueError:
            cur.execute(
                """
                SELECT * FROM workspaces
                WHERE name = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (ref,),
            )
        else:
            cur.execute("SELECT * FROM workspaces WHERE id = %s", (workspace_id,))
        row = cur.fetchone()

    if row is None:
        raise LookupError(f"no workspace matching {ref!r}")
    return dict(row)


def list_workspaces(settings: Settings | None = None) -> list[dict[str, Any]]:
    """Every workspace with its unit counts, newest first."""
    with connection(settings) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT w.id,
                   w.name,
                   w.mode,
                   w.status,
                   w.created_at,
                   count(u.id)                                       AS units,
                   count(u.id) FILTER (WHERE u.status = 'done')      AS done,
                   count(u.id) FILTER (WHERE u.status = 'pending')   AS pending
            FROM workspaces w
            LEFT JOIN work_units u ON u.workspace_id = w.id
            GROUP BY w.id, w.name, w.mode, w.status, w.created_at
            ORDER BY w.created_at DESC
            """
        )
        return [dict(row) for row in cur.fetchall()]


def workspace_summary(
    workspace_id: uuid.UUID, settings: Settings | None = None
) -> dict[str, Any]:
    """Status counts, dependency shape, and conflict tallies for one workspace."""
    with connection(settings) as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM workspaces WHERE id = %s", (workspace_id,))
        workspace = cur.fetchone()
        if workspace is None:
            raise LookupError(f"no workspace with id {workspace_id}")

        cur.execute(
            """
            SELECT status, count(*) AS n
            FROM work_units
            WHERE workspace_id = %s
            GROUP BY status
            """,
            (workspace_id,),
        )
        unit_status = {str(row["status"]): int(row["n"]) for row in cur.fetchall()}

        cur.execute(
            """
            SELECT count(*) AS n
            FROM unit_deps d
            JOIN work_units u ON u.id = d.unit_id
            WHERE u.workspace_id = %s
            """,
            (workspace_id,),
        )
        dep_row = cur.fetchone()

        cur.execute(
            """
            SELECT kind, count(*) AS n
            FROM conflict_log
            WHERE workspace_id = %s
            GROUP BY kind
            """,
            (workspace_id,),
        )
        conflicts = {str(row["kind"]): int(row["n"]) for row in cur.fetchall()}

        cur.execute(
            """
            SELECT status, count(*) AS n
            FROM decisions
            WHERE workspace_id = %s
            GROUP BY status
            """,
            (workspace_id,),
        )
        decisions = {str(row["status"]): int(row["n"]) for row in cur.fetchall()}

        cur.execute(
            """
            SELECT count(*) AS n
            FROM agent_sessions
            WHERE workspace_id = %s AND status = 'running'
            """,
            (workspace_id,),
        )
        agent_row = cur.fetchone()

    return {
        "workspace": dict(workspace),
        "units": unit_status,
        "unit_total": sum(unit_status.values()),
        "deps": int(dep_row["n"]) if dep_row else 0,
        "conflicts": conflicts,
        "decisions": decisions,
        "agents_running": int(agent_row["n"]) if agent_row else 0,
    }


def preview(task_spec: Mapping[str, Any], decomposer: str | None = None) -> Decomposition:
    """Decompose without writing anything. Used by `quorum decompose --dry-run`."""
    name = decomposer or str(task_spec.get("kind", "code_migration"))
    return get_decomposer(name).decompose(task_spec)
