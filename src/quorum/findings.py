"""Findings: what an agent discovered while doing its work.

A finding is an observation that outlives the unit that produced it — "this
module reaches into `requests.Session` internals", "the SSH transport cannot be
migrated without dropping a feature". Most are informational. Some carry
`invalidates=True`, and those are what trigger the Phase 5 cascade.

Findings are embedded on write. That is Phase 3 plumbing, not Phase 4 semantics:
it exercises the `VECTOR` write path and the distributed index end to end long
before any conflict logic depends on them, so an embedding-dimension mismatch
surfaces here rather than in the middle of contradiction detection.
"""

from __future__ import annotations

import uuid
from typing import Any

from psycopg import Cursor

from quorum.config import Settings, get_settings
from quorum.db import connection, run_serializable, vector_literal
from quorum.llm import LLMBackend, get_backend
from quorum.logging import get_logger

log = get_logger(__name__)


def record(
    workspace_id: uuid.UUID,
    content: str,
    *,
    unit_id: uuid.UUID | None = None,
    invalidates: bool = False,
    backend: LLMBackend | None = None,
    embed: bool = True,
    settings: Settings | None = None,
) -> uuid.UUID:
    """Write one finding, embedding it unless told not to."""
    settings = settings or get_settings()

    vector: str | None = None
    if embed:
        model = backend or get_backend(settings)
        embedding = model.embed(content)
        if embedding.dimensions != settings.embed_dim:
            raise ValueError(
                f"embedding width {embedding.dimensions} does not match the "
                f"VECTOR({settings.embed_dim}) column; check QUORUM_EMBED_DIM "
                f"against {model.name}"
            )
        vector = vector_literal(embedding.vector)

    def _insert(cur: Cursor) -> uuid.UUID:
        cur.execute(
            """
            INSERT INTO findings (workspace_id, unit_id, content, embedding, invalidates)
            VALUES (%s, %s, %s, %s::VECTOR, %s)
            RETURNING id
            """,
            (workspace_id, unit_id, content, vector, invalidates),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("findings INSERT did not return an id")
        return row["id"]

    finding_id = run_serializable(_insert, label="finding.record", settings=settings).value
    log.info(
        "finding.recorded",
        extra={
            "finding_id": finding_id,
            "workspace_id": workspace_id,
            "unit_id": unit_id,
            "invalidates": invalidates,
            "embedded": vector is not None,
        },
    )
    return finding_id


def listing(
    workspace_id: uuid.UUID,
    *,
    invalidating_only: bool = False,
    limit: int = 50,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Most recent findings first."""
    clause = "AND invalidates" if invalidating_only else ""
    with connection(settings) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, unit_id, content, invalidates, created_at
              FROM findings
             WHERE workspace_id = %s {clause}
             ORDER BY created_at DESC
             LIMIT %s
            """,  # noqa: S608 -- clause is a fixed literal, not user input
            (workspace_id, limit),
        )
        return [dict(row) for row in cur.fetchall()]


def nearest(
    workspace_id: uuid.UUID,
    vector: list[float],
    *,
    limit: int = 5,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Nearest findings by cosine distance, through the distributed index.

    Phase 3 uses this only to prove the index is reachable and returns sane
    neighbours. Phase 4 builds the semantic-conflict pre-check on the same shape
    of query against `decisions`.
    """
    with connection(settings) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, content, invalidates,
                   1 - (embedding <=> %s::VECTOR) AS similarity
              FROM findings
             WHERE workspace_id = %s AND embedding IS NOT NULL
             ORDER BY embedding <=> %s::VECTOR
             LIMIT %s
            """,
            (vector_literal(vector), workspace_id, vector_literal(vector), limit),
        )
        return [dict(row) for row in cur.fetchall()]
