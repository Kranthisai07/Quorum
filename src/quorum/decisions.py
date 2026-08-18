"""Conflict #2: two agents reaching contradictory conclusions independently.

Agent A decides "standardise the transport layer on httpx". Agent B, three files
away and with no knowledge of A, decides "keep the requests adapter for the unix
socket transport". Both are reasonable. Both cannot hold. Neither agent will
ever find out, because nothing they do overlaps.

The guard runs before a decision commits:

    embed -> ANN search in (workspace_id, scope) -> neighbours above threshold?
      -> classify each with a model: agrees | contradicts | unrelated
        -> contradicts: one wins, the loser is superseded, and the work built on
           the loser is invalidated (phase 5)
        -> agrees:      deduplicate, but only *after* classification
        -> unrelated:   record it

**Classify first, deduplicate second.** This ordering is not a style choice. A
near-duplicate gate placed ahead of classification is a documented way to lose
real conflicts silently: contradiction and near-duplication are indistinguishable
to cosine distance, so a dedup check rejects the contradictory write as "too
similar to something we already have" and the contradiction detector never sees
it. The conflict does not get resolved, and it does not get logged either. It
simply never happened, which is the worst of both.

**The guard has its own time-of-check-to-time-of-use window.** Embedding and
classification are network calls that cannot sit inside a transaction, so
between reading the neighbours and writing the verdict another agent can
supersede the incumbent. The write therefore re-validates the incumbent inside
the serializable transaction, and re-runs the guard if the ground moved. Getting
this wrong would mean superseding a decision that was already superseded --
resolving a conflict against a decision nobody holds any more.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from psycopg import Cursor

from quorum import conflicts
from quorum.classifier import Classifier, Judgement, build_classifier
from quorum.config import Settings, get_settings
from quorum.db import connection, run_autocommit, run_serializable, vector_literal
from quorum.llm import LLMBackend, get_backend
from quorum.logging import get_logger

log = get_logger(__name__)

Mode = Literal["safe", "naive"]

Status = Literal[
    "recorded",              # no near neighbour, or an unrelated one: written
    "recorded_superseding",  # contradicted an incumbent and won
    "rejected_contradicted", # contradicted an incumbent and lost
    "rejected_duplicate",    # agreed with an existing decision, so not rewritten
]

# How many times to re-run the guard when the incumbent changes underneath it.
MAX_RECHECKS = 3

# How many neighbours the ANN query returns before thresholding. Small on
# purpose: the index exists to make the model call affordable, and classifying
# a long tail of weak matches would give that back.
NEIGHBOUR_LIMIT = 5


@dataclass(frozen=True)
class Neighbour:
    """An existing decision the ANN search considers close enough to matter."""

    id: uuid.UUID
    agent_id: uuid.UUID | None
    statement: str
    rationale: str | None
    similarity: float


@dataclass
class DecisionOutcome:
    """What happened to a proposed decision, and why."""

    status: Status
    decision_id: uuid.UUID | None = None
    scope: str = ""
    statement: str = ""
    neighbours: list[Neighbour] = field(default_factory=list)
    judgement: Judgement | None = None
    superseded_id: uuid.UUID | None = None
    conflict_id: uuid.UUID | None = None
    rechecks: int = 0

    @property
    def written(self) -> bool:
        return self.status in ("recorded", "recorded_superseding")

    @property
    def conflicted(self) -> bool:
        return self.status in ("recorded_superseding", "rejected_contradicted")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "decision_id": str(self.decision_id) if self.decision_id else None,
            "scope": self.scope,
            "statement": self.statement,
            "neighbours": [
                {"id": str(n.id), "similarity": round(n.similarity, 4), "statement": n.statement}
                for n in self.neighbours
            ],
            "judgement": self.judgement.to_dict() if self.judgement else None,
            "superseded_id": str(self.superseded_id) if self.superseded_id else None,
            "conflict_id": str(self.conflict_id) if self.conflict_id else None,
            "rechecks": self.rechecks,
        }


def propose(
    workspace_id: uuid.UUID,
    scope: str,
    statement: str,
    *,
    agent_id: uuid.UUID | None = None,
    rationale: str | None = None,
    mode: Mode = "safe",
    backend: LLMBackend | None = None,
    classifier: Classifier | None = None,
    threshold: float | None = None,
    settings: Settings | None = None,
) -> DecisionOutcome:
    """Propose a decision, running the semantic guard first in safe mode."""
    settings = settings or get_settings()
    model = backend or get_backend(settings)

    if mode == "naive":
        return _propose_naive(
            workspace_id, scope, statement, agent_id, rationale, model, settings
        )

    judge = classifier or build_classifier(model)
    # The backend owns the threshold: it is a property of the embedding model,
    # not of Quorum. See LLMBackend.similarity_threshold.
    cutoff = model.similarity_threshold if threshold is None else threshold
    embedding = model.embed(statement)
    if embedding.dimensions != settings.embed_dim:
        raise ValueError(
            f"embedding width {embedding.dimensions} does not match the "
            f"VECTOR({settings.embed_dim}) column; check QUORUM_EMBED_DIM"
        )
    vector = embedding.vector

    for attempt in range(MAX_RECHECKS):
        neighbours = nearest(
            workspace_id, scope, vector, threshold=cutoff, settings=settings
        )

        if not neighbours:
            outcome = _write(
                workspace_id, scope, statement, agent_id, rationale, vector, settings
            )
            outcome.rechecks = attempt
            log.info("decision.recorded", extra=_event(outcome, "no_near_neighbour"))
            return outcome

        # Classify BEFORE any deduplication, and classify *every* candidate --
        # not just the nearest. A new decision can contradict any active
        # decision in the scope, and nearest-by-cosine is not the same as
        # most-opposed: two statements that agree often share more vocabulary
        # than two that conflict. Stopping at neighbours[0] would let a
        # contradiction two places down the list through unexamined.
        #
        # A contradiction found anywhere outranks a duplicate found earlier,
        # which is the same classify-then-dedupe rule applied within the loop.
        contradiction: tuple[Neighbour, Judgement] | None = None
        duplicate: tuple[Neighbour, Judgement] | None = None

        for candidate in neighbours:
            verdict = judge.classify(scope, candidate.statement, statement)
            if verdict.is_conflict:
                contradiction = (candidate, verdict)
                break
            if verdict.is_duplicate and duplicate is None:
                duplicate = (candidate, verdict)

        pair = contradiction or duplicate
        if pair is None:
            # Near neighbours existed, but none of them agree or conflict.
            outcome = _write(
                workspace_id, scope, statement, agent_id, rationale, vector, settings
            )
            outcome.neighbours = neighbours
            outcome.rechecks = attempt
            log.info("decision.recorded", extra=_event(outcome, "neighbours_unrelated"))
            return outcome

        incumbent, judgement = pair

        outcome = _apply(
            workspace_id=workspace_id,
            scope=scope,
            statement=statement,
            agent_id=agent_id,
            rationale=rationale,
            vector=vector,
            incumbent=incumbent,
            judgement=judgement,
            neighbours=neighbours,
            settings=settings,
        )
        if outcome is not None:
            outcome.rechecks = attempt
            log.info("decision.resolved", extra=_event(outcome, judgement.relation))
            return outcome

        # The incumbent moved while we were classifying. Re-run the guard
        # against whatever holds the scope now.
        log.info(
            "decision.recheck",
            extra={
                "workspace_id": workspace_id,
                "scope": scope,
                "incumbent": str(incumbent.id),
                "attempt": attempt + 1,
            },
        )

    raise RuntimeError(
        f"semantic guard did not settle for scope {scope!r} after {MAX_RECHECKS} "
        "rechecks; the scope is being rewritten faster than it can be adjudicated"
    )


def _event(outcome: DecisionOutcome, reason: str) -> dict[str, Any]:
    return {
        "decision_id": outcome.decision_id,
        "scope": outcome.scope,
        "decision_status": outcome.status,
        "reason": reason,
        "neighbours": len(outcome.neighbours),
        "top_similarity": round(outcome.neighbours[0].similarity, 4)
        if outcome.neighbours
        else None,
        "superseded_id": outcome.superseded_id,
    }


def _apply(
    *,
    workspace_id: uuid.UUID,
    scope: str,
    statement: str,
    agent_id: uuid.UUID | None,
    rationale: str | None,
    vector: list[float],
    incumbent: Neighbour,
    judgement: Judgement,
    neighbours: list[Neighbour],
    settings: Settings,
) -> DecisionOutcome | None:
    """Commit the verdict, or return None if the incumbent changed underneath us."""

    def _txn(cur: Cursor) -> DecisionOutcome | None:
        # Re-validate inside the transaction. If the incumbent is no longer
        # active, the judgement was made against a decision that no longer
        # governs this scope, and acting on it would supersede a ghost.
        cur.execute(
            "SELECT id, status, agent_id FROM decisions WHERE id = %s",
            (incumbent.id,),
        )
        current = cur.fetchone()
        if current is None or current["status"] != "active":
            return None

        if judgement.is_duplicate:
            # Deduplicate -- but only now, having established it is agreement
            # and not contradiction.
            return DecisionOutcome(
                status="rejected_duplicate",
                decision_id=incumbent.id,
                scope=scope,
                statement=statement,
                neighbours=neighbours,
                judgement=judgement,
            )

        if not judgement.is_conflict:
            new_id = _insert(cur, workspace_id, scope, statement, agent_id, rationale, vector)
            return DecisionOutcome(
                status="recorded",
                decision_id=new_id,
                scope=scope,
                statement=statement,
                neighbours=neighbours,
                judgement=judgement,
            )

        # Contradiction. Exactly one survives, and it is recorded either way.
        challenger_wins = judgement.winner == "challenger"
        detail = {
            "reason": "contradictory_decisions",
            "scope": scope,
            "incumbent_id": str(incumbent.id),
            "incumbent_statement": incumbent.statement,
            "challenger_statement": statement,
            "similarity": round(incumbent.similarity, 4),
            "relation": judgement.relation,
            "confidence": judgement.confidence,
            "winner": judgement.winner,
            "reasoning": judgement.reasoning,
            "classifier": judgement.model,
        }

        if challenger_wins:
            new_id = _insert(
                cur, workspace_id, scope, statement, agent_id, rationale, vector,
                supersedes_id=incumbent.id,
            )
            cur.execute(
                "UPDATE decisions SET status = 'superseded' WHERE id = %s", (incumbent.id,)
            )
            detail["superseded_id"] = str(incumbent.id)
            conflict_id = conflicts.record(
                cur,
                workspace_id=workspace_id,
                kind="semantic",
                agents=[agent_id, current["agent_id"]],
                detail=detail,
                resolution="challenger_superseded_incumbent",
            )
            return DecisionOutcome(
                status="recorded_superseding",
                decision_id=new_id,
                scope=scope,
                statement=statement,
                neighbours=neighbours,
                judgement=judgement,
                superseded_id=incumbent.id,
                conflict_id=conflict_id,
            )

        detail["rejected_challenger"] = True
        conflict_id = conflicts.record(
            cur,
            workspace_id=workspace_id,
            kind="semantic",
            agents=[agent_id, current["agent_id"]],
            detail=detail,
            resolution="incumbent_upheld",
        )
        return DecisionOutcome(
            status="rejected_contradicted",
            decision_id=incumbent.id,
            scope=scope,
            statement=statement,
            neighbours=neighbours,
            judgement=judgement,
            conflict_id=conflict_id,
        )

    return run_serializable(_txn, label="decision.apply", settings=settings).value


def _insert(
    cur: Cursor,
    workspace_id: uuid.UUID,
    scope: str,
    statement: str,
    agent_id: uuid.UUID | None,
    rationale: str | None,
    vector: list[float],
    *,
    supersedes_id: uuid.UUID | None = None,
) -> uuid.UUID:
    cur.execute(
        """
        INSERT INTO decisions
            (workspace_id, agent_id, scope, statement, rationale, embedding, supersedes_id)
        VALUES (%s, %s, %s, %s, %s, %s::VECTOR, %s)
        RETURNING id
        """,
        (
            workspace_id,
            agent_id,
            scope,
            statement,
            rationale,
            vector_literal(vector),
            supersedes_id,
        ),
    )
    row = cur.fetchone()
    if row is None:
        raise RuntimeError("decisions INSERT did not return an id")
    return row["id"]


def _write(
    workspace_id: uuid.UUID,
    scope: str,
    statement: str,
    agent_id: uuid.UUID | None,
    rationale: str | None,
    vector: list[float],
    settings: Settings,
) -> DecisionOutcome:
    """Unconditional write, for the case with no near neighbour at all."""

    def _txn(cur: Cursor) -> uuid.UUID:
        return _insert(cur, workspace_id, scope, statement, agent_id, rationale, vector)

    new_id = run_serializable(_txn, label="decision.write", settings=settings).value
    return DecisionOutcome(
        status="recorded", decision_id=new_id, scope=scope, statement=statement
    )


def _propose_naive(
    workspace_id: uuid.UUID,
    scope: str,
    statement: str,
    agent_id: uuid.UUID | None,
    rationale: str | None,
    model: LLMBackend,
    settings: Settings,
) -> DecisionOutcome:
    """No search, no classification, no supersession. Just write it.

    This is how an agent memory layer without a coordination story behaves, and
    it is why `naive` workspaces end up holding two contradictory decisions that
    are both marked `active` -- each one recorded successfully, by an agent that
    had no way to know the other existed.
    """
    vector = model.embed(statement).vector

    def _txn(cur: Cursor) -> uuid.UUID:
        return _insert(cur, workspace_id, scope, statement, agent_id, rationale, vector)

    new_id = run_autocommit(_txn, label="decision.write.naive", settings=settings).value
    log.info(
        "decision.recorded",
        extra={"decision_id": new_id, "scope": scope, "decision_mode": "naive"},
    )
    return DecisionOutcome(
        status="recorded", decision_id=new_id, scope=scope, statement=statement
    )


def nearest(
    workspace_id: uuid.UUID,
    scope: str,
    vector: list[float],
    *,
    threshold: float = 0.0,
    limit: int = NEIGHBOUR_LIMIT,
    settings: Settings | None = None,
) -> list[Neighbour]:
    """Active decisions in this scope, nearest first, above `threshold`.

    Goes through the distributed vector index, prefixed by `workspace_id`, so
    one workspace never searches another's decisions.
    """
    literal = vector_literal(vector)
    with connection(settings) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, agent_id, statement, rationale,
                   1 - (embedding <=> %s::VECTOR) AS similarity
              FROM decisions
             WHERE workspace_id = %s
               AND scope = %s
               AND status = 'active'
               AND embedding IS NOT NULL
             ORDER BY embedding <=> %s::VECTOR
             LIMIT %s
            """,
            (literal, workspace_id, scope, literal, limit),
        )
        rows = cur.fetchall()

    return [
        Neighbour(
            id=row["id"],
            agent_id=row["agent_id"],
            statement=str(row["statement"]),
            rationale=row["rationale"],
            similarity=float(row["similarity"]),
        )
        for row in rows
        if float(row["similarity"]) >= threshold
    ]


def listing(
    workspace_id: uuid.UUID,
    *,
    scope: str | None = None,
    include_superseded: bool = True,
    limit: int = 50,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    """Decisions in a workspace, newest first."""
    clauses = ["workspace_id = %s"]
    params: list[Any] = [workspace_id]
    if scope is not None:
        clauses.append("scope = %s")
        params.append(scope)
    if not include_superseded:
        clauses.append("status = 'active'")
    params.append(limit)

    with connection(settings) as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT id, agent_id, scope, statement, rationale, status,
                   supersedes_id, created_at
              FROM decisions
             WHERE {" AND ".join(clauses)}
             ORDER BY created_at DESC
             LIMIT %s
            """,  # noqa: S608 -- clauses are fixed literals, values are bound
            params,
        )
        return [dict(row) for row in cur.fetchall()]


def contradictions_outstanding(
    workspace_id: uuid.UUID, settings: Settings | None = None
) -> list[dict[str, Any]]:
    """Scopes that still hold more than one active decision.

    In `safe` mode this should always be empty: the guard resolves every
    contradiction before it commits. In `naive` mode it is the corruption, made
    countable -- and it is the headline number in the mode comparison.
    """
    with connection(settings) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT scope, count(*) AS active_decisions,
                   array_agg(statement) AS statements
              FROM decisions
             WHERE workspace_id = %s AND status = 'active'
             GROUP BY scope
            HAVING count(*) > 1
             ORDER BY count(*) DESC
            """,
            (workspace_id,),
        )
        return [dict(row) for row in cur.fetchall()]
