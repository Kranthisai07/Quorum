"""Conflict #2: semantic contradiction between independently-made decisions.

The tests that matter most here are the ones guarding *ordering*:

* `test_a_contradiction_that_is_also_a_near_duplicate_is_not_deduped` -- the
  documented failure where a dedup gate ahead of classification makes real
  conflicts vanish silently.
* `test_a_contradiction_below_the_nearest_neighbour_is_still_found` -- nearest
  by cosine is not most-opposed.
* `test_the_guard_rechecks_when_the_incumbent_moves` -- the guard's own
  time-of-check-to-time-of-use window, across a model call that cannot sit
  inside a transaction.
"""

from __future__ import annotations

import uuid

import pytest

from quorum import conflicts, decisions
from quorum.classifier import (
    HeuristicClassifier,
    Judgement,
    ModelClassifier,
    _parse,
    build_classifier,
)
from quorum.llm import Completion, StubBackend


class ScriptedClassifier:
    """Returns pre-arranged verdicts so ordering can be asserted exactly."""

    name = "scripted"

    def __init__(self, *verdicts: Judgement) -> None:
        self.verdicts = list(verdicts)
        self.calls: list[tuple[str, str]] = []

    def classify(self, scope: str, incumbent: str, challenger: str) -> Judgement:
        self.calls.append((incumbent, challenger))
        if self.verdicts:
            return self.verdicts.pop(0)
        return Judgement(relation="unrelated", confidence=0.1, reasoning="exhausted")


def _contradicts(winner: str = "incumbent") -> Judgement:
    return Judgement(
        relation="contradicts", confidence=0.9, reasoning="opposed", winner=winner  # type: ignore[arg-type]
    )


def _agrees() -> Judgement:
    return Judgement(relation="agrees", confidence=0.9, reasoning="same thing")


def _unrelated() -> Judgement:
    return Judgement(relation="unrelated", confidence=0.9, reasoning="different question")


class TestClassifierParsing:
    def test_plain_json(self):
        judgement = _parse('{"relation":"agrees","confidence":0.8,"reasoning":"r"}', "m")
        assert judgement.relation == "agrees"
        assert judgement.confidence == 0.8

    def test_fenced_json(self):
        judgement = _parse('```json\n{"relation":"unrelated","confidence":0.4}\n```', "m")
        assert judgement.relation == "unrelated"

    def test_json_embedded_in_prose(self):
        judgement = _parse(
            'Sure!\n{"relation":"contradicts","winner":"challenger"}\nHope that helps',
            "m",
        )
        assert judgement.relation == "contradicts"
        assert judgement.winner == "challenger"

    def test_unparseable_escalates_rather_than_passing(self):
        """Silence must never be read as 'no conflict'."""
        judgement = _parse("I could not decide.", "m")
        assert judgement.relation == "contradicts"
        assert judgement.winner == "incumbent"
        assert judgement.confidence == 0.0

    def test_an_unknown_relation_escalates(self):
        judgement = _parse('{"relation":"maybe","confidence":0.9}', "m")
        assert judgement.relation == "contradicts"

    def test_contradiction_without_a_winner_defaults_to_the_incumbent(self):
        judgement = _parse('{"relation":"contradicts","confidence":0.9}', "m")
        assert judgement.winner == "incumbent"

    def test_confidence_is_clamped(self):
        assert _parse('{"relation":"agrees","confidence":5}', "m").confidence == 1.0
        assert _parse('{"relation":"agrees","confidence":-2}', "m").confidence == 0.0

    def test_a_refusal_escalates(self):
        class Refusing:
            name = "refusing"
            similarity_threshold = 0.5

            def complete(self, prompt, *, system=None, max_tokens=8192):
                return Completion(text="", model="m", stop_reason="refusal")

            def embed(self, text):
                raise NotImplementedError

            def health(self):
                raise NotImplementedError

        judgement = ModelClassifier(Refusing()).classify("scope", "a", "b")
        assert judgement.relation == "contradicts"
        assert "declined" in judgement.reasoning


class TestHeuristicClassifier:
    def test_keep_versus_replace_is_a_contradiction(self):
        judgement = HeuristicClassifier().classify(
            "transport-adapter",
            "keep the requests adapter for the unix socket transport",
            "replace every adapter and standardise on httpx",
        )
        assert judgement.relation == "contradicts"

    def test_restatements_agree(self):
        judgement = HeuristicClassifier().classify(
            "timeout-policy",
            "timeouts default to 30 seconds on every client",
            "timeouts default to 30 seconds on every client instance",
        )
        assert judgement.relation == "agrees"

    def test_different_questions_are_unrelated(self):
        judgement = HeuristicClassifier().classify(
            "streaming", "stream responses lazily", "tls verification stays enabled"
        )
        assert judgement.relation == "unrelated"

    def test_the_stub_backend_gets_the_heuristic(self):
        assert isinstance(build_classifier(StubBackend()), HeuristicClassifier)


class TestThresholdOwnership:
    def test_the_backend_owns_the_threshold(self):
        """A single shared threshold across backends is a correctness bug.

        The stub's lexical vectors put a directly opposed pair at ~0.37, well
        under the 0.82 tuned for Titan. Sharing one constant makes the guard
        stop classifying anything, silently.
        """
        from quorum.config import get_settings
        from quorum.llm import BedrockBackend

        assert StubBackend().similarity_threshold == 0.0
        assert BedrockBackend().similarity_threshold == get_settings().semantic_threshold


@pytest.mark.integration
class TestSemanticGuard:
    @pytest.fixture
    def workspace(self, clean_workspaces, tiny_task_spec):
        from quorum.workspace import seed_workspace

        seeded = seed_workspace(
            name=f"sem-{uuid.uuid4().hex[:8]}",
            task_spec=tiny_task_spec,
            mode="safe",
            settings=clean_workspaces,
        )
        return seeded.workspace_id, clean_workspaces

    def test_a_lone_decision_is_recorded(self, workspace):
        workspace_id, settings = workspace
        outcome = decisions.propose(
            workspace_id, "transport-adapter", "adopt httpx everywhere", settings=settings
        )
        assert outcome.status == "recorded"
        assert outcome.decision_id is not None

    def test_an_unrelated_decision_is_recorded_alongside(self, workspace):
        workspace_id, settings = workspace
        judge = ScriptedClassifier(_unrelated())
        decisions.propose(workspace_id, "s", "first statement", settings=settings)

        outcome = decisions.propose(
            workspace_id, "s", "second statement", classifier=judge, settings=settings
        )

        assert outcome.status == "recorded"
        assert len(decisions.listing(workspace_id, settings=settings)) == 2

    def test_a_contradiction_upholds_the_incumbent(self, workspace):
        workspace_id, settings = workspace
        first = decisions.propose(workspace_id, "s", "adopt httpx", settings=settings)

        outcome = decisions.propose(
            workspace_id,
            "s",
            "keep requests",
            classifier=ScriptedClassifier(_contradicts("incumbent")),
            settings=settings,
        )

        assert outcome.status == "rejected_contradicted"
        assert outcome.conflict_id is not None
        active = decisions.listing(
            workspace_id, include_superseded=False, settings=settings
        )
        assert [row["id"] for row in active] == [first.decision_id]

    def test_a_winning_challenger_supersedes_the_incumbent(self, workspace):
        workspace_id, settings = workspace
        first = decisions.propose(workspace_id, "s", "adopt httpx", settings=settings)

        outcome = decisions.propose(
            workspace_id,
            "s",
            "keep requests",
            classifier=ScriptedClassifier(_contradicts("challenger")),
            settings=settings,
        )

        assert outcome.status == "recorded_superseding"
        assert outcome.superseded_id == first.decision_id

        rows = {row["id"]: row for row in decisions.listing(workspace_id, settings=settings)}
        assert rows[first.decision_id]["status"] == "superseded"
        assert rows[outcome.decision_id]["status"] == "active"
        assert rows[outcome.decision_id]["supersedes_id"] == first.decision_id

    def test_a_duplicate_is_not_written_twice(self, workspace):
        workspace_id, settings = workspace
        first = decisions.propose(workspace_id, "s", "adopt httpx", settings=settings)

        outcome = decisions.propose(
            workspace_id,
            "s",
            "adopt httpx everywhere",
            classifier=ScriptedClassifier(_agrees()),
            settings=settings,
        )

        assert outcome.status == "rejected_duplicate"
        assert outcome.decision_id == first.decision_id
        assert len(decisions.listing(workspace_id, settings=settings)) == 1

    def test_a_contradiction_that_is_also_a_near_duplicate_is_not_deduped(self, workspace):
        """The documented failure: dedupe ahead of classification loses conflicts.

        Contradiction and near-duplication are indistinguishable to cosine
        distance. If similarity alone could reject a write, this decision would
        disappear -- not resolved, not logged, just gone.
        """
        workspace_id, settings = workspace
        decisions.propose(workspace_id, "s", "adopt httpx for all transports", settings=settings)

        outcome = decisions.propose(
            workspace_id,
            "s",
            "adopt httpx for all transports except unix sockets",
            classifier=ScriptedClassifier(_contradicts("challenger")),
            settings=settings,
        )

        assert outcome.status == "recorded_superseding", (
            "a contradiction was silently swallowed as a near-duplicate"
        )
        assert outcome.conflict_id is not None

    def test_a_contradiction_below_the_nearest_neighbour_is_still_found(self, workspace):
        """Nearest by cosine is not the same as most-opposed."""
        workspace_id, settings = workspace
        decisions.propose(workspace_id, "s", "alpha statement", settings=settings)
        decisions.propose(
            workspace_id,
            "s",
            "beta statement",
            classifier=ScriptedClassifier(_unrelated()),
            settings=settings,
        )

        # First candidate is unrelated; the contradiction is the second.
        judge = ScriptedClassifier(_unrelated(), _contradicts("incumbent"))
        outcome = decisions.propose(
            workspace_id, "s", "gamma statement", classifier=judge, settings=settings
        )

        assert len(judge.calls) == 2, "stopped after the nearest neighbour"
        assert outcome.status == "rejected_contradicted"

    def test_a_contradiction_outranks_an_earlier_duplicate(self, workspace):
        workspace_id, settings = workspace
        decisions.propose(workspace_id, "s", "alpha", settings=settings)
        decisions.propose(
            workspace_id, "s", "beta", classifier=ScriptedClassifier(_unrelated()),
            settings=settings,
        )

        judge = ScriptedClassifier(_agrees(), _contradicts("challenger"))
        outcome = decisions.propose(
            workspace_id, "s", "gamma", classifier=judge, settings=settings
        )

        assert outcome.status == "recorded_superseding"

    def test_the_conflict_is_recorded_with_both_statements(self, workspace):
        workspace_id, settings = workspace
        decisions.propose(workspace_id, "s", "adopt httpx", settings=settings)
        decisions.propose(
            workspace_id,
            "s",
            "keep requests",
            classifier=ScriptedClassifier(_contradicts("challenger")),
            settings=settings,
        )

        rows = conflicts.listing(workspace_id, kind="semantic", settings=settings)
        assert len(rows) == 1
        detail = rows[0]["detail"]
        assert detail["incumbent_statement"] == "adopt httpx"
        assert detail["challenger_statement"] == "keep requests"
        assert detail["relation"] == "contradicts"
        assert "similarity" in detail
        assert rows[0]["resolution"] == "challenger_superseded_incumbent"

    def test_safe_mode_never_leaves_two_active_decisions_in_a_scope(self, workspace):
        workspace_id, settings = workspace
        decisions.propose(workspace_id, "s", "adopt httpx", settings=settings)
        decisions.propose(
            workspace_id, "s", "keep requests",
            classifier=ScriptedClassifier(_contradicts("challenger")),
            settings=settings,
        )

        assert decisions.contradictions_outstanding(workspace_id, settings) == []

    def test_the_guard_rechecks_when_the_incumbent_moves(self, workspace):
        """The guard's own TOCTOU window, across a model call.

        The classifier supersedes the incumbent as a side effect, simulating
        another agent winning the scope while this one was still deliberating.
        Acting on the stale judgement would supersede a decision that no longer
        governs anything.
        """
        workspace_id, settings = workspace
        first = decisions.propose(workspace_id, "s", "adopt httpx", settings=settings)

        class MovesTheGround:
            name = "moves"

            def __init__(self) -> None:
                self.calls = 0

            def classify(self, scope, incumbent, challenger):
                self.calls += 1
                if self.calls == 1:
                    from quorum.db import run_serializable

                    run_serializable(
                        lambda cur: cur.execute(
                            "UPDATE decisions SET status = 'superseded' WHERE id = %s",
                            (first.decision_id,),
                        ),
                        label="test.supersede",
                        settings=settings,
                    )
                return _contradicts("challenger")

        judge = MovesTheGround()
        outcome = decisions.propose(
            workspace_id, "s", "keep requests", classifier=judge, settings=settings
        )

        assert judge.calls >= 1
        assert outcome.rechecks >= 1, "the guard acted on a stale incumbent"
        assert outcome.written


@pytest.mark.integration
class TestNaiveMode:
    """The control group: no search, no classification, no supersession."""

    @pytest.fixture
    def workspace(self, clean_workspaces, tiny_task_spec):
        from quorum.workspace import seed_workspace

        seeded = seed_workspace(
            name=f"sem-naive-{uuid.uuid4().hex[:8]}",
            task_spec=tiny_task_spec,
            mode="naive",
            settings=clean_workspaces,
        )
        return seeded.workspace_id, clean_workspaces

    def test_contradictory_decisions_are_both_left_active(self, workspace):
        workspace_id, settings = workspace

        decisions.propose(
            workspace_id, "s", "adopt httpx everywhere", mode="naive", settings=settings
        )
        decisions.propose(
            workspace_id, "s", "keep requests for unix sockets", mode="naive", settings=settings
        )

        outstanding = decisions.contradictions_outstanding(workspace_id, settings)
        assert len(outstanding) == 1
        assert outstanding[0]["active_decisions"] == 2

    def test_naive_logs_no_semantic_conflicts(self, workspace):
        workspace_id, settings = workspace
        decisions.propose(workspace_id, "s", "adopt httpx", mode="naive", settings=settings)
        decisions.propose(workspace_id, "s", "keep requests", mode="naive", settings=settings)

        assert conflicts.counts(workspace_id, settings)["total"] == 0

    def test_naive_still_embeds_so_the_data_is_comparable(self, workspace):
        """Both modes store embeddings; only the guard differs."""
        workspace_id, settings = workspace
        decisions.propose(workspace_id, "s", "adopt httpx", mode="naive", settings=settings)

        probe = StubBackend(settings).embed("adopt httpx")
        assert decisions.nearest(workspace_id, "s", probe.vector, settings=settings)
