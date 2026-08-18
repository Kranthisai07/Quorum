"""The claim stress tests: the backbone of the correctness claim.

A single green concurrency run proves nothing, so safe mode is run hundreds of
times and asserted to have handed every unit to exactly one agent, every time.
The same harness, same agents, same workspace shape, run in naive mode, is
asserted to *fail* -- because a control group that never fails would not be
evidence of anything.

Set `QUORUM_STRESS_ITERATIONS` to trade runtime for confidence.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest

from quorum import claims, conflicts, runner, sessions, stress
from quorum.workspace import seed_workspace

pytestmark = pytest.mark.integration

AGENTS = 8
ITERATIONS = int(os.environ.get("QUORUM_STRESS_ITERATIONS", "200"))
NAIVE_ITERATIONS = max(5, ITERATIONS // 8)


@pytest.fixture
def contended_workspace(clean_workspaces, fixture_task_spec):
    """A real 16-unit workspace, dedicated to one stress run."""
    seeded = seed_workspace(
        name=f"stress-{uuid.uuid4().hex[:8]}",
        task_spec=fixture_task_spec,
        mode="safe",
        settings=clean_workspaces,
    )
    return seeded.workspace_id, clean_workspaces


@pytest.mark.timeout(900)
class TestSafeMode:
    """N agents, M units, hundreds of rounds, zero tolerance."""

    def test_never_double_claims(self, contended_workspace):
        workspace_id, settings = contended_workspace

        report = stress.run_stress(
            workspace_id,
            agents=AGENTS,
            iterations=ITERATIONS,
            mode="safe",
            settings=settings,
        )

        assert report.duplicate_claims == 0, (
            f"{report.duplicate_claims} double-claims across {ITERATIONS} iterations: "
            f"{report.samples[:3]}"
        )
        assert report.unclaimed_units == 0, "work was dropped rather than duplicated"
        assert report.max_agents_on_one_unit == 1
        assert report.ok

    def test_every_unit_is_claimed_exactly_once_per_round(self, contended_workspace):
        workspace_id, settings = contended_workspace

        report = stress.run_stress(
            workspace_id, agents=AGENTS, iterations=20, mode="safe", settings=settings
        )

        assert report.total_claims == report.units * 20
        assert report.iterations_with_duplicates == 0

    def test_contention_is_real_and_recorded(self, contended_workspace):
        """Zero duplicates could also mean the agents never actually raced.

        Note what is *not* asserted here: serialization retries. Under
        `FOR UPDATE` the losers block on the lock and commit cleanly once it is
        handed over, so a correct, heavily contended run has zero aborts. Using
        retry count as the contention signal is what hid the problem the first
        time round.
        """
        workspace_id, settings = contended_workspace

        report = stress.run_stress(
            workspace_id, agents=AGENTS, iterations=20, mode="safe", settings=settings
        )

        assert report.conflicts_logged > 0, "contention happened but was never logged"
        assert report.max_losers_on_one_unit >= 2, (
            "expected several agents to lose the same race"
        )

    def test_detection_does_not_manufacture_aborts(self, contended_workspace):
        """Instrumentation must not perturb what it measures.

        The conflict peek originally ran inside the claim transaction, where it
        left a read at a timestamp the winner's commit invalidated -- so the
        transaction could not refresh and aborted. It reported ~1.9 retries per
        claim that existed only because it was watching. Moved outside the
        transaction, the same detection costs no aborts at all.
        """
        workspace_id, settings = contended_workspace

        report = stress.run_stress(
            workspace_id, agents=AGENTS, iterations=20, mode="safe", settings=settings
        )

        assert report.conflicts_logged > 0, "detection was not actually active"
        assert report.txn_retries <= report.total_claims * 0.05, (
            f"{report.txn_retries} aborts over {report.total_claims} claims -- "
            "the detector is inducing the contention it reports"
        )

    def test_detection_changes_visibility_not_correctness(self, contended_workspace):
        """Turning the conflict feed off must cost evidence, never safety."""
        workspace_id, settings = contended_workspace

        watched = stress.run_stress(
            workspace_id,
            agents=AGENTS,
            iterations=10,
            mode="safe",
            detect_contention=True,
            settings=settings,
        )
        blind = stress.run_stress(
            workspace_id,
            agents=AGENTS,
            iterations=10,
            mode="safe",
            detect_contention=False,
            settings=settings,
        )

        assert watched.ok and blind.ok
        assert watched.duplicate_claims == blind.duplicate_claims == 0
        assert watched.conflicts_logged > 0
        assert blind.conflicts_logged == 0, (
            "an empty conflict feed on a contended workspace -- exactly the "
            "false reassurance the whole project is about"
        )

    def test_conflict_rows_name_both_sides(self, contended_workspace):
        workspace_id, settings = contended_workspace
        stress.run_stress(
            workspace_id, agents=4, iterations=10, mode="safe", settings=settings
        )

        rows = [
            row
            for row in conflicts.listing(workspace_id, kind="claim", limit=200, settings=settings)
            if row["detail"].get("reason") == "concurrent_claim"
        ]
        assert rows, "no concurrent-claim conflicts recorded"
        sample = rows[0]
        assert len(sample["agents"]) == 2, "a claim conflict has a winner and a loser"
        assert sample["detail"]["winner"] != sample["detail"]["loser"]
        assert sample["resolution"] == "serialized_by_cockroachdb"


@pytest.mark.timeout(600)
class TestNaiveMode:
    """The control group. These tests assert that it breaks."""

    def test_double_claims_every_single_round_under_a_barrier(self, contended_workspace):
        workspace_id, settings = contended_workspace

        report = stress.run_stress(
            workspace_id,
            agents=AGENTS,
            iterations=NAIVE_ITERATIONS,
            mode="naive",
            barrier=True,
            settings=settings,
        )

        assert report.ok is False
        assert report.iterations_with_duplicates == NAIVE_ITERATIONS, (
            "the barrier is supposed to make this deterministic"
        )
        assert report.max_agents_on_one_unit == AGENTS, (
            "every agent should have claimed the same unit"
        )

    def test_double_claims_without_any_help(self, contended_workspace):
        """The barrier does not manufacture the bug; it only removes the luck."""
        workspace_id, settings = contended_workspace

        report = stress.naive_natural_rate(
            workspace_id, agents=AGENTS, iterations=NAIVE_ITERATIONS, settings=settings
        )

        assert report.duplicate_claims > 0, (
            "naive mode did not race on its own -- the barrier result would then "
            "be an artifact rather than evidence"
        )

    def test_never_notices_its_own_conflicts(self, contended_workspace):
        """Naive mode's conflict log stays empty while it corrupts the workspace."""
        workspace_id, settings = contended_workspace

        report = stress.run_stress(
            workspace_id,
            agents=AGENTS,
            iterations=5,
            mode="naive",
            barrier=True,
            settings=settings,
        )

        assert report.duplicate_claims > 0
        assert report.conflicts_logged == 0
        assert conflicts.counts(workspace_id, settings)["total"] == 0

    def test_the_same_harness_produces_opposite_verdicts(self, contended_workspace):
        """Same agents, same units, same code path -- only the mode differs."""
        workspace_id, settings = contended_workspace

        safe = stress.run_stress(
            workspace_id, agents=AGENTS, iterations=5, mode="safe", settings=settings
        )
        naive = stress.run_stress(
            workspace_id,
            agents=AGENTS,
            iterations=5,
            mode="naive",
            barrier=True,
            settings=settings,
        )

        assert safe.ok is True
        assert naive.ok is False
        assert safe.agents == naive.agents
        assert safe.units == naive.units


@pytest.mark.timeout(300)
class TestAgentDeath:
    """An agent is SIGKILLed while holding a lease. No work may be lost."""

    def test_dead_agent_loses_no_work_and_blocks_nobody(self, clean_workspaces, tiny_task_spec):
        settings = clean_workspaces
        seeded = seed_workspace(
            name=f"death-{uuid.uuid4().hex[:8]}",
            task_spec=tiny_task_spec,
            mode="safe",
            settings=settings,
        )
        workspace_id = seeded.workspace_id

        # A real OS process, so the kill is a real kill.
        doomed = runner.spawn(
            {
                "workspace_id": str(workspace_id),
                "agent_name": "doomed",
                "mode": "safe",
                "max_units": 1,
                # `sleep` mode, so the agent holds its lease long enough to be
                # killed while holding it. Real migration work finishes in
                # milliseconds against the stub backend.
                "work_mode": "sleep",
                "work_seconds": 60,  # it will never finish
                "lease_seconds": 2,
            }
        )
        try:
            held = _wait_for_claim(workspace_id, settings, timeout=30)
            doomed.kill()
            doomed.wait(timeout=15)
        finally:
            if doomed.poll() is None:
                doomed.kill()

        # While the lease is still live the unit stays held: a slow agent must
        # not have its work stolen just because it went quiet for a moment.
        assert claims.reap_expired(workspace_id, settings=settings) == []
        assert _status_of(workspace_id, settings, held["id"]) == "claimed"

        # Once the lease lapses, the unit comes back with a new version.
        time.sleep(2.5)
        reclaimed = claims.reap_expired(workspace_id, settings=settings)
        assert [row["id"] for row in reclaimed] == [held["id"]]
        assert _status_of(workspace_id, settings, held["id"]) == "pending"

        # The death is in the audit trail, not just in the outcome.
        resolutions = {
            row["resolution"] for row in conflicts.listing(workspace_id, settings=settings)
        }
        assert "requeued_by_reaper" in resolutions

        # A healthy agent picks up exactly where the dead one left off.
        survivor = sessions.register(workspace_id, "survivor", settings)
        completed = 0
        while True:
            outcome = claims.claim_next(workspace_id, survivor.id, settings=settings)
            if outcome.unit is None:
                break
            claims.complete(
                outcome.unit, survivor.id, result_ref=f"done://{outcome.unit.target}",
                settings=settings,
            )
            completed += 1

        states = claims.unit_states(workspace_id, settings)
        assert completed == 3, "the dead agent's unit was not picked back up"
        assert all(u["status"] == "done" for u in states)
        assert all(u["result_ref"] is not None for u in states)
        assert len({u["target"] for u in states}) == 3, "no unit was processed twice"

    def test_a_live_agent_keeps_its_lease(self, clean_workspaces, tiny_task_spec):
        """The mirror image: heartbeating must protect work in progress."""
        settings = clean_workspaces
        seeded = seed_workspace(
            name=f"alive-{uuid.uuid4().hex[:8]}",
            task_spec=tiny_task_spec,
            mode="safe",
            settings=settings,
        )
        agent = sessions.register(seeded.workspace_id, "steady", settings)
        outcome = claims.claim_next(
            seeded.workspace_id, agent.id, lease_seconds=1, settings=settings
        )
        assert outcome.unit is not None

        beater = sessions.Heartbeater(
            agent.id, interval_seconds=0.2, settings=settings
        ).start()
        try:
            time.sleep(1.5)  # longer than the original lease
            assert claims.reap_expired(seeded.workspace_id, settings=settings) == []
        finally:
            beater.stop()

        assert _status_of(seeded.workspace_id, settings, outcome.unit.id) == "claimed"


def _wait_for_claim(workspace_id: uuid.UUID, settings, timeout: float) -> dict:
    """Block until some unit is claimed, or fail the test."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for unit in claims.unit_states(workspace_id, settings):
            if unit["status"] == "claimed":
                return unit
        time.sleep(0.1)
    pytest.fail(f"no unit was claimed within {timeout}s")


def _status_of(workspace_id: uuid.UUID, settings, unit_id: uuid.UUID) -> str:
    for unit in claims.unit_states(workspace_id, settings):
        if unit["id"] == unit_id:
            return str(unit["status"])
    pytest.fail(f"unit {unit_id} vanished")
