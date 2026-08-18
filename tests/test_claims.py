"""Claim engine: leases, contention, reclaim, and stale-write rejection.

Every path here is deterministic -- two agents, one unit, explicit ordering --
so a failure names the broken invariant instead of being a probabilistic
mystery. The probabilistic side of the argument lives in `test_stress.py`.
"""

from __future__ import annotations

import time
import uuid

import pytest

from quorum import claims, conflicts, sessions
from quorum.claims import StaleClaimError

pytestmark = pytest.mark.integration


@pytest.fixture
def workspace(clean_workspaces, tiny_task_spec):
    """A three-unit workspace with two registered agents."""
    from quorum.workspace import seed_workspace

    seeded = seed_workspace(
        name=f"claims-{uuid.uuid4().hex[:8]}",
        task_spec=tiny_task_spec,
        mode="safe",
        settings=clean_workspaces,
    )
    alice = sessions.register(seeded.workspace_id, "alice", clean_workspaces)
    bob = sessions.register(seeded.workspace_id, "bob", clean_workspaces)
    return seeded.workspace_id, alice, bob, clean_workspaces


class TestClaiming:
    def test_claim_takes_a_unit_and_sets_a_lease(self, workspace):
        workspace_id, alice, _bob, settings = workspace

        outcome = claims.claim_next(workspace_id, alice.id, settings=settings)

        assert outcome.claimed
        assert outcome.unit is not None
        state = {u["id"]: u for u in claims.unit_states(workspace_id, settings)}[
            outcome.unit.id
        ]
        assert state["status"] == "claimed"
        assert state["claimed_by"] == alice.id
        assert state["claim_expires_at"] is not None

    def test_two_agents_never_get_the_same_unit(self, workspace):
        workspace_id, alice, bob, settings = workspace

        first = claims.claim_next(workspace_id, alice.id, settings=settings)
        second = claims.claim_next(workspace_id, bob.id, settings=settings)

        assert first.unit is not None
        assert second.unit is not None
        assert first.unit.id != second.unit.id

    def test_claiming_is_deterministic_by_target(self, workspace):
        """Ordering by target is what makes every agent race for the same row."""
        workspace_id, alice, _bob, settings = workspace
        outcome = claims.claim_next(workspace_id, alice.id, settings=settings)
        assert outcome.unit is not None
        assert outcome.unit.target == "pkg/client.py"  # alphabetically first

    def test_empty_workspace_returns_no_unit(self, workspace):
        workspace_id, alice, _bob, settings = workspace
        for _ in range(3):
            claims.claim_next(workspace_id, alice.id, settings=settings)

        exhausted = claims.claim_next(workspace_id, alice.id, settings=settings)
        assert exhausted.unit is None
        assert not exhausted.claimed

    def test_claimed_unit_carries_its_spec(self, workspace):
        workspace_id, alice, _bob, settings = workspace
        outcome = claims.claim_next(workspace_id, alice.id, settings=settings)
        assert outcome.unit is not None
        assert outcome.unit.spec["to_library"] == "httpx"


class TestCompletion:
    def test_complete_marks_the_unit_done(self, workspace):
        workspace_id, alice, _bob, settings = workspace
        outcome = claims.claim_next(workspace_id, alice.id, settings=settings)
        assert outcome.unit is not None

        claims.complete(outcome.unit, alice.id, result_ref="s3://x", settings=settings)

        state = {u["id"]: u for u in claims.unit_states(workspace_id, settings)}[
            outcome.unit.id
        ]
        assert state["status"] == "done"
        assert state["result_ref"] == "s3://x"

    def test_completing_someone_elses_unit_is_refused(self, workspace):
        workspace_id, alice, bob, settings = workspace
        outcome = claims.claim_next(workspace_id, alice.id, settings=settings)
        assert outcome.unit is not None

        with pytest.raises(StaleClaimError):
            claims.complete(outcome.unit, bob.id, settings=settings)

        state = {u["id"]: u for u in claims.unit_states(workspace_id, settings)}[
            outcome.unit.id
        ]
        assert state["status"] == "claimed", "a foreign write must not land"

    def test_release_returns_the_unit_to_the_pool(self, workspace):
        workspace_id, alice, bob, settings = workspace
        outcome = claims.claim_next(workspace_id, alice.id, settings=settings)
        assert outcome.unit is not None

        assert claims.release(outcome.unit, alice.id, settings=settings) is True

        retaken = claims.claim_next(workspace_id, bob.id, settings=settings)
        assert retaken.unit is not None
        assert retaken.unit.id == outcome.unit.id

    def test_fail_records_the_reason(self, workspace):
        workspace_id, alice, _bob, settings = workspace
        outcome = claims.claim_next(workspace_id, alice.id, settings=settings)
        assert outcome.unit is not None

        claims.fail(outcome.unit, alice.id, reason="bedrock timeout", settings=settings)

        state = {u["id"]: u for u in claims.unit_states(workspace_id, settings)}[
            outcome.unit.id
        ]
        assert state["status"] == "failed"


class TestExpiredLeases:
    """What happens when an agent stops renewing -- the crashed-agent case."""

    def test_expired_lease_is_claimable_by_another_agent(self, workspace):
        workspace_id, alice, bob, settings = workspace
        alice_claim = claims.claim_next(workspace_id, alice.id, settings=settings)
        assert alice_claim.unit is not None
        claims.expire_lease_now(alice_claim.unit.id, settings)

        bob_claim = claims.claim_next(workspace_id, bob.id, settings=settings)

        assert bob_claim.unit is not None
        assert bob_claim.unit.id == alice_claim.unit.id
        assert bob_claim.unit.reclaimed is True

    def test_takeover_bumps_the_version(self, workspace):
        workspace_id, alice, bob, settings = workspace
        alice_claim = claims.claim_next(workspace_id, alice.id, settings=settings)
        assert alice_claim.unit is not None
        claims.expire_lease_now(alice_claim.unit.id, settings)

        bob_claim = claims.claim_next(workspace_id, bob.id, settings=settings)

        assert bob_claim.unit is not None
        assert bob_claim.unit.version == alice_claim.unit.version + 1

    def test_the_original_agent_cannot_overwrite_the_takeover(self, workspace):
        """The zombie-write case: a dead agent waking up must not clobber."""
        workspace_id, alice, bob, settings = workspace
        alice_claim = claims.claim_next(workspace_id, alice.id, settings=settings)
        assert alice_claim.unit is not None
        claims.expire_lease_now(alice_claim.unit.id, settings)
        bob_claim = claims.claim_next(workspace_id, bob.id, settings=settings)
        assert bob_claim.unit is not None

        with pytest.raises(StaleClaimError):
            claims.complete(alice_claim.unit, alice.id, result_ref="zombie", settings=settings)

        claims.complete(bob_claim.unit, bob.id, result_ref="real", settings=settings)
        state = {u["id"]: u for u in claims.unit_states(workspace_id, settings)}[
            bob_claim.unit.id
        ]
        assert state["result_ref"] == "real"

    def test_takeover_is_logged_as_a_conflict(self, workspace):
        workspace_id, alice, bob, settings = workspace
        alice_claim = claims.claim_next(workspace_id, alice.id, settings=settings)
        assert alice_claim.unit is not None
        claims.expire_lease_now(alice_claim.unit.id, settings)
        claims.claim_next(workspace_id, bob.id, settings=settings)

        logged = conflicts.listing(workspace_id, kind="claim", settings=settings)
        reasons = {row["detail"].get("reason") for row in logged}
        assert "expired_lease_taken_over" in reasons

    def test_stale_write_is_logged_as_a_conflict(self, workspace):
        workspace_id, alice, bob, settings = workspace
        alice_claim = claims.claim_next(workspace_id, alice.id, settings=settings)
        assert alice_claim.unit is not None
        claims.expire_lease_now(alice_claim.unit.id, settings)
        claims.claim_next(workspace_id, bob.id, settings=settings)

        with pytest.raises(StaleClaimError):
            claims.complete(alice_claim.unit, alice.id, settings=settings)

        reasons = {
            row["detail"].get("reason")
            for row in conflicts.listing(workspace_id, settings=settings)
        }
        assert "stale_write_rejected" in reasons


class TestReaper:
    def test_reaper_requeues_expired_units(self, workspace):
        workspace_id, alice, _bob, settings = workspace
        outcome = claims.claim_next(workspace_id, alice.id, settings=settings)
        assert outcome.unit is not None
        claims.expire_lease_now(outcome.unit.id, settings)

        reclaimed = claims.reap_expired(workspace_id, settings=settings)

        assert len(reclaimed) == 1
        assert reclaimed[0]["id"] == outcome.unit.id
        state = {u["id"]: u for u in claims.unit_states(workspace_id, settings)}[
            outcome.unit.id
        ]
        assert state["status"] == "pending"
        assert state["version"] == outcome.unit.version + 1

    def test_reaper_leaves_live_leases_alone(self, workspace):
        workspace_id, alice, _bob, settings = workspace
        claims.claim_next(workspace_id, alice.id, settings=settings)

        assert claims.reap_expired(workspace_id, settings=settings) == []

    def test_reaper_logs_every_reclaim(self, workspace):
        workspace_id, alice, _bob, settings = workspace
        outcome = claims.claim_next(workspace_id, alice.id, settings=settings)
        assert outcome.unit is not None
        claims.expire_lease_now(outcome.unit.id, settings)

        claims.reap_expired(workspace_id, settings=settings)

        resolutions = {
            row["resolution"] for row in conflicts.listing(workspace_id, settings=settings)
        }
        assert "requeued_by_reaper" in resolutions

    def test_naive_mode_has_no_reaper(self, workspace):
        """A unit held by a dead naive agent stays held. That is the point."""
        workspace_id, alice, _bob, settings = workspace
        outcome = claims.claim_next(workspace_id, alice.id, settings=settings)
        assert outcome.unit is not None
        claims.expire_lease_now(outcome.unit.id, settings)

        assert claims.reap_expired(workspace_id, mode="naive", settings=settings) == []


class TestHeartbeats:
    def test_heartbeat_extends_the_lease(self, workspace):
        workspace_id, alice, _bob, settings = workspace
        outcome = claims.claim_next(
            workspace_id, alice.id, lease_seconds=2, settings=settings
        )
        assert outcome.unit is not None
        before = {u["id"]: u for u in claims.unit_states(workspace_id, settings)}[
            outcome.unit.id
        ]["claim_expires_at"]

        renewed = sessions.heartbeat(alice.id, lease_seconds=300, settings=settings)

        after = {u["id"]: u for u in claims.unit_states(workspace_id, settings)}[
            outcome.unit.id
        ]["claim_expires_at"]
        assert renewed == 1
        assert after > before

    def test_heartbeat_keeps_a_session_live(self, workspace):
        workspace_id, alice, _bob, settings = workspace
        sessions.heartbeat(alice.id, settings=settings)

        names = {row["name"] for row in sessions.live(workspace_id, settings=settings)}
        assert "alice" in names

    def test_silent_sessions_are_marked_dead(self, workspace):
        workspace_id, _alice, _bob, settings = workspace
        time.sleep(0.05)

        dead = sessions.mark_stale_dead(workspace_id, timeout_seconds=0, settings=settings)

        assert len(dead) == 2
        assert sessions.live(workspace_id, timeout_seconds=0, settings=settings) == []

    def test_a_dead_session_does_not_block_its_units(self, workspace):
        """Marking a session dead is bookkeeping; the lease is the safety net."""
        workspace_id, alice, bob, settings = workspace
        outcome = claims.claim_next(workspace_id, alice.id, settings=settings)
        assert outcome.unit is not None
        sessions.mark_stale_dead(workspace_id, timeout_seconds=0, settings=settings)

        # Still held, because the lease has not expired yet.
        assert claims.reap_expired(workspace_id, settings=settings) == []

        claims.expire_lease_now(outcome.unit.id, settings)
        retaken = claims.claim_next(workspace_id, bob.id, settings=settings)
        assert retaken.unit is not None
        assert retaken.unit.id == outcome.unit.id


class TestNaiveMode:
    def test_naive_claim_still_records_a_holder(self, workspace):
        workspace_id, alice, _bob, settings = workspace
        outcome = claims.claim_next(workspace_id, alice.id, mode="naive", settings=settings)

        assert outcome.unit is not None
        state = {u["id"]: u for u in claims.unit_states(workspace_id, settings)}[
            outcome.unit.id
        ]
        assert state["claimed_by"] == alice.id

    def test_naive_complete_is_unguarded(self, workspace):
        """No version check: any agent can mark any unit done. That is the bug."""
        workspace_id, alice, bob, settings = workspace
        outcome = claims.claim_next(workspace_id, alice.id, settings=settings)
        assert outcome.unit is not None

        claims.complete(outcome.unit, bob.id, result_ref="wrong", mode="naive", settings=settings)

        state = {u["id"]: u for u in claims.unit_states(workspace_id, settings)}[
            outcome.unit.id
        ]
        assert state["status"] == "done"
        assert state["result_ref"] == "wrong"

    def test_naive_logs_no_conflicts(self, workspace):
        workspace_id, alice, bob, settings = workspace
        claims.claim_next(workspace_id, alice.id, mode="naive", settings=settings)
        claims.claim_next(workspace_id, bob.id, mode="naive", settings=settings)

        assert conflicts.counts(workspace_id, settings)["total"] == 0
