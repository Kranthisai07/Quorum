"""Migration work, findings, and the full agent loop.

The migration itself is not what Quorum is about, so these tests care about the
contract around it: that a malformed model response fails loudly instead of
storing garbage, that a discovery affecting other files is marked as such, that
the source tree is never written to, and that the loop as a whole leaves a
workspace where every unit is done and every result is retrievable.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from quorum.llm import Completion, StubBackend
from quorum.migration import (
    MigrationError,
    _is_invalidating,
    _parse,
    migrate_unit,
    unified_diff,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def unit_spec() -> dict:
    return {
        "path": "pkg/client.py",
        "from_library": "requests",
        "to_library": "httpx",
        "scopes": ["client-lifecycle", "streaming"],
        "evidence": {"import": [1], "session": [6]},
    }


class TestResponseParsing:
    def _completion(self, text: str) -> Completion:
        return Completion(text=text, model="test", stop_reason="end_turn")

    def test_extracts_code_and_finding(self):
        code, finding = _parse(
            self._completion(
                "<migrated>\n```python\nimport httpx\n```\n</migrated>\n"
                "<finding>\nSwapped the client.\n</finding>"
            ),
            "a.py",
        )
        assert code == "import httpx\n"
        assert finding == "Swapped the client."

    def test_accepts_an_unlabelled_fence(self):
        code, _ = _parse(
            self._completion("<migrated>\n```\nimport httpx\n```\n</migrated>"), "a.py"
        )
        assert code == "import httpx\n"

    def test_a_missing_migrated_block_is_an_error(self):
        """Better to fail the unit than to store something unusable."""
        with pytest.raises(MigrationError, match="no <migrated> block"):
            _parse(self._completion("I would rather not."), "a.py")

    def test_a_missing_finding_gets_a_placeholder(self):
        _, finding = _parse(
            self._completion("<migrated>\n```python\nx = 1\n```\n</migrated>"), "a.py"
        )
        assert "no finding" in finding

    def test_the_error_names_the_stop_reason(self):
        with pytest.raises(MigrationError, match="max_tokens"):
            _parse(Completion(text="truncated...", model="t", stop_reason="max_tokens"), "a.py")


class TestInvalidationMarker:
    def test_the_marker_is_recognised(self):
        assert _is_invalidating("AFFECTS OTHERS: every caller inherits this mapping")

    def test_the_marker_is_case_insensitive(self):
        assert _is_invalidating("affects others: something")

    def test_an_ordinary_finding_does_not_invalidate(self):
        assert not _is_invalidating("Swapped Session for Client in this file only.")

    def test_the_marker_must_lead(self):
        """Mid-sentence mentions must not trigger a cascade."""
        assert not _is_invalidating("This change AFFECTS OTHERS: no, actually it does not")


class TestUnifiedDiff:
    def test_identical_text_produces_no_diff(self):
        assert unified_diff("a.py", "x = 1\n", "x = 1\n") == ""

    def test_changes_are_labelled_with_the_target(self):
        diff = unified_diff("pkg/a.py", "x = 1\n", "x = 2\n")
        assert "a/pkg/a.py" in diff
        assert "b/pkg/a.py" in diff
        assert "-x = 1" in diff
        assert "+x = 2" in diff


class TestStubMigration:
    def test_produces_a_real_patch(self, unit_spec, tiny_repo):
        result = migrate_unit(unit_spec, repo_root=tiny_repo, backend=StubBackend())

        assert result.changed
        assert result.diff.startswith("--- a/pkg/client.py")
        assert result.changed_lines > 0
        assert result.model == "stub"

    def test_never_writes_to_the_source_tree(self, unit_spec, tiny_repo):
        before = (tiny_repo / "pkg" / "client.py").read_text(encoding="utf-8")

        migrate_unit(unit_spec, repo_root=tiny_repo, backend=StubBackend())

        assert (tiny_repo / "pkg" / "client.py").read_text(encoding="utf-8") == before

    def test_error_mapping_scope_produces_an_invalidating_finding(self, tiny_repo):
        """Gives Phase 5 a deterministic cascade trigger with no model involved."""
        spec = {
            "path": "pkg/errors.py",
            "from_library": "requests",
            "to_library": "httpx",
            "scopes": ["error-mapping"],
            "evidence": {"exceptions": [7]},
        }

        result = migrate_unit(spec, repo_root=tiny_repo, backend=StubBackend())

        assert result.invalidates
        assert result.finding.startswith("AFFECTS OTHERS:")

    def test_an_ordinary_file_does_not_invalidate(self, unit_spec, tiny_repo):
        result = migrate_unit(unit_spec, repo_root=tiny_repo, backend=StubBackend())
        assert not result.invalidates

    def test_a_missing_source_file_is_an_error(self, unit_spec, tiny_repo):
        unit_spec["path"] = "pkg/does_not_exist.py"
        with pytest.raises(MigrationError, match="source file not found"):
            migrate_unit(unit_spec, repo_root=tiny_repo, backend=StubBackend())


@pytest.mark.integration
class TestFindings:
    def test_findings_are_embedded_on_write(self, clean_workspaces, tiny_task_spec):
        from quorum import findings
        from quorum.workspace import seed_workspace

        settings = clean_workspaces
        seeded = seed_workspace(
            name=f"find-{uuid.uuid4().hex[:8]}",
            task_spec=tiny_task_spec,
            settings=settings,
        )
        findings.record(
            seeded.workspace_id, "errors.py maps requests exceptions", settings=settings
        )
        findings.record(
            seeded.workspace_id, "cascading invalidation of the dependency graph",
            settings=settings,
        )

        probe = StubBackend(settings).embed("requests exception mapping in errors")
        neighbours = findings.nearest(seeded.workspace_id, probe.vector, settings=settings)

        assert len(neighbours) == 2
        assert neighbours[0]["similarity"] > neighbours[1]["similarity"]
        assert "requests exceptions" in neighbours[0]["content"]

    def test_a_dimension_mismatch_is_caught_before_the_insert(
        self, clean_workspaces, tiny_task_spec
    ):
        """The failure Phase 4 would otherwise hit mid-classification."""
        from quorum import findings
        from quorum.config import Settings
        from quorum.workspace import seed_workspace

        settings = clean_workspaces
        seeded = seed_workspace(
            name=f"dim-{uuid.uuid4().hex[:8]}",
            task_spec=tiny_task_spec,
            settings=settings,
        )
        wrong = StubBackend(Settings(**{**settings.__dict__, "embed_dim": 512}))

        with pytest.raises(ValueError, match="does not match"):
            findings.record(
                seeded.workspace_id, "anything", backend=wrong, settings=settings
            )

    def test_invalidating_findings_can_be_listed_alone(self, clean_workspaces, tiny_task_spec):
        from quorum import findings
        from quorum.workspace import seed_workspace

        settings = clean_workspaces
        seeded = seed_workspace(
            name=f"inv-{uuid.uuid4().hex[:8]}", task_spec=tiny_task_spec, settings=settings
        )
        findings.record(seeded.workspace_id, "ordinary", settings=settings)
        findings.record(
            seeded.workspace_id, "AFFECTS OTHERS: big deal", invalidates=True, settings=settings
        )

        only = findings.listing(seeded.workspace_id, invalidating_only=True, settings=settings)
        assert [row["content"] for row in only] == ["AFFECTS OTHERS: big deal"]


@pytest.mark.integration
class TestAgentLoop:
    """Claim -> migrate -> artifact -> finding -> complete, end to end."""

    def test_agents_drain_a_workspace_and_leave_evidence(
        self, clean_workspaces, fixture_task_spec, tmp_path
    ):
        from quorum import claims, findings
        from quorum.artifacts import LocalArtifactStore
        from quorum.config import Settings
        from quorum.runner import run
        from quorum.workspace import seed_workspace

        settings = Settings(**{**clean_workspaces.__dict__, "artifact_dir": tmp_path})
        seeded = seed_workspace(
            name=f"loop-{uuid.uuid4().hex[:8]}",
            task_spec=fixture_task_spec,
            mode="safe",
            settings=settings,
        )

        report = run(
            str(seeded.workspace_id), agents=4, work_mode="migrate", settings=settings
        )

        assert report.errors == []
        assert report.duplicate_claims == {}
        assert report.total_completed == seeded.unit_count
        assert report.total_findings == seeded.unit_count

        units = claims.unit_states(seeded.workspace_id, settings)
        assert all(unit["status"] == "done" for unit in units)
        assert all(unit["result_ref"] for unit in units)

        # Every stored result is retrievable and is a real patch.
        store = LocalArtifactStore(settings)
        for unit in units:
            patch = store.get(str(unit["result_ref"]))
            assert patch.startswith("--- a/")

        # At least one agent flagged something affecting other files, which is
        # what Phase 5's cascade will key off.
        invalidating = findings.listing(
            seeded.workspace_id, invalidating_only=True, settings=settings
        )
        assert invalidating, "no invalidating findings -- Phase 5 would have no trigger"

    def test_the_vendored_fixture_is_never_modified(
        self, clean_workspaces, fixture_task_spec, tmp_path
    ):
        """`git status` must stay clean no matter how many times agents run."""
        import hashlib

        from quorum.config import Settings
        from quorum.runner import run
        from quorum.workspace import seed_workspace

        settings = Settings(**{**clean_workspaces.__dict__, "artifact_dir": tmp_path})
        repo = Path(fixture_task_spec["repo"])

        def fingerprint() -> str:
            digest = hashlib.sha256()
            for path in sorted(repo.rglob("*.py")):
                digest.update(path.read_bytes())
            return digest.hexdigest()

        before = fingerprint()
        seeded = seed_workspace(
            name=f"pristine-{uuid.uuid4().hex[:8]}",
            task_spec=fixture_task_spec,
            settings=settings,
        )
        run(str(seeded.workspace_id), agents=2, work_mode="migrate", settings=settings)

        assert fingerprint() == before, "agents wrote into the vendored fixture"
