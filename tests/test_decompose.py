"""Decomposition tests.

Decomposition is where the coordination problem gets its shape: how many units
there are to contend over, which decision scopes overlap, and which dependency
edges the invalidation cascade will later walk. All three are asserted exactly
against a hand-built repository, and loosely against the vendored real one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from quorum.decompose import Decomposition, WorkUnitSpec, get_decomposer, registered
from quorum.decompose.code_migration import CodeMigrationDecomposer


def _decompose(spec: dict[str, Any]) -> Decomposition:
    return get_decomposer("code_migration").decompose(spec)


def test_code_migration_decomposer_is_registered():
    assert "code_migration" in registered()
    assert isinstance(get_decomposer("code_migration"), CodeMigrationDecomposer)


def test_unknown_decomposer_names_the_registered_ones():
    with pytest.raises(KeyError, match="code_migration"):
        get_decomposer("does_not_exist")


def test_selects_only_files_with_strong_signals(tiny_task_spec):
    result = _decompose(tiny_task_spec)
    assert sorted(result.targets()) == [
        "pkg/client.py",
        "pkg/errors.py",
        "pkg/transport.py",
    ]


def test_excluded_paths_are_not_units(tiny_task_spec):
    result = _decompose(tiny_task_spec)
    assert not any(target.startswith("pkg/tests/") for target in result.targets())


def test_scopes_reflect_the_idioms_present(tiny_task_spec):
    by_target = {unit.target: unit for unit in _decompose(tiny_task_spec).units}

    assert "client-lifecycle" in by_target["pkg/client.py"].scopes
    assert "streaming" in by_target["pkg/client.py"].scopes
    assert "timeout-policy" in by_target["pkg/client.py"].scopes
    assert "error-mapping" in by_target["pkg/errors.py"].scopes
    assert "transport-adapter" in by_target["pkg/transport.py"].scopes
    # helpers.py has no signals at all and never became a unit.
    assert "pkg/helpers.py" not in by_target


def test_dependency_edges_come_from_real_imports(tiny_task_spec):
    result = _decompose(tiny_task_spec)
    # client.py does `from .errors import wrap`, so invalidating errors.py must
    # reconsider client.py -- and nothing else in this repo is coupled.
    assert result.deps == [("pkg/client.py", "pkg/errors.py")]


def test_evidence_records_line_numbers(tiny_task_spec):
    by_target = {unit.target: unit for unit in _decompose(tiny_task_spec).units}
    evidence = by_target["pkg/client.py"].spec["evidence"]
    assert evidence["import"] == [1]
    assert evidence["session"] == [6]
    assert evidence["raise_for_status"] == [8]


def test_unit_spec_carries_migration_instruction(tiny_task_spec):
    unit = next(u for u in _decompose(tiny_task_spec).units if u.target == "pkg/errors.py")
    assert unit.spec["from_library"] == "requests"
    assert unit.spec["to_library"] == "httpx"
    assert unit.spec["module"] == "pkg.errors"
    assert "httpx" in unit.spec["instruction"]


def test_missing_repo_is_rejected(tiny_task_spec, tmp_path: Path):
    tiny_task_spec["repo"] = str(tmp_path / "nope")
    with pytest.raises(ValueError, match="not a directory"):
        _decompose(tiny_task_spec)


def test_unknown_source_library_is_rejected(tiny_task_spec):
    tiny_task_spec["from_library"] = "urllib"
    with pytest.raises(ValueError, match="no signal table"):
        _decompose(tiny_task_spec)


def test_missing_package_root_is_rejected(tiny_task_spec):
    tiny_task_spec["package_roots"] = ["not_a_package"]
    with pytest.raises(ValueError, match="package root not found"):
        _decompose(tiny_task_spec)


class TestDecompositionValidation:
    """A decomposition the engine could not safely run must not be written."""

    def test_dangling_dependency_edge_is_rejected(self):
        decomposition = Decomposition(
            units=[WorkUnitSpec(target="a.py", spec={})],
            deps=[("a.py", "ghost.py")],
        )
        with pytest.raises(ValueError, match=r"unknown unit: ghost\.py"):
            decomposition.validate()

    def test_duplicate_targets_are_rejected(self):
        decomposition = Decomposition(
            units=[
                WorkUnitSpec(target="a.py", spec={}),
                WorkUnitSpec(target="a.py", spec={}),
            ]
        )
        with pytest.raises(ValueError, match="duplicate work unit targets"):
            decomposition.validate()

    def test_self_dependency_is_rejected(self):
        decomposition = Decomposition(
            units=[WorkUnitSpec(target="a.py", spec={})],
            deps=[("a.py", "a.py")],
        )
        with pytest.raises(ValueError, match="self-dependency"):
            decomposition.validate()

    def test_duplicate_edges_are_rejected(self):
        decomposition = Decomposition(
            units=[
                WorkUnitSpec(target="a.py", spec={}),
                WorkUnitSpec(target="b.py", spec={}),
            ],
            deps=[("a.py", "b.py"), ("a.py", "b.py")],
        )
        with pytest.raises(ValueError, match="duplicate dependency edges"):
            decomposition.validate()

    def test_scopes_are_deduplicated_and_sorted(self):
        decomposition = Decomposition(
            units=[
                WorkUnitSpec(target="a.py", spec={}, scopes=("streaming", "error-mapping")),
                WorkUnitSpec(target="b.py", spec={}, scopes=("streaming",)),
            ]
        )
        assert decomposition.scopes() == ["error-mapping", "streaming"]


class TestVendoredFixture:
    """Loose assertions against the real docker-py tree.

    Exact counts would break whenever the fixture is re-vendored, but the
    properties the coordination engine depends on must hold.
    """

    def test_produces_enough_units_to_contend_over(self, fixture_task_spec):
        result = _decompose(fixture_task_spec)
        assert len(result.units) >= 10, "too few units for contention to be meaningful"

    def test_dependency_graph_is_non_trivial(self, fixture_task_spec):
        result = _decompose(fixture_task_spec)
        assert len(result.deps) >= 5, "an invalidation cascade needs edges to walk"

    def test_scopes_overlap_across_units(self, fixture_task_spec):
        """Semantic conflict requires two agents landing in the same scope."""
        result = _decompose(fixture_task_spec)
        counts: dict[str, int] = {}
        for unit in result.units:
            for scope in unit.scopes:
                counts[scope] = counts.get(scope, 0) + 1
        shared = {scope: n for scope, n in counts.items() if n >= 2}
        assert len(shared) >= 3, f"not enough shared decision surface: {counts}"

    def test_known_hub_files_are_selected(self, fixture_task_spec):
        targets = _decompose(fixture_task_spec).targets()
        assert "docker/errors.py" in targets
        assert "docker/api/client.py" in targets

    def test_result_is_self_consistent(self, fixture_task_spec):
        _decompose(fixture_task_spec).validate()
