"""Workspace seeding: atomicity, shape, and lookup.

The interesting assertion here is `test_failed_seed_leaves_nothing_behind`. A
workspace whose work units committed but whose dependency edges did not would
give the Phase 5 invalidation cascade an incomplete graph to walk, and the
cascade would then report success while leaving stale work in place. That is the
exact failure Quorum claims serializable transactions prevent, so it is worth a
test from the first phase.
"""

from __future__ import annotations

import uuid

import psycopg
import pytest

from quorum.db import connection
from quorum.decompose import Decomposition, WorkUnitSpec
from quorum.workspace import (
    list_workspaces,
    resolve_workspace,
    seed_workspace,
    workspace_summary,
)

pytestmark = pytest.mark.integration


def _seed(spec, name="tiny", mode="safe", settings=None):
    return seed_workspace(name=name, task_spec=spec, mode=mode, settings=settings)


def test_seed_writes_units_and_deps(clean_workspaces, tiny_task_spec):
    result = _seed(tiny_task_spec, settings=clean_workspaces)

    assert result.unit_count == 3
    assert result.dep_count == 1
    assert result.mode == "safe"

    summary = workspace_summary(result.workspace_id, clean_workspaces)
    assert summary["units"] == {"pending": 3}
    assert summary["deps"] == 1
    assert summary["agents_running"] == 0
    assert summary["conflicts"] == {}


def test_seeded_units_carry_their_evidence(clean_workspaces, tiny_task_spec):
    result = _seed(tiny_task_spec, settings=clean_workspaces)

    with connection(clean_workspaces) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT target, spec, status, version FROM work_units WHERE workspace_id = %s",
            (result.workspace_id,),
        )
        rows = {str(row["target"]): row for row in cur.fetchall()}

    assert set(rows) == {"pkg/client.py", "pkg/errors.py", "pkg/transport.py"}
    client = rows["pkg/client.py"]
    assert client["status"] == "pending"
    assert client["version"] == 1
    assert client["spec"]["to_library"] == "httpx"
    assert "session" in client["spec"]["evidence"]


def test_dependency_edges_point_at_real_units(clean_workspaces, tiny_task_spec):
    result = _seed(tiny_task_spec, settings=clean_workspaces)

    with connection(clean_workspaces) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.target AS dependent, b.target AS depends_on
            FROM unit_deps d
            JOIN work_units a ON a.id = d.unit_id
            JOIN work_units b ON b.id = d.depends_on_unit_id
            WHERE a.workspace_id = %s
            """,
            (result.workspace_id,),
        )
        edges = [(str(row["dependent"]), str(row["depends_on"])) for row in cur.fetchall()]

    assert edges == [("pkg/client.py", "pkg/errors.py")]


def test_mode_is_persisted(clean_workspaces, tiny_task_spec):
    result = _seed(tiny_task_spec, mode="naive", settings=clean_workspaces)
    assert resolve_workspace(str(result.workspace_id), clean_workspaces)["mode"] == "naive"


def test_invalid_mode_is_rejected_by_the_database(clean_workspaces, tiny_task_spec):
    with pytest.raises(psycopg.errors.CheckViolation):
        _seed(tiny_task_spec, mode="whatever", settings=clean_workspaces)


def test_failed_seed_leaves_nothing_behind(clean_workspaces, tiny_task_spec, monkeypatch):
    """A seed that fails part-way must not leave a partial workspace.

    The stub decomposer emits a duplicate dependency edge, which survives to the
    INSERT and violates the unit_deps primary key. Because the whole seed runs
    in one serializable transaction, the workspace row and its 16 work units
    must roll back with it.
    """

    class DuplicateEdgeDecomposer:
        name = "duplicate_edges"

        def decompose(self, task_spec):
            units = [
                WorkUnitSpec(target="a.py", spec={"kind": "stub"}),
                WorkUnitSpec(target="b.py", spec={"kind": "stub"}),
            ]
            # Deliberately not validated: validate() would catch this first.
            return Decomposition(units=units, deps=[("a.py", "b.py"), ("a.py", "b.py")])

    monkeypatch.setattr(
        "quorum.workspace.get_decomposer", lambda _name: DuplicateEdgeDecomposer()
    )

    name = f"doomed-{uuid.uuid4().hex[:8]}"
    with pytest.raises(psycopg.errors.UniqueViolation):
        _seed(tiny_task_spec, name=name, settings=clean_workspaces)

    with connection(clean_workspaces) as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) AS n FROM workspaces WHERE name = %s", (name,))
        workspaces = cur.fetchone()
        cur.execute("SELECT count(*) AS n FROM work_units")
        units = cur.fetchone()
        cur.execute("SELECT count(*) AS n FROM unit_deps")
        deps = cur.fetchone()

    assert workspaces is not None and workspaces["n"] == 0
    assert units is not None and units["n"] == 0, "work units survived a rolled-back seed"
    assert deps is not None and deps["n"] == 0


def test_empty_decomposition_is_refused(clean_workspaces, tiny_task_spec, monkeypatch):
    class EmptyDecomposer:
        name = "empty"

        def decompose(self, task_spec):
            return Decomposition(units=[], deps=[])

    monkeypatch.setattr("quorum.workspace.get_decomposer", lambda _name: EmptyDecomposer())
    with pytest.raises(ValueError, match="no work units"):
        _seed(tiny_task_spec, settings=clean_workspaces)


def test_resolve_by_name_and_by_id(clean_workspaces, tiny_task_spec):
    name = f"named-{uuid.uuid4().hex[:8]}"
    result = _seed(tiny_task_spec, name=name, settings=clean_workspaces)

    by_name = resolve_workspace(name, clean_workspaces)
    by_id = resolve_workspace(str(result.workspace_id), clean_workspaces)
    assert by_name["id"] == by_id["id"] == result.workspace_id


def test_resolve_by_name_prefers_the_newest(clean_workspaces, tiny_task_spec):
    name = f"repeat-{uuid.uuid4().hex[:8]}"
    _seed(tiny_task_spec, name=name, settings=clean_workspaces)
    second = _seed(tiny_task_spec, name=name, settings=clean_workspaces)
    assert resolve_workspace(name, clean_workspaces)["id"] == second.workspace_id


def test_unknown_workspace_raises(clean_workspaces):
    with pytest.raises(LookupError):
        resolve_workspace(str(uuid.uuid4()), clean_workspaces)


def test_list_workspaces_reports_counts(clean_workspaces, tiny_task_spec):
    result = _seed(tiny_task_spec, settings=clean_workspaces)
    rows = {row["id"]: row for row in list_workspaces(clean_workspaces)}
    assert rows[result.workspace_id]["units"] == 3
    assert rows[result.workspace_id]["pending"] == 3
    assert rows[result.workspace_id]["done"] == 0


def test_real_fixture_seeds(clean_workspaces, fixture_task_spec):
    """End to end on the vendored repo: the Phase 1 acceptance check."""
    result = _seed(fixture_task_spec, name="docker-py", settings=clean_workspaces)
    assert result.unit_count >= 10
    assert result.dep_count >= 5
    assert "error-mapping" in result.scopes
    assert result.txn_retries == 0

    summary = workspace_summary(result.workspace_id, clean_workspaces)
    assert summary["unit_total"] == result.unit_count
    assert summary["deps"] == result.dep_count
