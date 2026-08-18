"""Migration runner tests: statement splitting, checksums, and applied schema."""

from __future__ import annotations

import pytest

from quorum import migrations
from quorum.db import connection


class TestStatementSplitting:
    """The splitter is small on purpose; these are the cases it must survive."""

    def test_splits_on_semicolons(self):
        assert migrations.split_statements("SELECT 1; SELECT 2;") == ["SELECT 1", "SELECT 2"]

    def test_ignores_trailing_empty_chunk(self):
        assert migrations.split_statements("SELECT 1;\n\n") == ["SELECT 1"]

    def test_semicolon_inside_a_literal_does_not_split(self):
        sql = "INSERT INTO t VALUES ('a;b');"
        assert migrations.split_statements(sql) == [sql[:-1]]

    def test_escaped_quote_inside_a_literal(self):
        sql = "INSERT INTO t VALUES ('it''s; fine');"
        assert migrations.split_statements(sql) == [sql[:-1]]

    def test_comment_only_chunks_are_dropped(self):
        sql = "-- a comment\n/* block */\nSELECT 1;\n-- trailing\n"
        assert migrations.split_statements(sql) == ["-- a comment\n/* block */\nSELECT 1"]

    def test_semicolon_inside_a_comment_does_not_split(self):
        sql = "-- one; two\nSELECT 1;"
        assert migrations.split_statements(sql) == ["-- one; two\nSELECT 1"]


class TestDiscovery:
    def test_migrations_are_discovered_in_order(self):
        found = migrations.discover()
        assert [m.version for m in found] == sorted(m.version for m in found)
        assert found, "no migrations shipped with the package"

    def test_core_schema_is_first(self):
        assert migrations.discover()[0].name == "core_schema"

    def test_vector_index_migration_opts_out_of_a_transaction(self):
        vector = next(m for m in migrations.discover() if "vector" in m.name)
        assert vector.transactional is False

    def test_checksum_is_stable(self):
        first = {m.label: m.checksum for m in migrations.discover()}
        second = {m.label: m.checksum for m in migrations.discover()}
        assert first == second

    def test_duplicate_versions_are_rejected(self):
        duplicated = [
            migrations.Migration(version=1, name="a", sql=""),
            migrations.Migration(version=1, name="b", sql=""),
        ]
        with pytest.raises(ValueError, match="duplicate migration version 1"):
            migrations._assert_unique_versions(duplicated)


@pytest.mark.integration
class TestAppliedSchema:
    def test_migrate_is_idempotent(self, database):
        assert migrations.migrate(database) == []

    def test_every_table_exists(self, database):
        expected = {
            "workspaces",
            "agent_sessions",
            "work_units",
            "unit_deps",
            "decisions",
            "findings",
            "conflict_log",
            "schema_migrations",
        }
        with connection(database) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            )
            found = {str(row["table_name"]) for row in cur.fetchall()}
        assert expected <= found

    def test_decisions_has_a_cosine_vector_index(self, database):
        """Semantic conflict detection is an ANN query, so the index must exist."""
        with connection(database) as conn, conn.cursor() as cur:
            cur.execute("SELECT create_statement FROM [SHOW CREATE TABLE decisions]")
            row = cur.fetchone()
        assert row is not None
        create = str(row["create_statement"])
        assert "VECTOR INDEX decisions_embedding_idx" in create
        assert "vector_cosine_ops" in create
        assert "workspace_id" in create, "ANN search must be scoped to one workspace"

    def test_findings_has_a_vector_index(self, database):
        with connection(database) as conn, conn.cursor() as cur:
            cur.execute("SELECT create_statement FROM [SHOW CREATE TABLE findings]")
            row = cur.fetchone()
        assert row is not None
        assert "VECTOR INDEX findings_embedding_idx" in str(row["create_statement"])

    def test_embedding_width_matches_configured_model(self, database):
        with connection(database) as conn, conn.cursor() as cur:
            cur.execute("SELECT create_statement FROM [SHOW CREATE TABLE decisions]")
            row = cur.fetchone()
        assert row is not None
        assert f"VECTOR({database.embed_dim})" in str(row["create_statement"])

    def test_unit_deps_can_be_walked_in_reverse(self, database):
        """The cascade asks "who depends on this?", which needs its own index."""
        with connection(database) as conn, conn.cursor() as cur:
            cur.execute("SELECT create_statement FROM [SHOW CREATE TABLE unit_deps]")
            row = cur.fetchone()
        assert row is not None
        assert "unit_deps_reverse_idx" in str(row["create_statement"])

    def test_work_units_status_is_constrained(self, database):
        with connection(database) as conn, conn.cursor() as cur:
            cur.execute("SELECT create_statement FROM [SHOW CREATE TABLE work_units]")
            row = cur.fetchone()
        assert row is not None
        assert "work_units_status_check" in str(row["create_statement"])

    def test_applied_versions_match_the_shipped_files(self, database):
        applied = migrations.applied_versions(database)
        shipped = {m.version: m.checksum for m in migrations.discover()}
        assert applied == shipped
