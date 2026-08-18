"""Config, logging, and transaction-plumbing tests.

Small surfaces, but each one has already caused a real failure: an IPv6-first
`localhost` that hung every connection, and an `extra={"name": ...}` that killed
a process from inside a logging call.
"""

from __future__ import annotations

import json
import logging

import pytest

from quorum import db
from quorum.config import Settings, get_settings
from quorum.db import TxnStats, vector_literal
from quorum.logging import JsonFormatter, configure_logging, get_logger, log_context


class TestSettings:
    def test_database_name_is_parsed_from_the_url(self):
        # conftest redirects the suite to `<database>_test`.
        assert get_settings().database_name() == "quorum_test"

    def test_admin_url_points_at_defaultdb(self):
        settings = get_settings()
        assert "/defaultdb" in settings.admin_url()
        assert settings.admin_url().endswith("sslmode=disable")

    def test_admin_url_survives_a_url_without_a_query(self):
        settings = Settings(
            **{
                **get_settings().__dict__,
                "db_url": "postgresql://root@127.0.0.1:26257/quorum",
            }
        )
        assert settings.admin_url() == "postgresql://root@127.0.0.1:26257/defaultdb"

    def test_local_default_avoids_ipv6_first_localhost(self):
        """`localhost` resolves to ::1 first on Windows and stalls every connect."""
        assert "localhost" not in get_settings().db_url

    def test_embedding_width_is_a_positive_int(self):
        assert get_settings().embed_dim > 0


class TestLogging:
    def _capture(self, logger_name: str, emit) -> list[str]:
        import io

        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())
        logger = get_logger(logger_name)
        logger.handlers[:] = [handler]
        logger.propagate = False
        logger.setLevel(logging.INFO)
        try:
            emit(logger)
        finally:
            logger.handlers.clear()
            logger.propagate = True
        return [line for line in stream.getvalue().splitlines() if line]

    def test_emits_one_json_object_per_record(self):
        lines = self._capture(
            "quorum.test.json",
            lambda log: log.info("unit.claimed", extra={"unit_id": "abc", "attempt": 2}),
        )
        payload = json.loads(lines[0])
        assert payload["event"] == "unit.claimed"
        assert payload["level"] == "INFO"
        assert payload["unit_id"] == "abc"
        assert payload["attempt"] == 2

    def test_extra_keys_may_collide_with_logrecord_attributes(self):
        """Agents log domain fields; logging must never be the thing that dies."""
        lines = self._capture(
            "quorum.test.collide",
            lambda log: log.info(
                "workspace.seeded",
                extra={"name": "docker-py", "module": "docker.errors", "args": [1, 2]},
            ),
        )
        payload = json.loads(lines[0])
        assert payload["name"] == "docker-py"
        assert payload["module"] == "docker.errors"
        assert payload["args"] == [1, 2]
        assert payload["logger"] == "quorum.test.collide"

    def test_uuid_fields_are_stringified(self):
        import uuid as uuid_module

        identifier = uuid_module.uuid4()
        lines = self._capture(
            "quorum.test.uuid",
            lambda log: log.info("x", extra={"workspace_id": identifier}),
        )
        assert json.loads(lines[0])["workspace_id"] == str(identifier)

    def test_log_context_tags_every_record_inside_the_block(self):
        def emit(log):
            with log_context(workspace_id="ws-1", agent="a-7"):
                log.info("claim.attempt")
            log.info("outside")

        lines = self._capture("quorum.test.context", emit)
        inside, outside = json.loads(lines[0]), json.loads(lines[1])
        assert inside["workspace_id"] == "ws-1"
        assert inside["agent"] == "a-7"
        assert "workspace_id" not in outside

    def test_configure_logging_accepts_both_formats(self):
        configure_logging("WARNING", "console")
        configure_logging("WARNING", "json")
        assert logging.getLogger().handlers


class TestTxnStats:
    def test_counters_are_reported_per_label(self):
        stats = TxnStats()
        stats.record_attempt("claim")
        stats.record_retry("claim")
        stats.record_attempt("claim")
        stats.record_commit("claim")
        snapshot = stats.snapshot()
        assert snapshot["claim"] == {
            "attempts": 2,
            "retries": 1,
            "commits": 1,
            "failures": 0,
        }

    def test_reset_clears_everything(self):
        stats = TxnStats()
        stats.record_attempt("x")
        stats.reset()
        assert stats.snapshot() == {}


class TestBackoff:
    def test_backoff_grows_but_stays_capped(self):
        for attempt in range(1, 12):
            for _ in range(20):
                delay = db._backoff_seconds(attempt)
                assert 0.0 <= delay <= 1.0

    def test_early_attempts_back_off_less_than_later_ones(self):
        early = max(db._backoff_seconds(1) for _ in range(200))
        late = max(db._backoff_seconds(6) for _ in range(200))
        assert early < late


class TestVectorLiteral:
    def test_formats_a_bracketed_list(self):
        assert vector_literal([1.0, 0.0, -0.5]) == "[1,0,-0.5]"

    def test_handles_an_empty_vector(self):
        assert vector_literal([]) == "[]"


@pytest.mark.integration
class TestTransactionExecution:
    def test_serializable_commits_and_reports_no_retries(self, database):
        result = db.run_serializable(
            lambda cur: cur.execute("SELECT 1 AS one").fetchone(),
            label="test.select",
            settings=database,
        )
        assert result.value["one"] == 1
        assert result.retries == 0
        assert result.attempts == 1

    def test_autocommit_path_also_works(self, database):
        result = db.run_autocommit(
            lambda cur: cur.execute("SELECT 2 AS two").fetchone(),
            label="test.select.naive",
            settings=database,
        )
        assert result.value["two"] == 2

    def test_ping_reports_a_cockroachdb_version(self, database):
        assert "CockroachDB" in db.ping(database)

    def test_non_retryable_errors_propagate(self, database):
        import psycopg

        with pytest.raises(psycopg.errors.UndefinedTable):
            db.run_serializable(
                lambda cur: cur.execute("SELECT * FROM table_that_does_not_exist"),
                label="test.bad_sql",
                settings=database,
            )
