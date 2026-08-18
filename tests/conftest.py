"""Shared test fixtures.

Tests split into two groups. Pure tests (decomposition, statement splitting,
logging, config) run anywhere with no database. Tests marked `integration` need
a reachable CockroachDB; they skip rather than fail when one is not running, so
`pytest` stays useful on a laptop with nothing started.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest

from quorum.config import Settings, get_settings
from quorum.logging import configure_logging

REPO_ROOT = Path(__file__).resolve().parents[1]

configure_logging("WARNING", "console")


def _redirect_to_test_database() -> None:
    """Point the whole suite at `<database>_test`.

    Integration tests truncate every table, and a developer who has just seeded
    a demo workspace should not lose it by running `pytest`. Same cluster, same
    migrations, separate database.
    """
    base = os.environ.get(
        "QUORUM_DB_URL", "postgresql://root@127.0.0.1:26257/quorum?sslmode=disable"
    )
    head, _, query = base.partition("?")
    prefix, _, name = head.rpartition("/")
    if not name.endswith("_test"):
        head = f"{prefix}/{name}_test"
    os.environ["QUORUM_DB_URL"] = f"{head}?{query}" if query else head
    get_settings.cache_clear()


def _force_offline_backend() -> None:
    """Pin the suite to the stub backend.

    Tests must never need AWS credentials and must never spend money. A `.env`
    configured for Bedrock -- as it is once the project is wired up for real --
    would otherwise make `pytest` issue paid API calls: slow, flaky, and
    billable. Tests that exercise Bedrock wiring construct the backend
    explicitly instead.
    """
    os.environ["QUORUM_LLM_BACKEND"] = "stub"
    get_settings.cache_clear()


_redirect_to_test_database()
_force_offline_backend()


_redirect_to_test_database()


@pytest.fixture(scope="session")
def settings() -> Settings:
    return get_settings()


@pytest.fixture(scope="session")
def database(settings: Settings) -> Settings:
    """Skip the whole test if the cluster is not reachable, else migrate it."""
    from quorum import migrations

    try:
        with psycopg.connect(settings.admin_url(), connect_timeout=3) as conn:
            conn.execute("SELECT 1")
    except psycopg.Error as exc:
        pytest.skip(f"CockroachDB not reachable at {settings.db_url}: {exc}")

    migrations.migrate(settings)
    return settings


@pytest.fixture
def clean_workspaces(database: Settings) -> Iterator[Settings]:
    """Delete every workspace and its children before and after a test.

    Deletion order follows the foreign keys. Tests that assert on counts need a
    workspace table that only they have written to.
    """
    _truncate(database)
    yield database
    _truncate(database)


def _truncate(settings: Settings) -> None:
    from quorum.db import connection

    with connection(settings) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            for table in (
                "conflict_log",
                "findings",
                "decisions",
                "unit_deps",
                "work_units",
                "agent_sessions",
                "workspaces",
            ):
                cur.execute(f"DELETE FROM {table}")


@pytest.fixture(scope="session")
def fixture_task_spec() -> dict[str, Any]:
    """The bundled docker-py task spec, with `repo` resolved to an absolute path."""
    spec_path = REPO_ROOT / "tasks" / "requests-to-httpx.json"
    spec: dict[str, Any] = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["repo"] = str((REPO_ROOT / str(spec["repo"])).resolve())
    return spec


@pytest.fixture
def tiny_repo(tmp_path: Path) -> Path:
    """A miniature package with a known import graph and known signals.

    Deliberately hand-built: the assertions about which files become work units
    and which dependency edges appear must be exact, which they cannot be
    against a repository that upstream keeps changing.
    """
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")

    (package / "errors.py").write_text(
        "import requests\n"
        "\n"
        "class ApiError(Exception):\n"
        "    pass\n"
        "\n"
        "def wrap(exc):\n"
        "    if isinstance(exc, requests.exceptions.ConnectionError):\n"
        "        raise ApiError from exc\n",
        encoding="utf-8",
    )

    (package / "client.py").write_text(
        "import requests\n"
        "\n"
        "from .errors import wrap\n"
        "\n"
        "def fetch(url):\n"
        "    session = requests.Session()\n"
        "    response = session.get(url, timeout=5, stream=True)\n"
        "    response.raise_for_status()\n"
        "    return response\n",
        encoding="utf-8",
    )

    (package / "transport.py").write_text(
        "from requests.adapters import HTTPAdapter\n"
        "\n"
        "class UnixAdapter(HTTPAdapter):\n"
        "    pass\n",
        encoding="utf-8",
    )

    # No requests idioms at all: must not become a work unit.
    (package / "helpers.py").write_text(
        "def slugify(value):\n    return value.lower()\n", encoding="utf-8"
    )

    # Excluded by path, even though it is full of signals.
    tests_dir = package / "tests"
    tests_dir.mkdir()
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")
    (tests_dir / "test_client.py").write_text(
        "import requests\n\ndef test_it():\n    requests.get('http://x')\n", encoding="utf-8"
    )

    return tmp_path


@pytest.fixture
def tiny_task_spec(tiny_repo: Path) -> dict[str, Any]:
    return {
        "name": f"tiny-{uuid.uuid4().hex[:8]}",
        "kind": "code_migration",
        "repo": str(tiny_repo),
        "package_roots": ["pkg"],
        "exclude": ["tests"],
        "from_library": "requests",
        "to_library": "httpx",
    }
