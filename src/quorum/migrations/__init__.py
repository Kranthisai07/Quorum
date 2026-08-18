"""Versioned schema migrations.

Migrations are plain `.sql` files in this package, named `NNN_description.sql`.
The runner applies unapplied files in version order and records each one in
`schema_migrations` with a checksum, so a migration that is edited after being
applied is caught instead of silently diverging.

A file may opt out of transactional application with a `-- quorum:no-transaction`
directive in its header comment. Vector index creation uses this: CockroachDB
backfills the index as a schema-change job, which does not belong inside the
same transaction as the DDL that queues it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import psycopg

from quorum.config import Settings, get_settings
from quorum.db import connection
from quorum.logging import get_logger

log = get_logger(__name__)

_FILENAME_RE = re.compile(r"^(\d+)_(.+)\.sql$")
_NO_TXN_DIRECTIVE = "quorum:no-transaction"

BOOTSTRAP_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INT PRIMARY KEY,
    name        STRING NOT NULL,
    checksum    STRING NOT NULL,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


@dataclass(frozen=True)
class Migration:
    """One migration file, resolved and hashed."""

    version: int
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()[:16]

    @property
    def transactional(self) -> bool:
        return _NO_TXN_DIRECTIVE not in self.sql

    @property
    def label(self) -> str:
        return f"{self.version:03d}_{self.name}"


def discover() -> list[Migration]:
    """Load every migration shipped with the package, in version order."""
    found: list[Migration] = []
    for entry in resources.files(__package__).iterdir():
        match = _FILENAME_RE.match(entry.name)
        if match is None:
            continue
        found.append(
            Migration(
                version=int(match.group(1)),
                name=match.group(2),
                sql=entry.read_text(encoding="utf-8"),
            )
        )
    found.sort(key=lambda m: m.version)
    _assert_unique_versions(found)
    return found


def _assert_unique_versions(migrations: list[Migration]) -> None:
    seen: dict[int, str] = {}
    for migration in migrations:
        if migration.version in seen:
            raise ValueError(
                f"duplicate migration version {migration.version}: "
                f"{seen[migration.version]} and {migration.name}"
            )
        seen[migration.version] = migration.name


def split_statements(sql: str) -> list[str]:
    """Split a SQL script into individual statements.

    Deliberately small: it understands line comments, block comments and
    single-quoted literals, which is everything the Quorum migrations use. It is
    not a general SQL parser and does not pretend to be one.
    """
    statements: list[str] = []
    buffer: list[str] = []
    index = 0
    length = len(sql)
    in_string = False
    in_line_comment = False
    in_block_comment = False

    while index < length:
        char = sql[index]
        pair = sql[index : index + 2]

        if in_line_comment:
            buffer.append(char)
            if char == "\n":
                in_line_comment = False
            index += 1
            continue
        if in_block_comment:
            buffer.append(char)
            if pair == "*/":
                buffer.append(sql[index + 1])
                in_block_comment = False
                index += 2
                continue
            index += 1
            continue
        if in_string:
            buffer.append(char)
            if char == "'":
                if pair == "''":  # escaped quote inside a literal
                    buffer.append(sql[index + 1])
                    index += 2
                    continue
                in_string = False
            index += 1
            continue

        if pair == "--":
            in_line_comment = True
            buffer.append(char)
            index += 1
            continue
        if pair == "/*":
            in_block_comment = True
            buffer.append(char)
            index += 1
            continue
        if char == "'":
            in_string = True
            buffer.append(char)
            index += 1
            continue
        if char == ";":
            statements.append("".join(buffer))
            buffer = []
            index += 1
            continue

        buffer.append(char)
        index += 1

    statements.append("".join(buffer))
    return [s.strip() for s in statements if _is_executable(s)]


def _is_executable(statement: str) -> bool:
    """True if the chunk contains anything other than comments and whitespace."""
    stripped = re.sub(r"/\*.*?\*/", "", statement, flags=re.DOTALL)
    stripped = re.sub(r"--[^\n]*", "", stripped)
    return bool(stripped.strip())


def ensure_database(settings: Settings | None = None) -> str:
    """Create the Quorum database if it is missing, and enable vector indexes.

    Connects to `defaultdb` because you cannot create a database from inside it.
    Both statements are idempotent, so this is safe to run on every startup.
    """
    settings = settings or get_settings()
    database = settings.database_name()
    with psycopg.connect(
        settings.admin_url(), autocommit=True, connect_timeout=10
    ) as conn, conn.cursor() as cur:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {_quote_ident(database)}")
        _enable_vector_indexes(cur)
    log.info("db.ready", extra={"database": database})
    return database


# SQLSTATEs that make the vector-index cluster setting safe to skip.
UNDEFINED_PARAMETER = "42P02"  # setting does not exist -- GA'd, already on
INSUFFICIENT_PRIVILEGE = "42501"  # restricted cloud role cannot set it


def _enable_vector_indexes(cur: psycopg.Cursor) -> None:
    """Turn on vector indexes where the cluster still gates them.

    Deliberately narrow. Only two failures mean "carry on": the setting no
    longer exists (v26+, where vector indexes are GA and on by default), and a
    role that is not allowed to touch cluster settings (CockroachDB Cloud, where
    the setting is managed for us). Anything else -- a genuinely unreachable
    cluster, an unsupported tier, a permissions problem elsewhere -- must raise
    here rather than resurface later as a confusing CREATE VECTOR INDEX failure.
    """
    try:
        cur.execute("SET CLUSTER SETTING feature.vector_index.enabled = true")
    except psycopg.Error as exc:
        sqlstate = getattr(exc, "sqlstate", None)
        if sqlstate == UNDEFINED_PARAMETER:
            log.debug(
                "vector_index.setting_absent",
                extra={"sqlstate": sqlstate, "detail": "GA on this cluster; nothing to enable"},
            )
            return
        if sqlstate == INSUFFICIENT_PRIVILEGE:
            log.warning(
                "vector_index.setting_denied",
                extra={
                    "sqlstate": sqlstate,
                    "detail": (
                        "role cannot SET CLUSTER SETTING; assuming vector indexes are "
                        "enabled by the operator. CREATE VECTOR INDEX will fail loudly "
                        "if they are not."
                    ),
                },
            )
            return
        raise


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def applied_versions(settings: Settings | None = None) -> dict[int, str]:
    """Version -> checksum for every migration already applied."""
    with connection(settings) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(BOOTSTRAP_SQL)
            cur.execute("SELECT version, checksum FROM schema_migrations")
            return {int(row["version"]): str(row["checksum"]) for row in cur.fetchall()}


def migrate(settings: Settings | None = None) -> list[Migration]:
    """Apply all pending migrations. Returns the ones that ran."""
    settings = settings or get_settings()
    ensure_database(settings)

    already = applied_versions(settings)
    pending: list[Migration] = []

    for migration in discover():
        recorded = already.get(migration.version)
        if recorded is None:
            pending.append(migration)
        elif recorded != migration.checksum:
            raise RuntimeError(
                f"migration {migration.label} was modified after it was applied "
                f"(recorded checksum {recorded}, file checksum {migration.checksum}). "
                "Add a new migration instead of editing an applied one."
            )

    for migration in pending:
        _apply(migration, settings)

    if not pending:
        log.info("migrate.up_to_date", extra={"applied": len(already)})
    return pending


def _apply(migration: Migration, settings: Settings) -> None:
    statements = split_statements(migration.sql)
    record = (
        "INSERT INTO schema_migrations (version, name, checksum) VALUES (%s, %s, %s)"
    )

    with connection(settings) as conn:
        if migration.transactional:
            conn.autocommit = False
            with conn.transaction(), conn.cursor() as cur:
                for statement in statements:
                    cur.execute(statement)  # type: ignore[arg-type]
                cur.execute(record, (migration.version, migration.name, migration.checksum))
        else:
            conn.autocommit = True
            with conn.cursor() as cur:
                for statement in statements:
                    cur.execute(statement)  # type: ignore[arg-type]
                cur.execute(record, (migration.version, migration.name, migration.checksum))

    log.info(
        "migrate.applied",
        extra={
            "migration": migration.label,
            "statements": len(statements),
            "transactional": migration.transactional,
            "checksum": migration.checksum,
        },
    )


def reset(settings: Settings | None = None) -> None:
    """Drop the Quorum database entirely. Local development convenience."""
    settings = settings or get_settings()
    database = settings.database_name()
    with psycopg.connect(
        settings.admin_url(), autocommit=True, connect_timeout=10
    ) as conn, conn.cursor() as cur:
        cur.execute(f"DROP DATABASE IF EXISTS {_quote_ident(database)} CASCADE")
    log.info("db.dropped", extra={"database": database})


def migrations_dir() -> Path:
    """Filesystem location of the migration files (for docs and tooling)."""
    return Path(str(resources.files(__package__)))
