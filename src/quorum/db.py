"""CockroachDB access: pooling, serializable transactions, retry accounting.

This module is where the central claim of Quorum lives or dies. CockroachDB runs
`SERIALIZABLE` by default, which means a transaction that would have produced an
anomaly is *aborted* (SQLSTATE 40001) rather than silently allowed. Correct
client behaviour is therefore mandatory: retry the whole unit of work, never
just the failing statement.

Two execution modes are exposed on purpose, because the difference between them
is the demo:

* :func:`run_serializable` -- one serializable transaction, retried on conflict.
  This is `safe` mode.
* :func:`run_autocommit` -- every statement its own implicit transaction, no
  retry, no cross-statement isolation. This is `naive` mode: how a typical
  agent memory layer behaves, and the control group we compare against.
"""

from __future__ import annotations

import random
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generic, TypeVar

import psycopg
from psycopg import Connection, Cursor
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from quorum.config import Settings, get_settings
from quorum.logging import get_logger

log = get_logger(__name__)

T = TypeVar("T")

# SQLSTATE 40001. CockroachDB raises this when it cannot serialize a
# transaction; the contract is that the client retries the whole transaction.
SERIALIZATION_FAILURE = "40001"


class TxnStats:
    """Process-wide counters for transaction outcomes.

    Retries are not a failure signal here -- they are the visible cost of
    serializability, and the dashboard reports them as such.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.attempts: Counter[str] = Counter()
        self.retries: Counter[str] = Counter()
        self.commits: Counter[str] = Counter()
        self.failures: Counter[str] = Counter()

    def record_attempt(self, label: str) -> None:
        with self._lock:
            self.attempts[label] += 1

    def record_retry(self, label: str) -> None:
        with self._lock:
            self.retries[label] += 1

    def record_commit(self, label: str) -> None:
        with self._lock:
            self.commits[label] += 1

    def record_failure(self, label: str) -> None:
        with self._lock:
            self.failures[label] += 1

    def snapshot(self) -> dict[str, dict[str, int]]:
        with self._lock:
            labels = set(self.attempts) | set(self.retries) | set(self.failures)
            return {
                label: {
                    "attempts": self.attempts[label],
                    "retries": self.retries[label],
                    "commits": self.commits[label],
                    "failures": self.failures[label],
                }
                for label in sorted(labels)
            }

    def reset(self) -> None:
        with self._lock:
            self.attempts.clear()
            self.retries.clear()
            self.commits.clear()
            self.failures.clear()


TXN_STATS = TxnStats()

_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()
_pool_url: str | None = None


def get_pool(settings: Settings | None = None) -> ConnectionPool:
    """Lazily build (and memoise) the process-wide connection pool."""
    global _pool, _pool_url
    settings = settings or get_settings()
    with _pool_lock:
        if _pool is None or _pool_url != settings.db_url:
            if _pool is not None:
                _pool.close()
            _pool = ConnectionPool(
                settings.db_url,
                min_size=1,
                max_size=16,
                kwargs={
                    "row_factory": dict_row,
                    "application_name": "quorum",
                    # Fail fast on an unreachable cluster instead of hanging.
                    "connect_timeout": 10,
                },
                open=True,
            )
            _pool_url = settings.db_url
        return _pool


def close_pool() -> None:
    """Close the pool. Tests and CLI exit paths call this."""
    global _pool, _pool_url
    with _pool_lock:
        if _pool is not None:
            _pool.close()
        _pool = None
        _pool_url = None


@contextmanager
def connection(settings: Settings | None = None) -> Iterator[Connection]:
    """Borrow a pooled connection."""
    with get_pool(settings).connection() as conn:
        yield conn


@dataclass(frozen=True)
class TxnResult(Generic[T]):
    """The value a transaction produced, plus what it cost to produce it."""

    value: T
    attempts: int
    retries: int
    duration_s: float


def _is_serialization_failure(exc: BaseException) -> bool:
    return (
        isinstance(exc, psycopg.Error)
        and getattr(exc, "sqlstate", None) == SERIALIZATION_FAILURE
    )


def run_serializable(
    fn: Callable[[Cursor], T],
    *,
    label: str,
    settings: Settings | None = None,
    max_retries: int | None = None,
) -> TxnResult[T]:
    """Run `fn` inside a single SERIALIZABLE transaction, retrying on 40001.

    `fn` receives a cursor and must be *idempotent under replay*: on a retry it
    is called again from scratch against a fresh transaction, so it must not
    depend on state it mutated during the aborted attempt.
    """
    settings = settings or get_settings()
    limit = settings.txn_max_retries if max_retries is None else max_retries
    started = time.monotonic()
    attempt = 0

    while True:
        attempt += 1
        TXN_STATS.record_attempt(label)
        try:
            with connection(settings) as conn:
                conn.autocommit = False
                with conn.transaction(), conn.cursor() as cur:
                    cur.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
                    value = fn(cur)
            TXN_STATS.record_commit(label)
            return TxnResult(
                value=value,
                attempts=attempt,
                retries=attempt - 1,
                duration_s=time.monotonic() - started,
            )
        except Exception as exc:
            if _is_serialization_failure(exc) and attempt <= limit:
                TXN_STATS.record_retry(label)
                backoff = _backoff_seconds(attempt)
                log.info(
                    "txn.retry",
                    extra={
                        "txn_label": label,
                        "attempt": attempt,
                        "backoff_s": round(backoff, 4),
                        "sqlstate": SERIALIZATION_FAILURE,
                    },
                )
                time.sleep(backoff)
                continue
            TXN_STATS.record_failure(label)
            raise


def run_autocommit(
    fn: Callable[[Cursor], T],
    *,
    label: str,
    settings: Settings | None = None,
) -> TxnResult[T]:
    """Run `fn` with autocommit on: every statement is its own transaction.

    Deliberately offers *no* isolation across statements and *no* retry. This is
    the `naive` execution path, and its anomalies are the point.
    """
    settings = settings or get_settings()
    started = time.monotonic()
    TXN_STATS.record_attempt(label)
    try:
        with connection(settings) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                value = fn(cur)
        TXN_STATS.record_commit(label)
    except Exception:
        TXN_STATS.record_failure(label)
        raise
    return TxnResult(value=value, attempts=1, retries=0, duration_s=time.monotonic() - started)


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff with full jitter, capped at one second."""
    ceiling = min(0.05 * (2 ** (attempt - 1)), 1.0)
    return random.uniform(0.0, ceiling)  # noqa: S311 -- jitter, not a secret


def vector_literal(values: Sequence[float]) -> str:
    """Format an embedding as a CockroachDB VECTOR literal."""
    return "[" + ",".join(f"{v:.7g}" for v in values) + "]"


def ping(settings: Settings | None = None) -> str:
    """Return the server version string. Used by `quorum db status`."""
    with connection(settings) as conn, conn.cursor() as cur:
        cur.execute("SELECT version() AS v")
        row = cur.fetchone()
        return "" if row is None else str(row["v"])
