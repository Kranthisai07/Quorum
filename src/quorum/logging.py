"""Structured (JSON) logging.

Every interesting event in Quorum is a coordination event, and coordination
events are only useful if they can be replayed after the fact. So logging is
structured from the first commit: it doubles as the observability story and as
the evidence trail behind the conflict feed.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
    | {"message", "asctime", "taskName"}
)

# Where colliding `extra` keys are parked. See `QuorumLogger.makeRecord`.
_OVERFLOW = "quorum_fields"

# Fields attached to every record emitted inside a `log_context` block.
_context: dict[str, Any] = {}


class QuorumLogger(logging.Logger):
    """A logger that tolerates `extra` keys colliding with LogRecord attributes.

    The stdlib raises `KeyError` if `extra` contains a name LogRecord already
    uses -- `name`, `module`, `args`, and a dozen others. Agents log
    domain-shaped fields (`name`, `status`, `module`), and a logging call is not
    allowed to be the thing that takes a coordination run down, so colliding
    keys are parked in a side dict and merged back by the formatter.
    """

    def makeRecord(  # signature is fixed by logging.Logger
        self,
        name: str,
        level: int,
        fn: str,
        lno: int,
        msg: object,
        args: Any,
        exc_info: Any,
        func: str | None = None,
        extra: dict[str, Any] | None = None,
        sinfo: str | None = None,
    ) -> logging.LogRecord:
        overflow: dict[str, Any] = {}
        safe: dict[str, Any] = {}
        for key, value in (extra or {}).items():
            (overflow if key in _RESERVED else safe)[key] = value
        if overflow:
            safe[_OVERFLOW] = overflow
        return super().makeRecord(
            name, level, fn, lno, msg, args, exc_info, func, safe or None, sinfo
        )


logging.setLoggerClass(QuorumLogger)


class JsonFormatter(logging.Formatter):
    """Render a log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
            "pid": os.getpid(),
        }
        payload.update(_context)
        payload.update(_record_fields(record))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable fallback for interactive debugging."""

    def format(self, record: logging.LogRecord) -> str:
        extras = dict(_context)
        extras.update(_record_fields(record))
        tail = " ".join(f"{k}={v}" for k, v in extras.items())
        base = f"{record.levelname:<5} {record.name} {record.getMessage()}"
        return f"{base}  {tail}" if tail else base


def _record_fields(record: logging.LogRecord) -> dict[str, Any]:
    """Structured fields carried by a record, overflow keys folded back in."""
    fields = {
        key: _coerce(value)
        for key, value in record.__dict__.items()
        if key not in _RESERVED and key != _OVERFLOW
    }
    overflow = record.__dict__.get(_OVERFLOW)
    if isinstance(overflow, dict):
        fields.update({key: _coerce(value) for key, value in overflow.items()})
    return fields


def _coerce(value: Any) -> Any:
    if isinstance(value, uuid.UUID):
        return str(value)
    return value


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Install the root handler. Safe to call more than once."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Attach `fields` to every log record emitted inside the block.

    Used to tag an agent's whole lifecycle with its workspace and session id
    without threading them through every call site.
    """
    previous = dict(_context)
    _context.update({k: _coerce(v) for k, v in fields.items()})
    try:
        yield
    finally:
        _context.clear()
        _context.update(previous)
