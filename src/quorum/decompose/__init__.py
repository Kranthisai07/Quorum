"""Pluggable task decomposition.

Importing this package registers the bundled decomposers, so
`get_decomposer("code_migration")` works without the caller knowing where the
implementation lives.
"""

from __future__ import annotations

from quorum.decompose.base import (
    Decomposer,
    Decomposition,
    WorkUnitSpec,
    get_decomposer,
    register,
    registered,
)
from quorum.decompose.code_migration import CodeMigrationDecomposer

__all__ = [
    "CodeMigrationDecomposer",
    "Decomposer",
    "Decomposition",
    "WorkUnitSpec",
    "get_decomposer",
    "register",
    "registered",
]
