"""Task decomposition contract.

Quorum coordinates concurrent agents; it does not care what they are working on.
Everything domain-specific lives behind :class:`Decomposer`, so the coordination
engine only ever sees work units, dependency edges, and decision scopes. Code
migration is the reference implementation, not a built-in assumption.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class WorkUnitSpec:
    """One independently claimable piece of work.

    `target` is the stable natural key inside a workspace (a file path, for code
    migration). `scopes` names the shared decision surfaces this unit is likely
    to touch -- the places where two agents can independently reach conflicting
    conclusions. It is an input to semantic conflict detection, not an output.
    """

    target: str
    spec: dict[str, Any]
    scopes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Decomposition:
    """The full result of decomposing a task spec."""

    units: list[WorkUnitSpec]
    # Edges are (unit_target, depends_on_target), both of which must appear in
    # `units`. Stored rather than derived so the cascade can walk them in SQL.
    deps: list[tuple[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def targets(self) -> set[str]:
        return {unit.target for unit in self.units}

    def validate(self) -> None:
        """Reject a decomposition the coordination engine could not run.

        Called before anything is written, because a dangling dependency edge
        would make the invalidation cascade quietly incomplete -- exactly the
        class of corruption Quorum exists to prevent.
        """
        targets = self.targets()
        if len(targets) != len(self.units):
            duplicates = _duplicates(unit.target for unit in self.units)
            raise ValueError(f"duplicate work unit targets: {sorted(duplicates)}")

        for unit_target, depends_on in self.deps:
            if unit_target not in targets:
                raise ValueError(f"dependency edge from unknown unit: {unit_target}")
            if depends_on not in targets:
                raise ValueError(f"dependency edge to unknown unit: {depends_on}")
            if unit_target == depends_on:
                raise ValueError(f"self-dependency on unit: {unit_target}")

        if len(set(self.deps)) != len(self.deps):
            raise ValueError("duplicate dependency edges")

    def scopes(self) -> list[str]:
        """Every distinct decision scope present, in stable order."""
        seen: dict[str, None] = {}
        for unit in self.units:
            for scope in unit.scopes:
                seen.setdefault(scope, None)
        return sorted(seen)


def _duplicates(values: Any) -> set[str]:
    seen: set[str] = set()
    dupes: set[str] = set()
    for value in values:
        if value in seen:
            dupes.add(value)
        seen.add(value)
    return dupes


@runtime_checkable
class Decomposer(Protocol):
    """Turns a task spec into claimable work units and their dependencies."""

    name: str

    def decompose(self, task_spec: Mapping[str, Any]) -> Decomposition:
        """Produce a validated decomposition, or raise ValueError."""
        ...


_REGISTRY: dict[str, Decomposer] = {}


def register(decomposer: Decomposer) -> Decomposer:
    """Register a decomposer under its `name`."""
    if decomposer.name in _REGISTRY:
        raise ValueError(f"decomposer already registered: {decomposer.name}")
    _REGISTRY[decomposer.name] = decomposer
    return decomposer


def get_decomposer(name: str) -> Decomposer:
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(f"unknown decomposer {name!r}; registered: {known}") from None


def registered() -> Sequence[str]:
    return sorted(_REGISTRY)
