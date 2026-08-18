"""Reference decomposer: migrate a repository off one HTTP library onto another.

Chosen because it decomposes cleanly into per-file work units while still
generating genuine cross-file coupling: the same conclusion about error
mapping or client lifetime has to hold in twenty files at once, and agents
working different files will reach that conclusion independently. That is the
raw material for semantic conflict.

Three things are produced per file:

* **evidence** -- which library-specific idioms appear, and where. This is what
  an agent reads before touching the file, and what makes a work unit worth
  claiming rather than a bare filename.
* **scopes** -- the shared decision surfaces the file touches. Two agents
  writing decisions into the same scope are the semantic-conflict candidates.
* **dependency edges** -- derived from the real Python import graph, so an
  invalidation actually follows how the code is coupled rather than a guess.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quorum.decompose.base import Decomposition, WorkUnitSpec, register
from quorum.logging import get_logger

log = get_logger(__name__)

MAX_EVIDENCE_LINES = 8


@dataclass(frozen=True)
class Signal:
    """A library-specific idiom worth migrating, and what it commits you to."""

    name: str
    pattern: re.Pattern[str]
    scope: str
    # Strong signals are enough on their own to make a file a work unit; weak
    # ones only count once a strong signal has already matched, so that a file
    # merely calling `.json()` on some unrelated object is not swept in.
    strong: bool = False


def _sig(name: str, pattern: str, scope: str, *, strong: bool = False) -> Signal:
    return Signal(name, re.compile(pattern), scope, strong)


# Tuned for requests -> httpx, the migration the bundled fixture needs. Adding a
# different migration means adding a signal table, not touching the engine.
REQUESTS_SIGNALS: tuple[Signal, ...] = (
    _sig(
        "import",
        r"^\s*(?:import\s+requests|from\s+requests\b)",
        "dependency-surface",
        strong=True,
    ),
    _sig("session", r"\brequests\.Session\b|\bSession\(\)", "client-lifecycle", strong=True),
    _sig(
        "adapter",
        r"\bHTTPAdapter\b|\bmount\(|\bpoolmanager\b|\bPoolManager\b",
        "transport-adapter",
        strong=True,
    ),
    _sig("urllib3", r"\burllib3\b", "transport-adapter", strong=True),
    _sig(
        "exceptions",
        r"\brequests\.exceptions\b|\bRequestException\b|\bConnectionError\b|\bReadTimeout\b",
        "error-mapping",
        strong=True,
    ),
    _sig("raise_for_status", r"\braise_for_status\(", "error-mapping", strong=True),
    _sig(
        "verb",
        r"\brequests\.(?:get|post|put|patch|delete|head|options|request)\(",
        "request-invocation",
        strong=True,
    ),
    _sig(
        "streaming",
        r"\bstream\s*=\s*True\b|\biter_content\(|\biter_lines\(|\bresponse\.raw\b",
        "streaming",
        strong=True,
    ),
    _sig("timeout", r"\btimeout\s*=", "timeout-policy"),
    _sig("tls", r"\bverify\s*=|\bcert\s*=|\bSSLContext\b", "tls-verification"),
    _sig(
        "decoding",
        r"\.status_code\b|\.raise_for_status\b|\bresp(?:onse)?\.json\(\)",
        "response-decoding",
    ),
)

SIGNAL_TABLES: dict[str, tuple[Signal, ...]] = {
    "requests": REQUESTS_SIGNALS,
}


class CodeMigrationDecomposer:
    """Decompose a repository migration into per-file work units."""

    name = "code_migration"

    def decompose(self, task_spec: Mapping[str, Any]) -> Decomposition:
        repo_root = Path(str(task_spec["repo"])).resolve()
        if not repo_root.is_dir():
            raise ValueError(f"repo path is not a directory: {repo_root}")

        from_library = str(task_spec.get("from_library", "requests"))
        to_library = str(task_spec.get("to_library", "httpx"))
        package_roots = [str(p) for p in task_spec.get("package_roots", ["."])]
        excludes = tuple(str(p) for p in task_spec.get("exclude", ("tests", "test", "docs")))

        signals = SIGNAL_TABLES.get(from_library)
        if signals is None:
            raise ValueError(
                f"no signal table for library {from_library!r}; "
                f"known: {sorted(SIGNAL_TABLES)}"
            )

        sources = list(self._iter_sources(repo_root, package_roots, excludes))
        module_index = _build_module_index(repo_root, sources)

        units: list[WorkUnitSpec] = []
        for path in sources:
            unit = self._scan(path, repo_root, signals, from_library, to_library)
            if unit is not None:
                units.append(unit)

        units.sort(key=lambda u: u.target)
        deps = _import_edges(repo_root, units, module_index)

        decomposition = Decomposition(
            units=units,
            deps=deps,
            metadata={
                "decomposer": self.name,
                "repo": _rel(repo_root, repo_root.parent),
                "from_library": from_library,
                "to_library": to_library,
                "files_scanned": len(sources),
                "files_selected": len(units),
                "dependency_edges": len(deps),
            },
        )
        decomposition.validate()
        log.info(
            "decompose.complete",
            extra={
                "decomposer": self.name,
                "units": len(units),
                "deps": len(deps),
                "scanned": len(sources),
                "scopes": decomposition.scopes(),
            },
        )
        return decomposition

    def _iter_sources(
        self, repo_root: Path, package_roots: Iterable[str], excludes: tuple[str, ...]
    ) -> Iterable[Path]:
        for package_root in package_roots:
            base = (repo_root / package_root).resolve()
            if not base.is_dir():
                raise ValueError(f"package root not found: {base}")
            for path in sorted(base.rglob("*.py")):
                rel = _rel(path, repo_root)
                parts = rel.split("/")
                if any(part in excludes for part in parts):
                    continue
                if any(part.startswith("test_") or part.endswith("_test.py") for part in parts):
                    continue
                yield path

    def _scan(
        self,
        path: Path,
        repo_root: Path,
        signals: tuple[Signal, ...],
        from_library: str,
        to_library: str,
    ) -> WorkUnitSpec | None:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            log.warning("decompose.skip_undecodable", extra={"path": str(path)})
            return None

        lines = text.splitlines()
        evidence: dict[str, list[int]] = {}
        scopes: list[str] = []
        strong_hits = 0
        total_hits = 0

        for signal in signals:
            hits = [i for i, line in enumerate(lines, start=1) if signal.pattern.search(line)]
            if not hits:
                continue
            evidence[signal.name] = hits[:MAX_EVIDENCE_LINES]
            total_hits += len(hits)
            if signal.strong:
                strong_hits += len(hits)
            if signal.scope not in scopes:
                scopes.append(signal.scope)

        if strong_hits == 0:
            return None

        target = _rel(path, repo_root)
        return WorkUnitSpec(
            target=target,
            scopes=tuple(scopes),
            spec={
                "kind": "code_migration",
                "path": target,
                "module": _module_name(path, repo_root),
                "from_library": from_library,
                "to_library": to_library,
                "loc": len(lines),
                "signal_hits": total_hits,
                "strong_hits": strong_hits,
                "evidence": evidence,
                "scopes": list(scopes),
                "instruction": (
                    f"Migrate {target} from {from_library} to {to_library}, "
                    f"preserving public behaviour."
                ),
            },
        )


def _rel(path: Path, root: Path) -> str:
    """Repo-relative POSIX path, so targets are stable across platforms."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def _module_name(path: Path, repo_root: Path) -> str:
    rel = _rel(path, repo_root)
    without_suffix = rel[: -len(".py")] if rel.endswith(".py") else rel
    parts = without_suffix.split("/")
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _build_module_index(repo_root: Path, sources: list[Path]) -> dict[str, str]:
    """Dotted module name -> repo-relative path, for every scanned source."""
    return {_module_name(path, repo_root): _rel(path, repo_root) for path in sources}


def _import_edges(
    repo_root: Path, units: list[WorkUnitSpec], module_index: dict[str, str]
) -> list[tuple[str, str]]:
    """Dependency edges taken from the real import graph.

    An edge `(a, b)` means "unit a imports unit b", so invalidating b must
    reconsider a. Edges pointing outside the unit set are dropped: they are real
    couplings, but nothing in this workspace can invalidate them.
    """
    unit_targets = {unit.target for unit in units}
    edges: set[tuple[str, str]] = set()

    for unit in units:
        path = repo_root / unit.target
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError) as exc:
            log.warning(
                "decompose.unparsed", extra={"path": unit.target, "reason": type(exc).__name__}
            )
            continue

        own_module = _module_name(path, repo_root)
        for imported in _imported_modules(tree, own_module):
            target = _resolve(imported, module_index)
            if target is None or target == unit.target or target not in unit_targets:
                continue
            edges.add((unit.target, target))

    return sorted(edges)


def _imported_modules(tree: ast.AST, own_module: str) -> Iterable[str]:
    """Dotted module names imported by a file, with relative imports resolved."""
    package_parts = own_module.split(".")[:-1]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module:
                    yield node.module
                    for alias in node.names:
                        yield f"{node.module}.{alias.name}"
                continue
            # Relative import: climb `level - 1` packages from this file.
            base = package_parts[: len(package_parts) - (node.level - 1)]
            prefix = base + ([node.module] if node.module else [])
            if prefix:
                yield ".".join(prefix)
            for alias in node.names:
                yield ".".join([*prefix, alias.name])


def _resolve(module: str, module_index: dict[str, str]) -> str | None:
    """Longest-prefix match of a dotted import against known modules.

    `from docker.errors import APIError` yields both `docker.errors` and
    `docker.errors.APIError`; only the former is a module, so we walk back up
    the dotted path until something resolves.
    """
    parts = module.split(".")
    while parts:
        candidate = ".".join(parts)
        if candidate in module_index:
            return module_index[candidate]
        parts.pop()
    return None


DECOMPOSER = register(CodeMigrationDecomposer())
