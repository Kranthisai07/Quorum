"""The actual work: migrate one file off one library onto another.

This is the part Quorum is *not* about. It exists so the coordination engine has
something real to coordinate, and so the demo shows real files changing. The
interesting properties are:

* It reads the source repository and never writes to it. Output goes to the
  artifact store, so `fixtures/` stays pristine across runs.
* It produces a unified diff, which is reviewable evidence rather than a claim.
* It returns a **finding** alongside the diff. Findings are how an agent tells
  the workspace something the next agent needs to know, and in Phase 5 an
  invalidating finding is what triggers a cascade.

Both backends implement the same contract. The stub produces a real, valid patch
(an annotated no-op) so that the whole pipeline — claim, work, artifact, finding,
complete — is exercisable with no credentials and no spend.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quorum.artifacts import write_result
from quorum.config import Settings, get_settings
from quorum.llm import Completion, LLMBackend, get_backend, truncate
from quorum.logging import get_logger

log = get_logger(__name__)

MAX_SOURCE_CHARS = 60_000

SYSTEM_PROMPT = """\
You are migrating one file of a Python codebase from one HTTP library to \
another, as part of a repo-wide migration being carried out by several agents \
working in parallel on different files.

Rules:
- Preserve the public behaviour and the public API of the module exactly.
- Change only what the migration requires. This is not a refactor.
- If something cannot be migrated without changing behaviour, leave it alone \
and say so in the finding.
- Other agents are migrating other files right now. Do not invent a shared \
helper module, and do not assume anything about files you cannot see.

Reply with exactly these two sections and nothing else:

<migrated>
```python
(the complete migrated file)
```
</migrated>

<finding>
(One or two sentences: what you changed, or what blocked you. If you discovered \
something that affects OTHER files -- a shared convention, an exception mapping \
every caller depends on, a behaviour that cannot be preserved -- say so \
explicitly and begin the sentence with "AFFECTS OTHERS:".)
</finding>
"""

_MIGRATED_RE = re.compile(
    r"<migrated>\s*```(?:python)?\s*(?P<code>.*?)```\s*</migrated>", re.DOTALL
)
_FINDING_RE = re.compile(r"<finding>\s*(?P<finding>.*?)\s*</finding>", re.DOTALL)


class MigrationError(RuntimeError):
    """The model produced something that could not be used."""


@dataclass(frozen=True)
class MigrationResult:
    """One migrated file, ready to be stored and reported."""

    target: str
    diff: str
    finding: str
    changed: bool
    invalidates: bool
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def changed_lines(self) -> int:
        return sum(
            1
            for line in self.diff.splitlines()
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        )


def migrate_unit(
    spec: dict[str, Any],
    *,
    repo_root: Path,
    backend: LLMBackend | None = None,
    settings: Settings | None = None,
) -> MigrationResult:
    """Migrate the file named by a work unit spec."""
    settings = settings or get_settings()
    model = backend or get_backend(settings)

    target = str(spec["path"])
    source_path = (repo_root / target).resolve()
    if not source_path.is_file():
        raise MigrationError(f"source file not found: {source_path}")

    original = source_path.read_text(encoding="utf-8")

    if model.name == "stub":
        return _stub_migration(target, original, spec)

    completion = model.complete(
        _build_prompt(target, original, spec),
        system=SYSTEM_PROMPT,
        max_tokens=32_000,
    )
    if completion.refused:
        raise MigrationError(f"model declined to migrate {target}")

    migrated, finding = _parse(completion, target)
    diff = unified_diff(target, original, migrated)

    return MigrationResult(
        target=target,
        diff=diff,
        finding=finding,
        changed=bool(diff.strip()),
        invalidates=_is_invalidating(finding),
        model=completion.model,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
    )


def _build_prompt(target: str, source: str, spec: dict[str, Any]) -> str:
    evidence = spec.get("evidence", {})
    evidence_lines = "\n".join(
        f"  - {name}: lines {', '.join(str(n) for n in lines)}"
        for name, lines in sorted(evidence.items())
    )
    constraints = "\n".join(f"  - {c}" for c in spec.get("constraints", []))
    scopes = ", ".join(spec.get("scopes", []))

    return f"""\
File: {target}
Migrate from: {spec.get("from_library")}
Migrate to: {spec.get("to_library")}

Library-specific idioms detected in this file:
{evidence_lines or "  (none recorded)"}

Decision scopes this file touches: {scopes or "(none)"}
{f"Repo-wide constraints:{chr(10)}{constraints}" if constraints else ""}

--- BEGIN {target} ---
{truncate(source, MAX_SOURCE_CHARS)}
--- END {target} ---
"""


def _parse(completion: Completion, target: str) -> tuple[str, str]:
    migrated_match = _MIGRATED_RE.search(completion.text)
    if migrated_match is None:
        raise MigrationError(
            f"no <migrated> block in the response for {target} "
            f"(stop_reason={completion.stop_reason})"
        )
    finding_match = _FINDING_RE.search(completion.text)
    finding = (
        finding_match.group("finding").strip()
        if finding_match
        else f"Migrated {target}; the model reported no finding."
    )
    return migrated_match.group("code"), finding


def _is_invalidating(finding: str) -> bool:
    """A finding that changes what other agents should do invalidates their work.

    Phase 5 walks the dependency graph from exactly these. The marker is part of
    the prompt contract rather than an inference, because "does this finding
    affect other files" is a question the agent that made the discovery is far
    better placed to answer than a regex is.
    """
    return finding.strip().upper().startswith("AFFECTS OTHERS:")


def unified_diff(target: str, original: str, migrated: str) -> str:
    """A reviewable patch, which is the artifact a human actually wants."""
    return "".join(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            migrated.splitlines(keepends=True),
            fromfile=f"a/{target}",
            tofile=f"b/{target}",
            n=3,
        )
    )


def _stub_migration(target: str, original: str, spec: dict[str, Any]) -> MigrationResult:
    """A real, valid patch that changes nothing, for credential-free runs.

    Deliberately not a fake diff: it annotates the file with the migration that
    *would* happen, so the artifact pipeline handles genuine patch text and the
    demo shows real files changing without ever calling a model.
    """
    from_library = spec.get("from_library", "requests")
    to_library = spec.get("to_library", "httpx")
    evidence = spec.get("evidence", {})
    hits = sum(len(lines) for lines in evidence.values())

    banner = (
        f"# quorum: stub migration -- {from_library} -> {to_library}\n"
        f"# {hits} library-specific idiom(s) across {len(evidence)} signal(s) "
        f"would be rewritten here.\n"
    )
    migrated = banner + original
    diff = unified_diff(target, original, migrated)

    finding = (
        f"Stub migration of {target}: {hits} {from_library} idiom(s) identified "
        f"across scopes {', '.join(spec.get('scopes', [])) or 'none'}. "
        f"No model was called."
    )
    # The error-mapping hub is the file whose decisions every caller inherits,
    # so the stub marks it invalidating -- giving Phase 5 a deterministic
    # cascade trigger that needs no model.
    if "error-mapping" in spec.get("scopes", []):
        finding = (
            f"AFFECTS OTHERS: {target} maps {from_library} exceptions for the "
            f"whole codebase; every caller inherits whatever mapping is chosen here."
        )

    return MigrationResult(
        target=target,
        diff=diff,
        finding=finding,
        changed=bool(diff.strip()),
        invalidates=_is_invalidating(finding),
        model="stub",
    )


def store_result(
    workspace_id: Any,
    result: MigrationResult,
    version: int,
    *,
    settings: Settings | None = None,
) -> str:
    """Persist the diff and return the `result_ref` for the work unit."""
    artifact = write_result(
        workspace_id, result.target, version, result.diff, settings=settings
    )
    log.info(
        "migration.stored",
        extra={
            "target": result.target,
            "ref": artifact.ref,
            "changed_lines": result.changed_lines,
            "bytes": artifact.size_bytes,
        },
    )
    return artifact.ref
