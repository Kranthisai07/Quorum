"""Deciding whether two decisions actually conflict.

The vector index finds *near neighbours*. It cannot tell you what the
relationship between them is, and this is the crux of conflict #2:

    "standardise the transport layer on httpx"
    "keep the requests adapter for the unix socket transport"

Those are directly opposed. They share almost no vocabulary, so lexical
similarity misses them entirely, and even a good embedding places them near each
other only because they are *about the same thing* -- which is equally true of
two decisions that agree. **Contradiction and near-duplication look identical to
cosine distance.** Only a judge that reads both statements can separate them.

That is why the pipeline is: ANN narrows thousands of decisions to a handful,
then a model classifies that handful. The index makes the judge affordable; the
judge makes the index meaningful.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from quorum.llm import LLMBackend
from quorum.logging import get_logger

log = get_logger(__name__)

Relation = Literal["agrees", "contradicts", "unrelated"]
Winner = Literal["incumbent", "challenger"]

SYSTEM_PROMPT = """\
You are adjudicating between two decisions made independently by different \
agents working on the same codebase at the same time. They govern the same \
scope, and a similarity search flagged them as related.

Classify the relationship as exactly one of:

- "agrees": they say compatible things. Following both is possible and \
produces consistent code. Near-restatements count as agreement.
- "contradicts": following both is impossible. Code written under one would \
have to be rewritten under the other. Two different answers to the same \
question is a contradiction, even if both are reasonable.
- "unrelated": they concern different questions that happen to share \
vocabulary.

Be strict about "contradicts": it triggers rework of finished code. But do not \
soften a genuine conflict into "agrees" because both options are defensible -- \
the whole point is that the codebase can only have one of them.

If and only if the relation is "contradicts", also choose which should win:

- "incumbent": the decision already recorded. Prefer it when it is more \
specific, better justified, or when other work already depends on it.
- "challenger": the new decision. Prefer it when it is better reasoned, \
handles a case the incumbent missed, or corrects an error.

Reply with only a JSON object, no prose and no code fence:

{"relation": "...", "confidence": 0.0-1.0,
 "winner": "incumbent" | "challenger" | null,
 "reasoning": "one or two sentences"}
"""


@dataclass(frozen=True)
class Judgement:
    """What the judge concluded about two decisions."""

    relation: Relation
    confidence: float
    reasoning: str
    winner: Winner | None = None
    model: str = "unknown"

    @property
    def is_conflict(self) -> bool:
        return self.relation == "contradicts"

    @property
    def is_duplicate(self) -> bool:
        return self.relation == "agrees"

    def to_dict(self) -> dict[str, Any]:
        return {
            "relation": self.relation,
            "confidence": self.confidence,
            "winner": self.winner,
            "reasoning": self.reasoning,
            "model": self.model,
        }


@runtime_checkable
class Classifier(Protocol):
    name: str

    def classify(self, scope: str, incumbent: str, challenger: str) -> Judgement: ...


class ModelClassifier:
    """Asks the configured model to adjudicate. The real implementation."""

    name = "model"

    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend

    def classify(self, scope: str, incumbent: str, challenger: str) -> Judgement:
        prompt = (
            f"Scope under contention: {scope}\n\n"
            f"Decision already recorded (incumbent):\n{incumbent}\n\n"
            f"New decision being proposed (challenger):\n{challenger}\n"
        )
        completion = self.backend.complete(prompt, system=SYSTEM_PROMPT, max_tokens=1024)

        if completion.refused:
            # A refusal must not be read as "no conflict". Escalate instead:
            # unresolved is a state a human can inspect, silence is not.
            log.warning("classifier.refused", extra={"scope": scope})
            return Judgement(
                relation="contradicts",
                confidence=0.0,
                reasoning=(
                    "Model declined to classify; escalated for review rather "
                    "than assumed safe."
                ),
                winner="incumbent",
                model=completion.model,
            )

        return _parse(completion.text, completion.model)


def _parse(text: str, model: str) -> Judgement:
    """Read the judge's JSON, tolerating a code fence or surrounding prose."""
    payload = _extract_json(text)
    if payload is None:
        log.warning("classifier.unparseable", extra={"response": text[:200]})
        return Judgement(
            relation="contradicts",
            confidence=0.0,
            reasoning=f"Unparseable classifier response: {text[:120]!r}. Escalated.",
            winner="incumbent",
            model=model,
        )

    relation = str(payload.get("relation", "")).lower()
    if relation not in ("agrees", "contradicts", "unrelated"):
        relation = "contradicts"  # unknown label escalates rather than disappears

    winner = payload.get("winner")
    if winner not in ("incumbent", "challenger"):
        winner = "incumbent" if relation == "contradicts" else None

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return Judgement(
        relation=relation,  # type: ignore[arg-type]
        confidence=max(0.0, min(1.0, confidence)),
        reasoning=str(payload.get("reasoning", "")).strip(),
        winner=winner,  # type: ignore[arg-type]
        model=model,
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# Words that mark a statement as preserving the status quo, versus changing it.
# Crude on purpose -- see HeuristicClassifier.
_KEEP = frozenset(
    {"keep", "keeps", "retain", "retains", "stay", "stays", "remain", "remains",
     "preserve", "preserves", "continue", "unchanged", "as-is"}
)
_CHANGE = frozenset(
    {"replace", "replaces", "migrate", "migrates", "adopt", "adopts", "switch",
     "switches", "standardise", "standardize", "move", "moves", "convert",
     "converts", "rewrite", "rewrites", "drop", "drops", "remove", "removes"}
)
_TOKEN_RE = re.compile(r"[a-z]+")

# Above this Jaccard overlap the heuristic calls two statements restatements.
_RESTATEMENT_OVERLAP = 0.6


class HeuristicClassifier:
    """Offline stand-in that can still detect an opposing decision.

    Deliberately crude, and documented as such: it looks for one statement
    preserving the status quo while the other changes it, within the same scope.
    That is enough to demonstrate the mechanism end to end with no credentials
    and no spend, and it is *not* a substitute for the model -- it has no
    understanding of what is being kept or changed, only of the verbs.

    The moment a real backend is configured, `build_classifier` uses the model
    instead. This exists so the pipeline is runnable and testable offline, not
    so anyone can claim semantic classification without a model.
    """

    name = "heuristic"

    def classify(self, scope: str, incumbent: str, challenger: str) -> Judgement:  # noqa: ARG002
        left = set(_TOKEN_RE.findall(incumbent.lower()))
        right = set(_TOKEN_RE.findall(challenger.lower()))

        opposed = (bool(left & _KEEP) and bool(right & _CHANGE)) or (
            bool(left & _CHANGE) and bool(right & _KEEP)
        )
        if opposed:
            return Judgement(
                relation="contradicts",
                confidence=0.6,
                reasoning=(
                    "Heuristic: one statement preserves the current approach while "
                    "the other changes it, within the same scope."
                ),
                winner="incumbent",
                model="heuristic",
            )

        overlap = len(left & right) / max(1, len(left | right))
        if overlap > _RESTATEMENT_OVERLAP:
            return Judgement(
                relation="agrees",
                confidence=0.5,
                reasoning="Heuristic: statements are near-restatements of each other.",
                model="heuristic",
            )

        return Judgement(
            relation="unrelated",
            confidence=0.3,
            reasoning="Heuristic: no opposing polarity and little overlap.",
            model="heuristic",
        )


def build_classifier(backend: LLMBackend) -> Classifier:
    """Model classifier when a real backend is configured, heuristic otherwise."""
    if backend.name == "stub":
        return HeuristicClassifier()
    return ModelClassifier(backend)
