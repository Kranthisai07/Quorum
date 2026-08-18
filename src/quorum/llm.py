"""Model access: reasoning and embeddings.

Amazon Bedrock serves these through **two different endpoints with two different
auth paths**, which is the single most surprising thing in this file and the
reason it exists as a layer rather than two inline calls:

* **Claude** (reasoning, and later the contradiction classifier) goes through the
  Messages API endpoint, `bedrock-mantle.{region}.api.aws/anthropic/v1/messages`,
  via `AnthropicBedrockMantle`. Model IDs there are clean and carry an
  `anthropic.` prefix — `anthropic.claude-opus-5`, no date or revision suffix.
* **Titan embeddings** are not on that endpoint at all. They go through the
  classic `bedrock-runtime` `InvokeModel` API via boto3, with the older-style
  id `amazon.titan-embed-text-v2:0`.

Getting these confused costs an hour at the worst possible moment, so
`quorum bedrock check` verifies both independently against a real account.

The `stub` backend is a first-class citizen, not a mock. Every test and the
entire Phase 2 stress suite run against it, so the coordination engine can be
exercised without credentials, without spend, and deterministically.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from quorum.config import Settings, get_settings
from quorum.logging import get_logger

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class Completion:
    """One model response, plus what it cost."""

    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    stop_reason: str | None = None

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"


@dataclass(frozen=True)
class Embedding:
    """One embedding vector, and the model that produced it."""

    vector: list[float]
    model: str
    input_tokens: int = 0

    @property
    def dimensions(self) -> int:
        return len(self.vector)


@dataclass
class HealthReport:
    """What `quorum bedrock check` found."""

    backend: str
    region: str | None = None
    text_model: str | None = None
    embed_model: str | None = None
    text_ok: bool = False
    embed_ok: bool = False
    embed_dimensions: int | None = None
    expected_dimensions: int | None = None
    gates: dict[str, dict[str, str]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.text_ok and self.embed_ok and not self.errors

    @property
    def dimensions_match(self) -> bool:
        return self.embed_dimensions == self.expected_dimensions

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "region": self.region,
            "text_model": self.text_model,
            "embed_model": self.embed_model,
            "text_ok": self.text_ok,
            "embed_ok": self.embed_ok,
            "embed_dimensions": self.embed_dimensions,
            "expected_dimensions": self.expected_dimensions,
            "dimensions_match": self.dimensions_match,
            "gates": self.gates,
            "ok": self.ok,
            "errors": self.errors,
            "notes": self.notes,
        }


@runtime_checkable
class LLMBackend(Protocol):
    """Everything Quorum needs from a model provider."""

    name: str

    def complete(
        self, prompt: str, *, system: str | None = None, max_tokens: int = 8192
    ) -> Completion:
        ...

    def embed(self, text: str) -> Embedding:
        ...

    def health(self) -> HealthReport:
        ...

    @property
    def similarity_threshold(self) -> float:
        """Cosine similarity above which two decisions are worth classifying.

        Owned by the backend, not by global config, because it is a property of
        the *embedding model*. Titan produces normalised semantic vectors where
        0.82 means "about the same thing". The stub produces lexical vectors on
        a completely different scale -- a directly opposed pair scores 0.37
        there. One shared constant across both is not a tuning problem, it is a
        correctness bug: the guard silently stops classifying anything.
        """
        ...


class StubBackend:
    """Deterministic offline backend. No credentials, no spend, no flakiness.

    Embeddings are hashed bag-of-words, L2-normalised to the configured width.
    That makes them *lexically* similar for similar text, which is enough to
    exercise vector storage, index wiring, and threshold plumbing — but it is not
    semantic, and it deliberately cannot tell "use httpx" from "stick with
    requests". Catching that is the classifier's job, and the reason Phase 4
    needs a real model rather than cosine distance alone.
    """

    name = "stub"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 8192,  # noqa: ARG002 -- the stub has nothing to cap
    ) -> Completion:
        digest = hashlib.sha256((system or "").encode() + prompt.encode()).hexdigest()[:12]
        return Completion(
            text=f"[stub completion {digest}]",
            model="stub",
            input_tokens=len(prompt) // 4,
            output_tokens=8,
            stop_reason="end_turn",
        )

    def embed(self, text: str) -> Embedding:
        return Embedding(
            vector=_hashed_embedding(text, self.settings.embed_dim),
            model="stub",
            input_tokens=len(text) // 4,
        )

    @property
    def similarity_threshold(self) -> float:
        """Zero: consider every active decision in the scope.

        The stub's lexical similarity cannot be trusted to *rank* opposition --
        an opposed pair can score below an unrelated one, since opposition is
        about meaning and this measures shared words. So it does not threshold
        at all; `scope` does the narrowing, and the (free, offline) heuristic
        classifier looks at each candidate. With a real embedding model the
        threshold does real work and this becomes 0.82.
        """
        return 0.0

    def health(self) -> HealthReport:
        vector = self.embed("health check")
        return HealthReport(
            backend=self.name,
            text_model="stub",
            embed_model="stub",
            text_ok=True,
            embed_ok=True,
            embed_dimensions=vector.dimensions,
            expected_dimensions=self.settings.embed_dim,
            notes=["stub backend: no AWS credentials required, no spend"],
        )


def _hashed_embedding(text: str, dimensions: int) -> list[float]:
    """Deterministic hashed bag-of-words, L2-normalised."""
    vector = [0.0] * dimensions
    for token in _TOKEN_RE.findall(text.lower()):
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]


class BedrockBackend:
    """Amazon Bedrock: Claude for reasoning, Titan for embeddings.

    Both clients are built lazily so that importing this module never touches
    the network or requires credentials — the stub path must stay usable on a
    laptop with no AWS account at all.
    """

    name = "bedrock"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._claude: Any | None = None
        self._runtime: Any | None = None

    @property
    def claude(self) -> Any:
        """Messages API client. Claude only."""
        if self._claude is None:
            try:
                from anthropic import AnthropicBedrockMantle
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    'Bedrock reasoning needs the AWS extra: pip install "quorum[aws]"'
                ) from exc
            self._claude = AnthropicBedrockMantle(aws_region=self.settings.aws_region)
        return self._claude

    @property
    def runtime(self) -> Any:
        """boto3 bedrock-runtime client. Titan embeddings only."""
        if self._runtime is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    'Bedrock embeddings need the AWS extra: pip install "quorum[aws]"'
                ) from exc
            self._runtime = boto3.client(
                "bedrock-runtime", region_name=self.settings.aws_region
            )
        return self._runtime

    def complete(
        self, prompt: str, *, system: str | None = None, max_tokens: int = 8192
    ) -> Completion:
        request: dict[str, Any] = {
            "model": self.settings.bedrock_text_model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            request["system"] = system
        if self.settings.bedrock_thinking == "adaptive":
            # Adaptive thinking, not a fixed budget: budget_tokens is rejected
            # outright on current models.
            request["thinking"] = {"type": "adaptive"}

        response = self.claude.messages.create(**request)

        # A safety decline arrives as HTTP 200 with stop_reason "refusal", so
        # `content` must never be read before `stop_reason` is checked.
        if getattr(response, "stop_reason", None) == "refusal":
            log.warning(
                "bedrock.refused",
                extra={"model": self.settings.bedrock_text_model},
            )
            return Completion(
                text="",
                model=str(response.model),
                stop_reason="refusal",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        return Completion(
            text=text,
            model=str(response.model),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            stop_reason=str(response.stop_reason),
        )

    def embed(self, text: str) -> Embedding:
        """Titan Text Embeddings V2, through InvokeModel rather than Messages."""
        body = json.dumps(
            {
                "inputText": text,
                "dimensions": self.settings.embed_dim,
                # Normalised vectors are what cosine distance assumes, and the
                # decisions index is vector_cosine_ops.
                "normalize": True,
            }
        )
        response = self.runtime.invoke_model(
            modelId=self.settings.bedrock_embed_model,
            body=body,
            accept="application/json",
            contentType="application/json",
        )
        payload = json.loads(response["body"].read())
        return Embedding(
            vector=[float(value) for value in payload["embedding"]],
            model=self.settings.bedrock_embed_model,
            input_tokens=int(payload.get("inputTextTokenCount", 0)),
        )

    @property
    def similarity_threshold(self) -> float:
        return self.settings.semantic_threshold

    def model_gates(self, model_id: str) -> dict[str, str]:
        """Which of Bedrock's four access gates this model has cleared.

        A 403 from Bedrock says only "not available for this account", which is
        four different problems wearing one message: the model is not offered in
        this region, IAM does not authorise it, the account is not entitled, or
        the provider's use-case agreement has not been accepted. Reporting the
        gate that actually failed is the difference between a one-click fix and
        an afternoon.
        """
        try:
            import boto3

            client = boto3.client("bedrock", region_name=self.settings.aws_region)
            response = client.get_foundation_model_availability(modelId=model_id)
        except Exception as exc:
            return {"lookup": f"unavailable ({type(exc).__name__})"}

        agreement = response.get("agreementAvailability", {})
        return {
            "region": str(response.get("regionAvailability", "?")),
            "authorization": str(response.get("authorizationStatus", "?")),
            "entitlement": str(response.get("entitlementAvailability", "?")),
            "agreement": str(agreement.get("status", "?")),
        }

    def health(self) -> HealthReport:
        """Exercise both endpoints independently and report which one failed."""
        report = HealthReport(
            backend=self.name,
            region=self.settings.aws_region,
            text_model=self.settings.bedrock_text_model,
            embed_model=self.settings.bedrock_embed_model,
            expected_dimensions=self.settings.embed_dim,
        )

        report.gates[self.settings.bedrock_text_model] = self.model_gates(
            self.settings.bedrock_text_model
        )
        report.gates[self.settings.bedrock_embed_model] = self.model_gates(
            self.settings.bedrock_embed_model
        )
        for model_id, gates in report.gates.items():
            if gates.get("agreement") == "NOT_AVAILABLE":
                report.notes.append(
                    f"{model_id}: provider use-case agreement not accepted. Every "
                    f"other gate is clear, so this is the only blocker -- accept it "
                    f"in the Bedrock console (Model catalog -> the model -> "
                    f"playground) and re-run."
                )
            elif gates.get("authorization") not in (None, "AUTHORIZED", "?"):
                report.notes.append(f"{model_id}: IAM does not authorise this model.")
            elif gates.get("entitlement") == "NOT_AVAILABLE":
                report.notes.append(f"{model_id}: account is not entitled to this model.")

        try:
            completion = self.complete("Reply with the single word: ready.", max_tokens=16)
            report.text_ok = bool(completion.text) or completion.refused
            if completion.refused:
                report.notes.append("text model refused the probe; endpoint is reachable")
            else:
                report.notes.append(f"text model replied: {completion.text.strip()[:60]!r}")
        except Exception as exc:
            report.errors.append(f"text ({self.settings.bedrock_text_model}): {exc}")

        try:
            embedding = self.embed("quorum embedding probe")
            report.embed_ok = True
            report.embed_dimensions = embedding.dimensions
            if not report.dimensions_match:
                report.errors.append(
                    f"embedding width {embedding.dimensions} does not match "
                    f"VECTOR({self.settings.embed_dim}) in the schema -- fix "
                    f"QUORUM_EMBED_DIM or add a migration before Phase 4"
                )
        except Exception as exc:
            report.errors.append(f"embeddings ({self.settings.bedrock_embed_model}): {exc}")

        return report


def get_backend(settings: Settings | None = None) -> LLMBackend:
    """Resolve the configured backend. Defaults to the stub."""
    settings = settings or get_settings()
    if settings.llm_backend == "bedrock":
        return BedrockBackend(settings)
    return StubBackend(settings)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity, for tests and for reporting neighbour distances."""
    if len(left) != len(right):
        raise ValueError(f"dimension mismatch: {len(left)} vs {len(right)}")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def truncate(text: str, limit: int) -> str:
    """Trim source text for a prompt without pretending it was complete."""
    if len(text) <= limit:
        return text
    head = textwrap.shorten(text[:limit], width=limit, placeholder="")
    return f"{head}\n\n... [truncated {len(text) - limit} characters] ..."
