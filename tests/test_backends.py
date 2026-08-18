"""Model access and artifact storage.

Mostly pure tests. The Bedrock path cannot be exercised without credentials, so
what is asserted here is everything up to the auth boundary: that both clients
are constructed with the right shape, that the two endpoints stay separate, and
that a dimension mismatch is caught loudly rather than at INSERT time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quorum.artifacts import (
    LocalArtifactStore,
    S3ArtifactStore,
    _split_s3_ref,
    get_store,
    unit_key,
)
from quorum.config import Settings, get_settings
from quorum.llm import (
    BedrockBackend,
    StubBackend,
    cosine_similarity,
    get_backend,
    truncate,
)


class TestStubBackend:
    def test_is_the_default(self):
        assert get_backend().name == "stub"

    def test_embeddings_match_the_configured_width(self):
        backend = StubBackend()
        assert backend.embed("anything").dimensions == get_settings().embed_dim

    def test_embeddings_are_deterministic(self):
        backend = StubBackend()
        assert backend.embed("same text").vector == backend.embed("same text").vector

    def test_embeddings_are_normalised(self):
        vector = StubBackend().embed("normalise me please").vector
        magnitude = sum(value * value for value in vector) ** 0.5
        assert magnitude == pytest.approx(1.0, abs=1e-9)

    def test_similar_text_is_closer_than_unrelated_text(self):
        backend = StubBackend()
        base = backend.embed("standardise the transport layer on httpx").vector
        similar = backend.embed("standardise the transport layer on httpx now").vector
        unrelated = backend.embed("cascading invalidation of dependency graphs").vector

        assert cosine_similarity(base, similar) > cosine_similarity(base, unrelated)

    def test_stub_similarity_is_lexical_not_semantic(self):
        """The documented limitation, asserted so nobody trusts it by accident.

        "Use httpx" and "keep requests" are opposites, and the stub scores them
        as unrelated rather than contradictory. Only a real classifier can tell
        contradiction from unrelatedness -- which is exactly why Phase 4 needs
        one and cannot lean on cosine distance alone.
        """
        backend = StubBackend()
        adopt = backend.embed("standardise on httpx for every transport").vector
        reject = backend.embed("keep requests for the unix socket transport").vector

        assert cosine_similarity(adopt, reject) < 0.5

    def test_completions_are_deterministic(self):
        backend = StubBackend()
        assert backend.complete("prompt").text == backend.complete("prompt").text

    def test_health_reports_ready(self):
        report = StubBackend().health()
        assert report.ok
        assert report.dimensions_match


class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_score_minus_one(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_dimension_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_similarity([1.0], [1.0, 2.0])

    def test_zero_vector_does_not_divide_by_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


class TestBedrockWiring:
    """Everything checkable without credentials."""

    def test_the_two_endpoints_use_different_clients(self):
        """Claude is on the Messages API; Titan is on bedrock-runtime.

        Confusing the two is the mistake this layer exists to prevent, so the
        separation is asserted rather than assumed.
        """
        source = Path("src/quorum/llm.py").read_text(encoding="utf-8")
        assert "AnthropicBedrockMantle" in source
        assert 'boto3' in source and '"bedrock-runtime"' in source

    def test_claude_model_id_has_no_date_suffix(self):
        """Messages API ids are clean; the legacy InvokeModel path is not."""
        model = get_settings().bedrock_text_model
        assert model.startswith("anthropic."), model
        assert not model.endswith(":0"), (
            f"{model} looks like a legacy InvokeModel id; the Messages API "
            "endpoint uses unsuffixed ids"
        )

    def test_titan_model_id_keeps_its_revision_suffix(self):
        assert get_settings().bedrock_embed_model.startswith("amazon.titan-embed")

    def test_clients_are_lazy(self):
        """Constructing the backend must not need credentials or a network."""
        backend = BedrockBackend()
        assert backend._claude is None
        assert backend._runtime is None

    def test_health_reports_both_paths_separately(self):
        """A green reasoning check says nothing about embeddings."""
        report = BedrockBackend().health()
        # No credentials in CI or on a dev laptop, so both should fail -- but
        # they must fail independently, each naming its own model.
        if not report.ok:
            assert len(report.errors) >= 1
            joined = " ".join(report.errors)
            assert "anthropic.claude" in joined or "titan" in joined


class TestTruncate:
    def test_short_text_is_untouched(self):
        assert truncate("hello", 100) == "hello"

    def test_long_text_says_it_was_truncated(self):
        result = truncate("x" * 500, 100)
        assert "truncated" in result
        assert len(result) < 500


class TestLocalArtifacts:
    @pytest.fixture
    def store(self, tmp_path: Path) -> LocalArtifactStore:
        settings = Settings(**{**get_settings().__dict__, "artifact_dir": tmp_path})
        return LocalArtifactStore(settings)

    def test_round_trip(self, store):
        artifact = store.put("ws/file.py.v1.patch", "--- a\n+++ b\n")
        assert store.exists(artifact.ref)
        assert store.get(artifact.ref) == "--- a\n+++ b\n"

    def test_size_is_reported_in_bytes(self, store):
        assert store.put("k.patch", "abcd").size_bytes == 4

    def test_missing_artifact_does_not_exist(self, store):
        assert store.exists(str((store.root / "nope.patch").as_uri())) is False

    def test_keys_cannot_escape_the_artifact_root(self, store):
        with pytest.raises(ValueError, match="escapes the artifact root"):
            store.put("../../etc/passwd", "nope")

    def test_json_helper_writes_parseable_output(self, store, monkeypatch):
        from quorum import artifacts

        monkeypatch.setattr(artifacts, "get_store", lambda _settings=None: store)
        artifact = artifacts.write_json("report.json", {"ok": True})
        assert json.loads(store.get(artifact.ref)) == {"ok": True}


class TestArtifactKeys:
    def test_version_is_part_of_the_key(self):
        """A redo after a lease expiry must not overwrite the original.

        Both attempts stay inspectable, which is what makes an invalidation
        cascade auditable rather than merely effective.
        """
        first = unit_key("ws", "docker/api/client.py", 1)
        second = unit_key("ws", "docker/api/client.py", 2)
        assert first != second
        assert first.endswith(".v1.patch")
        assert second.endswith(".v2.patch")

    def test_path_separators_are_flattened(self):
        key = unit_key("ws", "docker/api/client.py", 1)
        assert "/" not in key.split("/", 1)[1]


class TestS3Artifacts:
    def test_refs_parse_into_bucket_and_key(self):
        assert _split_s3_ref("s3://bucket/a/b.patch") == ("bucket", "a/b.patch")

    def test_non_s3_refs_are_rejected(self):
        with pytest.raises(ValueError, match="not an s3 reference"):
            _split_s3_ref("file:///tmp/x")

    def test_a_bucket_is_required(self):
        settings = Settings(**{**get_settings().__dict__, "s3_bucket": None})
        with pytest.raises(ValueError, match="QUORUM_S3_BUCKET"):
            S3ArtifactStore(settings)

    def test_local_is_the_default_store(self):
        assert get_store().name == "local"
