"""Environment-driven configuration.

Everything that differs between the local dev loop and the cloud deployment is
resolved here, so no other module needs to know which profile it is running
under. Local must always work without any AWS credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

Profile = Literal["local", "cloud"]
LLMBackend = Literal["stub", "bedrock"]
ArtifactBackend = Literal["local", "s3"]
Runner = Literal["local", "lambda"]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


@dataclass(frozen=True)
class Settings:
    """Resolved runtime configuration."""

    profile: Profile
    db_url: str
    llm_backend: LLMBackend
    aws_region: str
    bedrock_text_model: str
    bedrock_embed_model: str
    bedrock_thinking: str
    embed_dim: int
    artifact_backend: ArtifactBackend
    artifact_dir: Path
    s3_bucket: str | None
    runner: Runner
    lambda_function: str | None
    claim_lease_seconds: int
    heartbeat_seconds: int
    txn_max_retries: int
    semantic_threshold: float
    conflict_detection: bool
    log_level: str
    log_format: str
    repo_root: Path = field(default=REPO_ROOT)

    @property
    def cockroach_binary(self) -> Path:
        """Path to the vendored local CockroachDB binary (dev profile only)."""
        suffix = ".exe" if os.name == "nt" else ""
        return self.repo_root / ".tools" / "crdb" / f"cockroach{suffix}"

    @property
    def local_store_dir(self) -> Path:
        return self.repo_root / ".tools" / "crdb-data"

    def database_name(self) -> str:
        """Database name parsed out of the connection URL."""
        base = self.db_url.split("?", 1)[0]
        _, separator, name = base.rpartition("/")
        return name if separator and name else "quorum"

    def admin_url(self) -> str:
        """Same cluster, but pointed at `defaultdb` so we can CREATE DATABASE."""
        base, _, query = self.db_url.partition("?")
        head, _, _tail = base.rpartition("/")
        admin = f"{head}/defaultdb"
        return f"{admin}?{query}" if query else admin


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process, reading `.env` if present."""
    load_dotenv(REPO_ROOT / ".env", override=False)

    artifact_dir = Path(_env("QUORUM_ARTIFACT_DIR", "./artifacts"))
    if not artifact_dir.is_absolute():
        artifact_dir = (REPO_ROOT / artifact_dir).resolve()

    return Settings(
        profile=_env("QUORUM_PROFILE", "local"),  # type: ignore[arg-type]
        db_url=_env(
            "QUORUM_DB_URL",
            "postgresql://root@127.0.0.1:26257/quorum?sslmode=disable",
        ),
        llm_backend=_env("QUORUM_LLM_BACKEND", "stub"),  # type: ignore[arg-type]
        aws_region=_env("AWS_REGION", "us-east-1"),
        # Bedrock model IDs carry an `anthropic.` prefix.
        bedrock_text_model=_env("QUORUM_BEDROCK_TEXT_MODEL", "anthropic.claude-opus-5"),
        bedrock_embed_model=_env("QUORUM_BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v2:0"),
        bedrock_thinking=_env("QUORUM_BEDROCK_THINKING", "adaptive"),
        embed_dim=_env_int("QUORUM_EMBED_DIM", 1024),
        artifact_backend=_env("QUORUM_ARTIFACT_BACKEND", "local"),  # type: ignore[arg-type]
        artifact_dir=artifact_dir,
        s3_bucket=os.environ.get("QUORUM_S3_BUCKET") or None,
        runner=_env("QUORUM_RUNNER", "local"),  # type: ignore[arg-type]
        lambda_function=os.environ.get("QUORUM_LAMBDA_FUNCTION") or None,
        claim_lease_seconds=_env_int("QUORUM_CLAIM_LEASE_SECONDS", 30),
        heartbeat_seconds=_env_int("QUORUM_HEARTBEAT_SECONDS", 10),
        txn_max_retries=_env_int("QUORUM_TXN_MAX_RETRIES", 8),
        semantic_threshold=_env_float("QUORUM_SEMANTIC_THRESHOLD", 0.82),
        conflict_detection=_env("QUORUM_CONFLICT_DETECTION", "true").lower()
        not in {"0", "false", "no"},
        log_level=_env("QUORUM_LOG_LEVEL", "INFO"),
        log_format=_env("QUORUM_LOG_FORMAT", "json"),
    )
