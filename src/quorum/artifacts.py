"""Artifact storage: where an agent's actual output goes.

`work_units.result_ref` holds a pointer, never the payload. Migration diffs run
to tens of kilobytes and there is no reason to carry that through a coordination
transaction — the database's job here is agreement about *who did what*, not
blob storage.

Two backends behind one interface: the local filesystem for development, S3 for
the cloud profile. Both produce a `result_ref` string that round-trips through
:func:`load`, so nothing downstream needs to know which one is in use.

Agents never write into `fixtures/`. Results land here instead, which keeps the
vendored repository pristine across runs and `git status` clean.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse

from quorum.config import Settings, get_settings
from quorum.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Artifact:
    """A stored result and the reference that finds it again."""

    ref: str
    size_bytes: int


@runtime_checkable
class ArtifactStore(Protocol):
    name: str

    def put(self, key: str, payload: str) -> Artifact: ...

    def get(self, ref: str) -> str: ...

    def exists(self, ref: str) -> bool: ...


class LocalArtifactStore:
    """Files under `QUORUM_ARTIFACT_DIR`, referenced as `file://` URLs."""

    name = "local"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.root = self.settings.artifact_dir

    def _path(self, key: str) -> Path:
        # Keys are workspace/unit scoped and generated internally, but a key is
        # still a path component: refuse anything that could escape the root.
        candidate = (self.root / key).resolve()
        if not str(candidate).startswith(str(self.root.resolve())):
            raise ValueError(f"artifact key escapes the artifact root: {key!r}")
        return candidate

    def put(self, key: str, payload: str) -> Artifact:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        log.info("artifact.written", extra={"ref": key, "bytes": len(payload)})
        return Artifact(ref=path.as_uri(), size_bytes=len(payload.encode("utf-8")))

    def get(self, ref: str) -> str:
        return Path(_local_path_from_ref(ref)).read_text(encoding="utf-8")

    def exists(self, ref: str) -> bool:
        return Path(_local_path_from_ref(ref)).exists()


def _local_path_from_ref(ref: str) -> str:
    parsed = urlparse(ref)
    if parsed.scheme != "file":
        return ref
    # file:///D:/... -> D:/... (Windows drive letters arrive with a leading /)
    path = parsed.path
    drive_prefix = len("/X:")
    if len(path) >= drive_prefix and path[2] == ":":
        return path[1:]
    return path


class S3ArtifactStore:
    """Objects in `QUORUM_S3_BUCKET`, referenced as `s3://bucket/key`."""

    name = "s3"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        if not self.settings.s3_bucket:
            raise ValueError("QUORUM_S3_BUCKET must be set for the s3 artifact backend")
        self.bucket = self.settings.s3_bucket
        self._client: Any | None = None

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client("s3", region_name=self.settings.aws_region)
        return self._client

    def put(self, key: str, payload: str) -> Artifact:
        encoded = payload.encode("utf-8")
        self.client.put_object(
            Bucket=self.bucket, Key=key, Body=encoded, ContentType="text/plain"
        )
        log.info("artifact.written", extra={"ref": key, "bytes": len(encoded)})
        return Artifact(ref=f"s3://{self.bucket}/{key}", size_bytes=len(encoded))

    def get(self, ref: str) -> str:
        bucket, key = _split_s3_ref(ref)
        response = self.client.get_object(Bucket=bucket, Key=key)
        return str(response["Body"].read().decode("utf-8"))

    def exists(self, ref: str) -> bool:
        from botocore.exceptions import ClientError

        bucket, key = _split_s3_ref(ref)
        try:
            self.client.head_object(Bucket=bucket, Key=key)
        except ClientError:
            return False
        return True


def _split_s3_ref(ref: str) -> tuple[str, str]:
    parsed = urlparse(ref)
    if parsed.scheme != "s3":
        raise ValueError(f"not an s3 reference: {ref!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def get_store(settings: Settings | None = None) -> ArtifactStore:
    settings = settings or get_settings()
    if settings.artifact_backend == "s3":
        return S3ArtifactStore(settings)
    return LocalArtifactStore(settings)


def unit_key(workspace_id: Any, target: str, version: int) -> str:
    """Stable, collision-free key for one attempt at one work unit.

    The version is part of the key on purpose: when a lease expires and another
    agent redoes the unit at version+1, that result must not overwrite the
    original. Both remain inspectable, which is what makes an invalidation
    cascade auditable rather than merely effective.
    """
    safe_target = target.replace("/", "__").replace("\\", "__")
    return f"{workspace_id}/{safe_target}.v{version}.patch"


def write_result(
    workspace_id: Any,
    target: str,
    version: int,
    payload: str,
    *,
    settings: Settings | None = None,
) -> Artifact:
    """Store one migration result and return its reference."""
    return get_store(settings).put(unit_key(workspace_id, target, version), payload)


def write_json(
    key: str, payload: dict[str, Any], *, settings: Settings | None = None
) -> Artifact:
    return get_store(settings).put(key, json.dumps(payload, indent=2))
