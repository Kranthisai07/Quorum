"""Local single-node CockroachDB lifecycle.

Development must not require Docker, a cloud account, or a manual install: the
setup story for this project is one command from clone to running demo. So the
binary is fetched on demand into `.tools/` (gitignored) and started as a
detached single node.

This is the `local` profile only. The `cloud` profile talks to CockroachDB Cloud
and never touches anything in here.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import psycopg

from quorum.config import Settings, get_settings
from quorum.logging import get_logger

log = get_logger(__name__)

DOWNLOAD_BASE = "https://binaries.cockroachdb.com"
# Pinned rather than "latest" so a clone reproduces the cluster this was built
# and tested against. Vector indexes need v25.2 or newer.
COCKROACH_VERSION = "v26.2.5"

# Bound to the IPv4 loopback explicitly: on Windows "localhost" resolves to
# ::1 first, and every connection then pays the full connect timeout before
# falling back to 127.0.0.1.
LISTEN_HOST = "127.0.0.1"
SQL_PORT = 26257
HTTP_PORT = 8080


@dataclass(frozen=True)
class NodeStatus:
    """Observed state of the local node."""

    running: bool
    reachable: bool
    pid: int | None
    version: str | None
    sql_url: str
    console_url: str


def _platform_slug() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = "arm64" if machine in {"arm64", "aarch64"} else "amd64"
    if system == "windows":
        return f"windows-6.2-{arch}"
    if system == "darwin":
        return f"darwin-10.9-{arch}"
    return f"linux-{arch}"


def _archive_name() -> str:
    slug = _platform_slug()
    suffix = "zip" if slug.startswith("windows") else "tgz"
    return f"cockroach-{COCKROACH_VERSION}.{slug}.{suffix}"


def ensure_binary(settings: Settings | None = None) -> Path:
    """Return the local cockroach binary, downloading it if it is missing."""
    settings = settings or get_settings()
    binary = settings.cockroach_binary
    if binary.exists():
        return binary

    on_path = shutil.which("cockroach")
    if on_path:
        log.info("cockroach.using_path_binary", extra={"path": on_path})
        return Path(on_path)

    archive_name = _archive_name()
    url = f"{DOWNLOAD_BASE}/{archive_name}"
    tools = binary.parent.parent
    tools.mkdir(parents=True, exist_ok=True)
    archive_path = tools / archive_name

    log.info("cockroach.download", extra={"url": url, "dest": str(archive_path)})
    urllib.request.urlretrieve(url, archive_path)  # noqa: S310 -- fixed https host

    extract_dir = tools / "_extract"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True)

    if archive_name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_dir)
    else:
        import tarfile

        with tarfile.open(archive_path) as tf:
            tf.extractall(extract_dir)  # noqa: S202 -- trusted vendor archive

    extracted = next(extract_dir.glob("cockroach-*"))
    target_dir = binary.parent
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.move(str(extracted), str(target_dir))
    shutil.rmtree(extract_dir, ignore_errors=True)
    archive_path.unlink(missing_ok=True)

    if not binary.exists():
        raise RuntimeError(f"cockroach binary not found after extraction: {binary}")
    binary.chmod(0o755)
    log.info("cockroach.installed", extra={"path": str(binary), "version": COCKROACH_VERSION})
    return binary


def _pid_file(settings: Settings) -> Path:
    return settings.repo_root / ".tools" / "cockroach.pid"


def _read_pid(settings: Settings) -> int | None:
    pid_file = _pid_file(settings)
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except ValueError:
        return None


def _pid_alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(  # noqa: S603 -- fixed argv, no shell
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],  # noqa: S607 -- Windows builtin
            capture_output=True,
            text=True,
            check=False,
        )
        return str(pid) in out.stdout
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def start(settings: Settings | None = None, *, wait_seconds: float = 60.0) -> NodeStatus:
    """Start the local node if it is not already up, then wait until it answers."""
    settings = settings or get_settings()

    existing = status(settings)
    if existing.reachable:
        log.info("cockroach.already_running", extra={"pid": existing.pid})
        return existing

    binary = ensure_binary(settings)
    store = settings.local_store_dir
    logs = settings.repo_root / ".tools" / "logs"
    store.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    command = [
        str(binary),
        "start-single-node",
        "--insecure",
        f"--listen-addr={LISTEN_HOST}:{SQL_PORT}",
        f"--http-addr={LISTEN_HOST}:{HTTP_PORT}",
        f"--store={store}",
        f"--log-dir={logs}",
    ]

    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

    stdout_path = logs / "cockroach-stdout.log"
    with stdout_path.open("ab") as stdout:
        process = subprocess.Popen(  # noqa: S603 -- fixed argv, no shell
            command,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            start_new_session=os.name != "nt",
        )

    _pid_file(settings).write_text(str(process.pid))
    log.info("cockroach.starting", extra={"pid": process.pid, "store": str(store)})

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        current = status(settings)
        if current.reachable:
            log.info("cockroach.ready", extra={"pid": current.pid, "version": current.version})
            return current
        time.sleep(1.0)

    raise TimeoutError(
        f"local CockroachDB did not become reachable within {wait_seconds:.0f}s; "
        f"see {stdout_path}"
    )


def stop(settings: Settings | None = None) -> bool:
    """Stop the local node. Returns True if something was stopped."""
    settings = settings or get_settings()
    pid = _read_pid(settings)
    if pid is None or not _pid_alive(pid):
        _pid_file(settings).unlink(missing_ok=True)
        log.info("cockroach.not_running")
        return False

    if os.name == "nt":
        subprocess.run(  # noqa: S603 -- fixed argv, no shell
            ["taskkill", "/PID", str(pid), "/F"],  # noqa: S607 -- Windows builtin
            capture_output=True,
            check=False,
        )
    else:
        import signal

        os.kill(pid, signal.SIGTERM)

    _pid_file(settings).unlink(missing_ok=True)
    log.info("cockroach.stopped", extra={"pid": pid})
    return True


def status(settings: Settings | None = None) -> NodeStatus:
    """Probe the local node without starting anything."""
    settings = settings or get_settings()
    pid = _read_pid(settings)
    running = pid is not None and _pid_alive(pid)

    version: str | None = None
    reachable = False
    try:
        with psycopg.connect(settings.admin_url(), connect_timeout=3) as conn, conn.cursor() as cur:
            cur.execute("SELECT version()")
            row = cur.fetchone()
            version = None if row is None else str(row[0])
            reachable = True
    except psycopg.Error:
        reachable = False

    return NodeStatus(
        running=running,
        reachable=reachable,
        pid=pid,
        version=version,
        sql_url=settings.db_url,
        console_url=f"http://{LISTEN_HOST}:{HTTP_PORT}",
    )


def wipe(settings: Settings | None = None) -> None:
    """Stop the node and delete its store. Destroys all local data."""
    settings = settings or get_settings()
    stop(settings)
    store = settings.local_store_dir
    if store.exists():
        shutil.rmtree(store, ignore_errors=True)
    log.info("cockroach.wiped", extra={"store": str(store)})
