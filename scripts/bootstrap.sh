#!/usr/bin/env bash
# One command from clone to a running Quorum workspace (macOS / Linux).
#
# Creates a virtualenv, installs the package, downloads and starts a local
# single-node CockroachDB, applies migrations, and seeds a workspace from the
# vendored fixture. Safe to re-run: every step is idempotent.
#
# Usage: ./scripts/bootstrap.sh [--skip-seed]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SKIP_SEED=0
for arg in "$@"; do
    case "$arg" in
        --skip-seed) SKIP_SEED=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

step() { printf '\n==> %s\n' "$1"; }

step "Checking Python"
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python 3.11+ is required but '$PYTHON' was not found on PATH." >&2
    exit 1
fi
"$PYTHON" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(f"Python 3.11+ is required; found {sys.version.split()[0]}")
print(f"    python {sys.version.split()[0]} at {sys.executable}")
PY

VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
    step "Creating virtualenv (.venv)"
    "$PYTHON" -m venv .venv
else
    step "Reusing existing virtualenv (.venv)"
fi

step "Installing quorum and its dev dependencies"
"$VENV_PYTHON" -m pip install --quiet --upgrade pip
"$VENV_PYTHON" -m pip install --quiet -e ".[dev]"

QUORUM="$REPO_ROOT/.venv/bin/quorum"

step "Starting local CockroachDB (downloads the binary on first run)"
"$QUORUM" db up

step "Applying schema migrations"
"$QUORUM" migrate

if [ "$SKIP_SEED" -eq 0 ]; then
    step "Seeding a workspace from fixtures/docker-py"
    "$QUORUM" seed tasks/requests-to-httpx.json --name docker-py-safe --mode safe
fi

cat <<'EOF'

Quorum is ready.

  .venv/bin/quorum workspaces            list workspaces
  .venv/bin/quorum status docker-py-safe
  .venv/bin/quorum decompose --units
  .venv/bin/quorum db down               stop the local cluster

  DB Console: http://127.0.0.1:8080

EOF
