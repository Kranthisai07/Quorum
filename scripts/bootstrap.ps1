<#
.SYNOPSIS
    One command from clone to a running Quorum workspace (Windows).

.DESCRIPTION
    Creates a virtualenv, installs the package, downloads and starts a local
    single-node CockroachDB, applies migrations, and seeds a workspace from the
    vendored fixture. Safe to re-run: every step is idempotent.

.PARAMETER SkipSeed
    Set up the database but do not seed a workspace.
#>
[CmdletBinding()]
param(
    [switch]$SkipSeed
)

$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

function Write-Step($message) {
    Write-Host ""
    Write-Host "==> $message" -ForegroundColor Cyan
}

Write-Step "Checking Python"
$python = (Get-Command python -ErrorAction SilentlyContinue)
if ($null -eq $python) {
    throw "Python 3.11+ is required but 'python' was not found on PATH."
}
$version = & python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
Write-Host "    python $version at $($python.Source)"
if ([version]$version -lt [version]"3.11") {
    throw "Python 3.11+ is required; found $version."
}

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Step "Creating virtualenv (.venv)"
    & python -m venv .venv
} else {
    Write-Step "Reusing existing virtualenv (.venv)"
}

Write-Step "Installing quorum and its dev dependencies"
& $venvPython -m pip install --quiet --upgrade pip
& $venvPython -m pip install --quiet -e ".[dev]"

$quorum = Join-Path $repoRoot ".venv\Scripts\quorum.exe"

Write-Step "Starting local CockroachDB (downloads the binary on first run)"
& $quorum db up

Write-Step "Applying schema migrations"
& $quorum migrate

if (-not $SkipSeed) {
    Write-Step "Seeding a workspace from fixtures/docker-py"
    & $quorum seed tasks/requests-to-httpx.json --name docker-py-safe --mode safe
}

Write-Host ""
Write-Host "Quorum is ready." -ForegroundColor Green
Write-Host ""
Write-Host "  .venv\Scripts\quorum.exe workspaces      list workspaces"
Write-Host "  .venv\Scripts\quorum.exe status docker-py-safe"
Write-Host "  .venv\Scripts\quorum.exe decompose --units"
Write-Host "  .venv\Scripts\quorum.exe db down         stop the local cluster"
Write-Host ""
Write-Host "  DB Console: http://127.0.0.1:8080"
Write-Host ""
