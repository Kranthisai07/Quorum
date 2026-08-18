"""`quorum` command line interface.

Phase 1 surface: bring up local CockroachDB, apply migrations, decompose a real
repository into work units, and seed a workspace. Later phases add agent, run,
and compare commands to this same app.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from quorum import __version__
from quorum.config import get_settings
from quorum.logging import configure_logging

app = typer.Typer(
    name="quorum",
    help="Shared-memory coordination layer for concurrent AI agents, on CockroachDB.",
    no_args_is_help=True,
    add_completion=False,
)
db_app = typer.Typer(help="Local CockroachDB lifecycle (local profile only).", no_args_is_help=True)
app.add_typer(db_app, name="db")

console = Console()
err_console = Console(stderr=True)

DEFAULT_TASK = Path("tasks/requests-to-httpx.json")


@app.callback()
def _root(
    log_level: Annotated[
        str | None, typer.Option("--log-level", help="Override QUORUM_LOG_LEVEL.")
    ] = None,
    log_format: Annotated[
        str | None, typer.Option("--log-format", help="json or console.")
    ] = None,
) -> None:
    settings = get_settings()
    configure_logging(log_level or settings.log_level, log_format or settings.log_format)


@app.command()
def version() -> None:
    """Print the Quorum version and the resolved profile."""
    settings = get_settings()
    console.print(f"quorum {__version__}")
    console.print(f"profile      {settings.profile}")
    console.print(f"database     {_redact(settings.db_url)}")
    console.print(f"llm backend  {settings.llm_backend}")
    console.print(f"artifacts    {settings.artifact_backend}")
    console.print(f"runner       {settings.runner}")


@db_app.command("up")
def db_up(
    wait: Annotated[float, typer.Option(help="Seconds to wait for readiness.")] = 60.0,
) -> None:
    """Start (and install, if needed) the local single-node cluster."""
    from quorum import localdb

    status = localdb.start(wait_seconds=wait)
    console.print(f"[green]CockroachDB ready[/green]  pid={status.pid}")
    console.print(f"  sql      {_redact(status.sql_url)}")
    console.print(f"  console  {status.console_url}")
    console.print(f"  version  {status.version}")


@db_app.command("down")
def db_down() -> None:
    """Stop the local cluster."""
    from quorum import localdb

    stopped = localdb.stop()
    console.print("[yellow]stopped[/yellow]" if stopped else "not running")


@db_app.command("status")
def db_status() -> None:
    """Report whether the local cluster is up and reachable."""
    from quorum import localdb

    status = localdb.status()
    colour = "green" if status.reachable else "red"
    console.print(f"[{colour}]reachable={status.reachable}[/{colour}] running={status.running}")
    console.print(f"  pid      {status.pid}")
    console.print(f"  version  {status.version}")
    console.print(f"  console  {status.console_url}")


@db_app.command("wipe")
def db_wipe(
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation prompt.")] = False,
) -> None:
    """Stop the local cluster and delete its data directory. Destructive."""
    from quorum import localdb

    if not yes:
        typer.confirm("Delete the entire local CockroachDB store?", abort=True)
    localdb.wipe()
    console.print("[yellow]local store deleted[/yellow]")


@app.command()
def migrate(
    show: Annotated[
        bool, typer.Option("--status", help="List migrations without applying.")
    ] = False,
) -> None:
    """Apply pending schema migrations."""
    from quorum import migrations

    if show:
        applied = migrations.applied_versions()
        table = Table(title="migrations")
        table.add_column("version", justify="right")
        table.add_column("name")
        table.add_column("applied")
        table.add_column("transactional")
        for migration in migrations.discover():
            table.add_row(
                str(migration.version),
                migration.name,
                "yes" if migration.version in applied else "[yellow]pending[/yellow]",
                "yes" if migration.transactional else "no",
            )
        console.print(table)
        return

    applied = migrations.migrate()
    if not applied:
        console.print("schema up to date")
    for migration in applied:
        console.print(f"[green]applied[/green] {migration.label}")


@app.command()
def reset(
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation prompt.")] = False,
) -> None:
    """Drop the Quorum database and re-apply every migration. Destructive."""
    from quorum import migrations
    from quorum.db import close_pool

    if not yes:
        typer.confirm("Drop the quorum database and all workspaces?", abort=True)
    close_pool()
    migrations.reset()
    close_pool()
    applied = migrations.migrate()
    console.print(f"[green]reset complete[/green] ({len(applied)} migrations applied)")


@app.command()
def decompose(
    task: Annotated[Path, typer.Argument(help="Task spec JSON file.")] = DEFAULT_TASK,
    show_units: Annotated[bool, typer.Option("--units", help="List every work unit.")] = False,
) -> None:
    """Decompose a task spec into work units without writing anything."""
    from quorum.workspace import preview

    spec = _load_task(task)
    result = preview(spec)

    console.print(f"[bold]{spec.get('name', task.stem)}[/bold]")
    console.print(f"  units             {len(result.units)}")
    console.print(f"  dependency edges  {len(result.deps)}")
    console.print(f"  decision scopes   {', '.join(result.scopes())}")

    scope_counts = Counter(scope for unit in result.units for scope in unit.scopes)
    scope_table = Table(title="decision scopes (semantic conflict surface)")
    scope_table.add_column("scope")
    scope_table.add_column("units", justify="right")
    for scope, count in scope_counts.most_common():
        scope_table.add_row(scope, str(count))
    console.print(scope_table)

    if show_units:
        unit_table = Table(title="work units")
        unit_table.add_column("target")
        unit_table.add_column("loc", justify="right")
        unit_table.add_column("hits", justify="right")
        unit_table.add_column("scopes")
        for unit in result.units:
            unit_table.add_row(
                unit.target,
                str(unit.spec.get("loc", "")),
                str(unit.spec.get("signal_hits", "")),
                ",".join(unit.scopes),
            )
        console.print(unit_table)


@app.command()
def seed(
    task: Annotated[Path, typer.Argument(help="Task spec JSON file.")] = DEFAULT_TASK,
    name: Annotated[str | None, typer.Option("--name", help="Workspace name.")] = None,
    mode: Annotated[str, typer.Option("--mode", help="safe or naive.")] = "safe",
    migrate_first: Annotated[
        bool, typer.Option("--migrate/--no-migrate", help="Apply migrations first.")
    ] = True,
) -> None:
    """Seed a workspace from a task spec: decompose, then commit atomically."""
    from quorum import migrations
    from quorum.workspace import seed_workspace

    if mode not in {"safe", "naive"}:
        err_console.print(f"[red]invalid mode {mode!r}: expected safe or naive[/red]")
        raise typer.Exit(2)

    spec = _load_task(task)
    if migrate_first:
        migrations.migrate()

    result = seed_workspace(
        name=name or str(spec.get("name", task.stem)),
        task_spec=spec,
        mode=mode,  # type: ignore[arg-type]
    )

    console.print(f"[green]workspace seeded[/green] {result.workspace_id}")
    console.print(f"  name    {result.name}")
    console.print(f"  mode    {result.mode}")
    console.print(f"  units   {result.unit_count}")
    console.print(f"  deps    {result.dep_count}")
    console.print(f"  scopes  {', '.join(result.scopes)}")


@app.command()
def workspaces() -> None:
    """List every seeded workspace."""
    from quorum.workspace import list_workspaces

    rows = list_workspaces()
    if not rows:
        console.print("no workspaces yet -- run [bold]quorum seed[/bold]")
        return

    table = Table(title="workspaces")
    table.add_column("id")
    table.add_column("name")
    table.add_column("mode")
    table.add_column("units", justify="right")
    table.add_column("done", justify="right")
    table.add_column("pending", justify="right")
    table.add_column("created")
    for row in rows:
        table.add_row(
            str(row["id"]),
            str(row["name"]),
            str(row["mode"]),
            str(row["units"]),
            str(row["done"]),
            str(row["pending"]),
            row["created_at"].strftime("%Y-%m-%d %H:%M:%S"),
        )
    console.print(table)


@app.command()
def status(
    workspace: Annotated[str, typer.Argument(help="Workspace id or name.")],
) -> None:
    """Show unit, decision, and conflict counts for one workspace."""
    from quorum.workspace import resolve_workspace, workspace_summary

    try:
        found = resolve_workspace(workspace)
    except LookupError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    summary = workspace_summary(found["id"])
    console.print(f"[bold]{found['name']}[/bold]  {found['id']}")
    console.print(f"  mode            {found['mode']}")
    console.print(f"  agents running  {summary['agents_running']}")
    console.print(f"  dependency edges {summary['deps']}")

    table = Table(title="work units")
    table.add_column("status")
    table.add_column("count", justify="right")
    for unit_status in ("pending", "claimed", "done", "stale", "failed"):
        table.add_row(unit_status, str(summary["units"].get(unit_status, 0)))
    console.print(table)

    console.print(f"  decisions  {summary['decisions'] or '{}'}")
    console.print(f"  conflicts  {summary['conflicts'] or '{}'}")


def _load_task(path: Path) -> dict[str, Any]:
    """Read a task spec, resolving its `repo` relative to the spec file."""
    if not path.exists():
        err_console.print(f"[red]task spec not found: {path}[/red]")
        raise typer.Exit(2)

    spec: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    repo = spec.get("repo")
    if repo is not None:
        repo_path = Path(str(repo))
        if not repo_path.is_absolute():
            repo_path = (get_settings().repo_root / repo_path).resolve()
        spec["repo"] = str(repo_path)
    return spec


def _redact(url: str) -> str:
    """Hide any password embedded in a connection URL before printing it."""
    if "@" not in url or "://" not in url:
        return url
    scheme, _, rest = url.partition("://")
    credentials, _, host = rest.rpartition("@")
    if ":" in credentials:
        user, _, _password = credentials.partition(":")
        credentials = f"{user}:***"
    return f"{scheme}://{credentials}@{host}"


def main() -> None:
    try:
        app()
    finally:
        from quorum.db import close_pool

        close_pool()


if __name__ == "__main__":
    sys.exit(main())
