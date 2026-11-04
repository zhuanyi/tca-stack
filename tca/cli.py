"""Command-line entry point: ``tca db build`` and the discovered module sub-apps."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Annotated

import typer

from tca.core import storage

_MODULES = ("ust", "fx", "agent")

db_app = typer.Typer(help="Build and inspect the analytical store.", no_args_is_help=True)


@db_app.command("build")
def db_build(
    derived: Annotated[
        Path | None, typer.Option(help="Folder of parquet tables (default: <data>/derived).")
    ] = None,
    db_path: Annotated[
        Path | None, typer.Option(help="DuckDB file to create (default: <data>/tca.duckdb).")
    ] = None,
) -> None:
    """Rebuild the DuckDB file from the parquet tables, one table per file or part folder."""
    counts = storage.build_duckdb(derived, db_path)
    for name, rows in counts.items():
        typer.echo(f"{name}: {rows:,} rows")
    typer.echo(f"{len(counts)} tables built")


def _discover(app: typer.Typer) -> None:
    """Register ``tca.<module>.cli:app`` for every module that ships one."""
    for name in _MODULES:
        target = f"tca.{name}.cli"
        try:
            module = import_module(target)
        except ModuleNotFoundError as exc:
            if exc.name is None or not target.startswith(exc.name):
                raise
            continue
        app.add_typer(module.app, name=name)


def build_app() -> typer.Typer:
    """Assemble the root ``tca`` app with ``db`` and every available module sub-app."""
    app = typer.Typer(
        help="Transaction cost analysis for US Treasuries and spot FX.", no_args_is_help=True
    )
    app.add_typer(db_app, name="db")
    _discover(app)
    return app


app = build_app()
