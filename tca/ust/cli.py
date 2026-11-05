"""``tca ust``: fetch inputs, simulate, run the TCA, draw the figures, print the summary."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from tca.core import storage
from tca.ust import inputs, posttrade, simulate
from tca.ust.config import DEFAULT_CONFIG, REPO_ROOT, load_config

app = typer.Typer(help="Module 1: UST dealer-to-client RFQ TCA.", no_args_is_help=True)

ConfigOption = Annotated[Path, typer.Option(help="Simulator configuration YAML.")]
ProfileOption = Annotated[
    str | None,
    typer.Option("--profile", help="'ci' simulates the first 3 business days; default: env."),
]


def _set_profile(profile: str | None) -> None:
    if profile is None:
        return
    if profile not in ("ci", "full"):
        raise typer.BadParameter("profile must be 'ci' or 'full'")
    os.environ["TCA_PROFILE"] = profile


def _derived() -> Path:
    return storage.data_dir() / "derived"


@app.command("fetch-anchors")
def fetch_anchors(
    start: Annotated[str, typer.Option(help="First day, YYYY-MM-DD.")] = "2025-03-03",
    end: Annotated[str, typer.Option(help="Last day, YYYY-MM-DD.")] = "2025-05-30",
) -> None:
    """Download the public inputs into ``<raw>/ust`` (cached; see scripts/fetch_ust_anchors.py)."""
    script = REPO_ROOT / "scripts" / "fetch_ust_anchors.py"
    code = subprocess.run([sys.executable, str(script), "--start", start, "--end", end]).returncode
    raise typer.Exit(code)


@app.command("data")
def data(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Build the four committed input tables from the raw downloads."""
    cfg = load_config(config)
    tables = inputs.prepare(storage.raw_dir(), cfg)
    for name, frame in tables.items():
        path = storage.write_table(frame, name, _derived())
        typer.echo(f"{name}: {len(frame):,} rows -> {path.name}")


@app.command("simulate")
def simulate_cmd(
    config: ConfigOption = DEFAULT_CONFIG,
    seed: Annotated[int | None, typer.Option(help="Seed (default: from the config).")] = None,
    profile: ProfileOption = None,
) -> None:
    """Simulate the market and the RFQ flow; write the ``ust_*`` simulator tables."""
    _set_profile(profile)
    cfg = load_config(config)
    tables = simulate.run(cfg, cfg.seed if seed is None else seed)
    simulate.write(tables, _derived())
    for name, frame in tables.items():
        typer.echo(f"{name}: {len(frame):,} rows")


@app.command("tca")
def tca_cmd(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Compute per-fill costs, parent shortfall and the dealer scorecard; write ``ust_tca_*``."""
    cfg = load_config(config)
    tables = posttrade.read_tables(posttrade.SIMULATOR_TABLES, _derived())
    out = posttrade.build(cfg, tables)
    posttrade.write(out, _derived())
    for name, frame in out.items():
        typer.echo(f"{name}: {len(frame):,} rows")


@app.command("figures")
def figures_cmd(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Draw the figures under ``docs/figures``."""
    from tca.ust import figures

    for path in figures.draw_all(load_config(config), _derived()):
        typer.echo(str(path))


@app.command("summary")
def summary_cmd(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Print the numbers used in ``docs/ust.md``."""
    from tca.ust import summary

    typer.echo(summary.render(load_config(config), _derived()))
