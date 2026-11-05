"""``tca fx``: ticks to bars, profile, simulation, TCA tables, figures and the printed summary."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from tca.core import storage
from tca.core.profile import is_ci
from tca.fx import figures, paths, pipeline, summary, synthetic
from tca.fx.config import FxConfig, load_config

app = typer.Typer(help="Module 2: spot FX TCA with an algo replay engine.", no_args_is_help=True)

ProfileOption = Annotated[
    str | None, typer.Option("--profile", help="'ci' runs on the synthetic fixture; default: env.")
]
ConfigOption = Annotated[Path | None, typer.Option(help="Path to fx_sim.yaml.")]
TABLES = [
    "fx_orders",
    "fx_children",
    "fx_fill_attempts",
    "fx_fills",
    "fx_markouts",
    "fx_tca_fill",
    "fx_tca_parent",
    "fx_lp_scorecard",
    "fx_liquidity_profile",
]


def _setup(profile: str | None, config: Path | None = None) -> FxConfig:
    if profile is not None:
        if profile not in ("ci", "full"):
            raise typer.BadParameter("profile must be 'ci' or 'full'")
        os.environ["TCA_PROFILE"] = profile
    return load_config(config)


def _echo(label: str, frame_rows: int) -> None:
    typer.echo(f"{label}: {frame_rows:,} rows -> {paths.derived_dir()}")


def _run_fetcher(pairs: str, start: str, end: str, from_dir: Path | None) -> int:
    """Run ``scripts/fetch_fx_ticks.py`` for the given range and return its exit code."""
    cfg = load_config()
    command = [
        sys.executable,
        str(paths.REPO_ROOT / "scripts" / "fetch_fx_ticks.py"),
        "--pairs",
        pairs or ",".join(cfg.pairs.order),
        "--start",
        start or cfg.period.start.isoformat(),
        "--end",
        end or cfg.period.end.isoformat(),
    ]
    if from_dir is not None:
        command += ["--from-dir", str(from_dir)]
    env = {**os.environ, "TCA_DATA_RAW": str(storage.raw_dir())}
    return subprocess.run(command, env=env).returncode


@app.command("fetch-ticks")
def fetch_ticks(
    pairs: Annotated[str, typer.Option(help="Comma-separated pairs.")] = "",
    start: Annotated[str, typer.Option(help="First day, YYYY-MM-DD.")] = "",
    end: Annotated[str, typer.Option(help="Last day, YYYY-MM-DD.")] = "",
    from_dir: Annotated[
        Path | None, typer.Option(help="Read downloaded .bi5 files from here; no network.")
    ] = None,
) -> None:
    """Download Dukascopy ticks into <raw>/fx, resumable per hour (scripts/fetch_fx_ticks.py)."""
    raise typer.Exit(_run_fetcher(pairs, start, end, from_dir))


@app.command("make-synthetic-ticks")
def make_synthetic_ticks(
    days: Annotated[int, typer.Option(help="Number of weekdays.")] = 2,
    out: Annotated[Path, typer.Option(help="Output parquet file.")] = paths.FIXTURE,
    start: Annotated[str, typer.Option(help="First day, default the period start.")] = "",
    config: ConfigOption = None,
) -> None:
    """Write synthetic ticks for the four pairs.

    The time-of-day spread and tick-rate shape comes from the defaults in the ``synthetic`` section
    of the configuration, not from the measured liquidity profile, which does not exist before the
    real ticks are downloaded.
    """
    cfg = load_config(config)
    first = date.fromisoformat(start) if start else None
    frame = synthetic.make_synthetic_ticks(cfg, days, first)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False, compression="zstd", compression_level=9)
    typer.echo(f"{len(frame):,} ticks -> {out}")


@app.command()
def bars(profile: ProfileOption = None, config: ConfigOption = None) -> None:
    """Ticks to one-minute bars (``fx_bars_1m``)."""
    cfg = _setup(profile, config)
    _echo("fx_bars_1m", len(pipeline.stage_bars(cfg)))


@app.command("profile")
def liquidity_profile(profile: ProfileOption = None, config: ConfigOption = None) -> None:
    """Bars to the liquidity profile (``fx_liquidity_profile``)."""
    _setup(profile, config)
    _echo("fx_liquidity_profile", len(pipeline.stage_profile()))


@app.command()
def simulate(
    profile: ProfileOption = None,
    config: ConfigOption = None,
    seed: Annotated[int | None, typer.Option(help="Master seed; default from the config.")] = None,
) -> None:
    """Generate parent orders and replay them against the ticks."""
    cfg = _setup(profile, config)
    result = pipeline.stage_simulate(cfg, cfg.simulation.seed if seed is None else seed)
    for name, frame in result.tables().items():
        _echo(name, len(frame))


@app.command()
def tca(profile: ProfileOption = None, config: ConfigOption = None) -> None:
    """Fill-level and parent-level TCA and the LP scorecard."""
    _setup(profile, config)
    for name, frame in pipeline.stage_tca().items():
        _echo(name, len(frame))


@app.command("figures")
def make_figures(profile: ProfileOption = None, config: ConfigOption = None) -> None:
    """Draw the seven figures."""
    cfg = _setup(profile, config)
    names = figures.make_all(cfg, {name: pipeline.load(name) for name in TABLES})
    typer.echo(f"{len(names)} figures -> {paths.figures_dir()}")


@app.command("summary")
def print_summary(profile: ProfileOption = None, config: ConfigOption = None) -> None:
    """Print the numbers the findings in docs/fx.md are written from."""
    _setup(profile, config)
    typer.echo("\n".join(summary.summary_lines({name: pipeline.load(name) for name in TABLES})))


@app.command()
def data(profile: ProfileOption = None, config: ConfigOption = None) -> None:
    """Make the inputs: download ticks (full profile only), then bars and the liquidity profile."""
    cfg = _setup(profile, config)
    if not is_ci() and (code := _run_fetcher("", "", "", None)) != 0:
        raise typer.Exit(code)
    _echo("fx_bars_1m", len(pipeline.stage_bars(cfg)))
    _echo("fx_liquidity_profile", len(pipeline.stage_profile()))
