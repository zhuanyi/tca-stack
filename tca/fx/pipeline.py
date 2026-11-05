"""Glue between the stages: ticks, bars, profile, simulation and TCA tables, stored on disk."""

from __future__ import annotations

import pandas as pd

from tca.core import storage
from tca.core.profile import is_ci
from tca.fx import analysis, liquidity, paths
from tca.fx import bars as bars_module
from tca.fx.config import FxConfig
from tca.fx.market import Market
from tca.fx.simulate import Overrides, SimResult, run
from tca.fx.ticks import TickSource, default_source

# Tables above this many rows are split into part files so no parquet file nears the size limit.
ROWS_PER_PART = 100_000
PRICE_DECIMALS = 6
BP_DECIMALS = 4


def round_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Round floats before storage: 1e-4 for columns in bp or pips, 1e-6 otherwise (prices)."""
    out = frame.copy()
    for col in out.select_dtypes(include="float").columns:
        decimals = BP_DECIMALS if ("bp" in col or "pips" in col) else PRICE_DECIMALS
        out[col] = out[col].round(decimals)
    return out


def store(frame: pd.DataFrame, name: str) -> None:
    """Write ``frame`` to the derived folder of the active profile; part files when it is long."""
    derived = paths.derived_dir()
    frame = round_table(frame)
    if len(frame) > ROWS_PER_PART:
        storage.write_table_parts(frame, name, ROWS_PER_PART, derived)
    else:
        storage.write_table(frame, name, derived)


def load(name: str) -> pd.DataFrame:
    """Read a stored table of the active profile."""
    return storage.read_table(name, paths.derived_dir())


def build_bars(cfg: FxConfig, source: TickSource) -> pd.DataFrame:
    """One-minute bars for every configured pair and every day the source holds."""
    pieces = []
    for pair in cfg.pairs.order:
        for day in source.days(pair):
            tape = source.load(pair, day)
            if tape is not None:
                pieces.append(bars_module.bars_from_ticks(tape))
    if not pieces:
        raise FileNotFoundError(
            f"no ticks found for {', '.join(cfg.pairs.order)}; run `tca fx fetch-ticks` first"
        )
    return pd.concat(pieces, ignore_index=True)


def build_market(cfg: FxConfig, bars: pd.DataFrame) -> tuple[Market, pd.DataFrame]:
    """The market context and the liquidity profile it is built from."""
    profile = liquidity.liquidity_profile(bars)
    return Market(cfg, profile, bars_module.daily_vol(bars)), profile


def stage_bars(cfg: FxConfig) -> pd.DataFrame:
    """Ticks to ``fx_bars_1m``."""
    bars = build_bars(cfg, default_source())
    store(bars, "fx_bars_1m")
    return bars


def stage_profile() -> pd.DataFrame:
    """``fx_bars_1m`` to ``fx_liquidity_profile``."""
    profile = liquidity.round_profile(liquidity.liquidity_profile(load("fx_bars_1m")))
    store(profile, "fx_liquidity_profile")
    return profile


def stage_simulate(cfg: FxConfig, seed: int) -> SimResult:
    """Generate orders, replay them against the ticks and store the six simulated tables.

    The ``ci`` profile simulates at most ``simulation.ci_max_days`` days.
    """
    market, _ = build_market(cfg, load("fx_bars_1m"))
    overrides = Overrides(max_days=cfg.simulation.ci_max_days if is_ci() else None)
    result = run(cfg, seed, default_source(), market, overrides)
    for name, frame in result.tables().items():
        store(frame, name)
    return result


def stage_tca() -> dict[str, pd.DataFrame]:
    """Simulated tables to ``fx_tca_fill``, ``fx_tca_parent`` and ``fx_lp_scorecard``."""
    orders, children = load("fx_orders"), load("fx_children")
    attempts, fills = load("fx_fill_attempts"), load("fx_fills")
    tfill = analysis.tca_fill(orders, fills, load("fx_markouts"))
    out = {
        "fx_tca_fill": tfill,
        "fx_tca_parent": analysis.tca_parent(orders, fills, attempts, children, tfill),
        "fx_lp_scorecard": analysis.lp_scorecard(orders, children, attempts, fills, tfill),
    }
    for name, frame in out.items():
        store(frame, name)
    return out
