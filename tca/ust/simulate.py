"""Run the Module 1 simulator and write its tables.

``run`` is the library entry point : it takes the configuration, a seed and
:class:`Overrides` and returns the simulator tables as DataFrames without touching disk.
``write`` stores them under ``data/derived``.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from tca.core import bondmath, storage
from tca.core.calendar import NY
from tca.core.profile import is_ci
from tca.ust import dealers as dl
from tca.ust import engine, flow, market
from tca.ust.config import SimConfig
from tca.ust.inputs import Inputs, load_inputs

PART_ROWS = 120_000
_PRICE_DECIMALS = 6
_YIELD_DECIMALS = 8
_BAR_STEP_S = 60

TABLES = (
    "ust_dealers",
    "ust_orders",
    "ust_rfq",
    "ust_rfq_quotes",
    "ust_hedges",
    "ust_rfq_markouts",
    "ust_bars_1m",
)


@dataclass(frozen=True)
class Overrides:
    """Run-time changes to the configured simulator, all optional.

    ``leakage_scale`` multiplies the impact of every hedge clip; ``competition_widening_scale``
    multiplies the dealers' widening per extra dealer; ``n_dealers`` forces the number of
    dealers asked for every RFQ (and switches single-dealer axe trades off); ``rfqs`` replaces the
    generated order flow with an explicit list of RFQs, a frame with columns ``ts_send_utc``
    (aware), ``cusip``, ``side``, ``qty_mm`` and optionally ``n_dealers`` and ``is_axe``;
    ``days`` restricts the simulated business days.
    """

    leakage_scale: float | None = None
    competition_widening_scale: float | None = None
    n_dealers: int | None = None
    rfqs: pd.DataFrame | None = None
    days: Sequence[date] | None = None


def simulation_days(cfg: SimConfig, inputs: Inputs, overrides: Overrides) -> list[date]:
    """Business days to simulate: the override, the explicit RFQ days, the CI subset or all."""
    if overrides.days is not None:
        return sorted(overrides.days)
    if overrides.rfqs is not None:
        stamps = pd.to_datetime(overrides.rfqs["ts_send_utc"], utc=True).dt.tz_convert(NY)
        return sorted(set(stamps.dt.date))
    days = inputs.business_days(cfg.period.start, cfg.period.end)
    return days[: cfg.ci_days] if is_ci() else days


def _plan(ctx, mkt, dealers, d, seed, overrides) -> flow.DayPlan:
    if overrides.rfqs is None:
        return flow.plan_day(ctx.cfg, ctx.inputs, mkt, ctx.grid, d, seed, len(dealers))
    stamps = pd.to_datetime(overrides.rfqs["ts_send_utc"], utc=True)
    today = overrides.rfqs[(stamps.dt.tz_convert(NY).dt.date == d).to_numpy()]
    return flow.plan_explicit(today.reset_index(drop=True), mkt, ctx.grid, d, ctx.cfg, len(dealers))


def run(
    cfg: SimConfig, seed: int, overrides: Overrides | None = None, inputs: Inputs | None = None
) -> dict[str, pd.DataFrame]:
    """Simulate the configured period (3 days under ``TCA_PROFILE=ci``) and return the tables."""
    overrides = overrides or Overrides()
    inputs = inputs if inputs is not None else load_inputs()
    ctx = market.Context.build(cfg, inputs)
    dealers = dl.draw_dealers(cfg, seed)
    knobs = engine.Knobs(
        leakage_scale=cfg.leakage_scale
        if overrides.leakage_scale is None
        else overrides.leakage_scale,
        widening_scale=(
            cfg.competition_widening_scale
            if overrides.competition_widening_scale is None
            else overrides.competition_widening_scale
        ),
        forced_n_dealers=overrides.n_dealers,
    )
    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in TABLES}
    order_offset = rfq_offset = 0
    for d in simulation_days(cfg, inputs, overrides):
        mkt = market.build_day(ctx, d, seed)
        n_minutes = ctx.grid.n_seconds // _BAR_STEP_S
        dday = dl.draw_day(cfg, dealers, len(mkt.securities), n_minutes, d, seed)
        plan = _plan(ctx, mkt, dealers, d, seed, overrides)
        res = engine.run_day(cfg, mkt, dealers, dday, plan, ctx.grid, seed, knobs)
        for name, frame in _globalise(res, order_offset, rfq_offset).items():
            parts[name].append(frame)
        parts["ust_bars_1m"].append(_bars(mkt, res.ledgers, ctx.grid))
        order_offset += len(res.orders)
        rfq_offset += len(res.rfq)
    tables = {
        name: pd.concat(frames, ignore_index=True) for name, frames in parts.items() if frames
    }
    tables["ust_dealers"] = dealers.table
    return {name: _round(tables[name]) for name in TABLES}


def _globalise(
    res: engine.DayResult, order_offset: int, rfq_offset: int
) -> dict[str, pd.DataFrame]:
    """Replace the day-local ids by run-wide ids and name the key columns."""
    orders = res.orders.copy()
    orders["order_id"] = orders.pop("order") + order_offset + 1
    lead = orders["list_id"].to_numpy()
    orders["list_id"] = pd.array(np.where(lead >= 0, lead + order_offset + 1, -1), dtype="Int64")
    orders["list_id"] = orders["list_id"].mask(orders["list_id"] < 0)
    rfq = res.rfq.copy()
    rfq["rfq_id"] = rfq.pop("rfq") + rfq_offset + 1
    rfq["order_id"] = rfq.pop("order") + order_offset + 1
    quotes, hedges, marks = res.quotes.copy(), res.hedges.copy(), res.markouts.copy()
    for frame in (quotes, hedges, marks):
        frame["rfq_id"] = frame.pop("rfq") + rfq_offset + 1
    return {
        "ust_orders": _lead(orders, "order_id"),
        "ust_rfq": _lead(rfq, "rfq_id", "order_id"),
        "ust_rfq_quotes": _lead(quotes, "rfq_id"),
        "ust_hedges": _lead(hedges, "rfq_id"),
        "ust_rfq_markouts": _lead(marks, "rfq_id"),
    }


def _lead(frame: pd.DataFrame, *cols: str) -> pd.DataFrame:
    """Move ``cols`` to the front of ``frame``."""
    return frame[[*cols, *[c for c in frame.columns if c not in cols]]]


def _bars(mkt: market.DayMarket, ledgers, grid) -> pd.DataFrame:
    """One-minute bars of the effective mid, composite touch, depth and impact state."""
    t = np.arange(0, grid.n_seconds, _BAR_STEP_S, dtype=float)
    idx = grid.index(t)
    frames = []
    for r, sec in enumerate(mkt.securities):
        shift = ledgers[r].value(t)
        mid = mkt.base_px[r, idx] + shift
        hs = mkt.half_spread_32nds[r, idx]
        frames.append(
            pd.DataFrame(
                {
                    "cusip": sec.cusip,
                    "ts_utc": grid.to_utc(mkt.day, t),
                    "mid_yield": _yield(sec, mid),
                    "mid_px": mid,
                    "bid_px": mid - hs / 32.0,
                    "ask_px": mid + hs / 32.0,
                    "half_spread_32nds": hs,
                    "depth_mm": mkt.depth_mm[r],
                    "impact_state_32nds": shift * 32.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _yield(sec: market.SecurityDay, px: np.ndarray) -> np.ndarray:
    return bondmath.yield_from_price(sec.coupon, sec.maturity, sec.settle, px)


def _round(df: pd.DataFrame) -> pd.DataFrame:
    """Round prices to 1e-6 and yields to 1e-8 so the parquet files stay small."""
    out = df.copy()
    for col in out.select_dtypes(include="float").columns:
        decimals = _YIELD_DECIMALS if "yld" in col or "yield" in col else _PRICE_DECIMALS
        out[col] = out[col].round(decimals)
    return out


def write(tables: dict[str, pd.DataFrame], derived: Path | None = None) -> list[Path]:
    """Write the tables under ``derived``; tables over ``PART_ROWS`` rows go into part files."""
    root = derived if derived is not None else storage.data_dir() / "derived"
    paths = []
    for name, frame in tables.items():
        single, folder = root / f"{name}.parquet", root / name
        single.unlink(missing_ok=True)
        shutil.rmtree(folder, ignore_errors=True)
        if len(frame) > PART_ROWS:
            paths.append(storage.write_table_parts(frame, name, PART_ROWS, root))
        else:
            paths.append(storage.write_table(frame, name, root))
    return paths
