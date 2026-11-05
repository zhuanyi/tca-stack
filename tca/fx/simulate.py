"""Run the order generator and the replay engine over a sample of days: the library entry point."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from tca.core.conventions import pip_size
from tca.fx import lps as lp_module
from tca.fx import ticks
from tca.fx.config import FxConfig, check_broker_sets, load_brokers
from tca.fx.market import Market
from tca.fx.model import orders_from_frame
from tca.fx.orders import arrival_seconds, generate_orders, simulation_days
from tca.fx.passive import calibrate_lambda_fill, scan_passive
from tca.fx.pretrade import estimate_orders
from tca.fx.replay import Ids, ReplayEnv, replay_parent
from tca.fx.strategies import STRATEGIES, Strategy
from tca.fx.ticks import US_PER_S, TickSource

_NO_BROKER_STYLES = ("RISK_TRANSFER", "FIX_ORDER")


@dataclass(frozen=True)
class Overrides:
    """Ways to steer a run from code.

    ``orders`` replaces the generated flow (same columns as ``fx_orders``); ``broker`` forces that
    broker on every order that goes through the wheel; ``days`` restricts the simulated days;
    ``lambda_fill`` fixes the passive hazard per pair instead of calibrating it; ``strategies``
    adds or replaces execution strategies by style name.
    """

    orders: pd.DataFrame | None = None
    broker: str | None = None
    days: list[date] | None = None
    max_days: int | None = None
    lambda_fill: dict[str, float] | None = None
    strategies: dict[str, Strategy] = field(default_factory=dict)


@dataclass
class SimResult:
    """Tables produced by a run, with timestamps as timezone-aware UTC."""

    lps: pd.DataFrame
    orders: pd.DataFrame
    children: pd.DataFrame
    attempts: pd.DataFrame
    fills: pd.DataFrame
    markouts: pd.DataFrame
    lambda_fill: dict[str, float]

    def tables(self) -> dict[str, pd.DataFrame]:
        """The result as ``{table name: frame}`` using the documented table names."""
        return {
            "fx_lps": self.lps,
            "fx_orders": self.orders,
            "fx_children": self.children,
            "fx_fill_attempts": self.attempts,
            "fx_fills": self.fills,
            "fx_markouts": self.markouts,
        }


def calibrate_passive(
    cfg: FxConfig, source: TickSource, days: list[date], seed: int
) -> dict[str, float]:
    """Hazard per second ``lambda_fill`` of each pair that gives the target passive fill share.

    Postings are drawn at the arrival-time mixture on the first ``passive_calibration_days`` days,
    both sides, on the tape with no impact; a posting fills if it crosses or by the hazard while it
    sits at the limit.
    """
    sim = cfg.simulation
    out: dict[str, float] = {}
    timeout_us = int(cfg.passive.timeout_s * US_PER_S)
    latency_us = int(cfg.latency.decision_latency_ms * 1000)
    for p_idx, pair in enumerate(cfg.pairs.order):
        rng = np.random.default_rng([seed, 3, p_idx])
        exposures, crossed = [], []
        for day in days[: sim.passive_calibration_days]:
            tape = source.load(pair, day)
            if tape is None:
                continue
            n = max(1, sim.passive_calibration_samples // max(1, sim.passive_calibration_days))
            starts = ticks.day_start_us(day) + np.rint(
                arrival_seconds(cfg, rng, n) * US_PER_S
            ).astype("int64")
            for start, side in zip(starts, rng.choice([-1, 1], size=n), strict=True):
                i = int(tape.index_at(start))
                limit = tape.bid[i] if side > 0 else tape.ask[i]
                scan = scan_passive(
                    tape,
                    int(side),
                    float(limit),
                    int(start) + latency_us,
                    int(start) + latency_us + timeout_us,
                    lambda t: np.zeros(len(t)),
                    tol=pip_size(pair) / 20.0,
                )
                exposures.append(scan.exposure_s)
                crossed.append(scan.cross_us is not None)
        out[pair] = calibrate_lambda_fill(
            np.asarray(exposures), np.asarray(crossed), cfg.passive.target_fill_rate
        )
    return out


def run(
    cfg: FxConfig,
    seed: int,
    source: TickSource,
    market: Market,
    overrides: Overrides | None = None,
) -> SimResult:
    """Generate (or take) parent orders and replay them against the tape.

    Args:
        cfg: simulator configuration.
        seed: master seed; every random stream derives from it, so equal inputs give equal tables.
        source: tick provider.
        market: volume proxy, spreads and trailing volatility built from the bars.
        overrides: optional steering, see :class:`Overrides`.

    Returns:
        :class:`SimResult` with the six simulated tables.
    """
    ov = overrides or Overrides()
    strategies = {**STRATEGIES, **ov.strategies}
    lps = lp_module.generate_lps(cfg, seed)
    available = source.days(cfg.pairs.order[0])
    days = ov.days or simulation_days(cfg, available, ov.max_days)
    if ov.orders is not None:
        orders = ov.orders.copy()
        days = sorted({ts.date() for ts in orders["ts_arrival_utc"]})
    else:
        orders = generate_orders(cfg, seed, days, market)
    if ov.broker is not None:
        wheel = ~orders["style"].isin(_NO_BROKER_STYLES)
        orders["broker"] = orders["broker"].astype(object).where(~wheel, ov.broker)
    if "pretrade_est_bp" not in orders:
        orders["pretrade_est_bp"] = estimate_orders(orders, market, cfg)

    lambda_fill = ov.lambda_fill or calibrate_passive(cfg, source, days, seed)
    book = load_brokers()
    check_broker_sets(cfg, book)
    env = ReplayEnv(cfg, market, book, lps, lambda_fill, Ids(), strategies)
    parents = orders_from_frame(orders)

    rows: dict[str, list[dict]] = {"children": [], "attempts": [], "fills": [], "markouts": []}
    summaries = []
    for day in days:
        tapes = {pair: source.load(pair, day) for pair in cfg.pairs.order}
        for order in (p for p in parents if p.day == day):
            tape = tapes[order.pair]
            if tape is None:
                continue
            out = replay_parent(order, tape, env, np.random.default_rng([seed, 4, order.order_id]))
            for name in rows:
                rows[name].extend(getattr(out, name))
            summaries.append(out.summary)
    return _assemble(lps, orders, rows, summaries, lambda_fill)


_TS_COLUMNS = {
    "children": ["ts_decision_us", "tick_us"],
    "attempts": ["ts_sent_us", "tick_us"],
    "fills": ["ts_fill_us", "px_tick_us"],
}


def _frame(rows: list[dict], timestamps: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for col in timestamps:
        frame[col.removesuffix("_us") + "_utc"] = ticks.from_us(frame.pop(col).to_numpy())
    return frame


def _assemble(
    lps: pd.DataFrame,
    orders: pd.DataFrame,
    rows: dict[str, list[dict]],
    summaries: list[dict],
    lambda_fill: dict[str, float],
) -> SimResult:
    extra = _frame(summaries, ["first_decision_us", "ts_end_us"])
    full_orders = orders.merge(extra, on="order_id", how="left")
    return SimResult(
        lps=lps,
        orders=full_orders,
        children=_frame(rows["children"], _TS_COLUMNS["children"]),
        attempts=_frame(rows["attempts"], _TS_COLUMNS["attempts"]),
        fills=_frame(rows["fills"], _TS_COLUMNS["fills"]),
        markouts=pd.DataFrame(rows["markouts"]),
        lambda_fill=lambda_fill,
    )
