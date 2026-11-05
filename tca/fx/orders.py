"""Simulated client flow: parent orders per weekday, before any replay."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from tca.core.calendar import business_days, wmr_window_utc
from tca.fx import ticks
from tca.fx.config import FxConfig, hhmm_to_seconds
from tca.fx.market import Market
from tca.fx.model import ORDER_COLUMNS, URGENCIES, Order
from tca.fx.strategies import STRATEGIES, StrategyContext
from tca.fx.ticks import US_PER_S

_STYLE_BY_URGENCY = {"LOW": "TWAP", "MEDIUM": "POV", "HIGH": "IS", "FIX": "FIX_ORDER"}


def simulation_days(
    cfg: FxConfig, available: list[date], max_days: int | None = None
) -> list[date]:
    """Business days of the configured period that have ticks, at most ``max_days`` of them."""
    wanted = set(business_days(cfg.period.start, cfg.period.end)) & set(available)
    days = sorted(wanted)
    return days if max_days is None else days[:max_days]


def arrival_seconds(cfg: FxConfig, rng: np.random.Generator, n: int) -> np.ndarray:
    """Arrival second of day from the configured mixture of UTC windows."""
    windows = cfg.flow.arrival_time_utc
    weights = np.array([w.weight for w in windows])
    pick = rng.choice(len(windows), size=n, p=weights / weights.sum())
    lows = np.array([hhmm_to_seconds(w.start) for w in windows])[pick]
    highs = np.array([hhmm_to_seconds(w.end) for w in windows])[pick]
    return np.round(rng.uniform(lows, highs), 3)


def _notional(cfg: FxConfig, rng: np.random.Generator, n: int) -> np.ndarray:
    """Lognormal notionals clipped to the configured range, a share snapped to round sizes."""
    spec = cfg.flow.notional_usd
    raw = np.clip(rng.lognormal(np.log(spec.median), spec.sigma, n), spec.min, spec.max)
    snap = rng.random(n) < spec.round_snap_probability
    grid = np.asarray(spec.round_snap_values_mm) * 1e6
    nearest = grid[np.abs(np.log(raw)[:, None] - np.log(grid)[None, :]).argmin(axis=1)]
    return np.rint(np.where(snap, nearest, raw))


def _assign_brokers(
    cfg: FxConfig, rng: np.random.Generator, notional: np.ndarray, style: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Broker and assignment mode for each parent; ``None`` for risk transfer and fix orders."""
    spec = cfg.broker_assignment
    n = len(notional)
    modes = list(spec.assignment_mix)
    probs = np.array([spec.assignment_mix[m] for m in modes])
    mode = np.asarray(modes, dtype=object)[rng.choice(len(modes), size=n, p=probs / probs.sum())]
    uniform = np.asarray(spec.brokers, dtype=object)[rng.integers(0, len(spec.brokers), n)]
    large = notional > spec.discretionary_large_threshold_usd
    broker = np.where((mode == "DISCRETIONARY") & large, "A", uniform)
    wheel = ~np.isin(style, ["RISK_TRANSFER", "FIX_ORDER"])
    return np.where(wheel, broker, None), np.where(wheel, mode, None)


def _generate_day(cfg: FxConfig, seed: int, day: date, market: Market) -> pd.DataFrame:
    rng = np.random.default_rng([seed, 1, day.toordinal()])
    flow = cfg.flow
    n = int(rng.poisson(flow.parents_per_weekday.mean))
    pair = np.asarray(cfg.pairs.order, dtype=object)[
        rng.choice(len(cfg.pairs.order), size=n, p=[cfg.pairs.weights[p] for p in cfg.pairs.order])
    ]
    bias = rng.normal(0.0, flow.side.daily_bias_sigma)
    p_buy = float(
        np.clip(
            flow.side.buy_probability_base + bias,
            flow.side.buy_probability_floor,
            flow.side.buy_probability_cap,
        )
    )
    side = np.where(rng.random(n) < p_buy, 1, -1)
    arrival_s = arrival_seconds(cfg, rng, n)
    notional = _notional(cfg, rng, n)
    urgency = np.asarray(URGENCIES, dtype=object)[
        rng.choice(4, size=n, p=[flow.urgency_probabilities[u] for u in URGENCIES])
    ]
    is_rt = rng.random(n) < flow.high_share_risk_transfer
    style = np.array(
        [
            "RISK_TRANSFER" if (u == "HIGH" and rt) else _STYLE_BY_URGENCY[u]
            for u, rt in zip(urgency, is_rt, strict=True)
        ],
        dtype=object,
    )
    broker, assignment = _assign_brokers(cfg, rng, notional, style)
    trader = rng.integers(1, cfg.simulation.n_traders + 1, n)
    lead_lo, lead_hi = cfg.execution_styles.FIX_ORDER.arrival_lead_minutes
    lead_s = rng.uniform(lead_lo * 60.0, lead_hi * 60.0, n)

    midnight = ticks.day_start_us(day)
    arrival_us = midnight + np.rint(arrival_s * US_PER_S).astype("int64")
    fix_start = wmr_window_utc(day)[0]
    fix_start_us = int(fix_start.timestamp()) * US_PER_S
    fix_arrival = fix_start_us - np.rint(lead_s * US_PER_S).astype("int64")
    arrival_us = np.where(style == "FIX_ORDER", fix_arrival, arrival_us)
    order = np.argsort(arrival_us, kind="stable")

    ctx = StrategyContext(cfg, market)
    rows = []
    for i in order:
        draft = Order(
            order_id=0,
            day=day,
            pair=str(pair[i]),
            side=int(side[i]),
            notional_usd=float(notional[i]),
            ts_arrival_us=int(arrival_us[i]),
            urgency=str(urgency[i]),
            style=str(style[i]),
            broker=None if broker[i] is None else str(broker[i]),
            assignment=None if assignment[i] is None else str(assignment[i]),
            trader_id=int(trader[i]),
            planned_duration_s=0.0,
        )
        duration = STRATEGIES[draft.style].planned_duration_s(draft, ctx)
        rows.append({**draft.__dict__, "planned_duration_s": duration})
    return pd.DataFrame(rows)


def generate_orders(cfg: FxConfig, seed: int, days: list[date], market: Market) -> pd.DataFrame:
    """Parent orders for ``days``, ordered by day and arrival time, ``order_id`` from 1.

    Each weekday draws ``Poisson(mean)`` parents across the pairs, with arrival times from the
    configured mixture of UTC windows, lognormal notionals, a daily buy bias, an urgency that
    selects the execution style, and a broker for orders that go through the wheel. Fix orders
    arrive a random lead before the WM/R window. Each day has its own random stream, so one day
    does not depend on how many days precede it in the run.
    """
    frames = [_generate_day(cfg, seed, d, market) for d in days]
    if not frames:
        return pd.DataFrame(columns=ORDER_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["order_id"] = np.arange(1, len(out) + 1)
    out["ts_arrival_utc"] = ticks.from_us(out.pop("ts_arrival_us").to_numpy())
    return out.drop(columns=["day"])[ORDER_COLUMNS]
