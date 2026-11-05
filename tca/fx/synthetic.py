"""Synthetic tick generator used by the tests and the ``ci`` profile.

A Brownian mid with a time-of-day spread and tick-rate shape taken from ``config/fx_sim.yaml``
(the ``synthetic`` section). The shape is a configuration default, not the measured liquidity
profile: no real profile exists when the fixture is generated, and the fixture is not regenerated
when one does.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from tca.core.conventions import pip_size
from tca.fx import ticks
from tca.fx.config import FxConfig
from tca.fx.ticks import US_PER_S

_SECONDS_PER_YEAR = 365.0 * 86_400.0


def _tick_times(rng: np.random.Generator, weights: np.ndarray, n: int, start_us: int) -> np.ndarray:
    """Sorted, distinct tick times of a day, about ``n``, with hourly intensity from ``weights``."""
    per_hour = rng.multinomial(n, weights / weights.sum())
    hours = np.repeat(np.arange(24), per_hour)
    offsets = rng.integers(0, 3600 * US_PER_S, size=n)
    return np.unique(start_us + hours * 3600 * US_PER_S + offsets)


def synthetic_day(cfg: FxConfig, pair: str, day: date, level: float, seed: int) -> pd.DataFrame:
    """One pair-day of synthetic quotes.

    Args:
        cfg: configuration holding the ``synthetic`` section.
        pair: pair symbol; JPY pairs quote on a 0.001 grid, others on 0.00001.
        day: the UTC day.
        level: mid price at the start of the day, in quote units.
        seed: seed of the day's generator.

    Returns:
        Frame with ``ts_utc`` (UTC), ``bid``, ``ask`` (quote units, ``ask > bid``) and the
        indicative volumes ``ask_vol`` and ``bid_vol``.
    """
    syn = cfg.synthetic
    rng = np.random.default_rng(seed)
    start = ticks.day_start_us(day)
    t = _tick_times(rng, np.asarray(syn.hourly_tick_weight), syn.ticks_per_day, start)
    dt_years = np.diff(t, prepend=start) / US_PER_S / _SECONDS_PER_YEAR
    shocks = rng.standard_normal(len(t)) * syn.annual_vol[pair] * np.sqrt(dt_years)
    mid = level * np.exp(np.cumsum(shocks))

    hour = ((t - start) // (3600 * US_PER_S)).astype(int)
    half = (
        0.5
        * syn.base_spread_pips[pair]
        * np.asarray(syn.hourly_spread_multiplier)[hour]
        * rng.lognormal(0.0, 0.25, len(t))
        * pip_size(pair)
    )
    grid = pip_size(pair) / 10.0
    bid = np.floor((mid - half) / grid) * grid
    ask = np.maximum(np.ceil((mid + half) / grid) * grid, bid + grid)
    return pd.DataFrame(
        {
            "ts_utc": ticks.from_us(t),
            "bid": np.round(bid, 5),
            "ask": np.round(ask, 5),
            "ask_vol": np.round(rng.uniform(0.5, 4.0, len(t)), 1),
            "bid_vol": np.round(rng.uniform(0.5, 4.0, len(t)), 1),
        }
    )


def make_synthetic_ticks(
    cfg: FxConfig, days: int, start: date | None = None, seed: int | None = None
) -> pd.DataFrame:
    """Synthetic ticks for every configured pair on ``days`` consecutive weekdays.

    Args:
        cfg: configuration holding the pairs and the ``synthetic`` section.
        days: number of weekdays to generate.
        start: first day; defaults to the start of the configured period (a weekday).
        seed: base seed; defaults to the simulation seed.

    Returns:
        One frame for all pairs with a leading ``pair`` column; prices carry on from day to day.
    """
    first = start or cfg.period.start
    base_seed = cfg.simulation.seed if seed is None else seed
    weekdays: list[date] = []
    cursor = first
    while len(weekdays) < days:
        if cursor.weekday() < 5:
            weekdays.append(cursor)
        cursor += timedelta(days=1)
    frames = []
    for p_idx, pair in enumerate(cfg.pairs.order):
        level = cfg.synthetic.start_level[pair]
        for d_idx, day in enumerate(weekdays):
            frame = synthetic_day(cfg, pair, day, level, seed=base_seed + 1000 * p_idx + d_idx)
            level = float((frame["bid"].iloc[-1] + frame["ask"].iloc[-1]) / 2.0)
            frames.append(frame.assign(pair=pair))
    out = pd.concat(frames, ignore_index=True)
    return out[["pair", "ts_utc", "bid", "ask", "ask_vol", "bid_vol"]]
