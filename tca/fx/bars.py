"""One-minute bars from ticks, and daily realised volatility from the bars."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tca.core.conventions import pip_size
from tca.fx import ticks
from tca.fx.ticks import US_PER_S, TickDay

US_PER_MINUTE = 60 * US_PER_S
BAR_COLUMNS = [
    "pair",
    "ts_utc",
    "open",
    "high",
    "low",
    "close",
    "spread_tw_pips",
    "spread_tw_bp",
    "tick_count",
    "last_bid",
    "last_ask",
]


def _time_weighted_spread(day: TickDay, minutes: np.ndarray) -> np.ndarray:
    """Time-weighted quoted spread (quote units) of each minute in ``minutes`` (epoch minutes).

    A quote persists until the next tick, so a minute is weighted by how long each spread was
    on the screen inside it, including a spread carried in from an earlier minute. Time before
    the first tick of the day is not weighted.
    """
    t = day.t_us
    spread = day.ask - day.bid
    boundaries = np.arange(minutes[0] + 1, minutes[-1] + 1, dtype=np.int64) * US_PER_MINUTE
    timeline = np.sort(np.concatenate((t, boundaries)), kind="stable")
    owner = np.searchsorted(t, timeline[:-1], side="right") - 1
    duration = np.diff(timeline).astype(float)
    slot = (timeline[:-1] // US_PER_MINUTE - minutes[0]).astype(int)
    size = int(minutes[-1] - minutes[0]) + 1
    weighted = np.bincount(slot, weights=duration * spread[owner], minlength=size)
    total = np.bincount(slot, weights=duration, minlength=size)
    pick = (minutes - minutes[0]).astype(int)
    weighted, total = weighted[pick], total[pick]
    first_tick = spread[np.searchsorted(t, minutes * US_PER_MINUTE, side="left")]
    return np.where(total > 0, weighted / np.where(total > 0, total, 1.0), first_tick)


def bars_from_ticks(day: TickDay) -> pd.DataFrame:
    """One-minute bars of one pair-day; only minutes with at least one tick appear.

    Prices are mids; ``spread_tw_pips`` and ``spread_tw_bp`` are the time-weighted quoted spread
    in pips and in bp of the bar's close; ``tick_count`` counts the ticks in the minute.
    """
    t = day.t_us
    mid = day.mid
    minute = t // US_PER_MINUTE
    minutes, first, counts = np.unique(minute, return_index=True, return_counts=True)
    last = first + counts - 1
    close = mid[last]
    spread = _time_weighted_spread(day, minutes)
    return pd.DataFrame(
        {
            "pair": day.pair,
            "ts_utc": ticks.from_us(minutes * US_PER_MINUTE),
            "open": mid[first],
            "high": np.maximum.reduceat(mid, first),
            "low": np.minimum.reduceat(mid, first),
            "close": close,
            "spread_tw_pips": spread / pip_size(day.pair),
            "spread_tw_bp": spread / close * 1e4,
            "tick_count": counts.astype("int64"),
            "last_bid": day.bid[last],
            "last_ask": day.ask[last],
        }
    )[BAR_COLUMNS]


def round_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Round prices to 1e-6, pips to 1e-4 and bp to 1e-4 before storage."""
    out = bars.copy()
    for col in ("open", "high", "low", "close", "last_bid", "last_ask"):
        out[col] = out[col].round(6)
    for col in ("spread_tw_pips", "spread_tw_bp"):
        out[col] = out[col].round(4)
    return out


def daily_vol(bars: pd.DataFrame) -> pd.DataFrame:
    """Realised volatility of each pair-day from consecutive one-minute closes.

    Returns columns ``pair``, ``day`` and ``vol`` where ``vol = sqrt(sum r^2)`` over the day's
    log returns ``r`` between adjacent minutes with a bar (a fraction of price, not annualised).
    """
    frame = bars[["pair", "ts_utc", "close"]].copy()
    frame["day"] = frame["ts_utc"].dt.date
    frame["ret"] = frame.groupby(["pair", "day"], sort=True)["close"].transform(
        lambda c: np.log(c).diff()
    )
    frame["sq"] = frame["ret"] ** 2
    out = frame.groupby(["pair", "day"], sort=True)["sq"].sum().pow(0.5).rename("vol")
    return out.reset_index()
