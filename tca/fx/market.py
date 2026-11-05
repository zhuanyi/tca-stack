"""Market context for the simulator: volume proxy, half-spreads and trailing volatility."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from tca.fx.config import FxConfig
from tca.fx.liquidity import ALL_WEEKDAYS, BUCKET_SECONDS, BUCKETS_PER_DAY
from tca.fx.ticks import US_PER_DAY, US_PER_S, day_start_us


class Market:
    """Pair-level inputs to order generation, scheduling, impact and the pre-trade estimate.

    Args:
        cfg: simulator configuration (volume proxy and impact window).
        profile: ``fx_liquidity_profile``; the all-weekday rows give the intraday tick-rate shape
            that allocates the daily volume proxy, and the median spread.
        daily_vol: columns ``pair``, ``day``, ``vol`` as returned by :func:`tca.fx.bars.daily_vol`.
    """

    def __init__(self, cfg: FxConfig, profile: pd.DataFrame, daily_vol: pd.DataFrame) -> None:
        self._cfg = cfg
        overall = profile[profile["dow"] == ALL_WEEKDAYS]
        self._share: dict[str, np.ndarray] = {}
        self._half_spread_bp: dict[str, np.ndarray] = {}
        for pair, sub in overall.groupby("pair", sort=True):
            by_bucket = sub.set_index("bucket").reindex(range(BUCKETS_PER_DAY))
            rate = by_bucket["tick_rate_per_s"].fillna(0.0).to_numpy()
            self._share[pair] = rate / rate.sum()
            self._half_spread_bp[pair] = (
                by_bucket["spread_median_bp"].ffill().bfill().to_numpy() / 2.0
            )
        self._vol = {
            pair: sub.sort_values("day")[["day", "vol"]].reset_index(drop=True)
            for pair, sub in daily_vol.groupby("pair", sort=True)
        }

    @staticmethod
    def bucket_at(ts_us: int) -> int:
        """Five-minute bucket of UTC time of day containing ``ts_us``."""
        return int((ts_us % US_PER_DAY) // (BUCKET_SECONDS * US_PER_S))

    def volume_rate_usd_s(self, pair: str, ts_us: int) -> float:
        """Volume proxy in USD per second at ``ts_us``: daily turnover allocated by tick rate."""
        share = self._share[pair][self.bucket_at(ts_us)]
        return self._cfg.daily_volume_usd(pair) * share / BUCKET_SECONDS

    def half_spread_bp(self, pair: str, ts_us: int) -> float:
        """Median quoted half-spread in bp of mid for the bucket containing ``ts_us``."""
        return float(self._half_spread_bp[pair][self.bucket_at(ts_us)])

    def sigma_d(self, pair: str, day: date) -> float:
        """Daily volatility (fraction of price) used for impact on ``day``.

        The mean realised volatility of up to ``sigma_d_window_days`` earlier days; for the first
        day of the sample, which has no history, that day's own realised volatility.
        """
        table = self._vol[pair]
        prior = table[table["day"] < day].tail(self._cfg.impact.sigma_d_window_days)
        if len(prior):
            return float(prior["vol"].mean())
        return float(table.loc[table["day"] == day, "vol"].iloc[0])

    def day_window_us(self, day: date) -> tuple[int, int]:
        """Start and end of ``day`` in epoch microseconds."""
        start = day_start_us(day)
        return start, start + US_PER_DAY
