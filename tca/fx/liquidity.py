"""Liquidity profile by pair and five-minute bucket of UTC time of day, from one-minute bars."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tca.core.calendar import wmr_window_utc

BUCKET_MINUTES = 5
BUCKETS_PER_DAY = 24 * 60 // BUCKET_MINUTES
BUCKET_SECONDS = BUCKET_MINUTES * 60
ALL_WEEKDAYS = -1
MINUTES_PER_YEAR = 260 * 24 * 60

PROFILE_COLUMNS = [
    "pair",
    "dow",
    "bucket",
    "spread_median_pips",
    "spread_p90_pips",
    "spread_median_bp",
    "spread_p90_bp",
    "tick_rate_per_s",
    "realised_vol_ann",
    "n_days",
    "wmr_share",
    "wmr_window",
]


def _bucket_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """Bars with ``day``, ``dow``, ``bucket`` and the squared return to the previous minute."""
    frame = bars.sort_values(["pair", "ts_utc"], kind="stable").copy()
    frame["day"] = frame["ts_utc"].dt.date
    frame["dow"] = frame["ts_utc"].dt.day_of_week
    minute_of_day = frame["ts_utc"].dt.hour * 60 + frame["ts_utc"].dt.minute
    frame["bucket"] = minute_of_day // BUCKET_MINUTES
    gap = frame.groupby(["pair", "day"], sort=False)["ts_utc"].diff().dt.total_seconds()
    ret = frame.groupby(["pair", "day"], sort=False)["close"].transform(lambda c: np.log(c).diff())
    frame["sq_ret"] = np.where(gap == 60.0, ret**2, np.nan)
    return frame


def _aggregate(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(keys, sort=True)
    out = pd.DataFrame(
        {
            "spread_median_pips": grouped["spread_tw_pips"].median(),
            "spread_p90_pips": grouped["spread_tw_pips"].quantile(0.9),
            "spread_median_bp": grouped["spread_tw_bp"].median(),
            "spread_p90_bp": grouped["spread_tw_bp"].quantile(0.9),
            "ticks": grouped["tick_count"].sum(),
            "mean_sq_ret": grouped["sq_ret"].mean(),
            "n_days": grouped["day"].nunique(),
        }
    )
    out["tick_rate_per_s"] = out["ticks"] / (out["n_days"] * BUCKET_SECONDS)
    out["realised_vol_ann"] = np.sqrt(out["mean_sq_ret"] * MINUTES_PER_YEAR)
    return out.drop(columns=["ticks", "mean_sq_ret"]).reset_index()


def _wmr_share(days: pd.Series) -> np.ndarray:
    """Share of ``days`` on which each bucket overlaps the WM/R window (London local time)."""
    hits = np.zeros(BUCKETS_PER_DAY)
    for day in days.unique():
        start, end = wmr_window_utc(day)
        midnight = pd.Timestamp(day, tz="UTC")
        lo = (start - midnight).total_seconds()
        hi = (end - midnight).total_seconds()
        first = int(lo // BUCKET_SECONDS)
        last = int((hi - 1e-9) // BUCKET_SECONDS)
        hits[first : last + 1] += 1
    return hits / max(len(days.unique()), 1)


def liquidity_profile(bars: pd.DataFrame) -> pd.DataFrame:
    """Median and 90th-percentile spread, tick rate and volatility per pair and bucket.

    Args:
        bars: one-minute bars as produced by :func:`tca.fx.bars.bars_from_ticks`.

    Returns:
        One row per pair, weekday and five-minute bucket of UTC time of day, with ``dow`` equal to
        ``-1`` for the aggregate over all weekdays and ``0`` to ``4`` (Monday to Friday) for one
        weekday. Spreads are in pips and in bp of mid; ``tick_rate_per_s`` is ticks per second
        averaged over the days that traded in the bucket; ``realised_vol_ann`` annualises the
        one-minute mid return variance over 260 days of 24 hours; ``wmr_share`` is the share of
        sample days on which the bucket overlaps 15:57:30 to 16:02:30 London time and ``wmr_window``
        flags buckets where it is at least one half.
    """
    frame = _bucket_frame(bars)
    pieces = []
    for pair, sub in frame.groupby("pair", sort=True):
        by_dow = _aggregate(sub, ["dow", "bucket"])
        overall = _aggregate(sub, ["bucket"]).assign(dow=ALL_WEEKDAYS)
        table = pd.concat([overall, by_dow], ignore_index=True)
        shares = {ALL_WEEKDAYS: _wmr_share(sub["day"])}
        shares.update({int(d): _wmr_share(g["day"]) for d, g in sub.groupby("dow", sort=True)})
        table["wmr_share"] = [
            shares[dow][bucket] for dow, bucket in zip(table["dow"], table["bucket"], strict=True)
        ]
        table["wmr_window"] = table["wmr_share"] >= 0.5
        pieces.append(table.assign(pair=pair))
    out = pd.concat(pieces, ignore_index=True)
    out["bucket"] = out["bucket"].astype("int64")
    out["dow"] = out["dow"].astype("int64")
    out["n_days"] = out["n_days"].astype("int64")
    return out.sort_values(["pair", "dow", "bucket"], kind="stable").reset_index(drop=True)[
        PROFILE_COLUMNS
    ]


def round_profile(profile: pd.DataFrame) -> pd.DataFrame:
    """Round spreads, rates, volatility and shares to 1e-4 before storage."""
    out = profile.copy()
    for col in out.select_dtypes(include="float").columns:
        out[col] = out[col].round(4)
    return out
