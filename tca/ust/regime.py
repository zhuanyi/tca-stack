"""Weekly regime numbers and the simulated April 2025 episode against the NY Fed readings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tca.core import storage
from tca.core.calendar import NY
from tca.core.stats import bootstrap_ci
from tca.ust import market
from tca.ust.config import SimConfig
from tca.ust.inputs import load_inputs

READ_TENORS = (2, 5, 10)
_CORE_FROM, _CORE_TO = "08:35", "15:00"
_MOVING_AVERAGE_DAYS = 5


def _core_hours(bars: pd.DataFrame) -> pd.DataFrame:
    """Bars between 08:35 and 15:00 New York time, where the time-of-day multiplier is 1."""
    local = bars["ts_utc"].dt.tz_convert(NY)
    clock = local.dt.strftime("%H:%M")
    return bars[(clock >= _CORE_FROM) & (clock < _CORE_TO)].assign(date=local.dt.date)


def daily_liquidity(cfg: SimConfig, derived: Path | None = None) -> pd.DataFrame:
    """Per day and read tenor, the on-the-run composite full spread, depth and impact slope.

    ``full_spread_32nds`` is twice the mean half-spread of the core hours of the simulated bars,
    ``depth_mm`` the mean touch depth, and ``impact_32nds_per_100mm`` the configured slope times
    the day's impact multiplier (the impact of a fill is that slope times ``sqrt(q / 100mm)``).
    """
    inputs = load_inputs(derived)
    uni = inputs.universe[
        (inputs.universe["run_status"] == "OTR") & inputs.universe["tenor"].isin(READ_TENORS)
    ]
    bars = storage.read_table("ust_bars_1m", derived)
    bars = _core_hours(bars[bars["cusip"].isin(uni["cusip"])])
    bars = bars.merge(uni, on=["cusip", "date"])
    daily = (
        bars.groupby(["date", "tenor"])
        .agg(half_spread=("half_spread_32nds", "mean"), depth_mm=("depth_mm", "mean"))
        .reset_index()
    )
    daily["full_spread_32nds"] = 2.0 * daily.pop("half_spread")
    ctx = market.Context.build(cfg, inputs)
    mult = {d: ctx.multipliers(d).impact for d in sorted(set(daily["date"]))}
    slope = cfg.impact.price_impact_32nds_per_100mm
    daily["impact_32nds_per_100mm"] = [
        slope[t] * mult[d] for d, t in zip(daily["date"], daily["tenor"], strict=True)
    ]
    return daily


def april_peak(cfg: SimConfig, derived: Path | None = None) -> pd.DataFrame:
    """Simulated five-day-average extreme of each liquidity measure against the Fleming reading.

    Spread and impact are compared at their peak, depth at its trough. ``normal`` is the mean
    over the days before 2025-04-01, ``sim_ratio`` is peak over that mean, and ``reading_ratio``
    is the reading over the configured March level. The readings are
    ``composite_quote.april_2025_reference``.
    """
    q = cfg.composite_quote
    ref = q.april_2025_reference
    configured = {
        "full_spread_32nds": {t: 2.0 * q.hs_base_32nds[t] for t in READ_TENORS},
        "depth_mm": q.depth_mm,
        "impact_32nds_per_100mm": cfg.impact.price_impact_32nds_per_100mm,
    }
    daily = daily_liquidity(cfg, derived)
    readings = {
        "full_spread_32nds": (ref.full_spread_32nds_peak, "max"),
        "depth_mm": (ref.depth_mm_trough, "min"),
        "impact_32nds_per_100mm": (ref.price_impact_32nds_per_100mm_peak, "max"),
    }
    rows = []
    for tenor in READ_TENORS:
        d = daily[daily["tenor"] == tenor].set_index("date").sort_index()
        normal_days = d.index < pd.Timestamp("2025-04-01").date()
        for column, (reading, extreme) in readings.items():
            smooth = d[column].rolling(_MOVING_AVERAGE_DAYS, center=True, min_periods=3).mean()
            peak = smooth.max() if extreme == "max" else smooth.min()
            normal = float(d.loc[normal_days, column].mean())
            rows.append(
                (
                    tenor,
                    column,
                    normal,
                    float(peak),
                    smooth.idxmax() if extreme == "max" else smooth.idxmin(),
                    reading[tenor],
                    float(peak) / normal,
                    reading[tenor] / configured[column][tenor],
                )
            )
    return pd.DataFrame(
        rows,
        columns=[
            "tenor",
            "measure",
            "sim_normal",
            "sim_peak",
            "sim_peak_date",
            "reading",
            "sim_ratio",
            "reading_ratio",
        ],
    )


def _week_of(ts: pd.Series) -> pd.Series:
    """Monday (New York) starting the calendar week of each timestamp."""
    local = ts.dt.tz_convert(NY).dt.tz_localize(None)
    return local.dt.to_period("W-SUN").dt.start_time


def distribution(x, rng: np.random.Generator) -> dict[str, float]:
    """Median, mean with 95% bootstrap intervals, 10th and 90th percentile of ``x``; NaN dropped."""
    v = np.asarray(x, dtype=float)
    v = v[~np.isnan(v)]
    med_lo, med_hi = bootstrap_ci(v, stat=np.median, rng=rng)
    mean_lo, mean_hi = bootstrap_ci(v, rng=rng)
    return {
        "n": float(len(v)),
        "median": float(np.median(v)),
        "median_lo": med_lo,
        "median_hi": med_hi,
        "mean": float(v.mean()),
        "mean_lo": mean_lo,
        "mean_hi": mean_hi,
        "p10": float(np.quantile(v, 0.1)),
        "p90": float(np.quantile(v, 0.9)),
    }


def weekly(
    cfg: SimConfig, fills: pd.DataFrame, anchors: pd.DataFrame, parents: pd.DataFrame
) -> pd.DataFrame:
    """Per calendar week: fill cost and parent IS distributions, realised 10y vol, mean ratio ``r``.

    Columns ``cost_*`` describe ``cost_32nds`` of filled RFQs (``n`` is the fill count),
    ``is_*`` describe the parent ``is_bp`` (``n`` parents) with the same statistics.
    ``realised_vol_bp`` is the root mean square of the week's daily 10y par-yield changes
    (bp per day) and ``mean_r`` the week's mean volatility ratio. Unfilled or missing values
    are excluded from every bootstrap.
    """
    ref = cfg.composite_quote.regime_reference_tenor
    a = anchors[anchors["tenor"] == ref].copy()
    a = a[(a["date"] >= cfg.period.start) & (a["date"] <= cfg.period.end)]
    a["week"] = pd.to_datetime(a["date"]).dt.to_period("W-SUN").dt.start_time
    vol = a.groupby("week").agg(
        realised_vol_bp=("change_bp", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        mean_r=("regime_mult", "mean"),
    )
    rng = np.random.default_rng(cfg.seed)
    frames = []
    for prefix, frame, column, stamp in (
        ("cost", fills, "cost_32nds", "ts_send_utc"),
        ("is", parents, "is_bp", "ts_create_utc"),
    ):
        grouped = frame.assign(week=_week_of(frame[stamp])).groupby("week")
        rows = {w: distribution(g[column], rng) for w, g in grouped}
        frames.append(pd.DataFrame.from_dict(rows, orient="index").add_prefix(f"{prefix}_"))
    out = frames[0].join(frames[1]).join(vol)
    out.index.name = "week"
    return out.reset_index()


HIGH_VOL_BP = 7.0
SIZE_EDGES = (0, 10, 25, 50, 100, 500)
SIZE_LABELS = ("<=10", "10-25", "25-50", "50-100", "100-500")
_MIN_GAP_FILLS = 5


def high_vol_weeks(
    weekly_table: pd.DataFrame, threshold: float = HIGH_VOL_BP
) -> list[pd.Timestamp]:
    """Weeks whose realised 10y vol (bp per day) is above ``threshold``."""
    return list(weekly_table.loc[weekly_table["realised_vol_bp"] > threshold, "week"])


def on_the_run_pool(fills: pd.DataFrame, tenors: tuple[int, ...]) -> pd.DataFrame:
    """On-the-run fills of ``tenors`` with ``cost_per_hs``, cost over the composite half-spread."""
    sel = fills[fills["tenor"].isin(tenors) & (fills["run_status"] == "OTR")].copy()
    sel["cost_per_hs"] = sel["cost_32nds"] / sel["hs_send_32nds"]
    sel["week"] = _week_of(sel["ts_send_utc"])
    return sel


def size_gap(
    fills: pd.DataFrame, weeks: list[pd.Timestamp], value: str, rng: np.random.Generator
) -> pd.DataFrame:
    """Mean of ``value`` in ``weeks`` minus the mean in all other weeks, by size bucket.

    ``fills`` needs ``week`` and ``qty_mm``. Intervals are 95% bootstrap percentiles of the
    difference of resampled means; buckets with fewer than five fills in ``weeks`` are NaN.
    """
    bucket = pd.cut(fills["qty_mm"], list(SIZE_EDGES), labels=list(SIZE_LABELS))
    rows = []
    for label in SIZE_LABELS:
        g = fills[bucket == label]
        hot = g.loc[g["week"].isin(weeks), value].dropna().to_numpy()
        calm = g.loc[~g["week"].isin(weeks), value].dropna().to_numpy()
        if len(hot) < _MIN_GAP_FILLS:
            rows.append((label, len(hot), len(calm), np.nan, np.nan, np.nan))
            continue
        draws = [
            rng.choice(hot, len(hot)).mean() - rng.choice(calm, len(calm)).mean()
            for _ in range(2000)
        ]
        lo, hi = np.quantile(draws, [0.025, 0.975])
        rows.append((label, len(hot), len(calm), hot.mean() - calm.mean(), lo, hi))
    columns = ["size_mm", "n_high_vol", "n_calm", "gap", "lo", "hi"]
    return pd.DataFrame(rows, columns=columns).set_index("size_mm")


def weekly_otr10(
    cfg: SimConfig, fills: pd.DataFrame, anchors: pd.DataFrame, parents: pd.DataFrame
) -> pd.DataFrame:
    """``weekly`` restricted to 10y on-the-run fills and parents, so tenor mix cannot move it."""

    def keep(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[(frame["tenor"] == 10) & (frame["run_status"] == "OTR")]

    return weekly(cfg, keep(fills), anchors, keep(parents))
