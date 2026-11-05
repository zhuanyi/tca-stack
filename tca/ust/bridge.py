"""Intraday yield paths: a time-changed Brownian bridge pinned to the public daily closes.

Each tenor's path starts at the previous close and ends exactly on the day's close. Between
the two, increments follow a three-factor model (level, slope, curvature loadings from a PCA
of the period's daily anchor changes) with a variance clock that is U-shaped and spikes at
data releases, auction results, FOMC statements and the 15:00 marks.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from tca.core.calendar import NY
from tca.ust.config import TENORS, SimConfig
from tca.ust.grid import Grid, parse_hhmm

_RELEASE_TYPES = ("NFP", "CPI")
_AUCTION_TYPE = "AUCTION_RESULT"


@dataclass(frozen=True)
class Factors:
    """Cross-tenor structure estimated from the period's daily anchor changes.

    ``mix`` is (tenors x factors); each row has unit norm so a tenor's increment variance is
    set by its own daily variance. ``explained`` are the PCA variance shares and ``daily_sd`` the
    period standard deviation of each tenor's daily yield change as a decimal.
    """

    mix: np.ndarray
    explained: np.ndarray
    daily_sd: np.ndarray


def estimate_factors(anchors: pd.DataFrame, cfg: SimConfig) -> Factors:
    """PCA of the in-period daily anchor changes (63 days by 7 tenors) keeping ``n_factors``."""
    p = cfg.period
    in_period = anchors[(anchors["date"] >= p.start) & (anchors["date"] <= p.end)]
    changes = in_period.pivot(index="date", columns="tenor", values="change_bp")[list(TENORS)]
    cov = np.cov(changes.to_numpy(), rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1][: cfg.yield_process.n_factors]
    vectors = vectors[:, order] * np.where(vectors[:, order].sum(axis=0) < 0, -1.0, 1.0)
    scaled = vectors * np.sqrt(values[order])
    mix = scaled / np.linalg.norm(scaled, axis=1, keepdims=True)
    return Factors(mix, values[order] / values.sum(), changes.std(ddof=1).to_numpy() / 1e4)


def _baseline(grid: Grid, cfg: SimConfig) -> np.ndarray:
    """U-shaped variance weight on the grid, with a step down at 16:00."""
    vw = cfg.yield_process.variance_weight
    t = grid.seconds()
    knots = [(grid.offset(parse_hhmm(hhmm)), w) for hhmm, w in vw.baseline_points]
    base = np.interp(t, [k for k, _ in knots], [w for _, w in knots])
    base[t > knots[-1][0]] = vw.after_16_00
    return base


def _bump(t: np.ndarray, centre_s: float, height: float, halfwidth_s: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((t - centre_s) / halfwidth_s) ** 2)


def variance_weights(d: date, events: pd.DataFrame, grid: Grid, cfg: SimConfig) -> np.ndarray:
    """Variance weight on the grid for every tenor, shape (tenors, points), for business day ``d``.

    The weight is the baseline plus Gaussian bumps whose standard deviation is
    ``bump_halfwidth_s``: 08:30 on NFP and CPI days, 14:00 on FOMC days, the auction result
    (larger for the auctioned tenor) and the 15:00 marks every day.
    """
    vw = cfg.yield_process.variance_weight
    t = grid.seconds()
    sigma = vw.bump_halfwidth_s
    common = _baseline(grid, cfg) + _bump(
        t, grid.offset(parse_hhmm(vw.bump_close_marks.time)), vw.bump_close_marks.height, sigma
    )
    local = events["ts_utc"].dt.tz_convert(NY)
    today = events[local.dt.date == d]
    for row in today.itertuples():
        if row.type in _RELEASE_TYPES:
            common = common + _bump(
                t, grid.to_offset(d, row.ts_utc), vw.bump_data_release.height, sigma
            )
        elif row.type == "FOMC":
            common = common + _bump(t, grid.to_offset(d, row.ts_utc), vw.bump_fomc.height, sigma)
    out = np.tile(common, (len(TENORS), 1))
    for row in today[today["type"] == _AUCTION_TYPE].itertuples():
        centre = grid.to_offset(d, row.ts_utc)
        for i, tenor in enumerate(TENORS):
            bump = (
                vw.bump_auction_auctioned_tenor
                if tenor == row.tenor
                else vw.bump_auction_other_tenors
            )
            out[i] += _bump(t, centre, bump.height, sigma)
    return out


def bridge_paths(
    y_prev: np.ndarray,
    y_close: np.ndarray,
    daily_var: np.ndarray,
    weights: np.ndarray,
    factors: Factors,
    z: np.ndarray,
) -> np.ndarray:
    """Yield paths (tenors x points) from ``y_prev`` to ``y_close``, as decimals.

    ``daily_var`` is each tenor's day variance (decimal^2); ``weights`` the variance clock
    (tenors x points, the first column unused); ``z`` standard normals (factors x steps).
    With cumulative clock ``c(t) / c(T)`` the path is ``y_prev + B(t) - c(t)/c(T) * (B(T) -
    (y_close - y_prev))``, which ends exactly on ``y_close``.
    """
    w = weights[:, 1:]
    clock = w / w.sum(axis=1, keepdims=True)
    shocks = factors.mix @ z
    steps = np.sqrt(daily_var[:, None] * clock) * shocks
    b = np.concatenate([np.zeros((len(y_prev), 1)), np.cumsum(steps, axis=1)], axis=1)
    c = np.concatenate([np.zeros((len(y_prev), 1)), np.cumsum(clock, axis=1)], axis=1)
    drift = (y_close - y_prev)[:, None]
    return y_prev[:, None] + b - c * (b[:, -1:] - drift)
