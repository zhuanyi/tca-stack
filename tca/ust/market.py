"""The simulated interdealer market for one business day.

For every security of the day: the fundamental yield and clean price path on the one-second
grid, the composite half-spread and the touch depth. Impact from trades is not in these arrays;
it lives in :mod:`tca.ust.ledger` and is added on top to give the effective mid.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from tca.core import bondmath
from tca.ust import bridge
from tca.ust.calibrate import Elasticities, derive_elasticities, market_ratio
from tca.ust.config import TENORS, SimConfig
from tca.ust.grid import Grid, parse_hhmm
from tca.ust.inputs import Inputs

STREAM_MARKET = 1


@dataclass(frozen=True)
class Context:
    """Everything a day needs that does not depend on the day: inputs, grid, factors, regime."""

    cfg: SimConfig
    inputs: Inputs
    grid: Grid
    factors: bridge.Factors
    elasticities: Elasticities
    close: pd.DataFrame
    tenor_regime: pd.DataFrame
    ratio: pd.Series

    @classmethod
    def build(cls, cfg: SimConfig, inputs: Inputs) -> Context:
        a = inputs.anchors
        return cls(
            cfg=cfg,
            inputs=inputs,
            grid=Grid.from_config(cfg),
            factors=bridge.estimate_factors(a, cfg),
            elasticities=derive_elasticities(a, cfg),
            close=a.pivot(index="date", columns="tenor", values="close_yield")[list(TENORS)],
            tenor_regime=a.pivot(index="date", columns="tenor", values="regime_mult")[list(TENORS)],
            ratio=market_ratio(a, cfg),
        )

    def multipliers(self, d: date) -> Regime:
        """Spread, depth and impact multipliers of day ``d`` from the channel elasticities."""
        r = float(self.ratio[d])
        clip = self.cfg.composite_quote.regime_clip
        e = self.elasticities
        return Regime(
            ratio=r,
            spread=float(np.clip(r**e.spread, *clip.spread)),
            depth=float(np.clip(r**e.depth, *clip.depth)),
            impact=float(np.clip(r**e.impact, *clip.impact)),
        )


@dataclass(frozen=True)
class Regime:
    """Volatility ratio ``r`` of a day and the three channel multipliers derived from it."""

    ratio: float
    spread: float
    depth: float
    impact: float


@dataclass(frozen=True)
class SecurityDay:
    """A security on one day with its settlement date."""

    cusip: str
    tenor: int
    run_status: str
    coupon: float
    maturity: date
    settle: date


@dataclass
class DayMarket:
    """Fundamental market arrays of one day; rows follow ``securities``."""

    day: date
    securities: list[SecurityDay]
    regime: Regime
    base_px: np.ndarray
    base_yld: np.ndarray
    half_spread_32nds: np.ndarray
    depth_mm: np.ndarray
    impact_per_100mm_32nds: np.ndarray
    cum_px: np.ndarray

    def row(self, cusip: str) -> int:
        """Row of ``cusip`` in the day's arrays."""
        for i, s in enumerate(self.securities):
            if s.cusip == cusip:
                return i
        raise KeyError(f"{cusip} is not in the universe on {self.day}")

    def mean_base_px(self, row: int, start_idx, end_idx) -> np.ndarray:
        """Mean fundamental price over grid indices ``[start_idx, end_idx)``, at least one point."""
        lo = np.asarray(start_idx)
        hi = np.maximum(np.asarray(end_idx), lo + 1)
        return (self.cum_px[row, hi] - self.cum_px[row, lo]) / (hi - lo)


def day_securities(ctx: Context, d: date) -> list[SecurityDay]:
    """The day's 15 securities in table order, with ``settle = max(T+1, issue date)``."""
    uni = ctx.inputs.universe
    day = uni[uni["date"] == d]
    sec = ctx.inputs.securities.set_index("cusip")
    t1 = bondmath.settle_date(d)
    out = []
    for row in day.itertuples():
        s = sec.loc[row.cusip]
        out.append(
            SecurityDay(
                cusip=row.cusip,
                tenor=int(row.tenor),
                run_status=row.run_status,
                coupon=float(s["coupon"]),
                maturity=s["maturity_date"],
                settle=max(t1, s["issue_date"]),
            )
        )
    return out


def _ou_noise(rng: np.random.Generator, n_rows: int, n_steps: int, half_life_s: float, sd: float):
    """Stationary AR(1) paths with the given half-life and standard deviation, one per row."""
    phi = 0.5 ** (1.0 / half_life_s)
    shocks = rng.standard_normal((n_rows, n_steps))
    start = rng.standard_normal((n_rows, 1)) * sd
    first = start
    rest = lfilter(
        [1.0], [1.0, -phi], sd * np.sqrt(1.0 - phi**2) * shocks[:, 1:], axis=1, zi=phi * start
    )[0]
    return np.concatenate([first, rest], axis=1)


def _tod_multiplier(ctx: Context) -> np.ndarray:
    t = ctx.grid.seconds()
    out = np.ones_like(t)
    for band in ctx.cfg.composite_quote.tod_mult:
        lo, hi = ctx.grid.offset(parse_hhmm(band.from_)), ctx.grid.offset(parse_hhmm(band.to))
        out[(t >= lo) & (t < hi)] = band.mult
    return out


def build_day(ctx: Context, d: date, seed: int) -> DayMarket:
    """Simulate the fundamental paths, half-spreads and depth of business day ``d``."""
    cfg, grid = ctx.cfg, ctx.grid
    rng = np.random.default_rng([seed, STREAM_MARKET, d.toordinal()])
    prev = ctx.close.index[ctx.close.index.get_loc(d) - 1]
    sd = ctx.factors.daily_sd
    var = sd**2 * ctx.tenor_regime.loc[d].to_numpy()
    weights = bridge.variance_weights(d, ctx.inputs.events, grid, cfg)
    z = rng.standard_normal((cfg.yield_process.n_factors, grid.n_seconds))
    tenor_paths = bridge.bridge_paths(
        ctx.close.loc[prev].to_numpy(), ctx.close.loc[d].to_numpy(), var, weights, ctx.factors, z
    )

    secs = day_securities(ctx, d)
    n_sec, n_pts = len(secs), grid.n_seconds + 1
    noise_cfg = cfg.yield_process.security_noise
    noise = _ou_noise(
        rng, n_sec, n_pts, noise_cfg.ou_half_life_min * 60.0, noise_cfg.ou_stationary_sd_bp / 1e4
    )
    offsets = cfg.yield_process.off_the_run_yield_spread_bp
    rows = [TENORS.index(s.tenor) for s in secs]
    spread_bp = np.array([offsets[s.run_status] for s in secs]) / 1e4
    base_yld = tenor_paths[rows] + noise + spread_bp[:, None]
    base_px = np.stack(
        [
            bondmath.price_from_yield(s.coupon, s.maturity, s.settle, base_yld[i])
            for i, s in enumerate(secs)
        ]
    )

    regime = ctx.multipliers(d)
    q = cfg.composite_quote
    block = int(q.hs_eps_resample_min * 60)
    eps = rng.standard_normal((n_sec, grid.n_seconds // block + 1)) * q.hs_eps_sd
    eps_path = np.repeat(eps, block, axis=1)[:, :n_pts]
    hs_base = np.array([q.hs_base_32nds[s.tenor] * q.run_mult[s.run_status] for s in secs])
    hs = hs_base[:, None] * _tod_multiplier(ctx)[None, :] * regime.spread * np.exp(eps_path)
    depth = np.array(
        [q.depth_mm[s.tenor] * q.run_mult[s.run_status] ** q.depth_run_mult_exponent for s in secs]
    )
    impact = np.array([cfg.impact.price_impact_32nds_per_100mm[s.tenor] for s in secs])
    return DayMarket(
        day=d,
        securities=secs,
        regime=regime,
        base_px=base_px,
        base_yld=base_yld,
        half_spread_32nds=hs,
        depth_mm=depth * regime.depth,
        impact_per_100mm_32nds=impact * regime.impact,
        cum_px=np.concatenate([np.zeros((n_sec, 1)), np.cumsum(base_px, axis=1)], axis=1),
    )
