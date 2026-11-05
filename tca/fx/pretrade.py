"""Pre-trade cost estimate: half-spread plus square-root impact, with no fit to realised costs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tca.fx import ticks
from tca.fx.config import FxConfig
from tca.fx.market import Market

_STYLE_KEY = {"TWAP": "twap", "RISK_TRANSFER": "risk_transfer"}


def duration_factor(duration_s: float, cfg: FxConfig) -> float:
    """``g(T) = (T_ref / T) ** exponent`` clipped to the configured range, ``T`` in minutes.

    A zero duration (a single principal fill) gives the upper clip.
    """
    spec = cfg.pretrade
    minutes = max(duration_s / 60.0, 1e-9)
    low, high = spec.duration_clip
    return float(np.clip((spec.t_ref_minutes / minutes) ** spec.duration_exponent, low, high))


def estimate_bp(order_row: pd.Series, arrival_us: int, market: Market, cfg: FxConfig) -> float:
    """Estimated implementation shortfall of one order in bp of mid.

    ``est = hs_bp * f_style + Y * sigma_d_bp * sqrt(Q / V) * g(T)``: ``hs_bp`` is the median half
    spread of the arrival bucket, ``sigma_d_bp`` the trailing daily volatility in bp, ``Q`` the
    notional and ``V`` the daily volume proxy, both in USD.
    """
    pair = str(order_row["pair"])
    day = pd.Timestamp(arrival_us, unit="us").date()
    factor = cfg.pretrade.style_factor[_STYLE_KEY.get(str(order_row["style"]), "aggressive")]
    spread_term = market.half_spread_bp(pair, arrival_us) * factor
    sigma_bp = market.sigma_d(pair, day) * 1e4
    impact_term = (
        cfg.impact.y
        * sigma_bp
        * np.sqrt(float(order_row["notional_usd"]) / cfg.daily_volume_usd(pair))
        * duration_factor(float(order_row["planned_duration_s"]), cfg)
    )
    return float(spread_term + impact_term)


def estimate_orders(orders: pd.DataFrame, market: Market, cfg: FxConfig) -> np.ndarray:
    """Pre-trade estimate in bp for every row of ``orders`` (computed before any replay)."""
    arrival = ticks.to_us(orders["ts_arrival_utc"])
    return np.array(
        [
            estimate_bp(row, int(us), market, cfg)
            for (_, row), us in zip(orders.iterrows(), arrival, strict=True)
        ]
    )
