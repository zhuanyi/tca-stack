"""Sign conventions, unit conversions and cost formulas.

Cost is positive when execution is worse than the benchmark. ``side`` is +1 for a buy and -1
for a sell; for FX a buy is a purchase of the base currency. Every function broadcasts over
numpy arrays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SIDE_BUY = 1
SIDE_SELL = -1

_BP = 1e4


def pip_size(pair: str) -> float:
    """Return the price increment of one pip: 0.01 for JPY-quoted pairs, else 0.0001."""
    return 0.01 if pair.upper().replace("/", "").endswith("JPY") else 0.0001


def cost_bp(side, exec_px, bench_px):
    """Price cost in basis points of the benchmark price; positive when worse than benchmark."""
    return side * (exec_px - bench_px) / bench_px * _BP


def cost_pips(side, exec_px, bench_px, pair: str):
    """FX price cost in pips of ``pair``; positive when worse than benchmark."""
    return side * (exec_px - bench_px) / pip_size(pair)


def cost_32nds(side, exec_px, bench_px):
    """Bond price cost in 32nds of a point, for prices in points per 100 face."""
    return side * (exec_px - bench_px) * 32.0


def cost_yld_bp(side, exec_yld, bench_yld):
    """Yield cost in basis points for yields given as decimals.

    A buyer who pays a higher price receives a lower yield, so a worse buy is positive.
    """
    return side * (bench_yld - exec_yld) * _BP


def cost_usd(cost_px_bp, notional_usd):
    """Price cost in USD on a par notional: ``cost_px_bp / 1e4 * notional_usd``."""
    return cost_px_bp / _BP * notional_usd


def markout_bp(side, mid_later, exec_px):
    """Markout in basis points of the fill price; positive when the market moved in our favour."""
    return side * (mid_later - exec_px) / exec_px * _BP


def reversion(markout_h, markout_0):
    """Markout change from fill to horizon ``h`` in basis points; negative means price came back."""
    return markout_h - markout_0


def spread_cost_bp(side, fill_px, mid_at_fill):
    """Cost of a fill against the prevailing mid in basis points of that mid."""
    return side * (fill_px - mid_at_fill) / mid_at_fill * _BP


@dataclass(frozen=True)
class ISResult:
    """Implementation shortfall of a parent order, all costs in basis points of the decision mid.

    ``is_filled_bp`` is measured on filled quantity, ``opportunity_bp`` is already weighted by
    the unfilled fraction, and ``is_bp == filled_frac * is_filled_bp + opportunity_bp``.
    """

    is_filled_bp: float
    opportunity_bp: float
    is_bp: float
    filled_frac: float


def implementation_shortfall(
    side: int, fill_qty, fill_px, parent_qty: float, decision_mid: float, end_mid: float
) -> ISResult:
    """Implementation shortfall of a parent order against its decision mid.

    ``fill_qty`` and ``fill_px`` are the child fills (same units as ``parent_qty`` and the mids);
    either may be empty. With no fills ``is_filled_bp`` is 0 and the whole order is opportunity
    cost against ``end_mid``.
    """
    qty = np.atleast_1d(np.asarray(fill_qty, dtype=float))
    px = np.atleast_1d(np.asarray(fill_px, dtype=float))
    filled = float(qty.sum())
    x = filled / parent_qty
    avg_px = float(np.dot(qty, px) / filled) if filled > 0 else decision_mid
    is_filled = float(cost_bp(side, avg_px, decision_mid))
    opportunity = (1.0 - x) * float(side * (end_mid - decision_mid) / decision_mid * _BP)
    return ISResult(is_filled, opportunity, x * is_filled + opportunity, x)
