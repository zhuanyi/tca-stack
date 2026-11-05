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
_OVERFILL_TOL = 1e-9


def pip_size(pair: str) -> float:
    """Return the price increment of one pip: 0.01 for JPY-quoted pairs, else 0.0001.

    Follows the contract convention (JPY only); USDHUF and similar 0.01-pip pairs are not covered.
    """
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
    """Price cost in the currency of ``notional_usd`` on a par notional.

    For an FX pair with a base-currency notional the result is in the base currency; the name
    keeps for USD-quoted costs on a USD notional.
    """
    return cost_px_bp / _BP * notional_usd


def markout_bp(side, mid_later, exec_px):
    """Markout in basis points of ``exec_px``; positive when the market moved in our favour.

    ``cost_bp`` and ``spread_cost_bp`` are in bp of the benchmark or mid, so the two differ by
    the relative size of the cost (about 0.02% of the value at a 2 bp cost).
    """
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
    if side not in (SIDE_BUY, SIDE_SELL):
        raise ValueError(f"side must be +1 or -1, got {side!r}")
    if not parent_qty > 0:
        raise ValueError(f"parent_qty must be positive, got {parent_qty!r}")
    qty = np.atleast_1d(np.asarray(fill_qty, dtype=float))
    px = np.atleast_1d(np.asarray(fill_px, dtype=float))
    if np.any(qty < 0):
        raise ValueError("fill_qty must not contain negative quantities")
    filled = float(qty.sum())
    if filled > parent_qty * (1.0 + _OVERFILL_TOL):
        raise ValueError(f"filled quantity {filled} exceeds parent_qty {parent_qty}")
    x = filled / parent_qty
    avg_px = float(np.dot(qty, px) / filled) if filled > 0 else decision_mid
    is_filled = float(cost_bp(side, avg_px, decision_mid))
    opportunity = (1.0 - x) * float(side * (end_mid - decision_mid) / decision_mid * _BP)
    return ISResult(is_filled, opportunity, x * is_filled + opportunity, x)


@dataclass(frozen=True)
class ParentDecomposition:
    """Parent-order cost terms in basis points of the decision mid.

    ``is_filled_bp == delay_bp + timing_bp + execution_bp`` and
    ``is_bp == filled_frac * is_filled_bp + opportunity_bp``. With no fills, the four
    filled-quantity terms are NaN and ``is_bp == opportunity_bp``.
    """

    delay_bp: float
    timing_bp: float
    execution_bp: float
    opportunity_bp: float
    is_filled_bp: float
    is_bp: float
    filled_frac: float


def decompose_parent(
    side: int,
    decision_mid: float,
    first_ref_mid: float,
    fill_ref_mid,
    fill_qty,
    fill_px,
    parent_qty: float,
    end_mid: float,
) -> ParentDecomposition:
    """Split the implementation shortfall of a parent order into delay, timing and execution.

    All terms are in basis points of ``decision_mid``. With ``D = decision_mid``, ``q_i`` the fill
    quantities, ``Q_f = sum q_i``, ``Q = parent_qty`` and ``x = Q_f / Q``:

    - ``delay_bp = side * (first_ref_mid - D) / D * 1e4``
    - ``timing_bp = sum (q_i / Q_f) * side * (fill_ref_mid_i - first_ref_mid) / D * 1e4``
    - ``execution_bp = sum (q_i / Q_f) * side * (fill_px_i - fill_ref_mid_i) / D * 1e4``
    - ``is_filled_bp = delay_bp + timing_bp + execution_bp``
    - ``opportunity_bp = (1 - x) * side * (end_mid - D) / D * 1e4``
    - ``is_bp = x * is_filled_bp + opportunity_bp``

    The reference mid differs by module. UST passes the send mid of each fill as ``fill_ref_mid``,
    and ``first_ref_mid`` is the mid of the first send even when that RFQ was passed. FX passes the
    tape mid at each fill and the first decision mid as ``first_ref_mid``; its ``spread`` and
    ``impact`` terms split ``execution_bp``.

    With no fills the filled-quantity terms (``delay_bp``, ``timing_bp``, ``execution_bp``,
    ``is_filled_bp``) are NaN, unlike ``implementation_shortfall`` where ``is_filled_bp`` is 0,
    because a cost on nothing filled is undefined. ``first_ref_mid`` may then be NaN. A fill with
    zero quantity raises, because it is a bad record.

    ``fill_ref_mid``, ``fill_qty`` and ``fill_px`` accept scalars or sequences and may be empty.
    """
    if side not in (SIDE_BUY, SIDE_SELL):
        raise ValueError(f"side must be +1 or -1, got {side!r}")
    if not (np.isfinite(decision_mid) and decision_mid > 0):
        raise ValueError(f"decision_mid must be finite and positive, got {decision_mid!r}")
    if not (np.isfinite(parent_qty) and parent_qty > 0):
        raise ValueError(f"parent_qty must be finite and positive, got {parent_qty!r}")
    if not np.isfinite(end_mid):
        raise ValueError(f"end_mid must be finite, got {end_mid!r}")
    ref = np.atleast_1d(np.asarray(fill_ref_mid, dtype=float))
    qty = np.atleast_1d(np.asarray(fill_qty, dtype=float))
    px = np.atleast_1d(np.asarray(fill_px, dtype=float))
    if not (ref.shape == qty.shape == px.shape):
        raise ValueError("fill_ref_mid, fill_qty and fill_px must have the same length")
    if not (np.all(np.isfinite(ref)) and np.all(np.isfinite(qty)) and np.all(np.isfinite(px))):
        raise ValueError("fill values must be finite")
    if np.any(qty <= 0):
        raise ValueError("fill_qty must be positive")
    filled = float(qty.sum())
    if filled > parent_qty * (1.0 + _OVERFILL_TOL):
        raise ValueError(f"filled quantity {filled} exceeds parent_qty {parent_qty}")
    x = filled / parent_qty
    opportunity = (1.0 - x) * float(side * (end_mid - decision_mid) / decision_mid * _BP)
    if qty.size == 0:
        nan = float("nan")
        return ParentDecomposition(nan, nan, nan, opportunity, nan, opportunity, 0.0)
    if not np.isfinite(first_ref_mid):
        raise ValueError(f"first_ref_mid must be finite with fills, got {first_ref_mid!r}")
    w = qty / filled
    scale = side / decision_mid * _BP
    delay = float(scale * (first_ref_mid - decision_mid))
    timing = float(scale * np.dot(w, ref - first_ref_mid))
    execution = float(scale * np.dot(w, px - ref))
    is_filled = delay + timing + execution
    return ParentDecomposition(
        delay, timing, execution, opportunity, is_filled, x * is_filled + opportunity, x
    )
