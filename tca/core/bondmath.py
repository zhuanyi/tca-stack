"""US Treasury price and yield maths on the street convention.

Coupons and yields are decimals (0.0425 is 4.25%). Prices are clean, in points per 100 face.
Yields compound semi-annually with actual/actual accrual and a fractional first period; the
final coupon period uses simple interest.
"""

from __future__ import annotations

import calendar as _calendar
import math
from datetime import date, timedelta

import numpy as np

from tca.core.calendar import is_business_day

_BUMP = 0.5e-4
_NEWTON_TOL = 1e-13
_NEWTON_MAX_ITER = 100


def settle_date(trade_date: date) -> date:
    """Regular-way Treasury settlement: the next business day after ``trade_date`` (T+1)."""
    d = trade_date + timedelta(days=1)
    while not is_business_day(d):
        d += timedelta(days=1)
    return d


def _add_months(d: date, months: int, day: int) -> date:
    """Shift ``d`` by whole months, placing the result on ``day`` or the month end if shorter."""
    index = d.year * 12 + d.month - 1 + months
    year, month = divmod(index, 12)
    return date(year, month + 1, min(day, _calendar.monthrange(year, month + 1)[1]))


def _coupon_period(maturity: date, settle: date) -> tuple[date, date, int]:
    """Return the coupon dates bracketing ``settle`` and the number of coupons still to pay.

    ``prev`` is the last coupon date on or before ``settle`` and ``nxt`` the first one after it;
    the count includes the coupon paid at maturity.
    """
    if settle >= maturity:
        raise ValueError(f"settle {settle} must be before maturity {maturity}")
    k = 0
    while _add_months(maturity, -6 * (k + 1), maturity.day) > settle:
        k += 1
    nxt = _add_months(maturity, -6 * k, maturity.day)
    prev = _add_months(maturity, -6 * (k + 1), maturity.day)
    return prev, nxt, k + 1


def _period_fraction(maturity: date, settle: date) -> tuple[float, int]:
    """Return ``(w, n)``: unexpired fraction of the current coupon period and coupons left."""
    prev, nxt, n = _coupon_period(maturity, settle)
    return (nxt - settle).days / (nxt - prev).days, n


def accrued_interest(coupon: float, maturity: date, settle: date) -> float:
    """Accrued interest in points per 100 face, actual/actual within the coupon period."""
    w, _ = _period_fraction(maturity, settle)
    return 100.0 * coupon / 2.0 * (1.0 - w)


def _dirty_price(coupon: float, maturity: date, settle: date, yld):
    """Full price in points per 100 face, summing the coupons as a geometric series."""
    w, n = _period_fraction(maturity, settle)
    y = np.asarray(yld, dtype=float)
    half_coupon = 100.0 * coupon / 2.0
    if n == 1:
        return (100.0 + half_coupon) / (1.0 + w * y / 2.0)
    v = 1.0 / (1.0 + y / 2.0)
    vn = v**n
    with np.errstate(divide="ignore", invalid="ignore"):
        annuity = np.where(y == 0.0, float(n), (1.0 - vn) / (1.0 - v))
    return v**w * (half_coupon * annuity + 100.0 * v ** (n - 1))


def _scalar_if_scalar(value, like):
    return float(value) if np.ndim(like) == 0 else value


def price_from_yield(coupon: float, maturity: date, settle: date, yld):
    """Clean price in points per 100 face for a yield given as a decimal; vectorised in ``yld``."""
    dirty = _dirty_price(coupon, maturity, settle, yld)
    return _scalar_if_scalar(dirty - accrued_interest(coupon, maturity, settle), yld)


def yield_from_price(coupon: float, maturity: date, settle: date, clean_px):
    """Yield as a decimal for a clean price in points per 100 face; vectorised in ``clean_px``.

    Solved by Newton iteration with a central-difference slope, to better than 1e-12.
    """
    px = np.asarray(clean_px, dtype=float)
    y = np.full(px.shape, coupon if coupon > 0 else 0.03)
    for _ in range(_NEWTON_MAX_ITER):
        f = price_from_yield(coupon, maturity, settle, y) - px
        slope = (
            price_from_yield(coupon, maturity, settle, y + 1e-6)
            - price_from_yield(coupon, maturity, settle, y - 1e-6)
        ) / 2e-6
        step = f / slope
        y = np.maximum(y - step, -0.99)
        if np.max(np.abs(step)) < _NEWTON_TOL:
            break
    return _scalar_if_scalar(y, clean_px)


def dv01(coupon: float, maturity: date, settle: date, yld, face: float = 1_000_000.0):
    """USD change in value of ``face`` for a one basis point fall in yield; positive.

    Computed from prices at the yield plus and minus half a basis point.
    """
    up = _dirty_price(coupon, maturity, settle, np.asarray(yld, dtype=float) + _BUMP)
    down = _dirty_price(coupon, maturity, settle, np.asarray(yld, dtype=float) - _BUMP)
    return _scalar_if_scalar((down - up) / 100.0 * face, yld)


def format_32nds(price: float) -> str:
    """Format points as ``handle-32nds``, ``+`` marking a half: 99.515625 is ``99-16+``.

    Uses half-up rounding (``floor(price * 64 + 0.5)``) rather than banker's rounding so the
    output is deterministic near 32nd boundaries.
    """
    halves = math.floor(price * 64 + 0.5)
    handle, rem = divmod(halves, 64)
    thirty_seconds, half = divmod(rem, 2)
    return f"{handle}-{thirty_seconds:02d}{'+' if half else ''}"


def parse_32nds(text: str) -> float:
    """Parse ``handle-32nds`` text, with an optional trailing ``+`` for a half, into points."""
    handle, sep, frac = text.strip().partition("-")
    if not sep or not handle.isdigit() or not frac.rstrip("+").isdigit():
        raise ValueError(f"invalid 32nds price {text!r}; expected 'handle-32nds[+]'")
    half = frac.endswith("+")
    return int(handle) + (int(frac.rstrip("+")) + 0.5 * half) / 32.0
