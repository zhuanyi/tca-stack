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

from tca.core.calendar import is_business_day, is_settlement_day

_BUMP = 0.5e-4
_NEWTON_TOL = 1e-13
_NEWTON_MAX_ITER = 100


def settle_date(trade_date: date) -> date:
    """Regular-way Treasury settlement: the next settlement day after ``trade_date`` (T+1).

    A trade entered on a business day that is not a settlement day (Good Friday 2026-04-03)
    settles T+2, the second settlement day after it, as SIFMA recommends.
    """
    steps = 2 if is_business_day(trade_date) and not is_settlement_day(trade_date) else 1
    d = trade_date
    while steps:
        d += timedelta(days=1)
        steps -= is_settlement_day(d)
    return d


def _add_months(d: date, months: int, day: int, eom: bool = False) -> date:
    """Shift ``d`` by whole months onto ``day``, or the month end if shorter or ``eom`` is set."""
    index = d.year * 12 + d.month - 1 + months
    year, month = divmod(index, 12)
    last = _calendar.monthrange(year, month + 1)[1]
    return date(year, month + 1, last if eom else min(day, last))


def _coupon_period(maturity: date, settle: date) -> tuple[date, date, int]:
    """Return the coupon dates bracketing ``settle`` and the number of coupons still to pay.

    ``prev`` is the last coupon date on or before ``settle`` and ``nxt`` the first one after it;
    the count includes the coupon paid at maturity.
    """
    if settle >= maturity:
        raise ValueError(f"settle {settle} must be before maturity {maturity}")
    eom = maturity.day == _calendar.monthrange(maturity.year, maturity.month)[1]
    k = 0
    while _add_months(maturity, -6 * (k + 1), maturity.day, eom) > settle:
        k += 1
    nxt = _add_months(maturity, -6 * k, maturity.day, eom)
    prev = _add_months(maturity, -6 * (k + 1), maturity.day, eom)
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
    if np.any(y <= -2.0):
        raise ValueError("yield must be greater than -2 (200%)")
    half_coupon = 100.0 * coupon / 2.0
    if n == 1:
        return (100.0 + half_coupon) / (1.0 + w * y / 2.0)
    x = y / 2.0
    v = 1.0 / (1.0 + x)
    small = np.abs(x) < 1e-8
    safe_x = np.where(small, 1.0, x)
    with np.errstate(divide="ignore", invalid="ignore"):
        closed = -np.expm1(-n * np.log1p(safe_x)) * (1.0 + safe_x) / safe_x
    annuity = np.where(small, n - x * n * (n - 1) / 2.0, closed)
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
    if not np.all(np.isfinite(px)) or np.any(px <= 0):
        raise ValueError("clean_px must be finite and positive")
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


def format_32nds(price: float, *, exact: bool = True) -> str:
    """Format points as ``handle-32nds``: 99.515625 is ``99-16+``, 99.5078125 is ``99-162``.

    A half 32nd is ``+``; other ticks of 1/256 are one trailing digit counting eighths of a
    32nd (``99-162`` is 16 and a quarter). With ``exact=True`` a price off the 1/256 grid
    raises ``ValueError``; ``exact=False`` rounds half-up to the nearest half 32nd, so any
    fraction finer than the eighth digit (and any eighth that is not a half) is dropped. A
    non-finite price (nan or inf) raises ``ValueError``.
    """
    if not math.isfinite(price):
        raise ValueError(f"price {price!r} is not finite")
    if not exact:
        halves = math.floor(price * 64 + 0.5)
        handle, rem = divmod(halves, 64)
        thirty_seconds, half = divmod(rem, 2)
        return f"{handle}-{thirty_seconds:02d}{'+' if half else ''}"
    ticks = round(price * 256)
    if abs(price * 256 - ticks) > 1e-6:
        raise ValueError(
            f"price {price!r} is not on the 1/256 grid; pass exact=False to round to a half 32nd"
        )
    handle, rem = divmod(ticks, 256)
    thirty_seconds, eighths = divmod(rem, 8)
    suffix = "+" if eighths == 4 else (str(eighths) if eighths else "")
    return f"{handle}-{thirty_seconds:02d}{suffix}"


def parse_32nds(text: str) -> float:
    """Parse ``handle-32nds`` text into points; the fraction is ``NN``, ``NN+`` or ``NNd``.

    ``NN`` is whole 32nds (00 to 31), ``+`` a half and a trailing digit ``d`` (1 to 7) eighths of a
    32nd, so ``99-162`` is 16 and a quarter.
    """
    handle, sep, frac = text.strip().partition("-")
    err = ValueError(
        f"invalid 32nds price {text!r}; expected 'handle-NN', 'handle-NN+' or 'handle-NNd'"
    )
    if not sep or not handle.isdigit() or len(frac) not in (2, 3) or not frac[:2].isdigit():
        raise err
    thirty_seconds, tail = int(frac[:2]), frac[2:]
    if thirty_seconds > 31:
        raise err
    if tail == "+":
        eighths = 4
    elif tail == "":
        eighths = 0
    elif tail in "1234567":
        eighths = int(tail)
    else:
        raise err
    return int(handle) + (thirty_seconds + eighths / 8.0) / 32.0
