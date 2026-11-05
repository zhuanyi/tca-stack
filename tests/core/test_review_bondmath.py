"""Acceptance tests: month-end coupon rule (M1), 32nds (m1), tiny yields (m2)."""

from datetime import date

import numpy as np
import pytest

from tca.core import bondmath as bm


def _reference(coupon, prev, nxt, later, settle, y):
    """Clean price and accrued from an explicit cash-flow list (Treasury street formula)."""
    w = (nxt - settle).days / (nxt - prev).days
    cfs = [coupon * 50.0] * (1 + len(later))
    cfs[-1] += 100.0
    dirty = sum(cf / (1 + y / 2) ** (w + k) for k, cf in enumerate(cfs))
    accrued = coupon * 50.0 * (1.0 - w)
    return dirty - accrued, accrued


def test_eom_feb_maturity_accrued_matches_review_hand_value():
    # 28 Feb 2027 note: coupons fall on 31 Aug and 28 Feb, so 2025-08-31 .. 2026-02-28 is 181 days.
    acc = bm.accrued_interest(0.04, date(2027, 2, 28), date(2025, 9, 15))
    assert acc == pytest.approx(2.0 * 15 / 181, abs=1e-12)
    assert acc == pytest.approx(0.165746, abs=1e-6)


def test_eom_feb_maturity_price_matches_cash_flow_sum():
    px = bm.price_from_yield(0.04, date(2027, 2, 28), date(2025, 9, 15), 0.045)
    ref_px, _ = _reference(
        0.04,
        date(2025, 8, 31),
        date(2026, 2, 28),
        [date(2026, 8, 31), date(2027, 2, 28)],
        date(2025, 9, 15),
        0.045,
    )
    assert px == pytest.approx(ref_px, abs=1e-10)
    assert px == pytest.approx(99.30002, abs=1e-5)


def test_eom_april_maturity_pays_on_31_october():
    px = bm.price_from_yield(0.04, date(2027, 4, 30), date(2025, 12, 1), 0.045)
    ref_px, ref_acc = _reference(
        0.04,
        date(2025, 10, 31),
        date(2026, 4, 30),
        [date(2026, 10, 31), date(2027, 4, 30)],
        date(2025, 12, 1),
        0.045,
    )
    assert px == pytest.approx(ref_px, abs=1e-10)
    assert bm.accrued_interest(0.04, date(2027, 4, 30), date(2025, 12, 1)) == pytest.approx(
        ref_acc, abs=1e-12
    )


def test_eom_leap_day_maturity_uses_month_ends():
    # 29 Feb 2028 is the last day of February: previous coupon 31 Aug 2027, period 182 days.
    acc = bm.accrued_interest(0.04, date(2028, 2, 29), date(2027, 10, 1))
    assert acc == pytest.approx(2.0 * 31 / 182, abs=1e-12)


def test_eom_coupon_dates_are_month_ends_when_maturity_is_month_end():
    prev, nxt, n = bm._coupon_period(date(2026, 8, 31), date(2025, 9, 15))
    assert (prev, nxt, n) == (date(2025, 8, 31), date(2026, 2, 28), 2)


def test_non_eom_maturity_keeps_its_day_of_month():
    prev, nxt, n = bm._coupon_period(date(2027, 2, 15), date(2025, 9, 15))
    assert (prev, nxt, n) == (date(2025, 8, 15), date(2026, 2, 15), 3)
    # the 30th is not a month end in every month: no end-of-month rule for an Aug 30 maturity
    prev, nxt, _ = bm._coupon_period(date(2027, 8, 30), date(2026, 9, 15))
    assert (prev, nxt) == (date(2026, 8, 30), date(2027, 2, 28))


@pytest.mark.parametrize(
    ("price", "text"),
    [
        (99.515625, "99-16+"),
        (99 + 16.25 / 32, "99-162"),
        (99 + 16.75 / 32, "99-166"),
        (99 + 16.125 / 32, "99-161"),
        (99 + 16.875 / 32, "99-167"),
        (99 + 16 / 32, "99-16"),
        (100.0, "100-00"),
    ],
)
def test_format_32nds_is_exact_on_the_256ths_grid(price, text):
    assert bm.format_32nds(price) == text
    assert bm.parse_32nds(text) == price


def test_format_32nds_round_trips_every_256th_in_a_handle():
    for k in range(256):
        px = 99.0 + k / 256.0
        assert bm.parse_32nds(bm.format_32nds(px)) == px


def test_format_32nds_rejects_off_grid_price_instead_of_rounding():
    with pytest.raises(ValueError, match="256"):
        bm.format_32nds(99.0 + 16.1 / 32)


def test_format_32nds_exact_false_rounds_half_up_to_a_half_32nd():
    assert bm.format_32nds(99.0 + 16.1 / 32, exact=False) == "99-16"
    assert bm.format_32nds(99.0 + 16.3 / 32, exact=False) == "99-16+"
    assert bm.format_32nds(99.0 + 16.75 / 32, exact=False) == "99-17"


def test_parse_32nds_reads_eighths_of_a_32nd():
    assert bm.parse_32nds("99-165") == 99 + 16.625 / 32
    assert bm.parse_32nds("99-164") == bm.parse_32nds("99-16+")


@pytest.mark.parametrize("text", ["99-16x", "99-32", "99-4", "99-168", "99-169", "-16", "99-"])
def test_parse_32nds_rejects_bad_text(text):
    with pytest.raises(ValueError, match="invalid 32nds"):
        bm.parse_32nds(text)


def _direct_dirty(coupon, n, w, y):
    """Sum the cash flows one by one; no closed-form cancellation."""
    x = y / 2.0
    total = sum(coupon * 50.0 / (1 + x) ** (w + k) for k in range(n)) + 100.0 / (1 + x) ** (
        w + n - 1
    )
    return total


@pytest.mark.parametrize("y", [1e-17, 1e-12, 1e-9, 0.0])
def test_price_is_finite_and_continuous_at_tiny_yields(y):
    mat, settle = date(2035, 5, 15), date(2025, 5, 15)
    px = bm.price_from_yield(0.04, mat, settle, y)
    assert np.isfinite(px)
    # settle is a coupon date: w = 1, n = 20, accrued = 0
    assert px == pytest.approx(_direct_dirty(0.04, 20, 1.0, y), abs=1e-10)


def test_price_at_tiny_yield_vectorised():
    ys = np.array([-1e-17, 0.0, 1e-17, 1e-9, 0.04])
    px = bm.price_from_yield(0.04, date(2035, 5, 15), date(2025, 5, 15), ys)
    assert np.all(np.isfinite(px))
    assert px[0] == pytest.approx(px[2], abs=1e-9)


@pytest.mark.parametrize("px", [-5.0, 0.0, float("nan"), float("inf")])
def test_yield_from_price_rejects_non_positive_or_non_finite_price(px):
    with pytest.raises(ValueError, match="clean_px"):
        bm.yield_from_price(0.04, date(2035, 5, 15), date(2025, 5, 15), px)


def test_yield_from_price_rejects_one_bad_element_in_an_array():
    with pytest.raises(ValueError, match="clean_px"):
        bm.yield_from_price(0.04, date(2035, 5, 15), date(2025, 5, 15), np.array([99.0, -1.0]))


def test_price_from_yield_rejects_yield_at_or_below_minus_two():
    with pytest.raises(ValueError, match="yield"):
        bm.price_from_yield(0.04, date(2035, 5, 15), date(2025, 5, 15), -2.0)


def test_price_from_yield_nan_propagates():
    assert np.isnan(bm.price_from_yield(0.04, date(2035, 5, 15), date(2025, 5, 15), np.nan))


def test_price_from_yield_rejects_settle_after_maturity():
    with pytest.raises(ValueError, match="before maturity"):
        bm.price_from_yield(0.04, date(2027, 2, 15), date(2027, 3, 15), 0.05)


@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("price", [float("nan"), float("inf"), float("-inf")])
def test_format_32nds_rejects_non_finite_price(price, exact):
    with pytest.raises(ValueError, match="not finite"):
        bm.format_32nds(price, exact=exact)
