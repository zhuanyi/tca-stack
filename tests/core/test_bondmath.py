from datetime import date

import numpy as np
import pytest

from tca.core import bondmath as bm


def test_settle_date_is_next_business_day():
    assert bm.settle_date(date(2025, 3, 6)) == date(2025, 3, 7)
    assert bm.settle_date(date(2025, 3, 7)) == date(2025, 3, 10)
    assert bm.settle_date(date(2025, 4, 17)) == date(2025, 4, 21)


def test_accrued_interest_mid_period():
    # coupons 15 Jan / 15 Jul, settle 15 Apr: 90 of 181 days elapsed
    # accrued = 100 * 0.04 / 2 * 90 / 181 = 0.994475...
    got = bm.accrued_interest(0.04, date(2027, 7, 15), date(2025, 4, 15))
    assert got == pytest.approx(2.0 * 90 / 181, abs=1e-12)


def test_accrued_interest_zero_on_coupon_date():
    assert bm.accrued_interest(0.04, date(2027, 7, 15), date(2025, 7, 15)) == 0.0


def test_one_period_left_uses_simple_interest():
    # settle on a coupon date, maturity six months later, coupon 4%, yield 5%:
    # price = (100 + 2) / (1 + 0.05 / 2) = 102 / 1.025 = 99.51219512...
    got = bm.price_from_yield(0.04, date(2025, 7, 15), date(2025, 1, 15), 0.05)
    assert got == pytest.approx(102 / 1.025, abs=1e-9)


def test_two_periods_hand_computed():
    # coupon 4%, yield 5%, two semi-annual periods on a coupon date:
    # 2 / 1.025 + 102 / 1.025**2 = 1.951220 + 97.084474 = 99.035694...
    got = bm.price_from_yield(0.04, date(2026, 1, 15), date(2025, 1, 15), 0.05)
    assert got == pytest.approx(2 / 1.025 + 102 / 1.025**2, abs=1e-9)


@pytest.mark.parametrize("coupon", [0.02, 0.0425, 0.06])
@pytest.mark.parametrize("years", [1, 2, 5, 10, 30])
def test_par_bond_at_coupon_yield_on_coupon_date_prices_at_100(coupon, years):
    settle = date(2025, 5, 15)
    maturity = date(2025 + years, 5, 15)
    assert bm.price_from_yield(coupon, maturity, settle, coupon) == pytest.approx(100.0, abs=1e-9)


@pytest.mark.parametrize("settle", [date(2025, 3, 7), date(2025, 5, 15), date(2025, 8, 20)])
@pytest.mark.parametrize("years", [0.5, 2, 7, 30])
def test_price_yield_round_trip(settle, years):
    maturity = date(settle.year + int(years), 11, 15)
    for y in (0.01, 0.0425, 0.08):
        px = bm.price_from_yield(0.0375, maturity, settle, y)
        assert bm.yield_from_price(0.0375, maturity, settle, px) == pytest.approx(y, abs=1e-9)


def test_round_trip_vectorised():
    ys = np.linspace(0.005, 0.09, 200)
    settle, maturity = date(2025, 3, 7), date(2035, 2, 15)
    px = bm.price_from_yield(0.04, maturity, settle, ys)
    np.testing.assert_allclose(bm.yield_from_price(0.04, maturity, settle, px), ys, atol=1e-9)


def test_price_from_yield_scalar_returns_float():
    assert isinstance(bm.price_from_yield(0.04, date(2030, 5, 15), date(2025, 5, 15), 0.04), float)


def test_zero_yield_price_is_sum_of_cash_flows():
    # 10 periods of 2.0 plus 100 at zero yield
    assert bm.price_from_yield(0.04, date(2030, 5, 15), date(2025, 5, 15), 0.0) == pytest.approx(
        120.0, abs=1e-9
    )


def test_dv01_positive_and_increasing_in_maturity():
    settle = date(2025, 5, 15)
    values = [bm.dv01(0.04, date(2025 + n, 5, 15), settle, 0.04) for n in (2, 5, 10, 30)]
    assert all(v > 0 for v in values)
    assert values == sorted(values)


def test_dv01_two_year_par_bond_magnitude():
    # modified duration of a 2y 4% par bond is about 1.89, so dv01 on 1m face is
    # roughly 1e6 * 1.89 * 1e-4 = 189 USD
    got = bm.dv01(0.04, date(2027, 5, 15), date(2025, 5, 15), 0.04)
    assert got == pytest.approx(189.0, rel=0.01)


def test_dv01_scales_with_face():
    args = (0.04, date(2030, 5, 15), date(2025, 5, 15), 0.04)
    assert bm.dv01(*args, face=5_000_000.0) == pytest.approx(5 * bm.dv01(*args), rel=1e-12)


def test_format_32nds():
    assert bm.format_32nds(99.515625) == "99-16+"
    assert bm.format_32nds(100.0) == "100-00"
    assert bm.format_32nds(99.5) == "99-16"
    assert bm.format_32nds(99.0 + 1 / 32) == "99-01"
    assert bm.format_32nds(99.0 + 31.5 / 32) == "99-31+"


def test_parse_32nds_round_trip():
    for px in (99.515625, 100.0, 98.96875, 101.0 + 7.5 / 32):
        assert bm.parse_32nds(bm.format_32nds(px)) == px


def test_parse_32nds_plus_and_plain():
    assert bm.parse_32nds("99-16+") == 99.515625
    assert bm.parse_32nds("100-00") == 100.0


def test_parse_32nds_rejects_malformed_input():
    with pytest.raises(ValueError, match="invalid 32nds"):
        bm.parse_32nds("no-dash-here-extra")
    with pytest.raises(ValueError, match="invalid 32nds"):
        bm.parse_32nds("abc-16")
    with pytest.raises(ValueError, match="invalid 32nds"):
        bm.parse_32nds("99")


def test_coupon_period_rejects_matured_bond():
    with pytest.raises(ValueError, match=r"settle.*must be before maturity"):
        bm._coupon_period(date(2025, 1, 15), date(2025, 1, 15))
    with pytest.raises(ValueError, match=r"settle.*must be before maturity"):
        bm._coupon_period(date(2025, 1, 15), date(2026, 1, 1))


def test_format_32nds_half_up_rounding():
    # 99 + 31.5/32 = 99.984375 -> should be 99-31+ (half-up)
    assert bm.format_32nds(99.984375) == "99-31+"
    # price just below a boundary: 100 - 1/64 = 99.984375 -> 99-31+
    # price at exact boundary: 100.0 -> 100-00
    assert bm.format_32nds(100.0) == "100-00"


def test_price_from_yield_36000_array_shape_and_values():
    ys = np.linspace(0.02, 0.06, 36_000)
    settle, maturity = date(2025, 5, 15), date(2055, 5, 15)
    px = bm.price_from_yield(0.04, maturity, settle, ys)
    assert px.shape == ys.shape
    assert np.all(np.isfinite(px))
    # prices strictly decrease as yield rises
    assert np.all(np.diff(px) < 0)
    # 20 sampled yields equal the scalar call within 1e-10
    sample_idx = np.linspace(0, len(ys) - 1, 20, dtype=int)
    for i in sample_idx:
        assert px[i] == pytest.approx(
            bm.price_from_yield(0.04, maturity, settle, float(ys[i])), abs=1e-10
        )
