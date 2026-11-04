"""Sign and unit conventions, checked against hand-computed values."""

import numpy as np
import pytest

from tca.core import conventions as cv


def test_pip_size_jpy_and_default():
    assert cv.pip_size("USDJPY") == 0.01
    assert cv.pip_size("eur/jpy") == 0.01
    assert cv.pip_size("EURUSD") == 0.0001


def test_cost_bp_buy_above_benchmark_is_positive():
    # (100.5 - 100) / 100 * 1e4 = 50 bp
    assert cv.cost_bp(1, 100.5, 100.0) == pytest.approx(50.0, abs=1e-12)


def test_cost_bp_sell_mirrors_buy():
    # sell at 99.5 vs 100: -1 * (99.5 - 100) / 100 * 1e4 = +50 bp (worse than benchmark)
    assert cv.cost_bp(-1, 99.5, 100.0) == pytest.approx(50.0, abs=1e-12)
    # sell above benchmark is a negative cost: -1 * (100.5 - 100) / 100 * 1e4 = -50 bp
    assert cv.cost_bp(-1, 100.5, 100.0) == pytest.approx(-50.0, abs=1e-12)


def test_cost_bp_broadcasts():
    side = np.array([1, -1])
    out = cv.cost_bp(side, np.array([100.5, 99.5]), 100.0)
    np.testing.assert_allclose(out, [50.0, 50.0], atol=1e-12)


def test_cost_pips_usd_quoted_and_jpy():
    # EURUSD: buy at 1.1002 vs 1.1000: 0.0002 / 0.0001 = 2 pips
    assert cv.cost_pips(1, 1.1002, 1.1000, "EURUSD") == pytest.approx(2.0, abs=1e-9)
    # USDJPY: sell at 149.95 vs 150.00: -1 * (-0.05) / 0.01 = 5 pips
    assert cv.cost_pips(-1, 149.95, 150.00, "USDJPY") == pytest.approx(5.0, abs=1e-9)


def test_cost_32nds_buy_at_100_5_is_16():
    # (100.5 - 100) * 32 = 16 thirty-seconds
    assert cv.cost_32nds(1, 100.5, 100.0) == pytest.approx(16.0, abs=1e-12)
    assert cv.cost_32nds(-1, 100.5, 100.0) == pytest.approx(-16.0, abs=1e-12)


def test_cost_yld_bp_buy_at_lower_yield_is_positive():
    # buy: exec yield 4.30%, bench yield 4.32%: +1 * (0.0432 - 0.0430) * 1e4 = 2 bp
    assert cv.cost_yld_bp(1, 0.0430, 0.0432) == pytest.approx(2.0, abs=1e-9)
    # sell mirrors: exec yield 4.34% vs 4.32%: -1 * (0.0432 - 0.0434) * 1e4 = 2 bp
    assert cv.cost_yld_bp(-1, 0.0434, 0.0432) == pytest.approx(2.0, abs=1e-9)


def test_cost_usd_uses_par_notional():
    # 2.5 bp on 10,000,000 face: 2.5 / 1e4 * 1e7 = 2,500 USD
    assert cv.cost_usd(2.5, 10_000_000) == pytest.approx(2500.0, abs=1e-9)


def test_buy_above_benchmark_has_positive_cost_in_every_measure():
    px_cost = cv.cost_bp(1, 100.5, 100.0)
    assert px_cost > 0
    assert cv.cost_32nds(1, 100.5, 100.0) > 0
    assert cv.cost_usd(px_cost, 5_000_000) > 0
    assert cv.cost_yld_bp(1, 0.0425, 0.0426) > 0


def test_markout_buy_then_market_rises_is_positive():
    # buy at 100, later mid 100.1: (100.1 - 100) / 100 * 1e4 = 10 bp
    assert cv.markout_bp(1, 100.1, 100.0) == pytest.approx(10.0, abs=1e-9)
    # sell at 100, later mid 100.1: -10 bp (market moved against the seller)
    assert cv.markout_bp(-1, 100.1, 100.0) == pytest.approx(-10.0, abs=1e-9)


def test_reversion_is_difference_of_markouts():
    # aggressive buy: markout 0 bp at fill, -3 bp after 5 minutes: reversion = -3 - 0 = -3
    assert cv.reversion(-3.0, 0.0) == pytest.approx(-3.0, abs=1e-12)


def test_spread_cost_bp():
    # buy at 100.02 with mid 100.00: 0.02 / 100 * 1e4 = 2 bp
    assert cv.spread_cost_bp(1, 100.02, 100.0) == pytest.approx(2.0, abs=1e-9)
    assert cv.spread_cost_bp(-1, 99.98, 100.0) == pytest.approx(2.0, abs=1e-9)


def test_implementation_shortfall_full_fill():
    # buy 100 filled at 100.10, decision mid 100: is_filled = 0.10 / 100 * 1e4 = 10 bp, x = 1
    r = cv.implementation_shortfall(1, [100.0], [100.10], 100.0, 100.0, 101.0)
    assert r.filled_frac == pytest.approx(1.0)
    assert r.is_filled_bp == pytest.approx(10.0, abs=1e-9)
    assert r.opportunity_bp == pytest.approx(0.0, abs=1e-12)
    assert r.is_bp == pytest.approx(10.0, abs=1e-9)


def test_implementation_shortfall_partial_fill():
    # buy 100, filled 40 @ 100.10 and 20 @ 100.40: avg = (4004 + 2008) / 60 = 100.20
    # is_filled = 0.20 / 100 * 1e4 = 20 bp; x = 0.6
    # opportunity = 0.4 * (101 - 100) / 100 * 1e4 = 40 bp; is = 0.6 * 20 + 40 = 52 bp
    r = cv.implementation_shortfall(1, [40.0, 20.0], [100.10, 100.40], 100.0, 100.0, 101.0)
    assert r.filled_frac == pytest.approx(0.6)
    assert r.is_filled_bp == pytest.approx(20.0, abs=1e-9)
    assert r.opportunity_bp == pytest.approx(40.0, abs=1e-9)
    assert r.is_bp == pytest.approx(52.0, abs=1e-9)


def test_implementation_shortfall_zero_fill_is_pure_opportunity():
    # sell, nothing filled, end mid 99 vs decision 100: 1.0 * -1 * (99 - 100) / 100 * 1e4 = 100 bp
    r = cv.implementation_shortfall(-1, [], [], 100.0, 100.0, 99.0)
    assert r.filled_frac == 0.0
    assert r.is_filled_bp == 0.0
    assert r.opportunity_bp == pytest.approx(100.0, abs=1e-9)
    assert r.is_bp == pytest.approx(100.0, abs=1e-9)


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("fills", [([], []), ([30.0], [100.2]), ([30.0, 25.0], [100.2, 99.9])])
def test_implementation_shortfall_identity(side, fills):
    qty, px = fills
    r = cv.implementation_shortfall(side, qty, px, 100.0, 100.0, 100.7)
    assert abs(r.is_bp - (r.filled_frac * r.is_filled_bp + r.opportunity_bp)) <= 1e-12
