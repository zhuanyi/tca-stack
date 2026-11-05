"""Sign and unit conventions as the FX module uses them."""

from __future__ import annotations

import pytest

from tca.core.conventions import cost_bp, cost_pips, markout_bp, pip_size


def test_buy_above_mid_costs_in_bp_and_pips():
    assert cost_bp(1, 1.10010, 1.10000) == pytest.approx(0.9090909, rel=1e-6)
    assert cost_pips(1, 1.10010, 1.10000, "EURUSD") == pytest.approx(1.0)


def test_sell_below_mid_costs():
    assert cost_bp(-1, 1.09990, 1.10000) > 0
    assert cost_pips(-1, 1.09990, 1.10000, "EURUSD") == pytest.approx(1.0)


def test_buy_below_mid_is_a_gain():
    assert cost_bp(1, 1.09990, 1.10000) < 0


def test_jpy_pip_size():
    assert pip_size("USDJPY") == 0.01
    assert pip_size("EURUSD") == 0.0001
    assert cost_pips(1, 150.02, 150.00, "USDJPY") == pytest.approx(2.0)


def test_markout_positive_when_market_moves_in_favour():
    assert markout_bp(1, 1.1002, 1.1000) > 0
    assert markout_bp(-1, 1.0998, 1.1000) > 0
    assert markout_bp(1, 1.0998, 1.1000) < 0
