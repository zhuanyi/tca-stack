"""Child schedules of the execution strategies."""

from __future__ import annotations

from datetime import date
from itertools import pairwise

import numpy as np
import pytest

from tca.core.calendar import wmr_window_utc
from tca.fx.model import Order
from tca.fx.strategies import (
    STRATEGIES,
    StrategyContext,
    almgren_chriss_holdings,
    whole_dollar_split,
)
from tca.fx.ticks import US_PER_S, day_start_us

DAY = date(2025, 3, 5)


def _order(style: str, notional: float, arrival_us: int | None = None) -> Order:
    arrival = day_start_us(DAY) + 9 * 3600 * US_PER_S if arrival_us is None else arrival_us
    return Order(1, DAY, "EURUSD", 1, notional, arrival, "LOW", style, "A", "RANDOM", 1, 0.0)


@pytest.fixture
def ctx(cfg, market):
    return StrategyContext(cfg, market)


@pytest.mark.parametrize("style", ["TWAP", "IS", "RISK_TRANSFER", "FIX_ORDER"])
@pytest.mark.parametrize("notional", [1e6, 25e6, 137_500_000.0, 750e6])
def test_children_sum_to_parent(style, notional, ctx, rng):
    plan = STRATEGIES[style].schedule(_order(style, notional), ctx, rng)
    assert sum(c.qty_usd for c in plan) == pytest.approx(notional, abs=1.0)
    assert all(c.qty_usd > 0 for c in plan)
    offsets = [c.offset_s for c in plan]
    assert offsets == sorted(offsets)


def test_pov_never_exceeds_the_parent(ctx, rng):
    order = _order("POV", 25e6)
    plan = STRATEGIES["POV"].schedule(order, ctx, rng)
    assert sum(c.qty_usd for c in plan) <= 25e6


def test_twap_slices_are_equal_to_the_dollar(ctx, rng):
    plan = STRATEGIES["TWAP"].schedule(_order("TWAP", 40e6), ctx, rng)
    qty = np.array([c.qty_usd for c in plan])
    assert qty.max() - qty.min() <= 1.0
    assert plan[0].offset_s == 0.0


def test_almgren_chriss_is_monotone_and_front_loaded():
    holdings = almgren_chriss_holdings(30, kappa_t=3.0)
    assert holdings[0] == pytest.approx(1.0) and holdings[-1] == pytest.approx(0.0)
    assert (np.diff(holdings) < 0).all()
    slices = -np.diff(holdings)
    assert (np.diff(slices) < 0).all()
    assert slices[0] > 1.0 / 30


def test_almgren_chriss_without_risk_aversion_is_time_weighted():
    slices = -np.diff(almgren_chriss_holdings(10, kappa_t=0.0))
    assert slices == pytest.approx(np.full(10, 0.1))


def test_is_strategy_front_loads_a_large_order(ctx, rng):
    plan = STRATEGIES["IS"].schedule(_order("IS", 300e6), ctx, rng)
    qty = [c.qty_usd for c in plan]
    assert qty[0] > qty[-1]
    assert all(a >= b for a, b in pairwise(qty))


def test_whole_dollar_split_sums_exactly():
    parts = whole_dollar_split(100.0, np.ones(7))
    assert parts.sum() == 100.0
    assert (parts == np.round(parts)).all()


def test_fix_order_children_start_in_the_window(ctx, rng):
    start, end = wmr_window_utc(DAY)
    arrival = int(start.timestamp() * US_PER_S) - 600 * US_PER_S
    order = _order("FIX_ORDER", 50e6, arrival_us=arrival)
    plan = STRATEGIES["FIX_ORDER"].schedule(order, ctx, rng)
    first = arrival + plan[0].offset_s * US_PER_S
    last = arrival + plan[-1].offset_s * US_PER_S
    assert first == pytest.approx(start.timestamp() * US_PER_S)
    assert last < end.timestamp() * US_PER_S
