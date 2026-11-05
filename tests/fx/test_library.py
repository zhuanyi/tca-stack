"""The engine as a library: an explicit order list and a strategy added by name."""

from __future__ import annotations

import pandas as pd
import pytest

from tca.fx.simulate import Overrides, run
from tca.fx.ticks import from_us


@pytest.fixture
def explicit_orders(first_day, noon_us) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": [1, 2],
            "pair": ["EURUSD", "USDJPY"],
            "side": [1, -1],
            "notional_usd": [40e6, 25e6],
            "ts_arrival_utc": from_us([noon_us, noon_us + 60_000_000]),
            "urgency": ["HIGH", "HIGH"],
            "style": ["ONE_SHOT", "ONE_SHOT"],
            "broker": ["B", "D"],
            "assignment": ["RANDOM", "RANDOM"],
            "trader_id": [1, 2],
            "planned_duration_s": [0.0, 0.0],
        }
    )


def _run(cfg, source, market, orders, strategy, **extra):
    overrides = Overrides(
        orders=orders,
        strategies={"ONE_SHOT": strategy},
        lambda_fill={p: 0.0 for p in cfg.pairs.order},
        **extra,
    )
    return run(cfg, 7, source, market, overrides)


def test_a_strategy_is_added_by_name_and_fills_the_orders(
    cfg, source, market, explicit_orders, one_shot_strategy
):
    result = _run(cfg, source, market, explicit_orders, one_shot_strategy)
    assert len(result.children) == 2
    assert result.fills.groupby("order_id")["qty_usd"].sum().to_dict() == {1: 40e6, 2: 25e6}
    assert set(result.orders["style"]) == {"ONE_SHOT"}


def test_the_broker_override_replaces_the_wheel_assignment(
    cfg, source, market, explicit_orders, one_shot_strategy
):
    result = _run(cfg, source, market, explicit_orders, one_shot_strategy, broker="A")
    assert set(result.orders["broker"]) == {"A"}


def test_a_run_is_reproducible_from_the_seed(
    cfg, source, market, explicit_orders, one_shot_strategy
):
    first = _run(cfg, source, market, explicit_orders, one_shot_strategy)
    second = _run(cfg, source, market, explicit_orders, one_shot_strategy)
    pd.testing.assert_frame_equal(first.fills, second.fills)
    pd.testing.assert_frame_equal(first.attempts, second.attempts)


def test_an_unregistered_style_is_rejected(cfg, source, market, explicit_orders):
    with pytest.raises(KeyError):
        run(cfg, 7, source, market, Overrides(orders=explicit_orders))
