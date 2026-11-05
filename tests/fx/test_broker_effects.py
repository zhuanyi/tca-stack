"""The planted broker effects are visible on the same orders and the same seed."""

from __future__ import annotations

import pandas as pd
import pytest

from tca.fx.config import load_brokers
from tca.fx.simulate import Overrides, run

ORDER_COLUMNS = [
    "order_id",
    "pair",
    "side",
    "notional_usd",
    "ts_arrival_utc",
    "urgency",
    "style",
    "broker",
    "assignment",
    "trader_id",
    "planned_duration_s",
]
WHEEL_STYLES = ["TWAP", "POV", "IS"]


@pytest.fixture(scope="module")
def forced(cfg, source, market, sim):
    """The same orders sent through each broker in turn."""
    orders = sim.orders[ORDER_COLUMNS]
    runs = {}
    for broker in "ABCD":
        runs[broker] = run(
            cfg,
            20250303,
            source,
            market,
            Overrides(orders=orders, broker=broker, lambda_fill=sim.lambda_fill),
        )
    return runs


def _impact_state_bp(result) -> pd.Series:
    """Mean effective-mid offset at the fills, in bp of the tape mid in the trade direction."""
    f = result.fills.merge(result.orders[["order_id", "side", "style", "broker"]], on="order_id")
    f = f[f["broker"].notna() & f["style"].isin(WHEEL_STYLES)]
    f["offset_bp"] = f["side"] * f["impact_state_at_fill"] / f["mid_tape_at_fill"] * 1e4
    per_parent = f.groupby("order_id").agg(offset_bp=("offset_bp", "mean"), n=("fill_id", "size"))
    return per_parent[per_parent["n"] > 1]["offset_bp"]


def test_the_same_orders_are_replayed_for_every_broker(forced):
    ids = [set(r.orders["order_id"]) for r in forced.values()]
    assert all(i == ids[0] for i in ids)
    for result in forced.values():
        wheel = result.orders[result.orders["style"].isin(WHEEL_STYLES)]
        assert wheel["broker"].nunique() == 1


def test_broker_d_leaks_more_impact_at_fill_than_broker_b(forced):
    d, b = _impact_state_bp(forced["D"]), _impact_state_bp(forced["B"])
    assert len(d) > 30
    assert d.mean() > b.mean() + 0.5
    assert (d.reindex(b.index) > b).mean() > 0.9


def test_leakage_grows_with_the_number_of_fills(forced):
    f = forced["D"].fills.merge(forced["D"].orders[["order_id", "style"]], on="order_id")
    f = f[f["style"].isin(WHEEL_STYLES)].sort_values(["order_id", "fill_id"])
    f["k"] = f.groupby("order_id").cumcount()
    last, first = f.groupby("order_id").tail(1), f.groupby("order_id").head(1)
    assert last["impact_state_at_fill"].abs().mean() > first["impact_state_at_fill"].abs().mean()


def test_broker_d_has_the_best_spread_cost(forced):
    def spread_bp(result):
        f = result.fills.merge(result.orders[["order_id", "side", "style"]], on="order_id")
        f = f[f["style"].isin(["POV", "IS"])]
        return (f["side"] * (f["px"] - f["mid_eff_at_fill"]) / f["mid_tape_at_fill"] * 1e4).mean()

    costs = {b: spread_bp(r) for b, r in forced.items()}
    assert costs["D"] < costs["A"] < costs["B"]


def test_broker_a_pays_more_impact_than_broker_b(forced):
    assert _impact_state_bp(forced["A"]).mean() > _impact_state_bp(forced["B"]).mean()


def test_broker_effects_do_not_change_the_market_the_algo_sees(forced):
    a, b = forced["A"].orders, forced["B"].orders
    cols = ["decision_mid", "first_decision_mid"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_leakage_is_driven_by_the_configuration(cfg, source, market, sim, forced, monkeypatch):
    book = load_brokers()
    book["D"] = book["D"].model_copy(update={"post_fill_impact_bp": 0.0})
    monkeypatch.setattr("tca.fx.simulate.load_brokers", lambda: book)
    no_leak = run(
        cfg,
        20250303,
        source,
        market,
        Overrides(orders=sim.orders[ORDER_COLUMNS], broker="D", lambda_fill=sim.lambda_fill),
    )
    assert _impact_state_bp(no_leak).mean() == pytest.approx(_impact_state_bp(forced["B"]).mean())
