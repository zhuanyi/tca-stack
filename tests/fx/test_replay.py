"""Replay internals: the impact path, the interval benchmark and the stored markouts."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate

from tca.core.impact import ImpactState
from tca.fx.replay import impact_path, mean_impact

HALF_LIFE = 900.0


def test_impact_path_equals_the_impact_state_machine():
    adds_t = np.array([10.0, 40.0, 100.0, 400.0])
    amounts = np.array([1.0, -0.4, 2.0, 0.5])
    state = ImpactState(0.5, HALF_LIFE)
    expected = []
    for t, a in zip(adds_t, amounts, strict=True):
        state.add(t, a)
        expected.append(state.value(t))
    assert impact_path(adds_t, adds_t, amounts, 0.5, HALF_LIFE) == pytest.approx(expected)


def test_impact_path_ignores_adds_after_the_query_time():
    assert impact_path(np.array([5.0]), np.array([10.0]), np.array([3.0]), 0.5, HALF_LIFE)[0] == 0.0


def test_mean_impact_matches_numerical_integration():
    adds_t = np.array([0.0, 120.0, 800.0])
    amounts = np.array([1.0, 2.0, -0.5])
    area, _ = integrate.quad(
        lambda t: impact_path(np.array([t]), adds_t, amounts, 0.4, HALF_LIFE)[0],
        60.0,
        1500.0,
        points=[120.0, 800.0],
    )
    exact = mean_impact(60.0, 1500.0, adds_t, amounts, 0.4, HALF_LIFE)
    assert exact == pytest.approx(area / (1500.0 - 60.0), rel=1e-9)


def test_mean_impact_of_an_empty_window_is_the_point_value():
    adds_t, amounts = np.array([0.0]), np.array([1.0])
    point = impact_path(np.array([300.0]), adds_t, amounts, 0.5, HALF_LIFE)[0]
    assert mean_impact(300.0, 300.0, adds_t, amounts, 0.5, HALF_LIFE) == pytest.approx(point)


def test_every_filled_order_has_an_interval_benchmark(sim, parents):
    filled = parents[parents["filled_frac"] > 0]
    assert filled["cost_vs_interval_twap_bp"].notna().all()
    assert (
        sim.orders.loc[sim.orders["order_id"].isin(filled["order_id"]), "interval_twap_eff_mid"]
        .notna()
        .all()
    )


def test_interval_benchmark_sees_the_orders_own_impact(sim):
    """For a buy the effective mid is above the tape mid by the footprint, on average."""
    o = sim.orders.dropna(subset=["interval_twap_eff_mid"])
    o = o[o["style"].isin(["POV", "IS"]) & (o["notional_usd"] > 50e6)]
    gap = o["side"] * (o["interval_twap_eff_mid"] - o["twap_mid"]) / o["decision_mid"]
    assert len(o) > 5
    assert gap.mean() > 0


def test_markouts_have_one_row_per_fill_and_a_zero_horizon(sim):
    assert len(sim.markouts) == len(sim.fills)
    assert set(sim.markouts["fill_id"]) == set(sim.fills["fill_id"])
    assert "markout_eff_bp_0" in sim.markouts


def test_effective_markout_at_zero_horizon_reflects_the_fill_footprint(sim):
    """At horizon 0 the effective mid includes the fill's own impact, the tape mid does not."""
    f = sim.fills.merge(sim.markouts, on="fill_id").merge(sim.orders[["order_id", "side"]])
    aggressive = f[f["intent"] == "AGGRESSIVE"]
    assert (aggressive["markout_eff_bp_0"] - aggressive["markout_tape_bp_0"]).mean() > 0


def test_children_never_overlap_in_time(sim):
    f = sim.fills.sort_values(["order_id", "ts_fill_utc"])
    c = sim.children.set_index("child_id")["ts_decision_utc"]
    f["decision"] = f["child_id"].map(c)
    prior_fill = f.groupby("order_id")["ts_fill_utc"].shift()
    ok = f["decision"] >= prior_fill.fillna(f["decision"]) - np.timedelta64(1, "us")
    assert ok.mean() > 0.99
