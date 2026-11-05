"""Implementation shortfall and its decomposition."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tca.core.conventions import implementation_shortfall
from tca.fx import analysis

TOL = 1e-9
TERMS = ["delay_bp", "spread_bp", "impact_bp", "timing_bp"]


def test_filled_terms_sum_to_is_filled_bp(parents):
    filled = parents[parents["filled_frac"] > 0]
    assert len(filled) > 100
    assert (filled[TERMS].sum(axis=1) - filled["is_filled_bp"]).abs().max() < TOL


def test_is_bp_is_filled_share_times_filled_cost_plus_opportunity(parents):
    total = parents["filled_frac"] * parents["is_filled_bp"] + parents["opportunity_bp"]
    assert (total - parents["is_bp"]).abs().max() < TOL


def test_is_filled_bp_is_the_core_formula_on_the_stored_fills(sim, parents):
    fills = sim.fills.merge(sim.orders[["order_id", "side", "notional_usd", "decision_mid"]])
    for order_id, group in list(fills.groupby("order_id"))[:25]:
        row = parents.loc[parents["order_id"] == order_id].iloc[0]
        result = implementation_shortfall(
            int(row["side"]),
            group["qty_usd"],
            group["px"],
            row["notional_usd"],
            row["decision_mid"],
            sim.orders.set_index("order_id").loc[order_id, "end_mid"],
        )
        assert result.is_filled_bp == pytest.approx(row["is_filled_bp"], abs=TOL)
        assert result.is_bp == pytest.approx(row["is_bp"], abs=TOL)


def test_orders_that_fill_completely_have_no_opportunity_cost(parents):
    full = parents[parents["filled_frac"] > 1 - 1e-12]
    assert (full["opportunity_bp"].abs() < TOL).all()
    assert (full["is_bp"] - full["is_filled_bp"]).abs().max() < TOL


def _hand_inputs():
    """One buy of 100 with two fills (60 and 20), decision mid 100 and a 20% unfilled remainder."""
    orders = pd.DataFrame(
        {
            "order_id": [1],
            "pair": ["EURUSD"],
            "side": [1],
            "style": ["TWAP"],
            "urgency": ["LOW"],
            "broker": ["A"],
            "assignment": ["RANDOM"],
            "trader_id": [1],
            "notional_usd": [100.0],
            "ts_arrival_utc": [pd.Timestamp("2025-03-05", tz="UTC")],
            "planned_duration_s": [60.0],
            "pretrade_est_bp": [1.0],
            "decision_mid": [100.0],
            "first_decision_mid": [100.10],
            "end_mid": [100.40],
            "twap_mid": [100.2],
            "fix_mid": [np.nan],
            "interval_twap_eff_mid": [100.3],
        }
    )
    fills = pd.DataFrame(
        {
            "fill_id": [1, 2],
            "child_id": [1, 2],
            "order_id": [1, 1],
            "lp_id": [1, 2],
            "intent": ["AGGRESSIVE", "PASSIVE"],
            "qty_usd": [60.0, 20.0],
            "px": [100.30, 100.50],
            "mid_tape_at_fill": [100.20, 100.30],
            "mid_eff_at_fill": [100.25, 100.42],
            "impact_state_at_fill": [0.05, 0.12],
        }
    )
    attempts = pd.DataFrame({"child_id": [1], "lp_id": [1], "outcome": ["FILLED"]})
    children = pd.DataFrame({"child_id": [1, 2], "order_id": [1, 1]})
    tfill = pd.DataFrame(
        {
            "order_id": [1, 1],
            "qty_usd": [60.0, 20.0],
            "markout_eff_bp_1800": [1.0, 2.0],
            "markout_eff_bp_0": [0.0, 0.0],
        }
    )
    return orders, fills, attempts, children, tfill


def test_decomposition_matches_a_hand_computation():
    # Weights 0.75 and 0.25 of the filled 80; every term in bp of the decision mid 100.
    parent = analysis.tca_parent(*_hand_inputs()).iloc[0]
    assert parent["delay_bp"] == pytest.approx(10.0)
    assert parent["spread_bp"] == pytest.approx(5.75)
    assert parent["impact_bp"] == pytest.approx(6.75)
    assert parent["timing_bp"] == pytest.approx(12.5)
    assert parent["is_filled_bp"] == pytest.approx(35.0)
    assert parent["filled_frac"] == pytest.approx(0.8)
    assert parent["opportunity_bp"] == pytest.approx(8.0)
    assert parent["is_bp"] == pytest.approx(36.0)
    assert parent["avg_fill_px"] == pytest.approx(100.35)
    assert parent["cost_vs_interval_twap_bp"] == pytest.approx((100.35 - 100.3) / 100.3 * 1e4)
    assert parent["pct_passive_filled"] == pytest.approx(25.0)
    assert parent["reversion_30m_bp"] == pytest.approx(0.75 * 1.0 + 0.25 * 2.0)


def test_an_order_with_no_fill_is_all_opportunity_cost():
    orders, fills, attempts, children, tfill = _hand_inputs()
    parent = analysis.tca_parent(orders, fills.iloc[0:0], attempts, children, tfill.iloc[0:0]).iloc[
        0
    ]
    assert parent["filled_frac"] == 0.0
    assert parent["is_bp"] == pytest.approx(parent["opportunity_bp"])
    assert parent["opportunity_bp"] == pytest.approx(40.0)
    assert parent["delay_bp"] == 0.0
