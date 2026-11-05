from datetime import date, time

import numpy as np
import pandas as pd
import pytest

from tca.core.calendar import NY, local_to_utc
from tca.ust import costs


def test_identities_hold_on_the_simulated_parents(cfg, sim):
    p = sim["ust_tca_parent"]
    filled = p[p["filled_qty_mm"] > 0]
    parts = filled["delay_cost_bp"] + filled["timing_cost_bp"] + filled["execution_cost_bp"]
    assert np.allclose(parts, filled["is_filled_bp"], atol=1e-9)
    total = p["filled_frac"] * p["is_filled_bp"].fillna(0.0) + p["opportunity_cost_bp"]
    assert np.allclose(total, p["is_bp"], atol=1e-9)
    assert p["is_bp"].notna().all()


def test_worked_example_from_the_corrections(cfg, inputs):
    """Decision 100.00, sends at 100.00 and 100.50, each fill 0.02 above its send mid."""
    sec = inputs.securities
    cusip = sec[sec["tenor"] == 10]["cusip"].iloc[0]
    t0 = local_to_utc(date(2025, 3, 5), time(10, 0), NY)
    orders = pd.DataFrame(
        {
            "order_id": [1],
            "ts_create_utc": [t0],
            "cusip": [cusip],
            "tenor": [10],
            "run_status": ["OTR"],
            "side": [1],
            "qty_mm": [10.0],
            "decision_mid": [100.0],
            "end_mid": [100.5],
        }
    )
    rfq = pd.DataFrame(
        {
            "rfq_id": [1, 2],
            "order_id": [1, 1],
            "ts_send_utc": [t0, t0 + pd.Timedelta(minutes=30)],
            "mid_send": [100.0, 100.5],
        }
    )
    fills = pd.DataFrame(
        {
            "rfq_id": [1, 2],
            "order_id": [1, 1],
            "side": [1, 1],
            "qty_mm": [5.0, 5.0],
            "mid_send": [100.0, 100.5],
            "exec_px": [100.02, 100.52],
        }
    )
    out = costs.parent_shortfall(cfg, orders, rfq, fills, sec).iloc[0]
    assert out["delay_cost_bp"] == pytest.approx(0.0, abs=1e-9)
    assert out["timing_cost_bp"] == pytest.approx(25.0, abs=1e-9)
    assert out["execution_cost_bp"] == pytest.approx(2.0, abs=1e-9)
    assert out["is_filled_bp"] == pytest.approx(27.0, abs=1e-9)
    assert out["is_bp"] == pytest.approx(27.0, abs=1e-9)


def test_partial_fill_adds_opportunity_cost(cfg, inputs):
    sec = inputs.securities
    cusip = sec[sec["tenor"] == 10]["cusip"].iloc[0]
    t0 = local_to_utc(date(2025, 3, 5), time(10, 0), NY)
    orders = pd.DataFrame(
        {
            "order_id": [1],
            "ts_create_utc": [t0],
            "cusip": [cusip],
            "tenor": [10],
            "run_status": ["OTR"],
            "side": [-1],
            "qty_mm": [20.0],
            "decision_mid": [100.0],
            "end_mid": [99.0],
        }
    )
    rfq = pd.DataFrame({"rfq_id": [1], "order_id": [1], "ts_send_utc": [t0], "mid_send": [99.9]})
    fills = pd.DataFrame(
        {
            "rfq_id": [1],
            "order_id": [1],
            "side": [-1],
            "qty_mm": [10.0],
            "mid_send": [99.9],
            "exec_px": [99.88],
        }
    )
    out = costs.parent_shortfall(cfg, orders, rfq, fills, sec).iloc[0]
    assert out["is_filled_bp"] == pytest.approx(12.0, abs=1e-9)
    assert out["opportunity_cost_bp"] == pytest.approx(0.5 * 100.0, abs=1e-9)
    assert out["is_bp"] == pytest.approx(0.5 * 12.0 + 50.0, abs=1e-9)
