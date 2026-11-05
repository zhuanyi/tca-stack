"""UST parent_shortfall agrees with the core decompose_parent to 1e-9."""

import numpy as np
import pandas as pd
import pytest

from tca.core.conventions import decompose_parent
from tca.ust import costs

TERMS = [
    ("delay_bp", "delay_cost_bp"),
    ("timing_bp", "timing_cost_bp"),
    ("execution_bp", "execution_cost_bp"),
    ("opportunity_bp", "opportunity_cost_bp"),
    ("is_filled_bp", "is_filled_bp"),
    ("is_bp", "is_bp"),
    ("filled_frac", "filled_frac"),
]


@pytest.fixture(autouse=True)
def _stub_dv01(monkeypatch):
    monkeypatch.setattr(costs, "_add_dv01", lambda cfg, parent, securities: parent)


def _check(orders, rfq, fills):
    out = costs.parent_shortfall(None, orders, rfq, fills, None).set_index("order_id")
    rfq_sorted = rfq.sort_values(["order_id", "ts_send_utc", "rfq_id"], kind="stable")
    for o in orders.itertuples():
        f = fills[fills["order_id"] == o.order_id]
        sends = rfq_sorted[rfq_sorted["order_id"] == o.order_id]["mid_send"]
        first = float(sends.iloc[0]) if len(sends) else float("nan")
        r = decompose_parent(
            int(o.side),
            o.decision_mid,
            first,
            f["mid_send"].to_numpy(),
            f["qty_mm"].to_numpy(),
            f["exec_px"].to_numpy(),
            o.qty_mm,
            o.end_mid,
        )
        u = out.loc[o.order_id]
        for core_name, ust_name in TERMS:
            x, y = getattr(r, core_name), u[ust_name]
            if np.isnan(x):
                assert np.isnan(y), (o.order_id, core_name)
            else:
                assert y == pytest.approx(x, abs=1e-9), (o.order_id, core_name)


def test_hand_built_parents():
    orders = pd.DataFrame(
        {
            "order_id": [1, 2, 3],
            "side": [1, -1, 1],
            "decision_mid": [100.0, 99.5, 100.2],
            "qty_mm": [4.0, 2.0, 3.0],
            "end_mid": [101.0, 99.0, 100.0],
        }
    )
    # Order 1: partial fill with two RFQs. Order 2: a sell. Order 3: unfilled, no RFQ.
    rfq = pd.DataFrame(
        {
            "order_id": [1, 1, 2, 2],
            "rfq_id": [1, 2, 3, 4],
            "ts_send_utc": pd.to_datetime(
                ["2025-01-01 10:00", "2025-01-01 10:01", "2025-01-01 10:00", "2025-01-01 10:02"]
            ),
            "mid_send": [100.1, 100.5, 99.4, 99.1],
        }
    )
    fills = pd.DataFrame(
        {
            "order_id": [1, 1, 2],
            "side": [1, 1, -1],
            "qty_mm": [1.0, 1.0, 2.0],
            "mid_send": [100.1, 100.5, 99.1],
            "exec_px": [100.12, 100.52, 99.08],
        }
    )
    _check(orders, rfq, fills)


def test_seeded_random_parents():
    rng = np.random.default_rng(20261104)
    order_rows, rfq_rows, fill_rows = [], [], []
    rfq_id = 0
    for oid in range(1, 201):
        side = int(rng.choice([-1, 1]))
        dec = float(rng.uniform(90, 110))
        qty = float(rng.uniform(1, 50))
        order_rows.append(
            {
                "order_id": oid,
                "side": side,
                "decision_mid": dec,
                "qty_mm": qty,
                "end_mid": dec + float(rng.normal(0, 0.5)),
            }
        )
        n_rfq = int(rng.integers(0, 4))
        sends = []
        for k in range(n_rfq):
            rfq_id += 1
            m = dec + float(rng.normal(0, 0.3))
            sends.append(m)
            rfq_rows.append(
                {
                    "order_id": oid,
                    "rfq_id": rfq_id,
                    "ts_send_utc": pd.Timestamp("2025-01-01 10:00") + pd.Timedelta(minutes=k),
                    "mid_send": m,
                }
            )
        if n_rfq == 0:
            continue
        n_fill = int(rng.integers(0, n_rfq + 1))
        shares = (
            rng.dirichlet(np.ones(n_fill)) * qty * float(rng.uniform(0.1, 1.0)) if n_fill else []
        )
        for k in range(n_fill):
            fill_rows.append(
                {
                    "order_id": oid,
                    "side": side,
                    "qty_mm": float(shares[k]),
                    "mid_send": sends[k],
                    "exec_px": sends[k] + side * float(rng.uniform(0, 0.1)),
                }
            )
    orders = pd.DataFrame(order_rows)
    rfq = pd.DataFrame(rfq_rows)
    fills = pd.DataFrame(fill_rows)
    assert len(fills) > 50 and orders["order_id"].nunique() > fills["order_id"].nunique()
    _check(orders, rfq, fills)
