"""FX fill-level decomposition agrees with the core parent decomposition."""

import pandas as pd
import pytest

from tca.core.conventions import decompose_parent
from tca.fx.analysis import _decomposition_terms


def test_fx_terms_match_decompose_parent():
    orders = pd.DataFrame(
        {
            "order_id": [1, 2],
            "side": [1, -1],
            "decision_mid": [1.1000, 1.2500],
            "first_decision_mid": [1.1002, 1.2497],
        }
    )
    fills = pd.DataFrame(
        {
            "order_id": [1, 1, 2],
            "qty_usd": [1e6, 3e6, 2e6],
            "px": [1.1006, 1.1010, 1.2490],
            "mid_eff_at_fill": [1.1004, 1.1008, 1.2492],
            "mid_tape_at_fill": [1.1003, 1.1007, 1.2494],
            "intent": ["AGGRESSIVE", "PASSIVE", "AGGRESSIVE"],
            "impact_state_at_fill": [0.0, 0.0, 0.0],
        }
    )
    terms = _decomposition_terms(fills, orders).groupby("order_id").sum()
    for o in orders.itertuples():
        f = fills[fills["order_id"] == o.order_id]
        r = decompose_parent(
            o.side,
            o.decision_mid,
            o.first_decision_mid,
            f["mid_tape_at_fill"].to_numpy(),
            f["qty_usd"].to_numpy(),
            f["px"].to_numpy(),
            float(f["qty_usd"].sum()),
            o.decision_mid,
        )
        t = terms.loc[o.order_id]
        assert t["delay_bp"] == pytest.approx(r.delay_bp, abs=1e-9)
        assert t["timing_bp"] == pytest.approx(r.timing_bp, abs=1e-9)
        assert t["spread_bp"] + t["impact_bp"] == pytest.approx(r.execution_bp, abs=1e-9)
