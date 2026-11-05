"""Leakage pushes the mid in the trade's direction, more so the more dealers lose the RFQ."""

import numpy as np
import pytest

from tca.ust import simulate

SEEDS = (101, 202, 303)


def _drift_5m(cfg, inputs, rfqs, seed: int, n: int, leakage_scale: float) -> np.ndarray:
    """Side-signed change of the effective mid from the award to 5 minutes later, 32nds."""
    over = simulate.Overrides(rfqs=rfqs, n_dealers=n, leakage_scale=leakage_scale)
    t = simulate.run(cfg, seed, over, inputs)
    marks = t["ust_rfq_markouts"].merge(t["ust_rfq"][["rfq_id", "side"]], on="rfq_id")
    return (marks["side"] * (marks["mid_px_5m"] - marks["mid_px_0s"]) * 32.0).to_numpy()


def _gap(cfg, inputs, rfq_lists, leakage_scale: float) -> tuple[float, float]:
    """Mean drift at N=6 minus N=1 pooled over seeds, and its standard error."""
    n1 = np.concatenate([_drift_5m(cfg, inputs, r, s, 1, leakage_scale) for r, s in rfq_lists])
    n6 = np.concatenate([_drift_5m(cfg, inputs, r, s, 6, leakage_scale) for r, s in rfq_lists])
    se = np.hypot(n1.std(ddof=1) / np.sqrt(len(n1)), n6.std(ddof=1) / np.sqrt(len(n6)))
    return float(n6.mean() - n1.mean()), float(se)


@pytest.fixture(scope="module")
def rfq_lists(cfg, inputs, explicit_rfqs):
    days = inputs.business_days(cfg.period.start, cfg.period.end)[:6]
    return [(explicit_rfqs(days, 8, seed=s, sizes=(50.0,)), s) for s in SEEDS]


def test_more_dealers_lose_more_information(cfg, inputs, rfq_lists):
    gap, se = _gap(cfg, inputs, rfq_lists, leakage_scale=1.0)
    assert gap > 2 * se


def test_no_difference_without_leakage(cfg, inputs, rfq_lists):
    gap, se = _gap(cfg, inputs, rfq_lists, leakage_scale=0.0)
    assert abs(gap) < 2 * se
