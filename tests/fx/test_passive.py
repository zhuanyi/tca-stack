"""The passive child model."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from tca.fx.passive import calibrate_lambda_fill, hazard_fill, scan_passive
from tca.fx.ticks import US_PER_S, TickDay


def _tape(bid, ask, step_s=1.0) -> TickDay:
    t = (np.arange(len(bid)) * step_s * US_PER_S).astype("int64")
    return TickDay("EURUSD", date(2025, 3, 5), t, np.array(bid, float), np.array(ask, float))


def _no_impact(t):
    return np.zeros(len(t))


def test_buy_fills_when_the_ask_falls_to_the_limit():
    tape = _tape([1.1000, 1.1000, 1.0998], [1.1002, 1.1002, 1.1000])
    scan = scan_passive(tape, 1, 1.1000, 0, 10 * US_PER_S, _no_impact, tol=1e-6)
    assert scan.cross_idx == 2
    assert scan.cross_us == 2 * US_PER_S
    assert scan.exposure_s == pytest.approx(2.0)


def test_sell_fills_when_the_bid_rises_to_the_limit():
    tape = _tape([1.1000, 1.1000, 1.1002], [1.1002, 1.1002, 1.1004])
    scan = scan_passive(tape, -1, 1.1002, 0, 10 * US_PER_S, _no_impact, tol=1e-6)
    assert scan.cross_idx == 2


def test_no_cross_means_timeout():
    tape = _tape([1.1000] * 5, [1.1002] * 5)
    scan = scan_passive(tape, 1, 1.1000, 0, 5 * US_PER_S, _no_impact, tol=1e-6)
    assert scan.cross_us is None
    assert scan.exposure_s == pytest.approx(5.0)


def test_exposure_counts_only_time_at_the_limit():
    tape = _tape([1.1000, 1.0999, 1.1000, 1.1000], [1.1002, 1.1001, 1.1002, 1.1002])
    scan = scan_passive(tape, 1, 1.1000, 0, 4 * US_PER_S, _no_impact, tol=1e-6)
    assert scan.exposure_s == pytest.approx(3.0)


def test_hazard_fill_waits_for_the_exponential_clock():
    tape = _tape([1.1000] * 10, [1.1002] * 10)
    scan = scan_passive(tape, 1, 1.1000, 0, 10 * US_PER_S, _no_impact, tol=1e-6)
    assert hazard_fill(scan, lambda_fill=0.5, unit_exp=1.0) == (2 * US_PER_S, 1)
    assert hazard_fill(scan, lambda_fill=0.5, unit_exp=100.0) is None
    assert hazard_fill(scan, lambda_fill=0.0, unit_exp=1.0) is None


def test_calibration_hits_the_target_rate():
    rng = np.random.default_rng(3)
    exposure = rng.uniform(1.0, 20.0, 5000)
    crossed = rng.random(5000) < 0.1
    lam = calibrate_lambda_fill(exposure, crossed, 0.4)
    rate = np.mean(np.where(crossed, 1.0, 1.0 - np.exp(-lam * exposure)))
    assert rate == pytest.approx(0.4, abs=1e-6)


def test_calibration_needs_no_hazard_when_crossings_already_reach_the_target():
    assert calibrate_lambda_fill(np.ones(10), np.ones(10, dtype=bool), 0.4) == 0.0


def test_calibration_rejects_an_unreachable_target():
    with pytest.raises(ValueError, match="not reachable"):
        calibrate_lambda_fill(np.zeros(10), np.zeros(10, dtype=bool), 0.4)


def test_calibrated_passive_fill_share_is_near_the_target(sim, cfg):
    kids = sim.children[sim.children["intent"] == "PASSIVE"]
    share = sim.fills["child_id"].isin(kids["child_id"]).sum() / len(kids)
    assert share == pytest.approx(cfg.passive.target_fill_rate, abs=0.12)
