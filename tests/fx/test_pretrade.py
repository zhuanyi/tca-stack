"""The pre-trade estimate."""

from __future__ import annotations

import numpy as np
import pytest

from tca.fx.pretrade import duration_factor


def test_duration_factor_is_one_at_the_reference_duration(cfg):
    assert duration_factor(30 * 60, cfg) == pytest.approx(1.0)


def test_duration_factor_follows_the_quarter_power_and_clips(cfg):
    assert duration_factor(7.5 * 60, cfg) == pytest.approx(4.0**0.25)
    assert duration_factor(0.0, cfg) == 2.0
    assert duration_factor(24 * 3600, cfg) == 0.5


def test_estimate_is_computed_before_replay_and_positive(sim):
    assert sim.orders["pretrade_est_bp"].notna().all()
    assert (sim.orders["pretrade_est_bp"] > 0).all()


def test_estimate_rises_with_size_within_a_pair_and_style(sim):
    sub = sim.orders[(sim.orders["pair"] == "EURUSD") & (sim.orders["style"] == "IS")]
    small = sub[sub["notional_usd"] < 20e6]["pretrade_est_bp"].mean()
    large = sub[sub["notional_usd"] > 60e6]["pretrade_est_bp"].mean()
    if not (np.isnan(small) or np.isnan(large)):
        assert large > small
