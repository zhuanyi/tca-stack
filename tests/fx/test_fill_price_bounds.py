"""Fill prices decompose exactly into the effective touch plus the LP and broker markups.

Impact moves quotes away from the tape and markups can be negative, so a fill can sit inside the
tape touch. The test therefore checks the construction of the price, not a bound against the tape.
"""

from __future__ import annotations

import numpy as np
import pytest

from tca.fx import brokers
from tca.fx.config import load_brokers

TOL = 1e-9


def _joined(sim):
    f = sim.fills.merge(sim.orders[["order_id", "side", "broker"]], on="order_id")
    return f, f["side"].to_numpy()


def test_aggressive_fill_is_effective_touch_plus_markups(sim):
    f, side = _joined(sim)
    aggressive = (f["intent"] != "PASSIVE").to_numpy()
    touch = np.where(side > 0, f["ask_tape"], f["bid_tape"]) + f["impact_state_at_fill"]
    markups = f["lp_markup_px"] + f["broker_markup_px"]
    assert aggressive.sum() > 100
    assert np.abs(f["px"] - (touch + side * markups))[aggressive].max() < TOL


def test_broker_markup_matches_the_configuration(sim):
    f, _ = _joined(sim)
    book = load_brokers()
    wheel = f[f["broker"].notna() & (f["intent"] != "PASSIVE")]
    checked = 0
    for row in wheel.itertuples():
        mid = (row.bid_tape + row.ask_tape) / 2.0
        at_tick = brokers.markup_bp(book[row.broker], row.px_tick_utc.value // 1000)
        at_fill = brokers.markup_bp(book[row.broker], row.ts_fill_utc.value // 1000)
        if at_tick != at_fill:
            continue
        assert row.broker_markup_px == pytest.approx(at_tick / 1e4 * mid, abs=1e-9)
        checked += 1
    assert checked > 100


def test_fills_without_a_broker_carry_no_broker_markup(sim):
    f, _ = _joined(sim)
    assert (f.loc[f["broker"].isna(), "broker_markup_px"] == 0.0).all()


def test_passive_fill_is_at_the_limit_posted_at_the_effective_touch(sim):
    f, _ = _joined(sim)
    kids = sim.children.set_index("child_id")
    passive = f[f["intent"] == "PASSIVE"]
    assert len(passive) > 20
    child = kids.loc[passive["child_id"]]
    sides = passive["side"].to_numpy()
    posted_touch = np.where(sides > 0, child["bid_tape"], child["ask_tape"]) + child["impact_state"]
    assert np.abs(passive["px"].to_numpy() - child["limit_px"].to_numpy()).max() < TOL
    assert np.abs(child["limit_px"].to_numpy() - posted_touch.to_numpy()).max() < TOL


def test_some_fills_sit_inside_the_tape_touch(sim):
    f, side = _joined(sim)
    buys = f[(side > 0) & (f["intent"] != "PASSIVE")]
    assert (buys["px"] < buys["ask_tape"]).any()
