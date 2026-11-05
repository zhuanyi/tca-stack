"""Every action happens at least the decision latency after the tick it acts on."""

from __future__ import annotations

import pandas as pd

LATENCY = pd.Timedelta(milliseconds=20)


def test_child_decisions_follow_their_tick(sim):
    gap = sim.children["ts_decision_utc"] - sim.children["tick_utc"]
    assert (gap >= LATENCY).all()


def test_lp_attempts_follow_their_tick(sim):
    gap = sim.attempts["ts_sent_utc"] - sim.attempts["tick_utc"]
    assert (gap >= LATENCY).all()


def test_aggressive_fills_follow_the_tick_that_priced_them(sim):
    f = sim.fills[sim.fills["intent"] != "PASSIVE"]
    assert ((f["ts_fill_utc"] - f["px_tick_utc"]) >= LATENCY).all()


def test_configured_latency_is_twenty_milliseconds(cfg):
    assert cfg.latency.decision_latency_ms == 20
