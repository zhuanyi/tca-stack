"""Planted broker effects. Each effect is a small function driven by ``config/brokers.yaml``."""

from __future__ import annotations

from tca.fx.config import BrokerCfg, hhmm_to_seconds
from tca.fx.ticks import US_PER_DAY, US_PER_S


def markup_bp(broker: BrokerCfg | None, ts_us: int) -> float:
    """Broker fill markup in bp of mid at ``ts_us``; positive is worse for the client.

    A window in the configuration replaces the base markup while the UTC time of day is inside it.
    Without a broker (risk transfer, fix) there is no markup.
    """
    if broker is None:
        return 0.0
    seconds = (ts_us % US_PER_DAY) // US_PER_S
    for window in broker.markup_windows:
        if hhmm_to_seconds(window.start) <= seconds < hhmm_to_seconds(window.end):
            return window.markup_bp
    return broker.markup_bp


def impact_multiplier(broker: BrokerCfg | None) -> float:
    """Factor on the square-root impact increment of every fill (1.0 without a broker)."""
    return 1.0 if broker is None else broker.impact_multiplier


def post_fill_impact_bp(broker: BrokerCfg | None) -> float:
    """Extra permanent impact in bp of mid, in the trade direction, added after every fill.

    This is the broker's information leakage: its liquidity providers hedge ahead of the rest of
    the order. It is zero without a broker.
    """
    return 0.0 if broker is None else broker.post_fill_impact_bp
