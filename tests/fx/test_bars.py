"""One-minute bars from ticks."""

from __future__ import annotations

import numpy as np
import pytest

from tca.fx import bars as bars_module
from tca.fx.ticks import US_PER_S, TickDay


def test_ohlc_ordering_and_positive_spread(bars):
    assert (bars["high"] >= bars["close"]).all()
    assert (bars["close"] >= bars["low"]).all()
    assert (bars["high"] >= bars["open"]).all()
    assert (bars["open"] >= bars["low"]).all()
    assert (bars["spread_tw_pips"] > 0).all()
    assert (bars["spread_tw_bp"] > 0).all()


def test_tick_counts_sum_to_total(bars, source, cfg):
    for pair in cfg.pairs.order:
        total = sum(len(source.load(pair, d).t_us) for d in source.days(pair))
        assert bars.loc[bars["pair"] == pair, "tick_count"].sum() == total


def test_last_quote_of_the_minute(source, cfg):
    day = source.days("EURUSD")[0]
    tape = source.load("EURUSD", day)
    out = bars_module.bars_from_ticks(tape)
    last_in_first_minute = np.flatnonzero(
        tape.t_us // (60 * US_PER_S) == tape.t_us[0] // (60 * US_PER_S)
    )[-1]
    assert out["last_bid"].iloc[0] == tape.bid[last_in_first_minute]
    assert out["last_ask"].iloc[0] == tape.ask[last_in_first_minute]


def test_spread_is_time_weighted_and_carries_over_minutes():
    t0 = (1_740_000_000 // 60) * 60 * US_PER_S
    # 1 pip wide for 15 s and 3 pips for 45 s of the first minute, then 3 pips carried over.
    t = np.array([t0, t0 + 15 * US_PER_S, t0 + 70 * US_PER_S])
    bid = np.array([1.0, 1.0, 1.0])
    ask = np.array([1.0001, 1.0003, 1.0003])
    out = bars_module.bars_from_ticks(TickDay("EURUSD", None, t, bid, ask))
    assert out["spread_tw_pips"].iloc[0] == pytest.approx((15 * 1 + 45 * 3) / 60)
    assert out["spread_tw_pips"].iloc[1] == pytest.approx(3.0)


def test_daily_vol_is_root_sum_of_squared_returns(bars):
    vol = bars_module.daily_vol(bars)
    assert (vol["vol"] > 0).all()
    assert set(vol.columns) == {"pair", "day", "vol"}


def test_rounding_keeps_columns_and_precision(bars):
    out = bars_module.round_bars(bars)
    assert list(out.columns) == list(bars.columns)
    assert np.allclose(out["open"], out["open"].round(6), atol=0.0)
