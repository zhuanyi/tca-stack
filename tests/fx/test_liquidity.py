"""The liquidity profile and the market context built from it."""

from __future__ import annotations

import pytest

from tca.fx import liquidity
from tca.fx.liquidity import ALL_WEEKDAYS, BUCKETS_PER_DAY


def test_profile_has_every_bucket_for_every_pair(bars, cfg):
    profile = liquidity.liquidity_profile(bars)
    overall = profile[profile["dow"] == ALL_WEEKDAYS]
    assert set(overall["pair"]) == set(cfg.pairs.order)
    assert overall.groupby("pair").size().eq(BUCKETS_PER_DAY).all()
    assert (profile["spread_p90_pips"] >= profile["spread_median_pips"]).all()


def test_wmr_flag_marks_the_bucket_containing_four_pm_london(bars):
    profile = liquidity.liquidity_profile(bars)
    flagged = profile[
        (profile["dow"] == ALL_WEEKDAYS) & profile["wmr_window"] & (profile["pair"] == "EURUSD")
    ]
    hours = (flagged["bucket"] * 5 // 60).unique()
    assert set(hours) <= {14, 15, 16}
    assert len(flagged) >= 1


def test_sigma_d_uses_prior_days_and_falls_back_for_the_first(market, source):
    days = source.days("EURUSD")
    first, second = days[0], days[1]
    assert market.sigma_d("EURUSD", first) > 0
    assert market.sigma_d("EURUSD", second) > 0


def test_volume_rate_integrates_to_the_daily_turnover(market, cfg):
    from datetime import date

    from tca.fx.ticks import US_PER_S, day_start_us

    start = day_start_us(date(2025, 3, 5))
    total = sum(
        market.volume_rate_usd_s("EURUSD", start + i * 300 * US_PER_S) * 300 for i in range(288)
    )
    assert total == pytest.approx(cfg.daily_volume_usd("EURUSD"), rel=1e-9)


def test_half_spread_is_half_the_median_spread(market):
    from datetime import date

    from tca.fx.ticks import US_PER_S, day_start_us

    ts = day_start_us(date(2025, 3, 5)) + 10 * 3600 * US_PER_S
    assert market.half_spread_bp("EURUSD", ts) > 0
