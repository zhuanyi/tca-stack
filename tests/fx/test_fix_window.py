"""The WM/R fixing window and the proxy benchmark."""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pytest

from tca.core.calendar import wmr_window_utc
from tca.fx.benchmarks import fix_benchmark, twap_mid
from tca.fx.ticks import US_PER_S, TickDay


def test_window_in_winter_time_is_utc():
    start, end = wmr_window_utc(date(2025, 3, 28))
    assert start == datetime(2025, 3, 28, 15, 57, 30, tzinfo=UTC)
    assert end == datetime(2025, 3, 28, 16, 2, 30, tzinfo=UTC)


def test_window_in_summer_time_is_one_hour_earlier_in_utc():
    start, end = wmr_window_utc(date(2025, 4, 1))
    assert start == datetime(2025, 4, 1, 14, 57, 30, tzinfo=UTC)
    assert end == datetime(2025, 4, 1, 15, 2, 30, tzinfo=UTC)


def _tape(values: list[float], t0: int) -> TickDay:
    t = t0 + np.arange(len(values), dtype="int64") * US_PER_S
    mid = np.array(values)
    return TickDay("EURUSD", date(2025, 3, 28), t, mid - 0.00005, mid + 0.00005)


def test_fix_benchmark_is_median_of_one_second_mids():
    t0 = 1_743_000_000 * US_PER_S
    tape = _tape([1.1, 1.1, 1.1, 1.5, 1.1], t0)
    assert fix_benchmark(tape, (t0, t0 + 5 * US_PER_S)) == pytest.approx(1.1)


def test_twap_mid_weights_quotes_by_time_in_force():
    t0 = 1_743_000_000 * US_PER_S
    tape = TickDay(
        "EURUSD",
        date(2025, 3, 28),
        np.array([t0, t0 + 10 * US_PER_S]),
        np.array([1.0, 2.0]),
        np.array([1.0, 2.0]),
    )
    assert twap_mid(tape, t0, t0 + 40 * US_PER_S) == pytest.approx((10 * 1.0 + 30 * 2.0) / 40)
    assert twap_mid(tape, t0 + 5 * US_PER_S, t0 + 20 * US_PER_S) == pytest.approx(
        (5 * 1.0 + 10 * 2.0) / 15
    )
