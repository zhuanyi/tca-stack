"""Benchmarks read off the tape: WM/R fix proxy and time-weighted mid."""

from __future__ import annotations

import numpy as np

from tca.fx.ticks import US_PER_S, TickDay


def fix_benchmark(day: TickDay, window_us: tuple[int, int]) -> float:
    """Proxy for the WM/R fix: median of the 1-second tape mids across the fixing window.

    The official methodology uses trade and order snapshots; this proxy uses quotes only.
    ``window_us`` is ``(start, end)`` in epoch microseconds, end exclusive.
    """
    start, end = window_us
    grid = np.arange(start, end, US_PER_S, dtype=np.int64)
    return float(np.median(day.mid[day.index_at(grid)]))


def twap_mid(day: TickDay, start_us: int, end_us: int) -> float:
    """Time-weighted mean of the tape mid over ``[start_us, end_us)``.

    The quote in force at ``start_us`` is carried in from the last tick before the window.
    """
    first = int(day.index_at(start_us))
    last = int(day.index_at(end_us))
    knots = np.concatenate(([start_us], day.t_us[first + 1 : last + 1]))
    ends = np.concatenate((knots[1:], [end_us]))
    weights = (ends - knots).astype(float)
    if weights.sum() <= 0:
        return float(day.mid[first])
    return float(np.average(day.mid[first : last + 1], weights=weights))
