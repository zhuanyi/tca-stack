"""Passive child model: resting limit order, fill by tape cross or by a hazard at the touch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import optimize

from tca.fx.ticks import US_PER_S, TickDay

_LAMBDA_MAX = 1e3
_EPS = 1e-12


@dataclass(frozen=True)
class PassiveScan:
    """The resting order's path between posting and its end or first cross.

    ``knots_us`` are the instants the quote changed (the first is the posting time), ``duration_s``
    how long each state lasted, ``at_limit`` whether the order's own side of the book sat at the
    limit in that state, and ``tick_idx`` the tick that set the state. ``cross_us`` and
    ``cross_idx`` give the tick that crossed the limit, if any.
    """

    knots_us: np.ndarray
    duration_s: np.ndarray
    at_limit: np.ndarray
    tick_idx: np.ndarray
    cross_us: int | None
    cross_idx: int | None

    @property
    def exposure_s(self) -> float:
        """Seconds the order sat at the limit before it crossed or timed out."""
        return float(self.duration_s[self.at_limit].sum())


def scan_passive(
    day: TickDay,
    side: int,
    limit: float,
    start_us: int,
    end_us: int,
    impact_at: Callable[[np.ndarray], np.ndarray],
    tol: float,
) -> PassiveScan:
    """Walk the tape from ``start_us`` to ``end_us`` for a resting limit order.

    Args:
        day: the tape.
        side: +1 for a buy resting at the effective bid, -1 for a sell at the effective ask.
        limit: limit price in quote units.
        start_us: instant the order becomes live (epoch microseconds).
        end_us: instant it is cancelled.
        impact_at: effective-market impact in quote units at an array of epoch microseconds.
        tol: tolerance in quote units for "the book sits at the limit" (half a price grid step).

    A buy crosses when the effective ask falls to the limit or below; a sell when the effective bid
    rises to it or above.
    """
    first = int(day.index_at(start_us))
    stop = max(int(np.searchsorted(day.t_us, end_us, side="right")), first + 1)
    idx = np.arange(first, stop)
    knots = np.concatenate(([start_us], day.t_us[first + 1 : stop]))
    ends = np.concatenate((knots[1:], [end_us]))
    impact = impact_at(knots)
    bid = day.bid[idx] + impact
    ask = day.ask[idx] + impact
    if side > 0:
        crossed = ask <= limit + _EPS
        own_side = bid
    else:
        crossed = bid >= limit - _EPS
        own_side = ask
    at_limit = np.abs(own_side - limit) < tol
    duration = (ends - knots) / US_PER_S
    if crossed.any():
        k = int(np.argmax(crossed))
        return PassiveScan(
            knots[:k], duration[:k], at_limit[:k], idx[:k], int(knots[k]), int(idx[k])
        )
    return PassiveScan(knots, duration, at_limit, idx, None, None)


def hazard_fill(scan: PassiveScan, lambda_fill: float, unit_exp: float) -> tuple[int, int] | None:
    """Time and tick of a hazard fill while the order sits at the limit.

    The order fills when its accumulated time at the limit reaches ``unit_exp / lambda_fill``,
    where ``unit_exp`` is a draw from the unit exponential distribution. Returns
    ``(fill_us, tick_idx)`` or ``None`` when that time is not reached before the scan ends.
    """
    if lambda_fill <= 0.0 or len(scan.knots_us) == 0:
        return None
    cumulative = np.cumsum(np.where(scan.at_limit, scan.duration_s, 0.0))
    threshold = unit_exp / lambda_fill
    k = int(np.searchsorted(cumulative, threshold, side="left"))
    if k >= len(cumulative):
        return None
    before = cumulative[k - 1] if k else 0.0
    return int(scan.knots_us[k] + (threshold - before) * US_PER_S), int(scan.tick_idx[k])


def calibrate_lambda_fill(exposures_s: np.ndarray, crossed: np.ndarray, target: float) -> float:
    """Hazard per second that makes the expected passive fill rate equal ``target``.

    Args:
        exposures_s: for each sampled posting, seconds at the limit before a cross or timeout.
        crossed: whether the posting crossed the limit (and so fills regardless of the hazard).
        target: desired share of postings that fill.

    A posting fills with probability 1 if it crossed, else ``1 - exp(-lambda * exposure)``; the
    mean is increasing in ``lambda`` and is solved by bracketing. When the postings that cross
    on their own already fill at least ``target``, no hazard is needed and ``0.0`` is returned;
    the realised fill rate is then above the target.

    Raises:
        ValueError: ``target`` is above what any hazard reaches.
    """

    def rate(lam: float) -> float:
        return float(np.mean(np.where(crossed, 1.0, 1.0 - np.exp(-lam * exposures_s))))

    if rate(0.0) >= target:
        return 0.0
    if rate(_LAMBDA_MAX) < target:
        raise ValueError("target fill rate is not reachable with the sampled exposures")
    return float(optimize.brentq(lambda lam: rate(lam) - target, 0.0, _LAMBDA_MAX))
