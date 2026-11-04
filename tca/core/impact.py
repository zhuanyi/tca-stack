"""Square-root market impact and a decaying impact state shared by both simulators."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def sqrt_impact(q, v, y: float, sigma):
    """Square-root impact ``y * sigma * sqrt(q / v)`` in the units of ``sigma``.

    ``q`` is the order size and ``v`` the reference volume in the same units; ``y`` is the
    dimensionless impact coefficient.
    """
    return y * sigma * np.sqrt(q / v)


@dataclass
class ImpactState:
    """Impact on the price path: a permanent part plus a transient part decaying exponentially.

    Each ``add`` splits a signed impact (price units) into ``perm_frac`` permanent and the
    remainder transient; the transient halves every ``half_life_s`` seconds. Times are seconds
    on any clock that never runs backwards.
    """

    perm_frac: float
    half_life_s: float
    _permanent: float = field(default=0.0, init=False, repr=False)
    _transient: float = field(default=0.0, init=False, repr=False)
    _t_s: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.perm_frac <= 1.0:
            raise ValueError("perm_frac must be in [0, 1]")
        if self.half_life_s <= 0.0:
            raise ValueError("half_life_s must be positive")

    def _decay(self, t_s):
        return 0.5 ** ((t_s - self._t_s) / self.half_life_s)

    def _check(self, t_s) -> None:
        if np.any(np.asarray(t_s) < self._t_s):
            raise ValueError(f"time is earlier than the last update at {self._t_s}")

    def add(self, t_s: float, signed_impact: float) -> None:
        """Record an impact of ``signed_impact`` (price units) at time ``t_s`` seconds."""
        self._check(t_s)
        self._transient = (
            self._transient * self._decay(t_s) + (1.0 - self.perm_frac) * signed_impact
        )
        self._permanent += self.perm_frac * signed_impact
        self._t_s = float(t_s)

    def value(self, t_s: float) -> float:
        """Total impact in price units at ``t_s``, which must not precede the last ``add``."""
        self._check(t_s)
        return self._permanent + self._transient * float(self._decay(t_s))

    def value_at(self, t_s: np.ndarray) -> np.ndarray:
        """Total impact at each time in ``t_s``, assuming no further ``add``."""
        t = np.asarray(t_s, dtype=float)
        self._check(t)
        return self._permanent + self._transient * self._decay(t)

    def reset(self) -> None:
        """Clear all impact and rewind the clock to zero."""
        self._permanent = 0.0
        self._transient = 0.0
        self._t_s = 0.0
