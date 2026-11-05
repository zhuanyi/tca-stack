"""Per-security record of impact events and the effective-mid shift they cause.

Each event adds a signed price shift that is part permanent and part transient; the transient
part halves every ``half_life_s`` seconds. This is the same state as
:class:`tca.core.impact.ImpactState`, kept as an event list so that the shift can be read at
any past time, which the post-trade markouts need. Shifts are in price points per 100 face.
"""

from __future__ import annotations

import numpy as np

_CHUNK = 2_000_000
_LN2 = np.log(2.0)


class Ledger:
    """Impact events of one security on one day, in seconds since the session start."""

    def __init__(self, perm_frac: float, half_life_s: float) -> None:
        self.perm_frac = perm_frac
        self.half_life_s = half_life_s
        self._t = np.empty(64)
        self._v = np.empty(64)
        self._n = 0

    def __len__(self) -> int:
        return self._n

    def add(self, t_s, signed_shift) -> None:
        """Record shifts (price points, positive pushes the price up) at times ``t_s``."""
        t = np.atleast_1d(np.asarray(t_s, dtype=float))
        v = np.broadcast_to(np.asarray(signed_shift, dtype=float), t.shape)
        end = self._n + len(t)
        if end > len(self._t):
            cap = max(end, 2 * len(self._t))
            self._t = np.resize(self._t, cap)
            self._v = np.resize(self._v, cap)
        self._t[self._n : end] = t
        self._v[self._n : end] = v
        self._n = end

    def value(self, t_s) -> np.ndarray:
        """Total shift in price points at each time in ``t_s``, from events at or before it."""
        t = np.atleast_1d(np.asarray(t_s, dtype=float))
        et, ev = self._t[: self._n], self._v[: self._n]
        out = np.empty(len(t))
        step = max(1, _CHUNK // max(self._n, 1))
        for i in range(0, len(t), step):
            dt = t[i : i + step, None] - et[None, :]
            live = dt >= 0.0
            decay = np.where(live, 0.5 ** (np.where(live, dt, 0.0) / self.half_life_s), 0.0)
            kept = np.where(live, 1.0, 0.0)
            out[i : i + step] = self.perm_frac * (kept @ ev) + (1.0 - self.perm_frac) * (decay @ ev)
        return out

    def mean_over(self, start_s, end_s) -> np.ndarray:
        """Time-weighted mean shift over ``[start_s, end_s)``, exact for continuous time.

        Only events before ``end_s`` contribute, so the result never uses later information.
        """
        a = np.atleast_1d(np.asarray(start_s, dtype=float))
        b = np.atleast_1d(np.asarray(end_s, dtype=float))
        et, ev = self._t[: self._n], self._v[: self._n]
        out = np.zeros(len(a))
        tail = self.half_life_s / _LN2
        step = max(1, _CHUNK // max(self._n, 1))
        for i in range(0, len(a), step):
            ai, bi = a[i : i + step, None], b[i : i + step, None]
            lo = np.maximum(ai, et[None, :])
            hi = np.maximum(bi, et[None, :])
            transient = tail * (
                0.5 ** ((lo - et[None, :]) / self.half_life_s)
                - 0.5 ** ((hi - et[None, :]) / self.half_life_s)
            )
            area = self.perm_frac * (hi - lo) + (1.0 - self.perm_frac) * transient
            out[i : i + step] = (area @ ev) / np.maximum(bi[:, 0] - ai[:, 0], 1e-9)
        return out
