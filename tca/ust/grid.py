"""The one-second New York trading grid shared by the bridge, the market and the RFQ engine."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta

import numpy as np
import pandas as pd

from tca.core.calendar import NY, local_to_utc
from tca.ust.config import SimConfig


def parse_hhmm(text: str) -> time:
    """Parse ``HH:MM`` into a ``datetime.time``."""
    hh, mm = text.split(":")
    return time(int(hh), int(mm))


@dataclass(frozen=True)
class Grid:
    """Seconds since the session start on a business day, from ``start`` to ``end`` inclusive."""

    start: time
    end: time

    @classmethod
    def from_config(cls, cfg: SimConfig) -> Grid:
        return cls(parse_hhmm(cfg.time_grid.start), parse_hhmm(cfg.time_grid.end))

    @property
    def n_seconds(self) -> int:
        """Length of the session in seconds; the grid has ``n_seconds + 1`` points."""
        return _seconds(self.end) - _seconds(self.start)

    def seconds(self) -> np.ndarray:
        """Grid offsets in seconds from the session start."""
        return np.arange(self.n_seconds + 1, dtype=float)

    def offset(self, wall: time) -> float:
        """Seconds from the session start to New York wall-clock time ``wall``."""
        return float(_seconds(wall) - _seconds(self.start))

    def origin_utc(self, d: date) -> datetime:
        """The session start on ``d`` as an aware UTC datetime."""
        return local_to_utc(d, self.start, NY)

    def to_offset(self, d: date, ts: datetime | pd.Timestamp) -> float:
        """Seconds from the session start on ``d`` to the instant ``ts``."""
        return (pd.Timestamp(ts) - pd.Timestamp(self.origin_utc(d))).total_seconds()

    def to_utc(self, d: date, offset_s) -> pd.DatetimeIndex:
        """Aware UTC timestamps (microsecond resolution) for offsets in seconds from the start."""
        origin = pd.Timestamp(self.origin_utc(d).astimezone(UTC))
        us = np.round(np.asarray(offset_s, dtype=float) * 1e6).astype("int64")
        return (origin + pd.to_timedelta(us, unit="us")).astype("datetime64[us, UTC]")

    def index(self, offset_s) -> np.ndarray:
        """Integer grid index at or before each offset, clipped to the session."""
        idx = np.floor(np.asarray(offset_s, dtype=float)).astype(np.int64)
        return np.clip(idx, 0, self.n_seconds)


def _seconds(wall: time) -> int:
    return int(timedelta(hours=wall.hour, minutes=wall.minute, seconds=wall.second).total_seconds())
