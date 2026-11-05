"""Tick access: one pair-day of quotes as numpy arrays, from cached Dukascopy files or a fixture."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from tca.core.profile import is_ci
from tca.fx import paths

US_PER_S = 1_000_000
US_PER_DAY = 86_400 * US_PER_S


def day_start_us(day: date) -> int:
    """Microseconds since the epoch of 00:00 UTC on ``day``."""
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp()) * US_PER_S


def to_us(ts: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Microseconds since the epoch of a timezone-aware timestamp column."""
    return pd.DatetimeIndex(ts).as_unit("us").asi8


def from_us(us: np.ndarray) -> pd.DatetimeIndex:
    """Timezone-aware UTC timestamps (microsecond precision) from epoch microseconds."""
    return pd.DatetimeIndex(pd.to_datetime(np.asarray(us, dtype=np.int64), unit="us", utc=True))


@dataclass(frozen=True)
class TickDay:
    """All quotes of one pair on one UTC day, sorted by time.

    ``t_us`` is epoch microseconds; ``bid`` and ``ask`` are prices in quote units.
    """

    pair: str
    day: date
    t_us: np.ndarray
    bid: np.ndarray
    ask: np.ndarray

    @property
    def mid(self) -> np.ndarray:
        """Mid price ``(bid + ask) / 2`` of every tick."""
        return (self.bid + self.ask) / 2.0

    def index_at(self, t_us: int | np.ndarray) -> int | np.ndarray:
        """Index of the last tick at or before ``t_us``; ``0`` if it precedes the first tick."""
        idx = np.searchsorted(self.t_us, t_us, side="right") - 1
        return np.maximum(idx, 0)

    def frame(self) -> pd.DataFrame:
        """The ticks as a frame with ``ts_utc``, ``bid`` and ``ask``."""
        return pd.DataFrame({"ts_utc": from_us(self.t_us), "bid": self.bid, "ask": self.ask})


def tickday_from_frame(frame: pd.DataFrame, pair: str, day: date) -> TickDay:
    """Build a :class:`TickDay` from a frame with ``ts_utc``, ``bid`` and ``ask`` columns."""
    ordered = frame.sort_values("ts_utc", kind="stable")
    return TickDay(
        pair=pair,
        day=day,
        t_us=to_us(ordered["ts_utc"]),
        bid=ordered["bid"].to_numpy(dtype=float),
        ask=ordered["ask"].to_numpy(dtype=float),
    )


class TickSource(Protocol):
    """Provider of pair-days of ticks."""

    def days(self, pair: str) -> list[date]:
        """Days with ticks for ``pair``, ascending."""
        ...

    def load(self, pair: str, day: date) -> TickDay | None:
        """The ticks of ``pair`` on ``day``, or ``None`` when there are none."""
        ...


class RawTickSource:
    """Day files ``<root>/{pair}/{YYYY-MM-DD}.parquet`` written by ``scripts/fetch_fx_ticks.py``."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else paths.raw_fx_dir()

    def days(self, pair: str) -> list[date]:
        folder = self.root / pair
        return sorted(date.fromisoformat(p.stem) for p in folder.glob("????-??-??.parquet"))

    def load(self, pair: str, day: date) -> TickDay | None:
        path = self.root / pair / f"{day.isoformat()}.parquet"
        if not path.exists():
            return None
        frame = pd.read_parquet(path, columns=["ts_utc", "bid", "ask"])
        return tickday_from_frame(frame, pair, day) if len(frame) else None


class FixtureTickSource:
    """Ticks of every pair in one parquet file with a ``pair`` column (the synthetic fixture)."""

    def __init__(self, path: Path | None = None) -> None:
        frame = pd.read_parquet(Path(path) if path is not None else paths.fixture_path())
        frame["day"] = frame["ts_utc"].dt.date
        self._groups = {
            (pair, day): group.drop(columns=["pair", "day"])
            for (pair, day), group in frame.groupby(["pair", "day"], sort=True)
        }

    def days(self, pair: str) -> list[date]:
        return sorted(day for p, day in self._groups if p == pair)

    def load(self, pair: str, day: date) -> TickDay | None:
        group = self._groups.get((pair, day))
        return tickday_from_frame(group, pair, day) if group is not None else None


def default_source() -> TickSource:
    """The fixture under the ``ci`` profile, cached Dukascopy ticks otherwise."""
    return FixtureTickSource() if is_ci() else RawTickSource()
