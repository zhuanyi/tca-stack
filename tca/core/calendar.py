"""Business-day calendar, session times and the event calendar loader."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

NY = ZoneInfo("America/New_York")
LONDON = ZoneInfo("Europe/London")

_WMR_START = time(15, 57, 30)
_WMR_END = time(16, 2, 30)
_EXTRA_CLOSURES = frozenset({date(2025, 1, 9)})
_EVENT_COLUMNS = ["ts_utc", "type", "tenor", "source"]


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    nxt = date(year + month // 12, month % 12 + 1, 1)
    last = nxt - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Gregorian Easter Sunday (Meeus/Jones/Butcher algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = divmod(b, 4)
    g = (b - (b + 8) // 25 + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month, day = divmod(h + ell - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _observed(d: date) -> date | None:
    """Closure date for a fixed-date holiday: Sunday moves to Monday, Saturday has no closure."""
    if d.weekday() == 5:
        return None
    return d + timedelta(days=1) if d.weekday() == 6 else d


def _closures(year: int) -> frozenset[date]:
    fixed = [date(year, 1, 1), date(year, 6, 19), date(year, 7, 4), date(year, 11, 11)]
    fixed.append(date(year, 12, 25))
    days = {o for d in fixed if (o := _observed(d)) is not None}
    days |= {
        _nth_weekday(year, 1, 0, 3),
        _nth_weekday(year, 2, 0, 3),
        _easter(year) - timedelta(days=2),
        _last_weekday(year, 5, 0),
        _nth_weekday(year, 9, 0, 1),
        _nth_weekday(year, 10, 0, 2),
        _nth_weekday(year, 11, 3, 4),
    }
    return frozenset(days) | {d for d in _EXTRA_CLOSURES if d.year == year}


def is_business_day(d: date) -> bool:
    """True when ``d`` is a weekday not in the US bond-market full-day closure list.

    Closures follow the SIFMA recommended full-day holidays: New Year's Day, Martin Luther King Jr.
    Day, Presidents' Day, Good Friday, Memorial Day, Juneteenth, Independence Day, Labor Day,
    Columbus Day, Veterans Day, Thanksgiving and Christmas, with a Sunday holiday observed on the
    Monday. The national day of mourning on 2025-01-09 is included.
    """
    return d.weekday() < 5 and d not in _closures(d.year)


def business_days(start: date, end: date) -> list[date]:
    """Business days from ``start`` to ``end``, both inclusive."""
    n = (end - start).days + 1
    return [d for i in range(n) if is_business_day(d := start + timedelta(days=i))]


def local_to_utc(d: date, t: time, tz: ZoneInfo) -> datetime:
    """Convert a local wall-clock date and time in ``tz`` to a timezone-aware UTC datetime."""
    return datetime.combine(d, t, tzinfo=tz).astimezone(UTC)


def wmr_window_utc(d: date) -> tuple[datetime, datetime]:
    """Start and end in UTC of the 5-minute fixing window, 15:57:30 to 16:02:30 London time."""
    return local_to_utc(d, _WMR_START, LONDON), local_to_utc(d, _WMR_END, LONDON)


def load_events(path: Path) -> pd.DataFrame:
    """Load a YAML event calendar as columns ``ts_utc`` (UTC), ``type``, ``tenor``, ``source``.

    The file is either a list of mappings or a mapping with an ``events`` list. ``tenor`` and
    ``source`` are optional per event.
    """
    raw = yaml.safe_load(Path(path).read_text())
    rows = raw["events"] if isinstance(raw, dict) else raw
    df = pd.DataFrame(rows).reindex(columns=_EVENT_COLUMNS)
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True).astype("datetime64[us, UTC]")
    return df.sort_values("ts_utc", kind="stable").reset_index(drop=True)
