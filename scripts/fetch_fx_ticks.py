#!/usr/bin/env python3
"""Download Dukascopy historical tick files, resumable per hour.

Access policy (a sustained 5xx is treated the same way):
one request at a time, at least one second between requests, an honest
User-Agent, no proxies, no IP rotation, no parallelism. On HTTP 403/429, or a
5xx that persists, the script backs off 60 s, then doubles, for at most 4
attempts; if the hour is still unavailable the whole run stops with exit code 3
and a single "REFUSED <status> at <pair> <date> <hour>" line.

Layout under $TCA_DATA_RAW/fx:
    {PAIR}/{YYYY-MM-DD}/{HH}.parquet   decoded hour (written atomically)
    {PAIR}/{YYYY-MM-DD}/{HH}.empty     marker for an hour with no ticks
    {PAIR}/{YYYY-MM-DD}.parquet        whole day, written when the day is complete
The hourly folder is removed once the day file exists, so a restart skips it.

With --from-dir the same pipeline reads already-downloaded files from a local folder laid out
like the URL path, {PAIR}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5 with a zero-based month, and makes no
network request. A missing file is an hour without ticks.

Usage:
    scripts/fetch_fx_ticks.py --pairs EURUSD,USDJPY,GBPUSD,AUDUSD \
        --start 2025-03-03 --end 2025-05-30 [--from-dir /path/to/bi5]
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd

from tca.core.storage import raw_dir
from tca.fx.bi5 import parse_bi5

USER_AGENT = "tca-stack-research/0.1"
BASE_URL = "https://datafeed.dukascopy.com/datafeed/{pair}/{year:04d}/{month0:02d}/{day:02d}/{hour:02d}h_ticks.bi5"
MIN_INTERVAL_S = 1.0
SERVER_UNAVAILABLE_STATUSES = (403, 429, 500, 502, 503, 504)
REFUSAL_BACKOFF_START_S = 60.0
MAX_ATTEMPTS = 4
NETWORK_BACKOFF_START_S = 10.0
MAX_NETWORK_ATTEMPTS = 6
REQUEST_TIMEOUT_S = 60.0
PROGRESS_EVERY = 50

EXIT_REFUSED = 3
EXIT_ERROR = 2


class Refused(Exception):
    """The provider refused this hour after the allowed attempts."""

    def __init__(self, status: int) -> None:
        super().__init__(f"refused with HTTP {status}")
        self.status = status


def market_hours(d: date) -> list[int]:
    """UTC hours in which the FX market is open on ``d`` (fixed weekend rule)."""
    weekday = d.weekday()  # Monday = 0, Sunday = 6
    if weekday == 5:  # Saturday: closed all day
        return []
    if weekday == 4:  # Friday: market closes at 22:00 UTC
        return list(range(22))
    if weekday == 6:  # Sunday: market opens at 21:00 UTC
        return [21, 22, 23]
    return list(range(24))


def hour_url(pair: str, d: date, hour: int) -> str:
    return BASE_URL.format(pair=pair, year=d.year, month0=d.month - 1, day=d.day, hour=hour)


def local_hour_path(root: Path, pair: str, d: date, hour: int) -> Path:
    """Where ``hour`` of ``d`` sits under a local folder laid out like the download URL."""
    return (
        root
        / pair
        / f"{d.year:04d}"
        / f"{d.month - 1:02d}"
        / f"{d.day:02d}"
        / f"{hour:02d}h_ticks.bi5"
    )


def atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_hour_parquet(path: Path, df: pd.DataFrame) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, engine="pyarrow", index=False, compression="zstd")
    os.replace(tmp, path)


class Fetcher:
    def __init__(self, root: Path, from_dir: Path | None = None) -> None:
        self.root = root
        self.from_dir = from_dir
        self.client = (
            None
            if from_dir is not None
            else httpx.Client(
                headers={"User-Agent": USER_AGENT},
                timeout=httpx.Timeout(REQUEST_TIMEOUT_S, connect=15.0),
                follow_redirects=True,
            )
        )
        self._last_request = 0.0
        self.done = 0
        self.empty = 0
        self.refused = 0
        self.errors = 0
        self.not_found = 0
        self.processed = 0
        self.total = 0
        self.current: tuple[str, date, int] = ("", date.min, 0)
        self.started = time.monotonic()

    # -- polite access -----------------------------------------------------
    def _throttle(self) -> None:
        wait = MIN_INTERVAL_S - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _backoff(self, seconds: float, why: str, pair: str, d: date, hour: int) -> None:
        log(f"backoff {seconds:.0f}s ({why}) {pair} {d} {hour:02d}h")
        time.sleep(seconds)

    def fetch_hour(self, pair: str, d: date, hour: int) -> bytes:
        """GET one hourly file, or read it from the local folder. Returns b'' for no ticks."""
        if self.from_dir is not None:
            path = local_hour_path(self.from_dir, pair, d, hour)
            return path.read_bytes() if path.is_file() else b""
        url = hour_url(pair, d, hour)
        refusals = 0
        network_errors = 0
        while True:
            self._throttle()
            try:
                resp = self.client.get(url)
            except httpx.HTTPError as exc:
                network_errors += 1
                self.errors += 1
                if network_errors >= MAX_NETWORK_ATTEMPTS:
                    raise RuntimeError(f"network failure for {url}: {exc}") from exc
                self._backoff(
                    NETWORK_BACKOFF_START_S * (2 ** (network_errors - 1)),
                    f"network {type(exc).__name__}",
                    pair,
                    d,
                    hour,
                )
                continue

            if resp.status_code == 200:
                return resp.content
            if resp.status_code == 404:
                # An hour the provider does not have: kept as an .empty marker, but counted so
                # gaps in the data are visible in the report.
                self.not_found += 1
                return b""
            if resp.status_code in SERVER_UNAVAILABLE_STATUSES:
                # 429/403 are explicit refusals; a sustained 5xx (the provider
                # returns 503 when no backend is available) is treated the same
                # way: stop the run rather than hammer the service.
                refusals += 1
                self.refused += 1
                if refusals >= MAX_ATTEMPTS:
                    raise Refused(resp.status_code)
                self._backoff(
                    REFUSAL_BACKOFF_START_S * (2 ** (refusals - 1)),
                    f"HTTP {resp.status_code}",
                    pair,
                    d,
                    hour,
                )
                continue
            self.errors += 1
            raise RuntimeError(f"HTTP {resp.status_code} for {url}")

    # -- storage -----------------------------------------------------------
    def day_dir(self, pair: str, d: date) -> Path:
        return self.root / pair / d.isoformat()

    def day_file(self, pair: str, d: date) -> Path:
        return self.root / pair / f"{d.isoformat()}.parquet"

    def cached(self, pair: str, d: date, hour: int) -> str | None:
        """'done', 'empty' or None (hour not cached yet)."""
        folder = self.day_dir(pair, d)
        if (folder / f"{hour:02d}.parquet").exists():
            return "done"
        if (folder / f"{hour:02d}.empty").exists():
            return "empty"
        return None

    def complete_day(self, pair: str, d: date) -> None:
        folder = self.day_dir(pair, d)
        hours = market_hours(d)
        frames = []
        for hour in hours:
            path = folder / f"{hour:02d}.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
        if frames:
            day = pd.concat(frames, ignore_index=True).sort_values("ts_utc", ignore_index=True)
            out = self.day_file(pair, d)
            tmp = out.with_name(out.name + ".tmp")
            day.to_parquet(tmp, engine="pyarrow", index=False, compression="zstd")
            os.replace(tmp, out)
        else:
            # A day with no ticks at all still needs a marker so it is not refetched.
            out = self.day_file(pair, d)
            atomic_write_bytes(out.with_suffix(".empty"), b"no ticks\n")
        shutil.rmtree(folder, ignore_errors=True)

    # -- main loop ---------------------------------------------------------
    def process_hour(self, pair: str, d: date, hour: int) -> None:
        status = self.cached(pair, d, hour)
        if status == "done":
            self.done += 1
            self.processed += 1
            return
        if status == "empty":
            self.empty += 1
            self.processed += 1
            return

        folder = self.day_dir(pair, d)
        folder.mkdir(parents=True, exist_ok=True)
        self.current = (pair, d, hour)
        raw = self.fetch_hour(pair, d, hour)
        hour_start = datetime(d.year, d.month, d.day, hour, tzinfo=UTC)
        df = parse_bi5(raw, hour_start, pair)

        if df.empty:
            atomic_write_bytes(folder / f"{hour:02d}.empty", b"empty hour\n")
            self.empty += 1
        else:
            write_hour_parquet(folder / f"{hour:02d}.parquet", df)
            self.done += 1
        self.processed += 1

    def process_day(self, pair: str, d: date) -> None:
        hours = market_hours(d)
        if not hours:
            return
        if self.day_file(pair, d).exists() or self.day_file(pair, d).with_suffix(".empty").exists():
            self.done += len(hours)
            self.processed += len(hours)
            return

        for hour in hours:
            self.process_hour(pair, d, hour)
            if self.processed % PROGRESS_EVERY == 0:
                self.report()

        self.complete_day(pair, d)
        log(f"day complete {pair} {d} ({len(hours)} hours)")

    def report(self) -> None:
        elapsed = time.monotonic() - self.started
        rate = self.processed / elapsed if elapsed > 0 else 0.0
        remaining = max(self.total - self.processed, 0)
        eta = remaining / rate if rate > 0 else float("nan")
        log(
            f"files={self.processed}/{self.total} done={self.done} empty={self.empty} "
            f"not_found={self.not_found} refused={self.refused} errors={self.errors} "
            f"elapsed={fmt_duration(elapsed)} eta={fmt_duration(eta)}"
        )


def fmt_duration(seconds: float) -> str:
    if math.isnan(seconds) or seconds < 0:
        return "n/a"
    minutes = round(seconds / 60.0)
    return f"{minutes // 60}h{minutes % 60:02d}m"


def log(message: str) -> None:
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{stamp}] {message}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Dukascopy hourly tick files.")
    parser.add_argument("--pairs", required=True, help="comma-separated, e.g. EURUSD,USDJPY")
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument(
        "--from-dir", type=Path, help="read downloaded .bi5 files from this folder; no network"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    pairs = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    if not pairs:
        log("ERROR no pairs given")
        return EXIT_ERROR
    if args.end < args.start:
        log("ERROR --end is before --start")
        return EXIT_ERROR

    raw_root = raw_dir() / "fx"
    days = [args.start + timedelta(days=i) for i in range((args.end - args.start).days + 1)]

    fetcher = Fetcher(raw_root, args.from_dir)
    fetcher.total = sum(len(market_hours(d)) for _ in pairs for d in days)
    log(
        f"start pairs={'/'.join(pairs)} {args.start}..{args.end} "
        f"hours={fetcher.total} raw={raw_root}"
    )

    try:
        for pair in pairs:
            for d in days:
                fetcher.process_day(pair, d)
    except Refused as exc:
        # The current hour is identified by the last line context; print the strict form.
        log(
            f"REFUSED {exc.status} at {fetcher.current[0]} {fetcher.current[1]} "
            f"{fetcher.current[2]:02d}"
        )
        log("stopping: provider refused access; cached hours are kept")
        return EXIT_REFUSED
    except KeyboardInterrupt:
        log("interrupted; cached hours are kept")
        return 130
    except Exception as exc:  # report and stop, do not hide it
        log(f"ERROR {type(exc).__name__}: {exc}")
        fetcher.report()
        return EXIT_ERROR

    fetcher.report()
    log("finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
