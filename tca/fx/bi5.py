"""Parser for Dukascopy hourly tick files (``.bi5``).

Format (verified against a live file, see ``docs/fx_calibration.md``): the
downloaded body is an LZMA-alone stream (no end-of-stream marker). The decoded
payload is a sequence of 20-byte big-endian records::

    int32   ms_offset   milliseconds after the start of the hour
    int32   ask         ask price, scaled by price_scale()
    int32   bid         bid price, scaled by price_scale()
    float32 ask_vol     ask-side volume, indicative, in millions
    float32 bid_vol     bid-side volume, indicative, in millions

Note the field order: ask comes before bid.
"""

from __future__ import annotations

import lzma
from datetime import datetime

import numpy as np
import pandas as pd

# Big-endian record layout, one record per tick.
RECORD_DTYPE = np.dtype(
    [
        ("ms_offset", ">i4"),
        ("ask", ">i4"),
        ("bid", ">i4"),
        ("ask_vol", ">f4"),
        ("bid_vol", ">f4"),
    ]
)
RECORD_SIZE = 20  # bytes

COLUMNS = ["ts_utc", "ask", "bid", "ask_vol", "bid_vol"]


def price_scale(pair: str) -> float:
    """Integer-to-price divisor: 1e3 for JPY-quoted pairs, else 1e5."""
    return 1e3 if pair.upper().endswith("JPY") else 1e5


def empty_frame() -> pd.DataFrame:
    """Empty tick frame with the canonical columns and dtypes."""
    return pd.DataFrame(
        {
            "ts_utc": pd.Series([], dtype="datetime64[us, UTC]"),
            "ask": pd.Series([], dtype="float64"),
            "bid": pd.Series([], dtype="float64"),
            "ask_vol": pd.Series([], dtype="float64"),
            "bid_vol": pd.Series([], dtype="float64"),
        }
    )


def parse_bi5(raw: bytes, hour_start_utc: datetime, pair: str) -> pd.DataFrame:
    """Decode one hourly Dukascopy file into a tick frame.

    Args:
        raw: the bytes as downloaded (LZMA-compressed). Empty input is an hour
            with no ticks and returns an empty frame.
        hour_start_utc: the UTC hour the file covers; tick timestamps are this
            instant plus each record's millisecond offset.
        pair: pair symbol, used only for the price scale (JPY quoted or not).

    Returns:
        DataFrame with columns ts_utc (datetime64[us, UTC]), ask, bid, ask_vol,
        bid_vol (all float64), ordered as in the file.

    Raises:
        ValueError: the decoded payload is not a whole number of 20-byte
            records, or ``hour_start_utc`` is timezone-naive.
    """
    if not raw:
        return empty_frame()

    payload = lzma.LZMADecompressor(format=lzma.FORMAT_ALONE).decompress(raw)
    if len(payload) == 0:
        return empty_frame()
    if len(payload) % RECORD_SIZE != 0:
        raise ValueError(
            f"bi5 payload for {pair} {hour_start_utc} is {len(payload)} bytes, "
            f"not a multiple of {RECORD_SIZE}"
        )

    rec = np.frombuffer(payload, dtype=RECORD_DTYPE)

    base = pd.Timestamp(hour_start_utc)
    if base.tzinfo is None:
        raise ValueError("hour_start_utc must be timezone-aware")
    base = base.tz_convert("UTC")

    scale = price_scale(pair)
    ts = (base + pd.to_timedelta(rec["ms_offset"].astype("int64"), unit="ms")).as_unit("us")

    return pd.DataFrame(
        {
            "ts_utc": pd.Series(ts, dtype="datetime64[us, UTC]"),
            "ask": rec["ask"].astype("float64") / scale,
            "bid": rec["bid"].astype("float64") / scale,
            "ask_vol": rec["ask_vol"].astype("float64"),
            "bid_vol": rec["bid_vol"].astype("float64"),
        }
    )
