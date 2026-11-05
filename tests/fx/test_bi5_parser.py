"""Tests for the Dukascopy .bi5 parser, using synthetic bytes (no network)."""

from __future__ import annotations

import lzma
import struct
from datetime import UTC, datetime

import pandas as pd
import pytest

from tca.fx.bi5 import RECORD_SIZE, empty_frame, parse_bi5, price_scale

HOUR = datetime(2025, 3, 3, 10, 0, tzinfo=UTC)


def make_bi5(records: list[tuple[int, int, int, float, float]]) -> bytes:
    """Compress synthetic (ms, ask, bid, ask_vol, bid_vol) records as LZMA alone."""
    payload = b"".join(struct.pack(">iiiff", *r) for r in records)
    return lzma.compress(payload, format=lzma.FORMAT_ALONE)


def test_price_scale_eur_and_jpy() -> None:
    assert price_scale("EURUSD") == 1e5
    assert price_scale("GBPUSD") == 1e5
    assert price_scale("USDJPY") == 1e3
    assert price_scale("usdjpy") == 1e3


def test_eurusd_records_decode_with_correct_scaling() -> None:
    raw = make_bi5([(2212, 110311, 110307, 0.9, 3.6), (5000, 110315, 110310, 1.2, 2.4)])
    df = parse_bi5(raw, HOUR, "EURUSD")

    assert len(df) == 2
    assert list(df.columns) == ["ts_utc", "ask", "bid", "ask_vol", "bid_vol"]
    assert str(df["ts_utc"].dtype) == "datetime64[us, UTC]"
    for col in ("ask", "bid", "ask_vol", "bid_vol"):
        assert df[col].dtype == "float64"

    assert df.loc[0, "ask"] == pytest.approx(1.10311)
    assert df.loc[0, "bid"] == pytest.approx(1.10307)
    assert df.loc[0, "ask_vol"] == pytest.approx(0.9)
    assert df.loc[0, "bid_vol"] == pytest.approx(3.6)


def test_ask_is_not_swapped_with_bid() -> None:
    # Ask strictly above bid; swapping the fields would invert every row.
    raw = make_bi5([(0, 110500, 110400, 1.0, 1.0)])
    df = parse_bi5(raw, HOUR, "EURUSD")
    assert df.loc[0, "ask"] > df.loc[0, "bid"]
    assert (df["ask"] > df["bid"]).all()


def test_jpy_scaling() -> None:
    raw = make_bi5([(1000, 149123, 149118, 2.0, 2.0)])
    df = parse_bi5(raw, HOUR, "USDJPY")
    assert df.loc[0, "ask"] == pytest.approx(149.123)
    assert df.loc[0, "bid"] == pytest.approx(149.118)


def test_timestamps_are_hour_start_plus_ms_offset() -> None:
    raw = make_bi5([(0, 110000, 109999, 1.0, 1.0), (3_599_999, 110000, 109999, 1.0, 1.0)])
    df = parse_bi5(raw, HOUR, "EURUSD")

    assert df.loc[0, "ts_utc"] == pd.Timestamp("2025-03-03T10:00:00Z")
    assert df.loc[0, "ts_utc"] == pd.Timestamp(HOUR)
    assert df.loc[1, "ts_utc"] == pd.Timestamp("2025-03-03T10:59:59.999000Z")
    assert (df["ts_utc"].dt.hour == 10).all()
    assert df["ts_utc"].is_monotonic_increasing


def test_empty_payload_returns_empty_frame() -> None:
    df = parse_bi5(b"", HOUR, "EURUSD")
    expected = empty_frame()

    assert len(df) == 0
    assert list(df.columns) == list(expected.columns)
    assert df["ts_utc"].dtype == expected["ts_utc"].dtype
    for col in ("ask", "bid", "ask_vol", "bid_vol"):
        assert df[col].dtype == expected[col].dtype


def test_empty_compressed_payload_returns_empty_frame() -> None:
    df = parse_bi5(lzma.compress(b"", format=lzma.FORMAT_ALONE), HOUR, "EURUSD")
    assert len(df) == 0
    assert str(df["ts_utc"].dtype) == "datetime64[us, UTC]"


def test_payload_not_multiple_of_record_size_raises() -> None:
    raw = lzma.compress(b"\x00" * (RECORD_SIZE + 7), format=lzma.FORMAT_ALONE)
    with pytest.raises(ValueError):
        parse_bi5(raw, HOUR, "EURUSD")


def test_naive_hour_start_raises() -> None:
    raw = make_bi5([(0, 110000, 109999, 1.0, 1.0)])
    with pytest.raises(ValueError):
        parse_bi5(raw, datetime(2025, 3, 3, 10, 0), "EURUSD")
