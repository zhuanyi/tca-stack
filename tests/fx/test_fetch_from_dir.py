"""``--from-dir`` reads downloaded .bi5 files through the parser and makes no network request."""

from __future__ import annotations

import importlib.util
import lzma
import struct
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fetch_fx_ticks.py"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("fetch_fx_ticks", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bi5(records: list[tuple[int, int, int, float, float]]) -> bytes:
    payload = b"".join(struct.pack(">iiiff", *r) for r in records)
    return lzma.compress(payload, format=lzma.FORMAT_ALONE)


def _write_hour(root: Path, pair: str, day: date, hour: int, records) -> None:
    path = (
        root
        / pair
        / f"{day.year:04d}"
        / f"{day.month - 1:02d}"
        / f"{day.day:02d}"
        / f"{hour:02d}h_ticks.bi5"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_bi5(records))


def test_local_hour_path_uses_the_zero_based_month(script):
    path = script.local_hour_path(Path("/x"), "EURUSD", date(2025, 4, 3), 10)
    assert path == Path("/x/EURUSD/2025/03/03/10h_ticks.bi5")


def test_reads_local_files_without_any_request(script, tmp_path, monkeypatch):
    day = date(2025, 4, 3)
    source, raw = tmp_path / "bi5", tmp_path / "raw"
    for hour in range(24):
        _write_hour(
            source,
            "USDJPY",
            day,
            hour,
            [(1500, 150123, 150119, 1.0, 2.0), (2500, 150125, 150121, 1.0, 1.0)],
        )

    def no_network(*args, **kwargs):
        raise AssertionError("a network client was created")

    monkeypatch.setattr(httpx, "Client", no_network)
    monkeypatch.setenv("TCA_DATA_RAW", str(raw))
    code = script.main(
        [
            "--pairs",
            "USDJPY",
            "--start",
            "2025-04-03",
            "--end",
            "2025-04-03",
            "--from-dir",
            str(source),
        ]
    )
    assert code == 0
    ticks = pd.read_parquet(raw / "fx" / "USDJPY" / "2025-04-03.parquet")
    assert len(ticks) == 48
    assert ticks["ask"].iloc[0] == pytest.approx(150.123)
    assert ticks["ts_utc"].iloc[0] == pd.Timestamp("2025-04-03 00:00:01.5", tz="UTC")
    assert ticks["ts_utc"].is_monotonic_increasing


def test_a_missing_local_file_is_an_hour_without_ticks(script, tmp_path, monkeypatch):
    day = date(2025, 4, 3)
    source, raw = tmp_path / "bi5", tmp_path / "raw"
    _write_hour(source, "EURUSD", day, 10, [(0, 110311, 110307, 0.9, 3.6)])
    monkeypatch.setenv("TCA_DATA_RAW", str(raw))
    code = script.main(
        [
            "--pairs",
            "EURUSD",
            "--start",
            "2025-04-03",
            "--end",
            "2025-04-03",
            "--from-dir",
            str(source),
        ]
    )
    assert code == 0
    assert len(pd.read_parquet(raw / "fx" / "EURUSD" / "2025-04-03.parquet")) == 1


class _Resp:
    def __init__(self, status: int, content: bytes = b"") -> None:
        self.status_code = status
        self.content = content


class _FakeClient:
    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)

    def get(self, url):
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def _fetcher(script, tmp_path, monkeypatch, outcomes):
    monkeypatch.setattr(script.time, "sleep", lambda s: None)
    fetcher = script.Fetcher(tmp_path)
    fetcher.client = _FakeClient(outcomes)
    return fetcher


def test_404_is_counted_and_returns_no_ticks(script, tmp_path, monkeypatch):
    fetcher = _fetcher(script, tmp_path, monkeypatch, [_Resp(404)])
    assert fetcher.fetch_hour("EURUSD", date(2025, 4, 3), 10) == b""
    assert fetcher.not_found == 1


def test_refusals_are_counted_and_stop_after_the_allowed_attempts(script, tmp_path, monkeypatch):
    fetcher = _fetcher(script, tmp_path, monkeypatch, [_Resp(503)] * script.MAX_ATTEMPTS)
    with pytest.raises(script.Refused):
        fetcher.fetch_hour("EURUSD", date(2025, 4, 3), 10)
    assert fetcher.refused == script.MAX_ATTEMPTS


def test_network_errors_are_counted(script, tmp_path, monkeypatch):
    err = httpx.ConnectError("boom")
    fetcher = _fetcher(script, tmp_path, monkeypatch, [err, err, _Resp(200, b"x")])
    assert fetcher.fetch_hour("EURUSD", date(2025, 4, 3), 10) == b"x"
    assert fetcher.errors == 2


def test_report_prints_the_not_found_count(script, tmp_path, monkeypatch, capsys):
    fetcher = _fetcher(script, tmp_path, monkeypatch, [_Resp(404)])
    fetcher.fetch_hour("EURUSD", date(2025, 4, 3), 10)
    fetcher.report()
    assert "not_found=1 refused=0 errors=0" in capsys.readouterr().out
