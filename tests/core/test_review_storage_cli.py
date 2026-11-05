"""Acceptance tests: M2 regression, m10, m12, m13 and the cycler dependency."""

import tomllib
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest
from typer.testing import CliRunner

from tca import cli
from tca.core import calendar as cal
from tca.core import storage

runner = CliRunner()
ROOT = Path(__file__).resolve().parents[2]


def test_smaller_parts_rewrite_reads_back_only_the_new_rows(tmp_path):
    # M2 regression: 250 rows then 150 rows must read back as 150 rows.
    big = pd.DataFrame({"v": range(250)})
    small = pd.DataFrame({"v": range(150)})
    storage.write_table_parts(big, "fills", rows_per_part=100, derived=tmp_path)
    storage.write_table_parts(small, "fills", rows_per_part=100, derived=tmp_path)
    assert len(storage.read_table("fills", tmp_path)) == 150


def test_db_build_on_missing_folder_exits_cleanly(tmp_path):
    missing = tmp_path / "nope"
    result = runner.invoke(cli.app, ["db", "build", "--derived", str(missing)])
    assert result.exit_code == 1
    assert str(missing) in result.output
    assert "Traceback" not in result.output
    assert isinstance(result.exception, SystemExit)


def test_db_build_on_a_file_instead_of_folder_exits_cleanly(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    result = runner.invoke(cli.app, ["db", "build", "--derived", str(f)])
    assert result.exit_code == 1
    assert str(f) in result.output
    assert isinstance(result.exception, SystemExit)


def test_build_duckdb_missing_folder_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope"):
        storage.build_duckdb(tmp_path / "nope", tmp_path / "x.duckdb")


def test_build_duckdb_removes_a_leftover_wal_file(tmp_path):
    derived = tmp_path / "derived"
    storage.write_table(pd.DataFrame({"v": [1, 2, 3]}), "t", derived)
    db = tmp_path / "tca.duckdb"
    wal = tmp_path / "tca.duckdb.wal"
    wal.write_bytes(b"garbage")
    assert storage.build_duckdb(derived, db) == {"t": 3}
    assert not wal.exists()


def test_pyarrow_backed_utc_timestamps_are_accepted(tmp_path):
    ts = pd.Series(pd.date_range("2025-03-03", periods=3, tz="UTC"))
    df = pd.DataFrame(
        {"ts": ts.astype(pd.ArrowDtype(pa.timestamp("us", tz="UTC"))), "v": [1, 2, 3]}
    )
    storage.write_table(df, "arrow", tmp_path)
    assert str(storage.read_table("arrow", tmp_path)["ts"].dt.tz) == "UTC"


def test_pyarrow_backed_naive_timestamps_are_still_rejected(tmp_path):
    ts = pd.Series(pd.date_range("2025-03-03", periods=3))
    df = pd.DataFrame({"ts": ts.astype(pd.ArrowDtype(pa.timestamp("us")))})
    with pytest.raises(ValueError, match="naive"):
        storage.write_table(df, "arrow", tmp_path)


@pytest.mark.parametrize("stamp", ["2025-03-07 13:30:00", "2025-03-07T13:30:00"])
def test_load_events_rejects_naive_timestamps(tmp_path, stamp):
    p = tmp_path / "events.yaml"
    p.write_text(f"events:\n  - {{ts_utc: '{stamp}', type: NFP}}\n")
    with pytest.raises(ValueError, match="timezone"):
        cal.load_events(p)


def test_load_events_rejects_unquoted_naive_yaml_datetime(tmp_path):
    p = tmp_path / "events.yaml"
    p.write_text("events:\n  - {ts_utc: 2025-03-07 13:30:00, type: NFP}\n")
    with pytest.raises(ValueError, match="timezone"):
        cal.load_events(p)


def test_load_events_accepts_offsets_and_converts_to_utc(tmp_path):
    p = tmp_path / "events.yaml"
    p.write_text("events:\n  - {ts_utc: '2025-03-07T08:30:00-05:00', type: NFP}\n")
    df = cal.load_events(p)
    assert df["ts_utc"].iloc[0] == pd.Timestamp("2025-03-07T13:30:00Z")


def test_cycler_is_a_declared_dependency():
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    assert any(d.replace(" ", "").lower().startswith("cycler") for d in deps)
