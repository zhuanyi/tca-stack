import hashlib

import duckdb
import numpy as np
import pandas as pd
import pytest

from tca.core import storage


def _frame(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-03-03 14:00", periods=n, freq="s", tz="UTC")
    return pd.DataFrame(
        {
            "ts_utc": ts,
            "side": rng.choice([-1, 1], n),
            "px": rng.normal(100, 0.1, n).round(6),
            "label": rng.choice(["a", "b", "c"], n),
        }
    )


def _digest(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_write_twice_gives_identical_bytes(tmp_path):
    df = _frame()
    p1 = storage.write_table(df, "fills", tmp_path / "a")
    p2 = storage.write_table(df.copy(), "fills", tmp_path / "b")
    assert _digest(p1) == _digest(p2)
    storage.write_table(df, "fills", tmp_path / "a")
    assert _digest(p1) == _digest(p2)


def test_round_trip_preserves_values_and_utc_microseconds(tmp_path):
    df = _frame()
    storage.write_table(df, "fills", tmp_path)
    out = storage.read_table("fills", tmp_path)
    pd.testing.assert_frame_equal(out, df.assign(ts_utc=df["ts_utc"].astype("datetime64[us, UTC]")))
    assert str(out["ts_utc"].dtype) == "datetime64[us, UTC]"


def test_index_is_not_written(tmp_path):
    df = _frame()
    df.index = df.index + 10
    storage.write_table(df, "fills", tmp_path)
    out = storage.read_table("fills", tmp_path)
    assert list(out.columns) == list(df.columns)
    assert isinstance(out.index, pd.RangeIndex)


def test_naive_timestamp_raises(tmp_path):
    df = _frame()
    df["ts_utc"] = df["ts_utc"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="ts_utc"):
        storage.write_table(df, "fills", tmp_path)


def test_non_utc_timestamp_is_converted_to_utc(tmp_path):
    df = _frame()
    df["ts_ny"] = df["ts_utc"].dt.tz_convert("America/New_York")
    storage.write_table(df, "fills", tmp_path)
    out = storage.read_table("fills", tmp_path)
    assert str(out["ts_ny"].dtype) == "datetime64[us, UTC]"
    assert (out["ts_ny"] == df["ts_utc"]).all()


def test_file_over_limit_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "MAX_FILE_BYTES", 1000)
    with pytest.raises(ValueError, match="limit"):
        storage.write_table(_frame(5000), "fills", tmp_path)
    assert not (tmp_path / "fills.parquet").exists()


def test_round_floats():
    df = pd.DataFrame({"a": [1.23456789, 2.0], "b": [1, 2]})
    out = storage.round_floats(df, 4)
    assert out["a"].tolist() == [1.2346, 2.0]
    assert out["b"].tolist() == [1, 2]


def test_data_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TCA_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("TCA_DATA_RAW", raising=False)
    assert storage.data_dir() == tmp_path
    assert storage.raw_dir() == tmp_path / "raw"
    monkeypatch.setenv("TCA_DATA_RAW", str(tmp_path / "elsewhere"))
    assert storage.raw_dir() == tmp_path / "elsewhere"


def test_build_duckdb_single_and_part_folder(tmp_path):
    derived = tmp_path / "derived"
    storage.write_table(_frame(100, 1), "alpha", derived)
    big = _frame(250, 2)
    storage.write_table_parts(big, "beta", rows_per_part=100, derived=derived)
    assert sorted(p.name for p in (derived / "beta").glob("part-*.parquet")) == [
        "part-00000.parquet",
        "part-00001.parquet",
        "part-00002.parquet",
    ]
    pd.testing.assert_frame_equal(
        storage.read_table("beta", derived),
        big.assign(ts_utc=big["ts_utc"].astype("datetime64[us, UTC]")),
    )

    db_path = tmp_path / "tca.duckdb"
    counts = storage.build_duckdb(derived, db_path)
    assert counts == {"alpha": 100, "beta": 250}
    with duckdb.connect(str(db_path), read_only=True) as con:
        assert con.execute("select count(*) from alpha").fetchone() == (100,)
        assert con.execute("select count(*) from beta").fetchone() == (250,)
        dtype = con.execute(
            "select data_type from information_schema.columns "
            "where table_name = 'alpha' and column_name = 'ts_utc'"
        ).fetchone()
        assert dtype == ("TIMESTAMP WITH TIME ZONE",)


def test_build_duckdb_is_rebuilt_from_scratch(tmp_path):
    derived = tmp_path / "derived"
    storage.write_table(_frame(10), "alpha", derived)
    db_path = tmp_path / "tca.duckdb"
    storage.build_duckdb(derived, db_path)
    (derived / "alpha.parquet").unlink()
    storage.write_table(_frame(20), "gamma", derived)
    assert storage.build_duckdb(derived, db_path) == {"gamma": 20}


@pytest.mark.parametrize("rows_per_part", [0, -1])
def test_write_table_parts_rejects_non_positive_rows_per_part(tmp_path, rows_per_part):
    with pytest.raises(ValueError, match="rows_per_part"):
        storage.write_table_parts(
            _frame(10), "fills", rows_per_part=rows_per_part, derived=tmp_path
        )
    assert not (tmp_path / "fills").exists()


def test_read_table_rejects_file_and_folder_with_same_name(tmp_path):
    storage.write_table(_frame(2), "dup", tmp_path)
    storage.write_table_parts(_frame(5, 1), "dup", rows_per_part=2, derived=tmp_path)
    assert (tmp_path / "dup.parquet").exists()
    assert (tmp_path / "dup").is_dir()
    with pytest.raises(ValueError, match="dup"):
        storage.read_table("dup", tmp_path)


def test_build_duckdb_rejects_file_and_folder_with_same_name(tmp_path):
    derived = tmp_path / "derived"
    storage.write_table(_frame(2), "dup", derived)
    storage.write_table_parts(_frame(5, 1), "dup", rows_per_part=2, derived=derived)
    with pytest.raises(ValueError, match="dup"):
        storage.build_duckdb(derived, tmp_path / "tca.duckdb")
    assert not (tmp_path / "tca.duckdb").exists()


def test_check_folder_size_sums_every_parquet_file(tmp_path):
    derived = tmp_path / "derived"
    storage.write_table(_frame(50), "alpha", derived)
    storage.write_table_parts(_frame(120, 1), "beta", rows_per_part=50, derived=derived)
    expected = sum(p.stat().st_size for p in derived.rglob("*.parquet"))
    assert storage.check_folder_size(derived) == expected
    assert expected > (derived / "alpha.parquet").stat().st_size


def test_check_folder_size_raises_over_limit(tmp_path, monkeypatch):
    derived = tmp_path / "derived"
    storage.write_table(_frame(50), "alpha", derived)
    size = (derived / "alpha.parquet").stat().st_size
    monkeypatch.setattr(storage, "MAX_FOLDER_BYTES", size)
    assert storage.check_folder_size(derived) == size
    monkeypatch.setattr(storage, "MAX_FOLDER_BYTES", size - 1)
    with pytest.raises(ValueError, match="folder limit"):
        storage.check_folder_size(derived)


def test_build_duckdb_enforces_folder_limit(tmp_path, monkeypatch):
    derived = tmp_path / "derived"
    storage.write_table(_frame(50), "alpha", derived)
    monkeypatch.setattr(storage, "MAX_FOLDER_BYTES", 1)
    with pytest.raises(ValueError, match="folder limit"):
        storage.build_duckdb(derived, tmp_path / "tca.duckdb")
    assert not (tmp_path / "tca.duckdb").exists()


def test_build_duckdb_escapes_double_quotes_in_table_names(tmp_path):
    derived = tmp_path / "derived"
    storage.write_table(_frame(5), 'odd"name', derived)
    db_path = tmp_path / "tca.duckdb"
    with pytest.raises(ValueError, match="table name"):
        storage.build_duckdb(derived, db_path)


def test_write_table_parts_removes_stale_parts(tmp_path):
    df = _frame(100, seed=42)
    storage.write_table_parts(df, "fills", rows_per_part=50, derived=tmp_path)
    assert len(list((tmp_path / "fills").glob("part-*.parquet"))) == 2
    storage.write_table_parts(df, "fills", rows_per_part=100, derived=tmp_path)
    parts = sorted((tmp_path / "fills").glob("part-*.parquet"))
    assert len(parts) == 1
    assert parts[0].name == "part-00000.parquet"


def test_build_duckdb_atomic_no_partial_on_failure(tmp_path):
    derived = tmp_path / "derived"
    storage.write_table(_frame(10), "good", derived)
    db_path = tmp_path / "tca.duckdb"
    storage.build_duckdb(derived, db_path)
    old_bytes = db_path.read_bytes()
    (derived / "bad name!").mkdir()
    (derived / "bad name!" / "part-00000.parquet").write_bytes(b"not parquet")
    with pytest.raises(ValueError, match="table name"):
        storage.build_duckdb(derived, db_path)
    assert db_path.read_bytes() == old_bytes


def test_build_duckdb_none_derived_with_no_data_dir_raises_file_not_found(tmp_path, monkeypatch):
    monkeypatch.setenv("TCA_DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        storage.build_duckdb(None, tmp_path / "tca.duckdb")


def test_build_duckdb_rejects_invalid_table_names(tmp_path):
    derived = tmp_path / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "has space").mkdir()
    (derived / "has space" / "part-00000.parquet").write_bytes(b"x")
    with pytest.raises(ValueError, match="table name"):
        storage.build_duckdb(derived, tmp_path / "tca.duckdb")
