"""Parquet storage with deterministic bytes, and the DuckDB build from it.

Every table lives in ``<data>/derived`` either as ``<name>.parquet`` or, when a single file
would exceed the size limit, as a folder ``<name>/part-*.parquet``. Timestamps are stored as
timezone-aware UTC at microsecond precision.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_FOLDER_BYTES = 150 * 1024 * 1024
_ZSTD_LEVEL = 9
_UTC_US = "datetime64[us, UTC]"


def data_dir() -> Path:
    """Root data directory: ``$TCA_DATA_DIR`` if set, else ``data`` at the repository root."""
    env = os.environ.get("TCA_DATA_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[2] / "data"


def raw_dir() -> Path:
    """Directory for downloaded raw data: ``$TCA_DATA_RAW`` if set, else ``<data_dir>/raw``."""
    env = os.environ.get("TCA_DATA_RAW")
    return Path(env) if env else data_dir() / "raw"


def _derived(derived: Path | None) -> Path:
    return Path(derived) if derived is not None else data_dir() / "derived"


def round_floats(df: pd.DataFrame, decimals: int = 6) -> pd.DataFrame:
    """Return a copy with every float column rounded to ``decimals`` places (smaller parquet)."""
    out = df.copy()
    for col in out.select_dtypes(include="float").columns:
        out[col] = out[col].round(decimals)
    return out


def _normalise_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime columns to UTC microseconds; raise ``ValueError`` on a naive column."""
    out = df.copy()
    for col in out.columns:
        dtype = out[col].dtype
        if not pd.api.types.is_datetime64_any_dtype(dtype):
            continue
        if getattr(dtype, "tz", None) is None:
            raise ValueError(f"column {col!r} holds naive timestamps; localise to UTC first")
        out[col] = out[col].dt.tz_convert("UTC").astype(_UTC_US)
    return out


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    table = pa.Table.from_pandas(_normalise_timestamps(df), preserve_index=False)
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        compression_level=_ZSTD_LEVEL,
        coerce_timestamps="us",
        store_schema=False,
    )
    return sink.getvalue().to_pybytes()


def _write_file(df: pd.DataFrame, path: Path) -> Path:
    payload = _parquet_bytes(df)
    if len(payload) > MAX_FILE_BYTES:
        raise ValueError(
            f"{path.name} would be {len(payload)} bytes, over the {MAX_FILE_BYTES} byte limit; "
            "use write_table_parts or a wider layout"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def write_table(df: pd.DataFrame, name: str, derived: Path | None = None) -> Path:
    """Write ``df`` to ``<derived>/<name>.parquet`` and return the path.

    Output is zstd level 9 with no pandas index, so equal frames give identical bytes. Raises
    ``ValueError`` on a naive timestamp column or when the file would exceed ``MAX_FILE_BYTES``.
    """
    return _write_file(df, _derived(derived) / f"{name}.parquet")


def write_table_parts(
    df: pd.DataFrame, name: str, rows_per_part: int, derived: Path | None = None
) -> Path:
    """Write ``df`` as ``<derived>/<name>/part-NNNNN.parquet`` files of ``rows_per_part`` rows.

    Stale ``part-*.parquet`` files from a previous run with a different ``rows_per_part`` are
    removed before writing so that :func:`read_table` never mixes old and new data.
    """
    if rows_per_part <= 0:
        raise ValueError("rows_per_part must be positive")
    folder = _derived(derived) / name
    folder.mkdir(parents=True, exist_ok=True)
    for stale in folder.glob("part-*.parquet"):
        stale.unlink()
    for i, start in enumerate(range(0, max(len(df), 1), rows_per_part)):
        _write_file(df.iloc[start : start + rows_per_part], folder / f"part-{i:05d}.parquet")
    return folder


def check_folder_size(derived: Path | None = None) -> int:
    """Return the total bytes of every parquet file under ``derived``.

    Raises ``ValueError`` when the total exceeds ``MAX_FOLDER_BYTES``.

    The size check intentionally sums every ``*.parquet`` file under ``derived`` (including
    scratch sub-folders) so that orphaned files cannot silently consume disk; the subset
    actually loaded into DuckDB is tracked by :func:`_table_files`.
    """
    root = _derived(derived)
    total = sum(p.stat().st_size for p in root.rglob("*.parquet")) if root.exists() else 0
    if total > MAX_FOLDER_BYTES:
        raise ValueError(
            f"{root} holds {total} bytes, over the {MAX_FOLDER_BYTES} byte folder limit"
        )
    return total


def _table_files(derived: Path) -> dict[str, list[Path]]:
    """Map table name to its parquet files, in the order they must be read."""
    tables: dict[str, list[Path]] = {}
    for path in sorted(derived.iterdir()):
        if path.is_file() and path.suffix == ".parquet":
            name, files = path.stem, [path]
        elif path.is_dir() and (parts := sorted(path.glob("part-*.parquet"))):
            name, files = path.name, parts
        else:
            continue
        if name in tables:
            raise ValueError(
                f"table {name!r} exists as both {name}.parquet and {name}/part-*.parquet; "
                "delete one"
            )
        tables[name] = files
    return tables


def read_table(name: str, derived: Path | None = None) -> pd.DataFrame:
    """Read ``<name>.parquet`` or the concatenation of ``<name>/part-*.parquet``."""
    files = _table_files(_derived(derived))
    if name not in files:
        raise FileNotFoundError(f"no table {name!r} in {_derived(derived)}")
    frames = [pq.read_table(p).to_pandas() for p in files[name]]
    return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)


_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_table_name(name: str) -> str:
    """Return ``name`` unchanged when it is a safe DuckDB identifier, else raise."""
    if not _TABLE_NAME_RE.match(name):
        raise ValueError(
            f"table name {name!r} must match [A-Za-z_][A-Za-z0-9_]*; "
            "DuckDB cannot bind identifiers as parameters"
        )
    return name


def build_duckdb(derived: Path | None = None, db_path: Path | None = None) -> dict[str, int]:
    """Rebuild the DuckDB file from scratch, one table per parquet file or part folder.

    Table names are the file stems or folder names. Returns the row count of each table.

    The new database is built into a temporary file and atomically moved onto ``db_path``
    only after every table has been created, so a mid-build failure never destroys the
    previous good database.
    """
    files = _table_files(_derived(derived))
    check_folder_size(derived)
    target = Path(db_path) if db_path is not None else data_dir() / "tca.duckdb"
    target.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    fd, tmp_str = tempfile.mkstemp(suffix=".duckdb", dir=str(target.parent))
    os.close(fd)
    tmp_path = Path(tmp_str)
    tmp_path.unlink()  # let duckdb.create the file fresh
    try:
        with duckdb.connect(str(tmp_path)) as con:
            for name, paths in files.items():
                _validate_table_name(name)
                safe = name.replace('"', '""')
                listing = ", ".join("'" + str(p).replace("'", "''") + "'" for p in paths)
                con.execute(f'CREATE TABLE "{safe}" AS SELECT * FROM read_parquet([{listing}])')
                counts[name] = con.execute(f'SELECT count(*) FROM "{safe}"').fetchone()[0]
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return counts
