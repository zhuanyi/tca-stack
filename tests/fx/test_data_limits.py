"""Committed FX outputs respect the repository size limits (25 MB per file, 150 MB per folder)."""

from __future__ import annotations

from tca.fx.paths import FIXTURE, REPO_ROOT

MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_FOLDER_BYTES = 150 * 1024 * 1024


def test_committed_derived_files_are_within_the_limits():
    derived = REPO_ROOT / "data" / "derived"
    files = [p for p in derived.rglob("*.parquet")] if derived.exists() else []
    assert all(p.stat().st_size <= MAX_FILE_BYTES for p in files)
    assert sum(p.stat().st_size for p in files) <= MAX_FOLDER_BYTES


def test_the_synthetic_fixture_is_small():
    assert FIXTURE.stat().st_size < 5 * 1024 * 1024
