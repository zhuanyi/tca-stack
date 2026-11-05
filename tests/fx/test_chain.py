"""The whole chain on the synthetic fixture under the ci profile: outputs and determinism."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest
from typer.testing import CliRunner

from tca.cli import app

COMMANDS = ["bars", "profile", "simulate", "tca", "figures", "summary"]
FIGURES = [
    "fx_01_is_by_style_size",
    "fx_02_liquidity_heatmap",
    "fx_03_is_decomposition",
    "fx_04_cost_vs_size",
    "fx_05_lp_scorecard",
    "fx_06_pretrade_calibration",
    "fx_07_markouts_by_lp",
]
MAX_FILE_BYTES = 25 * 1024 * 1024


def _run_chain(data_dir: Path, monkeypatch) -> str:
    monkeypatch.setenv("TCA_DATA_DIR", str(data_dir))
    monkeypatch.delenv("TCA_FIGURES_DIR", raising=False)
    runner = CliRunner()
    output = ""
    for command in COMMANDS:
        result = runner.invoke(app, ["fx", command, "--profile", "ci"])
        assert result.exit_code == 0, f"{command}: {result.output}"
        output = result.output
    return output


def _digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in {".parquet", ".png"}
    }


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    patch = pytest.MonkeyPatch()
    roots = [tmp_path_factory.mktemp("run") for _ in range(2)]
    outputs = [_run_chain(r, patch) for r in roots]
    patch.undo()
    return roots, outputs


def test_every_table_and_figure_is_written(chain):
    root = chain[0][0]
    derived = {p.stem for p in (root / "ci" / "derived").glob("*.parquet")}
    assert {
        "fx_bars_1m",
        "fx_liquidity_profile",
        "fx_lps",
        "fx_orders",
        "fx_children",
        "fx_fill_attempts",
        "fx_fills",
        "fx_markouts",
        "fx_tca_fill",
        "fx_tca_parent",
        "fx_lp_scorecard",
    } <= derived
    figures = {p.stem for p in (root / "ci" / "figures").glob("*.png")}
    assert figures == set(FIGURES)


def test_figures_are_1600_by_900(chain):
    for path in (chain[0][0] / "ci" / "figures").glob("*.png"):
        assert struct.unpack(">II", path.read_bytes()[16:24]) == (1600, 900)


def test_same_seed_gives_byte_identical_outputs(chain):
    first, second = (_digest(r) for r in chain[0])
    assert first == second
    assert len(first) >= 18


def test_file_size_limits(chain):
    for path in (chain[0][0] / "ci").rglob("*.parquet"):
        assert path.stat().st_size <= MAX_FILE_BYTES


def test_summary_prints_the_findings_numbers(chain):
    text = chain[1][0]
    for heading in ("Implementation shortfall", "Brokers", "LP scorecard", "Pre-trade"):
        assert heading in text


def test_markouts_are_wide_with_both_families(chain):
    import pandas as pd

    frame = pd.read_parquet(chain[0][0] / "ci" / "derived" / "fx_markouts.parquet")
    expected = {
        f"markout_{kind}_bp_{h}"
        for kind in ("tape", "eff")
        for h in (0, 1, 5, 30, 60, 300, 900, 1800, 3600)
    }
    assert set(frame.columns) == {"fill_id"} | expected
    assert frame["fill_id"].is_unique


def test_the_ci_profile_does_not_touch_committed_outputs(chain):
    root = chain[0][0]
    assert not (root / "derived").exists()
