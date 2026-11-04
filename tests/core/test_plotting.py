import struct

import numpy as np
import pandas as pd
import pytest

from tca.plotting import common, style


@pytest.fixture(autouse=True)
def _figures_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("TCA_FIGURES_DIR", str(tmp_path / "figs"))
    style.apply()


def _png_size(path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", header[16:24])


def _render():
    fig, ax = style.figure("Cost by tenor", "Simulated RFQs, Mar-May 2025", "Source: simulation")
    ax.plot([1, 2, 3], [1, 4, 9])
    return fig


def test_save_is_exactly_1600_by_900(tmp_path):
    path = style.save(_render(), "ust_00_test")
    assert path == tmp_path / "figs" / "ust_00_test.png"
    assert _png_size(path) == (1600, 900)


def test_save_is_deterministic():
    first = style.save(_render(), "a").read_bytes()
    second = style.save(_render(), "b").read_bytes()
    assert first == second


def test_hero_font_is_larger():
    style.apply(hero=True)
    assert style.plt.rcParams["font.size"] == 14.0
    style.apply()
    assert style.plt.rcParams["font.size"] == 11.0


def test_palette_is_okabe_ito():
    assert style.PALETTE[:3] == ["#0072B2", "#E69F00", "#009E73"]
    assert len(style.PALETTE) == 8


def test_boxplot_with_n_labels_counts():
    _, ax = style.figure("t", "s", "src")
    groups = {"2Y": np.arange(10.0), "10Y": np.arange(25.0)}
    common.boxplot_with_n(ax, groups)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["2Y\nn=10", "10Y\nn=25"]


def test_binned_means_counts_and_ci_order(rng):
    x = rng.lognormal(size=1000)
    y = 2.0 * np.sqrt(x) + rng.normal(size=1000)
    table = common.binned_means(x, y, 8, rng=np.random.default_rng(1))
    assert list(table.columns) == ["bin", "x_mid", "mean", "lo", "hi", "n"]
    assert table["n"].sum() == 1000
    assert (table["lo"] <= table["mean"]).all()
    assert (table["mean"] <= table["hi"]).all()
    assert table["x_mid"].is_monotonic_increasing


def test_binned_means_with_explicit_edges():
    x = np.array([1.0, 2.0, 3.0, 11.0, 12.0])
    y = np.array([1.0, 3.0, 5.0, 10.0, 20.0])
    table = common.binned_means(x, y, [0.0, 5.0, 20.0])
    assert table["n"].tolist() == [3, 2]
    assert table["mean"].tolist() == [3.0, 15.0]


def test_cost_vs_size_recovers_square_root_exponent(rng):
    size = rng.lognormal(3.0, 1.0, 5000)
    cost = 0.2 + 0.5 * np.sqrt(size) + rng.normal(0, 0.5, 5000)
    _, ax = style.figure("t", "s", "src")
    _, _, c = common.cost_vs_size(ax, size, cost, rng=np.random.default_rng(2))
    assert ax.get_xscale() == "log"
    assert c == pytest.approx(0.5, abs=0.08)
    assert any(f"c = {c:.2f}" in t.get_text() for t in ax.texts)


def test_markout_curve_and_heatmap_render():
    fig, axes = style.figure("t", "s", "src", ncols=2, figsize=(10.67, 6))
    common.markout_curve(
        axes[0], [1, 5, 30], [0.1, 0.2, 0.1], [0.0, 0.1, 0.0], [0.2, 0.3, 0.2], "x", "#0072B2"
    )
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 10.0, 10.0]}, index=["r1", "r2", "r3"])
    common.heatmap_with_values(axes[1], frame)
    assert [t.get_text() for t in axes[1].texts][:2] == ["1.00", "10.00"]
    style.save(fig, "multi")


def test_binned_means_drops_nan_rows():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y = np.array([10.0, np.nan, 30.0, 40.0, 50.0])
    table = common.binned_means(x, y, 2, rng=np.random.default_rng(0))
    assert table["n"].sum() == 3


def test_cost_vs_size_rejects_non_positive_sizes():
    _, ax = style.figure("t", "s", "src")
    with pytest.raises(ValueError, match="strictly positive"):
        common.cost_vs_size(ax, [0, 1, 2], [1, 2, 3])


def test_heatmap_all_nan_does_not_raise():
    _, ax = style.figure("t", "s", "src")
    frame = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
    common.heatmap_with_values(ax, frame)


def test_apply_idempotent_with_open_figure():
    style.figure("t", "s", "src")
    style.apply()  # should not raise even though a figure exists
