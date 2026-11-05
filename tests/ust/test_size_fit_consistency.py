"""``tca ust summary`` and the ``ust_02`` / ``ust_08`` figures use one exponent fit."""

from __future__ import annotations

import pytest

from tca.core import storage
from tca.plotting import common
from tca.ust import figures, summary


@pytest.fixture
def captured(monkeypatch, tmp_path):
    """Record every ``(a, b, c, lo, hi)`` returned to a figure, without writing to docs/."""
    monkeypatch.setenv("TCA_FIGURES_DIR", str(tmp_path))
    calls = []
    real = common.cost_vs_size

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append(out)
        return out

    monkeypatch.setattr(common, "cost_vs_size", spy)
    return calls


def test_summary_exponents_equal_figure_exponents(captured):
    figures.cost_vs_size(None)
    fits = summary.size_fits(storage.read_table("ust_tca_fill"))
    assert len(captured) == 2
    for (_a, _b, c, lo, hi), run in zip(captured, ("OTR", "OLD"), strict=True):
        row = fits.loc[run]
        assert row["c"] == pytest.approx(c, abs=1e-12)
        assert row["c_lo"] == pytest.approx(lo, abs=1e-12)
        assert row["c_hi"] == pytest.approx(hi, abs=1e-12)


def test_summary_reports_bound_status():
    fits = summary.size_fits(storage.read_table("ust_tca_fill"))
    assert set(fits["c_bound"]) <= {"-", "lower", "upper"}
