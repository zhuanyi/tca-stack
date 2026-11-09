"""The three invariants of Note 3 on the CI-sized grid (specs/06_NOTE_RFQ_DEALER_COUNT.md s5).

The grid is N in {1, 3, 7}, 10y on-the-run, a single seed and two leakage levels, so the whole
file runs six simulator runs (about ten seconds). The zero-leakage cells switch the dealers'
competition widening off: with the configured widening (0.03 32nds per dealer beyond three) the
N = 7 cell costs about 0.02 more than N = 3, which is the flattening Figure 1 panel (b) of the
note reports; the invariant here is the competition benefit alone.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate, stats

from tca.ust.config import load_config
from tca.ust.inputs import load_inputs

ANALYSIS = Path(__file__).resolve().parents[2] / "notes" / "06_rfq_dealer_count" / "analysis.py"
CELLS = (1, 3, 7)


@pytest.fixture(scope="module")
def note():
    spec = importlib.util.spec_from_file_location("note06", ANALYSIS)
    module = importlib.util.module_from_spec(spec)
    assert module is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reduced(note):
    return note.run_reduced(load_config(), load_inputs())


def _by_n(reduced, leakage: float, column: str) -> np.ndarray:
    sel = reduced[np.isclose(reduced["leakage_scale"], leakage)].sort_values("n_dealers")
    assert list(sel["n_dealers"]) == list(CELLS)
    return sel[column].to_numpy(dtype=float)


def test_reduced_grid_is_the_ci_grid(reduced):
    assert len(reduced) == 2 * len(CELLS)
    assert set(reduced["tenor"]) == {10}
    assert set(reduced["status"]) == {"OTR"}
    assert np.isclose(reduced["size"], 50.0).all()
    assert (reduced["n_pairs"] == reduced["programmes"]).all()


def test_exec_cost_falls_with_dealers_without_leakage(reduced):
    cost = _by_n(reduced, leakage=0.0, column="exec_cost_1")
    assert cost[0] > cost[1] > cost[2], cost


def test_arrival_shift_rises_with_dealers_under_leakage(reduced):
    shift = _by_n(reduced, leakage=2.0, column="arrival_shift_2")
    assert shift[0] < shift[1] < shift[2], shift


def test_order_statistics_match_numerical_integration(note):
    def by_quantile(n: int) -> float:
        f = lambda x: n * x * stats.norm.pdf(x) * stats.norm.cdf(x) ** (n - 1)  # noqa: E731
        value, _ = integrate.quad(f, -10.0, 10.0, limit=200)
        return value

    assert note.order_statistic_max(1) == pytest.approx(0.0, abs=1e-9)
    for n in (2, 3, 4, 5, 7, 10):
        assert round(note.order_statistic_max(n), 2) == round(by_quantile(n), 2)
    constants = {row.n_dealers: row.emax for row in note.order_statistics().itertuples()}
    assert round(constants[2], 2) == 0.56
    assert round(constants[10], 2) == 1.54
