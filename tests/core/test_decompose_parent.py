"""Parent-order decomposition: worked example, identities, edge cases and guards."""

import math

import numpy as np
import pytest

from tca.core.conventions import decompose_parent, implementation_shortfall


def test_worked_example_from_the_spec_review():
    r = decompose_parent(
        side=1,
        decision_mid=100.0,
        first_ref_mid=100.0,
        fill_ref_mid=[100.0, 100.5],
        fill_qty=[1.0, 1.0],
        fill_px=[100.02, 100.52],
        parent_qty=2.0,
        end_mid=101.0,
    )
    assert r.delay_bp == pytest.approx(0.0, abs=1e-9)
    assert r.timing_bp == pytest.approx(25.0, abs=1e-9)
    assert r.execution_bp == pytest.approx(2.0, abs=1e-9)
    assert r.is_filled_bp == pytest.approx(27.0, abs=1e-9)
    assert r.opportunity_bp == pytest.approx(0.0, abs=1e-9)
    assert r.is_bp == pytest.approx(27.0, abs=1e-9)
    assert r.filled_frac == pytest.approx(1.0)


def test_partial_fill_adds_opportunity_cost():
    r = decompose_parent(
        side=1,
        decision_mid=100.0,
        first_ref_mid=100.0,
        fill_ref_mid=[100.0, 100.5],
        fill_qty=[1.0, 1.0],
        fill_px=[100.02, 100.52],
        parent_qty=4.0,
        end_mid=101.0,
    )
    assert r.filled_frac == pytest.approx(0.5)
    assert r.opportunity_bp == pytest.approx(50.0, abs=1e-9)
    assert r.is_bp == pytest.approx(63.5, abs=1e-9)


def test_sell_side_flips_the_signs():
    r = decompose_parent(
        side=-1,
        decision_mid=100.0,
        first_ref_mid=99.9,
        fill_ref_mid=[99.9, 99.4],
        fill_qty=[1.0, 1.0],
        fill_px=[99.88, 99.38],
        parent_qty=2.0,
        end_mid=99.0,
    )
    assert r.delay_bp == pytest.approx(10.0, abs=1e-9)
    assert r.timing_bp == pytest.approx(25.0, abs=1e-9)
    assert r.execution_bp == pytest.approx(2.0, abs=1e-9)


def test_identities_and_agreement_with_implementation_shortfall():
    rng = np.random.default_rng(7)
    for _ in range(200):
        n = int(rng.integers(1, 6))
        side = int(rng.choice([-1, 1]))
        decision = float(rng.uniform(50, 150))
        refs = decision + rng.normal(0, 0.5, n)
        qty = rng.uniform(0.1, 5.0, n)
        px = refs + rng.normal(0, 0.05, n)
        parent = float(qty.sum() * rng.uniform(1.0, 3.0))
        end = decision + float(rng.normal(0, 1.0))
        first = decision + float(rng.normal(0, 0.2))
        r = decompose_parent(side, decision, first, refs, qty, px, parent, end)
        assert r.is_filled_bp == pytest.approx(r.delay_bp + r.timing_bp + r.execution_bp, abs=1e-9)
        assert r.is_bp == pytest.approx(r.filled_frac * r.is_filled_bp + r.opportunity_bp, abs=1e-9)
        ref = implementation_shortfall(side, qty, px, parent, decision, end)
        assert r.is_filled_bp == pytest.approx(ref.is_filled_bp, abs=1e-9)
        assert r.opportunity_bp == pytest.approx(ref.opportunity_bp, abs=1e-9)
        assert r.is_bp == pytest.approx(ref.is_bp, abs=1e-9)


def test_no_fills_is_all_opportunity():
    r = decompose_parent(1, 100.0, 100.1, [], [], [], 2.0, 101.0)
    for term in (r.delay_bp, r.timing_bp, r.execution_bp, r.is_filled_bp):
        assert math.isnan(term)
    assert r.filled_frac == 0.0
    assert r.opportunity_bp == pytest.approx(100.0, abs=1e-9)
    assert r.is_bp == pytest.approx(100.0, abs=1e-9)


def test_no_fills_allows_a_nan_first_mid():
    r = decompose_parent(1, 100.0, float("nan"), [], [], [], 2.0, 101.0)
    assert math.isnan(r.delay_bp)
    assert r.is_bp == pytest.approx(100.0, abs=1e-9)


def test_a_fill_requires_a_finite_first_mid():
    with pytest.raises(ValueError):
        decompose_parent(1, 100.0, float("nan"), [100.0], [1.0], [100.0], 1.0, 100.0)


def test_result_is_frozen():
    r = decompose_parent(1, 100.0, 100.0, [100.0], [1.0], [100.0], 1.0, 100.0)
    with pytest.raises(AttributeError):
        r.is_bp = 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"side": 0},
        {"decision_mid": 0.0},
        {"decision_mid": float("nan")},
        {"parent_qty": 0.0},
        {"fill_qty": [1.0, 1.0]},
        {"fill_qty": [0.0]},
        {"fill_qty": [2.0]},
        {"fill_px": [float("inf")]},
        {"end_mid": float("nan")},
    ],
)
def test_bad_inputs_raise(kwargs):
    base = {
        "side": 1,
        "decision_mid": 100.0,
        "first_ref_mid": 100.0,
        "fill_ref_mid": [100.0],
        "fill_qty": [1.0],
        "fill_px": [100.0],
        "parent_qty": 1.0,
        "end_mid": 100.0,
    }
    with pytest.raises(ValueError):
        decompose_parent(**{**base, **kwargs})
