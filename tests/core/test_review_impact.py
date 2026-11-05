"""Acceptance tests for ImpactState accessors and time validation (m7)."""

import numpy as np
import pytest

from tca.core.impact import ImpactState


def test_permanent_and_transient_components_are_exposed():
    s = ImpactState(perm_frac=0.25, half_life_s=60.0)
    s.add(0.0, 8.0)
    assert s.permanent == pytest.approx(2.0)
    assert s.transient(0.0) == pytest.approx(6.0)
    assert s.transient(60.0) == pytest.approx(3.0)
    assert s.value(60.0) == pytest.approx(s.permanent + s.transient(60.0))
    np.testing.assert_allclose(s.transient_at(np.array([0.0, 60.0, 120.0])), [6.0, 3.0, 1.5])


def test_components_after_reset_are_zero():
    s = ImpactState(0.5, 10.0)
    s.add(0.0, 4.0)
    s.reset()
    assert s.permanent == 0.0
    assert s.transient(0.0) == 0.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_add_rejects_non_finite_time(bad):
    s = ImpactState(0.5, 10.0)
    with pytest.raises(ValueError, match="finite"):
        s.add(bad, 1.0)
    assert s.value(0.0) == 0.0


def test_add_rejects_non_finite_impact():
    s = ImpactState(0.5, 10.0)
    with pytest.raises(ValueError, match="finite"):
        s.add(1.0, float("nan"))


def test_value_rejects_nan_time_and_value_at_rejects_any_nan():
    s = ImpactState(0.5, 10.0)
    s.add(0.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        s.value(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        s.value_at(np.array([1.0, np.nan]))
