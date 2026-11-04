import numpy as np
import pytest

from tca.core.impact import ImpactState, sqrt_impact


def test_sqrt_impact_hand_value():
    # y * sigma * sqrt(q / v) = 0.8 * 10 * sqrt(0.04) = 1.6
    assert sqrt_impact(4.0, 100.0, 0.8, 10.0) == pytest.approx(1.6, abs=1e-12)


def test_sqrt_impact_broadcasts_and_quadruples_to_double():
    q = np.array([1.0, 4.0, 16.0])
    out = sqrt_impact(q, 100.0, 1.0, 1.0)
    np.testing.assert_allclose(out / out[0], [1.0, 2.0, 4.0])


def test_value_after_one_half_life_keeps_permanent_and_half_of_transient():
    state = ImpactState(perm_frac=0.3, half_life_s=60.0)
    state.add(0.0, 10.0)
    # permanent 0.3 * 10 = 3, transient 7 halves to 3.5 after one half-life
    assert state.value(0.0) == pytest.approx(10.0, abs=1e-12)
    assert state.value(60.0) == pytest.approx(0.3 * 10.0 + 0.5 * 0.7 * 10.0, abs=1e-12)
    assert state.value(120.0) == pytest.approx(3.0 + 0.25 * 7.0, abs=1e-12)


def test_value_decays_to_permanent_part():
    state = ImpactState(perm_frac=0.25, half_life_s=10.0)
    state.add(0.0, -8.0)
    assert state.value(1e6) == pytest.approx(-2.0, abs=1e-12)


def test_successive_adds_decay_earlier_transient_only():
    state = ImpactState(perm_frac=0.5, half_life_s=30.0)
    state.add(0.0, 4.0)
    state.add(30.0, 2.0)
    # at t=30: permanent 2 + 1 = 3, transient 2 * 0.5 + 1 = 2; at t=60: 3 + 2 * 0.5 = 4
    assert state.value(30.0) == pytest.approx(5.0, abs=1e-12)
    assert state.value(60.0) == pytest.approx(4.0, abs=1e-12)


def test_value_does_not_change_state():
    state = ImpactState(perm_frac=0.0, half_life_s=10.0)
    state.add(0.0, 1.0)
    state.value(50.0)
    assert state.value(10.0) == pytest.approx(0.5, abs=1e-12)


def test_value_at_matches_scalar_value():
    state = ImpactState(perm_frac=0.2, half_life_s=45.0)
    state.add(5.0, 3.0)
    t = np.array([5.0, 20.0, 100.0, 500.0])
    np.testing.assert_allclose(state.value_at(t), [state.value(x) for x in t], atol=1e-12)


def test_time_going_backwards_raises():
    state = ImpactState(perm_frac=0.2, half_life_s=45.0)
    state.add(10.0, 1.0)
    with pytest.raises(ValueError, match="earlier"):
        state.add(5.0, 1.0)
    with pytest.raises(ValueError, match="earlier"):
        state.value(5.0)


def test_reset_clears_state():
    state = ImpactState(perm_frac=0.2, half_life_s=45.0)
    state.add(10.0, 1.0)
    state.reset()
    assert state.value(0.0) == 0.0
    state.add(0.0, 1.0)
    assert state.value(0.0) == 1.0


def test_reset_accepts_clock_origin():
    state = ImpactState(perm_frac=0.3, half_life_s=10.0)
    state.add(100.0, 5.0)
    state.reset(t_s=100.0)
    assert state.value(100.0) == 0.0
    state.add(100.0, 2.0)
    assert state.value(100.0) == 2.0
    assert state.value(110.0) == pytest.approx(0.3 * 2.0 + 0.5 * 0.7 * 2.0, abs=1e-12)


def test_first_add_does_not_decay_from_zero():
    state = ImpactState(perm_frac=0.5, half_life_s=10.0)
    state.add(1000.0, 4.0)
    assert state.value(1000.0) == 4.0
    assert state.value(1010.0) == pytest.approx(0.5 * 4.0 + 0.5 * 0.5 * 4.0, abs=1e-12)


def test_invalid_parameters_raise():
    with pytest.raises(ValueError):
        ImpactState(perm_frac=1.5, half_life_s=10.0)
    with pytest.raises(ValueError):
        ImpactState(perm_frac=0.5, half_life_s=0.0)
