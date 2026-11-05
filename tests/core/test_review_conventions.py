"""Acceptance tests for convention guards (m4) and notes (m5, m6)."""

import pytest

from tca.core import conventions as cv


def test_implementation_shortfall_rejects_zero_parent_quantity():
    with pytest.raises(ValueError, match="parent_qty"):
        cv.implementation_shortfall(1, [1.0], [100.0], 0.0, 100.0, 100.0)


def test_implementation_shortfall_rejects_negative_parent_quantity():
    with pytest.raises(ValueError, match="parent_qty"):
        cv.implementation_shortfall(1, [], [], -5.0, 100.0, 100.0)


def test_implementation_shortfall_rejects_overfill():
    with pytest.raises(ValueError, match="exceeds"):
        cv.implementation_shortfall(1, [120.0], [100.1], 100.0, 100.0, 101.0)


def test_implementation_shortfall_rejects_negative_fill_quantity():
    with pytest.raises(ValueError, match="negative"):
        cv.implementation_shortfall(1, [-10.0, 60.0], [100.0, 100.0], 100.0, 100.0, 100.0)


def test_implementation_shortfall_rejects_bad_side():
    with pytest.raises(ValueError, match="side"):
        cv.implementation_shortfall(0, [10.0], [100.0], 100.0, 100.0, 100.0)


def test_implementation_shortfall_full_fill_within_float_tolerance_is_accepted():
    res = cv.implementation_shortfall(1, [33.3, 66.7], [100.0, 100.0], 100.0, 100.0, 100.0)
    assert res.filled_frac == pytest.approx(1.0)


def test_implementation_shortfall_partial_fill_still_matches_c7_identity():
    res = cv.implementation_shortfall(1, [60.0], [100.1], 100.0, 100.0, 100.5)
    assert res.is_filled_bp == pytest.approx(10.0)
    assert res.opportunity_bp == pytest.approx(0.4 * 50.0)
    assert res.is_bp == pytest.approx(0.6 * 10.0 + 20.0)
