"""Acceptance tests for stats guards (m8, m9)."""

import numpy as np
import pandas as pd
import pytest

from tca.core import stats


def test_bootstrap_ci_rejects_nan_and_inf():
    for bad in ([1.0, 2.0, np.nan], [1.0, np.inf, 3.0]):
        with pytest.raises(ValueError, match="finite"):
            stats.bootstrap_ci(bad)


def test_tau2_mom_needs_two_cells():
    with pytest.raises(ValueError, match="two cells"):
        stats.tau2_mom([1.0], [10], [1.0])


def test_tau2_mom_two_cells_still_works():
    assert stats.tau2_mom([0.0, 4.0], [100, 100], [1.0, 1.0]) == pytest.approx(8.0 - 0.01)


def test_eb_shrink_zero_within_variance_returns_the_cell_mean():
    assert stats.eb_shrink(1.5, 10, 0.0, 0.0, 1.0) == pytest.approx(1.5)


def test_required_n_rejects_zero_effect_and_negative_sd():
    with pytest.raises(ValueError, match="delta"):
        stats.required_n(0.0, 1.0)
    with pytest.raises(ValueError, match="sd"):
        stats.required_n(1.0, -1.0)


def test_required_n_known_value_unchanged():
    assert stats.required_n(1.0, 1.0) == 16


def _frame():
    rng = np.random.default_rng(1)
    n = 60
    df = pd.DataFrame({"x": rng.normal(size=n), "g": np.repeat(np.arange(12.0), 5)})
    df["y"] = 0.5 * df["x"] + rng.normal(size=n)
    df["b"] = (df["y"] > 0).astype(int)
    return df


def test_ols_cluster_rejects_missing_cluster_label():
    df = _frame()
    df.loc[3, "g"] = np.nan
    with pytest.raises(ValueError, match="cluster"):
        stats.ols_cluster(df, "y ~ x", "g")


def test_logit_cluster_rejects_missing_cluster_label():
    df = _frame()
    df.loc[3, "g"] = np.nan
    with pytest.raises(ValueError, match="cluster"):
        stats.logit_cluster(df, "b ~ x", "g")


def test_logit_cluster_prints_nothing(capsys):
    stats.logit_cluster(_frame(), "b ~ x", "g")
    assert capsys.readouterr().out == ""


def test_fit_power_law_rejects_non_positive_size():
    with pytest.raises(ValueError, match="size"):
        stats.fit_power_law([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="size"):
        stats.fit_power_law([-1.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])


def test_fit_power_law_rejects_too_few_or_non_finite_points():
    with pytest.raises(ValueError, match="at least 3"):
        stats.fit_power_law([1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        stats.fit_power_law([1.0, 2.0, 3.0, 4.0], [1.0, np.nan, 3.0, 4.0])
    with pytest.raises(ValueError, match="same length"):
        stats.fit_power_law([1.0, 2.0, 3.0], [1.0, 2.0])


def test_tau2_mom_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        stats.tau2_mom([0.0, 4.0, 8.0], [100, 100], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="same length"):
        stats.tau2_mom([0.0, 4.0], [100, 100], [1.0])
