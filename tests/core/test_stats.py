import numpy as np
import pandas as pd
import pytest

from tca.core import stats


def test_bootstrap_ci_deterministic_given_rng():
    x = np.random.default_rng(1).normal(2.0, 1.0, 300)
    a = stats.bootstrap_ci(x, rng=np.random.default_rng(7))
    b = stats.bootstrap_ci(x, rng=np.random.default_rng(7))
    assert a == b
    assert stats.bootstrap_ci(x) == stats.bootstrap_ci(x)


def test_bootstrap_ci_brackets_mean_and_has_sensible_width():
    x = np.random.default_rng(2).normal(5.0, 2.0, 2000)
    lo, hi = stats.bootstrap_ci(x, rng=np.random.default_rng(3))
    se = 2.0 / np.sqrt(2000)
    assert lo < x.mean() < hi
    assert (hi - lo) == pytest.approx(2 * 1.96 * se, rel=0.2)


def test_bootstrap_ci_other_statistic_and_chunking(monkeypatch):
    monkeypatch.setattr(stats, "_MAX_RESAMPLE_CELLS", 10_000)
    x = np.random.default_rng(4).normal(0.0, 1.0, 500)
    lo, hi = stats.bootstrap_ci(x, stat=np.median, n_boot=500, rng=np.random.default_rng(5))
    assert lo < np.median(x) < hi


@pytest.mark.parametrize("empty", [[], np.array([])])
def test_bootstrap_ci_empty_sample_raises(empty):
    with pytest.raises(ValueError, match="non-empty"):
        stats.bootstrap_ci(empty)


def test_ols_cluster_matches_hand_sandwich():
    rng = np.random.default_rng(10)
    g = np.repeat(np.arange(40), 5)
    x = rng.normal(size=200)
    y = 1.0 + 2.0 * x + rng.normal(size=40)[g] + rng.normal(size=200)
    df = pd.DataFrame({"y": y, "x": x, "g": g})
    res = stats.ols_cluster(df, "y ~ x", "g")

    X = np.column_stack([np.ones(200), x])
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    resid = y - X @ beta
    meat = sum(np.outer(s, s) for s in (X[g == k].T @ resid[g == k] for k in range(40)))
    n_g, n, k = 40, 200, 2
    cov = bread @ meat @ bread * n_g / (n_g - 1) * (n - 1) / (n - k)
    np.testing.assert_allclose(res.params.to_numpy(), beta, atol=1e-10)
    np.testing.assert_allclose(res.bse.to_numpy(), np.sqrt(np.diag(cov)), atol=1e-10)


def test_ols_cluster_handles_missing_rows():
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.5, 4.0, np.nan, 6.2, 7.1, 8.3],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "g": [0, 0, 1, 1, 2, 2, 3, 3],
        }
    )
    res = stats.ols_cluster(df, "y ~ x", "g")
    assert int(res.nobs) == 7


def test_logit_cluster_recovers_sign():
    rng = np.random.default_rng(11)
    g = np.repeat(np.arange(60), 10)
    x = rng.normal(size=600)
    p = 1 / (1 + np.exp(-(0.2 + 1.0 * x)))
    df = pd.DataFrame({"y": (rng.random(600) < p).astype(int), "x": x, "g": g})
    res = stats.logit_cluster(df, "y ~ x", "g")
    assert res.params["x"] > 0.5
    assert res.pvalues["x"] < 0.001


def test_tau2_mom_hand_value():
    # var([1, 2, 3], ddof=1) = 1; mean(s2 / n) = 4 / 10 = 0.4; tau2 = 0.6
    assert stats.tau2_mom([1.0, 2.0, 3.0], [10, 10, 10], [4.0, 4.0, 4.0]) == pytest.approx(0.6)


def test_tau2_mom_floors_at_zero():
    assert stats.tau2_mom([1.0, 1.01, 0.99], [10, 10, 10], [4.0, 4.0, 4.0]) == 0.0


def test_eb_shrink_hand_value():
    # (10 * 2 / 4 + 0 / 1) / (10 / 4 + 1) = 5 / 3.5
    assert stats.eb_shrink(2.0, 10, 4.0, 0.0, 1.0) == pytest.approx(5 / 3.5)
    # n = 1: (1 * 2 / 4) / (1 / 4 + 1) = 0.5 / 1.25
    assert stats.eb_shrink(2.0, 1, 4.0, 0.0, 1.0) == pytest.approx(0.4)


def test_eb_shrink_moves_small_n_more_than_large_n():
    small = abs(2.0 - stats.eb_shrink(2.0, 3, 4.0, 0.0, 1.0))
    large = abs(2.0 - stats.eb_shrink(2.0, 300, 4.0, 0.0, 1.0))
    assert small > large


def test_eb_shrink_zero_tau2_returns_prior_mean():
    out = stats.eb_shrink(np.array([1.0, 5.0]), np.array([10, 20]), 4.0, 2.5, 0.0)
    np.testing.assert_array_equal(out, [2.5, 2.5])


def test_required_n_hand_value_and_monotone_in_delta():
    # 2 * ((1.95996 + 0.84162) / 0.5) ** 2 = 62.79 -> 63 per arm
    assert stats.required_n(0.5, 1.0) == 63
    assert stats.required_n(0.25, 1.0) > stats.required_n(0.5, 1.0)
    assert stats.required_n(0.5, 1.0, power=0.9) > stats.required_n(0.5, 1.0)


@pytest.mark.parametrize(
    ("n", "expected"), [(2, 0.56), (3, 0.85), (4, 1.03), (5, 1.16), (7, 1.35), (10, 1.54)]
)
def test_expected_max_normal_table(n, expected):
    assert round(stats.expected_max_normal(n), 2) == expected


def test_expected_max_normal_one_is_zero():
    assert stats.expected_max_normal(1) == pytest.approx(0.0, abs=1e-9)


def test_fit_power_law_recovers_parameters():
    size = np.logspace(0, 3, 40)
    cost = 1.5 + 0.8 * size**0.5
    a, b, c = stats.fit_power_law(size, cost)
    assert (a, b, c) == pytest.approx((1.5, 0.8, 0.5), abs=1e-4)


def test_fit_power_law_noisy_exponent_in_range():
    rng = np.random.default_rng(6)
    size = np.logspace(0, 3, 400)
    cost = 0.5 + 0.3 * size**0.5 + rng.normal(0, 0.05, 400)
    _, _, c = stats.fit_power_law(size, cost)
    assert 0.4 < c < 0.6
