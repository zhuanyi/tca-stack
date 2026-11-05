"""Statistical helpers: bootstrap intervals, clustered regressions, shrinkage, power, fits."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import integrate, optimize
from scipy import stats as sp_stats

_MAX_RESAMPLE_CELLS = 4_000_000
_DEFAULT_SEED = 0


def bootstrap_ci(
    x,
    stat: Callable[..., float] = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap interval with coverage ``1 - alpha`` for ``stat`` of ``x``.

    ``stat`` must accept an ``axis`` keyword (``np.mean``, ``np.median``, ``np.std``). Without
    ``rng`` a generator seeded with 0 is used, so results are repeatable. Resamples are drawn in
    chunks to bound memory.
    """
    rng = rng if rng is not None else np.random.default_rng(_DEFAULT_SEED)
    values = np.asarray(x, dtype=float)
    if values.size == 0:
        raise ValueError("bootstrap_ci requires a non-empty sample")
    if not np.all(np.isfinite(values)):
        raise ValueError("bootstrap_ci requires finite values; drop NaN and inf first")
    chunk = max(1, _MAX_RESAMPLE_CELLS // len(values))
    if chunk == 1 and len(values) > _MAX_RESAMPLE_CELLS:
        raise ValueError(
            f"sample has {len(values)} rows, exceeding the cell budget of "
            f"{_MAX_RESAMPLE_CELLS}; increase _MAX_RESAMPLE_CELLS or downsample"
        )
    draws = np.concatenate(
        [
            stat(
                values[rng.integers(0, len(values), (min(chunk, n_boot - i), len(values)))], axis=1
            )
            for i in range(0, n_boot, chunk)
        ]
    )
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _fit_cluster(model, df: pd.DataFrame, cluster: str, disp: int | None = None):
    groups = df.loc[model.data.row_labels, cluster].to_numpy()
    if pd.isna(groups).any():
        raise ValueError(f"cluster column {cluster!r} has missing labels in the rows used")
    kwds = {"cov_type": "cluster", "cov_kwds": {"groups": groups}}
    if disp is not None:
        kwds["disp"] = disp
    return model.fit(**kwds)


def ols_cluster(df: pd.DataFrame, formula: str, cluster: str):
    """OLS fit of ``formula`` with standard errors clustered on column ``cluster``.

    Rows with missing values in the formula variables are dropped. Returns the statsmodels result.
    """
    return _fit_cluster(smf.ols(formula, df), df, cluster)


def logit_cluster(df: pd.DataFrame, formula: str, cluster: str):
    """Logit fit of ``formula`` with standard errors clustered on column ``cluster``."""
    return _fit_cluster(smf.logit(formula, df), df, cluster, disp=0)


def tau2_mom(xbar, n, s2) -> float:
    """Method-of-moments between-cell variance, floored at zero.

    ``xbar`` are cell means, ``n`` cell sizes and ``s2`` within-cell variances; the estimate is
    the variance of the means less the average sampling variance ``s2 / n``.
    """
    xbar, n, s2 = (np.asarray(a, dtype=float) for a in (xbar, n, s2))
    if len({a.size for a in (xbar, n, s2) if a.ndim > 0}) > 1:
        raise ValueError("tau2_mom needs xbar, n and s2 of the same length")
    if xbar.size < 2:
        raise ValueError("tau2_mom needs at least two cells")
    return float(max(0.0, np.var(xbar, ddof=1) - np.mean(s2 / n)))


def eb_shrink(xbar, n, s2, mu, tau2):
    """Empirical-Bayes posterior mean ``(n*xbar/s2 + mu/tau2) / (n/s2 + 1/tau2)``.

    ``mu`` is the prior mean and ``tau2`` the prior variance; with ``tau2 == 0`` every cell is
    shrunk fully to ``mu``.
    """
    xbar, n, s2 = (np.asarray(a, dtype=float) for a in (xbar, n, s2))
    if tau2 <= 0.0:
        return np.broadcast_to(np.asarray(mu, dtype=float), np.broadcast(xbar, n, s2).shape).copy()
    s2 = np.maximum(s2, 1e-300)
    return (n * xbar / s2 + mu / tau2) / (n / s2 + 1.0 / tau2)


def required_n(delta, sd, power: float = 0.8, alpha: float = 0.05):
    """Observations per arm to detect a mean difference ``delta`` with a two-sided two-sample test.

    ``delta`` and ``sd`` share units; uses the normal approximation and rounds up.
    """
    if np.any(np.asarray(delta) == 0):
        raise ValueError("delta must be non-zero")
    if np.any(np.asarray(sd) < 0):
        raise ValueError("sd must be non-negative")
    z = sp_stats.norm.ppf(1 - alpha / 2) + sp_stats.norm.ppf(power)
    n = np.ceil(2.0 * (z * np.asarray(sd) / np.asarray(delta)) ** 2)
    return int(n) if np.ndim(n) == 0 else n.astype(int)


def expected_max_normal(n: int) -> float:
    """Expected maximum of ``n`` independent standard normals, by numerical integration."""
    if n < 1:
        raise ValueError("n must be at least 1")
    value, _ = integrate.quad(
        lambda x: n * x * sp_stats.norm.pdf(x) * sp_stats.norm.cdf(x) ** (n - 1), -10.0, 10.0
    )
    return float(value)


EXPONENT_BOUNDS = (0.01, 3.0)


def exponent_at_bound(c: float, rel_tol: float = 1e-3) -> str | None:
    """Return ``"lower"`` or ``"upper"`` when a fitted exponent sits at a fit bound, else ``None``.

    An exponent within ``rel_tol`` of the span of ``EXPONENT_BOUNDS`` from a bound is treated as
    at the bound: the optimiser stopped at the constraint, so the value is not an estimate.
    """
    lo, hi = EXPONENT_BOUNDS
    tol = rel_tol * (hi - lo)
    if not np.isfinite(c):
        return None
    if c <= lo + tol:
        return "lower"
    if c >= hi - tol:
        return "upper"
    return None


def fit_power_law(size, cost) -> tuple[float, float, float]:
    """Least-squares fit of ``cost = a + b * size**c``; returns ``(a, b, c)``.

    The exponent is bounded to ``EXPONENT_BOUNDS`` = [0.01, 3]; ``size`` and ``cost`` share the
    units of the caller.
    """
    s = np.asarray(size, dtype=float)
    y = np.asarray(cost, dtype=float)
    if s.shape != y.shape:
        raise ValueError("size and cost must have the same length")
    if s.size < 3:
        raise ValueError("fit_power_law needs at least 3 points")
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(y))):
        raise ValueError("size and cost must be finite")
    if np.any(s <= 0):
        raise ValueError("size must be strictly positive")
    scale = s.max()
    u = s / scale

    def model(u, a, b, c):
        return a + b * u**c

    span = max(float(y.max() - y.min()), 1e-12)
    (a, b_scaled, c), _ = optimize.curve_fit(
        model,
        u,
        y,
        p0=(float(y.min()), span, 0.5),
        bounds=([-np.inf, -np.inf, EXPONENT_BOUNDS[0]], [np.inf, np.inf, EXPONENT_BOUNDS[1]]),
        maxfev=20_000,
    )
    return float(a), float(b_scaled / scale**c), float(c)


def fit_power_law_binned(
    size,
    cost,
    n_bins: int = 6,
    n_boot: int = 500,
    seed: int = 0,
) -> tuple[float, float, float, float, float]:
    """Fit ``cost = a + b * size**c`` to binned means and return ``(a, b, c, ci_lo, ci_hi)``.

    The data are grouped into ``n_bins`` equal-count quantile bins; the fit is to the bin medians
    of ``size`` and means of ``cost``. A percentile bootstrap over ``n_boot`` resamples of the
    raw ``(size, cost)`` pairs, recomputing the binned means each time with the original bin
    edges, gives a 95% interval for the exponent ``c``. ``seed`` fixes the RNG for repeatability.
    Pairs with a non-finite ``size`` or ``cost`` are dropped before binning.
    """
    s = np.asarray(size, dtype=float)
    y = np.asarray(cost, dtype=float)
    if s.shape != y.shape:
        raise ValueError("size and cost must have the same length")
    keep = np.isfinite(s) & np.isfinite(y)
    s, y = s[keep], y[keep]
    if len(s) < n_bins:
        raise ValueError(f"need at least n_bins={n_bins} points, got {len(s)}")

    # Original bin edges (quantiles of size).
    quantiles = np.quantile(s, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(quantiles)
    if len(edges) < 2:
        raise ValueError("size has no variation; cannot bin")

    def _binned_fit(s_arr: np.ndarray, y_arr: np.ndarray) -> tuple[float, float, float]:
        # Assign to the original edges; drop empty bins.
        idx = np.searchsorted(edges[1:-1], s_arr, side="right")
        bin_medians = []
        bin_means = []
        for b in range(len(edges) - 1):
            mask = idx == b
            if mask.sum() > 0:
                bin_medians.append(float(np.median(s_arr[mask])))
                bin_means.append(float(np.mean(y_arr[mask])))
        if len(bin_medians) < 3:
            return (float("nan"), float("nan"), float("nan"))
        return fit_power_law(np.array(bin_medians), np.array(bin_means))

    a, b, c = _binned_fit(s, y)

    rng = np.random.default_rng(seed)
    exponents = []
    for _ in range(n_boot):
        i = rng.integers(0, len(s), len(s))
        _, _, c_boot = _binned_fit(s[i], y[i])
        if not np.isnan(c_boot):
            exponents.append(c_boot)

    if len(exponents) < int(0.9 * n_boot):
        ci_lo, ci_hi = float("nan"), float("nan")
    else:
        ci_lo, ci_hi = (float(v) for v in np.quantile(exponents, [0.025, 0.975]))

    return float(a), float(b), float(c), ci_lo, ci_hi
