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
    chunk = max(1, _MAX_RESAMPLE_CELLS // len(values))
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


def _fit_cluster(model, df: pd.DataFrame, cluster: str):
    groups = df.loc[model.data.row_labels, cluster].to_numpy()
    return model.fit(cov_type="cluster", cov_kwds={"groups": groups})


def ols_cluster(df: pd.DataFrame, formula: str, cluster: str):
    """OLS fit of ``formula`` with standard errors clustered on column ``cluster``.

    Rows with missing values in the formula variables are dropped. Returns the statsmodels result.
    """
    return _fit_cluster(smf.ols(formula, df), df, cluster)


def logit_cluster(df: pd.DataFrame, formula: str, cluster: str):
    """Logit fit of ``formula`` with standard errors clustered on column ``cluster``."""
    return _fit_cluster(smf.logit(formula, df), df, cluster)


def tau2_mom(xbar, n, s2) -> float:
    """Method-of-moments between-cell variance, floored at zero.

    ``xbar`` are cell means, ``n`` cell sizes and ``s2`` within-cell variances; the estimate is
    the variance of the means less the average sampling variance ``s2 / n``.
    """
    xbar, n, s2 = (np.asarray(a, dtype=float) for a in (xbar, n, s2))
    return float(max(0.0, np.var(xbar, ddof=1) - np.mean(s2 / n)))


def eb_shrink(xbar, n, s2, mu, tau2):
    """Empirical-Bayes posterior mean ``(n*xbar/s2 + mu/tau2) / (n/s2 + 1/tau2)``.

    ``mu`` is the prior mean and ``tau2`` the prior variance; with ``tau2 == 0`` every cell is
    shrunk fully to ``mu``.
    """
    xbar, n, s2 = (np.asarray(a, dtype=float) for a in (xbar, n, s2))
    if tau2 <= 0.0:
        return np.broadcast_to(np.asarray(mu, dtype=float), np.broadcast(xbar, n, s2).shape).copy()
    return (n * xbar / s2 + mu / tau2) / (n / s2 + 1.0 / tau2)


def required_n(delta, sd, power: float = 0.8, alpha: float = 0.05):
    """Observations per arm to detect a mean difference ``delta`` with a two-sided two-sample test.

    ``delta`` and ``sd`` share units; uses the normal approximation and rounds up.
    """
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


def fit_power_law(size, cost) -> tuple[float, float, float]:
    """Least-squares fit of ``cost = a + b * size**c``; returns ``(a, b, c)``.

    The exponent is bounded to [0.01, 3]; ``size`` and ``cost`` share the units of the caller.
    """
    s = np.asarray(size, dtype=float)
    y = np.asarray(cost, dtype=float)
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
        bounds=([-np.inf, -np.inf, 0.01], [np.inf, np.inf, 3.0]),
        maxfev=20_000,
    )
    return float(a), float(b_scaled / scale**c), float(c)
