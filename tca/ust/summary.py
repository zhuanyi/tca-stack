"""The numbers quoted in ``docs/ust.md``, computed from the committed tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tca.core import storage
from tca.core.stats import bootstrap_ci, exponent_at_bound, fit_power_law_binned
from tca.ust import regime, scorecard
from tca.ust.calibrate import derive_elasticities
from tca.ust.config import SimConfig

_APRIL_WEEK = pd.Timestamp("2025-04-07")
_SIZE_EDGES = [0, 10, 25, 50, 100, 500]
_SIZE_LABELS = ["<=10", "10-25", "25-50", "50-100", "100-500"]


def _fmt(frame: pd.DataFrame, digits: int = 3) -> str:
    return frame.round(digits).to_string()


def cost_by_tenor(fills: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Fill count, median and mean cost (32nds) with a 95% CI on the mean, by tenor and run."""
    rows = []
    for (tenor, run), g in fills.groupby(["tenor", "run_status"]):
        x = g["cost_32nds"].dropna().to_numpy()
        lo, hi = bootstrap_ci(x, rng=rng)
        rows.append((tenor, run, len(x), np.median(x), x.mean(), lo, hi))
    columns = ["tenor", "run_status", "n", "median", "mean", "mean_lo", "mean_hi"]
    return pd.DataFrame(rows, columns=columns).set_index(["tenor", "run_status"])


SIZE_FIT_BINS = 8


def size_fits(fills: pd.DataFrame) -> pd.DataFrame:
    """Fitted ``a + b * size^c`` of 10y cost in 32nds against size in $mm, by run status.

    The same binned-means fit with bootstrap interval that ``ust_02`` draws (``n_bins`` is
    ``SIZE_FIT_BINS`` in both); ``off_the_run`` is the first off-the-run only, as in the figure.
    ``c_bound`` names the bound the exponent sits at, if any.
    """
    rows = []
    for run in ("OTR", "OLD"):
        g = fills[(fills["tenor"] == 10) & (fills["run_status"] == run)]
        a, b, c, lo, hi = fit_power_law_binned(g["qty_mm"], g["cost_32nds"], n_bins=SIZE_FIT_BINS)
        rows.append((run, len(g), a, b, c, lo, hi, exponent_at_bound(c) or "-"))
    columns = ["run_status", "n", "a", "b", "c", "c_lo", "c_hi", "c_bound"]
    return pd.DataFrame(rows, columns=columns).set_index("run_status")


def _mean_ci(x, rng: np.random.Generator) -> tuple[int, float, float, float]:
    v = np.asarray(x, dtype=float)
    v = v[~np.isnan(v)]
    lo, hi = bootstrap_ci(v, rng=rng)
    return len(v), float(v.mean()), lo, hi


def cost_by_size(fills: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Mean 10y on-the-run cost (32nds) with a 95% CI by size bucket ($mm face)."""
    g = fills[(fills["tenor"] == 10) & (fills["run_status"] == "OTR")]
    bucket = pd.cut(g["qty_mm"], _SIZE_EDGES, labels=_SIZE_LABELS)
    rows = {k: _mean_ci(v["cost_32nds"], rng) for k, v in g.groupby(bucket, observed=True)}
    return pd.DataFrame.from_dict(rows, orient="index", columns=["n", "mean", "lo", "hi"])


def is_by_policy(parents: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Mean parent implementation shortfall (bp) with a 95% CI by execution policy."""
    rows = {k: _mean_ci(v["is_bp"], rng) for k, v in parents.groupby("policy")}
    return pd.DataFrame.from_dict(rows, orient="index", columns=["n", "mean", "lo", "hi"])


def markout_by_n(fills: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Mean 5-minute markout (32nds) with a 95% CI by dealers asked, 10y OTR $10mm to $50mm."""
    g = fills[
        (fills["tenor"] == 10) & (fills["run_status"] == "OTR") & fills["qty_mm"].between(10, 50)
    ]
    rows = {k: _mean_ci(v["markout_32nds_5m"], rng) for k, v in g.groupby("n_dealers")}
    return pd.DataFrame.from_dict(rows, orient="index", columns=["n", "mean", "lo", "hi"])


def april_week(weekly: pd.DataFrame) -> pd.DataFrame:
    """The week of 2025-04-07 against the mean of the other weeks, for fills and parents."""
    stats = ["median", "mean", "p10", "p90"]
    cols = [f"{p}_{s}" for p in ("cost", "is") for s in stats] + ["realised_vol_bp"]
    is_april = weekly["week"] == _APRIL_WEEK
    out = pd.DataFrame({"april_week": weekly.loc[is_april, cols].iloc[0]})
    out["other_weeks_mean"] = weekly.loc[~is_april, cols].mean()
    return out


def _april_ci(weekly: pd.DataFrame) -> list[float]:
    row = weekly.loc[weekly["week"] == _APRIL_WEEK, ["cost_mean_lo", "cost_mean_hi"]].iloc[0]
    return row.round(3).tolist()


def render(cfg: SimConfig, derived: Path | None = None) -> str:
    """Plain-text summary of simulation size, costs, the April check and the dealer scorecard."""
    t = {n: storage.read_table(n, derived) for n in _TABLES}
    fills, parents, orders, rfq = (
        t["ust_tca_fill"],
        t["ust_tca_parent"],
        t["ust_orders"],
        t["ust_rfq"],
    )
    rng = np.random.default_rng(cfg.seed)
    weekly = regime.weekly(cfg, fills, t["ust_daily_anchor"], parents)
    weekly10 = regime.weekly_otr10(cfg, fills, t["ust_daily_anchor"], parents)
    hot = regime.high_vol_weeks(weekly10)
    pool = regime.on_the_run_pool(fills, (2, 5, 10))
    tenth = regime.on_the_run_pool(fills, (10,))
    out = [
        f"parents {len(orders):,}  rfqs {len(rfq):,}  fills {len(fills):,}  "
        f"quote rows {len(t['ust_rfq_quotes']):,}  hedge trades {len(t['ust_hedges']):,}",
        "rfq status: " + ", ".join(f"{k} {v:,}" for k, v in rfq["status"].value_counts().items()),
        f"median spread capture {fills['spread_capture_pct'].median():.1f}%  "
        f"mean winner margin vs cover {fills['winner_margin_32nds'].mean():.3f} 32nds  "
        f"median award latency {fills['award_latency_s'].median():.1f}s",
        "",
        "COST BY TENOR (32nds vs arrival mid)",
        _fmt(cost_by_tenor(fills, rng)),
        "",
        "10y COST VS SIZE FIT a + b*size^c",
        _fmt(size_fits(fills)),
        "",
        "10y OTR COST BY SIZE (32nds)",
        _fmt(cost_by_size(fills, rng)),
        "",
        "PARENT IS BY EXECUTION POLICY (bp)",
        _fmt(is_by_policy(parents, rng)),
        "",
        "5-MINUTE MARKOUT BY DEALERS ASKED, 10y OTR $10-50mm (32nds)",
        _fmt(markout_by_n(fills, rng)),
        "",
        "APRIL WEEK (2025-04-07) VS OTHER WEEKS, 10y OTR (cost 32nds, parent IS bp)",
        _fmt(april_week(weekly10)),
        f"mean cost 95% CI, april week: {_april_ci(weekly10)}",
        "n fills / parents in the april week: "
        f"{int(weekly10.loc[weekly10['week'] == _APRIL_WEEK, 'cost_n'].iloc[0])} / "
        f"{int(weekly10.loc[weekly10['week'] == _APRIL_WEEK, 'is_n'].iloc[0])}",
        "",
        "APRIL WEEK VS OTHER WEEKS, ALL TENORS AND RUN STATUSES (composition not adjusted)",
        _fmt(april_week(weekly)),
        "",
        f"HIGH-VOL WEEKS (realised 10y vol above {regime.HIGH_VOL_BP} bp/day): "
        + ", ".join(f"{w:%Y-%m-%d}" for w in hot),
        "GAP, HIGH-VOL MINUS CALM WEEKS, BY SIZE: 10y OTR, cost in 32nds",
        _fmt(regime.size_gap(tenth, hot, "cost_32nds", rng)),
        "GAP BY SIZE: OTR 2y+5y+10y pooled, cost per composite half-spread",
        _fmt(regime.size_gap(pool, hot, "cost_per_hs", rng)),
        "GAP BY SIZE: OTR 2y+5y+10y pooled, cost in 32nds",
        _fmt(regime.size_gap(pool, hot, "cost_32nds", rng)),
        f"fills in high-vol weeks: 10y OTR {int(tenth['week'].isin(hot).sum())}, "
        f"pooled {int(pool['week'].isin(hot).sum())}",
        "",
        "",
        "PARENT IS DECOMPOSITION, mean bp (filled parents)",
        _fmt(
            parents.loc[
                parents["filled_qty_mm"] > 0,
                ["delay_cost_bp", "timing_cost_bp", "execution_cost_bp", "is_filled_bp"],
            ]
            .mean()
            .to_frame("mean_bp")
        ),
        f"is_bp mean {parents['is_bp'].mean():.3f}  "
        f"filled fraction {parents['filled_frac'].mean():.3f}  "
        f"opportunity mean {parents['opportunity_cost_bp'].mean():.3f}",
        "",
        "APRIL PEAK VS READINGS",
        _fmt(regime.april_peak(cfg, derived).drop(columns="sim_peak_date")),
        "elasticities " + str(derive_elasticities(t["ust_daily_anchor"], cfg)),
        "",
        "DEALER SCORECARD (pooled)",
        _fmt(scorecard.dealer_overall(t["ust_dealer_scorecard"])),
    ]
    return "\n".join(out)


_TABLES = (
    "ust_tca_fill",
    "ust_tca_parent",
    "ust_orders",
    "ust_rfq",
    "ust_rfq_quotes",
    "ust_hedges",
    "ust_daily_anchor",
    "ust_dealer_scorecard",
)
