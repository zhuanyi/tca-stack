"""The numbers behind the module's findings, printed by ``tca fx summary``.

Findings in ``docs/fx.md`` are written from this output, never from the specification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tca.core.stats import bootstrap_ci, fit_power_law_binned
from tca.fx import analysis
from tca.fx.figures import STYLES, size_bucket

MULTI_CHILD_MIN_USD = 50e6


def _mean_ci(values: pd.Series, seed: int = 0) -> str:
    x = values.dropna().to_numpy()
    if len(x) < 2:
        return "n/a"
    lo, hi = bootstrap_ci(x, rng=np.random.default_rng(seed))
    return f"{x.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _block(title: str, frame: pd.DataFrame) -> list[str]:
    return ["", title, frame.to_string(index=False, float_format=lambda v: f"{v:.3f}")]


def counts(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Row counts of the simulated tables and the share of fully filled parents."""
    parents = tables["fx_tca_parent"]
    rows = {
        "parents": len(parents),
        "children": len(tables["fx_children"]),
        "lp attempts": len(tables["fx_fill_attempts"]),
        "fills": len(tables["fx_fills"]),
        "parents fully filled (%)": round(100.0 * (parents["filled_frac"] > 0.999999).mean(), 2),
    }
    return pd.DataFrame({"quantity": list(rows), "value": list(rows.values())})


def is_by_style(parents: pd.DataFrame) -> pd.DataFrame:
    """Mean IS with a 95% bootstrap interval, by style, and the same split by size bucket."""
    rows = []
    for name in [*STYLES, "FIX_ORDER"]:
        sub = parents[parents["style"] == name]
        rows.append(
            {"style": name, "bucket": "all", "n": len(sub), "is_bp": _mean_ci(sub["is_bp"])}
        )
        for bucket, group in sub.groupby(size_bucket(sub["notional_usd"]), observed=True):
            rows.append(
                {
                    "style": name,
                    "bucket": bucket,
                    "n": len(group),
                    "is_bp": _mean_ci(group["is_bp"]),
                }
            )
    return pd.DataFrame(rows)


def decomposition(parents: pd.DataFrame) -> pd.DataFrame:
    """Mean contribution of each term to ``is_bp`` by style (filled terms scaled by ``x``)."""
    x = parents["filled_frac"]
    frame = pd.DataFrame(
        {
            "style": parents["style"],
            "delay": x * parents["delay_bp"],
            "spread": x * parents["spread_bp"],
            "impact": x * parents["impact_bp"],
            "timing": x * parents["timing_bp"],
            "opportunity": parents["opportunity_bp"],
            "is_bp": parents["is_bp"],
        }
    )
    return frame.groupby("style").mean().reset_index()


def size_exponents(parents: pd.DataFrame) -> pd.DataFrame:
    """Fitted exponent ``c`` of ``a + b * size**c`` for IS against notional, per pair.

    FIX_ORDER is excluded: its IS swings with market drift from arrival to the fix window and
    is not comparable to the other styles. The fit is to binned means; the
    bootstrap 95% interval is for the exponent only, with a fixed seed.
    """
    rows = []
    filled = parents[parents["filled_frac"] > 0]
    non_fix = filled[filled["style"] != "FIX_ORDER"]
    for pair, group in non_fix.groupby("pair"):
        a, b, c, ci_lo, ci_hi = fit_power_law_binned(
            group["notional_usd"].to_numpy() / 1e6,
            group["is_bp"].to_numpy(),
            n_bins=6,
            n_boot=500,
            seed=20250303,
        )
        rows.append(
            {"pair": pair, "n": len(group), "a": a, "b": b, "c": c, "c_lo": ci_lo, "c_hi": ci_hi}
        )
    return pd.DataFrame(rows)


def brokers(parents: pd.DataFrame) -> pd.DataFrame:
    """Wheel orders by broker: IS, interval-benchmark cost and the effective-mid offset at fills.

    ``mean_impact_state_bp`` is the noise-free measure of leakage. The last two columns restrict to
    orders over 50mm with more than one fill.
    """
    wheel = parents[parents["broker"].notna() & (parents["filled_frac"] > 0)]
    big = wheel[(wheel["notional_usd"] > MULTI_CHILD_MIN_USD) & (wheel["n_fills"] > 1)]
    rows = []
    for name, group in wheel.groupby("broker"):
        large = big[big["broker"] == name]
        rows.append(
            {
                "broker": name,
                "n": len(group),
                "is_bp": _mean_ci(group["is_bp"]),
                "vs_interval_twap_bp": _mean_ci(group["cost_vs_interval_twap_bp"]),
                "impact_state_bp": group["mean_impact_state_bp"].mean(),
                "n_over_50mm": len(large),
                "is_bp_over_50mm": _mean_ci(large["is_bp"]),
            }
        )
    return pd.DataFrame(rows)


def execution(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Passive fill rate, LP reject rate and the share of fills that needed the ECN."""
    children, fills = tables["fx_children"], tables["fx_fills"]
    attempts = tables["fx_fill_attempts"]
    passive = children[children["intent"] == "PASSIVE"]
    lp = attempts[attempts["outcome"] != "ECN"]
    rows = {
        "passive children filled (%)": 100.0
        * fills["child_id"].isin(passive["child_id"]).sum()
        / len(passive),
        "LP attempts rejected (%)": 100.0 * (lp["outcome"] == "REJECTED").mean(),
        "aggressive fills by ECN (%)": 100.0
        * (attempts["outcome"] == "ECN").sum()
        / (fills["intent"] != "PASSIVE").sum(),
    }
    return pd.DataFrame({"quantity": list(rows), "value": list(rows.values())})


def summary_lines(tables: dict[str, pd.DataFrame]) -> list[str]:
    """The printed summary: every number the 30-second view of ``docs/fx.md`` draws on."""
    parents = tables["fx_tca_parent"]
    card = analysis.lp_scorecard(
        tables["fx_orders"],
        tables["fx_children"],
        tables["fx_fill_attempts"],
        tables["fx_fills"],
        tables["fx_tca_fill"],
        by=("lp_id",),
    )
    lines = ["Positive IS and costs are worse for the client; 95% bootstrap intervals in brackets."]
    lines += _block("Counts", counts(tables))
    lines += _block("Implementation shortfall (bp) by style and size", is_by_style(parents))
    lines += _block("Mean contribution to IS (bp) by style", decomposition(parents))
    lines += _block(
        "Shortfall against size, fitted a + b*size^c (size in USD mm)", size_exponents(parents)
    )
    lines += _block("Brokers (wheel orders)", brokers(parents))
    lines += _block("Execution", execution(tables))
    lines += _block(
        "LP scorecard (pooled; pips)",
        card[
            [
                "lp_id",
                "n_attempts",
                "fill_ratio",
                "reject_ratio",
                "asym_reject",
                "median_hold_ms",
                "markout_eff_pips_1",
                "markout_eff_pips_30",
                "cost_of_rejection_pips",
            ]
        ],
    )
    lines += _block("Pre-trade estimate against realised IS", analysis.pretrade_errors(parents))
    return lines
