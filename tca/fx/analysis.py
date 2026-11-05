"""The TCA layer: fill-level costs, parent-level implementation shortfall and the LP scorecard.

Sign convention (``tca.core.conventions``): costs are positive when worse for the client. Markouts
are positive when the market moved in the client's favour after the fill. Two families exist:
``markout_tape_bp_<h>`` is measured against the historical tape mid and so contains only market
noise, ``markout_eff_bp_<h>`` against the effective mid ``tape + impact_state`` and so contains the
order's own footprint. ``<h>`` is the horizon in seconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tca.core import conventions as conv
from tca.fx.lps import ECN_ID

ORDER_KEYS = [
    "order_id",
    "pair",
    "side",
    "style",
    "urgency",
    "broker",
    "assignment",
    "trader_id",
    "notional_usd",
    "ts_arrival_utc",
    "planned_duration_s",
]


def horizons_from(markouts: pd.DataFrame) -> list[int]:
    """Horizons in seconds present in a wide markout table, ascending."""
    prefix = "markout_tape_bp_"
    return sorted(int(c.removeprefix(prefix)) for c in markouts if c.startswith(prefix))


def _pip_sizes(pairs: pd.Series) -> np.ndarray:
    lookup = {p: conv.pip_size(p) for p in pairs.unique()}
    return pairs.map(lookup).to_numpy(dtype=float)


def tca_fill(orders: pd.DataFrame, fills: pd.DataFrame, markouts: pd.DataFrame) -> pd.DataFrame:
    """Cost and markouts of every fill (table ``fx_tca_fill``).

    ``spread_cost_bp`` and ``spread_cost_pips`` compare the price with the tape mid at the pricing
    tick; ``effective_spread_bp`` is twice the absolute distance in bp of mid;
    ``realised_spread_bp_5`` is minus twice the effective 5-second markout. ``n_rejects`` is the
    number of rejections before the fill.
    """
    f = fills.merge(orders[["order_id", "pair", "side", "style", "broker"]], on="order_id")
    f = f.merge(markouts, on="fill_id")
    side = f["side"].to_numpy()
    px = f["px"].to_numpy()
    mid = f["mid_tape_at_fill"].to_numpy()
    out = pd.DataFrame(
        {
            "fill_id": f["fill_id"],
            "order_id": f["order_id"],
            "child_id": f["child_id"],
            "pair": f["pair"],
            "side": f["side"],
            "style": f["style"],
            "broker": f["broker"],
            "lp_id": f["lp_id"],
            "intent": f["intent"],
            "qty_usd": f["qty_usd"],
            "px": f["px"],
            "ts_fill_utc": f["ts_fill_utc"],
            "spread_cost_bp": conv.spread_cost_bp(side, px, mid),
            "spread_cost_pips": side * (px - mid) / _pip_sizes(f["pair"]),
            "effective_spread_bp": 2.0 * np.abs(px - mid) / mid * 1e4,
            "n_rejects": f["n_rejects"],
        }
    )
    for col in markouts.columns.drop("fill_id"):
        out[col] = f[col].to_numpy()
    out["realised_spread_bp_5"] = -2.0 * out["markout_eff_bp_5"]
    return out


def _weighted_nanmean(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna()
    total = weights[valid].sum()
    return float((values[valid] * weights[valid]).sum() / total) if total > 0 else float("nan")


def _decomposition_terms(fills: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Fill-level decomposition terms in bp of the decision mid, weighted by ``q_i / Q_filled``."""
    keys = ["order_id", "side", "decision_mid", "first_decision_mid"]
    f = fills.merge(orders[keys], on="order_id")
    f["filled_total"] = f.groupby("order_id")["qty_usd"].transform("sum")
    w = f["qty_usd"] / f["filled_total"]
    scale = w * f["side"] / f["decision_mid"] * 1e4
    return pd.DataFrame(
        {
            "order_id": f["order_id"],
            "delay_bp": scale * (f["first_decision_mid"] - f["decision_mid"]),
            "spread_bp": scale * (f["px"] - f["mid_eff_at_fill"]),
            "impact_bp": scale * (f["mid_eff_at_fill"] - f["mid_tape_at_fill"]),
            "timing_bp": scale * (f["mid_tape_at_fill"] - f["first_decision_mid"]),
            "filled_usd": f["qty_usd"],
            "px_x_qty": f["px"] * f["qty_usd"],
            "passive_usd": np.where(f["intent"] == "PASSIVE", f["qty_usd"], 0.0),
            "impact_x_qty": f["qty_usd"]
            * f["side"]
            * f["impact_state_at_fill"]
            / f["mid_tape_at_fill"]
            * 1e4,
        }
    )


def _implementation_shortfall(out: pd.DataFrame, fills: pd.DataFrame) -> pd.DataFrame:
    """``is_filled_bp``, ``opportunity_bp``, ``is_bp`` and ``filled_frac`` from the core formula."""
    by_order = {
        k: (g["qty_usd"].to_numpy(), g["px"].to_numpy()) for k, g in fills.groupby("order_id")
    }
    empty = (np.empty(0), np.empty(0))
    results = [
        conv.implementation_shortfall(
            int(r.side),
            *by_order.get(r.order_id, empty),
            float(r.notional_usd),
            float(r.decision_mid),
            float(r.end_mid),
        )
        for r in out.itertuples()
    ]
    return pd.DataFrame(
        {
            "is_filled_bp": [r.is_filled_bp for r in results],
            "opportunity_bp": [r.opportunity_bp for r in results],
            "is_bp": [r.is_bp for r in results],
            "filled_frac": [r.filled_frac for r in results],
        },
        index=out.index,
    )


PARENT_COLUMNS = [
    "decision_mid",
    "avg_fill_px",
    "filled_frac",
    "is_bp",
    "is_filled_bp",
    "delay_bp",
    "spread_bp",
    "impact_bp",
    "timing_bp",
    "opportunity_bp",
    "twap_bench_bp",
    "cost_vs_interval_twap_bp",
    "fix_bench_bp",
    "reversion_30m_bp",
    "pct_passive_filled",
    "reject_rate",
    "n_lps_used",
    "n_fills",
    "mean_impact_state_bp",
    "pretrade_est_bp",
]


def tca_parent(
    orders: pd.DataFrame,
    fills: pd.DataFrame,
    attempts: pd.DataFrame,
    children: pd.DataFrame,
    tfill: pd.DataFrame,
) -> pd.DataFrame:
    """Implementation shortfall of every parent and its decomposition (``fx_tca_parent``).

    With ``D`` the decision mid, ``m0`` the tape mid at the first child's decision, ``w = q / Q_f``
    the share of fill ``i`` in the filled quantity ``Q_f`` and sums over fills, every term is in bp
    of ``D``, positive when worse for the client::

        delay   = sum w * side * (m0 - D)
        spread  = sum w * side * (px - mid_eff)
        impact  = sum w * side * (mid_eff - mid_tape)
        timing  = sum w * side * (mid_tape - m0)

    The four sum to ``is_filled_bp``, the cost of the filled quantity against ``D``. With
    ``x = Q_f / Q``, ``opportunity_bp = (1 - x) * side * (end_mid - D) / D`` and
    ``is_bp = x * is_filled_bp + opportunity_bp``, the total every figure uses.

    ``cost_vs_interval_twap_bp`` is the average fill price against the time-weighted effective mid
    from the first decision to the last fill, in bp of that benchmark; the benchmark moves with the
    order's own impact. ``twap_bench_bp`` and ``fix_bench_bp`` use the tape mid over the planned
    window and the fix proxy. ``pct_passive_filled`` is the percentage of filled USD that filled
    passively, ``reject_rate`` the share of LP attempts rejected, ``reversion_30m_bp`` the
    quantity-weighted change of the effective markout from fill to 30 minutes and
    ``mean_impact_state_bp`` the quantity-weighted effective-mid offset at the fills, in bp of the
    tape mid, in the trade direction.
    """
    extra = ["pretrade_est_bp", "decision_mid", "first_decision_mid", "end_mid", "twap_mid"]
    o = orders[ORDER_KEYS + extra + ["fix_mid", "interval_twap_eff_mid"]]
    summed = _decomposition_terms(fills, o).groupby("order_id", sort=True).sum()
    out = o.merge(summed, left_on="order_id", right_index=True, how="left")
    out[summed.columns] = out[summed.columns].fillna(0.0)
    traded = out["filled_usd"].where(out["filled_usd"] > 0, np.nan)
    out["avg_fill_px"] = out["px_x_qty"] / traded
    out["pct_passive_filled"] = 100.0 * out["passive_usd"] / traded
    out["mean_impact_state_bp"] = out["impact_x_qty"] / traded
    out["n_fills"] = out["order_id"].map(fills.groupby("order_id").size()).fillna(0).astype("int64")
    out = out.join(_implementation_shortfall(out, fills))
    out["twap_bench_bp"] = conv.cost_bp(out["side"], out["avg_fill_px"], out["twap_mid"])
    out["cost_vs_interval_twap_bp"] = conv.cost_bp(
        out["side"], out["avg_fill_px"], out["interval_twap_eff_mid"]
    )
    out["fix_bench_bp"] = conv.cost_bp(out["side"], out["avg_fill_px"], out["fix_mid"])

    rev = tfill.assign(rev=tfill["markout_eff_bp_1800"] - tfill["markout_eff_bp_0"])
    out["reversion_30m_bp"] = out["order_id"].map(
        {k: _weighted_nanmean(g["rev"], g["qty_usd"]) for k, g in rev.groupby("order_id")}
    )
    lp_attempts = attempts[attempts["lp_id"] != ECN_ID].merge(
        children[["child_id", "order_id"]], on="child_id"
    )
    reject_rate = lp_attempts.groupby("order_id")["outcome"].apply(
        lambda x: (x == "REJECTED").mean()
    )
    out["reject_rate"] = out["order_id"].map(reject_rate)
    lps_used = fills[fills["lp_id"] > 0].groupby("order_id")["lp_id"].nunique()
    out["n_lps_used"] = out["order_id"].map(lps_used).fillna(0).astype("int64")
    return out[ORDER_KEYS + PARENT_COLUMNS]


SCORECARD_KEYS = ("lp_id", "pair", "month")


def lp_scorecard(
    orders: pd.DataFrame,
    children: pd.DataFrame,
    attempts: pd.DataFrame,
    fills: pd.DataFrame,
    tfill: pd.DataFrame,
    by: tuple[str, ...] = SCORECARD_KEYS,
) -> pd.DataFrame:
    """Per LP (and ``by`` further keys): fill and reject behaviour, hold time, markups, markouts.

    ``by`` is a subset of ``lp_id``, ``pair`` and ``month`` (``YYYY-MM`` of the send time) and
    always contains ``lp_id``. ``asym_reject`` is the rejection rate on attempts where the tape mid
    moved against the LP during the hold minus the rate where it did not. ``cost_of_rejection``
    averages, over rejected attempts, how much worse the eventual fill of the child was than the
    rejected quote, in bp of the quote and in pips. Adverse selection by last look is a fraction of
    a pip, so pips are the unit the scorecard reports it in: ``markout_eff_pips_<h>`` is the mean
    effective markout of the LP's fills in pips. The ECN fallback is not an LP and is excluded.
    """
    a = attempts[attempts["lp_id"] != ECN_ID].merge(
        children[["child_id", "order_id"]], on="child_id"
    )
    a = a.merge(orders[["order_id", "pair", "side"]], on="order_id")
    a["month"] = a["ts_sent_utc"].dt.strftime("%Y-%m")
    a["final_px"] = a["child_id"].map(fills.groupby("child_id")["px"].last())
    a["rejected"] = a["outcome"] == "REJECTED"
    slip = a["side"] * (a["final_px"] - a["quoted_px"])
    a["slip_bp"] = np.where(a["rejected"], slip / a["quoted_px"] * 1e4, np.nan)
    a["slip_pips"] = np.where(a["rejected"], slip / _pip_sizes(a["pair"]), np.nan)

    keys = list(by)
    grouped = a.groupby(keys, sort=True)
    out = pd.DataFrame(
        {
            "n_attempts": grouped.size(),
            "fill_ratio": 1.0 - grouped["rejected"].mean(),
            "reject_ratio": grouped["rejected"].mean(),
            "median_hold_ms": grouped["hold_ms"].median(),
            "cost_of_rejection_bp": grouped["slip_bp"].mean(),
            "cost_of_rejection_pips": grouped["slip_pips"].mean(),
        }
    )
    against = a[a["moved_against"]].groupby(keys)["rejected"].mean()
    neutral = a[~a["moved_against"]].groupby(keys)["rejected"].mean()
    out["asym_reject"] = (against - neutral).reindex(out.index)
    peers = [k for k in keys if k != "lp_id"]
    total = out.groupby(peers)["n_attempts"].transform("sum") if peers else out["n_attempts"].sum()
    out["share_of_attempts"] = out["n_attempts"] / total

    t = tfill[tfill["lp_id"] > 0].copy()
    t["month"] = t["ts_fill_utc"].dt.strftime("%Y-%m")
    t = t.merge(fills[["fill_id", "lp_markup_px"]], on="fill_id")
    pip = _pip_sizes(t["pair"])
    t["markup_pips"] = t["lp_markup_px"] / pip
    for h in (1, 30):
        t[f"markout_eff_pips_{h}"] = t[f"markout_eff_bp_{h}"] * t["px"] / 1e4 / pip
    mo = t.groupby(keys, sort=True).agg(
        avg_markup_pips=("markup_pips", "mean"),
        markout_eff_bp_1=("markout_eff_bp_1", "mean"),
        markout_eff_bp_30=("markout_eff_bp_30", "mean"),
        markout_eff_pips_1=("markout_eff_pips_1", "mean"),
        markout_eff_pips_30=("markout_eff_pips_30", "mean"),
    )
    return out.join(mo).reset_index()


def pretrade_errors(parents: pd.DataFrame) -> pd.DataFrame:
    """Bias and RMSE of the pre-trade estimate against realised ``is_bp``, by style.

    Bias is the mean of ``pretrade_est_bp - is_bp`` in bp (positive: the estimate was too high);
    RMSE is the root mean square of the same difference. Only parents with a fill count.
    """
    filled = parents[parents["filled_frac"] > 0]
    err = filled["pretrade_est_bp"] - filled["is_bp"]
    grouped = err.groupby(filled["style"])
    return pd.DataFrame(
        {
            "n": grouped.size(),
            "bias_bp": grouped.mean(),
            "rmse_bp": np.sqrt((err**2).groupby(filled["style"]).mean()),
        }
    ).reset_index()
