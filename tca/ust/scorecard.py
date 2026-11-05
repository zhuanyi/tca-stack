"""Dealer scorecard: response, hit rate, rank, margin and markout, with a difficulty adjustment.

Hit rates depend on what a dealer is asked: large tickets, off-the-runs and crowded RFQs are
harder to win. The adjustment fits a logit of winning on those features with and without dealer
fixed effects; the fixed effects are reported and the expected wins without them give a
hit rate relative to expectation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tca.core import stats
from tca.core.calendar import NY
from tca.ust.config import SimConfig

_FEATURES = "np.log(qty_mm) + C(run_status) + C(tenor) + n_dealers"


def _bucket_of(cfg: SimConfig) -> dict[int, str]:
    return {t: name for name, tenors in cfg.tca.scorecard_tenor_buckets.items() for t in tenors}


def _responses(
    quotes: pd.DataFrame, rfq: pd.DataFrame, orders: pd.DataFrame, cfg: SimConfig
) -> pd.DataFrame:
    """Quote rows joined to their RFQ and parent, with the tenor bucket and month."""
    cols = [
        "rfq_id",
        "order_id",
        "qty_mm",
        "n_dealers",
        "ts_send_utc",
        "side",
        "exec_px",
        "cover_px",
    ]
    q = quotes.merge(rfq[cols], on="rfq_id", how="left")
    q = q.merge(orders[["order_id", "tenor", "run_status"]], on="order_id", how="left")
    q["tenor_bucket"] = q["tenor"].map(_bucket_of(cfg))
    q["month"] = q["ts_send_utc"].dt.tz_convert(NY).dt.strftime("%Y-%m")
    q["win"] = q["is_winner"].astype(int)
    q["margin_vs_cover_32nds"] = q["side"] * (q["cover_px"] - q["exec_px"]) * 32.0
    return q


def difficulty_model(responded: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Dealer fixed effects and each response's win probability without them.

    Returns a frame (``dealer_id``, ``fixed_effect``, ``se``, ``adj_hit_rate``) and the
    expected win probability of every row in ``responded`` from the model without dealer
    effects. ``fixed_effect`` is the logit difference to the first dealer; ``adj_hit_rate`` the
    mean predicted win probability over all responses if every response were this dealer's.
    """
    with_fe = stats.logit_cluster(responded, f"win ~ {_FEATURES} + C(dealer_id)", "order_id")
    without = stats.logit_cluster(responded, f"win ~ {_FEATURES}", "order_id")
    dealers = sorted(responded["dealer_id"].unique())
    rows = []
    for d in dealers:
        swapped = responded.assign(dealer_id=d)
        name = f"C(dealer_id)[T.{d}]"
        rows.append(
            (
                d,
                float(with_fe.params.get(name, 0.0)),
                float(with_fe.bse.get(name, np.nan)),
                float(np.mean(with_fe.predict(swapped))),
            )
        )
    effects = pd.DataFrame(rows, columns=["dealer_id", "fixed_effect", "se", "adj_hit_rate"])
    return effects, np.asarray(without.predict(responded))


def scorecard(
    cfg: SimConfig,
    quotes: pd.DataFrame,
    rfq: pd.DataFrame,
    orders: pd.DataFrame,
    fills: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scorecard by dealer x tenor bucket x month and the dealer fixed-effects table.

    Win-conditional statistics use only awarded RFQs. ``hit_vs_expected`` is wins divided by the
    wins the difficulty model without dealer effects expects for the same responses.
    """
    q = _responses(quotes, rfq, orders, cfg)
    q = q.merge(
        fills[["rfq_id", "markout_32nds_5m"]].rename(columns={"markout_32nds_5m": "markout_5m"}),
        on="rfq_id",
        how="left",
    )
    resp = q[q["responded"]].copy()
    effects, expected = difficulty_model(resp)
    resp["expected_win"] = expected
    wins = resp[resp["win"] == 1]
    keys = ["dealer_id", "tenor_bucket", "month"]
    out = (
        q.groupby(keys)
        .agg(
            quotes_requested=("rfq_id", "size"),
            responses=("responded", "sum"),
            median_latency_s=("latency_s", "median"),
        )
        .join(
            resp.groupby(keys).agg(
                wins=("win", "sum"),
                avg_rank=("rank", "mean"),
                expected_wins=("expected_win", "sum"),
            )
        )
        .join(
            wins.groupby(keys).agg(
                avg_margin_vs_cover_32nds=("margin_vs_cover_32nds", "mean"),
                avg_markout_5m_32nds=("markout_5m", "mean"),
            )
        )
        .reset_index()
    )
    out["response_rate"] = out["responses"] / out["quotes_requested"]
    out["hit_rate"] = out["wins"] / out["responses"]
    out["hit_vs_expected"] = out["wins"] / out["expected_wins"]
    return out.drop(columns=["expected_wins"]), effects


def dealer_overall(card: pd.DataFrame) -> pd.DataFrame:
    """One row per dealer from the scorecard: response rate, hit rate, rank, margin and markout.

    Rates are pooled counts; rank is weighted by responses, margin and markout by wins.
    """
    rows = []
    for dealer, g in card.groupby("dealer_id"):
        wins = g["wins"].fillna(0)
        resp = g["responses"]
        rows.append(
            (
                dealer,
                resp.sum() / g["quotes_requested"].sum(),
                wins.sum() / resp.sum(),
                float(np.average(g["avg_rank"], weights=resp)),
                float(np.average(g["avg_margin_vs_cover_32nds"].fillna(0), weights=wins)),
                float(np.average(g["avg_markout_5m_32nds"].fillna(0), weights=wins)),
            )
        )
    return pd.DataFrame(
        rows,
        columns=[
            "dealer_id",
            "response_rate",
            "hit_rate",
            "avg_rank",
            "margin_vs_cover_32nds",
            "markout_5m_32nds",
        ],
    ).set_index("dealer_id")
