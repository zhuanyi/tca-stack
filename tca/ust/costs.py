"""Post-trade TCA: per-fill costs and markouts, and parent-order implementation shortfall.

Every function takes the simulator tables and returns a new frame. Costs are positive when the
execution is worse than the benchmark; markouts are positive when the market moved in the
trader's favour after the fill (the opposite of the sign many desks use).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from tca.core import bondmath, conventions
from tca.core.calendar import NY
from tca.ust.config import SimConfig

DONE = "DONE"
_BP = 1e4


def horizon_label(seconds: int) -> str:
    """Column label for a markout horizon: ``0s``, ``30s``, ``5m``."""
    return f"{seconds // 60}m" if seconds >= 60 and seconds % 60 == 0 else f"{seconds}s"


def horizon_labels(cfg: SimConfig) -> list[str]:
    """Markout horizon labels in column order, ending with ``close``."""
    return [horizon_label(h) for h in cfg.tca.markout_horizons_s] + ["close"]


def _settle(trade_date: date, issue: date) -> date:
    return max(bondmath.settle_date(trade_date), issue)


def _yields(frame: pd.DataFrame, price_col: str, securities: pd.DataFrame) -> np.ndarray:
    """Yield of ``frame[price_col]`` for each row's security on its trade date."""
    sec = securities.set_index("cusip")
    out = np.full(len(frame), np.nan)
    keys = frame[["cusip", "trade_date"]].astype(str).agg("|".join, axis=1)
    for key in keys.unique():
        mask = (keys == key).to_numpy() & frame[price_col].notna().to_numpy()
        if not mask.any():
            continue
        cusip = key.split("|")[0]
        s = sec.loc[cusip]
        settle = _settle(frame.loc[mask, "trade_date"].iloc[0], s["issue_date"])
        out[mask] = bondmath.yield_from_price(
            s["coupon"], s["maturity_date"], settle, frame.loc[mask, price_col].to_numpy()
        )
    return out


def _dv01_usd(frame: pd.DataFrame, yld: np.ndarray, securities: pd.DataFrame) -> np.ndarray:
    """DV01 in USD of each row's face amount (``qty_mm``) at the yield ``yld``."""
    sec = securities.set_index("cusip")
    out = np.full(len(frame), np.nan)
    keys = frame[["cusip", "trade_date"]].astype(str).agg("|".join, axis=1)
    for key in keys.unique():
        mask = (keys == key).to_numpy()
        s = sec.loc[key.split("|")[0]]
        settle = _settle(frame.loc[mask, "trade_date"].iloc[0], s["issue_date"])
        face = frame.loc[mask, "qty_mm"].to_numpy() * 1e6
        out[mask] = bondmath.dv01(s["coupon"], s["maturity_date"], settle, yld[mask], face=face)
    return out


def fill_costs(
    cfg: SimConfig,
    orders: pd.DataFrame,
    rfq: pd.DataFrame,
    markouts: pd.DataFrame,
    securities: pd.DataFrame,
) -> pd.DataFrame:
    """One row per filled RFQ: benchmarks, four cost measures, quality statistics and markouts.

    Benchmarks at or before ``ts_send`` are ``mid_send`` (arrival), the composite bid and ask,
    ``best_px``, ``cover_px`` and ``vwap_proxy_px`` (time-weighted mid from parent creation to
    send). ``mid_award`` and ``mid_close15_px`` are post-trade and are not used for cost.
    """
    fills = rfq[rfq["status"] == DONE].copy()
    fills = fills.merge(
        orders[["order_id", "tenor", "run_status", "policy", "trader_id", "decision_mid"]],
        on="order_id",
        how="left",
    )
    local = fills["ts_send_utc"].dt.tz_convert(NY)
    fills["trade_date"] = local.dt.date
    fills["ny_time"] = local.dt.strftime("%H:%M:%S")
    fills["bucket_15m"] = (local.dt.hour * 60 + (local.dt.minute // 15) * 15).astype(int)
    fills["month"] = local.dt.strftime("%Y-%m")

    side, px, mid = (
        fills["side"].to_numpy(),
        fills["exec_px"].to_numpy(),
        fills["mid_send"].to_numpy(),
    )
    exec_yld = fills["exec_yld"].to_numpy()
    mid_yld = _yields(fills, "mid_send", securities)
    notional = fills["qty_mm"].to_numpy() * cfg.tca.face_usd_per_mm
    fills["cost_px_bp"] = conventions.cost_bp(side, px, mid)
    fills["cost_32nds"] = conventions.cost_32nds(side, px, mid)
    fills["cost_yld_bp"] = conventions.cost_yld_bp(side, exec_yld, mid_yld)
    fills["cost_usd"] = conventions.cost_usd(fills["cost_px_bp"].to_numpy(), notional)
    fills["dv01_usd"] = _dv01_usd(fills, exec_yld, securities)
    fills["cost_dv01_usd"] = fills["cost_yld_bp"] * fills["dv01_usd"]

    hs = fills["hs_send_32nds"].to_numpy()
    fills["spread_capture_pct"] = 100.0 * (1.0 - fills["cost_32nds"].to_numpy() / hs)
    fills["cost_vs_best_32nds"] = conventions.cost_32nds(side, px, fills["best_px"].to_numpy())
    fills["winner_margin_32nds"] = side * (fills["cover_px"].to_numpy() - px) * 32.0
    fills["response_rate"] = fills["n_responses"] / fills["n_dealers"]
    fills["award_latency_s"] = (fills["ts_award_utc"] - fills["ts_send_utc"]).dt.total_seconds()
    fills["cost_vwap_proxy_32nds"] = conventions.cost_32nds(
        side, px, fills["vwap_proxy_px"].to_numpy()
    )

    marks = markouts.set_index("rfq_id").loc[fills["rfq_id"]].reset_index(drop=True)
    for label in horizon_labels(cfg):
        later = marks[f"mid_px_{label}"].to_numpy()
        fills[f"markout_32nds_{label}"] = side * (later - px) * 32.0
        fills[f"markout_bp_{label}"] = conventions.markout_bp(side, later, px)
        fills[f"markout_yld_bp_{label}"] = (
            side * (exec_yld - marks[f"mid_yld_{label}"].to_numpy()) * _BP
        )
    fills["reversion_30m_32nds"] = conventions.reversion(
        fills["markout_32nds_30m"].to_numpy(), fills["markout_32nds_0s"].to_numpy()
    )
    return fills.drop(columns=["trade_date"]).reset_index(drop=True)


def parent_shortfall(
    cfg: SimConfig,
    orders: pd.DataFrame,
    rfq: pd.DataFrame,
    fills: pd.DataFrame,
    securities: pd.DataFrame,
) -> pd.DataFrame:
    """Implementation shortfall of each parent against its decision mid.

    With ``q_i`` the fill sizes, ``Q_f`` the filled quantity and ``x = Q_f / Q``, all terms in bp
    of the decision mid:

    * ``delay_cost_bp``: first RFQ send mid against the decision mid (the first send counts
      even when that RFQ was passed);
    * ``timing_cost_bp``: ``q_i / Q_f``-weighted drift of the send mids after the first send;
    * ``execution_cost_bp``: ``q_i / Q_f``-weighted fill price against the send mid;
    * ``is_filled_bp = delay + timing + execution`` and ``is_bp = x * is_filled_bp +
      opportunity_cost_bp``.

    Parents with no fill have NaN for the filled-quantity terms and ``is_bp`` equal to their
    opportunity cost.
    """
    first_send = (
        rfq.sort_values(["order_id", "ts_send_utc", "rfq_id"], kind="stable")
        .groupby("order_id")["mid_send"]
        .first()
        .rename("mid_send_1")
    )
    parent = orders.join(first_send, on="order_id")
    f = fills.merge(parent[["order_id", "mid_send_1"]], on="order_id")
    f["w"] = f["qty_mm"] / f.groupby("order_id")["qty_mm"].transform("sum")
    f["timing_term"] = f["w"] * f["side"] * (f["mid_send"] - f["mid_send_1"])
    f["execution_term"] = f["w"] * f["side"] * (f["exec_px"] - f["mid_send"])
    f["avg_term"] = f["w"] * f["exec_px"]
    agg = f.groupby("order_id").agg(
        filled_qty_mm=("qty_mm", "sum"),
        avg_fill_px=("avg_term", "sum"),
        timing_sum=("timing_term", "sum"),
        execution_sum=("execution_term", "sum"),
    )
    parent = parent.join(agg, on="order_id")
    parent["filled_qty_mm"] = parent["filled_qty_mm"].fillna(0.0)
    scale = _BP / parent["decision_mid"]
    side = parent["side"]
    parent["filled_frac"] = parent["filled_qty_mm"] / parent["qty_mm"]
    parent["delay_cost_bp"] = np.where(
        parent["filled_qty_mm"] > 0,
        side * (parent["mid_send_1"] - parent["decision_mid"]) * scale,
        np.nan,
    )
    parent["timing_cost_bp"] = parent["timing_sum"] * scale
    parent["execution_cost_bp"] = parent["execution_sum"] * scale
    parent["is_filled_bp"] = (
        parent["delay_cost_bp"] + parent["timing_cost_bp"] + parent["execution_cost_bp"]
    )
    parent["opportunity_cost_bp"] = (
        (1.0 - parent["filled_frac"]) * side * (parent["end_mid"] - parent["decision_mid"]) * scale
    )
    parent["is_bp"] = (
        parent["filled_frac"] * parent["is_filled_bp"].fillna(0.0) + parent["opportunity_cost_bp"]
    )
    parent = parent.drop(columns=["timing_sum", "execution_sum", "mid_send_1"])
    return _add_dv01(cfg, parent, securities)


def _add_dv01(cfg: SimConfig, parent: pd.DataFrame, securities: pd.DataFrame) -> pd.DataFrame:
    """Add the parent's DV01 and its cost expressed in yield bp and in DV01 dollars."""
    p = parent.copy()
    p["trade_date"] = p["ts_create_utc"].dt.tz_convert(NY).dt.date
    yld = _yields(p.assign(decision_mid=p["decision_mid"]), "decision_mid", securities)
    p["dv01_usd"] = _dv01_usd(p, yld, securities)
    price_dv01_per_100 = p["dv01_usd"] / (p["qty_mm"] * cfg.tca.face_usd_per_mm) * 100.0
    p["is_yld_bp"] = p["is_bp"] / _BP * p["decision_mid"] / price_dv01_per_100
    p["cost_dv01_usd"] = p["is_yld_bp"] * p["dv01_usd"]
    p["is_usd"] = conventions.cost_usd(p["is_bp"], p["qty_mm"] * cfg.tca.face_usd_per_mm)
    return p.drop(columns=["trade_date"])
