"""Client order flow: parent orders and the RFQs they are executed through.

``plan_day`` draws one business day of parents (arrival, security, side, size, execution
policy, dealers to ask). A list RFQ is stored as several single-RFQ parents that share a
``list_id`` and a send time, so that every parent has one security and one decision mid.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from tca.core.calendar import NY
from tca.ust.config import SimConfig
from tca.ust.grid import Grid, parse_hhmm
from tca.ust.inputs import Inputs
from tca.ust.market import DayMarket

STREAM_FLOW = 3
POLICIES = ("SINGLE", "SPLIT", "LIST")
_MINUTE = 60.0
_AUCTION_TYPE = "AUCTION_RESULT"


@dataclass
class DayPlan:
    """Planned parents and RFQs of one day; times are seconds since the session start.

    ``orders`` has one row per parent (``order``, ``ts_create_s``, ``row``, ``side``, ``qty_mm``,
    ``policy``, ``list_id``, ``trader_id``); ``rfqs`` one row per child (``order``, ``child``,
    ``ts_send_s``, ``row``, ``side``, ``qty_mm``, ``n_dealers``, ``is_list``, ``plan_axe``).
    """

    orders: pd.DataFrame
    rfqs: pd.DataFrame


def _auction_blackout(
    inputs: Inputs, d: date, grid: Grid, minutes: float
) -> dict[int, tuple[float, float]]:
    """Per auctioned tenor, the (start, end) seconds before the result in which no order arrives."""
    ev = inputs.events
    today = ev[(ev["type"] == _AUCTION_TYPE) & (ev["ts_utc"].dt.tz_convert(NY).dt.date == d)]
    out = {}
    for row in today.itertuples():
        end = grid.to_offset(d, row.ts_utc)
        out[int(row.tenor)] = (end - minutes * _MINUTE, end)
    return out


def _arrival_times(rng: np.random.Generator, n: int, grid: Grid, cfg: SimConfig) -> np.ndarray:
    mix = cfg.order_flow.arrival_mixture
    weights = np.array([m.weight for m in mix])
    pick = rng.choice(len(mix), size=n, p=weights / weights.sum())
    lo = np.array([grid.offset(parse_hhmm(m.from_)) for m in mix])[pick]
    hi = np.array([grid.offset(parse_hhmm(m.to)) for m in mix])[pick]
    return rng.uniform(lo, hi)


def _avoid_blackout(
    rng: np.random.Generator, t: np.ndarray, tenor: np.ndarray, windows, redraw
) -> np.ndarray:
    """Redraw arrival times that fall in the pre-auction window of their tenor."""
    out = t.copy()
    for _ in range(50):
        bad = np.zeros(len(out), dtype=bool)
        for tn, (lo, hi) in windows.items():
            bad |= (tenor == tn) & (out >= lo) & (out < hi)
        if not bad.any():
            return out
        out[bad] = redraw(int(bad.sum()))
    for tn, (lo, hi) in windows.items():
        out[(tenor == tn) & (out >= lo) & (out < hi)] = hi
    return out


def _snap(rng: np.random.Generator, size: np.ndarray, cfg: SimConfig) -> np.ndarray:
    sz = cfg.order_flow.size_mm
    values = np.array(sz.round_snap_values)
    snapped = values[np.abs(size[:, None] - values[None, :]).argmin(axis=1)]
    return np.where(rng.random(len(size)) < sz.round_snap_prob, snapped, size)


def draw_sizes(rng: np.random.Generator, tenor: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Order sizes in $mm face: lognormal by tenor, rounded, partly snapped to round numbers."""
    sz = cfg.order_flow.size_mm
    median = np.array([sz.median[t] for t in tenor])
    raw = np.round(median * np.exp(sz.lognormal_sigma * rng.standard_normal(len(tenor))))
    clipped = np.clip(raw, *sz.clip_mm)
    return np.clip(_snap(rng, clipped, cfg), *sz.clip_mm)


def default_n_dealers(qty_mm, run_status: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """Default number of dealers asked, by size and run status (before randomisation)."""
    p = cfg.order_flow.n_dealers_policy
    q = np.asarray(qty_mm, dtype=float)
    otr = run_status == "OTR"
    otr_n = np.select(
        [q < 10, q < 50, q <= 150],
        [p.otr_under_10mm, p.otr_under_50mm, p.otr_50_to_150mm],
        default=p.otr_over_150mm,
    )
    off_n = np.where(q > 25, p.off_the_run_over_25mm, p.otr_under_50mm)
    return np.where(otr, otr_n, off_n)


def jitter_n_dealers(
    rng: np.random.Generator, n: np.ndarray, n_max: int, cfg: SimConfig
) -> np.ndarray:
    """Move ``n`` by one in either direction with the configured probability, within [1, n_max]."""
    j = cfg.order_flow.n_dealers_policy.jitter
    move = rng.random(len(n)) < j["prob"]
    sign = np.where(rng.random(len(n)) < 0.5, -1, 1)
    return np.clip(n + move * sign * int(j["delta"]), 1, n_max)


def _split_sizes(qty: int, n_children: int) -> list[int]:
    base, extra = divmod(qty, n_children)
    return [base + (1 if i < extra else 0) for i in range(n_children)]


def plan_day(
    cfg: SimConfig,
    inputs: Inputs,
    market: DayMarket,
    grid: Grid,
    d: date,
    seed: int,
    n_dealers_max: int,
) -> DayPlan:
    """Draw the parents and RFQs of business day ``d`` (see the module docstring)."""
    of = cfg.order_flow
    rng = np.random.default_rng([seed, STREAM_FLOW, d.toordinal()])
    n = int(rng.poisson(of.parents_per_day_poisson))

    tenors = np.array(sorted(of.tenor_weights))
    w = np.array([of.tenor_weights[t] for t in tenors])
    tenor = rng.choice(tenors, size=n, p=w / w.sum())
    t_create = _arrival_times(rng, n, grid, cfg)
    windows = _auction_blackout(inputs, d, grid, of.no_order_before_auction_result_min)
    t_create = _avoid_blackout(
        rng, t_create, tenor, windows, lambda k: _arrival_times(rng, k, grid, cfg)
    )

    status_names = list(of.run_status_probs)
    status = rng.choice(status_names, size=n, p=list(of.run_status_probs.values()))
    available = {(s.tenor, s.run_status): i for i, s in enumerate(market.securities)}
    status = np.array(
        [s if (t, s) in available else "OTR" for t, s in zip(tenor, status, strict=True)]
    )
    row = np.array([available[(t, s)] for t, s in zip(tenor, status, strict=True)])

    bias = rng.normal(0.0, float(of.side["daily_bias_sd"]))
    lo, hi = of.side["p_buy_clip"]
    p_buy = float(np.clip(0.5 + bias, lo, hi))
    side = np.where(rng.random(n) < p_buy, 1, -1)
    qty = draw_sizes(rng, tenor, cfg).astype(int)
    policy_p = [of.execution_policy_probs[k] for k in ("single_rfq", "split", "list")]
    policy = rng.choice(len(POLICIES), size=n, p=policy_p)
    delay = of.send_delay_s["median"] * np.exp(of.send_delay_s["sigma"] * rng.standard_normal(n))
    traders = rng.integers(1, int(of.traders["n"]) + 1, size=n)

    orders, rfqs = [], []
    counter = 0
    horizon = grid.n_seconds - 5 * _MINUTE
    lo_n, hi_n = of.list_rfq["n_securities"]
    lo_c, hi_c = (int(x) for x in of.split["n_children"])
    lo_g, hi_g = of.split["spacing_min"]
    for i in range(n):
        legs = [(int(row[i]), int(qty[i]))]
        list_id = -1
        if POLICIES[policy[i]] == "LIST":
            k = int(rng.integers(lo_n, hi_n + 1))
            others = rng.choice(len(market.securities), size=k - 1, replace=False)
            sizes = draw_sizes(rng, np.array([market.securities[o].tenor for o in others]), cfg)
            legs += [(int(o), int(s)) for o, s in zip(others, sizes, strict=True)]
            list_id = counter
        n_children = 1
        if POLICIES[policy[i]] == "SPLIT":
            n_children = min(int(rng.integers(lo_c, hi_c + 1)), int(qty[i]))
        sends = [t_create[i] + delay[i]]
        for _ in range(n_children - 1):
            sends.append(sends[-1] + rng.uniform(lo_g, hi_g) * _MINUTE)
        sends = [s for s in sends if s <= horizon] or sends[:1]
        for r, q in legs:
            sizes = _split_sizes(int(q), len(sends)) if len(legs) == 1 else [int(q)]
            orders.append(
                (
                    counter,
                    t_create[i],
                    r,
                    side[i],
                    int(q),
                    POLICIES[policy[i]],
                    list_id,
                    f"T{traders[i]:02d}",
                    len(sends) if len(legs) == 1 else 1,
                )
            )
            for c, (ts, cq) in enumerate(zip(sends[: len(sizes)], sizes, strict=False)):
                rfqs.append((counter, c, ts, r, side[i], cq, len(legs) > 1))
            counter += 1

    order_df = pd.DataFrame(
        orders,
        columns=[
            "order",
            "ts_create_s",
            "row",
            "side",
            "qty_mm",
            "policy",
            "list_id",
            "trader_id",
            "n_children",
        ],
    )
    rfq_df = pd.DataFrame(
        rfqs, columns=["order", "child", "ts_send_s", "row", "side", "qty_mm", "is_list"]
    )
    status_of = np.array([market.securities[r].run_status for r in rfq_df["row"]])
    n_default = default_n_dealers(rfq_df["qty_mm"], status_of, cfg)
    rfq_df["n_dealers"] = jitter_n_dealers(rng, n_default, n_dealers_max, cfg)
    rfq_df["plan_axe"] = rng.random(len(rfq_df)) < of.n_dealers_policy.single_dealer_axe_prob
    rfq_df = rfq_df.sort_values(["ts_send_s", "order", "child"], kind="stable").reset_index(
        drop=True
    )
    return DayPlan(orders=order_df, rfqs=rfq_df)


def plan_explicit(
    rfqs: pd.DataFrame, market: DayMarket, grid: Grid, d: date, cfg: SimConfig, n_dealers_max: int
) -> DayPlan:
    """Turn an explicit RFQ list for day ``d`` into a plan with one single-RFQ parent per row.

    ``rfqs`` needs ``ts_send_utc``, ``cusip``, ``side`` and ``qty_mm``; ``n_dealers`` is optional
    (default policy) and ``is_axe`` marks single-dealer axe trades.
    """
    rows = np.array([market.row(c) for c in rfqs["cusip"]])
    ts = np.array([grid.to_offset(d, t) for t in rfqs["ts_send_utc"]])
    status = np.array([market.securities[r].run_status for r in rows])
    n = len(rfqs)
    n_default = default_n_dealers(rfqs["qty_mm"].to_numpy(), status, cfg)
    n_dealers = rfqs["n_dealers"].to_numpy() if "n_dealers" in rfqs else n_default
    order = np.arange(n)
    side = rfqs["side"].to_numpy().astype(int)
    qty = rfqs["qty_mm"].to_numpy().astype(int)
    orders = pd.DataFrame(
        {
            "order": order,
            "ts_create_s": ts,
            "row": rows,
            "side": side,
            "qty_mm": qty,
            "policy": "SINGLE",
            "list_id": -1,
            "trader_id": "T01",
            "n_children": 1,
        }
    )
    plan_axe = rfqs["is_axe"].to_numpy() if "is_axe" in rfqs else np.zeros(n, dtype=bool)
    out = pd.DataFrame(
        {
            "order": order,
            "child": 0,
            "ts_send_s": ts,
            "row": rows,
            "side": side,
            "qty_mm": qty,
            "is_list": False,
            "n_dealers": np.clip(n_dealers, 1, n_dealers_max),
            "plan_axe": plan_axe,
        }
    )
    return DayPlan(
        orders=orders,
        rfqs=out.sort_values(["ts_send_s", "order"], kind="stable").reset_index(drop=True),
    )
