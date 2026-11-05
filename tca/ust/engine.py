"""The RFQ engine: dealer quotes, award, impact and leakage for one business day.

All per-RFQ random numbers and quote components are drawn and computed for the whole day with
array operations. Only the impact of earlier trades on later quotes forces a sequential pass,
and that pass touches a handful of small arrays per RFQ. Random draws do not depend on the
number of dealers asked, so changing ``n_dealers`` or ``leakage_scale`` on the same RFQ list
re-prices the same market (common random numbers).

Quotes stay live until the award: every responder quotes off the effective mid at the award time,
so the client holds no free option on the move during the RFQ. Latency decides who has responded
by the award and when the award happens.

Quotes stay live until the award: every responder quotes off the effective mid at the award time,
so the client holds no free option on the move during the RFQ. Latency decides who has responded
by the award and when the award happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from tca.core import bondmath
from tca.core.impact import sqrt_impact
from tca.ust import dealers as dl
from tca.ust.config import SimConfig
from tca.ust.flow import DayPlan
from tca.ust.grid import Grid, parse_hhmm
from tca.ust.ledger import Ledger
from tca.ust.market import DayMarket

STREAM_RFQ = 4
DONE, PASSED, NO_RESPONSE = "DONE", "PASSED", "NO_RESPONSE"
_MINUTE = 60.0
_IMPACT_REFERENCE_MM = 100.0


@dataclass(frozen=True)
class Knobs:
    """Run-time overrides of the configuration; ``None`` keeps the configured value."""

    leakage_scale: float
    widening_scale: float
    forced_n_dealers: int | None


@dataclass
class DayResult:
    """Tables of one simulated day (local ids) and the impact ledgers that produced them."""

    orders: pd.DataFrame
    rfq: pd.DataFrame
    quotes: pd.DataFrame
    hedges: pd.DataFrame
    markouts: pd.DataFrame
    ledgers: list[Ledger]


@dataclass
class _Outcome:
    """Per-RFQ results of the sequential pass, arrays indexed by RFQ (and dealer)."""

    selected: np.ndarray
    n_asked: np.ndarray
    axe_trade: np.ndarray
    axe_favours: np.ndarray
    responded: np.ndarray
    latency: np.ndarray
    px: np.ndarray
    ref_mid: np.ndarray
    hs: np.ndarray
    mid_send: np.ndarray
    t_award: np.ndarray
    status: np.ndarray
    winner: np.ndarray
    cover: np.ndarray
    best: np.ndarray
    hedge_parts: list


@dataclass
class _Draws:
    """Random numbers for every RFQ and dealer of the day."""

    select: np.ndarray
    respond: np.ndarray
    latency_z: np.ndarray
    noise_z: np.ndarray
    hedge: np.ndarray
    clip_lose: np.ndarray
    clip_win: np.ndarray


def _draw(rng: np.random.Generator, n_rfq: int, n_dealers: int, clips: int) -> _Draws:
    return _Draws(
        select=rng.random((n_rfq, n_dealers)),
        respond=rng.random((n_rfq, n_dealers)),
        latency_z=rng.standard_normal((n_rfq, n_dealers)),
        noise_z=rng.standard_normal((n_rfq, n_dealers)),
        hedge=rng.random((n_rfq, n_dealers)),
        clip_lose=rng.random((n_rfq, n_dealers, clips)),
        clip_win=rng.random((n_rfq, clips)),
    )


def fill_impact_32nds(market: DayMarket, row, qty_mm) -> np.ndarray:
    """Impact in 32nds of trading ``qty_mm`` in security ``row``: ``PI * sqrt(q / 100mm)``."""
    return sqrt_impact(qty_mm, _IMPACT_REFERENCE_MM, 1.0, market.impact_per_100mm_32nds[row])


def _selection(
    plan: pd.DataFrame,
    draws: _Draws,
    dealers: dl.Dealers,
    axe_favours: np.ndarray,
    cfg: SimConfig,
    knobs: Knobs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Which dealers are asked: boolean (rfq x dealer), number asked, and the axe-trade flag."""
    weights = np.array([cfg.dealers.tier_selection_weight[t] for t in dealers.col("tier")])
    key = np.log(draws.select) / weights[None, :]
    n = plan["n_dealers"].to_numpy().copy()
    axe_trade = np.zeros(len(plan), dtype=bool)
    if knobs.forced_n_dealers is not None:
        n[:] = knobs.forced_n_dealers
    else:
        axe_trade = plan["plan_axe"].to_numpy() & axe_favours.any(axis=1)
        n[axe_trade] = 1
        key = np.where(axe_trade[:, None], np.where(axe_favours, key + 1e6, -np.inf), key)
    order = np.argsort(-key, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(key.shape[1])[None, :].repeat(len(plan), 0), axis=1)
    return rank < n[:, None], n, axe_trade


def _hedge_events(
    k: int,
    winner: int,
    t_award: float,
    side: int,
    qty: float,
    selected: np.ndarray,
    dealers: dl.Dealers,
    draws: _Draws,
    market: DayMarket,
    row: int,
    cfg: SimConfig,
    knobs: Knobs,
) -> tuple[np.ndarray, ...]:
    """Hedge clips of an award: dealer index, winner flag, time, clip size, signed impact."""
    lk = cfg.leakage
    clips = lk.hedge_clips
    losers = np.flatnonzero(selected & (np.arange(len(selected)) != winner))
    losers = losers[draws.hedge[k, losers] < lk.p_hedge_lose]
    dealer = np.concatenate([[winner], losers])
    size = np.concatenate(
        [
            [(1.0 - dealers.col("internalisation")[winner]) * qty],
            np.full(len(losers), lk.phi_lose * qty),
        ]
    )
    window = np.vstack([lk.win_hedge_window_min, *[lk.lose_hedge_window_min] * len(losers)])
    u = np.vstack([draws.clip_win[k][None, :], draws.clip_lose[k, losers]])
    when = t_award + (window[:, :1] + (window[:, 1:] - window[:, :1]) * u) * _MINUTE
    clip_qty = (size / clips)[:, None].repeat(clips, 1)
    impact = side * knobs.leakage_scale * fill_impact_32nds(market, row, clip_qty)
    is_winner = (np.arange(len(dealer)) == 0)[:, None].repeat(clips, 1)
    return (
        np.repeat(dealer, clips),
        is_winner.ravel(),
        when.ravel(),
        clip_qty.ravel(),
        impact.ravel(),
    )


def run_day(
    cfg: SimConfig,
    market: DayMarket,
    dealers: dl.Dealers,
    dday: dl.DealerDay,
    plan: DayPlan,
    grid: Grid,
    seed: int,
    knobs: Knobs,
) -> DayResult:
    """Simulate every RFQ of ``plan`` in time order and build the day's tables."""
    d = market.day
    rf = plan.rfqs
    n_rfq, n_dealers = len(rf), len(dealers)
    rng = np.random.default_rng([seed, STREAM_RFQ, d.toordinal()])
    draws = _draw(rng, n_rfq, n_dealers, cfg.leakage.hedge_clips)

    row = rf["row"].to_numpy()
    side = rf["side"].to_numpy()
    qty = rf["qty_mm"].to_numpy().astype(float)
    t_send = rf["ts_send_s"].to_numpy()
    i_send = grid.index(t_send)
    hs = market.half_spread_32nds[row, i_send]

    axe = dday.axe[:, row].T
    axe_favours = axe == side[:, None]
    selected, n_asked, axe_trade = _selection(rf, draws, dealers, axe_favours, cfg, knobs)

    dc = cfg.dealers
    resp_p = dealers.col("response_prob_base")[None, :] * np.where(
        qty[:, None] > dealers.col("no_quote_size_mm")[None, :], 0.5, 1.0
    )
    responded = selected & (draws.respond < resp_p)
    latency = dealers.col("response_latency_median_s")[None, :] * np.exp(
        dc.response_latency_sigma * draws.latency_z
    )

    minute = np.minimum((t_send // _MINUTE).astype(int), dday.inventory_mm.shape[2] - 1)
    inventory = dday.inventory_mm[:, row, minute].T
    skew_price = (
        -dc.inventory.skew_coeff * hs[:, None] * inventory / dc.inventory.inventory_scale_mm
    )
    widening = (
        dealers.col("competition_widening")[None, :]
        * knobs.widening_scale
        * np.maximum(n_asked - dc.competition_free_dealers, 0)[:, None]
    )
    premium = dl.premium_32nds(
        hs[:, None],
        dealers.col("spread_mult")[None, :],
        dl.axe_adjustment(cfg, axe, side[:, None]),
        dealers.col("size_charge_k")[None, :] * np.sqrt(qty / _IMPACT_REFERENCE_MM)[:, None],
        widening,
        side[:, None] * skew_price,
        draws.noise_z * dc.quote.noise_sd_hs * hs[:, None],
        dc.quote.floor_half_spread_frac,
    )

    life = cfg.order_flow.rfq_life_cycle
    tick = dc.quote.tick_denominator
    ledgers = [
        Ledger(cfg.impact.perm_frac, cfg.impact.decay_half_life_min * _MINUTE)
        for _ in market.securities
    ]
    px = np.full((n_rfq, n_dealers), np.nan)
    mid_send = np.empty(n_rfq)
    t_award = np.full(n_rfq, np.nan)
    status = np.empty(n_rfq, dtype=object)
    winner = np.full(n_rfq, -1)
    cover = np.full(n_rfq, -1)
    best = np.full(n_rfq, np.nan)
    ref_mid = np.full((n_rfq, n_dealers), np.nan)
    hedge_parts: list[tuple[int, tuple[np.ndarray, ...]]] = []

    for k in range(n_rfq):
        ledger = ledgers[row[k]]
        shift = ledger.value(t_send[k])[0]
        mid_send[k] = market.base_px[row[k], i_send[k]] + shift
        cand = np.flatnonzero(responded[k])
        if len(cand) == 0:
            status[k] = NO_RESPONSE
            continue
        t_award[k] = t_send[k] + max(float(latency[k, cand].max()), life.min_award_wait_s)
        ta_k = min(t_award[k], float(grid.n_seconds))
        ref = market.base_px[row[k], grid.index(ta_k)] + ledger.value(ta_k)[0]
        ref_mid[k] = ref
        px[k] = dl.quote_price(ref, side[k], premium[k], tick)
        order = cand[np.lexsort((latency[k, cand], side[k] * px[k, cand]))]
        winner[k] = order[0]
        cover[k] = order[1] if len(order) > 1 else -1
        best[k] = px[k, order[0]]
        limit = life.pass_threshold_hs * hs[k] + life.pass_slack_ticks * 32.0 / tick
        if side[k] * (best[k] - ref) * 32.0 > limit:
            status[k] = PASSED
            continue
        status[k] = DONE
        ta = min(t_award[k], float(grid.n_seconds))
        ledger.add(ta, side[k] * fill_impact_32nds(market, row[k], qty[k]) / 32.0)
        parts = _hedge_events(
            k,
            int(winner[k]),
            ta,
            int(side[k]),
            qty[k],
            selected[k],
            dealers,
            draws,
            market,
            int(row[k]),
            cfg,
            knobs,
        )
        times = np.minimum(parts[2], float(grid.n_seconds))
        ledger.add(times, parts[4] / 32.0)
        hedge_parts.append((k, (*parts[:2], times, *parts[3:])))

    out = _Outcome(
        selected,
        n_asked,
        axe_trade,
        axe_favours,
        responded,
        latency,
        px,
        ref_mid,
        hs,
        mid_send,
        t_award,
        status,
        winner,
        cover,
        best,
        hedge_parts,
    )
    return _assemble(cfg, market, dealers, plan, grid, ledgers, out)


def _yield_of(market: DayMarket, row: np.ndarray, px: np.ndarray) -> np.ndarray:
    """Yields for prices in security rows ``row``; NaN where the price is NaN."""
    out = np.full(px.shape, np.nan)
    for r in np.unique(row):
        s = market.securities[r]
        mask = (row == r) & ~np.isnan(px)
        if mask.any():
            out[mask] = bondmath.yield_from_price(s.coupon, s.maturity, s.settle, px[mask])
    return out


def _mid_at(market: DayMarket, ledgers: list[Ledger], grid: Grid, row: np.ndarray, t_s: np.ndarray):
    """Effective mid (fundamental plus impact) in security rows ``row`` at times ``t_s`` seconds."""
    t = np.minimum(np.asarray(t_s, dtype=float), float(grid.n_seconds))
    out = np.empty(len(t))
    for r in np.unique(row):
        m = row == r
        out[m] = market.base_px[r, grid.index(t[m])] + ledgers[r].value(t[m])
    return out


def _assemble(
    cfg: SimConfig,
    market: DayMarket,
    dealers: dl.Dealers,
    plan: DayPlan,
    grid: Grid,
    ledgers: list[Ledger],
    o: _Outcome,
) -> DayResult:
    """Turn the sequential pass into the day's tables, reading mids from the finished ledgers."""
    px, winner, cover, best, status = o.px, o.winner, o.cover, o.best, o.status
    hs, mid_send, responded, latency = o.hs, o.mid_send, o.responded, o.latency
    d = market.day
    rf, od = plan.rfqs, plan.orders
    n_rfq = len(rf)
    row = rf["row"].to_numpy()
    side = rf["side"].to_numpy()
    cusip_of = np.array([s.cusip for s in market.securities])
    dealer_ids = dealers.table["dealer_id"].to_numpy()
    done = status == DONE
    has_quote = winner >= 0

    idx = np.arange(n_rfq)
    exec_px = np.where(done, best, np.nan)
    cover_px = np.where(cover >= 0, px[idx, np.maximum(cover, 0)], np.nan)
    ta = np.where(has_quote, o.t_award, np.nan)
    mid_award = np.full(n_rfq, np.nan)
    resolved = np.where(
        has_quote, ta, rf["ts_send_s"].to_numpy() + cfg.order_flow.rfq_life_cycle.min_award_wait_s
    )
    if has_quote.any():
        mid_award[has_quote] = _mid_at(market, ledgers, grid, row[has_quote], ta[has_quote])

    horizons = list(cfg.tca.markout_horizons_s)
    labels = [_label(h) for h in horizons] + ["close"]
    mark = {"rfq": idx[done]}
    mids = np.empty((done.sum(), len(labels)))
    for j, h in enumerate(horizons):
        mids[:, j] = _mid_at(market, ledgers, grid, row[done], ta[done] + h)
    mids[:, -1] = _mid_at(
        market, ledgers, grid, row[done], np.full(done.sum(), grid.n_seconds, dtype=float)
    )
    for j, label in enumerate(labels):
        mark[f"mid_px_{label}"] = mids[:, j]
        mark[f"mid_yld_{label}"] = _yield_of(market, row[done], mids[:, j])
    markouts = pd.DataFrame(mark)

    t_create = od["ts_create_s"].to_numpy()[rf["order"].to_numpy()]
    i_create = grid.index(t_create)
    i_send = grid.index(rf["ts_send_s"].to_numpy())
    vwap = np.empty(n_rfq)
    for r in np.unique(row):
        m = row == r
        base_mean = market.mean_base_px(r, i_create[m], i_send[m])
        vwap[m] = base_mean + ledgers[r].mean_over(t_create[m], rf["ts_send_s"].to_numpy()[m])
    close_s = grid.offset(parse_hhmm("15:00"))
    mid_close = _mid_at(market, ledgers, grid, row, np.full(n_rfq, close_s))

    rfq = pd.DataFrame(
        {
            "rfq": idx,
            "order": rf["order"].to_numpy(),
            "cusip": cusip_of[row],
            "ts_send_utc": grid.to_utc(d, rf["ts_send_s"].to_numpy()),
            "ts_award_utc": _utc_or_nat(grid, d, ta),
            "side": side,
            "qty_mm": rf["qty_mm"].to_numpy(),
            "n_dealers": o.n_asked,
            "n_responses": responded.sum(axis=1),
            "status": status,
            "winner_dealer": np.where(done, dealer_ids[np.maximum(winner, 0)], None),
            "exec_px": exec_px,
            "exec_yld": _yield_of(market, row, exec_px),
            "cover_px": cover_px,
            "best_px": np.where(has_quote, best, np.nan),
            "composite_bid": mid_send - hs / 32.0,
            "composite_ask": mid_send + hs / 32.0,
            "mid_send": mid_send,
            "mid_award": mid_award,
            "is_list": rf["is_list"].to_numpy(),
            "is_axe": o.axe_trade,
            "hs_send_32nds": hs,
            "vwap_proxy_px": vwap,
            "mid_close15_px": mid_close,
        }
    )

    rr, dd = np.nonzero(o.selected)
    ranks = _ranks(side, px, latency, responded)
    quote_px = px[rr, dd]
    quotes = pd.DataFrame(
        {
            "rfq": rr,
            "dealer_id": dealer_ids[dd],
            "responded": responded[rr, dd],
            "latency_s": np.where(responded[rr, dd], latency[rr, dd], np.nan),
            "quote_px": quote_px,
            "quote_yld": _yield_of(market, row[rr], quote_px),
            "rank": ranks[rr, dd],
            "is_winner": done[rr] & (winner[rr] == dd),
            "is_cover": done[rr] & (cover[rr] == dd),
            "was_axed": o.axe_favours[rr, dd],
            "ref_mid_px": o.ref_mid[rr, dd],
            "hs_32nds": hs[rr],
        }
    )
    hedges = _hedge_table(o.hedge_parts, dealer_ids, rf, grid, d)

    orders = _order_table(market, plan, grid, ledgers, resolved)
    return DayResult(orders, rfq, quotes, hedges, markouts, ledgers)


def _label(seconds: int) -> str:
    """Column label for a horizon: ``0s``, ``30s``, ``5m``."""
    return f"{seconds // 60}m" if seconds >= 60 and seconds % 60 == 0 else f"{seconds}s"


def _utc_or_nat(grid: Grid, d: date, t_s: np.ndarray) -> pd.DatetimeIndex:
    """UTC timestamps for offsets in seconds, ``NaT`` where the offset is NaN."""
    filled = grid.to_utc(d, np.where(np.isnan(t_s), 0.0, t_s))
    return filled.where(~np.isnan(t_s))


def _ranks(
    side: np.ndarray, px: np.ndarray, latency: np.ndarray, responded: np.ndarray
) -> np.ndarray:
    """Rank 1..m of each responding dealer's quote, best first; ties go to the faster dealer."""
    score = np.where(responded, side[:, None] * px, np.inf)
    order = np.lexsort((latency, score), axis=1)
    ranks = np.empty(order.shape)
    np.put_along_axis(
        ranks, order, np.arange(1, order.shape[1] + 1)[None, :].repeat(len(side), 0), axis=1
    )
    return np.where(responded, ranks, np.nan)


def _hedge_table(parts, dealer_ids, rf: pd.DataFrame, grid: Grid, d: date) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(
            columns=["rfq", "dealer_id", "role", "ts_utc", "qty_mm", "impact_32nds"]
        )
    k = np.concatenate([np.full(len(p[0]), i) for i, p in parts])
    cols = [np.concatenate([p[j] for _, p in parts]) for j in range(5)]
    dealer, is_winner, when, qty, impact = cols
    return pd.DataFrame(
        {
            "rfq": k,
            "dealer_id": dealer_ids[dealer.astype(int)],
            "role": np.where(is_winner, "WINNER", "LOSER"),
            "ts_utc": grid.to_utc(d, when),
            "qty_mm": qty,
            "impact_32nds": impact,
        }
    )


def _order_table(
    market: DayMarket, plan: DayPlan, grid: Grid, ledgers: list[Ledger], resolved: np.ndarray
) -> pd.DataFrame:
    """Parent orders with the decision mid at creation and the mid when the last RFQ resolved."""
    od = plan.orders
    d = market.day
    row = od["row"].to_numpy()
    rf = plan.rfqs
    end_s = (
        pd.Series(resolved).groupby(rf["order"].to_numpy()).max().reindex(od["order"]).to_numpy()
    )
    decision = _mid_at(market, ledgers, grid, row, od["ts_create_s"].to_numpy())
    end_mid = _mid_at(market, ledgers, grid, row, end_s)
    secs = market.securities
    orders = pd.DataFrame(
        {
            "order": od["order"].to_numpy(),
            "ts_create_utc": grid.to_utc(d, od["ts_create_s"].to_numpy()),
            "cusip": [secs[r].cusip for r in row],
            "tenor": [secs[r].tenor for r in row],
            "run_status": [secs[r].run_status for r in row],
            "side": od["side"].to_numpy(),
            "qty_mm": od["qty_mm"].to_numpy(),
            "policy": od["policy"].to_numpy(),
            "n_children": od["n_children"].to_numpy(),
            "trader_id": od["trader_id"].to_numpy(),
            "list_id": od["list_id"].to_numpy(),
            "decision_mid": decision,
            "ts_end_utc": grid.to_utc(d, end_s),
            "end_mid": end_mid,
        }
    )
    return orders
