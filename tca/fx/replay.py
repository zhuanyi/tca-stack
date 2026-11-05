"""Replay of one parent order against the tape.

The effective market is ``tape + impact_state``, where the impact state is the order's own footprint
(a square-root law in cumulative filled quantity, half permanent and half decaying) plus any broker
effect. Children are priced and filled as in the module design: aggressive children go to
liquidity providers that may reject on last look, passive children rest at the effective touch.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tca.core.calendar import wmr_window_utc
from tca.core.conventions import markout_bp, pip_size
from tca.core.impact import ImpactState
from tca.fx import brokers, lps
from tca.fx.benchmarks import fix_benchmark, twap_mid
from tca.fx.config import BrokerCfg, FxConfig
from tca.fx.market import Market
from tca.fx.model import AGGRESSIVE, PASSIVE, RISK_TRANSFER_INTENT, ChildPlan, Order
from tca.fx.passive import hazard_fill, scan_passive
from tca.fx.strategies import STRATEGIES, Strategy, StrategyContext
from tca.fx.ticks import US_PER_DAY, US_PER_S, TickDay, day_start_us

_LAST_QUOTE_GRACE_US = 60 * US_PER_S


def impact_path(
    t_s: np.ndarray, add_t_s: np.ndarray, amounts: np.ndarray, perm_frac: float, half_life_s: float
) -> np.ndarray:
    """Impact at times ``t_s`` from a list of adds, equal to :class:`~tca.core.impact.ImpactState`.

    Each add of ``amounts[k]`` at ``add_t_s[k]`` contributes ``perm_frac`` of itself forever and the
    rest decaying with half-life ``half_life_s``; adds later than the query time do not count.
    Unlike ``ImpactState.value_at`` it sees adds that happen after some of the query times.
    """
    t = np.asarray(t_s, dtype=float)
    elapsed = t[None, :] - np.asarray(add_t_s, dtype=float)[:, None]
    weight = perm_frac + (1.0 - perm_frac) * 0.5 ** (np.maximum(elapsed, 0.0) / half_life_s)
    return (np.asarray(amounts)[:, None] * weight * (elapsed >= 0.0)).sum(axis=0)


def mean_impact(
    start_s: float,
    end_s: float,
    add_t_s: np.ndarray,
    amounts: np.ndarray,
    perm_frac: float,
    half_life_s: float,
) -> float:
    """Time-weighted mean of the impact path over ``[start_s, end_s]``, exact.

    The path is the one of :func:`impact_path`. A window of zero length returns the impact at
    ``start_s``.
    """
    if end_s <= start_s:
        return float(impact_path(np.array([start_s]), add_t_s, amounts, perm_frac, half_life_s)[0])
    t_add = np.asarray(add_t_s, dtype=float)
    lo = np.maximum(start_s, t_add)
    span = np.maximum(end_s - lo, 0.0)
    tau = half_life_s / np.log(2.0)
    decay = tau * (0.5 ** ((lo - t_add) / half_life_s) - 0.5 ** ((end_s - t_add) / half_life_s))
    area = perm_frac * span + (1.0 - perm_frac) * np.where(span > 0.0, decay, 0.0)
    return float((np.asarray(amounts) * area).sum() / (end_s - start_s))


@dataclass
class Ids:
    """Run-wide counters for child and fill identifiers."""

    child: itertools.count = field(default_factory=lambda: itertools.count(1))
    fill: itertools.count = field(default_factory=lambda: itertools.count(1))


@dataclass(frozen=True)
class ReplayEnv:
    """Everything the replay needs besides the order and the tape."""

    cfg: FxConfig
    market: Market
    brokers: dict[str, BrokerCfg]
    lps: pd.DataFrame
    lambda_fill: dict[str, float]
    ids: Ids
    strategies: dict[str, Strategy] = field(default_factory=lambda: dict(STRATEGIES))


@dataclass
class ParentOutput:
    """Rows produced by one parent; timestamps are epoch microseconds until assembly."""

    children: list[dict] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)
    fills: list[dict] = field(default_factory=list)
    markouts: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


class _Replay:
    def __init__(
        self, order: Order, tape: TickDay, env: ReplayEnv, rng: np.random.Generator
    ) -> None:
        cfg = env.cfg
        self.order, self.tape, self.env, self.rng = order, tape, env, rng
        self.day0 = day_start_us(order.day)
        half_life_s = cfg.impact.half_life_minutes * 60.0
        self.state = ImpactState(cfg.impact.perm_frac, half_life_s)
        self.extra = ImpactState(1.0, half_life_s)
        self.broker = env.brokers[order.broker] if order.broker else None
        self.sigma = env.market.sigma_d(order.pair, order.day)
        self.volume = cfg.daily_volume_usd(order.pair)
        self.pip = pip_size(order.pair)
        self.latency_us = int(cfg.latency.decision_latency_ms * 1000)
        self.cum_qty = 0.0
        self.last_end_us = order.ts_arrival_us
        self.out = ParentOutput()
        self._lp = env.lps
        self._clip = self._lp["max_clip_usd"].to_numpy()
        self._markup_px = self._lp["markup_pips"].to_numpy() * self.pip
        self._add_t_s: list[float] = []
        self._add_main: list[float] = []
        self._add_extra: list[float] = []

    def _t_s(self, us: int | np.ndarray) -> float | np.ndarray:
        return (us - self.day0) / US_PER_S

    def _impact(self, us: int) -> float:
        t = self._t_s(us)
        return self.state.value(t) + self.extra.value(t)

    def _impact_array(self, us: np.ndarray) -> np.ndarray:
        t = self._t_s(us)
        return self.state.value_at(t) + self.extra.value_at(t)

    def _add_child(self, **fields) -> int:
        child_id = next(self.env.ids.child)
        self.out.children.append({"child_id": child_id, "order_id": self.order.order_id, **fields})
        return child_id

    def _record_fill(
        self,
        child_id: int,
        lp_id: int,
        intent: str,
        ts_fill_us: int,
        px: float,
        qty: float,
        tick_i: int,
        impact: float,
        lp_markup_px: float,
        broker_markup_px: float,
        n_rejects: int,
    ) -> None:
        """Store a fill, then add its impact to the order's footprint."""
        o, bid, ask = self.order, float(self.tape.bid[tick_i]), float(self.tape.ask[tick_i])
        mid = (bid + ask) / 2.0
        self.out.fills.append(
            {
                "fill_id": next(self.env.ids.fill),
                "child_id": child_id,
                "order_id": o.order_id,
                "lp_id": lp_id,
                "intent": intent,
                "ts_fill_us": ts_fill_us,
                "px": px,
                "qty_usd": qty,
                "px_tick_us": int(self.tape.t_us[tick_i]),
                "bid_tape": bid,
                "ask_tape": ask,
                "mid_tape_at_fill": mid,
                "impact_state_at_fill": impact,
                "mid_eff_at_fill": mid + impact,
                "lp_markup_px": lp_markup_px,
                "broker_markup_px": broker_markup_px,
                "n_rejects": n_rejects,
            }
        )
        impact_cfg = self.env.cfg.impact
        scale = impact_cfg.passive_impact_multiple if intent == PASSIVE else 1.0
        scale *= brokers.impact_multiplier(self.broker)
        marginal = np.sqrt((self.cum_qty + qty) / self.volume) - np.sqrt(self.cum_qty / self.volume)
        main = o.side * scale * impact_cfg.y * self.sigma * mid * marginal
        extra_bp = brokers.post_fill_impact_bp(self.broker)
        extra = o.side * extra_bp / 1e4 * mid
        t_s = self._t_s(ts_fill_us)
        self.state.add(t_s, main)
        self.extra.add(t_s, extra)
        self._add_t_s.append(t_s)
        self._add_main.append(main)
        self._add_extra.append(extra)
        self.cum_qty += qty
        self.last_end_us = ts_fill_us

    def _broker_markup_px(self, ts_us: int, mid: float) -> float:
        return brokers.markup_bp(self.broker, ts_us) / 1e4 * mid

    def _passive(self, qty: float, t_sched_us: int) -> bool:
        """Rest a child at the effective touch; True if it filled, else it timed out."""
        o, tape, cfg = self.order, self.tape, self.env.cfg
        i = int(tape.index_at(t_sched_us))
        sent_us = t_sched_us + self.latency_us
        impact = self._impact(sent_us)
        bid, ask = float(tape.bid[i]), float(tape.ask[i])
        limit = bid + impact if o.side > 0 else ask + impact
        child_id = self._add_child(
            ts_decision_us=sent_us,
            qty_usd=qty,
            intent=PASSIVE,
            limit_px=limit,
            tick_us=int(tape.t_us[i]),
            bid_tape=bid,
            ask_tape=ask,
            impact_state=impact,
            origin_child_id=None,
        )
        end_us = sent_us + int(cfg.passive.timeout_s * US_PER_S)
        scan = scan_passive(
            tape, o.side, limit, sent_us, end_us, self._impact_array, tol=self.pip / 20.0
        )
        hazard = hazard_fill(
            scan, self.env.lambda_fill[o.pair], float(self.rng.standard_exponential())
        )
        if hazard is not None:
            fill_us, tick_i = hazard
        elif scan.cross_us is not None:
            fill_us, tick_i = scan.cross_us, scan.cross_idx
        else:
            self.last_end_us = end_us
            return False
        self._record_fill(
            child_id,
            lps.PASSIVE_ID,
            PASSIVE,
            fill_us,
            limit,
            qty,
            tick_i,
            self._impact(fill_us),
            float("nan"),
            0.0,
            0,
        )
        return True

    def _aggressive(self, qty: float, t_sched_us: int, intent: str, origin: int | None) -> None:
        """Send a child to LPs best price first; after the allowed rejections cross on the ECN."""
        o, tape, cfg = self.order, self.tape, self.env.cfg
        ll = cfg.last_look
        lp_cfg = cfg.liquidity_providers
        n_lp = len(self._lp)
        view_us, tried, rejects, child_id = t_sched_us, set(), 0, None
        while True:
            i = int(tape.index_at(view_us))
            sent_us = view_us + self.latency_us
            impact = self._impact(sent_us)
            bid, ask = float(tape.bid[i]), float(tape.ask[i])
            mid = (bid + ask) / 2.0
            touch = (ask if o.side > 0 else bid) + impact
            if child_id is None:
                child_id = self._add_child(
                    ts_decision_us=sent_us,
                    qty_usd=qty,
                    intent=intent,
                    limit_px=float("nan"),
                    tick_us=int(tape.t_us[i]),
                    bid_tape=bid,
                    ask_tape=ask,
                    impact_state=impact,
                    origin_child_id=origin,
                )
            eligible = [j for j in range(n_lp) if j not in tried and self._clip[j] >= qty]
            attempt = {
                "child_id": child_id,
                "ts_sent_us": sent_us,
                "tick_us": int(tape.t_us[i]),
                "bid_tape": bid,
                "ask_tape": ask,
                "impact_state": impact,
            }
            broker_px = self._broker_markup_px(sent_us, mid)
            if rejects >= ll.max_rejections_before_ecn or not eligible:
                markup = ll.ecn_markup_pips * self.pip
                quoted = touch + o.side * markup
                self.out.attempts.append(
                    {
                        **attempt,
                        "lp_id": lps.ECN_ID,
                        "hold_ms": 0.0,
                        "quoted_px": quoted,
                        "outcome": "ECN",
                        "moved_against": False,
                    }
                )
                px = quoted + o.side * broker_px
                self._record_fill(
                    child_id,
                    lps.ECN_ID,
                    intent,
                    sent_us,
                    px,
                    qty,
                    i,
                    impact,
                    markup,
                    broker_px,
                    rejects,
                )
                return
            noise = self.rng.standard_normal(n_lp) * lp_cfg.quote_noise_pips * self.pip
            hold_z = float(self.rng.standard_normal())
            draw = float(self.rng.random())
            keys = o.side * (touch + o.side * self._markup_px + noise)
            j = min(eligible, key=lambda k: keys[k])
            quoted = touch + o.side * self._markup_px[j]
            hold_ms = float(
                self._lp["hold_median_ms"].iloc[j] * np.exp(self._lp["hold_sigma"].iloc[j] * hold_z)
            )
            hold_us = int(hold_ms * 1000)
            later = int(tape.index_at(sent_us + hold_us))
            against = bool(o.side * (tape.mid[later] - tape.mid[i]) > 0.0)
            p_reject = self._lp["reject_base"].iloc[j] + self._lp["reject_asym"].iloc[j] * against
            lp_id = int(self._lp["lp_id"].iloc[j])
            rejected = draw < p_reject
            self.out.attempts.append(
                {
                    **attempt,
                    "lp_id": lp_id,
                    "hold_ms": hold_ms,
                    "quoted_px": quoted,
                    "outcome": "REJECTED" if rejected else "FILLED",
                    "moved_against": against,
                }
            )
            if rejected:
                tried.add(j)
                rejects += 1
                view_us = sent_us + hold_us
                continue
            px = quoted + o.side * broker_px
            self._record_fill(
                child_id,
                lp_id,
                intent,
                sent_us + hold_us,
                px,
                qty,
                i,
                impact,
                float(self._markup_px[j]),
                broker_px,
                rejects,
            )
            return

    def _risk_transfer(self, qty: float, t_sched_us: int) -> None:
        """Single principal fill at ``mid +/- (hs * hs_multiple + k_rt * sqrt(Q / V) * mid)``."""
        o, tape = self.order, self.tape
        style = self.env.cfg.execution_styles.RISK_TRANSFER
        i = int(tape.index_at(t_sched_us))
        sent_us = t_sched_us + self.latency_us
        impact = self._impact(sent_us)
        bid, ask = float(tape.bid[i]), float(tape.ask[i])
        mid, half = (bid + ask) / 2.0, (ask - bid) / 2.0
        charge = style.hs_multiple * half + style.k_rt * np.sqrt(qty / self.volume) * mid
        child_id = self._add_child(
            ts_decision_us=sent_us,
            qty_usd=qty,
            intent=RISK_TRANSFER_INTENT,
            limit_px=float("nan"),
            tick_us=int(tape.t_us[i]),
            bid_tape=bid,
            ask_tape=ask,
            impact_state=impact,
            origin_child_id=None,
        )
        px = mid + impact + o.side * charge
        self._record_fill(
            child_id,
            lps.PRINCIPAL_ID,
            RISK_TRANSFER_INTENT,
            sent_us,
            px,
            qty,
            i,
            impact,
            float(charge - half),
            0.0,
            0,
        )

    def run(self) -> ParentOutput:
        """Execute the schedule and fill in the order summary and the markouts."""
        o, tape, cfg = self.order, self.tape, self.env.cfg
        ctx = StrategyContext(cfg, self.env.market)
        plan: list[ChildPlan] = self.env.strategies[o.style].schedule(o, ctx, self.rng)
        last_tick = int(tape.t_us[-1])
        for child in plan:
            t_sched = max(o.ts_arrival_us + round(child.offset_s * US_PER_S), self.last_end_us)
            if t_sched > last_tick:
                break
            if child.intent == PASSIVE:
                if self._passive(child.qty_usd, t_sched):
                    continue
                passive_id = self.out.children[-1]["child_id"]
                self._aggressive(child.qty_usd, self.last_end_us, AGGRESSIVE, passive_id)
            elif child.intent == RISK_TRANSFER_INTENT:
                self._risk_transfer(child.qty_usd, t_sched)
            else:
                self._aggressive(child.qty_usd, t_sched, child.intent, None)
        self._summarise()
        self._markouts()
        return self.out

    def _summarise(self) -> None:
        o, tape = self.order, self.tape
        decision_mid = float(tape.mid[tape.index_at(o.ts_arrival_us)])
        out = self.out
        first_us = out.children[0]["ts_decision_us"] if out.children else o.ts_arrival_us
        planned_end = first_us + int(o.planned_duration_s * US_PER_S)
        end_us = min(max(planned_end, self.last_end_us), int(tape.t_us[-1]))
        fix_mid = float("nan")
        if o.style == "FIX_ORDER":
            start, end = wmr_window_utc(o.day)
            window = (int(start.timestamp()) * US_PER_S, int(end.timestamp()) * US_PER_S)
            fix_mid = fix_benchmark(tape, window)
        out.summary = {
            "order_id": o.order_id,
            "decision_mid": decision_mid,
            "first_decision_us": first_us,
            "first_decision_mid": float(tape.mid[tape.index_at(first_us)]),
            "ts_end_us": end_us,
            "end_mid": float(tape.mid[tape.index_at(end_us)]),
            "twap_mid": twap_mid(tape, first_us, max(planned_end, first_us)),
            "fix_mid": fix_mid,
            "interval_twap_eff_mid": self._interval_twap_eff_mid(first_us),
        }

    def _interval_twap_eff_mid(self, first_us: int) -> float:
        """Time-weighted effective mid from the first child's decision to the last fill.

        ``nan`` for an order with no fill. This is the benchmark of ``cost_vs_interval_twap_bp``.
        """
        if not self.out.fills:
            return float("nan")
        end_us = max(max(f["ts_fill_us"] for f in self.out.fills), first_us)
        tape_mid = twap_mid(self.tape, first_us, end_us)
        half_life_s = self.env.cfg.impact.half_life_minutes * 60.0
        add_t = np.asarray(self._add_t_s)
        args = (self._t_s(first_us), self._t_s(end_us), add_t)
        perm = self.env.cfg.impact.perm_frac
        footprint = mean_impact(*args, np.asarray(self._add_main), perm, half_life_s)
        leakage = mean_impact(*args, np.asarray(self._add_extra), 1.0, half_life_s)
        return tape_mid + footprint + leakage

    def _markouts(self) -> None:
        """Markouts in bp of each fill at the configured horizons, tape and effective mid."""
        fills = self.out.fills
        if not fills:
            return
        horizons = np.asarray(self.env.cfg.simulation.markout_horizons_s, dtype=np.int64)
        fill_us = np.array([f["ts_fill_us"] for f in fills], dtype=np.int64)
        when = fill_us[:, None] + horizons[None, :] * US_PER_S
        tape = self.tape
        mid_tape = tape.mid[tape.index_at(when)]
        half_life_s = self.env.cfg.impact.half_life_minutes * 60.0
        t_s = self._t_s(when).ravel()
        add_t = np.asarray(self._add_t_s)
        impact = impact_path(
            t_s, add_t, np.asarray(self._add_main), self.env.cfg.impact.perm_frac, half_life_s
        ) + impact_path(t_s, add_t, np.asarray(self._add_extra), 1.0, half_life_s)
        mid_eff = mid_tape + impact.reshape(when.shape)
        day_end = self.day0 + US_PER_DAY
        valid = (when <= int(tape.t_us[-1]) + _LAST_QUOTE_GRACE_US) & (when < day_end)
        mid_tape = np.where(valid, mid_tape, np.nan)
        mid_eff = np.where(valid, mid_eff, np.nan)
        side, px = self.order.side, np.array([f["px"] for f in fills])[:, None]
        tape_bp = markout_bp(side, mid_tape, px)
        eff_bp = markout_bp(side, mid_eff, px)
        for k, f in enumerate(fills):
            row = {"fill_id": f["fill_id"]}
            for h, horizon in enumerate(horizons):
                row[f"markout_tape_bp_{horizon}"] = tape_bp[k, h]
                row[f"markout_eff_bp_{horizon}"] = eff_bp[k, h]
            self.out.markouts.append(row)


def replay_parent(
    order: Order, tape: TickDay, env: ReplayEnv, rng: np.random.Generator
) -> ParentOutput:
    """Replay ``order`` against ``tape`` (the order's pair on the order's day).

    Args:
        order: the parent.
        tape: ticks of ``order.pair`` on ``order.day``.
        env: configuration, LPs, brokers, calibrated passive hazard and id counters.
        rng: the order's random stream; draws are consumed in a fixed pattern that does not depend
            on the broker, so two runs that differ only in broker see the same market.

    Returns:
        Children, LP attempts, fills, markout mids and an order summary, all timestamps in epoch
        microseconds. A child that starts after the last tick of the day is dropped and its
        quantity stays unfilled.
    """
    return _Replay(order, tape, env, rng).run()
