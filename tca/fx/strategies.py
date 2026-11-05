"""Execution strategies: how a parent order is cut into children.

A strategy is registered under a style name and turns an :class:`~tca.fx.model.Order` into a list
of :class:`~tca.fx.model.ChildPlan`. New strategies are added with :func:`register_strategy` and are
then usable as the ``style`` of any order handed to the replay engine.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tca.core.calendar import wmr_window_utc
from tca.fx.config import DurationRule, FxConfig
from tca.fx.market import Market
from tca.fx.model import (
    AGGRESSIVE,
    FIX_INTENT,
    PASSIVE,
    RISK_TRANSFER_INTENT,
    ChildPlan,
    Order,
)
from tca.fx.ticks import US_PER_S

MIN_CHILD_USD = 1_000.0


@dataclass(frozen=True)
class StrategyContext:
    """What a strategy may look at: configuration and the market's volume proxy."""

    cfg: FxConfig
    market: Market


class Strategy(Protocol):
    """Interface of an execution strategy."""

    def planned_duration_s(self, order: Order, ctx: StrategyContext) -> float:
        """Planned execution length in seconds, from the first child to the end of the schedule."""
        ...

    def schedule(
        self, order: Order, ctx: StrategyContext, rng: np.random.Generator
    ) -> list[ChildPlan]:
        """Children of ``order``: offsets from arrival in seconds, USD quantities, intents."""
        ...


STRATEGIES: dict[str, Strategy] = {}


def register_strategy(name: str) -> Callable[[type], type]:
    """Class decorator that registers an instance of the class as the strategy of style ``name``."""

    def wrap(cls: type) -> type:
        STRATEGIES[name] = cls()
        return cls

    return wrap


def whole_dollar_split(total: float, weights: np.ndarray) -> np.ndarray:
    """Split ``total`` USD into whole-dollar parts in proportion to ``weights``; they sum to it."""
    cumulative = np.rint(np.cumsum(weights) / weights.sum() * total)
    cumulative[-1] = total
    return np.diff(cumulative, prepend=0.0)


def duration_minutes(rule: DurationRule, notional_usd: float) -> float:
    """``clip(notional_mm / notional_mm_per_minute, min, max)`` in minutes."""
    minutes = notional_usd / 1e6 / rule.notional_mm_per_minute
    return float(np.clip(minutes, rule.min_minutes, rule.max_minutes))


def almgren_chriss_holdings(n_slices: int, kappa_t: float) -> np.ndarray:
    """Remaining fraction of the order at the start of each slice and at the end, length ``n + 1``.

    ``x_k = sinh(kappa_t * (1 - k / n)) / sinh(kappa_t)``; falls back to a straight line when
    ``kappa_t`` is negligible, which is the time-weighted schedule.
    """
    progress = np.arange(n_slices + 1) / n_slices
    if kappa_t < 1e-9:
        return 1.0 - progress
    return np.sinh(kappa_t * (1.0 - progress)) / np.sinh(kappa_t)


@register_strategy("TWAP")
class Twap:
    """Equal slices at jittered intervals; each slice rests passively first."""

    def planned_duration_s(self, order: Order, ctx: StrategyContext) -> float:
        return 60.0 * duration_minutes(
            ctx.cfg.execution_styles.TWAP.duration_rule, order.notional_usd
        )

    def schedule(
        self, order: Order, ctx: StrategyContext, rng: np.random.Generator
    ) -> list[ChildPlan]:
        low, high = ctx.cfg.execution_styles.TWAP.slice_interval_s
        duration = self.planned_duration_s(order, ctx)
        offsets = [0.0]
        while (nxt := offsets[-1] + rng.uniform(low, high)) < duration:
            offsets.append(nxt)
        qty = whole_dollar_split(order.notional_usd, np.ones(len(offsets)))
        return [ChildPlan(o, float(q), PASSIVE) for o, q in zip(offsets, qty, strict=True)]


@register_strategy("POV")
class Pov:
    """Aggressive slices sized to a share of the volume proxy until filled or the cap is reached."""

    def planned_duration_s(self, order: Order, ctx: StrategyContext) -> float:
        plan = self._plan(order, ctx)
        step = ctx.cfg.execution_styles.POV.slice_interval_s
        return plan[-1].offset_s + step if plan else 0.0

    def schedule(
        self, order: Order, ctx: StrategyContext, rng: np.random.Generator
    ) -> list[ChildPlan]:
        return self._plan(order, ctx)

    @staticmethod
    def _plan(order: Order, ctx: StrategyContext) -> list[ChildPlan]:
        pov = ctx.cfg.execution_styles.POV
        cap_s = pov.max_duration_minutes * 60.0
        remaining = order.notional_usd
        plan: list[ChildPlan] = []
        offset = 0.0
        while remaining > 0 and offset < cap_s:
            ts = order.ts_arrival_us + int(offset * US_PER_S)
            rate = ctx.market.volume_rate_usd_s(order.pair, ts)
            qty = float(min(remaining, np.floor(pov.participation * rate * pov.slice_interval_s)))
            if qty >= MIN_CHILD_USD or (qty > 0 and qty == remaining):
                plan.append(ChildPlan(offset, qty, AGGRESSIVE))
                remaining -= qty
            offset += pov.slice_interval_s
        return plan


@register_strategy("IS")
class ArrivalPriceIs:
    """Almgren-Chriss trajectory of aggressive slices, front-loaded for positive risk aversion."""

    def planned_duration_s(self, order: Order, ctx: StrategyContext) -> float:
        return 60.0 * duration_minutes(
            ctx.cfg.execution_styles.IS.duration_rule, order.notional_usd
        )

    def schedule(
        self, order: Order, ctx: StrategyContext, rng: np.random.Generator
    ) -> list[ChildPlan]:
        style = ctx.cfg.execution_styles.IS
        duration = self.planned_duration_s(order, ctx)
        n = max(1, int(np.ceil(duration / style.slice_interval_s)))
        kappa_t = float(np.sqrt(style.risk_aversion * order.notional_usd))
        remaining = np.rint(almgren_chriss_holdings(n, kappa_t) * order.notional_usd)
        qty = remaining[:-1] - remaining[1:]
        step = duration / n
        return [ChildPlan(k * step, float(q), AGGRESSIVE) for k, q in enumerate(qty) if q > 0]


@register_strategy("RISK_TRANSFER")
class RiskTransfer:
    """One principal fill of the whole order at arrival."""

    def planned_duration_s(self, order: Order, ctx: StrategyContext) -> float:
        return 0.0

    def schedule(
        self, order: Order, ctx: StrategyContext, rng: np.random.Generator
    ) -> list[ChildPlan]:
        return [ChildPlan(0.0, order.notional_usd, RISK_TRANSFER_INTENT)]


@register_strategy("FIX_ORDER")
class FixOrder:
    """Equal aggressive children across the WM/R window, first child at the window start."""

    def planned_duration_s(self, order: Order, ctx: StrategyContext) -> float:
        return 60.0 * ctx.cfg.execution_styles.FIX_ORDER.duration_minutes

    def schedule(
        self, order: Order, ctx: StrategyContext, rng: np.random.Generator
    ) -> list[ChildPlan]:
        style = ctx.cfg.execution_styles.FIX_ORDER
        window_start = wmr_window_utc(order.day)[0]
        start_us = int(window_start.timestamp()) * US_PER_S + window_start.microsecond
        lead_s = (start_us - order.ts_arrival_us) / US_PER_S
        n = int(self.planned_duration_s(order, ctx) / style.aggressive_child_interval_s)
        qty = whole_dollar_split(order.notional_usd, np.ones(n))
        return [
            ChildPlan(lead_s + k * style.aggressive_child_interval_s, float(q), FIX_INTENT)
            for k, q in enumerate(qty)
        ]
