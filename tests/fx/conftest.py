"""Shared fixtures for the FX tests: synthetic ticks, a small simulation, a one-shot strategy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from tca.fx import analysis, lps, pipeline
from tca.fx.config import FxConfig, load_brokers, load_config
from tca.fx.market import Market
from tca.fx.model import AGGRESSIVE, ChildPlan, Order
from tca.fx.replay import Ids, ReplayEnv
from tca.fx.simulate import Overrides, SimResult, run
from tca.fx.ticks import FixtureTickSource, TickDay, day_start_us

SEED = 20250303
ONE_SHOT = "ONE_SHOT"


@dataclass(frozen=True)
class OneShot:
    """A strategy added by name from outside the package: one aggressive child at arrival."""

    def planned_duration_s(self, order, ctx) -> float:
        return 0.0

    def schedule(self, order, ctx, rng) -> list[ChildPlan]:
        return [ChildPlan(0.0, order.notional_usd, AGGRESSIVE)]


@pytest.fixture(scope="session")
def one_shot_strategy() -> OneShot:
    return OneShot()


@pytest.fixture(scope="session")
def cfg() -> FxConfig:
    return load_config()


@pytest.fixture(scope="session")
def source() -> FixtureTickSource:
    return FixtureTickSource()


@pytest.fixture(scope="session")
def bars(cfg, source) -> pd.DataFrame:
    return pipeline.build_bars(cfg, source)


@pytest.fixture(scope="session")
def market(cfg, bars) -> Market:
    return pipeline.build_market(cfg, bars)[0]


@pytest.fixture(scope="session")
def sim(cfg, source, market) -> SimResult:
    return run(cfg, SEED, source, market, Overrides(max_days=2))


@pytest.fixture(scope="session")
def tfill(sim) -> pd.DataFrame:
    return analysis.tca_fill(sim.orders, sim.fills, sim.markouts)


@pytest.fixture(scope="session")
def parents(sim, tfill) -> pd.DataFrame:
    return analysis.tca_parent(sim.orders, sim.fills, sim.attempts, sim.children, tfill)


@pytest.fixture
def one_shot_env(cfg, market):
    """Factory for a replay environment with a single LP and the one-shot strategy."""

    def make(reject_base: float, reject_asym: float, hold_ms: float = 100.0) -> ReplayEnv:
        lp = lps.generate_lps(cfg, SEED).head(1).copy()
        lp[["reject_base", "reject_asym", "hold_median_ms", "hold_sigma"]] = [
            reject_base,
            reject_asym,
            hold_ms,
            0.0,
        ]
        lp["max_clip_usd"] = 1e12
        lp["markup_pips"] = 0.0
        return ReplayEnv(
            cfg,
            market,
            load_brokers(),
            lp,
            {p: 0.0 for p in cfg.pairs.order},
            Ids(),
            {ONE_SHOT: OneShot()},
        )

    return make


@pytest.fixture
def make_tape():
    """Factory of EURUSD ticks every 10 ms for one second from ``start_us``; the mid ramps."""

    def make(day, rising: bool, start_us: int) -> TickDay:
        n = 100
        t = start_us + np.arange(n) * 10_000
        step = 0.00001 * np.arange(n) * (1 if rising else -1)
        bid = 1.08000 + step
        return TickDay("EURUSD", day, t.astype("int64"), bid, bid + 0.00010)

    return make


@pytest.fixture
def make_order():
    """Factory of a one-shot parent order of 1mm USD in EURUSD."""

    def make(day, ts_arrival_us: int, side: int = 1) -> Order:
        return Order(
            order_id=1,
            day=day,
            pair="EURUSD",
            side=side,
            notional_usd=1e6,
            ts_arrival_us=ts_arrival_us,
            urgency="HIGH",
            style=ONE_SHOT,
            broker=None,
            assignment=None,
            trader_id=1,
            planned_duration_s=0.0,
        )

    return make


@pytest.fixture(scope="session")
def first_day(source, cfg):
    return source.days(cfg.pairs.order[0])[0]


@pytest.fixture(scope="session")
def noon_us(first_day) -> int:
    return day_start_us(first_day) + 12 * 3600 * 1_000_000
