"""The passive fill-rate claims, checked for structure on the synthetic fixture.

The full-sample numbers in docs/fx.md come from scripts/fx_fill_rate_seeds.py, which needs the
cached Dukascopy ticks and 77 days of tape and cannot run in CI. These tests assert only the
structure of the claim: crossing alone fills no more children than the calibrated hazard, a pair
whose crossing rate already clears the target gets no hazard at all, and dropping crossings
strictly lowers the passive fill rate.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from tca.fx.simulate import Overrides, run

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "fx_fill_rate_seeds.py"


@pytest.fixture(scope="module")
def seed_script():
    spec = importlib.util.spec_from_file_location("fx_fill_rate_seeds", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rates(result, pair_of: pd.Series) -> dict[str, float]:
    """Passive fill percentage of a run, pooled and per pair."""
    tables = result.tables()
    children = tables["fx_children"]
    children = children[children["intent"] == "PASSIVE"]
    filled = children["child_id"].isin(tables["fx_fills"]["child_id"])
    pairs = children["order_id"].map(pair_of)
    out = {"pooled": 100.0 * float(filled.mean())}
    for pair in sorted(pairs.dropna().unique()):
        mask = pairs == pair
        out[pair] = 100.0 * float(filled[mask].mean())
    return out


@pytest.fixture(scope="module")
def pair_of(sim) -> pd.Series:
    return sim.orders.set_index("order_id")["pair"]


@pytest.fixture(scope="module")
def baseline(sim, pair_of) -> dict[str, float]:
    return _rates(sim, pair_of)


@pytest.fixture(scope="module")
def crossing(cfg, source, market, pair_of) -> dict[str, float]:
    """The same flow with lambda_fill forced to zero, so only crossings and timeouts fill."""
    zeros = dict.fromkeys(cfg.pairs.order, 0.0)
    result = run(cfg, cfg.simulation.seed, source, market, Overrides(max_days=2, lambda_fill=zeros))
    return _rates(result, pair_of)


def test_crossing_alone_fills_fewer_children_than_the_calibrated_hazard(baseline, crossing):
    assert crossing["pooled"] < baseline["pooled"] - 1.0


def test_no_pair_fills_more_often_without_the_hazard(baseline, crossing):
    for pair, rate in crossing.items():
        assert rate <= baseline[pair] + 1e-9, pair


def test_crossing_alone_clears_the_calibration_target(cfg, crossing):
    """The target cannot be reached through the hazard when crossings alone beat it."""
    assert crossing["pooled"] > 100.0 * cfg.passive.target_fill_rate


def test_dropping_half_the_crossings_lowers_the_fill_rate(
    seed_script, cfg, source, market, pair_of, baseline
):
    zeros = dict.fromkeys(cfg.pairs.order, 0.0)
    with seed_script._crossing_haircut(cfg.simulation.seed, 0.5):
        cut = run(
            cfg, cfg.simulation.seed, source, market, Overrides(max_days=2, lambda_fill=zeros)
        )
    assert _rates(cut, pair_of)["pooled"] < baseline["pooled"] - 5.0
