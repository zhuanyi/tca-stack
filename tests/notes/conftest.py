"""Shared fixtures for the note tests: the analysis module and a synthetic order population.

The note tests must fit the 15-minute CI budget and must not depend on committed market tables,
so they generate a small panel with the same columns as the real analysis panel and the planted
effects read from config/brokers.yaml.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ANALYSIS_PATH = Path(__file__).resolve().parents[2] / "notes" / "04_algo_wheel" / "analysis.py"


@pytest.fixture(scope="session")
def analysis():
    """The note's analysis module, loaded from its path (the folder name is not a module name)."""
    spec = importlib.util.spec_from_file_location("algo_wheel_analysis", ANALYSIS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def book(analysis):
    return analysis.load_brokers()


@pytest.fixture(scope="session")
def panel(analysis, book):
    """A synthetic panel with the planted effects applied, 1,200 orders over 40 days."""
    return synthetic_panel(analysis, book, n=1200, seed=7)


def synthetic_panel(analysis, book, n: int = 1200, seed: int = 7) -> pd.DataFrame:
    """Build a panel shaped like analysis.add_features with the planted effects in is_bp.

    Assignment follows config/fx_sim.yaml: the RANDOM/DISCRETIONARY mode is drawn
    independently of size, and only DISCRETIONARY orders above $100mm are rerouted to A, as in
    tca/fx/orders.py. Broker A also stays in the uniform pool for the other orders, again as in
    the simulation, so A carries both the hard orders and a random sample.
    """
    rng = np.random.default_rng(seed)
    brokers = sorted(book)
    pairs = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD"]
    styles = ["TWAP", "POV", "IS"]
    days = pd.Timestamp("2025-03-03", tz="UTC") + pd.to_timedelta(rng.integers(0, 40, n), unit="D")
    # Fixed test constants: a compact intraday mix, not a copy of the configured weights. They
    # only need to span the four hour buckets the regression controls for.
    weights = np.array(
        [
            0.02,
            0.02,
            0.01,
            0.01,
            0.01,
            0.01,
            0.02,  # 0-6
            0.09,
            0.09,
            0.09,
            0.09,
            0.09,  # 7-11 London
            0.08,
            0.07,
            0.06,
            0.06,  # 12-15 overlap
            0.04,
            0.03,
            0.03,
            0.02,
            0.02,
            0.01,  # 16-21
        ]
    )
    hour = rng.choice(np.arange(0, 22), size=n, p=weights / weights.sum())
    minute = rng.integers(0, 60, n)
    ts = days + pd.to_timedelta(hour * 60 + minute, unit="m")
    pair = rng.choice(pairs, n)
    style = rng.choice(styles, n)
    notional = np.exp(rng.normal(np.log(25e6), 1.0, n))
    book["A"]  # the discretionary destination must exist; fail loudly if the config changes
    mode = rng.random(n) < 0.3
    rerouted = mode & (notional > 100e6)
    broker = np.where(rerouted, "A", rng.choice(brokers, n))
    assignment = np.where(mode, "DISCRETIONARY", "RANDOM")

    volume = {p: 600e9 if p == "EURUSD" else 400e9 for p in pairs}
    log_size = np.log(notional / np.array([volume[p] for p in pair]))
    # Impact grows with size (square-root), so A's impact_multiplier shows up as a size slope.
    impact = 0.17 * np.sqrt(notional / 25e6) * rng.lognormal(0.0, 0.3, n)
    n_fills = rng.poisson(11, n) + 1
    passive = rng.uniform(10.0, 30.0, n)

    markup = np.array([_markup(book[b], int(h)) for b, h in zip(broker, hour, strict=True)])
    own = (
        markup
        + (np.array([book[b].impact_multiplier for b in broker]) - 1.0) * impact
        + np.array([book[b].post_fill_impact_bp for b in broker]) * n_fills
    )
    base = (
        0.30
        + 0.30 * log_size
        + pd.Series(pair)
        .map({"EURUSD": -0.10, "USDJPY": 0.10, "GBPUSD": 0.20, "AUDUSD": 0.30})
        .to_numpy()
        + pd.Series(style).map({"TWAP": 0.0, "POV": -0.65, "IS": 0.10}).to_numpy()
        + rng.normal(0.0, 1.0, n)
    )
    return pd.DataFrame(
        {
            "order_id": np.arange(n),
            "is_bp": base + own,
            "broker": broker,
            "assignment": assignment,
            "pair": pair,
            "style": style,
            "notional_usd": notional,
            "ts_arrival_utc": ts,
            "log_size_rel": log_size,
            "sigma_d_bp": pd.Series(pair)
            .map({"EURUSD": 50.0, "USDJPY": 55.0, "GBPUSD": 60.0, "AUDUSD": 65.0})
            .to_numpy()
            + rng.normal(0.0, 4.0, n),
            "hs_bp_at_arrival": rng.uniform(0.15, 0.45, n),
            "hour": hour,
            "hour_bucket": pd.cut(
                hour, [0, 7, 12, 16, 24], right=False, labels=["pre", "london", "overlap", "late"]
            ).astype(str),
            "london_morning": ((hour >= 7) & (hour < 12)).astype(int),
            "day": ts.date,
            "impact_bp": impact,
            "n_fills": n_fills,
            "pct_passive_filled": passive,
            "size_bucket": pd.cut(
                notional, [0, 10e6, 50e6, np.inf], labels=["<10mm", "10-50mm", ">50mm"]
            ),
        }
    )


def _markup(cfg, hour: int) -> float:
    """Configured markup of one broker at an hour, using its window when inside it."""
    for window in cfg.markup_windows:
        start = int(window.start[:2]) * 60 + int(window.start[3:])
        end = int(window.end[:2]) * 60 + int(window.end[3:])
        if start <= hour * 60 < end:
            return window.markup_bp
    return cfg.markup_bp
