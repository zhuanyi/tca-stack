"""Shared fixtures: the committed inputs, a three-day simulation and an explicit-RFQ factory."""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd
import pytest

from tca.core.calendar import NY, local_to_utc
from tca.ust import posttrade, simulate
from tca.ust.config import load_config
from tca.ust.inputs import load_inputs

SEED = 20250303


@pytest.fixture(scope="session")
def seed() -> int:
    return SEED


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def inputs():
    return load_inputs()


@pytest.fixture(scope="session")
def three_days(cfg, inputs):
    return inputs.business_days(cfg.period.start, cfg.period.end)[:3]


@pytest.fixture(scope="session")
def sim(cfg, inputs, three_days):
    """Simulator tables and TCA tables for the first three business days."""
    tables = simulate.run(cfg, SEED, simulate.Overrides(days=three_days), inputs)
    tca = posttrade.build(cfg, {**tables, "ust_securities": inputs.securities})
    return {**tables, **tca}


@pytest.fixture(scope="session")
def explicit_rfqs(inputs):
    """Factory of RFQ lists: ``n_per_day`` 10y on-the-run tickets an hour apart on each day."""

    def make(days, n_per_day: int, seed: int, sizes=(25.0,)) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        uni = inputs.universe
        rows = []
        for d in days:
            cusip = uni[(uni["date"] == d) & (uni["tenor"] == 10) & (uni["run_status"] == "OTR")][
                "cusip"
            ].iloc[0]
            for k in range(n_per_day):
                ts = local_to_utc(d, time(9 + k * 40 // 60, (k * 40) % 60), NY)
                rows.append((ts, cusip, int(rng.choice([-1, 1])), float(rng.choice(sizes))))
        return pd.DataFrame(rows, columns=["ts_send_utc", "cusip", "side", "qty_mm"])

    return make
