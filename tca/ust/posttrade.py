"""The ``tca ust tca`` step: read the simulator tables, write the post-trade TCA tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tca.core import storage
from tca.ust import costs, scorecard
from tca.ust.config import SimConfig

SIMULATOR_TABLES = (
    "ust_securities",
    "ust_orders",
    "ust_rfq",
    "ust_rfq_quotes",
    "ust_hedges",
    "ust_rfq_markouts",
    "ust_bars_1m",
    "ust_dealers",
)
TCA_TABLES = ("ust_tca_fill", "ust_tca_parent", "ust_dealer_scorecard", "ust_dealer_effects")


def read_tables(names: tuple[str, ...], derived: Path | None = None) -> dict[str, pd.DataFrame]:
    """Read the named tables from ``derived`` (default ``<data>/derived``)."""
    return {name: storage.read_table(name, derived) for name in names}


def build(cfg: SimConfig, tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Per-fill costs, per-parent shortfall and the dealer scorecard from the simulator tables."""
    fills = costs.fill_costs(
        cfg,
        tables["ust_orders"],
        tables["ust_rfq"],
        tables["ust_rfq_markouts"],
        tables["ust_securities"],
    )
    parent = costs.parent_shortfall(
        cfg, tables["ust_orders"], tables["ust_rfq"], fills, tables["ust_securities"]
    )
    card, effects = scorecard.scorecard(
        cfg, tables["ust_rfq_quotes"], tables["ust_rfq"], tables["ust_orders"], fills
    )
    return {
        "ust_tca_fill": fills,
        "ust_tca_parent": parent,
        "ust_dealer_scorecard": card,
        "ust_dealer_effects": effects,
    }


def write(tables: dict[str, pd.DataFrame], derived: Path | None = None) -> list[Path]:
    """Write the TCA tables with floats rounded to 1e-6 (bp columns to 1e-4)."""
    paths = []
    for name, frame in tables.items():
        out = frame.copy()
        for col in out.select_dtypes(include="float").columns:
            out[col] = out[col].round(4 if col.endswith("_bp") or "_bp_" in col else 6)
        paths.append(storage.write_table(out, name, derived))
    return paths
