"""Derive the regime elasticities from the April 2025 readings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tca.ust.config import SimConfig

_READ_TENORS = (2, 5, 10)


@dataclass(frozen=True)
class Elasticities:
    """Exponents ``e`` in ``multiplier = r ** e`` for r = rv_5d / rv_median, one per channel."""

    spread: float
    depth: float
    impact: float


def market_ratio(anchors: pd.DataFrame, cfg: SimConfig) -> pd.Series:
    """Daily volatility ratio ``r`` of the reference tenor, indexed by date."""
    q = cfg.composite_quote
    ref = anchors[anchors["tenor"] == q.regime_reference_tenor]
    return ref.set_index("date")["regime_mult"]


def derive_elasticities(anchors: pd.DataFrame, cfg: SimConfig) -> Elasticities:
    """Exponents that reproduce the April readings at the ratio ``r`` of the matched day.

    For each channel ``e = mean(ln(peak / normal)) / ln(r)`` over the 2y, 5y and 10y readings.
    Normal levels are the configured March 2025 values: twice ``hs_base`` for the full spread,
    ``depth_mm`` and the price impact per $100mm; peaks are ``april_2025_reference``.
    """
    q = cfg.composite_quote
    ref = q.april_2025_reference
    ln_r = float(np.log(market_ratio(anchors, cfg)[q.regime_matched_on]))

    def fit(peak: dict[int, float], normal: dict[int, float]) -> float:
        return float(np.mean([np.log(peak[t] / normal[t]) for t in _READ_TENORS]) / ln_r)

    normal_spread = {t: 2.0 * q.hs_base_32nds[t] for t in _READ_TENORS}
    return Elasticities(
        spread=fit(ref.full_spread_32nds_peak, normal_spread),
        depth=fit(ref.depth_mm_trough, q.depth_mm),
        impact=fit(ref.price_impact_32nds_per_100mm_peak, cfg.impact.price_impact_32nds_per_100mm),
    )
