"""The twelve dealers: seeded attributes, daily axes and inventory, and the quote rule.

A dealer's quote for a client buy is ``mid + premium`` and for a client sell ``mid - premium``,
with the premium in 32nds built from the dealer's spread, any axe, a size charge, a
competition term, an inventory skew and noise, and floored at a fraction of the composite
half-spread. Prices are rounded away from the mid to the 1/256 tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from tca.ust.config import SimConfig

STREAM_DEALERS = 0
STREAM_DEALER_DAY = 2


@dataclass(frozen=True)
class Dealers:
    """Dealer attributes as arrays, in ``table`` row order."""

    table: pd.DataFrame

    def col(self, name: str) -> np.ndarray:
        return self.table[name].to_numpy()

    def __len__(self) -> int:
        return len(self.table)


def draw_dealers(cfg: SimConfig, seed: int) -> Dealers:
    """Draw the dealer attributes once from the configured ranges (written as ``ust_dealers``)."""
    dc = cfg.dealers
    rng = np.random.default_rng([seed, STREAM_DEALERS])
    tiers = np.repeat(sorted(dc.tier_counts), [dc.tier_counts[t] for t in sorted(dc.tier_counts)])
    n = len(tiers)
    if n != dc.n:
        raise ValueError(f"tier_counts sum to {n}, but dealers.n is {dc.n}")
    lo = np.array([dc.spread_mult_range[t][0] for t in tiers])
    hi = np.array([dc.spread_mult_range[t][1] for t in tiers])

    def uniform(bounds: tuple[float, float]) -> np.ndarray:
        return rng.uniform(bounds[0], bounds[1], n)

    table = pd.DataFrame(
        {
            "dealer_id": [f"D{i + 1:02d}" for i in range(n)],
            "tier": tiers,
            "spread_mult": rng.uniform(lo, hi),
            "axe_prob": uniform(dc.axe_prob_range),
            "response_prob_base": uniform(dc.response_prob_base_range),
            "response_latency_median_s": uniform(dc.response_latency_median_s_range),
            "no_quote_size_mm": uniform(dc.no_quote_size_mm_range),
            "internalisation": uniform(dc.internalisation_range),
            "size_charge_k": uniform(dc.size_charge_k_range),
            "competition_widening": uniform(dc.competition_widening_range),
        }
    )
    return Dealers(table)


@dataclass(frozen=True)
class DealerDay:
    """Per-day dealer state: the axe direction and the inventory path of each security.

    ``axe[d, s]`` is +1 when dealer ``d`` wants to sell security ``s`` (tightens its offer),
    -1 when it wants to buy, 0 when not axed. ``inventory_mm[d, s, m]`` is the position in $mm
    at the start of minute ``m`` (positive is long).
    """

    axe: np.ndarray
    inventory_mm: np.ndarray


def draw_day(cfg: SimConfig, dealers: Dealers, n_sec: int, n_minutes: int, d: date, seed: int):
    """Draw the day's axes and inventory paths for all dealers and ``n_sec`` securities."""
    dc = cfg.dealers
    rng = np.random.default_rng([seed, STREAM_DEALER_DAY, d.toordinal()])
    n = len(dealers)
    axed = rng.random((n, n_sec)) < dealers.col("axe_prob")[:, None]
    direction = np.where(rng.random((n, n_sec)) < 0.5, 1, -1)
    phi = 0.5 ** (1.0 / dc.inventory.ou_half_life_min)
    sd = dc.inventory.ou_stationary_sd_scale * dc.inventory.inventory_scale_mm
    start = rng.standard_normal((n, n_sec, 1)) * sd
    shocks = rng.standard_normal((n, n_sec, n_minutes - 1)) * sd * np.sqrt(1.0 - phi**2)
    rest = lfilter([1.0], [1.0, -phi], shocks, axis=2, zi=phi * start)[0]
    return DealerDay(
        axe=(axed * direction).astype(np.int8), inventory_mm=np.concatenate([start, rest], axis=2)
    )


def axe_adjustment(cfg: SimConfig, axe: np.ndarray, side) -> np.ndarray:
    """Spread multiplier: ``1 - effect`` when the axe favours the client, ``1 + effect`` against."""
    effect = cfg.dealers.axe_effect
    return 1.0 - effect * axe * np.asarray(side)


def premium_32nds(
    hs, spread_mult, axe_adj, size_charge, widening, shift, noise, floor_frac: float
) -> np.ndarray:
    """Quote premium over the effective mid in 32nds, never below ``floor_frac * hs``.

    ``hs`` is the composite half-spread; ``shift`` is the signed inventory skew already
    expressed as a cost to the client (positive makes the quote worse for them).
    """
    raw = hs * spread_mult * axe_adj + size_charge + widening + shift + noise
    return np.maximum(raw, floor_frac * hs)


def quote_price(mid, side, premium_32nds_, tick_denominator: int) -> np.ndarray:
    """Quoted clean price: asks rounded up, bids rounded down to ``1 / tick_denominator``."""
    px = mid + np.asarray(side) * premium_32nds_ / 32.0
    scaled = px * tick_denominator
    return np.where(np.asarray(side) > 0, np.ceil(scaled), np.floor(scaled)) / tick_denominator
