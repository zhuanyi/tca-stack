"""Liquidity-provider population."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tca.fx.config import FxConfig

LP_COLUMNS = [
    "lp_id",
    "name",
    "markup_pips",
    "reject_base",
    "reject_asym",
    "hold_median_ms",
    "hold_sigma",
    "max_clip_usd",
]
ECN_ID = 0
PRINCIPAL_ID = -1
PASSIVE_ID = -2


def generate_lps(cfg: FxConfig, seed: int) -> pd.DataFrame:
    """Draw the LPs: attributes uniform in the configured ranges, clips as configured.

    ``lp_id`` runs from 1; ``0`` denotes the ECN fallback, ``-1`` the principal that takes a
    risk-transfer order and ``-2`` a resting passive order, none of which is a row here.
    ``markup_pips`` is relative to the tape touch and positive is worse for the client;
    ``max_clip_usd`` is the largest child it quotes.
    """
    lp = cfg.liquidity_providers
    rng = np.random.default_rng([seed, 2])
    n = lp.count
    return pd.DataFrame(
        {
            "lp_id": np.arange(1, n + 1),
            "name": [f"LP{i}" for i in range(1, n + 1)],
            "markup_pips": rng.uniform(*lp.markup_pips, n),
            "reject_base": rng.uniform(*lp.reject_base, n),
            "reject_asym": rng.uniform(*lp.reject_asym, n),
            "hold_median_ms": rng.uniform(*lp.hold_ms.median_range, n),
            "hold_sigma": lp.hold_ms.sigma,
            "max_clip_usd": np.asarray(lp.max_clip_mm_by_lp_mm[:n], dtype=float) * 1e6,
        }
    )[LP_COLUMNS]
