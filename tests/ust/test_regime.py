import numpy as np
import pandas as pd

from tca.ust import regime
from tca.ust.calibrate import derive_elasticities
from tca.ust.market import Context


def test_elasticities_match_the_corrections(cfg, inputs):
    e = derive_elasticities(inputs.anchors, cfg)
    assert 0.05 < e.spread < 0.25
    assert -1.1 < e.depth < -0.6
    assert 0.5 < e.impact < 0.95


def test_multipliers_are_clipped(cfg, inputs):
    ctx = Context.build(cfg, inputs)
    clip = cfg.composite_quote.regime_clip
    for d in inputs.business_days(cfg.period.start, cfg.period.end):
        m = ctx.multipliers(d)
        assert clip.spread[0] <= m.spread <= clip.spread[1]
        assert clip.depth[0] <= m.depth <= clip.depth[1]
        assert clip.impact[0] <= m.impact <= clip.impact[1]


def test_distribution_ignores_missing_values():
    x = np.array([0.1, 0.4, np.nan, 0.2, 0.9, np.nan])
    rng = np.random.default_rng(0)
    d = regime.distribution(x, rng)
    clean = regime.distribution(x[~np.isnan(x)], np.random.default_rng(0))
    assert d == clean
    assert d["n"] == 4
    assert pd.notna(list(d.values())).all()
