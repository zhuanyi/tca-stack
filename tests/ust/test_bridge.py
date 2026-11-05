from datetime import date

import numpy as np

from tca.ust import bridge
from tca.ust.config import TENORS
from tca.ust.grid import Grid, parse_hhmm


def test_every_path_ends_on_the_anchor_close(cfg, inputs):
    grid = Grid.from_config(cfg)
    factors = bridge.estimate_factors(inputs.anchors, cfg)
    rng = np.random.default_rng(1)
    weights = bridge.variance_weights(date(2025, 3, 12), inputs.events, grid, cfg)
    y_prev = np.full(len(TENORS), 0.04)
    y_close = y_prev + rng.normal(0, 0.0007, len(TENORS))
    for _ in range(5):
        z = rng.standard_normal((cfg.yield_process.n_factors, grid.n_seconds))
        paths = bridge.bridge_paths(y_prev, y_close, (factors.daily_sd) ** 2, weights, factors, z)
        assert np.abs(paths[:, -1] - y_close).max() < 1e-9
        assert np.abs(paths[:, 0] - y_prev).max() < 1e-12


def _window_weight(d: date, inputs, grid, cfg) -> float:
    w = bridge.variance_weights(d, inputs.events, grid, cfg)[0]
    lo = int(grid.offset(parse_hhmm("08:25")))
    hi = int(grid.offset(parse_hhmm("08:35")))
    return float(w[lo:hi].sum() / w[1:].sum())


def test_cpi_day_allocates_more_variance_to_0830(cfg, inputs):
    grid = Grid.from_config(cfg)
    cpi = _window_weight(date(2025, 3, 12), inputs, grid, cfg)
    quiet = _window_weight(date(2025, 3, 13), inputs, grid, cfg)
    assert cpi > 2 * quiet
