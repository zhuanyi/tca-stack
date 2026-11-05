"""Same seed, byte-identical tables; another seed, different ones."""

from __future__ import annotations

import hashlib
from pathlib import Path

from tca.core import storage
from tca.fx import pipeline
from tca.fx.simulate import Overrides, SimResult, run


def _hashes(result: SimResult, folder: Path) -> dict[str, str]:
    out = {}
    for name, frame in result.tables().items():
        path = storage.write_table(pipeline.round_table(frame), name, folder)
        out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _run(cfg, source, market, seed: int) -> SimResult:
    return run(cfg, seed, source, market, Overrides(max_days=1))


def test_same_seed_gives_byte_identical_tables(cfg, source, market, tmp_path):
    first = _hashes(_run(cfg, source, market, 11), tmp_path / "a")
    second = _hashes(_run(cfg, source, market, 11), tmp_path / "b")
    assert first == second


def test_a_different_seed_gives_different_tables(cfg, source, market, tmp_path):
    first = _hashes(_run(cfg, source, market, 11), tmp_path / "a")
    other = _hashes(_run(cfg, source, market, 12), tmp_path / "b")
    assert first["fx_orders"] != other["fx_orders"]
    assert first["fx_fills"] != other["fx_fills"]


def test_the_order_flow_of_a_day_does_not_depend_on_the_days_before_it(cfg, source, market):
    both = run(cfg, 5, source, market, Overrides(max_days=2)).orders
    second_day = source.days("EURUSD")[1]
    only = run(cfg, 5, source, market, Overrides(days=[second_day])).orders
    cols = ["pair", "side", "notional_usd", "ts_arrival_utc", "style", "broker"]
    on_day = both[both["ts_arrival_utc"].dt.date == second_day][cols].reset_index(drop=True)
    assert on_day.equals(only[cols].reset_index(drop=True))


def test_lp_population_is_fixed_by_the_seed(cfg, source, market):
    a, b = _run(cfg, source, market, 3), _run(cfg, source, market, 3)
    assert a.lps.equals(b.lps)
    assert not a.lps.equals(_run(cfg, source, market, 4).lps)
