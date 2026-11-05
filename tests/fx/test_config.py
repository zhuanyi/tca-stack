"""Configuration files: the planted broker effects and the calibration sources."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from tca.fx.config import CONFIG_DIR, check_broker_sets, hhmm_to_seconds, load_brokers


def test_brokers_match_the_specified_effects():
    book = load_brokers()
    assert book["A"].markup_bp == -0.15 and book["A"].impact_multiplier == 1.3
    assert book["B"].markup_bp == 0.20 and book["B"].impact_multiplier == 1.0
    assert book["C"].markup_bp == 0.25
    assert [(w.start, w.end, w.markup_bp) for w in book["C"].markup_windows] == [
        ("07:00", "12:00", -0.10)
    ]
    assert book["D"].markup_bp == -0.25 and book["D"].post_fill_impact_bp == 0.3
    assert all(book[b].post_fill_impact_bp == 0.0 for b in "ABC")


def test_broker_d_is_described_as_leakage_not_adverse_selection():
    text = Path(CONFIG_DIR / "brokers.yaml").read_text()
    assert "pre-hedge and signal" in text
    assert "not adverse selection" in text


NUMERIC_ENTRY = re.compile(r"^\s*(?:- )?[\w.]+:\s*[-\[{]?\d")


@pytest.mark.parametrize("name", ["fx_sim.yaml", "brokers.yaml"])
def test_every_number_in_the_configs_has_a_source(name):
    for line in (CONFIG_DIR / name).read_text().splitlines():
        if NUMERIC_ENTRY.match(line) and not line.lstrip().startswith("version:"):
            assert "source" in line, f"{name}: no source on: {line}"


def test_weights_and_mixtures_are_normalised(cfg):
    assert sum(cfg.pairs.weights.values()) == pytest.approx(1.0)
    assert sum(w.weight for w in cfg.flow.arrival_time_utc) == pytest.approx(1.0)
    assert sum(cfg.flow.urgency_probabilities.values()) == pytest.approx(1.0)
    assert sum(cfg.broker_assignment.assignment_mix.values()) == pytest.approx(1.0)


def test_volume_proxy_covers_every_pair(cfg):
    assert set(cfg.volume_proxy.spot_usd_bn_per_day) == set(cfg.pairs.order)
    assert cfg.daily_volume_usd("EURUSD") == pytest.approx(626.6e9)


def test_hhmm_conversion():
    assert hhmm_to_seconds("07:30") == 7 * 3600 + 1800


def test_the_raw_config_loads_as_yaml():
    assert yaml.safe_load((CONFIG_DIR / "fx_sim.yaml").read_text())["version"] == 1


def test_shipped_broker_sets_agree(cfg):
    check_broker_sets(cfg, load_brokers())


def test_a_broker_set_mismatch_raises(cfg):
    book = load_brokers()
    book.pop("D")
    with pytest.raises(ValueError, match="broker sets differ"):
        check_broker_sets(cfg, book)
    book["E"] = load_brokers()["A"]
    with pytest.raises(ValueError, match="broker sets differ"):
        check_broker_sets(cfg, book)
