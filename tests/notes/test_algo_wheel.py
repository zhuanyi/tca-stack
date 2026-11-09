"""Tests of the algo-wheel note: the planted table, the adjustment, shrinkage, power, learning.

They run on a synthetic panel built from config/brokers.yaml, so they are fast, deterministic and
independent of the committed market tables (the CI profile has none).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import yaml

CONFIG = Path(__file__).resolve().parents[2] / "config" / "brokers.yaml"


def test_planted_table_equals_config(analysis, book):
    """Table 1 must be the configuration, not a copy of it in code."""
    raw = yaml.safe_load(CONFIG.read_text())["brokers"]
    table = analysis._table1(book).set_index("broker")
    assert sorted(table.index) == sorted(raw)
    for broker, cfg in raw.items():
        row = table.loc[broker]
        assert row["markup_bp"] == pytest.approx(cfg["markup_bp"])
        assert row["impact_multiplier"] == pytest.approx(cfg.get("impact_multiplier", 1.0))
        assert row["post_fill_impact_bp"] == pytest.approx(cfg.get("post_fill_impact_bp", 0.0))
        windows = cfg.get("markup_windows", [])
        if windows:
            window = windows[0]
            assert (
                row["markup_window"]
                == f"{window['start']}-{window['end']}: {window['markup_bp']:+.2f}"
            )
        else:
            assert row["markup_window"] == "-"


def test_regression_recovers_b_penalty_and_d_drift(analysis, panel, book):
    """B's markup penalty and D's leakage must both come out positive at these planted values."""
    adjustment = analysis.difficulty_adjustment(panel, sorted(book))
    effects = adjustment.effects.set_index("broker")
    assert effects.loc["A", "coef"] == 0.0
    assert effects.loc["B", "coef"] > 0
    assert effects.loc["D", "coef"] > effects.loc["B", "coef"]

    interaction = adjustment.interaction
    assert interaction.params["C(broker)[T.B]:log_size_rel"] < 0
    assert "C(broker)[B]:london_morning" in interaction.params.index
    assert "C(broker)[C]:london_morning" in interaction.params.index


def test_power_needs_more_orders_for_smaller_effects(analysis):
    """The power table must rise monotonically as the effect to detect shrinks."""
    table = analysis.power_table(2.8, 2.7, 900, 300.0)
    adjusted = table.sort_values("delta_bp")["n_adjusted"].to_numpy()
    assert np.all(np.diff(adjusted) < 0)
    assert table["months"].iloc[0] > table["months"].iloc[-1]
    assert table.loc[table["delta_bp"] == 0.25, "months"].iloc[0] > 0


def test_power_rejects_zero_effect(analysis):
    with pytest.raises(ValueError):
        analysis.stats.required_n(0.0, 1.0)


def test_shrinkage_moves_small_cells_more(analysis, panel, book):
    """The empirical-Bayes move must be larger where the cell is smaller."""
    adjustment = analysis.difficulty_adjustment(panel, sorted(book))
    cells = analysis.shrinkage_cells(panel, adjustment, sorted(book))
    assert len(cells) == 4 * 4 * 3
    summary = analysis.shrinkage_summary(cells)
    assert summary["small_n_move"] > summary["large_n_move"]
    assert summary["move_vs_n_spearman"] < 0
    largest = cells.sort_values("n").iloc[-1]
    assert abs(largest["shrunk"] - largest["raw"]) < 0.05 * max(abs(largest["raw"]), 1.0)


def test_thompson_beats_winner_takes_all(analysis, panel, book):
    """Over 20 seeds the adaptive policy must pay less regret than locking in after the warm-up."""
    result = analysis.learning_experiment(panel, book, n_orders=800, warmup=150, batch=50, seeds=20)
    summary = result.summary
    assert summary["ts_bp"] < summary["wta_bp"]
    assert summary["equal_split_bp"] > summary["ts_bp"]
    assert set(summary["true_mean_cost"]) == set(book)


def test_counterfactual_costs_add_back_the_planted_effect(analysis, panel, book):
    """The counterfactual matrix must return the observed cost on the row's own broker."""
    costs = analysis.counterfactual_costs(panel, book)
    brokers = sorted(book)
    own = panel["broker"].map({b: i for i, b in enumerate(brokers)}).to_numpy()
    np.testing.assert_allclose(costs[np.arange(len(panel)), own], panel["is_bp"].to_numpy())
    assert costs.shape == (len(panel), len(brokers))


def test_dealer_formulas_use_the_expected_controls(analysis):
    assert analysis.DEALER_HIT_FORMULA.startswith("win ~ log_qty")
    assert "C(dealer_id)" in analysis.DEALER_MARGIN_FORMULA
    assert "n_dealers" in analysis.DEALER_MARGIN_FORMULA


NOTE = Path(__file__).resolve().parents[2] / "notes" / "04_algo_wheel" / "NOTE.md"
RESULTS = Path(__file__).resolve().parents[2] / "notes" / "04_algo_wheel" / "results.json"
FIGURES = Path(__file__).resolve().parents[2] / "notes" / "04_algo_wheel" / "figures"
COMMENT = re.compile(r"<!-- results: (.*?) -->")


def _lookup(results, key):
    node = results
    for part in key.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def test_every_note_number_matches_results_json():
    """Every results: key=value comment in NOTE.md must match results.json."""
    results = json.loads(RESULTS.read_text())
    checked = 0
    for line in NOTE.read_text().splitlines():
        for block in COMMENT.findall(line):
            for pair in block.split("; "):
                key, claimed = pair.split("=", 1)
                actual = _lookup(results, key)
                if isinstance(actual, str):
                    assert claimed == actual, f"{key}: {claimed} != {actual}"
                else:
                    assert abs(float(claimed) - float(actual)) <= 1.5e-3, (
                        f"{key}: NOTE says {claimed}, results.json has {actual}"
                    )
                checked += 1
    assert checked > 60


def test_note_references_every_figure():
    """All five figures are committed and embedded in the note."""
    text = NOTE.read_text()
    for name in (
        "01_naive_ranking",
        "02_difficulty_adjustment",
        "03_shrinkage",
        "04_power",
        "05_learning",
    ):
        assert (FIGURES / f"{name}.png").exists()
        assert f"figures/{name}.png" in text


def test_note_states_the_planted_effects_are_not_market_evidence():
    """The note must say openly that the effects are planted."""
    text = NOTE.read_text().lower()
    assert "planted" in text
    assert "not the market" in text or "not evidence about" in text
