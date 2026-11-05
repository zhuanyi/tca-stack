"""Last look: rejection depends on the tape moving against the LP during the hold."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tca.fx.replay import replay_parent


def _replay(cfg, env, tape, order):
    return replay_parent(order, tape, env, np.random.default_rng(1))


def _outcomes(out) -> list[str]:
    return [a["outcome"] for a in out.attempts]


def test_full_asymmetry_rejects_when_the_tape_moves_against_the_lp(
    cfg, one_shot_env, make_tape, make_order, first_day, noon_us
):
    tape = make_tape(first_day, rising=True, start_us=noon_us)
    out = _replay(cfg, one_shot_env(0.0, 1.0), tape, make_order(first_day, noon_us))
    assert _outcomes(out)[0] == "REJECTED"
    assert out.attempts[0]["moved_against"] is True


def test_full_asymmetry_accepts_when_the_tape_moves_in_the_lp_favour(
    cfg, one_shot_env, make_tape, make_order, first_day, noon_us
):
    tape = make_tape(first_day, rising=False, start_us=noon_us)
    out = _replay(cfg, one_shot_env(0.0, 1.0), tape, make_order(first_day, noon_us))
    assert _outcomes(out) == ["FILLED"]


def test_no_rejection_without_base_or_asymmetric_probability(
    cfg, one_shot_env, make_tape, make_order, first_day, noon_us
):
    tape = make_tape(first_day, rising=True, start_us=noon_us)
    out = _replay(cfg, one_shot_env(0.0, 0.0), tape, make_order(first_day, noon_us))
    assert _outcomes(out) == ["FILLED"]
    assert out.attempts[0]["moved_against"] is True


def test_sell_is_rejected_when_the_tape_falls(
    cfg, one_shot_env, make_tape, make_order, first_day, noon_us
):
    tape = make_tape(first_day, rising=False, start_us=noon_us)
    order = make_order(first_day, noon_us, side=-1)
    out = _replay(cfg, one_shot_env(0.0, 1.0), tape, order)
    assert _outcomes(out)[0] == "REJECTED"


def test_after_the_allowed_rejections_the_child_crosses_on_the_ecn(
    cfg, one_shot_env, make_tape, make_order, first_day, noon_us
):
    tape = make_tape(first_day, rising=True, start_us=noon_us)
    out = _replay(cfg, one_shot_env(1.0, 0.0), tape, make_order(first_day, noon_us))
    assert _outcomes(out) == ["REJECTED", "ECN"]
    fill = out.fills[0]
    assert fill["lp_id"] == 0
    assert fill["lp_markup_px"] == cfg.last_look.ecn_markup_pips * 1e-4


def test_attempts_are_recorded_for_every_rejection_and_fill(
    cfg, one_shot_env, make_tape, make_order, first_day, noon_us
):
    tape = make_tape(first_day, rising=True, start_us=noon_us)
    out = _replay(cfg, one_shot_env(1.0, 0.0), tape, make_order(first_day, noon_us))
    frame = pd.DataFrame(out.attempts)
    assert frame["child_id"].nunique() == 1
    assert out.fills[0]["n_rejects"] == (frame["outcome"] == "REJECTED").sum()
