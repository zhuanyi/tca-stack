import numpy as np
import pytest


@pytest.fixture(scope="module")
def quotes(sim):
    q = sim["ust_rfq_quotes"].merge(
        sim["ust_rfq"][["rfq_id", "side", "status", "mid_award"]], on="rfq_id"
    )
    return q[q["responded"]].copy()


def test_every_quote_is_above_the_effective_mid_by_the_floor(cfg, quotes):
    floor = cfg.dealers.quote.floor_half_spread_frac
    premium = quotes["side"] * (quotes["quote_px"] - quotes["ref_mid_px"]) * 32.0
    assert (premium >= floor * quotes["hs_32nds"] - 1e-6).all()


def test_quotes_are_on_the_256th_grid(cfg, quotes):
    tick = cfg.dealers.quote.tick_denominator
    assert np.allclose(quotes["quote_px"] * tick, np.round(quotes["quote_px"] * tick), atol=1e-4)


def test_winner_has_the_best_quote_and_cover_the_second(quotes):
    done = quotes[quotes["status"] == "DONE"]
    for _, g in done.groupby("rfq_id"):
        cost = (g["side"] * g["quote_px"]).to_numpy()
        winner = g[g["is_winner"]]
        assert len(winner) == 1
        assert (winner["side"] * winner["quote_px"]).iloc[0] <= cost.min() + 1e-9
        cover = g[g["is_cover"]]
        if len(g) > 1 and len(cover):
            assert np.sort(cost)[1] == pytest.approx((cover["side"] * cover["quote_px"]).iloc[0])


def test_cost_vs_best_is_zero_for_every_award(sim):
    assert (sim["ust_tca_fill"]["cost_vs_best_32nds"].abs() < 1e-9).all()
