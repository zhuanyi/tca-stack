import numpy as np

from tca.core import conventions


def test_buy_above_arrival_is_costly_in_every_measure():
    side, px, mid = 1, 99.52, 99.50
    cost_px = conventions.cost_bp(side, px, mid)
    assert cost_px > 0
    assert conventions.cost_32nds(side, px, mid) > 0
    assert conventions.cost_yld_bp(side, 0.0429, 0.0430) > 0
    assert conventions.cost_usd(cost_px, 10e6) > 0


def test_fill_tables_agree_on_the_sign_of_cost(sim):
    f = sim["ust_tca_fill"]
    sign = np.sign(f["cost_32nds"])
    assert (np.sign(f["cost_px_bp"]) == sign).all()
    assert (np.sign(f["cost_usd"]) == sign).all()
    clear = f["cost_32nds"].abs() > 0.02
    assert (np.sign(f.loc[clear, "cost_yld_bp"]) == sign[clear]).all()
    assert (f["dv01_usd"] > 0).all()


def test_markouts_use_the_overview_sign(sim):
    """Positive markout means the later mid is better for the trader than the fill."""
    f = sim["ust_tca_fill"]
    marks = sim["ust_rfq_markouts"].set_index("rfq_id").loc[f["rfq_id"]]
    expected = f["side"].to_numpy() * (marks["mid_px_5m"].to_numpy() - f["exec_px"].to_numpy()) * 32
    assert np.allclose(f["markout_32nds_5m"], expected, atol=1e-6)
