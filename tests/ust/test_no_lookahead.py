import numpy as np

from tca.ust import engine, market
from tca.ust.ledger import Ledger

_BENCHMARKS = ["mid_send", "composite_bid", "composite_ask", "hs_send_32nds", "vwap_proxy_px"]


def test_send_mid_is_rebuilt_from_events_before_the_send(cfg, inputs, three_days, sim, seed):
    """``mid_send`` equals the fundamental plus the impact of trades and hedges up to the send."""
    ctx = market.Context.build(cfg, inputs)
    rfq = sim["ust_rfq"]
    hedges = sim["ust_hedges"]
    done = rfq[rfq["status"] == "DONE"]
    checked = 0
    for d in three_days:
        mkt = market.build_day(ctx, d, seed)
        day = rfq[rfq["ts_send_utc"].dt.tz_convert("America/New_York").dt.date == d]
        for cusip, group in day.groupby("cusip"):
            row = mkt.row(cusip)
            fills = done[done["rfq_id"].isin(group["rfq_id"])]
            ledger = Ledger(cfg.impact.perm_frac, cfg.impact.decay_half_life_min * 60.0)
            for f in fills.itertuples():
                t = ctx.grid.to_offset(d, f.ts_award_utc)
                shift = f.side * engine.fill_impact_32nds(mkt, row, f.qty_mm) / 32.0
                ledger.add(t, shift)
            h = hedges[hedges["rfq_id"].isin(fills["rfq_id"])]
            ledger.add(
                np.array([ctx.grid.to_offset(d, t) for t in h["ts_utc"]]),
                h["impact_32nds"].to_numpy() / 32.0,
            )
            for r in group.itertuples():
                t = ctx.grid.to_offset(d, r.ts_send_utc)
                expected = mkt.base_px[row, ctx.grid.index(t)] + ledger.value(t)[0]
                assert abs(expected - r.mid_send) < 1e-5
                checked += 1
    assert checked > 100


def test_pre_trade_benchmarks_are_stamped_before_the_award(sim):
    f = sim["ust_tca_fill"]
    assert (f["ts_award_utc"] >= f["ts_send_utc"]).all()
    assert np.isfinite(f[_BENCHMARKS].to_numpy()).all()
