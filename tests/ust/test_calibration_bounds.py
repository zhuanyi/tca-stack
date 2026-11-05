import numpy as np
import pytest

from tca.core import storage
from tca.core.calendar import NY
from tca.ust import market, simulate


def test_median_half_spread_of_the_10y_otr_is_near_its_configured_level(cfg, inputs):
    try:
        bars = storage.read_table("ust_bars_1m")
    except FileNotFoundError:
        pytest.skip("ust_bars_1m is not published; run `tca ust simulate` to build it")
    uni = inputs.universe
    otr = uni[(uni["tenor"] == 10) & (uni["run_status"] == "OTR")]["cusip"].unique()
    realised = bars.loc[bars["cusip"].isin(otr), "half_spread_32nds"].median()
    q = cfg.composite_quote
    configured = q.hs_base_32nds[10] * q.run_mult["OTR"]
    assert abs(realised / configured - 1.0) < 0.40


def test_impact_slope_recovers_the_configured_price_impact(cfg, inputs, explicit_rfqs):
    """Mid move from send to award against ``sqrt(q / 100) * impact multiplier``, no leakage."""
    days = inputs.business_days(cfg.period.start, cfg.period.end)[:20]
    rfqs = explicit_rfqs(days, 10, seed=5, sizes=(5.0, 10.0, 25.0, 50.0, 100.0, 200.0))
    rfqs["n_dealers"] = 3
    t = simulate.run(cfg, 9, simulate.Overrides(rfqs=rfqs, leakage_scale=0.0, days=days), inputs)
    fills = t["ust_rfq"].merge(t["ust_rfq_markouts"][["rfq_id", "mid_px_0s"]], on="rfq_id")
    fills["day"] = fills["ts_send_utc"].dt.tz_convert(NY).dt.date
    ctx = market.Context.build(cfg, inputs)
    mult = fills["day"].map(lambda d: ctx.multipliers(d).impact)
    x = np.sqrt(fills["qty_mm"] / 100.0) * mult
    y = fills["side"] * (fills["mid_px_0s"] - fills["mid_send"]) * 32.0
    slope = float((x * y).sum() / (x * x).sum())
    assert abs(slope / cfg.impact.price_impact_32nds_per_100mm[10] - 1.0) < 0.50
