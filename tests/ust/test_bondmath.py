from datetime import date

import numpy as np

from tca.core import bondmath


def test_par_bond_at_coupon_yield_prices_at_100():
    px_issue = bondmath.price_from_yield(0.045, date(2035, 2, 15), date(2025, 2, 15), 0.045)
    assert abs(px_issue - 100.0) < 1e-9


def test_auction_prices_at_the_high_yield_are_near_par(inputs):
    """New issues carry the high yield rounded down to 1/8, so they price just below par.

    Reopenings keep an older coupon and are excluded: a new issue's coupon is within 1/8 of a
    percentage point below its high yield.
    """
    sec = inputs.securities
    gap = sec["auction_high_yield"] - sec["coupon"]
    sec = sec[(gap >= 0) & (gap <= 0.00125 + 1e-9)]
    assert len(sec) > 10
    for s in sec.itertuples():
        px = bondmath.price_from_yield(
            s.coupon, s.maturity_date, s.issue_date, s.auction_high_yield
        )
        assert 98.5 < px <= 100.0 + 1e-9, (s.cusip, px)
        back = bondmath.yield_from_price(s.coupon, s.maturity_date, s.issue_date, px)
        assert (
            abs(bondmath.price_from_yield(s.coupon, s.maturity_date, s.issue_date, back) - px)
            < 1 / 256
        )


def test_dv01_is_positive_and_grows_with_maturity():
    settle = date(2025, 3, 4)
    dv = [bondmath.dv01(0.04, date(2025 + n, 2, 15), settle, 0.04) for n in (2, 5, 10, 30)]
    assert all(v > 0 for v in dv)
    assert np.all(np.diff(dv) > 0)


def test_32nds_formatting():
    assert bondmath.format_32nds(99.515625) == "99-16+"
