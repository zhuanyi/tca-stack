"""Treasury price from yield against published auction results (R3).

Source: TreasuryDirect auction results, read from
https://www.treasurydirect.gov/TA_WS/securities/search?format=json&type=Note&dateFieldName=issueDate&startDate=2025-03-01&endDate=2025-04-02
on 2026-11-04. Fields used: interestRate, maturityDate, issueDate, highYield, highPrice.
"""

from datetime import date

import pytest

from tca.core.bondmath import price_from_yield

_ONE_256TH = 1 / 256


@pytest.mark.parametrize(
    ("cusip", "coupon_pct", "maturity", "issue", "high_yield_pct", "high_price", "tolerance"),
    [
        # Month-end new issues settle on a coupon date, so there is no accrued-interest rounding.
        ("91282CMV0", 3.875, date(2027, 3, 31), date(2025, 3, 31), 3.984, 99.792438, 1e-5),
        ("91282CMU2", 4.000, date(2030, 3, 31), date(2025, 3, 31), 4.100, 99.552043, 1e-5),
        ("91282CMT5", 4.125, date(2032, 3, 31), date(2025, 3, 31), 4.233, 99.351591, 1e-5),
        # Issued after the dated date, so these carry accrued interest.
        ("91282CMS7", 3.875, date(2028, 3, 15), date(2025, 3, 17), 3.908, 99.907190, _ONE_256TH),
        ("91282CMM0", 4.625, date(2035, 2, 15), date(2025, 3, 17), 4.310, 102.513731, _ONE_256TH),
    ],
)
def test_price_matches_the_published_auction_price(
    cusip, coupon_pct, maturity, issue, high_yield_pct, high_price, tolerance
):
    px = float(price_from_yield(coupon_pct / 100, maturity, issue, high_yield_pct / 100))
    assert px == pytest.approx(high_price, abs=tolerance), cusip
