"""Acceptance tests: Saturday holidays, Juneteenth, Good Friday 2026 (M3, M4)."""

from datetime import date, timedelta

import pytest

from tca.core import bondmath as bm
from tca.core import calendar as cal


def _weekday_closures(year):
    d, out = date(year, 1, 1), []
    while d.year == year:
        if d.weekday() < 5 and not cal.is_business_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def test_2026_full_closures_match_sifma():
    assert _weekday_closures(2026) == [
        date(2026, 1, 1),
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 5, 25),
        date(2026, 6, 19),
        date(2026, 7, 3),
        date(2026, 9, 7),
        date(2026, 10, 12),
        date(2026, 11, 11),
        date(2026, 11, 26),
        date(2026, 12, 25),
    ]


def test_2025_full_closures_are_unchanged():
    assert _weekday_closures(2025) == [
        date(2025, 1, 1),
        date(2025, 1, 9),
        date(2025, 1, 20),
        date(2025, 2, 17),
        date(2025, 4, 18),
        date(2025, 5, 26),
        date(2025, 6, 19),
        date(2025, 7, 4),
        date(2025, 9, 1),
        date(2025, 10, 13),
        date(2025, 11, 11),
        date(2025, 11, 27),
        date(2025, 12, 25),
    ]


@pytest.mark.parametrize(
    "friday",
    [
        date(2020, 7, 3),  # Independence Day fell on Saturday 4 Jul
        date(2021, 12, 24),  # Christmas on Saturday
        date(2023, 11, 10),  # Veterans Day on Saturday
        date(2026, 7, 3),
        date(2027, 6, 18),  # Juneteenth on Saturday, observed from 2022
        date(2027, 12, 24),
    ],
)
def test_saturday_holiday_closes_the_preceding_friday(friday):
    assert friday.weekday() == 4
    assert not cal.is_business_day(friday)


def test_sunday_holiday_still_closes_monday():
    assert not cal.is_business_day(date(2021, 7, 5))
    assert not cal.is_business_day(date(2022, 6, 20))  # Juneteenth 2022 fell on Sunday


def test_saturday_new_year_is_an_early_close_not_a_full_close():
    # 1 Jan 2022 was a Saturday; 31 Dec 2021 was an early close
    assert cal.is_business_day(date(2021, 12, 31))
    assert cal.is_early_close(date(2021, 12, 31))
    assert cal.is_settlement_day(date(2021, 12, 31))


@pytest.mark.parametrize("d", [date(2019, 6, 19), date(2020, 6, 19)])
def test_no_juneteenth_closure_before_2022(d):
    assert cal.is_business_day(d)


def test_good_friday_2026_is_an_early_close_but_not_a_settlement_day():
    gf = date(2026, 4, 3)
    assert cal.is_business_day(gf)
    assert cal.is_early_close(gf)
    assert not cal.is_settlement_day(gf)
    assert gf in cal.business_days(date(2026, 3, 30), date(2026, 4, 7))


def test_good_friday_2025_remains_a_full_close():
    assert not cal.is_business_day(date(2025, 4, 18))
    assert not cal.is_early_close(date(2025, 4, 18))


def test_settle_date_around_good_friday_2026():
    assert bm.settle_date(date(2026, 4, 1)) == date(2026, 4, 2)
    assert bm.settle_date(date(2026, 4, 2)) == date(2026, 4, 6)  # 3 Apr is not a settlement day
    assert bm.settle_date(date(2026, 4, 3)) == date(2026, 4, 7)  # SIFMA: T+2 for trades of 3 Apr


def test_settle_date_skips_saturday_holiday_friday():
    assert bm.settle_date(date(2026, 7, 2)) == date(2026, 7, 6)


def test_plain_days_are_unchanged():
    assert cal.is_settlement_day(date(2025, 3, 10))
    assert not cal.is_early_close(date(2025, 3, 10))
    assert bm.settle_date(date(2025, 3, 7)) == date(2025, 3, 10)
    assert bm.settle_date(date(2025, 5, 23)) == date(2025, 5, 27)  # Memorial Day 26 May
    assert not cal.is_settlement_day(date(2025, 3, 8))  # a Saturday
