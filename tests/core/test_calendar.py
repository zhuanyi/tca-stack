from datetime import UTC, date, time

import pytest

from tca.core import calendar as cal


def test_business_days_spring_2025_is_63():
    # 13 full weeks = 65 weekdays, minus Good Friday (Apr 18) and Memorial Day (May 26)
    days = cal.business_days(date(2025, 3, 3), date(2025, 5, 30))
    assert len(days) == 63
    assert date(2025, 4, 18) not in days
    assert date(2025, 5, 26) not in days


@pytest.mark.parametrize(
    ("d", "expected"),
    [
        (date(2025, 1, 1), False),
        (date(2025, 1, 9), False),
        (date(2025, 1, 20), False),
        (date(2025, 6, 19), False),
        (date(2025, 7, 4), False),
        (date(2025, 11, 27), False),
        (date(2025, 12, 25), False),
        (date(2025, 3, 8), False),
        (date(2025, 3, 10), True),
    ],
)
def test_is_business_day(d, expected):
    assert cal.is_business_day(d) is expected


def test_sunday_holiday_is_observed_on_monday():
    # 2021-07-04 is a Sunday
    assert not cal.is_business_day(date(2021, 7, 5))


def test_saturday_holiday_does_not_close_friday():
    # 2026-07-04 is a Saturday
    assert cal.is_business_day(date(2026, 7, 3))


def test_local_to_utc_respects_dst():
    winter = cal.local_to_utc(date(2025, 1, 15), time(9, 30), cal.NY)
    summer = cal.local_to_utc(date(2025, 7, 15), time(9, 30), cal.NY)
    assert winter.tzinfo == UTC
    assert (winter.hour, summer.hour) == (14, 13)


def test_wmr_window_before_and_after_london_dst():
    start, end = cal.wmr_window_utc(date(2025, 3, 28))
    assert (start.hour, start.minute, start.second) == (15, 57, 30)
    assert (end.hour, end.minute, end.second) == (16, 2, 30)
    start, end = cal.wmr_window_utc(date(2025, 4, 1))
    assert (start.hour, start.minute, start.second) == (14, 57, 30)
    assert (end.hour, end.minute, end.second) == (15, 2, 30)


def test_load_events_yaml(tmp_path):
    p = tmp_path / "events.yaml"
    p.write_text(
        "events:\n"
        "  - {ts_utc: '2025-03-07T13:30:00Z', type: NFP, source: BLS}\n"
        "  - {ts_utc: '2025-03-12T12:30:00Z', type: CPI, tenor: 10Y, source: BLS}\n"
    )
    df = cal.load_events(p)
    assert list(df.columns) == ["ts_utc", "type", "tenor", "source"]
    assert str(df["ts_utc"].dt.tz) == "UTC"
    assert df["tenor"].isna().tolist() == [True, False]


@pytest.mark.parametrize("year", [2024, 2025, 2026])
def test_good_friday_is_closed(year):
    good_friday = {2024: date(2024, 3, 29), 2025: date(2025, 4, 18), 2026: date(2026, 4, 3)}
    assert not cal.is_business_day(good_friday[year])
