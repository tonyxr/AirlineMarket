import datetime as _dt
from typing import Dict


def holiday_dates(year: int) -> Dict[str, _dt.date]:
    def nth_weekday(month: int, weekday: int, n: int) -> _dt.date:
        day = _dt.date(year, month, 1)
        offset = (weekday - day.weekday()) % 7
        return day + _dt.timedelta(days=offset + 7 * (n - 1))

    def last_weekday(month: int, weekday: int) -> _dt.date:
        day = _dt.date(year, month + 1, 1) - _dt.timedelta(days=1)
        return day - _dt.timedelta(days=(day.weekday() - weekday) % 7)

    return {
        "thanksgiving": nth_weekday(11, 3, 4),
        "christmas_new_year": _dt.date(year, 12, 25),
        "independence_day": _dt.date(year, 7, 4),
        "memorial_day": last_weekday(5, 0),
        "labor_day": nth_weekday(9, 0, 1),
    }
