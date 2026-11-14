from ...vendor.ride_response.calendar import holiday_dates


def calendar_dates(year):
    """
    Combine the vendored holiday list with the installed federal calendar (model §§1–3,
    calibration assumptions).
    """
    from pandas.tseries.holiday import USFederalHolidayCalendar

    return sorted(
        set(map(str, holiday_dates(year)))
        | set(
            str(x.date())
            for x in USFederalHolidayCalendar().holidays(
                f"{year}-01-01", f"{year}-12-31"
            )
        )
    )
