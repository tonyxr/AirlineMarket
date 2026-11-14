from .common import stream, persist


def load_panel(scope, **kwargs):
    """
    Normalize filtered quote paths, dates and multi-leg segment fields (model §§1–3,
    calibration assumptions).
    """
    import pandas as pd

    out = kwargs.pop("out", None)
    f = stream(
        scope,
        "flightprices",
        (
            "legId",
            "searchDate",
            "flightDate",
            "startingAirport",
            "destinationAirport",
            "totalFare",
            "seatsRemaining",
        ),
        out=False,
        **kwargs,
    )
    f["days_before_departure"] = (
        pd.to_datetime(f.flightDate) - pd.to_datetime(f.searchDate)
    ).dt.days
    for k in f:
        if k.startswith("segments"):
            f[k] = f[k].fillna("").astype(str).str.split("||", regex=False)
    return persist(
        f.sort_values(["flightDate", "legId", "searchDate"]), scope, "flightprices", out
    )
