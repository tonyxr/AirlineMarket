from .common import stream, persist


def load_segments(scope, **kwargs):
    """
    Aggregate capacity and traffic by carrier, segment and aircraft type (model §§1–3,
    calibration assumptions).
    """
    import numpy as np

    out = kwargs.pop("out", None)
    f = stream(
        scope,
        "t100",
        ("carrier", "origin", "dest", "seats", "departures_performed", "passengers"),
        out=False,
        **kwargs,
    )
    keys = [
        k
        for k in ["year", "month", "carrier", "origin", "dest", "aircraft_type"]
        if k in f
    ]
    g = (
        f.groupby(keys, dropna=False)[["seats", "departures_performed", "passengers"]]
        .sum()
        .reset_index()
    )
    g["seats_per_departure"] = g.seats / g.departures_performed.replace(0, np.nan)
    g["load_factor"] = g.passengers / g.seats.replace(0, np.nan)
    g.attrs = f.attrs
    return persist(g, scope, "t100", out)
