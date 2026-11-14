from .common import stream, persist


def load_schedule(scope, date=None, **kwargs):
    """
    Read the dated flight schedule and observed cancellation indicators (model §§1–3,
    calibration assumptions).
    """
    out = kwargs.pop("out", None)
    f = stream(
        scope,
        "ontime",
        (
            "flight_date",
            "carrier",
            "flight_number",
            "origin",
            "dest",
            "crs_dep_time",
            "crs_arr_time",
            "cancelled",
        ),
        out=False,
        **kwargs,
    )
    f["cancelled"] = f.cancelled.fillna(0).astype(int)
    f.attrs["cancellation_rates"] = {
        "/".join(k): float(v)
        for k, v in f.groupby(["carrier", "origin"]).cancelled.mean().to_dict().items()
    }
    return persist(f, scope, "ontime", out)
