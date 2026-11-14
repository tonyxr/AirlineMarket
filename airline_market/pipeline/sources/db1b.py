from .common import stream, persist
from ...core.errors import DataValidationError


def load_markets(scope, *, expansion=None, **kwargs):
    """
    Aggregate expanded directional passengers and weighted cash-fare quantiles (model
    §§1–3, calibration assumptions).
    """
    import numpy as np

    out = kwargs.pop("out", None)
    f = stream(
        scope,
        "db1b",
        ("origin", "dest", "ticket_carrier", "passengers", "market_fare"),
        out=False,
        **kwargs,
    )
    factor = (
        expansion if expansion is not None else scope.params.get("sample_expansion")
    )
    if factor is None:
        raise DataValidationError("declare DB1B/OD40 sample_expansion explicitly")
    f["expanded_passengers"] = f.passengers * factor
    if "market_coupons" not in f:
        f["market_coupons"] = 1
    rows = []
    for keys, g in f.groupby(["origin", "dest", "ticket_carrier", "market_coupons"]):
        valid = g[g.market_fare.notna() & (g.passengers > 0)].sort_values("market_fare")
        weights = valid.passengers.to_numpy()
        values = valid.market_fare.to_numpy()
        if not len(values):
            continue
        qs = np.interp(
            [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99],
            (np.cumsum(weights) - 0.5 * weights) / weights.sum(),
            values,
        )
        row = dict(zip(["origin", "dest", "carrier", "coupons"], keys))
        row.update(
            passengers=float(g.expanded_passengers.sum()),
            sample_expansion=factor,
            measurement_break="DB1B versus DB1C/OD40 requires harmonized sampling and definitions",
        )
        row.update(
            dict(zip(["p05", "p10", "p25", "p50", "p75", "p90", "p99"], map(float, qs)))
        )
        rows.append(row)
    import pandas as pd

    result = pd.DataFrame(rows)
    result.attrs = f.attrs
    return persist(result, scope, "db1b", out)


def load_coupons(scope, **kwargs):
    """
    Read the itinerary coupon sequence with normalized required columns (model §§1–3,
    calibration assumptions).
    """
    return stream(scope, "coupon", ("itin_id", "seq_num", "origin", "dest"), **kwargs)
