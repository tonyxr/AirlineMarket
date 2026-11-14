"""Marketing-network attribution with explicit capacity conservation (V1)."""

import hashlib


PHYSICAL_KEY = [
    "flight_date",
    "operating_carrier_id",
    "operating_flight_number",
    "origin",
    "dest",
    "crs_dep_time",
]
LEG_MONTH = ["year", "month", "operating_carrier_id", "origin", "dest"]


def canonical_flights(frame):
    """Deduplicate physical/network records; preserve shared marketing ambiguity.

    A physical departure contributes one count in total. If it has multiple
    marketed networks, each receives an explicit equal ownership weight for
    the allocation estimator. No duplicate flag is interpreted as a deletion
    rule without checking the flight identity.
    """
    import pandas as pd

    f = frame.copy()
    n_before = len(f)
    keys = PHYSICAL_KEY + ["marketing_carrier"]
    critical = ["cancelled", "diverted", "marketing_carrier_id", "operating_carrier"]
    variation = f.groupby(keys, dropna=False)[critical].nunique(dropna=False)
    if (variation > 1).any().any():
        raise ValueError(
            "conflicting outcome/carrier records for one physical/network flight"
        )
    f = f.drop_duplicates(keys).copy()
    networks = f.groupby(PHYSICAL_KEY, dropna=False).marketing_carrier.transform(
        "nunique"
    )
    f["schedule_weight"] = 1.0 / networks
    f["performed_weight"] = f.schedule_weight * (f.cancelled == 0)
    f["partner_operated"] = f.operating_carrier != f.marketing_carrier
    f["marketing_ambiguous"] = networks > 1
    f["physical_flight_id"] = [
        hashlib.sha256("|".join(map(str, row)).encode()).hexdigest()[:24]
        for row in f[PHYSICAL_KEY].itertuples(index=False, name=None)
    ]
    physical = f.drop_duplicates(PHYSICAL_KEY)
    checks = {
        "raw_scoped_rows": n_before,
        "canonical_network_rows": len(f),
        "duplicate_network_rows_removed": n_before - len(f),
        "physical_scheduled_flights": len(physical),
        "physical_performed_flights": int((physical.cancelled == 0).sum()),
        "ambiguous_physical_flights": int(physical.marketing_ambiguous.sum()),
        "weighted_schedule_error": float(f.schedule_weight.sum() - len(physical)),
        "weighted_performed_error": float(
            f.performed_weight.sum() - (physical.cancelled == 0).sum()
        ),
    }
    return f.sort_values(PHYSICAL_KEY + ["marketing_carrier"]).reset_index(
        drop=True
    ), checks


def allocate_capacity(operator_leg_month, flights):
    """Estimate marketed seats from performed-flight shares without losing seats.

    This estimates allocation of observed operating-carrier totals. It does
    not observe contract seat entitlements or true unsold rival inventory.
    Unmatched source totals remain in an explicit UNMAPPED category.
    """
    import numpy as np

    counts = (
        flights.groupby(LEG_MONTH + ["marketing_carrier"], dropna=False)
        .agg(
            marketing_scheduled_flights=("schedule_weight", "sum"),
            marketing_performed_flights=("performed_weight", "sum"),
        )
        .reset_index()
    )
    counts["operator_performed_flights"] = counts.groupby(
        LEG_MONTH
    ).marketing_performed_flights.transform("sum")
    counts["allocation_share"] = (
        counts.marketing_performed_flights
        / counts.operator_performed_flights.replace(0, np.nan)
    )
    counts = counts[counts.allocation_share.notna()].copy()
    source = operator_leg_month.sort_values(LEG_MONTH).reset_index(drop=True).copy()
    source["source_leg_month_id"] = np.arange(len(source), dtype=int)
    result = source.merge(counts, on=LEG_MONTH, how="left", validate="one_to_many")
    missing = result.allocation_share.isna()
    result.loc[missing, "marketing_carrier"] = "UNMAPPED"
    result.loc[missing, "allocation_share"] = 1.0
    result["allocation_status"] = np.where(
        missing, "unmapped", "observed_performed_flight_share"
    )
    measures = ["seats", "passengers", "departures_scheduled", "departures_performed"]
    for name in measures:
        result["allocated_" + name] = result[name] * result.allocation_share
    reconciled = result.groupby("source_leg_month_id")[
        ["allocated_" + n for n in measures]
    ].sum()
    checks = {}
    for name in measures:
        errors = reconciled["allocated_" + name].to_numpy() - source[name].to_numpy()
        checks[name] = {
            "source_total": float(source[name].sum()),
            "allocated_total": float(result["allocated_" + name].sum()),
            "max_absolute_leg_month_error": float(np.max(np.abs(errors), initial=0)),
            "relative_total_error": float(
                abs(errors.sum()) / max(1, source[name].sum())
            ),
        }
    checks["unmapped_seats"] = float(result.loc[missing, "allocated_seats"].sum())
    checks["unmapped_seat_share"] = checks["unmapped_seats"] / max(
        1, float(source.seats.sum())
    )
    checks["method"] = (
        "Observed operating-route-month totals split by performed-flight marketing shares; unmapped retained. Fractional allocations are estimates, not integer departures observed by marketing carrier."
    )
    return result, checks


def departure_comparison(operator_leg_month, flights):
    """Report T-100/On-Time differences rather than rescaling flights to force identity."""
    counts = (
        flights.groupby(LEG_MONTH)
        .performed_weight.sum()
        .rename("ontime_performed_flights")
        .reset_index()
    )
    comparison = operator_leg_month.merge(counts, on=LEG_MONTH, how="outer")
    comparison["departure_difference"] = comparison.ontime_performed_flights.fillna(
        0
    ) - comparison.departures_performed.fillna(0)
    comparison["absolute_relative_difference"] = (
        comparison.departure_difference.abs()
        / comparison.departures_performed.fillna(0).clip(lower=1)
    )
    return comparison
