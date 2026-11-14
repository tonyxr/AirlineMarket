"""V1 diagnostic of scheduled flights versus flown T-100 segments.

This consumes the already downloaded reporting-carrier diversion fields. It
does not replace the marketing schedule, relabel unmatched seats, or rescale
counts. Residual differences remain failures to reconcile, not passed cells.
"""

from collections import Counter
from pathlib import Path
import argparse
import json

from .scoped_io import batches, atomic_json, sha256
from .regional import LEG_MONTH


COLUMNS = [
    "Year",
    "Month",
    "FlightDate",
    "Reporting_Airline",
    "DOT_ID_Reporting_Airline",
    "Flight_Number_Reporting_Airline",
    "Origin",
    "Dest",
    "CRSDepTime",
    "Cancelled",
    "Diverted",
    "DivAirportLandings",
    "DivReachedDest",
    *[f"Div{k}Airport" for k in range(1, 6)],
]


def flown_path(row):
    """Return reported physical hops, or an explicit insufficient-data reason."""

    def number(name):
        """Interpret blank diversion flags as unknown, not an observed zero."""
        value = row.get(name)
        return None if value in (None, "", "nan") else int(float(value))

    cancelled, diverted = number("Cancelled"), number("Diverted")
    landings = number("DivAirportLandings")
    if not diverted and not landings:
        return ([] if cancelled else [row["Origin"], row["Dest"]]), None
    airports = [
        row[f"Div{k}Airport"]
        for k in range(1, 6)
        if row.get(f"Div{k}Airport") not in (None, "", "nan")
    ]
    if not airports:
        return [], "diversion_without_airport"
    path = [row["Origin"], *airports]
    if number("DivReachedDest") == 1 and path[-1] != row["Dest"]:
        path.append(row["Dest"])
    elif not cancelled and number("DivReachedDest") is None:
        return [], "unknown_diversion_destination_reached"
    return path, None


def run(config):
    """Compare both definitions and preserve unresolved route-month discrepancies."""
    import pandas as pd

    out = Path(config["out"])
    out.mkdir(parents=True, exist_ok=True)
    identity = {str(p): sha256(p) for p in config["reporting_files"]}
    identity[config["processing_results"]] = sha256(config["processing_results"])
    identity[config["scope"]] = sha256(config["scope"])
    identity[__file__] = sha256(__file__)
    prior = out / "COMPLETE.json"
    if prior.exists():
        old = json.loads(prior.read_text())
        if old["config"] != config or old["input_hashes"] != identity:
            raise ValueError(
                "reconciliation identity changed; use a new revision/output"
            )
        for name, digest in old["output_hashes"].items():
            if sha256(out / name) != digest:
                raise ValueError("changed reconciliation artifact " + name)
        return json.loads((out / "summary.json").read_text())
    scope = json.loads(Path(config["scope"]).read_text())
    pool, hub = set(scope["airport_pool"]), scope["hub_candidate"]
    scheduled, performed, flown, missing = Counter(), Counter(), Counter(), Counter()
    scanned = diverted_rows = 0

    def relevant(origin, dest):
        """Match exactly the registered physical leg geography."""
        return origin in pool and dest in pool and hub in (origin, dest)

    for source in config["reporting_files"]:
        for batch in batches(source, COLUMNS):
            frame = batch.to_pandas()
            scanned += len(frame)
            local = frame[
                frame.Origin.isin(pool)
                & frame.Dest.isin(pool)
                & ((frame.Origin == hub) | (frame.Dest == hub))
            ].copy()
            local["cancelled_numeric"] = pd.to_numeric(local.Cancelled)
            names = ["Year", "Month", "DOT_ID_Reporting_Airline", "Origin", "Dest"]
            for key, group in local.groupby(names):
                key = (int(key[0]), int(key[1]), *key[2:])
                scheduled[key] += len(group)
                performed[key] += int((group.cancelled_numeric == 0).sum())
            regular = local[(local.Cancelled == "0.00") | (local.Cancelled == "0")]
            regular = regular[
                (regular.Diverted.isin(["0.00", "0"]))
                & regular.DivAirportLandings.isin(["0.00", "0", ""])
            ]
            for key, count in regular.groupby(names).size().items():
                flown[(int(key[0]), int(key[1]), *key[2:])] += int(count)
            irregular = frame[
                (pd.to_numeric(frame.Diverted, errors="coerce").fillna(0) != 0)
                | (
                    pd.to_numeric(frame.DivAirportLandings, errors="coerce").fillna(0)
                    != 0
                )
            ]
            diverted_rows += len(irregular)
            for row in irregular.to_dict("records"):
                path, reason = flown_path(row)
                if reason:
                    missing[reason] += 1
                for origin, dest in zip(path, path[1:]):
                    if origin != dest and relevant(origin, dest):
                        flown[
                            (
                                int(row["Year"]),
                                int(row["Month"]),
                                row["DOT_ID_Reporting_Airline"],
                                origin,
                                dest,
                            )
                        ] += 1
        atomic_json(out / "progress.json", {"source": source, "rows_scanned": scanned})
    evidence = json.loads(Path(config["processing_results"]).read_text())
    capacity = Path(evidence["stages"]["capacity"]["output"])
    flights = pd.read_parquet(
        Path(evidence["stages"]["flights"]["output"]) / "flights.parquet"
    )
    t100 = pd.read_parquet(capacity / "operator_leg_month.parquet")
    t100.operating_carrier_id = t100.operating_carrier_id.astype(str)
    marketing = (
        flights.groupby(LEG_MONTH)
        .agg(
            marketing_scheduled=("schedule_weight", "sum"),
            marketing_performed=("performed_weight", "sum"),
        )
        .reset_index()
    )
    marketing.operating_carrier_id = marketing.operating_carrier_id.astype(str)
    keys = set(scheduled) | set(performed) | set(flown)
    diagnostic = pd.DataFrame(
        [
            {
                **dict(zip(LEG_MONTH, key)),
                "reporting_scheduled": scheduled[key],
                "reporting_performed": performed[key],
                "reconstructed_flown_segments": flown[key],
            }
            for key in sorted(keys)
        ]
    )
    result = t100.merge(marketing, on=LEG_MONTH, how="outer").merge(
        diagnostic, on=LEG_MONTH, how="outer"
    )
    for name in [
        "marketing_scheduled",
        "marketing_performed",
        "reporting_scheduled",
        "reporting_performed",
        "reconstructed_flown_segments",
        "departures_performed",
    ]:
        result[name] = result[name].fillna(0)
    result["marketing_vs_reporting_schedule"] = (
        result.marketing_scheduled - result.reporting_scheduled
    )
    result["scheduled_performed_minus_t100"] = (
        result.marketing_performed - result.departures_performed
    )
    result["flown_minus_t100"] = (
        result.reconstructed_flown_segments - result.departures_performed
    )
    result["fully_explained_by_flown_reconstruction"] = (
        result.scheduled_performed_minus_t100 != 0
    ) & (result.flown_minus_t100 == 0)
    result["residual_relative_error"] = (
        result.flown_minus_t100.abs() / result.departures_performed.clip(lower=1)
    )
    result.to_csv(out / "route_month_reconciliation.csv", index=False)
    summary = {
        "status": "diagnostic_complete",
        "raw_reporting_rows": scanned,
        "diversion_or_air_return_rows": diverted_rows,
        "unreconstructable_national_diversion_records": dict(missing),
        "marketing_scheduled": float(result.marketing_scheduled.sum()),
        "reporting_scheduled": float(result.reporting_scheduled.sum()),
        "marketing_vs_reporting_max_cell_difference": float(
            result.marketing_vs_reporting_schedule.abs().max()
        ),
        "t100_segments": float(result.departures_performed.sum()),
        "marketing_performed": float(result.marketing_performed.sum()),
        "reconstructed_flown_segments": float(
            result.reconstructed_flown_segments.sum()
        ),
        "mismatched_cells_before": int(
            (result.scheduled_performed_minus_t100 != 0).sum()
        ),
        "mismatched_cells_after": int((result.flown_minus_t100 != 0).sum()),
        "cells_fully_explained_by_flown_reconstruction": int(
            result.fully_explained_by_flown_reconstruction.sum()
        ),
        "remaining_cells_above_2_percent": int(
            (result.residual_relative_error > 0.02).sum()
        ),
        "sum_absolute_residual_departures": float(result.flown_minus_t100.abs().sum()),
        "V1_status": "not_yet_passed",
        "interpretation": "Diagnostic, not a replacement data series. Flown hops use reported diversion airports and destination-reached flags, attributed to the scheduled local departure month. Reporting coverage, ad-hoc operations and month rollover can remain different; no unresolved cell is declared explained.",
    }
    atomic_json(out / "summary.json", summary)
    atomic_json(
        prior,
        {
            "config": config,
            "input_hashes": identity,
            "implementation_sha256": sha256(__file__),
            "output_hashes": {
                name: sha256(out / name)
                for name in ("summary.json", "route_month_reconciliation.csv")
            },
        },
    )
    return summary


def main():
    """Run a declared local-only diagnostic without fetching additional datasets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(Path(args.config).read_text())), indent=2))


if __name__ == "__main__":
    main()
