"""Construct Q2-only empirical targets from the reduced, attributed tables.

This prepares observations for P3; it does not fit simulator parameters or
evaluate Q3. Quote medians are exact over observed offers, using bounded
Parquet batches and cent-value frequency tables rather than a full panel.
"""

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import fcntl
import json
import time

from .scoped_io import atomic_json, sha256


def histogram_median(histogram):
    """Compute the exact ordinary median of integer-weighted cent observations."""
    total = sum(histogram.values())
    if total <= 0:
        raise ValueError("median requires observed positive-weight fares")
    if any(weight < 0 or weight != int(weight) for weight in histogram.values()):
        raise ValueError("fare frequencies must be nonnegative integer counts")
    positions = [(int(total) - 1) // 2, int(total) // 2]
    selected, running = [], 0
    for value, weight in sorted(histogram.items()):
        previous, running = running, running + int(weight)
        for position in positions:
            if previous <= position < running:
                selected.append(value)
    return sum(selected) / 200.0


def _quote_targets(path, out):
    """Summarize core-network offers without inventing unobserved search days."""
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    columns = [
        "quarter",
        "split",
        "flightDate",
        "market",
        "path_network",
        "days_bin",
        "isNonStop",
        "totalFare",
        "baseFare",
        "model_path_eligible",
        "left_truncated_by_collection_start",
        "seatsRemaining",
    ]
    histograms, pooled = defaultdict(Counter), defaultdict(Counter)
    rows = core = invalid = left_truncated = 0
    seats = Counter()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=100000, columns=columns):
        frame = batch.to_pandas()
        rows += len(frame)
        if not (
            (frame.quarter == 2)
            & (frame.split == "calibration")
            & frame.flightDate.between("2022-04-01", "2022-06-30")
        ).all():
            raise ValueError("Q2 target builder received a noncalibration departure")
        frame = frame[frame.model_path_eligible].copy()
        core += len(frame)
        left_truncated += int(frame.left_truncated_by_collection_start.sum())
        valid = np.isfinite(frame.totalFare) & (frame.totalFare > 0)
        invalid += int((~valid).sum())
        frame = frame[valid].copy()
        frame["fare_cents"] = np.rint(frame.totalFare * 100).astype(np.int64)
        frame["market_type"] = np.where(
            frame.isNonStop.isin([True, "True", "true"]), "nonstop_hub", "connecting"
        )
        names = ["market", "path_network", "market_type", "days_bin"]
        for key, count in (
            frame.groupby(names + ["fare_cents"], observed=True).size().items()
        ):
            histograms[key[:-1]][int(key[-1])] += int(count)
            pooled[(key[2], key[3])][int(key[-1])] += int(count)
        for value, count in frame.seatsRemaining.dropna().value_counts().items():
            seats[str(float(value))] += int(count)
        atomic_json(
            out / "progress.json",
            {
                "stage": "Q2_quotes",
                "rows_scanned": rows,
                "core_offers": core,
                "fare_frequency_cells": sum(map(len, histograms.values())),
            },
        )
    targets = []
    for key, values in sorted(histograms.items()):
        targets.append(
            {
                **dict(zip(names, key)),
                "observed_offers": sum(values.values()),
                "median_total_fare": histogram_median(values),
            }
        )
    pd.DataFrame(targets).to_csv(out / "q2_quote_targets_by_market.csv", index=False)
    pd.DataFrame(
        [
            {
                "market_type": key[0],
                "days_bin": key[1],
                "observed_offers": sum(values.values()),
                "median_total_fare": histogram_median(values),
            }
            for key, values in sorted(pooled.items())
        ]
    ).to_csv(out / "q2_quote_targets_by_type.csv", index=False)
    return {
        "scoped_offers_scanned": rows,
        "core_offers": core,
        "invalid_core_fares_excluded": invalid,
        "left_truncated_core_offers": left_truncated,
        "market_carrier_type_bin_targets": len(targets),
        "seats_display_counts": dict(seats),
        "sampling_unit": "Observed core-network offer/search, not a purchase or random sample of passengers",
        "left_truncation_rule": "Observed conditional fares retained; unobserved earlier search days contribute no observations, never zeros",
        "fare_basis": "All-in totalFare, including purchase taxes/fees; baseFare retained in input",
        "seats_interpretation": "Reported display only; neither true residual inventory nor a validated likelihood model",
    }


def _traffic_targets(path, out):
    """Use all-carrier denominators and separate represented from other paths."""
    import numpy as np
    import pandas as pd

    frame = pd.read_parquet(path)
    if not ((frame.quarter == 2) & (frame.split == "calibration")).all():
        raise ValueError("traffic targets must use Q2 only")
    total = frame.groupby("market").passengers_expanded.sum()
    counts = frame.groupby(
        ["market", "path_network", "model_path_eligible", "hub_connecting"],
        as_index=False,
    ).agg(
        passengers=("passengers_expanded", "sum"),
        sample_passengers=("passengers_sample", "sum"),
    )
    counts["all_carrier_market_passengers"] = counts.market.map(total)
    counts["share_of_all_carrier_market"] = (
        counts.passengers / counts.all_carrier_market_passengers
    )
    counts.to_csv(out / "q2_passenger_targets.csv", index=False)
    valid = frame[frame.model_path_eligible & frame.fare_eligible].copy()
    valid["fare_cents"] = np.rint(valid.market_fare * 100).astype(np.int64)
    fares = []
    for key, group in valid.groupby(["market", "path_network", "n_coupons"]):
        histogram = group.groupby("fare_cents").passengers_expanded.sum().to_dict()
        fares.append(
            {
                "market": key[0],
                "path_network": key[1],
                "n_coupons": int(key[2]),
                "paid_fare_median": histogram_median(histogram),
                "expanded_passengers": float(group.passengers_expanded.sum()),
            }
        )
    pd.DataFrame(fares).to_csv(out / "q2_paid_fare_initialization.csv", index=False)
    return {
        "rows": len(frame),
        "expanded_passengers": float(frame.passengers_expanded.sum()),
        "represented_passengers": float(
            frame.loc[frame.model_path_eligible, "passengers_expanded"].sum()
        ),
        "flow_scaling_error": float(
            frame.passengers_expanded.sum() - 10 * frame.passengers_sample.sum()
        ),
        "denominator": "All carrier/path passengers inside each selected O&D; outside airlines are not nonpurchase",
        "paid_fare_role": "Initialization/diagnostic; DB1B paid fares and posted quote medians are different populations",
    }


def run(config):
    """Prepare one source-attributed target set, reusing its complete receipt."""
    import pandas as pd

    root = Path(config["out"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        processing = json.loads(Path(config["processing_results"]).read_text())
        sources = {
            name: Path(stage["output"]) for name, stage in processing["stages"].items()
        }
        files = {
            "quotes": sources["quotes"] / "quotes_q2.parquet",
            "traffic": sources["markets"] / "db1b_scoped_q2.parquet",
            "capacity": sources["capacity"] / "marketing_leg_month.parquet",
            "flights": sources["flights"] / "flights.parquet",
            "weather": sources["weather"] / "weather_daily.parquet",
        }
        identity = {
            "config": config,
            "source_sha256": sha256(__file__),
            "processing_results_sha256": sha256(config["processing_results"]),
            "inputs": {
                key: {"path": str(path), "sha256": sha256(path)}
                for key, path in files.items()
            },
        }
        identity_path = root / "identity.json"
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError("target inputs/source changed; use a new revision/output")
        atomic_json(identity_path, identity)
        for receipt in sorted(root.glob("attempt-*/COMPLETE.json")):
            old = json.loads(receipt.read_text())
            if all(
                sha256(receipt.parent / name) == digest
                for name, digest in old["artifacts"].items()
            ):
                return json.loads((receipt.parent / "summary.json").read_text())
        out = root / f"attempt-{len(list(root.glob('attempt-*'))) + 1:04}"
        out.mkdir()
        started = time.perf_counter()
        atomic_json(root / "status.json", {"status": "running", "attempt": str(out)})
        try:
            quotes = _quote_targets(files["quotes"], out)
            traffic = _traffic_targets(files["traffic"], out)
            capacity = pd.read_parquet(
                files["capacity"], filters=[("month", ">=", 4), ("month", "<=", 6)]
            )
            capacity["load_factor"] = (
                capacity.allocated_passengers
                / capacity.allocated_seats.replace(0, float("nan"))
            )
            capacity["seats_per_performed_departure"] = (
                capacity.allocated_seats
                / capacity.allocated_departures_performed.replace(0, float("nan"))
            )
            capacity.to_csv(out / "q2_capacity_targets.csv", index=False)
            flights = pd.read_parquet(
                files["flights"], filters=[("split", "==", "calibration")]
            )
            daily = flights.groupby(
                ["flight_date", "marketing_carrier"], as_index=False
            ).agg(
                scheduled=("schedule_weight", "sum"),
                performed=("performed_weight", "sum"),
            )
            daily["cancellation_rate"] = 1 - daily.performed / daily.scheduled
            weather = pd.read_parquet(
                files["weather"],
                filters=[("DATE", ">=", "2022-04-01"), ("DATE", "<=", "2022-06-30")],
            )
            daily.merge(
                weather,
                left_on="flight_date",
                right_on="DATE",
                how="left",
                validate="many_to_one",
            ).to_csv(out / "q2_daily_supply_weather.csv", index=False)
            result = {
                "status": "targets_prepared_not_fitted",
                "quotes": quotes,
                "traffic": traffic,
                "capacity_rows": len(capacity),
                "daily_network_rows": len(daily),
                "seconds": time.perf_counter() - started,
                "fit_quarter": 2,
                "Q3_read_for_target_construction": False,
                "P3_status": "not_run",
                "V5_status": "not_run",
                "unidentified": "Nonpurchase conversion, segment taste/arrival decomposition, hidden inventory and contract seat allocations are not directly identified",
                "profit_basis": "Current generic simulator cash uses charged fares; empirical net airline profit requires explicit tax/pass-through treatment",
            }
            atomic_json(out / "summary.json", result)
            artifacts = {
                p.name: sha256(p)
                for p in out.iterdir()
                if p.is_file() and p.name != "progress.json"
            }
            atomic_json(
                out / "COMPLETE.json", {"identity": identity, "artifacts": artifacts}
            )
            atomic_json(
                root / "status.json", {"status": result["status"], "attempt": str(out)}
            )
            return result
        except Exception as error:
            atomic_json(
                out / "failure.json",
                {
                    "error": type(error).__name__,
                    "message": str(error),
                    "seconds": time.perf_counter() - started,
                },
            )
            atomic_json(
                root / "status.json",
                {"status": "failed", "attempt": str(out), "reason": str(error)},
            )
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(Path(args.config).read_text())), indent=2))
