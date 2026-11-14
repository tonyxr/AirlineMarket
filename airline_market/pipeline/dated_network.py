"""Audited dated resources/products from the reduced marketing-carrier tables.

This builds the empirical network input, not a calibrated MarketSpec. Seats
per flight use T-100 seats/performed departures without rescaling to make
the On-Time counts agree. Cancelled, ambiguous and unmatched records remain
explicit. Airport time zones and elapsed minutes determine connection times.
"""

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import argparse
import fcntl
import hashlib
import json
import math
import time

from .regional import LEG_MONTH
from .scoped_io import ParquetSink, atomic_json, sha256


def departure_time(day, hhmm, zone):
    """Interpret the source local clock, including 2400 as next-day midnight."""
    clock = int(float(hhmm))
    if clock == 2400:
        return datetime.fromisoformat(day).replace(tzinfo=ZoneInfo(zone)) + timedelta(
            days=1
        )
    hour, minute = divmod(clock, 100)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("invalid scheduled local clock")
    return datetime.fromisoformat(day).replace(
        hour=hour, minute=minute, tzinfo=ZoneInfo(zone)
    )


def timed_flight(row, zones):
    """Use elapsed flight time on UTC, then verify the destination's local clock."""
    elapsed = float(row.scheduled_elapsed_minutes)
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("invalid_scheduled_elapsed_minutes")
    departure = departure_time(row.flight_date, row.crs_dep_time, zones[row.origin])
    arrival = (
        departure.astimezone(timezone.utc) + timedelta(minutes=elapsed)
    ).astimezone(ZoneInfo(zones[row.dest]))
    raw = int(float(row.crs_arr_time))
    if raw == 2400:
        raw = 0
    hour, minute = divmod(raw, 100)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("invalid_scheduled_arrival_clock")
    difference = abs(arrival.hour * 60 + arrival.minute - hour * 60 - minute)
    if min(difference, 1440 - difference) > 1:
        raise ValueError("elapsed_time_arrival_clock_disagreement")
    return departure, arrival


def capacity_prior(mean, relative_spread):
    """Declared two-point prior preserving the observed mean; spread is assumed."""
    if not math.isfinite(mean) or mean < 0 or not 0 <= relative_spread < 1:
        raise ValueError("invalid mean capacity or prior spread")
    low = max(0, math.floor(mean * (1 - relative_spread)))
    high = math.ceil(mean * (1 + relative_spread))
    probability = (mean - low) / (high - low) if high != low else 0.0
    return low, high, probability


def build_legs(flights, capacities, config, scope):
    """Retain network/operator identities; do not invent defaults for missing seats."""
    import pandas as pd

    zones = config["airport_timezones"]
    missing = set(scope["airport_pool"]) - set(zones)
    if missing:
        raise ValueError("missing declared airport time zones: " + str(sorted(missing)))
    lookup = {
        tuple(getattr(row, key) for key in LEG_MONTH): row
        for row in capacities.itertuples(index=False)
    }
    legs, excluded = [], []
    for row in flights.itertuples(index=False):
        if row.marketing_carrier not in scope["carriers"]:
            continue
        reason = None
        if row.marketing_ambiguous:
            reason = "shared_marketing_requires_joint_resource_model"
        elif row.cancelled:
            reason = "cancelled_observed_service"
        source = lookup.get(tuple(getattr(row, key) for key in LEG_MONTH))
        if reason is None and (source is None or source.departures_performed <= 0):
            reason = "missing_positive_t100_performed_departures"
        if reason is None:
            try:
                departure, arrival = timed_flight(row, zones)
            except ValueError as error:
                reason = str(error)
        if reason is None:
            cohort = departure.date().isoformat()
            if not config["cohort_window"][0] <= cohort <= config["cohort_window"][1]:
                reason = "outside_actual_departure_cohort_window"
        if reason:
            excluded.append(
                {
                    "physical_flight_id": row.physical_flight_id,
                    "flight_date": row.flight_date,
                    "marketing_carrier": row.marketing_carrier,
                    "operating_carrier": row.operating_carrier,
                    "origin": row.origin,
                    "dest": row.dest,
                    "reason": reason,
                }
            )
            continue
        mean = float(source.seats / source.departures_performed)
        low, high, prob = capacity_prior(mean, config["capacity_prior_relative_spread"])
        legs.append(
            {
                "id": row.marketing_carrier + "_" + row.physical_flight_id,
                "physical_flight_id": row.physical_flight_id,
                "source_flight_date": row.flight_date,
                "cohort_date": cohort,
                "year": row.year,
                "month": row.month,
                "marketing_carrier": row.marketing_carrier,
                "operating_carrier": row.operating_carrier,
                "operating_carrier_id": row.operating_carrier_id,
                "operating_flight_number": row.operating_flight_number,
                "marketing_flight_number": row.marketing_flight_number,
                "partner_operated": row.partner_operated,
                "origin": row.origin,
                "dest": row.dest,
                "departure_local": departure.isoformat(),
                "arrival_local": arrival.isoformat(),
                "departure_utc_seconds": departure.timestamp(),
                "arrival_utc_seconds": arrival.timestamp(),
                "scheduled_elapsed_minutes": float(row.scheduled_elapsed_minutes),
                "distance_mi": float(row.distance),
                "mean_initial_capacity": mean,
                "capacity_low": low,
                "capacity_high": high,
                "probability_high": prob,
                "capacity_public": False,
                "split": "calibration" if cohort <= "2022-06-30" else "validation",
            }
        )
    result = pd.DataFrame(legs)
    if result.empty:
        raise ValueError("no valid observed-service core legs")
    if result.physical_flight_id.duplicated().any():
        raise ValueError(
            "a physical flight would be duplicated as independent core resources"
        )
    return result.sort_values(
        ["cohort_date", "marketing_carrier", "departure_utc_seconds", "id"]
    ), pd.DataFrame(excluded)


def product_rows(legs, selected_markets, hub, connection_window):
    """Generate all same-cohort single/two-leg core products, using UTC waits."""
    counts = Counter()

    def product(path):
        """Keep stable ordered resource IDs and elapsed journey duration."""
        first, last = path[0], path[-1]
        market = first.origin + "-" + last.dest
        if market not in selected_markets:
            return None
        identifier = hashlib.sha256(
            "|".join(row.id for row in path).encode()
        ).hexdigest()[:24]
        counts[first.cohort_date] += 1
        return {
            "id": first.marketing_carrier + "_" + identifier,
            "carrier": first.marketing_carrier,
            "cohort_date": first.cohort_date,
            "split": first.split,
            "market": market,
            "origin": first.origin,
            "dest": last.dest,
            "leg_ids": [row.id for row in path],
            "n_legs": len(path),
            "duration_h": (last.arrival_utc_seconds - first.departure_utc_seconds)
            / 3600,
            "connection_wait_minutes": 0.0
            if len(path) == 1
            else (last.departure_utc_seconds - first.arrival_utc_seconds) / 60,
            "distance_mi": sum(row.distance_mi for row in path),
            "departure_local": first.departure_local,
            "departure_utc_seconds": first.departure_utc_seconds,
            "arrival_utc_seconds": last.arrival_utc_seconds,
        }

    for (_, _), group in legs.groupby(["cohort_date", "marketing_carrier"], sort=True):
        records = list(group.itertuples(index=False))
        for row in records:
            item = product([row])
            if item:
                yield item
        incoming = [row for row in records if row.dest == hub]
        outgoing = [row for row in records if row.origin == hub]
        for a in incoming:
            for b in outgoing:
                wait = (b.departure_utc_seconds - a.arrival_utc_seconds) / 60
                if (
                    a.origin != b.dest
                    and connection_window[0] <= wait <= connection_window[1]
                ):
                    item = product([a, b])
                    if item:
                        yield item


def run(config):
    """Build reproducible calendar tables and expose source reconciliation errors."""
    import pandas as pd

    root = Path(config["out"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        scope = json.loads(Path(config["scope"]).read_text())
        evidence = json.loads(Path(config["processing_results"]).read_text())
        roots = {
            name: Path(stage["output"]) for name, stage in evidence["stages"].items()
        }
        paths = {
            "flights": roots["flights"] / "flights.parquet",
            "operator_capacity": roots["capacity"] / "operator_leg_month.parquet",
            "marketing_capacity": roots["capacity"] / "marketing_leg_month.parquet",
            "selected_markets": roots["markets"] / "model_markets.csv",
        }
        identity = {
            "config": config,
            "implementation_sha256": sha256(__file__),
            "scope_sha256": sha256(config["scope"]),
            "inputs": {
                name: {"path": str(path), "sha256": sha256(path)}
                for name, path in paths.items()
            },
        }
        if (root / "identity.json").exists() and json.loads(
            (root / "identity.json").read_text()
        ) != identity:
            raise ValueError(
                "dated-network source/input identity changed; use a new revision/output"
            )
        atomic_json(root / "identity.json", identity)
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
            flights = pd.read_parquet(paths["flights"])
            operator = pd.read_parquet(paths["operator_capacity"])
            legs, excluded = build_legs(flights, operator, config, scope)
            legs.to_parquet(out / "legs.parquet", index=False)
            excluded.to_csv(out / "excluded_marketing_flights.csv", index=False)
            markets = set(pd.read_csv(paths["selected_markets"]).market)
            sink = ParquetSink(out / "products.parquet")
            buffer = []
            product_counts, connection_count, max_degree = Counter(), 0, 0
            degree = defaultdict(set)
            for item in product_rows(
                legs,
                markets,
                scope["hub_candidate"],
                config["connection_window_minutes"],
            ):
                buffer.append(item)
                product_counts[item["cohort_date"]] += 1
                connection_count += int(item["n_legs"] == 2)
                for leg in item["leg_ids"]:
                    degree[leg].add(item["market"])
                if len(buffer) >= 50000:
                    sink.write(pd.DataFrame(buffer))
                    buffer.clear()
                    atomic_json(
                        out / "progress.json",
                        {"stage": "products", "products_written": sink.rows},
                    )
            if buffer:
                sink.write(pd.DataFrame(buffer))
            product_artifact = sink.close()
            max_degree = max(map(len, degree.values()), default=0)
            dates = (
                legs.groupby(["cohort_date", "split"])
                .agg(
                    legs=("id", "size"),
                    partner_legs=("partner_operated", "sum"),
                    expected_seats=("mean_initial_capacity", "sum"),
                )
                .reset_index()
            )
            dates["products"] = (
                dates.cohort_date.map(product_counts).fillna(0).astype(int)
            )
            dates.to_csv(out / "cohort_counts.csv", index=False)
            keys = LEG_MONTH + ["marketing_carrier"]
            compiled = legs.groupby(keys, as_index=False).agg(
                compiled_departures=("id", "size"),
                compiled_expected_seats=("mean_initial_capacity", "sum"),
            )
            target = pd.read_parquet(paths["marketing_capacity"])
            target = target[target.marketing_carrier.isin(scope["carriers"])]
            checks = target.merge(compiled, on=keys, how="outer")
            checks["compiled_departures"] = checks.compiled_departures.fillna(0)
            checks["compiled_expected_seats"] = checks.compiled_expected_seats.fillna(0)
            checks["departures_difference"] = (
                checks.compiled_departures
                - checks.allocated_departures_performed.fillna(0)
            )
            checks["relative_seat_error"] = abs(
                checks.compiled_expected_seats - checks.allocated_seats.fillna(0)
            ) / checks.allocated_seats.fillna(0).clip(lower=1)
            checks.to_csv(out / "leg_month_reconciliation.csv", index=False)
            summary = {
                "status": "dated_network_prepared_not_calibrated",
                "n_cohorts": len(dates),
                "n_legs": len(legs),
                "n_products": product_artifact["rows"],
                "n_connecting_products": connection_count,
                "partner_operated_legs": int(legs.partner_operated.sum()),
                "max_leg_shared_markets": max_degree,
                "excluded_flights": excluded.reason.value_counts().to_dict()
                if len(excluded)
                else {},
                "source_seat_total_for_core_networks": float(
                    checks.allocated_seats.sum()
                ),
                "compiled_expected_seat_total": float(
                    checks.compiled_expected_seats.sum()
                ),
                "leg_month_rows": len(checks),
                "seat_rows_above_0_1_percent": int(
                    (checks.relative_seat_error > 0.001).sum()
                ),
                "nonzero_departure_rows": int(
                    (checks.departures_difference.abs() > 1e-9).sum()
                ),
                "seconds": time.perf_counter() - started,
                "V1_status": "not_yet_passed; source mismatches/excluded records retained, no rescaling",
                "calibration_status": "not_run",
                "market_spec_status": "not_yet_built",
                "capacity_assumption": "T100 operator-route-month seats/performed departures; two-point private prior preserves that mean; relative spread is assumed, not inferred hidden inventory",
                "calendar_rule": "Observed performed service only; cancelled flights are excluded explicitly, not counted as forecasts or unobserved stochastic cancellations. UTC elapsed-time connections, same actual local departure cohort, retained Q2-selected markets only.",
                "Q3_role": "Schedule/capacity structure only; Q3 traffic/fares and calibration parameters not read",
                "artifacts": {"legs": "legs.parquet", "products": "products.parquet"},
            }
            atomic_json(out / "summary.json", summary)
            atomic_json(
                out / "COMPLETE.json",
                {
                    "identity": identity,
                    "artifacts": {
                        p.name: sha256(p)
                        for p in out.iterdir()
                        if p.is_file() and p.name != "progress.json"
                    },
                },
            )
            atomic_json(
                root / "status.json", {"status": summary["status"], "attempt": str(out)}
            )
            return summary
        except Exception as error:
            atomic_json(
                out / "failure.json",
                {
                    "type": type(error).__name__,
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
