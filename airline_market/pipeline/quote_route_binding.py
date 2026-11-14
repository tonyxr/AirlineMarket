"""Exact reported UTC-path binding to existing observed-service dated resources."""

import itertools
import json
import math
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo


class BindingIndex:
    """Keep all compatible resource/product candidates; never select by row order."""

    def __init__(self, legs, products):
        self.legs = {r["id"]: r for r in legs}
        if len(self.legs) != len(legs):
            raise ValueError("duplicate dated leg IDs")
        self.exact = defaultdict(list)
        self.departure = defaultdict(list)
        self.route_day = defaultdict(list)
        for r in legs:
            prefix = (r["marketing_carrier"], r["origin"], r["dest"])
            self.exact[
                (
                    *prefix,
                    float(r["departure_utc_seconds"]),
                    float(r["arrival_utc_seconds"]),
                )
            ].append(r["id"])
            self.departure[(*prefix, float(r["departure_utc_seconds"]))].append(r["id"])
            self.route_day[(*prefix, r["cohort_date"])].append(r["id"])
        self.products = defaultdict(list)
        for p in products:
            self.products[tuple(p["leg_ids"])].append(p["id"])
        for index in [self.exact, self.departure, self.route_day, self.products]:
            for key in index:
                index[key].sort()

    def compatible(self, ids, code):
        """Reported carrier may name the marketer or actual retained operator."""
        return [
            i
            for i in ids
            if code
            in (self.legs[i]["marketing_carrier"], self.legs[i]["operating_carrier"])
        ]


def bind_reported(values, index, zones, window, hub):
    """Apply exact timestamps and the previously declared same-day/window rule.

    Values0..8 are date, O,D, resolved network, departure/arrival epoch vectors,
    departure/arrival airport vectors and reported segment carriers. Fare/cabin
    distinctions are intentionally excluded from this physical-path signature.
    """
    row = {
        "departure_date": values[0],
        "origin": values[1],
        "dest": values[2],
        "network": values[3],
        "reported_physical_payload": json.dumps(values, separators=(",", ":")),
        "structural_valid": False,
        "connection_window_ok": False,
        "same_local_cohort": False,
        "n_segments": 0,
        "wait_minutes": None,
        "leg_candidates": [],
        "departure_candidate_counts": [],
        "route_day_candidate_counts": [],
        "dated_product_ids": [],
        "status": "invalid_reported_path",
        "reason": None,
    }
    try:
        arrays = [str(v).split("||") for v in values[4:9]]
        deps, arrs, origins, dests, codes = arrays
        n = len(deps)
        row["n_segments"] = n
        if n not in [1, 2] or any(len(x) != n for x in arrays):
            raise ValueError("segment_vector_lengths")
        dep = [float(v) for v in deps]
        arr = [float(v) for v in arrs]
        if any(not math.isfinite(v) or not v.is_integer() for v in dep + arr):
            raise ValueError("invalid_epoch_seconds")
        dep = list(map(int, dep))
        arr = list(map(int, arr))
        if any(a <= d for d, a in zip(dep, arr)):
            raise ValueError("nonpositive_segment_duration")
        if (origins[0], dests[-1]) != (values[1], values[2]):
            raise ValueError("endpoint_mismatch")
        if any(dests[k] != origins[k + 1] for k in range(n - 1)):
            raise ValueError("disconnected_airports")
        if n == 2 and (dests[0] != hub or origins[0] == dests[-1]):
            raise ValueError("not_declared_hub_path")
        if n == 1 and hub not in (origins[0], dests[0]):
            raise ValueError("not_declared_hub_local")
        dates = [
            datetime.fromtimestamp(t, ZoneInfo(zones[o])).date().isoformat()
            for t, o in zip(dep, origins)
        ]
        if dates[0] != values[0]:
            raise ValueError("first_local_departure_date_mismatch")
        wait = 0.0 if n == 1 else (dep[1] - arr[0]) / 60
        if wait < 0:
            raise ValueError("departure_precedes_connection_arrival")
        row.update(
            structural_valid=True,
            wait_minutes=wait,
            same_local_cohort=all(t == values[0] for t in dates),
            connection_window_ok=n == 1 or window[0] <= wait <= window[1],
        )
        for o, d, t, u, code, day in zip(origins, dests, dep, arr, codes, dates):
            prefix = (values[3], o, d)
            row["leg_candidates"].append(
                index.compatible(index.exact.get((*prefix, t, u), []), code)
            )
            row["departure_candidate_counts"].append(
                len(index.compatible(index.departure.get((*prefix, t), []), code))
            )
            row["route_day_candidate_counts"].append(
                len(index.compatible(index.route_day.get((*prefix, day), []), code))
            )
        if not row["connection_window_ok"]:
            row["status"] = "outside_declared_connection_window"
        elif not row["same_local_cohort"]:
            row["status"] = "outside_same_day_compiler"
        elif any(not ids for ids in row["leg_candidates"]):
            row["status"] = "missing_exact_leg"
        else:
            product_ids = []
            for path in itertools.product(*row["leg_candidates"]):
                product_ids.extend(index.products.get(path, []))
            row["dated_product_ids"] = sorted(product_ids)
            row["status"] = (
                "no_existing_dated_product"
                if not product_ids
                else "unique_exact_dated_product"
                if len(product_ids) == 1
                else "ambiguous_exact_dated_products"
            )
    except (ValueError, KeyError, OverflowError, OSError) as error:
        row["reason"] = str(error)
    return row
