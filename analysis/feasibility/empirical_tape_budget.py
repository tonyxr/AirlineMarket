"""Analytical dense-tape sizing on an audited Q2 network, without allocating it.

Arrival scales and a uniform booking-period profile below are explicit,
uncalibrated sensitivities. Only the physical leg/product table is empirical.
"""

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from airline_market.campaign.artifacts import atomic_json, file_hash, utc_now
from airline_market.core.rng import StreamRegistry


def main():
    """Measure count-only setup and calculate the exact proposed array dimensions."""
    root = Path("results/V1/20261111_dated_network_v1/attempt-0001")
    date = "2022-06-15"
    cohorts = pd.read_csv(root / "cohort_counts.csv")
    row = cohorts.loc[cohorts.cohort_date == date].iloc[0]
    products = pd.read_parquet(root / "products.parquet", filters=[("cohort_date", "==", date)])
    counts = products.groupby("market").size()
    T, P = 60, len(products)
    sizes = []
    for scale in [0.8, 1.2, 2.0]:
        started = time.perf_counter()
        rng = StreamRegistry(400000, "calibration/resource/count_only").generator(int(scale * 10))
        arrivals = rng.poisson(scale * row.expected_seats / T, size=T)
        count_seconds = time.perf_counter() - started
        max_requests = int(arrivals.max())
        dense_bytes = T * max_requests * (P + 1) * np.dtype(np.float64).itemsize
        sizes.append({"assumed_requests_per_expected_seat": scale,
                      "assumed_period_profile": "uniform; uncalibrated",
                      "count_only_seed": 400000, "potential_requests": int(arrivals.sum()),
                      "maximum_period_requests": max_requests,
                      "count_only_seconds": count_seconds,
                      "dense_gumbel_shape": [T, max_requests, P + 1],
                      "dense_gumbel_bytes_analytical": dense_bytes,
                      "dense_gumbel_GiB_analytical": dense_bytes / 2**30})
    result = {"created_at": utc_now(), "cohort": date, "split": "Q2 calibration structure",
              "physical_legs": int(row.legs), "physical_products": P,
              "expected_initial_seats": float(row.expected_seats),
              "markets": len(counts), "median_products_per_market": float(counts.median()),
              "max_products_per_market": int(counts.max()), "rows": sizes,
              "formula": "float64 gumbels: T * max_requests_per_period * (global_products+1) *8 bytes, as sim/tape.py allocates; excludes every other array/object and temporary workspace.",
              "full_tape_measured_seconds": None, "full_tape_measured_peak_bytes": None,
              "calibrated_demand": False, "calibration_fit_run": False,
              "interpretation": "Analytical memory sensitivity and measured count-only setup, not a full tape/simulator benchmark or demand fit. CEM cohorts can execute sequentially; multiplying by512 is not a mandatory resident-memory requirement. A full behavioral MarketSpec and bounded actual tape/cohort measurement are still required. Market-local taste generation is a prospective optimization, not implemented here.",
              "source_hashes": {str(root / name): file_hash(root / name)
                                for name in ["cohort_counts.csv", "products.parquet"]}}
    out = Path("results/P3/20261112_resource_projection_v1")
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "dense_tape_projection.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("source_hashes", "interpretation")}, indent=2))


if __name__ == "__main__":
    main()
