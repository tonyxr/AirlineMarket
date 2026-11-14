"""Resolve independent F2 cells; I4 exact scale awaits the F1 decision."""

from pathlib import Path
import hashlib
import json

from airline_market.pipeline.catalog import MarketScope
from airline_market.pipeline.compiler import compile_scope
from airline_market.spec import compile_market, dump_spec


def main():
    """Prepare fixture/Poisson benchmarks without changing active campaign code."""
    sources = {
        name: str(Path("tests/fixtures") / name / "tiny.csv")
        for name in ["t100", "db1b", "ontime", "flightprices", "noaa", "coupon"]
    }
    scope = MarketScope.from_dict(
        "I9_local_fixture_uncalibrated",
        {
            "hubs": ["SEA"],
            "carriers": ["AS", "DL"],
            "departure_date": "2024-08-14",
            "periods": [60, 30, 0],
            "sample_expansion": 10,
            "min_market_pax_per_day": 1,
            "sources": sources,
        },
    )
    spec = compile_scope(scope)
    path = Path("config/instances/I9_fixture.json")
    dump_spec(spec, path)
    cases = [
        ("I0", "toy_hub", {}),
        ("I3", "hub_feeder_small", {}),
        ("I5", "config/instances/I5.json", {}),
        ("I6", "config/instances/I6.json", {}),
        ("I9_fixture", str(path), {}),
    ]
    operations = [
        "engine_cohort",
        "batch_sampler_cohort",
        "kernel_transition",
        "exact_belief_update",
        "particle_belief_update_2000",
        "particle_belief_update_5000",
        "particle_belief_update_20000",
        "factorized_belief_update",
        "mpc_decision",
    ]
    config = {
        "schema": "airline-campaign/1",
        "id": "F2_independent",
        "phase": 1,
        "implementation_revision": "poisson-catalog-reference-v1",
        "task": "benchmark",
        "design": "list",
        "workers": 1,
        "memory_budget_gb": 2,
        "fixed": {"warmup": 1, "repeats": 5, "n_scenarios": 256, "reservation_gb": 2},
        "cells": [
            {
                "instance": name,
                "generator": generator,
                "instance_params": params,
                "operations": [operation],
            }
            for name, generator, params in cases
            for operation in operations
        ],
        "limits": {"seconds": 900, "rss_gb": 2},
        "out": "results/F2/20261111_independent_reference_v1",
        "note": "I4 exact-scale cells await F1 and are not counted complete. One isolated cell per operation preserves unsupported/limited operations without losing other timings. Large exact priors/Poisson enumeration may hit declared 15-minute/2-GiB measurement limits; these are not throughput estimates. Timing excludes setup and reports it separately. Existing batch sampler is Bernoulli-only; failures must be reported explicitly.",
        "fixture": {
            "spec_hash": compile_market(spec).spec_hash,
            "calibrated": False,
            "source_hashes": {
                name: hashlib.sha256(Path(source).read_bytes()).hexdigest()
                for name, source in sources.items()
            },
        },
    }
    Path("config/campaigns/F2.json").write_text(json.dumps(config, indent=2) + "\n")
    print(
        json.dumps(
            {"cells": len(config["cells"]), "I4": "pending F1", "fixture": str(path)}
        )
    )


if __name__ == "__main__":
    main()
