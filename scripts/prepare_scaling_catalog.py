"""Prepare the declared I5/I6 instances and prospective F3 design."""

from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import json

from airline_market.core.hashing import config_hash
from airline_market.pipeline.scaling import scaling_network
from airline_market.spec import dump_spec, compile_market


def write(path: str, data: dict) -> None:
    """Write a human-readable descriptor without changing any live F1 input."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    """Persist medium/large JSON specs and all 144 F3 parameter cells."""
    instances = []
    for identifier, spokes, carriers, periods in [("I5", 8, 2, 20), ("I6", 16, 3, 30)]:
        parameters = {
            "n_spokes": spokes,
            "n_carriers": carriers,
            "periods": periods,
            "name": "hub_feeder_medium" if identifier == "I5" else "hub_feeder_large",
        }
        spec = scaling_network(parameters)
        path = f"config/instances/{identifier}.json"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        dump_spec(spec, path)
        instances.append(
            {
                "id": identifier,
                "path": path,
                "parameters": parameters,
                "spec_hash": compile_market(spec).spec_hash,
                "calibrated": False,
            }
        )
    write(
        "config/instances/scaling_catalog.json",
        {
            "schema": "airline-instance-catalog/1",
            "generator": "airline_market.pipeline.scaling.scaling_network",
            "revision": "poisson-oligopoly-v1",
            "instances": instances,
            "I8": {
                "n_spokes": [2, 4, 8, 16, 32, 50],
                "n_carriers": [2, 3, 4],
                "periods": [10, 20, 40, 60],
            },
        },
    )
    cells = [
        dict(n_spokes=spokes, n_carriers=carriers, periods=periods)
        for spokes, carriers, periods in product(
            [2, 4, 8, 16, 32, 50], [2, 3, 4], [10, 20, 40, 60]
        )
    ]
    common = {
        "schema": "airline-scaling-study/1",
        "id": "F3",
        "phase": 1,
        "revision": "poisson-oligopoly-reference-v1",
        "fixed": {"controller_mode": "external_all", "requests_per_initial_seat": 1.2},
        "n_cohorts": 64,
        "n_profile_cohorts": 1,
        "profile_filter": {"n_spokes": 50, "n_carriers": 3, "periods": 60},
        "seed_start": 200000,
        "workers": 1,
        "limits": {"seconds": 14400, "rss_gb": 3},
        "cells": cells,
        "timing_scope": "Engine creation, control reset, tape generation, booking and accounting included. Compile/setup and separate profiled cohorts excluded.",
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    common["out"] = f"results/F3/{stamp}_{config_hash(common)[:8]}"
    write("config/campaigns/F3.json", common)
    pilot = {
        **common,
        "id": "F3_pilot_largest",
        "n_cohorts": 1,
        "cells": [{"n_spokes": 50, "n_carriers": 3, "periods": 60}],
        "limits": {"seconds": 3600, "rss_gb": 3},
        "note": "One unprofiled timing cohort plus one additional profiled cohort. A pilot does not establish the 64-cohort criterion.",
    }
    pilot["out"] = (
        f"results/F3_pilot/{stamp}_{config_hash({key: value for key, value in pilot.items() if key != 'out'})[:8]}"
    )
    write("config/campaigns/F3_pilot.json", pilot)


if __name__ == "__main__":
    main()
