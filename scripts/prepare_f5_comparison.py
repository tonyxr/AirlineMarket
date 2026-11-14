"""Resolve train-only tail receipts into pilot and full F5 comparison catalogs."""

from pathlib import Path
import argparse
import json

from airline_market.campaign.artifacts import file_hash
from airline_market.campaign.runner import completed_attempt


def catalogs(root, pilot_output, full_output, pilot_cohorts=2, pilot_scenarios=128):
    """Keep every registered budget/method/tail factor and explicit input hashes."""
    root = Path(root)
    inputs = []
    for cell in json.loads((root / "cells.json").read_text()):
        found = completed_attempt(root / "cells" / cell["cell_id"], cell["config_hash"])
        if found is None or found[1]["status"] != "completed":
            raise ValueError("train-only tail fitting is incomplete")
        path = found[0] / "tails.json"
        if file_hash(path) != found[1]["tails_sha256"]:
            raise ValueError("fitted-tail artifact changed")
        inputs.append(
            {
                "instance": cell["parameters"]["instance"],
                "instance_generator": cell["generator"],
                "T": cell["parameters"]["T"],
                "actions": cell["parameters"]["actions"],
                "tails_file": str(path),
                "tails_sha256": file_hash(path),
            }
        )
    methods = [
        {"method": "all", "selection": "all", "protect_incumbent": False, "k_se": 0},
        {
            "method": "split",
            "selection": "split",
            "protect_incumbent": False,
            "k_se": 0,
        },
        *[
            {
                "method": f"protected_{k}",
                "selection": "all",
                "protect_incumbent": True,
                "k_se": k,
            }
            for k in (1, 2, 3)
        ],
        {
            "method": "racing",
            "selection": "racing",
            "protect_incumbent": True,
            "k_se": 2,
        },
    ]
    base = {
        "schema": "airline-campaign/1",
        "phase": 1,
        "implementation_revision": "f5-indexed-v1",
        "task": "study:airline_market.experiments.f5_study:compare",
        "design": "list",
        "workers": 1,
        "memory_budget_gb": 2,
        "limits": {"seconds": 86400, "rss_gb": 2},
        "note": "Exact Q at common fitted-fixed behavior states and every on-policy decision. Physical tapes paired across methods; indexed planning prefixes shared across budgets/selection, independent reporting. Strict argmax agreement separate from Q ties. Racing uses64->declared max; the2048 row is the full racing configuration, smaller caps are budget ablations. Tail fits are training only, not final P1 tuning.",
    }

    def build(identifier, output, count, scenarios):
        return {
            **base,
            "id": identifier,
            "fixed": {"seed_start": 200000, "n_cohorts": count},
            "cells": [
                {**source, "tail": tail, "scenarios": n, **method}
                for source in inputs
                for tail in ("fixed", "schedule")
                for n in scenarios
                for method in methods
            ],
            "out": output,
            "pilot_only": count < 256,
        }

    return build("F5_pilot", pilot_output, pilot_cohorts, [pilot_scenarios]), build(
        "F5_full", full_output, 256, [32, 64, 128, 256, 512, 1024, 2048]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tails-root", required=True)
    args = parser.parse_args()
    pilot, full = catalogs(
        args.tails_root,
        "results/F5_pilot/20261111_indexed_v1",
        "results/F5/20261111_indexed_v1",
    )
    for name, value in [("F5_pilot", pilot), ("F5_full", full)]:
        path = Path("config/campaigns") / (name + ".json")
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError("changed derived catalog; preserve the existing revision")
        path.write_text(json.dumps(value, indent=2) + "\n")
        print(
            name, len(value["cells"]), "cells", value["fixed"]["n_cohorts"], "cohorts"
        )
