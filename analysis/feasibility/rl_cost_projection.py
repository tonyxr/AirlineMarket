"""Declared D4 grid-budget projections; no selected training budget or timing claim."""

import csv
import json
import sys
from pathlib import Path

from airline_market.campaign.artifacts import atomic_json, file_hash, utc_now


def project(root):
    """Apply plan page19 grid cohort counts only to matching sealed F6 measurements."""
    table = root / "training_throughput.csv"
    budgets = {"grid_2": 200000, "grid_4": 300000, "I5": 500000, "grid_16": 1000000}
    rows = []
    with table.open() as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "completed" or row["instance"] not in budgets:
                continue
            seconds = float(row["training_seconds_per_cohort"])
            cohorts = budgets[row["instance"]]
            rows.append(
                {
                    "instance": row["instance"],
                    "learner": row["learner"],
                    "cohorts_per_update": int(row["cohorts_per_update"]),
                    "measured_seconds_per_cohort": seconds,
                    "declared_grid_training_cohorts": cohorts,
                    "projected_training_hours_per_seed": cohorts * seconds / 3600,
                    "projected_within_48h": cohorts * seconds <= 48 * 3600,
                    "results_sha256": row["results_sha256"],
                }
            )
    report = {
        "updated_at": utc_now(),
        "rows": rows,
        "source_table_sha256": file_hash(table),
        "training_budget_selected": None,
        "scope": "Linear projections from three measured updates after warm-up under concurrent workload. Not full training runs, uncontended benchmarks, a tuning result or whole-campaign ETA. I5 exactly aliases grid8/2carriers/T20. Distinct I6 has no automatically assigned grid budget here.",
        "decision": "Keep final RL budget null. Finish grid; follow F6 pivot order16spokes3seeds then cap8 if required. A seed-count reduction does not shorten a per-seed run. If8spokes also projects above48h at its declared500k cohorts, record that failure and investigate measured costs; do not silently lower training cohorts.",
    }
    atomic_json(root / "declared_budget_projection.json", report)
    return {
        "matched_measurements": len(rows),
        "projected_over_48h": sum(not r["projected_within_48h"] for r in rows),
        "training_budget_selected": None,
    }


if __name__ == "__main__":
    print(json.dumps(project(Path(sys.argv[1])), indent=2))
