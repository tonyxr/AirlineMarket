"""Aggregate sealed F5 cells without treating pilot accuracy as final acceptance."""

from collections import defaultdict
from pathlib import Path
import argparse
import json

import pandas as pd

from airline_market.campaign.artifacts import atomic_json, file_hash, utc_now
from airline_market.campaign.runner import completed_attempt


def aggregate(root):
    """Retain strict/set agreement, paired uncertainty, costs and shared-state audit."""
    root = Path(root)
    records, provenance = [], []
    sealed = 0
    signatures = defaultdict(set)
    for cell in json.loads((root / "cells.json").read_text()):
        p = cell["parameters"]
        found = completed_attempt(root / "cells" / cell["cell_id"], cell["config_hash"])
        if found is None:
            continue
        sealed += 1
        directory, result = found
        common = {
            "cell_id": cell["cell_id"],
            **{key: p.get(key) for key in ["instance", "tail", "scenarios", "method"]},
            "status": result["status"],
            "cohorts": result.get("n_cohorts"),
            "wall_seconds": result.get("wall_seconds"),
        }
        if result["status"] != "completed":
            records.append({**common, "reason": result.get("reason")})
            continue
        decisions = json.loads((directory / "decisions.json").read_text())
        episodes = json.loads((directory / "episodes.json").read_text())
        if len(episodes) != result["n_cohorts"]:
            raise ValueError("incomplete F5 cohort artifact")
        for d in decisions:
            if d["mode"] == "shared_fixed_trajectory":
                signatures[(p["instance"], d["seed"], d["period"])].add(
                    d["state_signature"]
                )
        for mode, metrics in result["comparison"].items():
            final = result["n_cohorts"] >= 256
            ci = result["paired_95_ci"]
            revenue = result["mean_cohort_revenue"]
            records.append(
                {
                    **common,
                    "mode": mode,
                    **metrics,
                    "mean_direct_minus_fixed_profit": result[
                        "mean_direct_minus_fixed_profit"
                    ],
                    "ci_low": None if ci is None else ci[0],
                    "ci_high": None if ci is None else ci[1],
                    "mean_cohort_revenue": revenue,
                    "agreement_pass": None
                    if not final
                    else metrics["q_optimal_within_1e_12"] >= 0.95,
                    "regret_pass": None
                    if not final
                    else metrics["mean_regret_currency"] <= 0.0005 * revenue,
                    "contrast_pass": None
                    if not final or ci is None
                    else ci[0] >= -0.001 * revenue,
                    "criterion_status": result["criterion_status"],
                    "projected_256_cohort_wall_at_same_scenarios": result[
                        "wall_seconds"
                    ]
                    * 256
                    / result["n_cohorts"],
                }
            )
        provenance.append(
            {
                "cell_id": cell["cell_id"],
                "files": {
                    name: file_hash(directory / name)
                    for name in ["results.json", "decisions.json", "episodes.json"]
                },
            }
        )
    pd.DataFrame(records).to_csv(root / "comparison.csv", index=False)
    mismatches = [
        {
            "instance": key[0],
            "seed": key[1],
            "period": key[2],
            "distinct_signatures": len(values),
        }
        for key, values in signatures.items()
        if len(values) > 1
    ]
    report = {
        "created_at": utc_now(),
        "sealed_cells": sealed,
        "completed_comparison_cells": len(provenance),
        "configured_cells": len(json.loads((root / "cells.json").read_text())),
        "status": "partial"
        if sealed < len(json.loads((root / "cells.json").read_text()))
        else "complete"
        if sealed == len(provenance)
        else "complete_with_failures",
        "shared_state_contexts_checked": len(signatures),
        "shared_state_mismatches": mismatches,
        "primary_decision_metrics": "On-policy Q-optimal-set agreement and regret at all visited decisions; shared fixed trajectories compare methods at identical states",
        "selection_status": "not_selected_from_pilot"
        if any(r.get("criterion_status") == "pilot_only" for r in records)
        else "full_budget_evidence_available_no_automatic_selection"
        if provenance
        else "no_successful_comparison",
        "cost_projection_note": "Same-scenario linear extrapolation from observed wall/cohorts only. Includes cold exact-Q/setup and assumes constant per-cohort cost; it is not measured256-cohort cost or an extrapolation to larger scenario budgets.",
        "provenance": provenance,
    }
    atomic_json(root / "comparison_summary.json", report)
    return {
        key: report[key]
        for key in [
            "status",
            "sealed_cells",
            "configured_cells",
            "shared_state_contexts_checked",
            "shared_state_mismatches",
            "selection_status",
        ]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    print(json.dumps(aggregate(args.root), indent=2))
