"""Aggregate F4 small-instance evidence without inventing a global particle budget."""

from pathlib import Path
import argparse
import json

import pandas as pd

from airline_market.campaign.artifacts import atomic_json, file_hash, utc_now
from airline_market.campaign.runner import completed_attempt


def aggregate(root):
    """Retain complete-law TV, inventory projection, decision tests and failures."""
    root = Path(root)
    cells = json.loads((root / "cells.json").read_text())
    rows, provenance = [], []
    for cell in cells:
        found = completed_attempt(root / "cells" / cell["cell_id"], cell["config_hash"])
        if found is None:
            continue
        directory, result = found
        p = cell["parameters"]
        row = {
            "cell_id": cell["cell_id"],
            "instance": p["instance"],
            "arm": p["arm"],
            "particles": p["particles"],
            "status": result["status"],
            "wall_seconds": result.get("wall_seconds"),
            "reason": result.get("reason"),
        }
        if result["status"] == "completed":
            path = directory / "posterior_rows.json"
            if file_hash(path) != result["rows_sha256"]:
                raise ValueError("changed F4 posterior artifact")
            frames = json.loads(path.read_text())
            if len(frames) != result["observations"]:
                raise ValueError("incomplete F4 frame artifact")
            row.update(
                {
                    key: result[key]
                    for key in [
                        "n_cohorts",
                        "observations",
                        "decisions",
                        "mean_tv",
                        "max_tv",
                        "mean_inventory_tv",
                        "max_leg_marginal_error",
                        "max_joint_event_error",
                        "decision_agreement",
                        "strict_coefficient_agreement",
                        "mean_update_ms",
                        "mean_prior_reset_seconds",
                        "provisional_criteria_pass",
                    ]
                }
            )
            row["ess_post_resampling_min"] = min(frame["ess"] for frame in frames)
            row["ess_pre_resampling_min"] = None
            provenance.append(
                {
                    "cell_id": cell["cell_id"],
                    "rows_sha256": file_hash(path),
                    "results_sha256": file_hash(directory / "results.json"),
                }
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(root / "particles.csv", index=False)
    summary = {
        "created_at": utc_now(),
        "configured_cells": len(cells),
        "sealed_cells": len(rows),
        "completed_accuracy_cells": len(provenance),
        "status": "partial"
        if len(rows) < len(cells)
        else "complete"
        if len(rows) == len(provenance)
        else "complete_with_failures",
        "particle_selection_status": "not_selected",
        "remaining": "I4 corners at measured exact scale; I5/I6 100k-particle references; decision confirmation with final F5 config",
        "ess_scope": "v1 rows report ESS of current weights after any resampling. Pre-resampling ESS was not written by this task and remains null; do not interpret post-resampling ESS as evidence against degeneracy. Record pre-resampling diagnostics in subsequent fresh-confirmation/hub studies.",
        "provenance": provenance,
    }
    atomic_json(root / "particles_summary.json", summary)
    return {
        key: summary[key]
        for key in [
            "status",
            "configured_cells",
            "sealed_cells",
            "completed_accuracy_cells",
            "particle_selection_status",
        ]
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    print(json.dumps(aggregate(args.root), indent=2))
