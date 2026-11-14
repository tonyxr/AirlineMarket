"""Aggregate new I4 evidence with genuine pre/post ESS and preserved limits."""

import argparse
import json
from pathlib import Path

import pandas as pd

from airline_market.campaign.artifacts import atomic_json, file_hash, utc_now
from airline_market.campaign.runner import completed_attempt


def validate_frames(frames, parameters):
    """Require every seed/period and keep missing initial pre-ESS distinct."""
    p = parameters
    expected = [
        (seed, t)
        for seed in range(p["seed_start"], p["seed_start"] + p["n_cohorts"])
        for t in range(p["T"] + 1)
    ]
    if [(f["seed"], f["period"]) for f in frames] != expected:
        raise ValueError("missing or repeated validation posterior frame")
    for f in frames:
        if not 0 <= f["full_hidden_tv"] <= 1 + 1e-12:
            raise ValueError("invalid full-hidden TV")
        pre, post = f["pre_resampling_ess"], f["post_resampling_ess"]
        if f["period"] == 0:
            if pre is not None:
                raise ValueError("initial pre-ESS was not retained")
        elif pre is None or not 0 < pre <= p["particles"] * (1 + 1e-12):
            raise ValueError("missing or invalid update pre-ESS")
        if not 0 < post <= p["particles"] * (1 + 1e-12):
            raise ValueError("invalid post-ESS")
        if (f["posted_price_agreement"] is None) != (f["period"] == p["T"]):
            raise ValueError(
                "decision frame is missing or terminal decision is present"
            )


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
            "source_spec_sha256": p["source_spec_sha256"],
            "connecting_share": p["connecting_share"],
            "rival_capacity_prior": p["rival_capacity_prior"],
            "lag_periods": p["lag_periods"],
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
            validate_frames(frames, p)
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
            row["ess_post_resampling_min"] = min(
                frame["post_resampling_ess"] for frame in frames
            )
            row["ess_pre_resampling_min"] = min(
                frame["pre_resampling_ess"] for frame in frames if frame["period"] > 0
            )
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
        "remaining": "Final largest-scale confirmation; I5/I6 100k-particle full-trajectory references; final F5 planner decision confirmation",
        "ess_scope": "New rows retain update pre-resampling diagnostic and actual post-resampling ESS separately. Initial pre-ESS remains null. These I4 corners are at the feasible T5/A4 lower bound, not a selected largest scale; provisional S128 planner and global N remain unselected.",
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
