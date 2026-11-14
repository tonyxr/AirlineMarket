"""Audit sealed paired one-hypothesis tape costs without selecting a particle budget."""

import csv
import json
import sys
from pathlib import Path

from airline_market.campaign.artifacts import atomic_json, file_hash, utc_now
from airline_market.campaign.runner import completed_attempt


def summarize(root):
    rows = []
    for cell in json.loads((root / "cells.json").read_text()):
        found = completed_attempt(root / "cells" / cell["cell_id"], cell["config_hash"])
        if not found:
            continue
        directory, result = found
        row = {
            "instance": cell["parameters"]["instance"],
            "status": result["status"],
            "wall_seconds": result.get("wall_seconds"),
            "peak_rss_bytes": result.get("peak_rss_bytes"),
            "original_proposal_seconds": None,
            "step_proposal_seconds": None,
            "single_case_wall_ratio": None,
            "successor_states": None,
            "joint_mass_estimate": None,
            "original_stored_array_bytes": None,
            "step_stored_array_bytes": None,
            "paired_identity": None,
            "results_sha256": file_hash(directory / "results.json"),
        }
        if result["status"] == "completed":
            old, new = result["cases"]["original"], result["cases"]["step_tape"]
            if (
                not result["prior_identity"]
                or not result["proposal_full_state_mass_rng_identity"]
            ):
                raise ValueError("incomplete numerical parity")
            if (
                old["proposal_and_rng_fingerprint"]
                != new["proposal_and_rng_fingerprint"]
            ):
                raise ValueError("proposal hash mismatch")
            if (
                old["estimated_joint_observation_mass"]
                != new["estimated_joint_observation_mass"]
            ):
                raise ValueError("changed zero/positive estimate")
            row.update(
                original_proposal_seconds=old["proposal_seconds"],
                step_proposal_seconds=new["proposal_seconds"],
                single_case_wall_ratio=result["single_case_proposal_wall_ratio"],
                successor_states=old["successor_states"],
                joint_mass_estimate=old["estimated_joint_observation_mass"],
                original_stored_array_bytes=result["original_stored_array_bytes"],
                step_stored_array_bytes=result["step_stored_array_bytes"],
                paired_identity=True,
            )
        rows.append(row)
    with (root / "paired_proposal_costs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]) if rows else ["status"]
        )
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "updated_at": utc_now(),
        "configured_cells": 2,
        "sealed_cells": len(rows),
        "rows": rows,
        "N_selected": None,
        "full_update_throughput": None,
        "scope": "One initial hypothesis and128draws per method/hub after identical prior setup. Descriptive, unrandomized per-hub timings; no fullupdate,100k/fullhorizon accuracy, F3acceptance or scientific budget. Empty joint estimates and old full-TV/limit failures remain unchanged.",
    }
    atomic_json(root / "paired_proposal_summary.json", result)
    return {k: v for k, v in result.items() if k != "rows"}


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2))
