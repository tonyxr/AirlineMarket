"""Audit conditional-tail trials against saved original values, retaining limits."""

from pathlib import Path
import argparse
import json

from airline_market.campaign.artifacts import atomic_json, file_hash, utc_now
from airline_market.campaign.runner import completed_attempt


def summarize(root: Path, tolerance: float = 1e-12) -> dict:
    """Check completed receipts and physical identity without rerunning an anchor."""
    rows = []
    for cell in sorted((root / "cells").iterdir()):
        configs = sorted(cell.glob("attempt-*/config.json"))
        if not configs:
            continue
        config = json.loads(configs[-1].read_text())
        parameters = config["parameters"]
        finished = completed_attempt(cell, config["config_hash"])
        row = {"cell": str(cell), "arm": parameters["arm"], "T": parameters["T"],
               "actions": parameters["actions"], "status": "not_completed"}
        if finished:
            directory, result = finished
            row.update(status=result["status"], value=result.get("value"),
                       solve_seconds=result.get("solve_seconds"),
                       peak_rss_bytes=result.get("peak_rss_bytes"),
                       results_sha256=file_hash(directory / "results.json"),
                       prefix_and_final_nodes=result.get("prefix_and_final_nodes"),
                       tail_vector_entries=result.get("tail_vector_entries"),
                       penultimate_belief_queries=result.get("penultimate_belief_queries"))
            source_path = Path(parameters["source_result"])
            source_config = json.loads((source_path.parent / "config.json").read_text())
            source = completed_attempt(source_path.parent.parent, source_config["config_hash"])
            row["anchor_check"] = "not_checked"
            if source and result["status"] == "solved":
                old = source[1]
                same_spec = result["spec_hash"] == old["spec_hash"]
                same_design = all(parameters[k] == source_config["parameters"][k]
                                  for k in ("T", "actions", "arm"))
                row.update(same_spec=same_spec, same_design=same_design,
                           original_status=old["status"],
                           original_results_sha256=file_hash(source_path))
                if old["status"] == "solved":
                    error = abs(result["value"] - old["value"])
                    row.update(absolute_error=error,
                               anchor_check="passed" if same_spec and same_design and error <= tolerance else "failed")
                else:
                    row["anchor_check"] = "original_value_unavailable"
        rows.append(row)
    return {"schema": "airline-conditional-tail-audit/1", "created_at": utc_now(),
            "absolute_tolerance": tolerance, "trials": rows,
            "scope": "Saved matching model values only. No general-kernel or lag-history certificate is implied. Limited pilot results do not establish failure at the larger registered F1 resource budget."}


def main():
    """Publish analysis separately from sealed solver results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    atomic_json(args.root / "audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
