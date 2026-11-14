"""Audit paired general/compact solves without treating limits as agreement."""

from pathlib import Path
import argparse
import json

from airline_market.campaign.artifacts import atomic_json, file_hash, utc_now
from airline_market.campaign.runner import completed_attempt


def compare(root: Path, tolerance: float = 1e-10) -> dict:
    """Verify receipts, model identity and absolute value differences (F1/V7)."""
    rows = []
    for cell in sorted((root / "cells").iterdir()):
        configs = sorted(cell.glob("attempt-*/config.json"))
        if not configs:
            continue
        config = json.loads(configs[-1].read_text())
        parameters = config["parameters"]
        reference = completed_attempt(cell, config["config_hash"])
        source = Path(parameters["compact_source_cell"])
        compact_config = json.loads((source / "config.json").read_text())
        compact = completed_attempt(source.parent, compact_config["config_hash"])
        row = {
            "arm": parameters["arm"],
            "reference_cell": str(cell),
            "compact_source_cell": str(source),
            "status": "not_checked",
        }
        if not reference or not compact:
            row["reason"] = "a completed matching receipt is missing"
        else:
            rd, cd = reference[1], compact[1]
            row.update(reference_status=rd["status"], compact_status=cd["status"])
            if rd["status"] != "solved" or cd["status"] != "solved":
                row["reason"] = "both cells must be solved; limits are not agreement"
            else:
                same_spec = (
                    rd["spec_hash"]
                    == cd["spec_hash"]
                    == parameters["expected_spec_hash"]
                )
                same_design = all(
                    parameters[key] == compact_config["parameters"][key]
                    for key in ("T", "actions", "arm")
                )
                error = abs(rd["value"] - cd["value"])
                row.update(
                    status="passed"
                    if same_spec and same_design and error <= tolerance
                    else "failed",
                    spec_hash=rd["spec_hash"],
                    same_spec=same_spec,
                    same_design=same_design,
                    reference_value=rd["value"],
                    compact_value=cd["value"],
                    absolute_error=error,
                    reference_nodes=rd["nodes"],
                    compact_nodes=cd["nodes"],
                    reference_results_sha256=file_hash(reference[0] / "results.json"),
                    compact_results_sha256=file_hash(compact[0] / "results.json"),
                )
        rows.append(row)
    return {
        "schema": "airline-exact-crosschecks/1",
        "created_at": utc_now(),
        "absolute_tolerance": tolerance,
        "status": "passed"
        if rows and all(row["status"] == "passed" for row in rows)
        else "incomplete_or_failed",
        "checks": rows,
        "scope": "Only the listed completed compact/general cells. This is not a frontier-wide certificate or a check of compact-incompatible models.",
    }


def main():
    """Save an explicit audit alongside the unmodified scientific solve outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = compare(args.root)
    atomic_json(args.root / "crosschecks.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
