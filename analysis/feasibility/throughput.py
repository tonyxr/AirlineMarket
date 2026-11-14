"""Aggregate completed F2 timings and explicit unsupported/limited operations."""

from pathlib import Path
import argparse
import csv
import json

from airline_market.campaign.artifacts import atomic_json, utc_now
from airline_market.campaign.design import expand_campaign, load_campaign
from airline_market.campaign.runner import completed_attempt


def aggregate(config: dict) -> dict:
    """Keep missing operation costs null; never substitute a limited cell's wall time."""
    root = Path(config["out"])
    rows = []
    for cell in expand_campaign(config):
        p = cell["parameters"]
        found = completed_attempt(root / "cells" / cell["cell_id"], cell["config_hash"])
        result = found[1] if found else {}
        benchmarks = result.get("benchmarks", [])
        measured = (
            benchmarks[0]
            if result.get("status") == "completed" and len(benchmarks) == 1
            else {}
        )
        rows.append(
            {
                "instance": p["instance"],
                "operation": p["operations"][0],
                "status": result.get("status", "unstarted_or_running"),
                "median_seconds": measured.get("median_seconds"),
                "iqr_seconds": measured.get("iqr_seconds"),
                "median_setup_seconds": measured.get("median_setup_seconds"),
                "repeats": measured.get("repeats"),
                "reason": result.get("reason", result.get("error", "")),
                "cell_id": cell["cell_id"],
            }
        )
    root.mkdir(parents=True, exist_ok=True)
    with (root / "throughput.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "updated_at": utc_now(),
        "rows": rows,
        "completed_operations": sum(row["median_seconds"] is not None for row in rows),
        "configured_operations": len(rows),
        "I4_exact_scale": "pending F1; omitted rather than assigned a guessed scale",
        "note": "A failed/limited operation has no throughput estimate. Setup is reported separately, matching the R0 harness.",
    }
    atomic_json(root / "throughput_summary.json", result)
    return result


def main():
    """Write the partial or finished throughput table from matching receipts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    result = aggregate(load_campaign(args.config))
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "rows"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
