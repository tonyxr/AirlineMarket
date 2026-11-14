"""Combine new exact lag-one anchors with sealed, unchanged lag-zero results."""

from pathlib import Path
import argparse
import json

from airline_market.campaign.artifacts import atomic_json, file_hash, utc_now
from airline_market.campaign.runner import completed_attempt


def summarize(root, reuse_path):
    """Keep limits separate and report anchor coverage without declaring a frontier."""
    reuse = json.loads(reuse_path.read_text())
    rows = []
    for source in reuse["records"]:
        directory = Path(source["source_cell"])
        for name, key in [("spec.json", "source_spec_sha256"),
                          ("results.json", "source_result_sha256"),
                          ("COMPLETE.json", "source_receipt_sha256")]:
            if file_hash(directory / name) != source[key]:
                raise ValueError(f"changed reused anchor: {directory / name}")
        cfg = json.loads((directory / "config.json").read_text())
        _, result = completed_attempt(directory.parent, cfg["config_hash"])
        rows.append({"parameters": source["parameters"], "reused": True,
                     "status": result["status"], "value": result.get("value"),
                     "wall_seconds": result["wall_seconds"], "source": str(directory),
                     "result_sha256": source["source_result_sha256"]})
    for cell in json.loads((root / "cells.json").read_text()):
        found = completed_attempt(root / "cells" / cell["cell_id"], cell["config_hash"])
        row = {"parameters": cell["parameters"], "reused": False,
               "status": "not_completed", "value": None, "wall_seconds": None}
        if found:
            directory, result = found
            row.update(status=result["status"], value=result.get("value"),
                       wall_seconds=result["wall_seconds"], source=str(directory),
                       result_sha256=file_hash(directory / "results.json"))
        rows.append(row)
    corners = {}
    for row in rows:
        p = row["parameters"]
        key = f"share={p['connecting_share']};prior={p['rival_capacity_prior']};lag={p['lag_periods']}"
        corners.setdefault(key, {})[p["arm"]] = {
            "status": row["status"], "value": row["value"],
            "within_hour": row["status"] == "solved" and row["wall_seconds"] <= 3600}
    complete = len(rows) == 40 and len(corners) == 8 and all(
        len(arms) == 5 and all(x["within_hour"] for x in arms.values())
        for arms in corners.values())
    result = {"updated_at": utc_now(), "reused_lag0": len(reuse["records"]),
              "new_lag1_solved": sum(not x["reused"] and x["status"] == "solved" for x in rows),
              "anchor_T": 4, "anchor_actions": 4,
              "all_eight_corners_five_arms_within_hour": complete,
              "largest_exact_scale_selected": False,
              "corners": corners, "rows": rows,
              "scope": "Anchor coverage only. New cells use a stricter 1GiB resource ceiling. Limits have no exact value and do not establish failure at12GiB. Larger frontier and independent kernel checks remain separate."}
    atomic_json(root / "anchor_coverage.json", result)
    return {k: v for k, v in result.items() if k not in ("corners", "rows")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("reuse", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.reuse), indent=2))
