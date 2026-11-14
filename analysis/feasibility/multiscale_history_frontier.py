"""Audit separate40-cell scales without conflating identical corner/arm labels."""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from airline_market.campaign.artifacts import atomic_json, file_hash, utc_now
from airline_market.campaign.runner import completed_attempt


def summarize(root):
    groups = defaultdict(list)
    for c in json.loads((root / "cells.json").read_text()):
        p = c["parameters"]
        if file_hash(p["source_spec"]) != p["source_spec_sha256"]:
            raise ValueError("frozen frontier physical source changed")
        row = {
            "cell_id": c["cell_id"],
            "T": p["T"],
            "actions": p["actions"],
            "connecting_share": p["connecting_share"],
            "rival_capacity_prior": p["rival_capacity_prior"],
            "lag_periods": p["lag_periods"],
            "arm": p["arm"],
            "status": "not_completed",
            "value": None,
            "wall_seconds": None,
            "results_sha256": None,
        }
        found = completed_attempt(root / "cells" / c["cell_id"], c["config_hash"])
        if found:
            d, r = found
            row.update(
                status=r["status"],
                value=r.get("value"),
                wall_seconds=r["wall_seconds"],
                results_sha256=file_hash(d / "results.json"),
            )
        groups[(p["T"], p["actions"])].append(row)
    scales = []
    allrows = []
    for scale, rows in sorted(groups.items()):
        keys = {
            (
                r["connecting_share"],
                r["rival_capacity_prior"],
                r["lag_periods"],
                r["arm"],
            )
            for r in rows
        }
        if len(rows) != 40 or len(keys) != 40:
            raise ValueError(
                "each declared scale must contain40unique corner/arm factors"
            )
        corners = defaultdict(dict)
        for r in rows:
            corners[
                (r["connecting_share"], r["rival_capacity_prior"], r["lag_periods"])
            ][r["arm"]] = r
        if len(corners) != 8 or any(
            set(a) != set(["direct", "linked", "unrelated", "oracle", "full_state"])
            for a in corners.values()
        ):
            raise ValueError("missing8corner x5arm coverage")
        limited = [r for r in rows if r["status"] not in ["solved", "not_completed"]]
        over = [r for r in rows if r["status"] == "solved" and r["wall_seconds"] > 3600]
        complete = all(r["status"] == "solved" for r in rows)
        criterion = (complete and not over) if complete or limited or over else None
        violations = []
        for corner, arms in corners.items():
            for low, high in [
                ("direct", "linked"),
                ("linked", "oracle"),
                ("oracle", "full_state"),
            ]:
                if all(arms[a]["status"] == "solved" for a in [low, high]):
                    delta = arms[high]["value"] - arms[low]["value"]
                    if delta < -1e-12:
                        violations.append(
                            {"corner": corner, "low": low, "high": high, "gap": delta}
                        )
        scales.append(
            {
                "T": scale[0],
                "actions": scale[1],
                "configured_cells": 40,
                "sealed_cells": sum(r["status"] != "not_completed" for r in rows),
                "status_counts": dict(Counter(r["status"] for r in rows)),
                "all8corners5arms_within_hour_under_measured_limits": criterion,
                "within_corner_order_violations": violations,
            }
        )
        allrows.extend(rows)
    report = {
        "updated_at": utc_now(),
        "scales": scales,
        "largest_exact_scale_selected": False,
        "known_verified_lower_bound": "results/F1_history/20261112_T5A4_v1/frontier_coverage.json",
        "scope": "Each(T,actions)scale retains its own8corners/5arms. Only sealed matching receipts; unknown/limited values remain null. Feasibility is under the measured preliminary limits, not proof of12GiB failure. No across-scale value mixing or tree ETA.",
    }
    atomic_json(root / "multiscale_coverage.json", report)
    with (root / "multiscale_frontier.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0]))
        w.writeheader()
        w.writerows(allrows)
    return report


if __name__ == "__main__":
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2))
