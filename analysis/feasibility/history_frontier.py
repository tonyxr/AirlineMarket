"""Audit sealed history-frontier values without rerunning any exact solve."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from airline_market.campaign.artifacts import atomic_json, file_hash, utc_now
from airline_market.campaign.runner import completed_attempt


def summarize(root):
    """Keep incomplete/limited cells separate from feasible eight-corner coverage."""
    rows, corners, sources = [], defaultdict(dict), []
    for cell in json.loads((root / "cells.json").read_text()):
        p = cell["parameters"]
        row = {"cell_id": cell["cell_id"], "parameters": p,
               "status": "not_completed", "value": None, "wall_seconds": None}
        found = completed_attempt(root / "cells" / cell["cell_id"], cell["config_hash"])
        if found:
            directory, result = found
            row.update(status=result["status"], value=result.get("value"),
                       wall_seconds=result["wall_seconds"], source=str(directory),
                       results_sha256=file_hash(directory / "results.json"))
        if file_hash(p["source_spec"]) != p["source_spec_sha256"]:
            raise ValueError("history frontier frozen source changed")
        source = json.loads(Path(p["source_spec"]).read_text())
        quote_features = {"direct_rival_gap", "min_rival_gap", "linked_scarcity_signal", "rival_availability"}
        focal = source["metadata"].get("focal", source["carriers"][0]["code"])
        pricing = next(x for x in source["pricing"] if x["carrier"] == focal)
        features = [pair[0]["name"] for pair in pricing["fixed_features"]]
        features += [f["name"] for f in pricing["controllable_features"]]
        sources.append(sorted(set(features) & quote_features))
        key = (p["connecting_share"], p["rival_capacity_prior"], p["lag_periods"])
        corners[key][p["arm"]] = row
        rows.append(row)
    violations, unrelated, lag_contrasts = [], [], []
    for corner, arms in sorted(corners.items()):
        for low, high in [("direct", "linked"), ("linked", "oracle"), ("oracle", "full_state")]:
            if all(a in arms and arms[a]["status"] == "solved" for a in [low, high]):
                gap = arms[high]["value"] - arms[low]["value"]
                if gap < -1e-12:
                    violations.append({"corner": corner, "low": low, "high": high, "gap": gap})
        if all(a in arms and arms[a]["status"] == "solved" for a in ["direct", "unrelated"]):
            unrelated.append({"corner": corner,
                "unrelated_minus_direct": arms["unrelated"]["value"] - arms["direct"]["value"]})
        if corner[2] == 0 and (*corner[:2], 1) in corners:
            stale = corners[(*corner[:2], 1)]
            for arm, row in arms.items():
                if row["status"] == "solved" and stale.get(arm, {}).get("status") == "solved":
                    lag_contrasts.append({"corner": corner[:2], "arm": arm,
                        "lag0_minus_lag1": row["value"] - stale[arm]["value"]})
    complete = len(rows) == 40 and len(corners) == 8 and all(
        row["status"] == "solved" and row["wall_seconds"] <= 3600 for row in rows)
    report = {"updated_at": utc_now(), "configured_cells": len(rows),
        "scales": sorted({(r["parameters"]["T"], r["parameters"]["actions"]) for r in rows}),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "all_eight_corners_five_arms_within_hour": complete,
        "largest_exact_scale_selected": False,
        "maximum_sealed_solve_seconds": max((r["wall_seconds"] for r in rows if r["status"] == "solved"), default=None),
        "within_corner_information_order_violations": violations,
        "unrelated_direct_contrasts": unrelated, "descriptive_lag_contrasts": lag_contrasts,
        "quote_dependent_pricing_features": sorted({f for fs in sources for f in fs}),
        "lag_interpretation": "If quote-dependent pricing features change with lag, the feasible finite price class can also change. These cross-spec lag contrasts are descriptive; a pure-information lag claim needs a common price-rule confirmation. Within-corner arm comparisons are kept separate.",
        "scope": "Only the listed sealed history-frontier receipts. Missing/limited values remain null; no solve ETA, largest-scale choice, full V3 certificate, or rerun. The preliminary1GiB limits remain distinct from the registered12GiB ceiling.",
        "rows": rows}
    atomic_json(root / "frontier_coverage.json", report)
    return {k: v for k, v in report.items() if k not in
            ("rows", "unrelated_direct_contrasts", "descriptive_lag_contrasts")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
