"""Audit bounded hub SMC probes without treating prior resets as full F4 sizing."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from airline_market.campaign.artifacts import atomic_json, file_hash, utc_now
from airline_market.campaign.runner import completed_attempt


def summarize(root):
    """Keep per-stage timings, paired scientific hashes, strict failures and limits."""
    cells = json.loads((root / "cells.json").read_text())
    rows, groups = [], defaultdict(dict)
    for cell in cells:
        p = cell["parameters"]
        row = {
            "cell_id": cell["cell_id"],
            "instance": p["instance"],
            "particles": p["particles"],
            "implementation": p["implementation"],
            "operation": p["operation"],
            "status": "not_completed",
            "prior_reset_seconds": None,
            "update_seconds": None,
            "peak_rss_bytes": None,
            "wholeworker_seconds": None,
        }
        found = completed_attempt(root / "cells" / cell["cell_id"], cell["config_hash"])
        measurements = None
        if found:
            directory, result = found
            row.update(
                status=result["status"],
                peak_rss_bytes=result.get("peak_rss_bytes"),
                wholeworker_seconds=result["wall_seconds"],
                reason=result.get("reason"),
                results_sha256=file_hash(directory / "results.json"),
            )
            path = directory / "measurements.json"
            if path.exists():
                measurements = json.loads(path.read_text())
                if (
                    result.get("measurements_sha256")
                    and file_hash(path) != result["measurements_sha256"]
                ):
                    raise ValueError("sealed probe measurements changed")
                row.update(
                    prior_reset_seconds=measurements["prior_reset_seconds"],
                    update_seconds=measurements["update_seconds"],
                    initial=measurements["initial"],
                    after_update=measurements["after_update"],
                    failed_stage=measurements.get("failed_stage"),
                    measurements_sha256=file_hash(path),
                )
        groups[(p["instance"], p["particles"], p["operation"])][p["implementation"]] = (
            p,
            row,
            measurements,
        )
        rows.append(row)
    pairs = []
    for key, group in groups.items():
        if set(group) != {"cached_reference", "template"}:
            continue
        pa, a, ma = group["cached_reference"]
        pb, b, mb = group["template"]
        same_design = all(pa[k] == pb[k] for k in pa if k != "implementation")
        pair = {
            "instance": key[0],
            "particles": key[1],
            "operation": key[2],
            "status": "not_checked",
            "reference_status": a["status"],
            "template_status": b["status"],
            "same_design": same_design,
        }
        if not same_design:
            raise ValueError(
                "paired hub probes have different physical/planning inputs"
            )
        if ma and mb and ma["initial"] and mb["initial"]:
            initial_equal = (
                ma["initial_observation_sha256"] == mb["initial_observation_sha256"]
                and ma["initial"] == mb["initial"]
            )
            pair["initial_scientific_identity"] = initial_equal
            if not initial_equal:
                raise ValueError(
                    "construction substitution changed initial particles or RNG"
                )
            if a["prior_reset_seconds"] and b["prior_reset_seconds"]:
                pair["single_case_reset_wall_ratio"] = (
                    a["prior_reset_seconds"] / b["prior_reset_seconds"]
                )
            if a["status"] == b["status"] == "completed":
                if ma["after_update"] != mb["after_update"]:
                    raise ValueError("construction substitution changed the SMC update")
                pair["status"] = "passed"
            elif a["status"] == b["status"] == "failed":
                same_failure = (
                    ma.get("error_type") == mb.get("error_type")
                    and ma.get("failed_stage") == mb.get("failed_stage")
                    and ma.get("reason") == mb.get("reason")
                    and ma.get("failed_rng_sha256") == mb.get("failed_rng_sha256")
                )
                pair.update(
                    status="same_strict_failure"
                    if same_failure
                    else "different_failure",
                    same_failure_and_RNG=same_failure,
                )
                if not same_failure:
                    raise ValueError("paired strict failure or RNG differs")
        pairs.append(pair)
    result = {
        "updated_at": utc_now(),
        "configured_cells": len(cells),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "pairs": pairs,
        "rows": rows,
        "N_selected": None,
        "full_F4_pass": None,
        "scope": "One-cohort engineering stages. Initial100k-particle reset is not a full reference trajectory or exact posterior. Single-case wall ratios under contention are not randomized throughput estimates. Strict zero weights/limits remain failures and give no whole-operation throughput. Full64-cohort/all-period/final-planner sizing remains pending.",
    }
    atomic_json(root / "probe_summary.json", result)
    return {k: v for k, v in result.items() if k not in ["rows", "pairs"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
