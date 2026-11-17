"""S1/R4: pinned saved-policy analysis through the existing resource runner.
Revision: 2 (Nov 16, 2026): W4/R0.4 uses the canonical immutable-result runner.
"""

import json
from pathlib import Path

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.campaign.runner import completed_attempt
from airline_market.experiments.screen_saved_metrics import analyze_settled_setting

ARMS = {"direct", "linked", "unrelated", "oracle", "full_state"}


def pin_setting(root, setting_id):
    """Pin one fully acknowledged solved setting, without new native traversal.

    Policy streams were not original transitive artifacts. Their pinned hashes
    are post-completion snapshots, retained under exactly that scope.
    """
    root = Path(root).resolve()
    acknowledged = set(
        json.loads((root / "settled_cells.json").read_bytes())["cell_ids"]
    )
    cells = [
        c
        for c in json.loads((root / "cells.json").read_bytes())
        if c["parameters"]["setting_id"] == setting_id
    ]
    if len(cells) != 5 or {c["parameters"]["arm"] for c in cells} != ARMS:
        raise ValueError("exactly five registered arms required")
    if any(c["cell_id"] not in acknowledged for c in cells):
        raise ValueError("all five returned publisher futures must be acknowledged")
    sources = {c["parameters"]["source_spec_sha256"] for c in cells}
    if len(sources) != 1:
        raise ValueError("arm physical sources differ")
    files = {root / "config.json", root / "cells.json"}
    for cell in cells:
        found = completed_attempt(
            root / "cells" / cell["cell_id"], cell["config_hash"]
        )
        if found is None or found[1]["status"] != "solved":
            raise ValueError("all five acknowledged arms must be solved")
        attempt, result = found
        if result["parameters"] != cell["parameters"]:
            raise ValueError("saved parameters differ")
        source = Path(cell["parameters"]["source_spec"]).resolve()
        if file_hash(source) != cell["parameters"]["source_spec_sha256"]:
            raise ValueError("physical source changed")
        files.update(
            {
                source,
                attempt / "results.json",
                attempt / "COMPLETE.json",
                attempt / "optimal_policy.pkl.gz",
            }
        )
    return {
        "plan_id": "S1/R4",
        "source_root": str(root),
        "setting_id": setting_id,
        "source_spec_sha256": next(iter(sources)),
        "reservation_gb": 0.5,
        "pinned_files": [
            {"path": str(p), "sha256": file_hash(p)} for p in sorted(files)
        ],
        "policy_hash_scope": "post-completion snapshots, not original transitive seals",
    }


def verify_inputs(parameters):
    """Check registered bytes before and after; never repair a saved input."""
    if not parameters.get("pinned_files"):
        raise ValueError("pinned source manifest required")
    if parameters.get("reservation_gb") != 0.5:
        raise ValueError("this runner uses the released 0.5 GiB slot")
    for source in parameters["pinned_files"]:
        if file_hash(source["path"]) != source["sha256"]:
            raise ValueError("saved-policy input changed: " + source["path"])


def run(cell, context):
    """Run the verified five-arm analysis once; parent enforces wall/RSS limits."""
    p = cell["parameters"]
    if (
        cell["limits"].get("reservation_gb") != 0.5
        or cell["limits"].get("rss_gb") != 0.5
    ):
        raise ValueError("explicit 0.5 GiB resource controls required")
    current = {"stage": "pinned_inputs", "setting_id": p["setting_id"]}
    context.counters = lambda: dict(current)
    verify_inputs(p)
    current["stage"] = "saved_policy_native_transition_traversal"
    metrics = analyze_settled_setting(p["source_root"], p["setting_id"])
    verify_inputs(p)
    metrics.update(
        source_manifest_scope=p["policy_hash_scope"],
        sampled_physical_cohorts=0,
        new_rng_calls=0,
        bellman_optimizations=0,
        actor_calls=0,
    )
    path = context.directory / "saved_policy_metrics.json"
    atomic_json(path, metrics)
    current["stage"] = "complete"
    return {
        "plan_id": "S1/R4",
        "status": "completed",
        "value": None,
        "setting_id": p["setting_id"],
        "metrics_path": path.name,
        "metrics_sha256": file_hash(path),
        "sampled_physical_cohorts": 0,
        "new_rng_calls": 0,
        "bellman_optimizations": 0,
        "actor_calls": 0,
        "native_transition_enumeration": True,
        "scientific_budget": None,
        "policy_hash_scope": p["policy_hash_scope"],
    }
