"""Read-only derived evidence from completed pairs after a reporting-only failure.

The original FAILED receipt/value remain unchanged. New seals cover this audit
and its explicitly post-failure snapshots, never retroactively the original run.
"""

import json
from pathlib import Path

import numpy as np

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.experiments.checkpoint_pair_study import (
    check_inputs,
    paired_summary,
)
from airline_market.spec import compile_market, load_spec


def read_pairs(p):
    """Check all saved source bytes before interpreting any complete pair."""
    directory = Path(p["source_directory"])
    names = [
        "COMPLETE.json",
        "results.json",
        "config.json",
        "spec.json",
        "paired_summary.json",
        "pairs.jsonl",
        "run.log",
        "environment.json",
        "progress.json",
    ]
    for name in names:
        path = directory / name
        if (
            str(path) not in p["source_hashes"]
            or file_hash(path) != p["source_hashes"][str(path)]
        ):
            raise ValueError("saved-pair source snapshot mismatch: " + name)
    receipt = json.loads((directory / "COMPLETE.json").read_text())
    failed = json.loads((directory / "results.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    if (
        receipt["status"] != "failed"
        or failed["status"] != "failed"
        or failed["value"] is not None
        or receipt["results_sha256"] != file_hash(directory / "results.json")
        or receipt["config_hash"] != config["config_hash"]
        or "KeyError: 'status'" not in (directory / "run.log").read_text()
    ):
        raise ValueError("audit requires the preserved original reporting failure")
    source = config["parameters"]
    check_inputs(source)
    if (
        source["seed_start"] != p["seed_start"]
        or source["n_cohorts"] != p["n_cohorts"]
        or p["seed_start"] < 200000
        or p["seed_start"] + p["n_cohorts"] > 300000
    ):
        raise ValueError("saved-pair cohort declaration mismatch")
    compiled = compile_market(load_spec(directory / "spec.json"))
    old = json.loads((directory / "paired_summary.json").read_text())
    if (
        compiled.spec_hash != source["compiled_spec_hash"]
        or old["spec_hash"] != compiled.spec_hash
        or old["pairs_sha256"] != file_hash(directory / "pairs.jsonl")
        or not old["inference_state_unchanged"]
    ):
        raise ValueError("saved-pair physical/state/row identity mismatch")
    rows = [json.loads(x) for x in (directory / "pairs.jsonl").read_text().splitlines()]
    if [r["seed"] for r in rows] != list(
        range(p["seed_start"], p["seed_start"] + p["n_cohorts"])
    ):
        raise ValueError("missing, reordered or repeated saved pairs")
    T = compiled.spec.horizon.n_periods
    float32_error = 0.0
    for row in rows:
        e = row["policy_episode"]
        if (
            len(e["observations"]) != T + 1
            or len(e["actions"]) != T
            or len(e["rewards"]) != T
            or len(row["behavior_logp"]) != T
            or e["dones"] != [0.0] * (T - 1) + [1.0]
            or not row["deterministic"]
            or len(row["physical_tape_sha256"]) != 64
        ):
            raise ValueError("incomplete saved native episode")
        for values in [
            e["observations"],
            e["actions"],
            e["rewards"],
            row["behavior_logp"],
        ]:
            if not np.isfinite(values).all():
                raise ValueError("nonfinite saved native episode")
        if row["difference_J"] != row["ppo_J"] - row["fixed_J"]:
            raise ValueError("saved paired arithmetic mismatch")
        for name in ["ppo", "fixed"]:
            if abs(row[name + "_J"] - row[name + "_accounting"]["objective"]) > 1e-12:
                raise ValueError("native reward/accounting identity mismatch")
        float32_error = max(float32_error, abs(sum(e["rewards"]) - row["ppo_J"]))
    # Native Episode stores float32 rewards; primary J/accounting remains float64.
    if float32_error > 2e-6:
        raise ValueError("saved float32 episode rewards differ from native totals")
    summary = paired_summary(rows)
    if any(old[k] != v for k, v in summary.items()):
        raise ValueError("saved summary differs from all1280raw paired rows")
    return rows, old, summary, failed, float32_error


def run(cell, context):
    """Return the worker's required status while leaving every original file intact."""
    p = cell["parameters"]
    rows, old, summary, failed, rounding = read_pairs(p)
    report = {
        "status": "completed",
        "schema": "airline-post-report-failure-pair-audit/1",
        "source_worker_status": "failed",
        "source_worker_value": None,
        "source_receipt_scope": "Original FAILED receipt seals failed results only. Source pairs and summary are post-failure SHA snapshots; this new seal covers derived audit, not retroactive original success.",
        "source_directory": p["source_directory"],
        "source_hashes": p["source_hashes"],
        "ordered_physical_cohorts": len(rows),
        "profile_measurements": 2 * len(rows),
        "new_physical_cohorts": 0,
        "new_actor_forwards": 0,
        "new_optimizer_updates": 0,
        "spec_hash": old["spec_hash"],
        "fixed_theta": old["fixed_theta"],
        "checkpoint_sha256": old["checkpoint_sha256"],
        "baseline_sha256": old["baseline_sha256"],
        "inference_state_sha256": old["inference_state_sha256"],
        "source_inference_state_unchanged_assertion": True,
        "native_float32_reward_sum_max_error": rounding,
        **summary,
        "source_evaluation_wall_seconds": failed["wall_seconds"],
        "source_evaluation_peak_rss_bytes": failed["peak_rss_bytes"],
        "source_evaluation_cpu_seconds": None,
        "cost_scope": "Original failed-report evaluation envelope includes all1280complete pairs; CPU not retained. New audit worker timings are separate and are not policy inference/training costs.",
        "units": old["units"],
        "scope": old["scope"],
        "full_V7_pass": None,
        "scientific_RL_budget": None,
        "test_seeds_opened": False,
    }
    atomic_json(context.directory / "saved_pairs_audit.json", report)
    return {
        **report,
        "saved_pairs_audit_sha256": file_hash(
            context.directory / "saved_pairs_audit.json"
        ),
    }
