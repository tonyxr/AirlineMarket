"""S1/R4 bug fix: qualify reference-construction RNG separately from traversal."""

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.experiments.screen_metrics_study import (
    analyze_settled_setting,
    verify_inputs,
)


def run(cell, context):
    """Same exhaustive analysis, corrected RNG provenance; no source reruns."""
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
    qualifier = {
        "new_rng_calls": None,
        "reference_constructor_generates_tapes": True,
        "reference_construction_seed": 0,
        "rng_scope": "history_dp constructs ExactKernel base tapes; native finite traversal enumerates probabilities and does not sample physical cohorts. Raw RNG-call count unmeasured.",
        "traversal_sampling_calls": 0,
        "sampled_physical_cohorts": 0,
        "bellman_optimizations": 0,
        "actor_calls": 0,
    }
    metrics.update(source_manifest_scope=p["policy_hash_scope"], **qualifier)
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
        "native_transition_enumeration": True,
        "scientific_budget": None,
        "policy_hash_scope": p["policy_hash_scope"],
        **qualifier,
    }
