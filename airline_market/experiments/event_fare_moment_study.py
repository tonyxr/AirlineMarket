"""A.11/A.15/V10: all prepared Market rows, new fare/passenger moment bank."""

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.pipeline.event_fare_moments import aggregate


def run(cell, context):
    p = cell["parameters"]
    current = {"stage": "input_identity"}
    context.counters = lambda: dict(current)
    for x in p["pinned_files"]:
        if file_hash(x["path"]) != x["sha256"]:
            raise ValueError("changed moment input: " + x["path"])
    result = aggregate(
        p["sources"], context.directory, lambda **x: current.update(x), p["batch_rows"]
    )
    atomic_json(context.directory / "fare_moment_summary.json", result)
    current["stage"] = "complete"
    return {
        **{k: v for k, v in result.items() if k not in ["months"]},
        "value": None,
        "simulated_physical_cohorts": 0,
        "fits": 0,
        "raw_rng_call_count": None,
        "artifact_hashes": {
            n: file_hash(context.directory / n)
            for n in [
                "fare_moment_summary.json",
                "event_fare_moments.parquet",
                "moment_groups.sqlite",
            ]
        },
    }
