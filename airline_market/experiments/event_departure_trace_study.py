"""A.15/V10: new prepared-source discrepancy trace, no ingestion/solver replay."""

import json
from pathlib import Path

import pandas as pd

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.pipeline.event_departure_trace import run_trace


def run(cell, context):
    p = cell["parameters"]
    current = {"stage": "verify_inputs"}
    context.counters = lambda: dict(current)
    for x in p["pinned_files"]:
        if file_hash(x["path"]) != x["sha256"]:
            raise ValueError("changed departure-trace input: " + x["path"])
    current["stage"] = "source_group_trace"
    data = run_trace(
        json.loads(Path(p["comparison"]).read_bytes()),
        p["prepared_index"],
        p["T100_projection"],
    )
    atomic_json(context.directory / "departure_trace.json", data)
    pd.DataFrame(data["trace_rows"]).to_csv(
        context.directory / "departure_trace.csv", index=False
    )
    current["stage"] = "complete"
    return {
        **{
            k: v
            for k, v in data.items()
            if k not in ["trace_rows", "source_witnesses", "T100_witnesses"]
        },
        "value": None,
        "simulated_physical_cohorts": 0,
        "fits": 0,
        "raw_rng_call_count": None,
        "artifact_hashes": {
            n: file_hash(context.directory / n)
            for n in ["departure_trace.json", "departure_trace.csv"]
        },
    }
