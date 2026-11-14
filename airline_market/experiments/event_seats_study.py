"""A.15/V10: new prepared-service/T100 comparison; no prior study replay."""

import json
from pathlib import Path

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.pipeline.event_seats import run_reconciliation


def run(cell, context):
    p = cell["parameters"]
    current = {"stage": "input_identity"}
    context.counters = lambda: dict(current)
    for x in p["pinned_files"]:
        if file_hash(x["path"]) != x["sha256"]:
            raise ValueError("changed reconciliation input: " + x["path"])
    scope = json.loads(Path(p["candidate_scope"]).read_bytes())
    result = run_reconciliation(
        p["T100_archive"],
        p["source_parquet"],
        scope,
        set(p["coverage_months"]),
        context.directory,
        lambda **x: current.update(x),
    )
    atomic_json(context.directory / "reconciliation.json", result)
    import pandas as pd

    pd.DataFrame(result["operator_route_months"]).to_csv(
        context.directory / "operator_route_months.csv", index=False
    )
    pd.DataFrame(result["T100_strata"]).to_csv(
        context.directory / "T100_all_classes.csv", index=False
    )
    current["stage"] = "complete"
    files = [
        "reconciliation.json",
        "operator_route_months.csv",
        "T100_all_classes.csv",
        "t100_scoped_rows.parquet",
        "reported_counts.sqlite",
    ]
    return {
        **{
            k: v
            for k, v in result.items()
            if k not in ["T100_strata", "operator_route_months"]
        },
        "value": None,
        "simulated_physical_cohorts": 0,
        "fits": 0,
        "raw_rng_call_count": None,
        "artifact_hashes": {f: file_hash(context.directory / f) for f in files},
    }
