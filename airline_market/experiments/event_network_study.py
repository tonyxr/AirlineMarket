"""A.15/V10: new dated on-time compiler; saved candidate scope, no DB1C replay."""

import json
from pathlib import Path

from airline_market.pipeline.event_network import compile_service

from airline_market.campaign.artifacts import atomic_json, file_hash


def run(cell, context):
    p = cell["parameters"]
    current = {"stage": "source_identity"}
    context.counters = lambda: dict(current)
    pinned = [
        {"path": p["candidate_scope"], "sha256": p["candidate_scope_sha256"]},
        *p["marketing_sources"],
        {"path": p["timezone_config"], "sha256": p["timezone_config_sha256"]},
    ]
    for source in pinned:
        if file_hash(source["path"]) != source["sha256"]:
            raise ValueError("changed pinned event input: " + source["path"])
    scope = json.loads(Path(p["candidate_scope"]).read_bytes())
    zones = json.loads(Path(p["timezone_config"]).read_bytes())["airport_timezones"]
    summary = compile_service(
        p["marketing_sources"],
        scope,
        zones,
        context.directory,
        lambda **x: current.update(x),
    )
    for source in pinned:
        if file_hash(source["path"]) != source["sha256"]:
            raise ValueError("event input changed during compile")
    summary.update(
        input_hash_scope=p["candidate_scope_hash_scope"],
        simulated_physical_cohorts=0,
        fits=0,
        raw_rng_call_count=None,
    )
    atomic_json(context.directory / "network_summary.json", summary)
    files = [
        "source_rows.parquet",
        "daily_marketing.parquet",
        "physical_resources.parquet",
        "nonstop_products.parquet",
        "service_index.sqlite",
        "network_summary.json",
    ]
    current["stage"] = "complete"
    return {
        **summary,
        "value": None,
        "artifact_hashes": {f: file_hash(context.directory / f) for f in files},
    }
