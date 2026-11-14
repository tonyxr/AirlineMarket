"""S1-r1: exact arms and true-world policy diagnostics on frozen new sources."""

import gzip
import pickle
import time
from pathlib import Path

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.errors import KernelLimitError
from airline_market.experiments.screen_diagnostics import exact_diagnostics
from airline_market.solvers.lagged_dp import history_dp
from airline_market.spec import compile_market, dump_spec, load_spec


def run(cell, context):
    """Solve one registered arm; preserve optimal reachable policy records for S1 metrics."""
    p = cell["parameters"]
    source = Path(p["source_spec"])
    if file_hash(source) != p["source_spec_sha256"]:
        raise ValueError("S1 source changed")
    c = compile_market(load_spec(source))
    dump_spec(c.spec, context.directory / "spec.json")
    dp = history_dp(c, "A", p["arm"], max_nodes=cell["limits"]["max_nodes"])
    current = {"stage": "solving", "setting_id": p["setting_id"], "arm": p["arm"]}
    context.counters = lambda: {
        **current,
        "nodes": dp.finite.memo_entries(),
        "nodes_per_period": dict(dp.finite.counts),
    }
    started = time.perf_counter()
    try:
        value = dp.solve().value
        current.update(stage="policy_diagnostics")
        count = 0
        with gzip.open(context.directory / "optimal_policy.pkl.gz", "wb") as stream:

            def record(node, index):
                nonlocal count
                pickle.dump((node, int(index)), stream, protocol=4)
                count += 1

            diagnostics = exact_diagnostics(dp, policy_sink=record)
        atomic_json(context.directory / "diagnostics.json", diagnostics)
        outcome = {
            "status": "solved",
            "value": value,
            "exact": True,
            "diagnostics": diagnostics,
            "policy_records": count,
            "policy_path": "optimal_policy.pkl.gz",
            "first_policy_difference": "pending cross-arm comparison from saved optimal policy records",
        }
    except KernelLimitError as error:
        outcome = {
            "status": "node_limit",
            "value": None,
            "exact": False,
            "reason": str(error),
        }
    return {
        **outcome,
        "spec_hash": c.spec_hash,
        "nodes": dp.finite.memo_entries(),
        "solve_seconds": time.perf_counter() - started,
        "plan_id": "S1-r1",
        "setting_id": p["setting_id"],
        "arm": p["arm"],
        "plausibility_preflight": "All52 settings structurally fail both-carrier70pct LF; retained per explicit owner decision. Direct-policy diagnostics remain measured.",
        "scientific_budget": None,
    }
