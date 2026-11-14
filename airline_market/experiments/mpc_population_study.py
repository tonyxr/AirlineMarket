"""Bounded two-sided population study with preserved sampled MPC deviations."""

from pathlib import Path

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.hashing import canonical_json
from airline_market.game.mpc_population import sampled_mpc_psro
from airline_market.game.type_belief import HistoryPolicy
from airline_market.game.type_mpc import TypeMPCConfig
from airline_market.spec import compile_market, dump_spec, load_spec


def run(cell, context):
    p = cell["parameters"]
    source = Path(p["source_spec"])
    if file_hash(source) != p["source_spec_sha256"]:
        raise ValueError("changed frozen two-sided source")
    c = compile_market(load_spec(source))
    dump_spec(c.spec, context.directory / "spec.json")
    populations = {}
    for carrier in c.spec.carriers:
        code = carrier.code
        grid = next(q.theta_grid for q in c.spec.pricing if q.carrier == code)
        populations[code] = [
            HistoryPolicy(grid, {}, default=i, arm=p["arms"][code])
            for i in p["initial_action_indices"]
        ]
    latest = {"stage": "initializing"}
    context.counters = lambda: latest.copy()

    def progress(event):
        small = {
            k: v for k, v in event.items() if isinstance(v, (str, int, float, bool))
        }
        latest.update(small)
        # Append each completed payoff/cohort/node before later work can hit a limit.
        with (context.directory / "events.jsonl").open("a") as f:
            f.write(canonical_json(event) + "\n")
        if event["stage"] == "response_complete":
            atomic_json(
                context.directory
                / (
                    "response_"
                    + str(event["iteration"])
                    + "_"
                    + event["focal"]
                    + ".json"
                ),
                event,
            )

    result = sampled_mpc_psro(
        c,
        populations,
        TypeMPCConfig(**p["mpc"]),
        max_iters=p["max_iters"],
        tolerance=p["tolerance"],
        payoff_seed_start=p["payoff_seed_start"],
        n_payoff=p["n_payoff"],
        deviation_seed_start=p["deviation_seed_start"],
        n_deviation=p["n_deviation"],
        max_policy_nodes=p["max_policy_nodes"],
        max_support=p["max_support"],
        progress=progress,
    )
    policies = {
        code: [q.to_dict() for q in bank]
        for code, bank in result.pop("populations").items()
    }
    atomic_json(context.directory / "policies.json", policies)
    atomic_json(context.directory / "iterations.json", result["history"])
    search_status = result.pop("status")
    return {
        **result,
        "status": "completed",
        "search_status": search_status,
        "spec_hash": c.spec_hash,
        "policies_sha256": file_hash(context.directory / "policies.json"),
        "iterations_sha256": file_hash(context.directory / "iterations.json"),
        "test_seeds_opened": False,
        "is_original_I3": False,
    }
