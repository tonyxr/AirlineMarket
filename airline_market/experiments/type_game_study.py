"""Durable exact-mixture game pilots; original asymmetric instances are preserved."""

from dataclasses import replace
from pathlib import Path

from ..campaign.artifacts import atomic_json, file_hash
from ..core.hashing import spec_hash
from ..game.type_belief import HistoryPolicy
from ..game.type_psro import exact_type_psro
from ..solvers.brute_history import binary_sale_micro
from ..spec import compile_market, dump_spec, load_spec
from ..spec.model import InformationArmSpec


def micro_spec(parameters):
    """Declare two real pricing controls and explicitly simultaneous timing."""
    s = binary_sale_micro(
        {
            "T": parameters.get("T", 2),
            "rival_support": [2],
            "own_capacity": 2,
            "arrival": 0.9,
        }
    )
    pricing = s.pricing[0]
    segments = s.segments
    if parameters.get("symmetric", False):
        segments = tuple(
            replace(
                g,
                classes=tuple(
                    replace(k, carrier_pref={"A": 0, "B": 0}) for k in g.classes
                ),
            )
            for g in segments
        )
    return replace(
        s,
        name="two_sided_episode_type_micro",
        segments=segments,
        timing=replace(s.timing, mode="simultaneous", leaders=()),
        pricing=(pricing, replace(pricing, carrier="B", base={"B_p": 100})),
        information=tuple(
            InformationArmSpec(name="own", observer=c, requests=()) for c in ["A", "B"]
        ),
        metadata={
            "family": "F7_two_sided_micro",
            "parameters": parameters,
            "declaration": "Both carriers select {-40,+40} intercepts. Simultaneous actions; own-only information; persistent hidden opponent policy type. Original asymmetric models unchanged.",
        },
    )


def run(cell, context):
    """Run actual exact-payoff PSRO and preserve every iteration, including limits."""
    p = cell["parameters"]
    if "source_spec" in p:
        path = Path(p["source_spec"])
        if file_hash(path) != p["source_spec_sha256"]:
            raise ValueError("registered two-sided source changed")
        spec = load_spec(path)
    else:
        spec = micro_spec(p)
    compiled = compile_market(spec)
    dump_spec(spec, context.directory / "spec.json")
    populations = {}
    for i, carrier in enumerate(spec.carriers):
        arm = next(a.name for a in spec.information if a.observer == carrier.code)
        populations[carrier.code] = [
            HistoryPolicy(
                spec.pricing[i].theta_grid,
                {},
                default=int(p.get("initial_action_indices", [0, 1])[i]),
                arm=arm,
            )
        ]
    events = []
    latest = {"stage": "initializing"}
    context.counters = lambda: latest.copy()

    def progress(event):
        """Publish exact work counters without changing registered configuration."""
        latest.update(event)
        events.append(event)
        atomic_json(context.directory / "events.json", events)

    result = exact_type_psro(
        compiled,
        populations,
        max_iters=int(p.get("max_iters", 12)),
        tol=float(p.get("tolerance", 1e-6)),
        max_nodes=int(cell["limits"]["max_nodes"]),
        progress=progress,
    )
    policies = {
        c: [policy.to_dict() for policy in pop]
        for c, pop in result.pop("populations").items()
    }
    atomic_json(context.directory / "policies.json", policies)
    atomic_json(context.directory / "iterations.json", result["history"])
    return {
        **result,
        "spec_hash": compiled.spec_hash,
        "policies_sha256": file_hash(context.directory / "policies.json"),
        "population_hashes": {
            c: [spec_hash(policy) for policy in pop] for c, pop in policies.items()
        },
        "is_original_I1": False,
        "scope_note": "New nondegenerate two-sided pricing variant or micro precursor. Exact deterministic finite-grid own-history responses, not continuous-price or unrestricted airline-market Nash; no MPC or empirical game claim. Unchanged I1/I3 reference specs have a zero-dimensional B action class and are not silently relabeled.",
    }
