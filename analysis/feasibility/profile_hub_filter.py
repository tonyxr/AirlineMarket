"""Profile unchanged N500 hub initialization and one conditional hypothesis.

This is a new profiling-only operation, not a rerun of a completed timing cell.
No runtime sources, particle law, proposal budget or scientific criterion change.
"""

import cProfile
import json
import os
import pstats
import sys
import threading
import time
import tracemalloc
from pathlib import Path

import numpy as np
import psutil

from airline_market.belief.template_particle import TemplateParticleBelief
from airline_market.campaign.artifacts import atomic_json, file_hash, utc_now
from airline_market.core.rng import StreamRegistry
from airline_market.core.types import CoefficientAction
from airline_market.experiments.template_particle_study import fingerprint
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.spec import compile_market, load_spec


def graph_size(root, seen=None):
    """Count reachable concrete Python objects once, excluding globals/code refs.

    NumPy-owned array buffers are included by getsizeof. Shared views do not
    double-count base buffers. Native allocator/RNG internals are unmeasured.
    """
    seen = set() if seen is None else seen
    stack = [root]
    size = 0
    while stack:
        obj = stack.pop()
        key = id(obj)
        if key in seen:
            continue
        seen.add(key)
        size += sys.getsizeof(obj)
        if isinstance(obj, dict):
            stack.extend(obj.keys())
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        elif isinstance(obj, np.ndarray):
            if obj.base is not None:
                stack.append(obj.base)
        elif obj.__class__.__module__.startswith("airline_market") and hasattr(
            obj, "__dict__"
        ):
            stack.append(vars(obj))
    return size


def top_profile(profiler, path):
    """Keep calls, exclusive and cumulative seconds; never sum overlapping totals."""
    stats = pstats.Stats(profiler)
    rows = [
        {
            "file": k[0],
            "line": k[1],
            "function": k[2],
            "primitive_calls": v[0],
            "calls": v[1],
            "exclusive_seconds": v[2],
            "cumulative_seconds": v[3],
        }
        for k, v in stats.stats.items()
    ]
    atomic_json(
        path,
        {
            "total_profile_seconds": stats.total_tt,
            "exclusive_top": sorted(
                rows, key=lambda r: r["exclusive_seconds"], reverse=True
            )[:40],
            "cumulative_top": sorted(
                rows, key=lambda r: r["cumulative_seconds"], reverse=True
            )[:40],
        },
    )


def profile(config):
    """Use model-generated hypotheses and permitted before/after observations only."""
    out = Path(config["out"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETE.json").exists():
        raise ValueError("completed matching profiling operation exists")
    atomic_json(out / "config.json", config)
    started = time.perf_counter()
    process = psutil.Process()
    current = {
        "status": "running",
        "pid": os.getpid(),
        "created_at": process.create_time(),
        "stage": "setup",
    }
    peak = [process.memory_info().rss]
    stop = threading.Event()

    def monitor():
        while not stop.wait(1):
            rss = process.memory_info().rss
            peak[0] = max(peak[0], rss)
            atomic_json(
                out / "status.json",
                {
                    **current,
                    "updated_at": utc_now(),
                    "elapsed_seconds": time.perf_counter() - started,
                    "rss_bytes": rss,
                },
            )
            reason = (
                "memory_limit"
                if rss > config["rss_bytes_limit"]
                else "time_limit"
                if time.perf_counter() - started > config["seconds_limit"]
                else None
            )
            if reason:
                atomic_json(
                    out / "limit.json",
                    {**current, "status": reason, "peak_rss_bytes": peak[0]},
                )
                os._exit(75)

    threading.Thread(target=monitor, daemon=True).start()
    path = Path(config["source_spec"])
    if file_hash(path) != config["source_spec_sha256"]:
        raise ValueError("changed profile physical input")
    c = compile_market(load_spec(path))
    env = FocalCarrierEnv(c, "A", "direct")
    before, _ = env.reset(config["validation_seed"])
    belief = TemplateParticleBelief(
        c,
        "A",
        "direct",
        n_particles=500,
        seed=StreamRegistry(773, "plan/F4_hub/filter").child_seed(0),
        inner_draws=128,
        strict=True,
    )
    tracemalloc.start(8)
    prof = cProfile.Profile()
    current["stage"] = "profiled_prior_reset"
    prof.enable()
    belief.reset(before)
    prof.disable()
    reset_snapshot = tracemalloc.take_snapshot()
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    prof.dump_stats(str(out / "prior.prof"))
    top_profile(prof, out / "prior_profile.json")
    actual = fingerprint(belief)
    if actual != config["expected_prior_fingerprint"]:
        raise AssertionError(
            "profiled initialization differs from sealed scientific fingerprint"
        )
    # Archive ordered capacity coordinates/weights/RNG for independent initial-law checks.
    rival_legs = [
        leg
        for i in range(len(c.spec.carriers))
        if i != belief.focal
        for leg in c.legs_of[i]
    ]
    capacity = np.array(
        [[v for r in h.rivals for v in r.capacity] for h in belief.particles],
        dtype=np.int64,
    )
    np.savez(out / "initial_cloud.npz", capacities=capacity, weights=belief.weights)
    atomic_json(out / "initial_rng.json", belief.rng.bit_generator.state)
    hidden = list(belief.mass)
    quotes = [h.kernel.initial_quotes for h in hidden]
    known = [h.kernel.carriers[belief.focal] for h in hidden]
    # TrueQuote.seats may differ even when the observed flag signal is constant.
    groups = {}
    seen = set()
    for name, root in [
        ("compiled", c),
        ("particles_and_mass", (belief.particles, belief.mass, belief.weights)),
        ("kernel", belief.kernel),
        ("prior_template", belief._prior_template),
        ("history_and_observation", (belief.history, belief.last_observation)),
    ]:
        groups[name] = graph_size(root, seen)
    memory = {
        "ordered_incremental_reachable_bytes": groups,
        "ordered_union_bytes": sum(groups.values()),
        "cloud_reachable_bytes": graph_size(
            (belief.particles, belief.mass, belief.weights)
        ),
        "initial_quotes_isolated_union_bytes": graph_size(quotes),
        "initial_quotes_unique_identity_count": len({id(q) for q in quotes}),
        "initial_quotes_unique_value_count": len(set(quotes)),
        "known_carrier_isolated_union_bytes": graph_size(known),
        "known_carrier_unique_identity_count": len({id(q) for q in known}),
        "known_carrier_unique_value_count": len(set(known)),
        "tracemalloc_current_bytes": traced_current,
        "tracemalloc_peak_bytes": traced_peak,
        "allocation_top": [
            {"traceback": str(s.traceback), "size_bytes": s.size, "count": s.count}
            for s in reset_snapshot.statistics("lineno")[:35]
        ],
        "scope": "Reachable Python-object unions and traced allocations are distinct from RSS. Root incremental attribution is order-dependent; isolated field unions overlap and must not be added. No100k extrapolation or actual memory saving is claimed.",
    }
    atomic_json(out / "memory.json", memory)
    atomic_json(
        out / "prior.json",
        {
            "spec_hash": c.spec_hash,
            "fingerprint": actual,
            "matches_sealed": True,
            "support": len(belief.mass),
            "particle_count": 500,
            "rival_leg_ids": [c.spec.legs[leg].id for leg in rival_legs],
            "kernel_engine_calls": belief.kernel.engine_calls,
            "reset_is_profiled_not_cold_timing": True,
        },
    )
    action = CoefficientAction(c.spec.pricing[0].theta_grid[0])
    after, _, _, _ = env.step(action)
    current["stage"] = "one_hypothesis_128_draw_proposal"
    calls_before = belief.kernel.engine_calls
    prof = cProfile.Profile()
    prof.enable()
    proposals = belief.conditional_successors(
        belief.particles[0], before, action, after
    )
    prof.disable()
    prof.dump_stats(str(out / "proposal.prof"))
    top_profile(prof, out / "proposal_profile.json")
    atomic_json(
        out / "proposal.json",
        {
            "hypotheses_profiled": 1,
            "inner_draws": 128,
            "kernel_engine_calls": belief.kernel.engine_calls - calls_before,
            "retained_successor_states": len(proposals),
            "estimated_joint_observation_mass": sum(proposals.values()),
            "exact_proposals_enabled": belief.exact_proposals,
            "whole_particle_update": False,
            "update_throughput": None,
            "scope": "One existing hidden hypothesis and unchanged128-draw conditional simulator. Empty mass is retained; no weight recovery or full update/accuracy claim. Profiling times include overhead.",
        },
    )
    current.update(status="completed", stage="finished")
    stop.set()
    result = {
        **current,
        "updated_at": utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "peak_observed_rss_bytes": peak[0],
        "source_spec_sha256": file_hash(path),
        "prior_scientific_identity": True,
        "update_throughput": None,
        "scope": "Profiling-only resource diagnostic, not cold timing, full update, posterior accuracy, full100k reference or N selection.",
    }
    atomic_json(out / "results.json", result)
    atomic_json(out / "status.json", result)
    atomic_json(
        out / "COMPLETE.json", {"results_sha256": file_hash(out / "results.json")}
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    profile(json.loads(Path(sys.argv[1]).read_text()))
