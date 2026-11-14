"""Bounded native persistent-type MPC resource precursor, not PSRO convergence."""

import time
from pathlib import Path

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.game.payoffs import MixedController, PolicyRef, controller
from airline_market.game.type_mpc import EpisodeTypeMPCController, TypeMPCConfig
from airline_market.sim.accounting import carrier_metrics
from airline_market.sim.engine import MarketEngine
from airline_market.spec import compile_market, dump_spec, load_spec


def probe(cell, context):
    """Measure actual full native cohorts and retain every completed decision."""
    p = cell["parameters"]
    path = Path(p["source_spec"])
    if file_hash(path) != p["source_spec_sha256"]:
        raise ValueError("changed registered two-sided physical source")
    c = compile_market(load_spec(path))
    if len(c.spec.carriers) != 2 or any(
        not q.controllable_features or len(q.theta_grid or ()) < 2
        for q in c.spec.pricing
    ):
        raise ValueError("probe requires two nondegenerate coefficient action classes")
    dump_spec(c.spec, context.directory / "spec.json")
    focal = p["focal"]
    rival = next(x.code for x in c.spec.carriers if x.code != focal)
    seeds = list(range(p["seed_start"], p["seed_start"] + p["n_cohorts"]))
    if not seeds or any(not 200000 <= seed < 300000 for seed in seeds):
        raise ValueError(
            "probe uses validation cohorts only; final seeds remain closed"
        )
    refs = [
        PolicyRef(
            "controller",
            {"controller": "schedule", "params": {"thetas_by_period": schedule}},
            p["rival_arm"],
        )
        for schedule in p["opponent_schedules"]
    ]
    probabilities = p["type_probabilities"]
    types = []
    for ref in refs:
        child = controller(ref, rival, 0)
        child.arm_name = p["rival_arm"]
        types.append({rival: child})
    cfg = TypeMPCConfig(**p["mpc"])
    rows = []
    latest = {"stage": "initializing", "cohorts": len(seeds)}
    context.counters = lambda: latest.copy()
    for k, seed in enumerate(seeds):
        response = EpisodeTypeMPCController(
            focal,
            p["arm"],
            types,
            probabilities,
            cfg,
            episode_index=k,
            max_support=p.get("max_support", 20000),
        )
        mixture = MixedController(
            {"policies": refs, "probabilities": probabilities}, rival
        )
        engine = MarketEngine(c, {focal: response, rival: mixture})
        started = time.perf_counter()
        engine.reset(seed)
        try:
            while not engine.terminal():
                latest.update(
                    stage="native_joint_type_mpc",
                    cohort=k + 1,
                    period=engine.world.t,
                    completed_decisions=len(response.decisions),
                )
                engine.step()
                atomic_json(context.directory / "decisions.json", response.decisions)
        finally:
            atomic_json(context.directory / "decisions.json", response.decisions)
        rows.append(
            {
                "seed": seed,
                "seconds": time.perf_counter() - started,
                "realized_private_type": mixture.choice,
                "decisions": response.decisions,
                "metrics": carrier_metrics(engine.world, c, c.carrier_index[focal]),
            }
        )
        atomic_json(context.directory / "cohorts.json", rows)
    return {
        "status": "completed",
        "spec_hash": c.spec_hash,
        "focal": focal,
        "n_cohorts": len(rows),
        "decisions": sum(len(x["decisions"]) for x in rows),
        "cohort_seconds": [x["seconds"] for x in rows],
        "cohorts_sha256": file_hash(context.directory / "cohorts.json"),
        "type_probabilities": probabilities,
        "mpc": p["mpc"],
        "PSRO_iterations": None,
        "certified": False,
        "final_game_budget": None,
        "scope": "Native persistent-type one-step/fixed-tail MPC resource precursor on an explicitly two-sided variant. Whole cohort includes reset/exact filtering/planning/native steps/durable decisions. Final reporting/artifact work separate. No terminal filter update, tuned-tail optimum, PSRO result, exact deviation/Nash certificate, original-I3 or empirical claim.",
    }
