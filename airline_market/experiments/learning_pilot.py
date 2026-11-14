"""F6 training-only fixed fits and registered 20k-cohort learning pilots.

Existing flat/structured trainers remain unchanged. Initial and final policies
are compared on paired validation cohorts; best-checkpoint contrasts are
descriptive because those cohorts also select the checkpoint.
"""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from ..campaign.artifacts import atomic_json, file_hash
from ..campaign.instances import build_instance
from ..core.rng import StreamRegistry
from ..core.types import CoefficientAction
from ..learn.adapters import RecurrentEnvAdapter
from ..learn.ppo import collect
from ..learn.structured_runtime import StructuredAdapter, StructuredPPOAgent
from ..learn.structured_train import collect_structured, seed_in_split, train_structured
from ..pomdp.env import FocalCarrierEnv
from ..sim.sparse_access import sparse_access
from ..solvers.fixed_search import episode_value
from ..spec import compile_market, dump_spec, load_spec
from ..vendor.ride_response.recurrent.ppo import RecurrentPPOAgent, RecurrentPPOConfig
from .f6_study import flat_train


def market(parameters):
    """Load the frozen hub input or the same full-grid toy used in F5/F6."""
    if "spec_file" in parameters:
        path = Path(parameters["spec_file"])
        if file_hash(path) != parameters["spec_sha256"]:
            raise ValueError("learning-pilot frozen specification changed")
        spec = load_spec(path)
    else:
        spec = build_instance("toy_hub", {"T": parameters.get("T", 10),
                                           "actions": parameters.get("actions", 18)})
    return compile_market(spec)


def validation_seeds(parameters):
    """Use only the declared validation interval, with at least two cohorts."""
    start, count = parameters.get("validation_seed_start", 200000), parameters.get("n_validation", 256)
    if count < 2:
        raise ValueError("pilot contrasts require at least two validation cohorts")
    seeds = list(range(start, start + count))
    for seed in seeds:
        seed_in_split(seed, False)
    return seeds


def fit_fixed(cell, context):
    """Reuse a matching F5 fit or fit the full coefficient grid on training only."""
    p = cell["parameters"]
    compiled = market(p)
    dump_spec(compiled.spec, context.directory / "spec.json")
    focal, arm = p.get("focal", "A"), p.get("arm", "direct")
    start, count = p.get("train_seed_start", 100000), p.get("n_fit", 64)
    if count < 1:
        raise ValueError("fixed fitting requires training cohorts")
    seed_in_split(start, True)
    seed_in_split(start + count - 1, True)
    seeds = validation_seeds(p)
    pricing = next(x for x in compiled.spec.pricing if x.carrier == focal)
    current = {"stage": "fixed_setup"}
    context.counters = lambda: dict(current)
    table, reused = [], None
    with sparse_access():
        if "reuse_tails" in p:
            path = Path(p["reuse_tails"])
            if file_hash(path) != p["reuse_tails_sha256"]:
                raise ValueError("reused fixed-fit artifact changed")
            source = json.loads(path.read_text())
            if (source["spec_hash"] != compiled.spec_hash or source["arm"] != arm
                    or source["focal"] != focal or source["training_seed_start"] != start
                    or source["training_cohorts"] != count):
                raise ValueError("fixed-fit model, arm or training range mismatch")
            table = source["fixed_training_table"]
            if [tuple(x["coefficients"]) for x in table] != list(pricing.theta_grid):
                raise ValueError("fixed-fit coefficient grid mismatch")
            best = tuple(source["fixed"])
            if best != tuple(max(table, key=lambda x: x["training_objective"])["coefficients"]):
                raise ValueError("reused fixed policy is not the saved grid maximizer")
            reused = {"path": str(path), "sha256": file_hash(path)}
        else:
            env = FocalCarrierEnv(compiled, focal, arm)
            for index, theta in enumerate(pricing.theta_grid):
                values = []
                for k in range(count):
                    current.update(stage="fixed_fit", candidate=index + 1,
                                   candidates=len(pricing.theta_grid), training_cohort=k + 1,
                                   training_cohorts=count)
                    values.append(episode_value(env, start + k,
                                                lambda obs: CoefficientAction(tuple(theta))))
                table.append({"coefficients": list(theta),
                              "training_objective": float(np.mean(values))})
            best = tuple(max(table, key=lambda x: x["training_objective"])["coefficients"])
        env = FocalCarrierEnv(compiled, focal, arm)
        values, revenue = [], []
        for k, seed in enumerate(seeds):
            current.update(stage="fixed_validation", cohort=k + 1, cohorts=len(seeds))
            values.append(episode_value(env, seed, lambda obs: CoefficientAction(best)))
            revenue.append(float(env.engine.world.carriers[env.i].revenue))
    result = {"schema": "airline-fixed-training-baseline/1", "status": "completed",
              "spec_hash": compiled.spec_hash, "focal": focal, "arm": arm,
              "fixed": list(best), "fixed_training_table": table,
              "training_seed_start": start, "training_cohorts": count,
              "reused_fit": reused, "validation_seeds": seeds,
              "validation_values": values, "validation_revenue": revenue,
              "selection": "Full declared discrete coefficient grid, selected on common training tapes only; not a continuous/global optimum or final P1 tuning."}
    atomic_json(context.directory / "baseline.json", result)
    return {"status": "completed", "spec_hash": compiled.spec_hash,
            "baseline_sha256": file_hash(context.directory / "baseline.json"),
            "fit_reused": reused is not None, "coefficients": list(best),
            "validation_mean": float(np.mean(values)), "n_validation": len(seeds)}


def initial_agent(compiled, cfg):
    """Reconstruct the native trainer's exact named-seed initialization contract."""
    focal, arm = cfg.get("focal", "A"), cfg.get("arm", "direct")
    env = FocalCarrierEnv(compiled, focal, arm)
    flat = cfg["learner"] == "ppo"
    adapter = RecurrentEnvAdapter(env) if flat else StructuredAdapter(env)
    identity = tuple(k for k, name in enumerate(adapter.names())
                     if (flat and name.startswith("inventory/"))
                     or (not flat and k >= adapter.layout.global_dim)
                     or name == "clock" or name.startswith("exo/")
                     or name.endswith(("/available", "/missing")))
    namespace = "train/flat/init" if flat else "train/structured/init"
    seed = StreamRegistry(cfg.get("network_seed", 0), namespace).child_seed() % 2**31
    hyper = RecurrentPPOConfig(gamma=cfg.get("ppo", {}).get("gamma", 1.0),
                               **{k: v for k, v in cfg.get("ppo", {}).items() if k != "gamma"},
                               observation_identity_indices=identity, seed=seed)
    with torch.random.fork_rng(devices=[]):
        agent = (RecurrentPPOAgent(len(adapter.names()), adapter.action_transform, hyper)
                 if flat else StructuredPPOAgent(adapter, hyper, auxiliary=cfg["learner"] == "srl_ca",
                                                 aux_weight=cfg.get("aux_weight", 0.3)))
    if flat:
        adapter.configure_actor(agent)
    return adapter, agent


def untrained_values(compiled, cfg, seeds):
    """Evaluate initialized policies without labels, credit branches or parameter updates."""
    adapter, agent = initial_agent(compiled, cfg)
    result = {}
    for deterministic, label in [(True, "deterministic"), (False, "sampled")]:
        values = []
        for seed in seeds:
            if cfg["learner"] == "ppo":
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(StreamRegistry(seed, "validation/flat/policy").child_seed() % 2**31)
                    values.append(collect(adapter, agent, seed, deterministic=deterministic)[2])
            else:
                values.append(collect_structured(adapter, agent, seed, training=False,
                                                 deterministic=deterministic)[2])
        result[label + "_values"] = values
        result[label] = float(np.mean(values))
    return result


def paired_summary(values, baseline, seed=0):
    """Report a paired cohort bootstrap for a predeclared fixed-time pilot contrast."""
    values, baseline = np.asarray(values), np.asarray(baseline)
    if values.shape != baseline.shape or values.ndim != 1 or len(values) < 2:
        raise ValueError("paired pilot contrast needs matching cohort vectors")
    difference = values - baseline
    rng = StreamRegistry(seed, "boot/f6/pilot").generator()
    samples = rng.integers(0, len(difference), (2000, len(difference)))
    bootstrap = difference[samples].mean(1)
    return {"mean": float(difference.mean()),
            "se": float(difference.std(ddof=1) / np.sqrt(len(difference))),
            "bootstrap95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
            "n": len(difference), "bootstrap_replicates": 2000}


def pilot(cell, context):
    """Run the fixed 20k budget and preserve favorable and unfavorable learning curves."""
    p = cell["parameters"]
    compiled = market(p)
    dump_spec(compiled.spec, context.directory / "spec.json")
    path = Path(p["baseline_file"])
    if file_hash(path) != p["baseline_sha256"]:
        raise ValueError("pilot frozen baseline changed")
    baseline = json.loads(path.read_text())
    seeds = validation_seeds(p)
    if (baseline["spec_hash"] != compiled.spec_hash or baseline["validation_seeds"] != seeds
            or baseline["arm"] != p.get("arm", "direct")
            or baseline["focal"] != p.get("focal", "A")):
        raise ValueError("pilot baseline model/arm/cohort mismatch")
    cfg = {k: p[k] for k in ("learner", "n_train", "cohorts_per_update", "event_legs",
                             "network_seed", "n_validation", "ppo", "credit", "aux_weight",
                             "validate_every_cohorts", "arm", "focal", "validation_seed_start")
           if k in p}
    current = {"stage": "untrained_validation"}
    context.counters = lambda: dict(current)

    def progress(row):
        """Publish the native learner's actual consumed-cohort counters."""
        current.update(row)

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with sparse_access():
        initial = untrained_values(compiled, cfg, seeds)
        atomic_json(context.directory / "initial_validation.json", initial)
        trained = (flat_train(compiled, cfg, context.directory, progress)
                   if cfg["learner"] == "ppo"
                   else train_structured(compiled, cfg, context.directory, progress))
    learning = json.loads((context.directory / "learning.json").read_text())
    curve = []
    for row in learning["validation"]:
        curve.append({"cohorts": row["cohorts"], "deterministic": row["deterministic"],
                      "sampled": row["sampled"],
                      "deterministic_minus_fixed": paired_summary(
                          row["deterministic_values"], baseline["validation_values"]),
                      "deterministic_minus_initial": paired_summary(
                          row["deterministic_values"], initial["deterministic_values"])})
    final = curve[-1]
    # Include explicit feature/action metadata in the completed pilot checkpoints.
    # The native, currently running trainers are not changed or patched.
    adapter, _ = initial_agent(compiled, cfg)
    parameter_counts = None
    for name in ("best.pt", "last.pt"):
        checkpoint = context.directory / name
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        payload["pilot_contract"] = {"observation_names": list(adapter.names()),
                                     "action_transform": asdict(adapter.action_transform),
                                     "initialization": "Named train initialization and per-cohort policy streams; optimizer shuffle state in checkpoint.",
                                     "baseline_sha256": p["baseline_sha256"],
                                     "final_test_seeds_opened": False}
        parameter_counts = {key: sum(x.numel() for name, x in payload[key].items()
                                     if not name.endswith("input_mask"))
                            for key in ("actor", "value")}
        temporary = checkpoint.with_suffix(".pilot.tmp.pt")
        torch.save(payload, temporary)
        temporary.replace(checkpoint)
    report = {"schema": "airline-learning-pilot/1", "learner": cfg["learner"],
              "instance": p["instance"], "training_cohorts": cfg["n_train"],
              "network_seed": cfg.get("network_seed", 0), "validation_seeds": seeds,
              "initial": initial, "curve": curve, "final": final,
              "parameter_counts": parameter_counts,
              "interpretation": "One training seed, validation pilot only. Fixed-time final contrasts are paired; best-checkpoint contrasts are descriptive selection data. No final-method or multi-seed performance claim.",
              "learning_signal": "Descriptive initial-to-final gain; no invented pass criterion or hyperparameter selection.",
              "test_seeds_opened": False}
    atomic_json(context.directory / "pilot.json", report)
    return {**trained, "instance": p["instance"], "spec_hash": compiled.spec_hash,
            "baseline_sha256": p["baseline_sha256"], "final_minus_fixed": final["deterministic_minus_fixed"],
            "final_minus_initial": final["deterministic_minus_initial"],
            "pilot_sha256": file_hash(context.directory / "pilot.json"),
            "parameter_counts": parameter_counts, "final_budget_selected": False,
            "learning_signal": "Single-seed pilot contrasts recorded; no final-method or multi-seed claim."}
