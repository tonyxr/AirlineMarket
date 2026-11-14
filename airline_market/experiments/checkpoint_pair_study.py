"""Fresh paired validation of a frozen final flat-PPO checkpoint and fixed fit.

This performs inference only. The comparator is the saved training-grid winner,
not a continuous/global fixed optimum or a newly selected P1 baseline.
"""

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.hashing import canonical_json, schema_hash
from airline_market.core.rng import StreamRegistry
from airline_market.core.types import CoefficientAction
from airline_market.learn.adapters import RecurrentEnvAdapter
from airline_market.learn.ppo import collect, load_checkpoint
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.sim.accounting import carrier_metrics
from airline_market.sim.sparse_access import sparse_access
from airline_market.solvers.fixed_search import episode_value
from airline_market.spec import compile_market, dump_spec, load_spec


def validation_seeds(start, count):
    """Reject wrong/reserved splits before constructing any physical environment."""
    if not isinstance(start, int) or not isinstance(count, int) or count < 2:
        raise ValueError("paired validation requires at least two integer cohorts")
    if start < 200000 or start + count > 300000:
        raise ValueError("all paired cohorts must be in the validation interval")
    return list(range(start, start + count))


def check_inputs(parameters):
    """Verify prospective source bytes and the actual scope of old result seals."""
    for role in (
        "source_spec",
        "checkpoint",
        "checkpoint_manifest",
        "parent_results",
        "parent_receipt",
        "baseline_file",
        "baseline_results",
        "baseline_receipt",
    ):
        if parameters[role] not in parameters["input_hashes"]:
            raise ValueError("missing prospective input hash: " + role)
    for path, expected in parameters["input_hashes"].items():
        if file_hash(Path(path)) != expected:
            raise ValueError("frozen paired input changed: " + path)
    for prefix in ("parent", "baseline"):
        receipt = json.loads(Path(parameters[prefix + "_receipt"]).read_text())
        if receipt["status"] != "completed" or receipt["results_sha256"] != file_hash(
            Path(parameters[prefix + "_results"])
        ):
            raise ValueError("invalid source result receipt: " + prefix)


def load_sources(p, seeds):
    """Load saved physical bytes, training-grid fit and last checkpoint contract."""
    check_inputs(p)
    c = compile_market(load_spec(Path(p["source_spec"])))
    parent = json.loads(Path(p["parent_results"]).read_text())
    baseline = json.loads(Path(p["baseline_file"]).read_text())
    if any(
        x != c.spec_hash
        for x in (p["compiled_spec_hash"], parent["spec_hash"], baseline["spec_hash"])
    ):
        raise ValueError("paired compiled physical specification mismatch")
    if (
        parent["learner"] != "ppo"
        or parent["policy_class"] != "coefficient_recurrent"
        or parent["parameters"]["focal"] != p["focal"]
        or parent["parameters"]["arm"] != p["arm"]
        or baseline["focal"] != p["focal"]
        or baseline["arm"] != p["arm"]
        or parent["baseline_sha256"] != file_hash(Path(p["baseline_file"]))
    ):
        raise ValueError("paired learner/focal/arm/baseline mismatch")
    manifest = json.loads(Path(p["checkpoint_manifest"]).read_text())
    entries = [
        x
        for x in manifest["checkpoints"]
        if Path(x["path"]).resolve() == Path(p["checkpoint"]).resolve()
    ]
    if (
        Path(p["checkpoint"]).name != "last.pt"
        or len(entries) != 1
        or entries[0]["sha256"] != file_hash(Path(p["checkpoint"]))
    ):
        raise ValueError("requires the saved final last checkpoint, not selection")
    payload = torch.load(p["checkpoint"], map_location="cpu", weights_only=False)
    meta = payload["metadata"]
    if (
        meta["learner"] != "ppo"
        or meta["arm"] != p["arm"]
        or meta["consumed"] != parent["cohorts"]
        or payload["updates"] != parent["updates"]
        or payload["pilot_contract"]["baseline_sha256"] != parent["baseline_sha256"]
    ):
        raise ValueError("checkpoint differs from its final source training contract")
    old = set(baseline["validation_seeds"])
    old.update(meta["validation"][0].get("seeds", []))
    q = parent["parameters"]
    old.update(
        range(
            q["validation_seed_start"], q["validation_seed_start"] + q["n_validation"]
        )
    )
    if old.intersection(seeds):
        raise ValueError("fresh paired validation overlaps the old evaluation")
    pricing = next(x for x in c.spec.pricing if x.carrier == p["focal"])
    table = baseline["fixed_training_table"]
    if [tuple(x["coefficients"]) for x in table] != list(pricing.theta_grid) or tuple(
        baseline["fixed"]
    ) != tuple(max(table, key=lambda x: x["training_objective"])["coefficients"]):
        raise ValueError("fixed comparator is not the saved full training-grid winner")
    return c, baseline, payload


def inference_fingerprint(agent):
    """Track parameters, optimizers, normalization, update count and private RNG."""
    return schema_hash(
        {
            "actor": agent.actor.state_dict(),
            "value": agent.value.state_dict(),
            "actor_optimizer": agent.actor_optimizer.state_dict(),
            "value_optimizer": agent.value_optimizer.state_dict(),
            "normalizer": agent.observation_normalizer.state_dict(),
            "updates": agent.updates,
            "rng": agent.rng.bit_generator.state,
        }
    )


def one_pair(adapter, agent, fixed_env, theta, seed):
    """Use unchanged native inference and fixed execution on identical exogenous tape."""
    started = time.perf_counter()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(
            StreamRegistry(seed, "validation/flat/policy").child_seed() % 2**31
        )
        episode, logp, value = collect(adapter, agent, seed, deterministic=True)
    policy_seconds = time.perf_counter() - started
    started = time.perf_counter()
    fixed = episode_value(fixed_env, seed, lambda obs: CoefficientAction(tuple(theta)))
    fixed_seconds = time.perf_counter() - started
    tape = schema_hash(adapter.env.engine.tape)
    if tape != schema_hash(fixed_env.engine.tape):
        raise AssertionError("paired physical tapes differ")
    if not np.isfinite([value, fixed, *logp]).all():
        raise ValueError("nonfinite native paired evaluation")
    policy_accounting = carrier_metrics(
        adapter.env.engine.world, adapter.env.compiled, adapter.env.i
    )
    fixed_accounting = carrier_metrics(
        fixed_env.engine.world, fixed_env.compiled, fixed_env.i
    )
    if (
        max(
            abs(value - policy_accounting["objective"]),
            abs(fixed - fixed_accounting["objective"]),
        )
        > 1e-12
    ):
        raise AssertionError(
            "summed native rewards differ from terminal objective accounting"
        )
    return {
        "seed": seed,
        "ppo_J": value,
        "fixed_J": fixed,
        "difference_J": value - fixed,
        "physical_tape_sha256": tape,
        "deterministic": True,
        "ppo_seconds": policy_seconds,
        "fixed_seconds": fixed_seconds,
        "policy_episode": asdict(episode),
        "behavior_logp": logp,
        "ppo_accounting": policy_accounting,
        "fixed_accounting": fixed_accounting,
    }


def paired_summary(rows, replicates=2000):
    """Bootstrap complete paired cohorts; preserve raw signs and objective units."""
    delta = np.asarray([x["difference_J"] for x in rows], dtype=float)
    if len(delta) < 2 or not np.isfinite(delta).all():
        raise ValueError("paired summary needs finite complete cohorts")
    rng = StreamRegistry(773, "boot/v7/frozen-ppo-pair").generator()
    means = np.empty(replicates)
    for i in range(replicates):
        means[i] = delta[rng.integers(len(delta), size=len(delta))].mean()
    return {
        "n": len(delta),
        "mean_difference_J": float(delta.mean()),
        "se_J": float(delta.std(ddof=1) / np.sqrt(len(delta))),
        "bootstrap95_J": np.quantile(means, [0.025, 0.975]).tolist(),
        "bootstrap_replicates": replicates,
        "checkpoint_meets_fixed_point_criterion": bool(delta.mean() >= 0),
        "ppo_mean_J": float(np.mean([x["ppo_J"] for x in rows])),
        "fixed_mean_J": float(np.mean([x["fixed_J"] for x in rows])),
        "wins": int((delta > 0).sum()),
        "ties": int((delta == 0).sum()),
        "losses": int((delta < 0).sum()),
    }


def run(cell, context):
    """Infer once on each new paired seed; no training, refit or checkpoint selection."""
    p = cell["parameters"]
    seeds = validation_seeds(p["seed_start"], p["n_cohorts"])
    c, baseline, payload = load_sources(p, seeds)
    dump_spec(c.spec, context.directory / "spec.json")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    adapter = RecurrentEnvAdapter(FocalCarrierEnv(c, p["focal"], p["arm"]))
    fixed_env = FocalCarrierEnv(c, p["focal"], p["arm"])
    with torch.random.fork_rng(devices=[]):
        agent = load_checkpoint(p["checkpoint"], adapter)
    before = inference_fingerprint(agent)
    rows, current = (
        [],
        {"stage": "paired_validation", "cohorts": len(seeds), "completed": 0},
    )
    context.counters = lambda: dict(current)
    path = context.directory / "pairs.jsonl"
    with path.open("x") as output, sparse_access():
        for k, seed in enumerate(seeds):
            current.update(seed=seed, completed=k)
            row = one_pair(adapter, agent, fixed_env, baseline["fixed"], seed)
            output.write(canonical_json(row) + "\n")
            output.flush()
            rows.append(row)
            current["completed"] = k + 1
    if before != inference_fingerprint(agent):
        raise AssertionError(
            "inference changed weights/optimizer/normalizer/private RNG"
        )
    check_inputs(p)
    summary = {
        "schema": "airline-frozen-checkpoint-pair/1",
        "spec_hash": c.spec_hash,
        "seed_start": seeds[0],
        "seed_end": seeds[-1],
        "cohorts": len(seeds),
        "checkpoint_sha256": file_hash(Path(p["checkpoint"])),
        "baseline_sha256": file_hash(Path(p["baseline_file"])),
        "fixed_theta": baseline["fixed"],
        "training_cohorts": payload["metadata"]["consumed"],
        "updates": agent.updates,
        "inference_state_sha256": before,
        "inference_state_unchanged": True,
        "pairs_sha256": file_hash(path),
        **paired_summary(rows),
        "units": "Native normalized profit-plus-share objective J, not USD profit or revenue percent.",
        "scope": "One fixed-time last checkpoint/default/NN0 against saved training-grid winner; no tuning, single-leg five-seed learning, full V7, P1/P2 or scientific budget claim.",
        "full_V7_pass": None,
        "scientific_RL_budget": None,
        "test_seeds_opened": False,
    }
    atomic_json(context.directory / "paired_summary.json", summary)
    return {
        **summary,
        "paired_summary_sha256": file_hash(context.directory / "paired_summary.json"),
    }
