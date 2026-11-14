"""Isolated F6 throughput and learning pilots with actual structured policy classes."""

from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import torch

from ..campaign.artifacts import atomic_json, file_hash
from ..campaign.instances import build_instance
from ..core.rng import StreamRegistry
from ..learn.adapters import RecurrentEnvAdapter
from ..learn.ppo import collect, save_checkpoint
from ..learn.structured_train import train_structured, seed_in_split
from ..pipeline.scaling import scaling_network
from ..pomdp.env import FocalCarrierEnv
from ..sim.sparse_access import sparse_access
from ..spec import compile_market, dump_spec, load_spec
from ..vendor.ride_response.recurrent.ppo import RecurrentPPOAgent, RecurrentPPOConfig


def flat_train(compiled, cfg, directory, progress):
    """Use the existing flat recurrent PPO update with no final-test evaluation."""
    train = RecurrentEnvAdapter(FocalCarrierEnv(compiled, cfg.get("focal", "A"), cfg.get("arm", "direct")))
    validation = RecurrentEnvAdapter(FocalCarrierEnv(compiled, cfg.get("focal", "A"), cfg.get("arm", "direct")))
    identity = tuple(k for k, name in enumerate(train.names())
                     if name == "clock" or name.startswith(("inventory/", "exo/"))
                     or name.endswith(("/available", "/missing")))
    hyper = {"gamma": 1.0, **cfg.get("ppo", {}), "observation_identity_indices": identity,
             "seed": StreamRegistry(cfg.get("network_seed", 0), "train/flat/init").child_seed() % 2**31}
    with torch.random.fork_rng(devices=[]):
        agent = RecurrentPPOAgent(len(train.names()), train.action_transform, RecurrentPPOConfig(**hyper))
    train.configure_actor(agent)
    train_start, validation_start = cfg.get("train_seed_start", 100000), cfg.get("validation_seed_start", 200000)
    n_train, batch = cfg["n_train"], cfg["cohorts_per_update"]
    validation_seeds = range(validation_start, validation_start + cfg.get("n_validation", 64))
    for seed in [train_start, train_start + n_train - 1]:
        seed_in_split(seed, True)
    for seed in validation_seeds:
        seed_in_split(seed, False)
    history, validations, consumed, best = [], [], 0, None
    cadence = cfg.get("validate_every_cohorts", 2000)
    next_validation = cadence
    started = time.perf_counter()
    while consumed < n_train:
        rollout_started = time.perf_counter()
        returns = []
        for k in range(min(batch, n_train - consumed)):
            seed = train_start + consumed + k
            progress({"stage": "training", "cohorts_completed": consumed + k,
                      "cohorts_total": n_train, "update": agent.updates + 1})
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(StreamRegistry(seed, "train/flat/policy").child_seed() % 2**31)
                episode, logp, value = collect(train, agent, seed)
            agent.add_episode(episode, actions_are_physical=False, behavior_log_probabilities=logp)
            returns.append(value)
        rollout_seconds = time.perf_counter() - rollout_started
        update_started = time.perf_counter()
        stats = agent.update()
        update_seconds = time.perf_counter() - update_started
        consumed += len(returns)
        history.append({"cohorts": consumed, "return_mean": float(np.mean(returns)),
                        "rollout_seconds": rollout_seconds, "update_seconds": update_seconds, **stats})
        selected = False
        if consumed >= next_validation or consumed == n_train:
            row = {"cohorts": consumed}
            for deterministic, name in [(True, "deterministic"), (False, "sampled")]:
                values = []
                for seed in validation_seeds:
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(StreamRegistry(seed, "validation/flat/policy").child_seed() % 2**31)
                        values.append(collect(validation, agent, seed, deterministic=deterministic)[2])
                row[name] = float(np.mean(values))
                row[name + "_values"] = values
            validations.append(row)
            selected = best is None or row["deterministic"] > best
            best = row["deterministic"] if selected else best
            next_validation = (consumed // cadence + 1) * cadence
        metadata = {"learner": "ppo", "consumed": consumed, "history": history,
                    "validation": validations, "test_seeds_opened": False}
        save_checkpoint(agent, directory / "last.pt", train, metadata)
        if selected:
            save_checkpoint(agent, directory / "best.pt", train, metadata)
        atomic_json(directory / "learning.json", metadata)
    return {"status": "completed", "learner": "ppo", "policy_class": train.policy_class,
            "cohorts": consumed, "updates": agent.updates, "best_validation": best,
            "seconds": time.perf_counter() - started,
            "rollout_seconds": sum(x["rollout_seconds"] for x in history),
            "update_seconds": sum(x["update_seconds"] for x in history),
            "behavior_logp_checks": agent.behavior_log_probability_checks,
            "max_behavior_logp_error": agent.max_behavior_log_probability_error,
            "learning_signal": "not established by implementation/throughput runs",
            "test_seeds_opened": False}


def run(cell, context):
    """Measure cohort collection and optimizer work separately inside one owned child."""
    p = cell["parameters"]
    if "spec_file" in p:
        path = Path(p["spec_file"])
        if file_hash(path) != p["spec_sha256"]:
            raise ValueError("F6 frozen input specification changed")
        spec = load_spec(path)
    elif p.get("network") == "scaling":
        spec = scaling_network(p["network_params"])
    else:
        spec = build_instance("toy_hub", {"T": p.get("T", 10), "actions": 18})
    dump_spec(spec, context.directory / "spec.json")
    compiled = compile_market(spec)
    cfg = {k: p[k] for k in ("learner", "n_train", "cohorts_per_update", "event_legs",
                             "network_seed", "n_validation", "ppo", "credit", "aux_weight",
                             "validate_every_cohorts", "arm") if k in p}
    current = {"stage": "setup"}
    context.counters = lambda: dict(current)

    def progress(row):
        """Publish meaningful cell counters without changing frozen configs."""
        current.update(row)

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with sparse_access():
        result = (flat_train(compiled, cfg, context.directory, progress)
                  if cfg["learner"] == "ppo"
                  else train_structured(compiled, cfg, context.directory, progress))
    learning = __import__("json").loads((context.directory / "learning.json").read_text())
    rows = learning["history"]
    measured = rows[1:]  # First update is a declared warm-up, never a measured replicate.
    rollout = [x["rollout_seconds"] / x["episodes"] for x in measured]
    updates = [x["update_seconds"] for x in measured]
    result.update(stage=p["stage"], spec_hash=compiled.spec_hash,
                  measured_updates=len(measured), warmup_updates=min(1, len(rows)),
                  median_seconds_per_cohort=float(np.median(rollout)) if rollout else None,
                  median_seconds_per_update=float(np.median(updates)) if updates else None,
                  throughput_scope="Training rollout includes engine/tape construction, observations, action mapping and any declared credit branches; optimizer cost excludes validation/checkpoint I/O.",
                  validation_cadence="At the first completed update reaching each 2000-cohort mark; actual cohort count recorded.",
                  final_budget_selected=False)
    return result
