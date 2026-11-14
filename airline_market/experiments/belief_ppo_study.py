"""Exact-small M-PPO-B learning precursor; not the24-trial I5 tuning study."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.rng import StreamRegistry
from airline_market.experiments.learning_pilot import paired_summary
from airline_market.learn.belief_ppo import adapter, initial_agent, train_belief
from airline_market.learn.ppo import collect
from airline_market.learn.structured_train import seed_in_split
from airline_market.sim.sparse_access import sparse_access
from airline_market.spec import compile_market, dump_spec, load_spec


def evaluate(c, cfg, agent=None):
    """Use the same declared physical/policy validation streams as the flat pilot."""
    view, fresh = initial_agent(c, cfg)
    agent = fresh if agent is None else agent
    seeds = list(
        range(
            cfg.get("validation_seed_start", 200000),
            cfg.get("validation_seed_start", 200000) + cfg["n_validation"],
        )
    )
    result = {}
    for seed in seeds:
        seed_in_split(seed, False)
    for deterministic, label in [(True, "deterministic"), (False, "sampled")]:
        values = []
        for seed in seeds:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(
                    StreamRegistry(seed, "validation/flat/policy").child_seed() % 2**31
                )
                values.append(
                    collect(view, agent, seed, deterministic=deterministic)[2]
                )
        result[label + "_values"] = values
        result[label] = float(np.mean(values))
    return result


def pilot(cell, context):
    """Execute registered training and paired initial/final/fixed validation."""
    p = cell["parameters"]
    path = Path(p["source_spec"])
    if file_hash(path) != p["source_spec_sha256"]:
        raise ValueError("changed frozen belief-PPO physical specification")
    c = compile_market(load_spec(path))
    dump_spec(c.spec, context.directory / "spec.json")
    base = Path(p["baseline_file"])
    if file_hash(base) != p["baseline_sha256"]:
        raise ValueError("changed frozen training-fitted baseline")
    baseline = json.loads(base.read_text())
    cfg = {
        k: p[k]
        for k in [
            "n_train",
            "cohorts_per_update",
            "network_seed",
            "n_validation",
            "validation_seed_start",
            "train_seed_start",
            "validate_every_cohorts",
            "focal",
            "arm",
            "joint_events",
            "ppo",
        ]
        if k in p
    }
    seeds = list(
        range(
            cfg.get("validation_seed_start", 200000),
            cfg.get("validation_seed_start", 200000) + cfg["n_validation"],
        )
    )
    if (
        baseline["spec_hash"] != c.spec_hash
        or baseline["focal"] != cfg.get("focal", "A")
        or baseline["arm"] != cfg.get("arm", "direct")
        or baseline["validation_seeds"] != seeds
    ):
        raise ValueError("baseline physical/arm/validation pairing mismatch")
    current = {"stage": "untrained_validation"}
    context.counters = lambda: dict(current)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with sparse_access():
        initial = evaluate(c, cfg)
        atomic_json(context.directory / "initial_validation.json", initial)
        trained = train_belief(c, cfg, context.directory, current.update)
    learning = json.loads((context.directory / "learning.json").read_text())
    curve = []
    for row in learning["validation"]:
        curve.append(
            {
                "cohorts": row["cohorts"],
                "deterministic": row["deterministic"],
                "sampled": row["sampled"],
                "deterministic_minus_fixed": paired_summary(
                    row["deterministic_values"], baseline["validation_values"]
                ),
                "deterministic_minus_initial": paired_summary(
                    row["deterministic_values"], initial["deterministic_values"]
                ),
            }
        )
    view = adapter(c, cfg)
    checkpoints = []
    for name in ["best.pt", "last.pt"]:
        weight = context.directory / name
        checkpoints.append(
            {
                "file": name,
                "sha256": file_hash(weight),
                "bytes": weight.stat().st_size,
                "stored_local_ignored_binary": True,
            }
        )
    atomic_json(
        context.directory / "checkpoint_manifest.json",
        {
            "checkpoints": checkpoints,
            "names": view.names(),
            "schema_hash": view.schema_hash(),
            "belief_contract": view.contract(),
            "action_transform": asdict(view.action_transform),
            "baseline_sha256": p["baseline_sha256"],
        },
    )
    report = {
        "initial": initial,
        "curve": curve,
        "final": curve[-1],
        "validation_seeds": seeds,
        "network_seed": cfg.get("network_seed", 0),
        "belief_contract": view.contract(),
        "baseline_sha256": p["baseline_sha256"],
        "schema_hash": view.schema_hash(),
        "scope": "One exact-small M-PPO-B seed/default-budget learning precursor. Not24x50k I5hyperparameter trials, hubPF certification or a final ranking. Best checkpoint uses selection data and is descriptive.",
        "test_seeds_opened": False,
    }
    atomic_json(context.directory / "pilot.json", report)
    return {
        **trained,
        "spec_hash": c.spec_hash,
        "instance": p["instance"],
        "final_minus_fixed": curve[-1]["deterministic_minus_fixed"],
        "final_minus_initial": curve[-1]["deterministic_minus_initial"],
        "baseline_sha256": p["baseline_sha256"],
        "pilot_sha256": file_hash(context.directory / "pilot.json"),
        "full_P2_tuning_completed": False,
        "final_budget_selected": False,
    }
