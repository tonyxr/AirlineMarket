"""Native R3 t+1 forecasts and exact posterior calibration on I3.

The head sees pre-action history. Exact after-observation probabilities are
Rao-Blackwell targets for bin calibration, not pointwise pre-action forecasts.
"""

import json
from pathlib import Path

import numpy as np
import torch

from airline_market.belief import ExactBelief
from airline_market.belief.events import AllOf, LegInventoryAtMost
from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.hashing import jsonable, schema_hash
from airline_market.core.rng import StreamRegistry
from airline_market.experiments.learning_pilot import initial_agent
from airline_market.learn.structured_train import (
    collect_structured,
    joint_label,
    load_structured,
    save_structured,
)
from airline_market.spec import compile_market, dump_spec, load_spec


def event_contract(c, focal, legs):
    """Bind the native one-event semantics; no same-width changed-event loading."""
    if len(legs) < 2 or len(set(legs)) != len(legs):
        raise ValueError("joint event requires distinct named rival legs")
    owners = [c.spec.legs[c.leg_index[name]].carrier for name in legs]
    if focal in owners or len(set(owners)) != 1:
        raise ValueError("joint event requires legs of one rival")
    return {
        "legs": list(legs),
        "rival": owners[0],
        "threshold": 1,
        "horizon": 1,
        "input": "permitted pre-action observation and recurrent history",
        "target": "realized inventory after one native step, including terminal",
        "action_conditioning": "head has no current-action input; stochastic native policy",
    }


def load_event_agent(path, adapter, contract):
    """Validate both the existing physical/schema contract and named event meaning."""
    agent, metadata = load_structured(path, adapter)
    if metadata.get("event_contract") != contract or agent.actor.aux_head is None:
        raise ValueError("checkpoint joint-event semantic contract mismatch")
    return agent, metadata


def act_with_forecast(agent, x, **kwargs):
    """Observe the exact native pre-sampling actor features without a second act."""
    features = []
    handle = agent.actor.norm.register_forward_hook(
        lambda module, args, output: features.append(output.detach().clone())
    )
    try:
        physical, hidden, info = agent.act(x, **kwargs)
    finally:
        handle.remove()
    if len(features) != 1 or agent.actor.aux_head is None:
        raise ValueError("expected one native actor feature vector and an event head")
    with torch.no_grad():
        probability = float(torch.sigmoid(agent.actor.aux_head(features[0]))[0, 0, 0])
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise FloatingPointError("invalid head forecast")
    return physical, hidden, info, probability


def calibration(rows, bins=10):
    """Exact-posterior ECE primary; proper binary Brier includes q(1-q)."""
    if not rows or bins != 10:
        raise ValueError("registered calibration requires rows and ten fixed bins")
    p = np.asarray([r["forecast"] for r in rows], dtype=float)
    q = np.asarray([r["posterior_probability"] for r in rows], dtype=float)
    y = np.asarray([r["realized_label"] for r in rows], dtype=float)
    if (
        not np.isfinite([p, q, y]).all()
        or np.any((p < 0) | (p > 1))
        or np.any((q < -1e-12) | (q > 1 + 1e-12))
        or not np.isin(y, [0.0, 1.0]).all()
    ):
        raise ValueError("invalid probability or binary label")
    # Binning depends only on the issued forecast; preserve raw posterior roundoff.
    membership = np.minimum((p * bins).astype(int), bins - 1)
    table = []
    for k in range(bins):
        selected = membership == k
        n = int(selected.sum())
        table.append(
            {
                "bin": k,
                "left": k / bins,
                "right": (k + 1) / bins,
                "n": n,
                "forecast": float(p[selected].mean()) if n else None,
                "posterior": float(q[selected].mean()) if n else None,
                "realized": float(y[selected].mean()) if n else None,
            }
        )
    ece = sum(
        r["n"] * abs(r["forecast"] - r["posterior"]) for r in table if r["n"]
    ) / len(rows)
    binary_ece = sum(
        r["n"] * abs(r["forecast"] - r["realized"]) for r in table if r["n"]
    ) / len(rows)
    return {
        "forecasts": len(rows),
        "cohorts": len({r["seed"] for r in rows}),
        "bins": table,
        "posterior_ECE": ece,
        "realized_ECE": binary_ece,
        "expected_binary_brier": float(np.mean(p * p - 2 * p * q + q)),
        "realized_binary_brier": float(np.mean((p - y) ** 2)),
        "posterior_squared_error_diagnostic": float(np.mean((p - q) ** 2)),
        "registered_0_03_pass": ece <= 0.03,
    }


def cohort_bootstrap(rows, repetitions=2000):
    """Resample complete physical cohorts, retaining all dependent period rows."""
    seeds = sorted({r["seed"] for r in rows})
    if len(seeds) < 2:
        return {
            "repetitions": repetitions,
            "ECE_95_interval": None,
            "scope": "fewer than two independent physical cohort IDs",
        }
    delta = np.zeros((len(seeds), 10))
    sizes = np.zeros(len(seeds))
    lookup = {seed: k for k, seed in enumerate(seeds)}
    for row in rows:
        k = lookup[row["seed"]]
        b = min(int(row["forecast"] * 10), 9)
        delta[k, b] += row["forecast"] - row["posterior_probability"]
        sizes[k] += 1
    rng = StreamRegistry(773, "analysis/V7/R3/cohort-bootstrap").generator()
    values = []
    for _ in range(repetitions):
        index = rng.integers(len(seeds), size=len(seeds))
        values.append(
            float(np.abs(delta[index].sum(axis=0)).sum() / sizes[index].sum())
        )
    return {
        "repetitions": repetitions,
        "ECE_95_interval": np.quantile(values, [0.025, 0.975]).tolist(),
        "unit": "whole physical cohort, all period forecasts retained",
        "scope": "conditional on this fixed policy/one NN initialization, not training-seed or F9 primary uncertainty",
    }


def validate_interval(start, count, training):
    lo, hi = (100000, 200000) if training else (200000, 300000)
    if count < 1 or not lo <= start < start + count <= hi:
        raise ValueError("event study seed interval leaves its registered split")
    return range(start, start + count)


def evaluate(
    agent, adapter, seeds, contract, directory, label, progress, max_support=20000
):
    """All periods: issue forecast, execute native action, then filter and label."""
    c = adapter.c
    focal = adapter.env.focal
    event = AllOf(
        tuple(LegInventoryAtMost(contract["rival"], l, 1) for l in contract["legs"])
    )
    rows = []
    target = directory / f"predictions_{label}.jsonl"
    for seed in seeds:
        validate_interval(seed, 1, False)
        x = adapter.reset(seed)
        belief = ExactBelief(c, focal, adapter.env.arm.name)
        belief.reset(adapter.obs)
        hidden, previous, previous_coordinates, reward = None, None, None, 0.0
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(
                StreamRegistry(seed, "validation/V7/R3/policy").child_seed() % 2**31
            )
            for t in range(c.spec.horizon.n_periods):
                progress(
                    {
                        "stage": "calibration_" + label,
                        "seed": seed,
                        "period": t,
                        "forecasts_completed": len(rows),
                        "belief_support": len(belief.mass),
                    }
                )
                before = adapter.obs
                physical, hidden, info, p = act_with_forecast(
                    agent,
                    x,
                    hidden=hidden,
                    previous_action=previous,
                    previous_coordinates=previous_coordinates,
                    previous_reward=reward,
                )
                action = adapter.price_action(physical, before)
                x, reward, done, _ = adapter.step(physical)
                belief.update(before, action, adapter.obs)
                if len(belief.mass) > max_support:
                    raise RuntimeError(
                        "declared exact posterior support limit; no pruning"
                    )
                event_mass = [
                    (h, w, event.evaluate(h, c, adapter.env.i))
                    for h, w in belief.mass.items()
                ]
                q = sum(w for h, w, happened in event_mass if happened)
                # Truth is read after both forecast and Bayesian update, for audit only.
                truth = joint_label(adapter.env, contract["legs"])
                row = {
                    "seed": seed,
                    "period": t,
                    "target_period": t + 1,
                    "forecast": p,
                    "posterior_probability": q,
                    "realized_label": truth,
                    "physical_action": physical.tolist(),
                    "prices": action.prices,
                    "before_observation": jsonable(before),
                    "after_observation": jsonable(adapter.obs),
                    "posterior_support": len(belief.mass),
                    "kernel": type(belief.kernel).__name__,
                    "posterior_sha256": schema_hash(
                        sorted((schema_hash(h), w, bool(e)) for h, w, e in event_mass)
                    ),
                    "posterior_mass": sum(belief.mass.values()),
                    "terminal": done,
                }
                with target.open("a") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)
                if done != (t == c.spec.horizon.n_periods - 1):
                    raise AssertionError("unexpected native terminal timing")
                previous, previous_coordinates = physical, info["policy_coordinates"]
    summary = calibration(rows)
    summary["cohort_bootstrap"] = cohort_bootstrap(rows)
    summary["by_period"] = {
        str(t): calibration([r for r in rows if r["period"] == t])
        for t in range(c.spec.horizon.n_periods)
    }
    atomic_json(directory / f"calibration_{label}.json", summary)
    return summary


def train_native(c, cfg, contract, directory, progress):
    """Native SRL-CA collector/PPO updates, exact declared cohort count, no tuning."""
    seeds = validate_interval(cfg["train_seed_start"], cfg["n_train"], True)
    adapter, agent = initial_agent(c, cfg["actor"])
    history, consumed = [], 0
    batch = cfg["batch"]
    if batch < 1:
        raise ValueError("positive native batch required")
    while consumed < len(seeds):
        totals = []
        for seed in seeds[consumed : consumed + batch]:
            progress(
                {
                    "stage": "training",
                    "cohorts_completed": consumed + len(totals),
                    "cohorts_total": len(seeds),
                    "update": agent.updates + 1,
                }
            )
            episode, logp, total = collect_structured(
                adapter,
                agent,
                seed,
                training=True,
                credit=cfg["credit"],
                event_legs=tuple(contract["legs"]),
            )
            agent.add_episode(
                episode, actions_are_physical=False, behavior_log_probabilities=logp
            )
            totals.append(total)
        stats = agent.update()
        consumed += len(totals)
        history.append(
            {
                "cohorts": consumed,
                "update": agent.updates,
                "mean_objective": float(np.mean(totals)),
                **{k: float(v) for k, v in stats.items()},
            }
        )
        metadata = {
            "consumed": consumed,
            "updates": agent.updates,
            "event_contract": contract,
            "event_contract_sha256": schema_hash(contract),
            "physical_training_interval": [
                cfg["train_seed_start"],
                cfg["train_seed_start"] + consumed,
            ],
            "variant": "srl_ca",
            "validation_used_for_selection": False,
        }
        save_structured(agent, directory / "last.pt", adapter, metadata)
        atomic_json(directory / "learning.json", {"history": history, **metadata})
    return agent, adapter, metadata


def run(cell, context):
    """New I3/default20k learner and predeclared initial/final exact event audit."""
    p = cell["parameters"]
    validate_interval(p["train_seed_start"], p["n_train"], True)
    seeds = validate_interval(p["validation_seed_start"], p["n_validation"], False)
    source = Path(p["source_spec"])
    if file_hash(source) != p["source_spec_sha256"]:
        raise ValueError("registered physical source changed")
    c = compile_market(load_spec(source))
    if (
        c.spec_hash != p["compiled_spec_hash"]
        or c.spec.demand.mode != "single_request_bernoulli"
    ):
        raise ValueError("registered exact Bernoulli model required")
    if p["actor"]["learner"] != "srl_ca":
        raise ValueError("R3 requires the actual auxiliary-head learner")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    directory = context.directory
    dump_spec(c.spec, directory / "spec.json")
    contract = event_contract(c, p["actor"].get("focal", "A"), p["event_legs"])
    atomic_json(directory / "event_contract.json", contract)
    current = {"stage": "initializing"}
    context.counters = lambda: current.copy()

    def progress(row):
        current.update(row)

    adapter, initial = initial_agent(c, p["actor"])
    initial_summary = evaluate(
        initial,
        adapter,
        seeds,
        contract,
        directory,
        "initial",
        progress,
        p["max_support"],
    )
    del initial
    agent, adapter, metadata = train_native(c, p, contract, directory, progress)
    del agent
    agent, restored_metadata = load_event_agent(
        directory / "last.pt", adapter, contract
    )
    if metadata != restored_metadata:
        raise AssertionError("native final checkpoint metadata changed")
    final_summary = evaluate(
        agent, adapter, seeds, contract, directory, "final", progress, p["max_support"]
    )
    files = [
        "event_contract.json",
        "learning.json",
        "predictions_initial.jsonl",
        "predictions_final.jsonl",
        "calibration_initial.json",
        "calibration_final.json",
    ]
    hashes = {name: file_hash(directory / name) for name in files}
    atomic_json(
        directory / "checkpoint_manifest.json",
        {
            "last.pt": file_hash(directory / "last.pt"),
            "event_contract_sha256": schema_hash(contract),
            "metadata": metadata,
            "weights_local_gitignored": True,
        },
    )
    return {
        "status": "completed",
        "spec_hash": c.spec_hash,
        "training_cohorts": p["n_train"],
        "updates": agent.updates,
        "validation_cohorts_per_policy": len(seeds),
        "event_contract": contract,
        "initial": initial_summary,
        "final": final_summary,
        "artifact_hashes": hashes,
        "checkpoint_manifest_sha256": file_hash(directory / "checkpoint_manifest.json"),
        "full_V7_pass": None,
        "scientific_RL_budget": None,
        "scope": "One registered I3/direct SRL-CA pilot, not fullP2/TR/learner ranking. Native t+1 label semantics; pre-action head compared by calibration bins to exact after-observation posteriors via tower property, not pointwise posterior equality. Proper binary Brier retains posterior Bernoulli variance. All four periods/terminal retained, same256 physical IDs/initial-final policies have different action histories. FinalECE<=.03 is this fixed-policy/initialization criterion only, not fullV7 or training-seed coverage. No checkpoint selection or retuning on calibration data.",
    }
