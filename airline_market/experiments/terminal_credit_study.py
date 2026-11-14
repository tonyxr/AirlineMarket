"""Native R2 terminal-decision signs versus full remaining-horizon exact Q.

This training-label diagnostic conditions on the actual physical state. Actors
receive permitted observations only. It does not integrate a recurrent Gaussian
continuation away from terminal or certify the full V7 credit requirement.
"""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.hashing import canonical_json, schema_hash
from airline_market.core.rng import StreamRegistry
from airline_market.experiments.learning_pilot import initial_agent
from airline_market.learn.structured_train import branch_credit
from airline_market.sim.kernel import ExactKernel
from airline_market.spec import compile_market, dump_spec, load_spec


def validation_seeds(start, count):
    """Validate the full interval before any physical cohort is opened."""
    if count < 1 or not 200000 <= start < start + count <= 300000:
        raise ValueError("terminal credit uses validation physical seeds only")
    return range(start, start + count)


def alternatives(agent, physical, previous, seed, K=4):
    """Expose the original native alternative law without modifying that source."""
    if K < 2:
        raise ValueError("credit requires at least two alternatives")
    hold = (
        previous
        if previous is not None
        else agent.action_transform.coordinates_to_physical(np.zeros(agent.action_dim))
    )
    rng = StreamRegistry(seed, "train/structured/credit/alternatives").generator()
    return [physical, hold] + [
        agent.action_transform.coordinates_to_physical(
            rng.normal(size=agent.action_dim)
        )
        for _ in range(K - 2)
    ]


def terminal_q(kernel, adapter, actions):
    """Exact conditional Q at the last decision, including service/share reward."""
    env = adapter.env
    T = env.compiled.spec.horizon.n_periods
    if env.engine.world.t != T - 1 or adapter.obs.t != T - 1:
        raise ValueError("exact terminal Q requires the last decision")
    state = kernel.encode(env.engine.world, env.i)
    values, masses, counts = [], [], []
    for physical in actions:
        action = adapter.price_action(physical, adapter.obs)
        rows = kernel.transition(
            state, env.i, action, arm=env.arm, observation=adapter.obs
        )
        mass = sum(p for p, _, _ in rows)
        if not np.isfinite(mass) or abs(mass - 1.0) > 1e-12:
            raise ValueError("exact transition probability mass is not one")
        total = 0.0
        for p, child, signal in rows:
            if child.t != T:
                raise ValueError("terminal Q has a nonterminal successor")
            world = kernel.engine(child, env.i).world
            total += p * (
                signal.cash_delta / env.objective.spec.profit_scale
                + env.objective.terminal_reward(world)
            )
        values.append(total)
        masses.append(mass)
        counts.append(len(rows))
    if not np.isfinite(values).all():
        raise FloatingPointError("nonfinite exact terminal Q")
    return np.asarray(values), masses, counts, state


def contrasts(values):
    """Centered credit used by R2 and distinct hold-relative diagnostic."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("invalid counterfactual return vector")
    return {
        "centered": float(values[0] - values.mean()),
        "hold_relative": float(values[0] - values[1]),
    }


def sign(value, tolerance=1e-12):
    """Three-way signs, with declared symmetric numerical-zero tolerance."""
    if not np.isfinite(value) or tolerance < 0:
        raise ValueError("invalid sign input")
    return int(value > tolerance) - int(value < -tolerance)


def summarize(rows, tolerance=1e-12):
    """Keep zeros and every mismatch; nonzero denominator is reported separately."""
    if not rows:
        raise ValueError("no completed terminal comparisons")
    result = {}
    for name in ("centered", "hold_relative"):
        pairs = [
            (sign(r["exact"][name], tolerance), sign(r["estimated"][name], tolerance))
            for r in rows
        ]
        nonzero = [(a, b) for a, b in pairs if a]
        confusion = {
            f"{a}:{b}": pairs.count((a, b)) for a in (-1, 0, 1) for b in (-1, 0, 1)
        }
        rate = sum(a == b for a, b in pairs) / len(pairs)
        result[name] = {
            "count": len(pairs),
            "exact_zero_count": len(pairs) - len(nonzero),
            "agreement": rate,
            "disagreement_count": sum(a != b for a, b in pairs),
            "exact_nonzero_count": len(nonzero),
            "nonzero_agreement": sum(a == b for a, b in nonzero) / len(nonzero)
            if nonzero
            else None,
            "confusion_exact_estimated": confusion,
            "terminal_subset_meets_0_90": rate >= 0.90,
        }
    return result


def one_cohort(adapter, agent, kernel, seed, K=4, horizon=5):
    """One fresh observable-policy trajectory; exact state used only after action."""
    validation_seeds(seed, 1)
    x = adapter.reset(seed)
    hidden, previous, previous_coordinates, reward = None, None, None, 0.0
    T = adapter.c.spec.horizon.n_periods
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(
            StreamRegistry(seed, "validation/V7/terminal/policy").child_seed() % 2**31
        )
        for t in range(T):
            physical, hidden, info = agent.act(
                x,
                hidden=hidden,
                previous_action=previous,
                previous_coordinates=previous_coordinates,
                previous_reward=reward,
            )
            coordinates = info["policy_coordinates"]
            if t == T - 1:
                credit_seed = (
                    StreamRegistry(seed, "validation/V7/terminal/credit").child_seed()
                    % 2**31
                )
                choices = alternatives(agent, physical, previous, credit_seed, K)
                snapshot = hashlib.sha256(
                    pickle.dumps(adapter.env.snapshot())
                ).hexdigest()
                rng = torch.random.get_rng_state().clone()
                sampled = branch_credit(
                    adapter,
                    agent,
                    physical,
                    hidden,
                    previous,
                    seed=credit_seed,
                    K=K,
                    horizon=horizon,
                    coordinates=coordinates,
                    previous_coordinates=previous_coordinates,
                )
                if hashlib.sha256(
                    pickle.dumps(adapter.env.snapshot())
                ).hexdigest() != snapshot or not torch.equal(
                    rng, torch.random.get_rng_state()
                ):
                    raise AssertionError(
                        "native credit changed parent physical state or torch RNG"
                    )
                exact, masses, counts, state = terminal_q(kernel, adapter, choices)
                estimated = contrasts(sampled["returns"])
                if any(estimated[k] != sampled[k] for k in estimated):
                    raise AssertionError(
                        "native alternative/contrast arithmetic mismatch"
                    )
                return {
                    "seed": seed,
                    "period": t,
                    "credit_seed": credit_seed,
                    "K": K,
                    "H": horizon,
                    "physical_alternatives": [a.tolist() for a in choices],
                    "price_alternatives": [
                        adapter.price_action(a, adapter.obs).prices for a in choices
                    ],
                    "sampled_returns": sampled["returns"],
                    "exact_Q": exact.tolist(),
                    "estimated": estimated,
                    "exact": contrasts(exact),
                    "exact_probability_mass": masses,
                    "exact_transition_rows": counts,
                    "state": json.loads(canonical_json(state)),
                    "observation": json.loads(canonical_json(adapter.obs)),
                    "parent_snapshot_sha256": snapshot,
                    "parent_unchanged": True,
                }
            x, reward, done, _ = adapter.step(physical)
            if done:
                raise AssertionError("physical episode terminated before last decision")
            previous, previous_coordinates = physical, coordinates
    raise AssertionError("missing terminal decision")


def run(cell, context):
    """Prospectively registered I0/I1 terminal subset, with every row durable."""
    p = cell["parameters"]
    seeds = validation_seeds(p["seed_start"], p["n_cohorts"])
    source = Path(p["source_spec"])
    if file_hash(source) != p["source_spec_sha256"]:
        raise ValueError("changed registered physical source")
    c = compile_market(load_spec(source))
    if (
        c.spec_hash != p["compiled_spec_hash"]
        or c.spec.demand.mode != "single_request_bernoulli"
    ):
        raise ValueError("terminal credit requires registered exact Bernoulli law")
    if p["K"] != 4 or p["H"] != 5 or p["sign_tolerance"] != 1e-12:
        raise ValueError("registered native credit configuration changed")
    torch.set_num_threads(1)
    adapter, agent = initial_agent(c, p["actor"])
    kernel = ExactKernel(c)
    atomic_json(
        context.directory / "actor_initialization.json",
        {
            "actor_config": p["actor"],
            "schema_hash": adapter.schema_hash(),
            "actor_weights_sha256": schema_hash(
                {
                    k: v.detach().cpu().numpy()
                    for k, v in agent.actor.state_dict().items()
                }
            ),
            "optimizer_updates": agent.updates,
            "scope": "Named-seed initial weights only, no training or saved-weight substitution.",
        },
    )
    dump_spec(c.spec, context.directory / "spec.json")
    current = {"stage": "terminal_credit", "cohorts_total": len(seeds)}
    context.counters = lambda: current.copy()
    rows = []
    for seed in seeds:
        current.update(seed=seed, cohorts_completed=len(rows))
        row = one_cohort(adapter, agent, kernel, seed, p["K"], p["H"])
        with (context.directory / "comparisons.jsonl").open("a") as f:
            f.write(canonical_json(row) + "\n")
        rows.append(row)
    summary = summarize(rows, p["sign_tolerance"])
    atomic_json(context.directory / "sign_summary.json", summary)
    return {
        "status": "completed",
        "instance": p["instance"],
        "spec_hash": c.spec_hash,
        "cohorts": len(rows),
        "summary": summary,
        "comparisons_sha256": file_hash(context.directory / "comparisons.jsonl"),
        "summary_sha256": file_hash(context.directory / "sign_summary.json"),
        "full_V7_R2_pass": None,
        "scientific_RL_budget": None,
        "scope": "Native H5/K4 terminal decisions on fresh256 validation cohorts per original I0/I1 model. Initial NNseed0/default SRL-C actor; no optimizer training or learned-performance claim. Exact full-remaining-horizon Q conditional on the training physical state, read after actor action. Centered primary/hold-relative diagnostic, all zero and mismatched signs retained. This subset does not verify earlier stochastic recurrent continuations, belief-conditioned optimal Q, full R2, full V7 or primary F9.",
    }
