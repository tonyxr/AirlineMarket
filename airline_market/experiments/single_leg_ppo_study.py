"""Native PPO monopoly learning against the complete executed cent-price class."""

import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.hashing import canonical_json, schema_hash
from airline_market.core.rng import StreamRegistry
from airline_market.core.types import CoefficientAction
from airline_market.experiments.checkpoint_pair_study import inference_fingerprint
from airline_market.experiments.learning_pilot import initial_agent
from airline_market.learn.adapters import RecurrentEnvAdapter
from airline_market.learn.ppo import collect, save_checkpoint
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.sim.accounting import carrier_metrics
from airline_market.sim.kernel import ExactKernel
from airline_market.sim.sparse_access import sparse_access
from airline_market.sim.tape import ExogenousTape
from airline_market.solvers.fixed_search import episode_value
from airline_market.spec import compile_market, dump_spec, load_spec


def market(p):
    if file_hash(Path(p["source_spec"])) != p["source_spec_sha256"]:
        raise ValueError("single-leg frozen specification changed")
    c = compile_market(load_spec(Path(p["source_spec"])))
    if c.spec_hash != p["compiled_spec_hash"]:
        raise ValueError("single-leg compiled specification mismatch")
    return c


def contract(c):
    """Require the stated one-class, no-shock, full-observation monopoly law."""
    s = c.spec
    if (
        len(s.carriers) != 1
        or len(s.legs) != 1
        or len(s.products) != 1
        or len(s.markets) != 1
        or len(s.segments) != 1
        or len(s.segments[0].classes) != 1
        or len(s.exogenous.states) != 1
        or s.exogenous.transitions
        or s.capacity_process.events
        or s.background
        or s.quotes.requests
        or s.capacity_priors
        or s.demand.mode != "single_request_bernoulli"
        or s.choice.model != "latent_class_mnl"
        or s.objectives[0].share_weight != 0
    ):
        raise ValueError("benchmark requires the declared simple monopoly law")
    leg, prod, taste, pricing = (
        s.legs[0],
        s.products[0],
        s.segments[0].classes[0],
        s.pricing[0],
    )
    if (
        len(leg.capacity.support) != 1
        or leg.capacity.probabilities != (1.0,)
        or pricing.fixed_features
        or pricing.time_ramp
        or pricing.update_every != 1
        or [f.name for f in pricing.controllable_features] != ["intercept"]
        or len(pricing.theta_bounds) != 1
        or taste.product_shift
        or taste.time_penalty_per_hour
        or taste.schedule_penalty_per_hour
        or s.demand.holiday_multiplier != 1
        or s.demand.exogenous_multipliers
    ):
        raise ValueError("benchmark price/state/demand contract changed")
    if any(
        s.demand.market_weights.value_at(t) != (1.0,)
        or s.demand.segment_weights.value_at(t) != (1.0,)
        for t in range(s.horizon.n_periods)
    ):
        raise ValueError("monopoly needs one market and one segment")
    lo, hi = pricing.theta_bounds[0]
    base = pricing.base[prod.id]
    if (
        abs(base + lo - prod.fare_floor) > 1e-12
        or abs(base + hi - prod.fare_cap) > 1e-12
    ):
        raise ValueError(
            "native coefficient range must span the full declared fare bounds"
        )
    cents = [round(prod.fare_floor * 100), round(prod.fare_cap * 100)]
    if any(
        abs(x / 100 - y) > 1e-12
        for x, y in zip(cents, [prod.fare_floor, prod.fare_cap])
    ):
        raise ValueError("benchmark fare bounds must be integer cents")
    return {
        "T": s.horizon.n_periods,
        "capacity": leg.capacity.support[0],
        "base": base,
        "cost": prod.service_cost + leg.cost_per_pax,
        "value": taste.market_value[s.markets[0].id]
        + taste.carrier_pref.get(s.carriers[0].code, 0.0),
        "alpha": taste.alpha,
        "cent_bounds": cents,
        "profit_scale": s.objectives[0].profit_scale,
    }


def classical_dp(c):
    """Enumerate EVERY attainable cent; no unrounded-continuous optimum claim."""
    q = contract(c)
    fares = np.arange(q["cent_bounds"][0], q["cent_bounds"][1] + 1) / 100
    values = np.zeros((q["T"] + 1, q["capacity"] + 1))
    prices = np.full((q["T"], q["capacity"] + 1), fares[0])
    for t in range(q["T"] - 1, -1, -1):
        arrival = min(
            c.spec.demand.arrival_probability_cap,
            c.spec.demand.arrival_probability.value_at(t),
        )
        probability = arrival * expit(q["value"] - q["alpha"] * fares)
        for stock in range(1, q["capacity"] + 1):
            candidate = values[t + 1, stock] + probability * (
                fares - q["cost"] + values[t + 1, stock - 1] - values[t + 1, stock]
            )
            k = int(np.argmax(candidate))
            values[t, stock], prices[t, stock] = candidate[k], fares[k]
    return {
        **q,
        "values_USD": values.tolist(),
        "prices_USD": prices.tolist(),
        "V_star_USD": float(values[0, q["capacity"]]),
        "attainable_fare_count": len(fares),
        "price_class": "Complete native cent-rounded bounded price image, not unrounded continuous optimum.",
    }


def exact_tree(c, selector):
    """General reference transitions, permitted observations and full recurrent histories."""
    kernel = ExactKernel(c)
    env = FocalCarrierEnv(c, "A", "direct")
    arm = env.arm
    nodes = []

    def visit(
        state, obs, hidden=None, previous=None, coordinates=None, reward=0.0, path=()
    ):
        if state.t == c.spec.horizon.n_periods:
            return 0.0
        action, after_hidden, after_coordinates = selector(
            obs, hidden, previous, coordinates, reward
        )
        before = kernel.engine(state).world
        transitions = kernel.transition(
            state, 0, CoefficientAction(tuple(action)), arm=arm, observation=obs
        )
        mass = sum(p for p, _, _ in transitions)
        if abs(mass - 1.0) > 1e-12:
            raise ValueError(
                "exact policy transition mass is not one; no renormalization"
            )
        total = 0.0
        for branch, (p, after, signal) in enumerate(transitions):
            world = kernel.engine(after).world
            r = env.objective.period_reward(0, before, world)
            if after.t == c.spec.horizon.n_periods:
                r += env.objective.terminal_reward(world)
            total += p * (
                r
                + visit(
                    after,
                    signal.observation,
                    after_hidden,
                    action,
                    after_coordinates,
                    r,
                    (*path, branch),
                )
            )
        nodes.append(
            {
                "period": state.t,
                "history_branch_path": path,
                "hidden_state_sha256": schema_hash(hidden)
                if hidden is not None
                else None,
                "observation": obs,
                "physical_action": action,
                "previous_action": previous,
                "previous_coordinates": coordinates,
                "previous_reward": reward,
                "mass": mass,
                "branches": len(transitions),
                "conditional_J": total,
            }
        )
        return total

    initial = kernel.initial_distribution(0)
    if abs(sum(p for p, _ in initial) - 1.0) > 1e-12:
        raise ValueError("initial reference mass is not one")
    value = sum(p * visit(s, kernel.engine(s).observe(0, arm)) for p, s in initial)
    return value, nodes


def exact_actor(adapter, agent):
    """No hidden-state input to the actor, no optimizer/normalizer/private RNG mutation."""
    before = inference_fingerprint(agent)

    def policy(obs, hidden, previous, coordinates, reward):
        a, h, info = agent.act(
            adapter.encode(obs, {}),
            hidden=hidden,
            previous_action=previous,
            previous_coordinates=coordinates,
            previous_reward=reward,
            deterministic=True,
        )
        return a, h, info["policy_coordinates"]

    with torch.random.fork_rng(devices=[]):
        value, nodes = exact_tree(adapter.env.compiled, policy)
    if before != inference_fingerprint(agent):
        raise AssertionError("exact audit mutated learner state")
    return value, nodes


class StoredTapeEnv(FocalCarrierEnv):
    """Native reset/step with a saved immutable exogenous tape instead of regeneration."""

    tape = None

    def reset(self, seed):
        if self.tape is None or self.tape.seed != seed:
            raise ValueError("saved validation tape/seed mismatch")
        self.engine.reset(seed, tape=self.tape)
        return self.engine.observe(self.i, self.arm), self._info()


def tape_from_bank(bank, index, seed, T):
    args = {
        f.name: bank[f.name][index]
        for f in fields(ExogenousTape)
        if f.name not in {"seed", "arrival_counts", "arrival_marks"}
    }
    result = ExogenousTape(
        seed=seed,
        arrival_counts=None,
        arrival_marks=tuple(() for _ in range(T)),
        **args,
    )
    result.freeze()
    return result


def validation_seeds(p):
    start, count = p["validation_seed_start"], p["n_validation"]
    if count < 2 or start < 200000 or start + count > 300000:
        raise ValueError("single-leg validation interval invalid")
    return list(range(start, start + count))


def prepare(cell, context):
    """One DP reference/baseline and one physical tape bank reused by ALL five learners."""
    p = cell["parameters"]
    seeds = validation_seeds(p)
    c = market(p)
    target = classical_dp(c)
    table = np.asarray(target["prices_USD"])
    policy = lambda obs: CoefficientAction(
        (table[obs.t, obs.own_inventory[0]] - target["base"],)
    )
    exact, nodes = exact_tree(c, lambda o, h, a, z, r: (policy(o).theta, None, None))
    if abs(exact * target["profit_scale"] - target["V_star_USD"]) > 1e-10:
        raise AssertionError(
            "classical DP differs from native general-kernel policy value"
        )
    target.update(native_reference_J=exact, native_reference_nodes=len(nodes))
    atomic_json(context.directory / "dp_target.json", target)
    atomic_json(context.directory / "reference_tree.json", nodes)
    env = StoredTapeEnv(c, "A", "direct")
    tapes, rows = [], []
    current = {
        "stage": "DP_validation_tape_bank",
        "completed": 0,
        "cohorts": len(seeds),
    }
    context.counters = lambda: dict(current)
    with sparse_access():
        for k, seed in enumerate(seeds):
            tape = ExogenousTape.generate(c, seed)
            tapes.append(tape)
            env.tape = tape
            value = episode_value(env, seed, policy)
            rows.append(
                {
                    "seed": seed,
                    "DP_J": value,
                    "tape_sha256": schema_hash(tape),
                    "accounting": carrier_metrics(env.engine.world, c, 0),
                }
            )
            current["completed"] = k + 1
    arrays = {
        f.name: np.stack([getattr(t, f.name) for t in tapes])
        for f in fields(ExogenousTape)
        if f.name not in {"seed", "arrival_counts", "arrival_marks"}
    }
    np.savez_compressed(context.directory / "validation_tapes.npz", **arrays)
    atomic_json(context.directory / "DP_validation.json", rows)
    dump_spec(c.spec, context.directory / "spec.json")
    return {
        "status": "completed",
        "spec_hash": c.spec_hash,
        "V_star_USD": target["V_star_USD"],
        "cohorts": len(seeds),
        "native_reference_nodes": len(nodes),
        "dp_target_sha256": file_hash(context.directory / "dp_target.json"),
        "reference_tree_sha256": file_hash(context.directory / "reference_tree.json"),
        "validation_tapes_sha256": file_hash(
            context.directory / "validation_tapes.npz"
        ),
        "DP_validation_sha256": file_hash(context.directory / "DP_validation.json"),
    }


def train(cell, context):
    """Full declared cohorts for one initialization, unchanged native collector/update."""
    p = cell["parameters"]
    seeds = validation_seeds(p)
    if p["train_seed_start"] < 100000 or p["train_seed_start"] + p["n_train"] > 200000:
        raise ValueError("single-leg training interval invalid")
    if p["network_seed"] not in range(10):
        raise ValueError("network initialization outside declared0–9")
    for path, h in p["reference_hashes"].items():
        if file_hash(Path(path)) != h:
            raise ValueError("frozen single-leg reference changed")
    ref = Path(p["reference_directory"])
    required = [
        "COMPLETE.json",
        "results.json",
        "spec.json",
        "dp_target.json",
        "DP_validation.json",
        "validation_tapes.npz",
        "reference_tree.json",
    ]
    if any(str(ref / name) not in p["reference_hashes"] for name in required):
        raise ValueError("missing mandatory frozen reference hash")
    receipt = json.loads((ref / "COMPLETE.json").read_text())
    if receipt["status"] != "completed" or receipt["results_sha256"] != file_hash(
        ref / "results.json"
    ):
        raise ValueError("invalid DP reference result receipt")
    c = market(p)
    q = contract(c)
    target = json.loads((ref / "dp_target.json").read_text())
    reference_result = json.loads((ref / "results.json").read_text())
    if (
        reference_result["spec_hash"] != c.spec_hash
        or any(target[k] != v for k, v in q.items())
        or target["attainable_fare_count"]
        != q["cent_bounds"][1] - q["cent_bounds"][0] + 1
        or target["V_star_USD"] <= 0
    ):
        raise ValueError(
            "DP reference does not match the declared native price/model class"
        )
    baseline = json.loads((ref / "DP_validation.json").read_text())
    if [r["seed"] for r in baseline] != seeds:
        raise ValueError("single-leg saved DP validation sequence mismatch")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    adapter, agent = initial_agent(
        c, {"learner": "ppo", "network_seed": p["network_seed"], "ppo": p["ppo"]}
    )
    dump_spec(c.spec, context.directory / "spec.json")
    current = {
        "stage": "initial_exact",
        "training_completed": 0,
        "training_cohorts": p["n_train"],
        "updates": 0,
    }
    context.counters = lambda: dict(current)
    history, curve, consumed = [], [], 0
    next_audit = p["audit_every_cohorts"]

    def audit():
        value, nodes = exact_actor(adapter, agent)
        ratio = value * q["profit_scale"] / target["V_star_USD"]
        if value * q["profit_scale"] > target["V_star_USD"] + 1e-10:
            raise AssertionError(
                "native policy expectation exceeds same-class DP optimum"
            )
        row = {
            "training_cohorts": consumed,
            "updates": agent.updates,
            "exact_J": value,
            "exact_USD": value * q["profit_scale"],
            "ratio_to_V_star": ratio,
            "meets_0_98": ratio >= 0.98,
            "native_reference_nodes": len(nodes),
        }
        atomic_json(context.directory / (f"exact_tree_{consumed:06d}.json"), nodes)
        curve.append(row)
        atomic_json(context.directory / "exact_curve.json", curve)
        save_checkpoint(
            agent,
            context.directory / f"checkpoint_{consumed:06d}.pt",
            adapter,
            {"consumed": consumed, "network_seed": p["network_seed"], "exact": row},
        )

    with sparse_access():
        audit()
        with (context.directory / "updates.jsonl").open("x") as log:
            while consumed < p["n_train"]:
                current.update(stage="training", updates=agent.updates)
                returns = []
                started = time.perf_counter()
                for k in range(min(p["cohorts_per_update"], p["n_train"] - consumed)):
                    seed = p["train_seed_start"] + consumed + k
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(
                            StreamRegistry(seed, "train/flat/policy").child_seed()
                            % 2**31
                        )
                        episode, logp, value = collect(adapter, agent, seed)
                    agent.add_episode(
                        episode,
                        actions_are_physical=False,
                        behavior_log_probabilities=logp,
                    )
                    returns.append(value)
                    current["training_completed"] = consumed + k + 1
                rollout_seconds = time.perf_counter() - started
                started = time.perf_counter()
                stats = agent.update()
                if not np.isfinite(list(stats.values())).all():
                    raise ValueError("nonfinite native optimizer statistic")
                consumed += len(returns)
                history.append(
                    {
                        "cohorts": consumed,
                        "mean_return": float(np.mean(returns)),
                        "rollout_seconds": rollout_seconds,
                        "update_seconds": time.perf_counter() - started,
                        **stats,
                    }
                )
                log.write(canonical_json(history[-1]) + "\n")
                log.flush()
                current.update(training_completed=consumed, updates=agent.updates)
                if consumed >= next_audit or consumed == p["n_train"]:
                    current["stage"] = "exact_policy_audit"
                    audit()
                    next_audit = (consumed // p["audit_every_cohorts"] + 1) * p[
                        "audit_every_cohorts"
                    ]
                save_checkpoint(
                    agent,
                    context.directory / "last.pt",
                    adapter,
                    {
                        "consumed": consumed,
                        "network_seed": p["network_seed"],
                        "test_seeds_opened": False,
                    },
                )
        current.clear()
        current.update(
            stage="final_validation",
            completed=0,
            cohorts=len(seeds),
            training_completed=consumed,
            training_cohorts=p["n_train"],
            updates=agent.updates,
        )
        validation = RecurrentEnvAdapter(StoredTapeEnv(c, "A", "direct"))
        fingerprint = inference_fingerprint(agent)
        pairs = []
        with np.load(ref / "validation_tapes.npz", allow_pickle=False) as archive:
            bank = {name: archive[name] for name in archive.files}
        with (context.directory / "validation_pairs.jsonl").open("x") as log:
            for k, seed in enumerate(seeds):
                tape = tape_from_bank(bank, k, seed, q["T"])
                if schema_hash(tape) != baseline[k]["tape_sha256"]:
                    raise ValueError("saved physical tape identity mismatch")
                validation.env.tape = tape
                with torch.random.fork_rng(devices=[]):
                    value = collect(validation, agent, seed, deterministic=True)[2]
                row = {
                    "seed": seed,
                    "PPO_J": value,
                    "DP_J": baseline[k]["DP_J"],
                    "difference_J": value - baseline[k]["DP_J"],
                    "tape_sha256": baseline[k]["tape_sha256"],
                }
                pairs.append(row)
                log.write(canonical_json(row) + "\n")
                log.flush()
                current["completed"] = k + 1
        if inference_fingerprint(agent) != fingerprint:
            raise AssertionError("final validation mutated learner state")
    checkpoint = context.directory / "last.pt"
    atomic_json(
        context.directory / "checkpoint_manifest.json",
        {
            "checkpoints": [
                {
                    "path": str(x),
                    "sha256": file_hash(x),
                    "bytes": x.stat().st_size,
                    "storage": "local/Gitignored",
                }
                for x in sorted(context.directory.glob("*.pt"))
            ]
        },
    )
    report = {
        "status": "completed",
        "spec_hash": c.spec_hash,
        "network_seed": p["network_seed"],
        "training_cohorts": consumed,
        "updates": agent.updates,
        "training_environment_steps": consumed * q["T"],
        "validation_cohorts": len(seeds),
        "V_star_USD": target["V_star_USD"],
        "initial": curve[0],
        "final": curve[-1],
        "first_observed_0_98_cohorts": next(
            (r["training_cohorts"] for r in curve if r["meets_0_98"]), None
        ),
        "initial_already_meets_0_98": curve[0]["meets_0_98"],
        "final_point_pass": curve[-1]["meets_0_98"],
        "exact_curve_sha256": file_hash(context.directory / "exact_curve.json"),
        "updates_sha256": file_hash(context.directory / "updates.jsonl"),
        "validation_pairs_sha256": file_hash(
            context.directory / "validation_pairs.jsonl"
        ),
        "last_checkpoint_sha256": file_hash(checkpoint),
        "checkpoint_manifest_sha256": file_hash(
            context.directory / "checkpoint_manifest.json"
        ),
        "validation_mean_J": float(np.mean([r["PPO_J"] for r in pairs])),
        "all_five_seed_criterion": None,
        "full_V7_pass": None,
        "scientific_RL_budget": None,
        "test_seeds_opened": False,
    }
    atomic_json(context.directory / "single_leg_learning.json", report)
    return {
        **report,
        "single_leg_learning_sha256": file_hash(
            context.directory / "single_leg_learning.json"
        ),
    }
