"""Training-only network credit, delayed event labels and structured checkpoints."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import torch

from ..campaign.artifacts import atomic_json
from ..core.rng import StreamRegistry
from ..core.hashing import schema_hash
from ..pomdp.env import FocalCarrierEnv
from ..sim.ledger import inventory
from ..vendor.ride_response.recurrent.common import EpisodeBuilder
from ..vendor.ride_response.recurrent.ppo import RecurrentPPOConfig
from .structured_runtime import StructuredAdapter, StructuredPPOAgent


def seed_in_split(seed, training):
    """Reject final/calibration seeds in this train/validation-only runner."""
    low, high = (100000, 200000) if training else (200000, 300000)
    if not low <= seed < high:
        raise ValueError("structured rollout seed is outside its registered split")


def joint_label(env, legs):
    """Read a realized training label separately from the actor's observation path."""
    values = []
    for name in legs:
        leg = env.compiled.leg_index[name]
        owner = int(env.compiled.leg_carrier[leg])
        local = env.compiled.legs_of[owner].index(leg)
        values.append(inventory(env.engine.world.carriers[owner], env.compiled, owner)[local] <= 1)
    return float(all(values))


def branch_credit(adapter, agent, physical, state, previous, *, seed, K=4, horizon=5,
                  coordinates=None, previous_coordinates=None):
    """Compare latent actions on the same training tape and future policy-noise stream."""
    rng = StreamRegistry(seed, "train/structured/credit/alternatives").generator()
    if K < 2 or horizon is not None and horizon < 1:
        raise ValueError("credit requires K>=2 and a positive horizon")
    hold = previous if previous is not None else agent.action_transform.coordinates_to_physical(
        np.zeros(agent.action_dim))
    alternatives = [physical, hold] + [agent.action_transform.coordinates_to_physical(
        rng.normal(size=agent.action_dim)) for _ in range(K - 2)]
    coordinate_rows = [coordinates, previous_coordinates] + [None] * (K - 2)
    returns = []
    for alternative, alternative_coordinates in zip(alternatives, coordinate_rows):
        branch = adapter.env.branch()
        hidden = deepcopy(state)
        after, reward, done, _ = branch.step(adapter.price_action(alternative, adapter.obs))
        total = reward
        last_action = alternative
        last_coordinates = alternative_coordinates
        stop = min(adapter.env.compiled.spec.horizon.n_periods,
                   adapter.obs.t + horizon) if horizon is not None else adapter.env.compiled.spec.horizon.n_periods
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(StreamRegistry(seed, "train/structured/credit/follow").child_seed() % 2**31)
            while not done and branch.engine.world.t < stop:
                action, hidden, info = agent.act(adapter.encode(after), hidden=hidden,
                                                previous_action=last_action,
                                                previous_coordinates=last_coordinates,
                                                previous_reward=reward)
                after, reward, done, _ = branch.step(adapter.price_action(action, after))
                total += reward
                last_action = action
                last_coordinates = info["policy_coordinates"]
        returns.append(total)
    return {"centered": float(returns[0] - np.mean(returns)), "returns": returns,
            "hold_relative": float(returns[0] - returns[1])}


def collect_structured(adapter, agent, seed, *, training, deterministic=False,
                       credit=None, event_legs=()):
    """Keep hidden labels and counterfactual branching strictly in training collection."""
    seed_in_split(seed, training)
    if not training and (credit or event_legs):
        raise ValueError("evaluation cannot request training labels or branch credit")
    x = adapter.reset(seed)
    builder = EpisodeBuilder(x)
    state, previous, reward, done = None, None, 0.0, False
    previous_coordinates = None
    logp, credits, truths, branch_returns = [], [], [], []
    total = 0.0
    credit_rng = StreamRegistry(seed, "train/structured/credit/select").generator()
    policy_seed = StreamRegistry(seed, "train/structured/policy" if training
                                 else "validation/structured/policy").child_seed() % 2**31
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(policy_seed)
        while not done:
            if event_legs:
                truths.append(joint_label(adapter.env, event_legs))
            action, state, info = agent.act(x, hidden=state, previous_action=previous,
                                           previous_coordinates=previous_coordinates,
                                           previous_reward=reward, deterministic=deterministic)
            advantage = float("nan")
            if credit and credit_rng.random() < credit.get("fraction", 0.1):
                result = branch_credit(adapter, agent, action, state, previous,
                                       seed=int(credit_rng.integers(2**31)),
                                       K=credit.get("K", 4), horizon=credit.get("H", 5),
                                       coordinates=info["policy_coordinates"],
                                       previous_coordinates=previous_coordinates)
                advantage = result["centered"] * agent.config.reward_scale
                branch_returns.append(result["returns"])
            after, reward, done, _ = adapter.step(action)
            builder.append(info["policy_coordinates"], reward, after, done)
            logp.append(info["log_probability"])
            credits.append(advantage)
            total += reward
            previous, x = action, after
            previous_coordinates = info["policy_coordinates"]
    episode = builder.finish()
    if credit:
        episode.metadata.update(counterfactual_advantages=credits,
                                counterfactual_mix=credit.get("mix", 0.5),
                                branch_returns=branch_returns)
    if event_legs:
        # Target at t+1 under the realized training policy. Terminal inventory
        # supplies the final label; no no-further-sales projection is substituted.
        truths.append(joint_label(adapter.env, event_legs))
        episode.metadata["joint_event_labels"] = np.asarray(truths[1:], dtype=np.float32)[:, None]
    return episode, logp, total


def save_structured(agent, path, adapter, metadata):
    """Atomically save structured weights, optimizers, schema and sampling state."""
    path = Path(path)
    payload = {"format": "airline-structured-ppo/1", "config": asdict(agent.config),
               "spec_hash": adapter.env.compiled.spec_hash,
               "schema_hash": adapter.schema_hash(), "layout": asdict(adapter.layout),
               "actor": agent.actor.state_dict(), "value": agent.value.state_dict(),
               "actor_optimizer": agent.actor_optimizer.state_dict(),
               "value_optimizer": agent.value_optimizer.state_dict(),
               "normalizer": agent.observation_normalizer.state_dict(),
               "numpy_rng": agent.rng.bit_generator.state, "updates": agent.updates,
               "behavior_logp_checks": agent.behavior_log_probability_checks,
               "max_behavior_logp_error": agent.max_behavior_log_probability_error,
               "auxiliary": agent.actor.aux_head is not None, "aux_weight": agent.aux_weight,
               "metadata": metadata}
    temporary = path.with_suffix(".tmp.pt")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_structured(path, adapter):
    """Restore only matching structured information and physical-model contracts."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (payload["format"] != "airline-structured-ppo/1"
            or payload["spec_hash"] != adapter.env.compiled.spec_hash
            or payload["schema_hash"] != adapter.schema_hash()):
        raise ValueError("structured checkpoint model/schema mismatch")
    with torch.random.fork_rng(devices=[]):
        agent = StructuredPPOAgent(adapter, RecurrentPPOConfig(**payload["config"]),
                                   auxiliary=payload["auxiliary"], aux_weight=payload["aux_weight"])
    for name in ("actor", "value", "actor_optimizer", "value_optimizer"):
        getattr(agent, name).load_state_dict(payload[name])
    agent.observation_normalizer.load_state_dict(payload["normalizer"])
    agent.rng.bit_generator.state = payload["numpy_rng"]
    agent.updates = payload["updates"]
    agent.behavior_log_probability_checks = payload["behavior_logp_checks"]
    agent.max_behavior_log_probability_error = payload["max_behavior_logp_error"]
    return agent, payload["metadata"]


def train_structured(compiled, config, out, progress=None):
    """Run actual SRL/SRL-C/SRL-CA updates; select only on validation cohorts."""
    cfg = deepcopy(config)
    variant = cfg["learner"]
    if variant not in ("srl", "srl_c", "srl_ca"):
        raise ValueError("structured training requires srl, srl_c, or srl_ca")
    n_train = int(cfg["n_train"])
    seed_start = int(cfg.get("train_seed_start", 100000))
    validation = list(range(int(cfg.get("validation_seed_start", 200000)),
                            int(cfg.get("validation_seed_start", 200000)) + int(cfg.get("n_validation", 64))))
    seed_in_split(seed_start, True)
    seed_in_split(seed_start + n_train - 1, True)
    for seed in validation:
        seed_in_split(seed, False)
    focal, arm = cfg.get("focal", compiled.spec.carriers[0].code), cfg.get("arm", "direct")
    adapter = StructuredAdapter(FocalCarrierEnv(compiled, focal, arm))
    evaluation = StructuredAdapter(FocalCarrierEnv(compiled, focal, arm))
    root = int(cfg.get("network_seed", 0))
    init_seed = StreamRegistry(root, "train/structured/init").child_seed() % 2**31
    identity = tuple(k for k, name in enumerate(adapter.names())
                     if k >= adapter.layout.global_dim or name == "clock"
                     or name.startswith("exo/") or name.endswith(("/available", "/missing")))
    hyper = {"gamma": 1.0, **cfg.get("ppo", {}), "seed": init_seed,
             "observation_identity_indices": identity}
    if hyper["gamma"] != 1.0:
        raise ValueError("structured cohort credit uses undiscounted returns")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng(devices=[]):
        agent = StructuredPPOAgent(adapter, RecurrentPPOConfig(**hyper),
                                   auxiliary=variant == "srl_ca", aux_weight=cfg.get("aux_weight", 0.3))
    credit = cfg.get("credit", {"fraction": 0.1, "K": 4, "H": 5, "mix": 0.5}) if variant != "srl" else None
    event_legs = tuple(cfg.get("event_legs", ())) if variant == "srl_ca" else ()
    if variant == "srl_ca" and not event_legs:
        raise ValueError("SRL-CA requires explicit joint-event rival legs")
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    history, validations, consumed, best = [], [], 0, None
    fingerprint = schema_hash((compiled.spec_hash, adapter.schema_hash(),
                               {k: v for k, v in cfg.items() if k != "resume_checkpoint"}))
    if cfg.get("resume_checkpoint"):
        agent, metadata = load_structured(cfg["resume_checkpoint"], adapter)
        if metadata["training_fingerprint"] != fingerprint:
            raise ValueError("resumed structured training configuration changed")
        consumed, history, validations, best = (metadata[k] for k in ("consumed", "history", "validation", "best"))
    batch = int(cfg.get("cohorts_per_update", 64))
    cadence = int(cfg.get("validate_every_cohorts", 2000))
    next_validation = (consumed // cadence + 1) * cadence
    started = time.perf_counter()
    while consumed < n_train:
        roll_started = time.perf_counter()
        returns = []
        for k in range(min(batch, n_train - consumed)):
            seed = seed_start + consumed + k
            if progress:
                progress({"stage": "training", "cohorts_completed": consumed + k,
                          "cohorts_total": n_train, "update": agent.updates + 1})
            episode, logp, total = collect_structured(adapter, agent, seed, training=True,
                                                      credit=credit, event_legs=event_legs)
            agent.add_episode(episode, actions_are_physical=False, behavior_log_probabilities=logp)
            returns.append(total)
        roll_seconds = time.perf_counter() - roll_started
        update_started = time.perf_counter()
        stats = agent.update()
        update_seconds = time.perf_counter() - update_started
        consumed += len(returns)
        history.append({"cohorts": consumed, "return_mean": float(np.mean(returns)),
                        "rollout_seconds": roll_seconds, "update_seconds": update_seconds,
                        **{k: float(v) for k, v in stats.items()}})
        selected = False
        if consumed >= next_validation or consumed == n_train:
            row = {"cohorts": consumed}
            for deterministic, name in [(True, "deterministic"), (False, "sampled")]:
                values = [collect_structured(evaluation, agent, seed, training=False,
                                             deterministic=deterministic)[2] for seed in validation]
                row[name] = float(np.mean(values))
                row[name + "_values"] = values
            validations.append(row)
            selected = best is None or row["deterministic"] > best
            best = row["deterministic"] if selected else best
            next_validation = (consumed // cadence + 1) * cadence
        metadata = {"training_fingerprint": fingerprint, "consumed": consumed,
                    "history": history, "validation": validations, "best": best,
                    "variant": variant, "policy_class": adapter.policy_class,
                    "event": {"legs": event_legs, "threshold": 1, "horizon": 1},
                    "test_seeds_opened": False}
        save_structured(agent, directory / "last.pt", adapter, metadata)
        if selected:
            save_structured(agent, directory / "best.pt", adapter, metadata)
        atomic_json(directory / "learning.json", metadata)
    return {"status": "completed", "learner": variant, "policy_class": adapter.policy_class,
            "cohorts": consumed, "updates": agent.updates, "best_validation": best,
            "seconds": time.perf_counter() - started,
            "rollout_seconds": sum(x["rollout_seconds"] for x in history),
            "update_seconds": sum(x["update_seconds"] for x in history),
            "behavior_logp_checks": agent.behavior_log_probability_checks,
            "max_behavior_logp_error": agent.max_behavior_log_probability_error,
            "event_labels": "realized rival both-leg inventory<=1 at t+1; training only" if event_legs else None,
            "learning_signal": "not established by implementation/throughput runs",
            "test_seeds_opened": False}
