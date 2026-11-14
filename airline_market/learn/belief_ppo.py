"""Actual recurrent PPO with observation-derived exact small-network belief features.

No existing trainer/adapter is modified. Native coefficient actions and PPO
updates are reused. Hub particle beliefs and P2 tuning budgets are not substituted.
"""

import shutil
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from airline_market.belief.events import AllOf, LegInventoryAtMost
from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.hashing import schema_hash
from airline_market.core.rng import StreamRegistry
from airline_market.learn.adapters import RecurrentEnvAdapter
from airline_market.learn.ppo import collect, load_checkpoint, save_checkpoint
from airline_market.learn.structured_train import seed_in_split
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.vendor.ride_response.recurrent.ppo import (
    RecurrentPPOAgent,
    RecurrentPPOConfig,
)


class ExactBeliefAdapter(RecurrentEnvAdapter):
    """Own the registered-model environment and bind event meanings to its schema."""

    def __init__(self, compiled, focal, arm, events=(), *, critic_privileged=False):
        if compiled.spec.demand.mode != "single_request_bernoulli":
            raise ValueError(
                "this precursor uses exact small Bernoulli beliefs; hub filtering remains separate"
            )
        normalized = tuple(
            (str(e["id"]), str(e["rival"]), tuple(e["legs"]), int(e["k"]))
            for e in events
        )
        if len({e[0] for e in normalized}) != len(normalized):
            raise ValueError("joint-event IDs must be unique")
        for name, rival, legs, k in normalized:
            if (
                rival == focal
                or not legs
                or any(
                    compiled.spec.legs[compiled.leg_index[leg]].carrier != rival
                    for leg in legs
                )
            ):
                raise ValueError("joint-event legs must belong to the declared rival")
        self.event_contract = normalized
        joint = tuple(
            AllOf(tuple(LegInventoryAtMost(rival, leg, k) for leg in legs))
            for _, rival, legs, k in normalized
        )
        env = FocalCarrierEnv(compiled, focal, arm, critic_privileged=critic_privileged)
        super().__init__(env, belief_features=True, joint_events=joint)
        self.policy_class = "coefficient_recurrent_belief_features_exact_v1"
        start = self.public_dim - len(normalized)
        for k, event in enumerate(normalized):
            self._names[start + k] = "belief/event/" + event[0]

    def contract(self):
        return {
            "backend": "ExactBelief with registered spec controllers; no evaluation state input",
            "features": "existing per-rival-leg mean,p0,p1(exactlyone), plus named all-ofstock<=k events",
            "events": [
                {"id": name, "rival": rival, "legs": list(legs), "k": k}
                for name, rival, legs, k in self.event_contract
            ],
            "policy_class": self.policy_class,
        }

    def schema_hash(self):
        return schema_hash(
            (
                self.names(),
                self.policy_class,
                self.event_contract,
                "public identity mask plus belief statistics",
            )
        )


def adapter(compiled, cfg):
    return ExactBeliefAdapter(
        compiled,
        cfg.get("focal", "A"),
        cfg.get("arm", "direct"),
        cfg.get("joint_events", ()),
        critic_privileged=cfg.get("critic_privileged", False),
    )


def initial_agent(compiled, cfg):
    network_seed = cfg.get("network_seed", 0)
    if not 0 <= network_seed <= 9:
        raise ValueError("registered network initialization seeds are0–9")
    train = adapter(compiled, cfg)
    identity = tuple(
        k
        for k, name in enumerate(train.names())
        if name == "clock"
        or name.startswith(("inventory/", "exo/", "belief/"))
        or name.endswith(("/available", "/missing"))
    )
    hyper = {
        "gamma": 1.0,
        **cfg.get("ppo", {}),
        "observation_identity_indices": identity,
        "seed": StreamRegistry(network_seed, "train/flat/init").child_seed() % 2**31,
    }
    with torch.random.fork_rng(devices=[]):
        agent = RecurrentPPOAgent(
            len(train.names()), train.action_transform, RecurrentPPOConfig(**hyper)
        )
    train.configure_actor(agent)
    return train, agent


def training_contract(compiled, cfg, train):
    return {
        "spec_hash": compiled.spec_hash,
        "schema_hash": train.schema_hash(),
        "network_seed": cfg.get("network_seed", 0),
        "ppo": cfg.get("ppo", {}),
        "focal": cfg.get("focal", "A"),
        "arm": cfg.get("arm", "direct"),
        "train_seed_start": cfg.get("train_seed_start", 100000),
        "validation_seed_start": cfg.get("validation_seed_start", 200000),
        "n_validation": cfg.get("n_validation", 64),
        "cohorts_per_update": cfg["cohorts_per_update"],
        "validate_every_cohorts": cfg.get("validate_every_cohorts", 2000),
        "belief_contract": train.contract(),
    }


def atomic_checkpoint(agent, path, train, metadata):
    temporary = path.with_suffix(".tmp.pt")
    save_checkpoint(agent, temporary, train, metadata)
    temporary.replace(path)


def train_belief(compiled, cfg, directory, progress):
    """Run native PPO updates with exact, observation-derived belief actor inputs."""
    train, agent = initial_agent(compiled, cfg)
    validation = adapter(compiled, cfg)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    contract = training_contract(compiled, cfg, train)
    train_start, validation_start = (
        cfg.get("train_seed_start", 100000),
        cfg.get("validation_seed_start", 200000),
    )
    n_train, batch = cfg["n_train"], cfg["cohorts_per_update"]
    if n_train < 1 or batch < 1:
        raise ValueError("training budget and batch size must be positive")
    validation_seeds = range(
        validation_start, validation_start + cfg.get("n_validation", 64)
    )
    for seed in [train_start, train_start + n_train - 1]:
        seed_in_split(seed, True)
    for seed in validation_seeds:
        seed_in_split(seed, False)
    history, validations, consumed, best = [], [], 0, None
    cadence = cfg.get("validate_every_cohorts", 2000)
    if cadence < 1:
        raise ValueError("validation cadence must be positive")
    next_validation = cadence
    if cfg.get("resume_file"):
        resume = Path(cfg["resume_file"])
        if file_hash(resume) != cfg["resume_sha256"]:
            raise ValueError("changed resume checkpoint")
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        if metadata["training_contract"] != contract:
            raise ValueError("changed belief training contract")
        with torch.random.fork_rng(devices=[]):
            agent = load_checkpoint(resume, train, restore_rng=True)
        history, validations = (
            deepcopy(metadata["history"]),
            deepcopy(metadata["validation"]),
        )
        consumed = metadata["consumed"]
        if consumed > n_train:
            raise ValueError("resume is beyond the requested training budget")
        best = metadata["best_validation"]
        next_validation = (consumed // cadence + 1) * cadence
        prior_best = resume.with_name("best.pt")
        prior_payload = torch.load(prior_best, map_location="cpu", weights_only=False)
        if prior_payload["metadata"]["training_contract"] != contract:
            raise ValueError("changed selected-checkpoint contract")
        if prior_best.resolve() != (directory / "best.pt").resolve():
            shutil.copy2(prior_best, directory / "best.pt")
    started = time.perf_counter()
    while consumed < n_train:
        rollout_started = time.perf_counter()
        returns = []
        for k in range(min(batch, n_train - consumed)):
            seed = train_start + consumed + k
            progress(
                {
                    "stage": "training",
                    "cohorts_completed": consumed + k,
                    "cohorts_total": n_train,
                    "update": agent.updates + 1,
                }
            )
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(
                    StreamRegistry(seed, "train/flat/policy").child_seed() % 2**31
                )
                episode, logp, value = collect(train, agent, seed)
            agent.add_episode(
                episode, actions_are_physical=False, behavior_log_probabilities=logp
            )
            returns.append(value)
        rollout_seconds = time.perf_counter() - rollout_started
        update_started = time.perf_counter()
        stats = agent.update()
        update_seconds = time.perf_counter() - update_started
        consumed += len(returns)
        history.append(
            {
                "cohorts": consumed,
                "return_mean": float(np.mean(returns)),
                "rollout_seconds": rollout_seconds,
                "update_seconds": update_seconds,
                **stats,
            }
        )
        selected = False
        if consumed >= next_validation or consumed == n_train:
            row = {"cohorts": consumed}
            for deterministic, name in [(True, "deterministic"), (False, "sampled")]:
                values = []
                for seed in validation_seeds:
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(
                            StreamRegistry(seed, "validation/flat/policy").child_seed()
                            % 2**31
                        )
                        values.append(
                            collect(
                                validation, agent, seed, deterministic=deterministic
                            )[2]
                        )
                row[name] = float(np.mean(values))
                row[name + "_values"] = values
            validations.append(row)
            selected = best is None or row["deterministic"] > best
            best = row["deterministic"] if selected else best
            next_validation = (consumed // cadence + 1) * cadence
        metadata = {
            "learner": "ppo_b_exact",
            "consumed": consumed,
            "history": history,
            "validation": validations,
            "test_seeds_opened": False,
            "best_validation": best,
            "training_contract": contract,
            "belief_contract": train.contract(),
            "normalization": "original public identity indices plus all declared belief statistics",
        }
        atomic_checkpoint(agent, directory / "last.pt", train, metadata)
        if selected:
            atomic_checkpoint(agent, directory / "best.pt", train, metadata)
        atomic_json(directory / "learning.json", metadata)
    return {
        "status": "completed",
        "learner": "ppo_b_exact",
        "policy_class": train.policy_class,
        "cohorts": consumed,
        "updates": agent.updates,
        "best_validation": best,
        "seconds": time.perf_counter() - started,
        "rollout_seconds": sum(x["rollout_seconds"] for x in history),
        "update_seconds": sum(x["update_seconds"] for x in history),
        "behavior_logp_checks": agent.behavior_log_probability_checks,
        "max_behavior_logp_error": agent.max_behavior_log_probability_error,
        "learning_signal": "not established by implementation/throughput runs",
        "test_seeds_opened": False,
    }
