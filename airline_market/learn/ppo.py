"""Recurrent PPO training on departure cohorts, with versioned checkpoints."""

from dataclasses import dataclass, asdict
from pathlib import Path
import json, copy
import numpy as np
import torch
from ..spec import load_spec, compile_market
from ..core.hashing import jsonable
from ..core.rng import StreamRegistry
from ..core.registry import append_manifest
from ..pomdp.env import FocalCarrierEnv
from ..eval.seeds import SeedPlan
from ..vendor.ride_response.recurrent.ppo import RecurrentPPOAgent, RecurrentPPOConfig
from ..vendor.ride_response.recurrent.common import EpisodeBuilder
from .adapters import RecurrentEnvAdapter


@dataclass
class TrainResult:
    """
    Selected checkpoint and declared training/validation learning curves (Eqs. 18–19,
    26–27).
    """

    checkpoint: str
    history: list
    validation: list
    policy_class: str
    selected_update: int


def collect(adapter, agent, seed, *, deterministic=False, credit=None):
    """
    Collect one cohort episode, preserving native actions and behavior probabilities
    (Eqs. 18–19, 26–27).
    """
    if credit and adapter.belief_features:
        raise ValueError(
            "counterfactual credit requires a branch belief adapter; belief-feature actors cannot receive zero-filled features"
        )
    x = adapter.reset(seed)
    builder = EpisodeBuilder(x)
    state = None
    previous = None
    previous_coordinates = None
    reward = 0.0
    done = False
    logp = []
    total = 0.0
    cf = []
    credit_rng = StreamRegistry(seed, "train/credit").generator()
    while not done:
        action, state, info = agent.act(
            x,
            previous_action=previous,
            previous_coordinates=previous_coordinates,
            previous_reward=reward,
            hidden=state,
            deterministic=deterministic,
        )
        advantage = float("nan")
        if credit and credit_rng.random() < credit.get("fraction", 0.1):
            from .credit import counterfactual_credit
            from ..core.types import CoefficientAction

            hold = (
                np.asarray(previous)
                if previous is not None
                else np.mean(np.asarray(adapter.env.pricing.theta_bounds), axis=1)
            )
            alternatives = [action, hold] + [
                adapter.action_transform.coordinates_to_physical(
                    credit_rng.normal(size=len(action))
                )
                for _ in range(max(0, credit.get("K", 4) - 2))
            ]

            def factory():
                """
                Create isolated branch-policy recurrence for a common-tape credit
                rollout (Eqs. 18–19, 26–27).
                """
                h = copy.deepcopy(state)

                def policy(obs, previous_action, previous_reward, rng):
                    """
                    Return an observation-to-action selector using the declared planning
                    state (Eqs. 18–19, 26–27).
                    """
                    nonlocal h
                    # Public vector only; the critic-only columns are inert to the actor.
                    values = list(adapter.vectorizer(obs))
                    values.extend([0.0] * (len(adapter.names()) - len(values)))
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(int(rng.integers(2**31)))
                        a, h, _ = agent.act(
                            values,
                            hidden=h,
                            previous_action=previous_action.theta
                            if isinstance(previous_action, CoefficientAction)
                            else previous_action,
                            previous_reward=previous_reward,
                            deterministic=False,
                        )
                    return a

                return policy

            result = counterfactual_credit(
                adapter.env,
                alternatives,
                factory,
                horizon=credit.get("H"),
                seed=int(credit_rng.integers(2**31)),
            )
            advantage = float(result["centered"][0]) * agent.config.reward_scale
        cf.append(advantage)
        after, reward, done, _ = adapter.step(action)
        builder.append(info["policy_coordinates"], reward, after, done)
        logp.append(info["log_probability"])
        previous = action
        previous_coordinates = info["policy_coordinates"]
        x = after
        total += reward
    episode = builder.finish()
    if credit:
        episode.metadata.update(
            counterfactual_advantages=cf, counterfactual_mix=credit.get("mix", 1.0)
        )
    return episode, logp, total


def save_checkpoint(agent, path, adapter, extra=None):
    """
    Save model, schema/spec identity, RNG state and selection metadata (Eqs. 18–19,
    26–27).
    """
    metadata = {
        "schema_hash": adapter.schema_hash(),
        "observation_names": adapter.names(),
        "spec_hash": adapter.env.compiled.spec_hash,
        "arm": adapter.env.arm.name,
        "policy_class": adapter.policy_class,
        "public_dim": adapter.public_dim,
        **(extra or {}),
    }
    agent.save(path, metadata=metadata)
    payload = torch.load(path, weights_only=False)
    payload["torch_rng"] = torch.get_rng_state()
    payload["numpy_rng"] = agent.rng.bit_generator.state
    torch.save(payload, path)


def load_checkpoint(path, adapter, *, restore_rng=False):
    """
    Load only a schema/spec-compatible policy and optionally restore RNG state (Eqs.
    18–19, 26–27).
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    if (
        metadata["schema_hash"] != adapter.schema_hash()
        or metadata["spec_hash"] != adapter.env.compiled.spec_hash
    ):
        raise ValueError("checkpoint observation schema or market spec hash mismatch")
    agent = RecurrentPPOAgent.load(path)
    adapter.configure_actor(agent)
    if restore_rng:
        torch.set_rng_state(payload["torch_rng"])
        agent.rng.bit_generator.state = payload["numpy_rng"]
    return agent


def train_ppo(config):
    """
    Train on cohort tapes, select on validation, then open the held-out test split (Eqs.
    18–19, 26–27).
    """
    cfg = copy.deepcopy(config)
    c = compile_market(load_spec(cfg["spec"]))
    seeds = SeedPlan.from_config(cfg["seeds"])
    from ..eval.sealing import validate_seed_release

    validate_seed_release(seeds, cfg.get("phase", "development"))
    focal = cfg.get("focal", c.spec.carriers[0].code)
    arm = cfg.get("arm", "direct")

    def adapter():
        """
        Construct an environment adapter with the declared information and critic scope
        (Eqs. 18–19, 26–27).
        """
        from ..agents import make_controller
        from ..spec.model import ControllerAssignment
        from ..game.payoffs import PolicyRef, controller

        rivals = {
            i: controller(PolicyRef(**r), i, 0)
            if "kind" in r
            else make_controller(ControllerAssignment(carrier=i, **r))
            for i, r in cfg.get("rival_controllers", {}).items()
        }
        return RecurrentEnvAdapter(
            FocalCarrierEnv(
                c,
                focal,
                arm,
                rivals,
                critic_privileged=cfg.get("critic_privileged", False),
            ),
            belief_features=cfg.get("belief_features", False),
        )

    train = adapter()
    validation_adapter = adapter()
    root = int(cfg.get("seed", 0))
    train_seed = StreamRegistry(root, "train").child_seed() % 2**31
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(cfg.get("torch_threads", 1)))
    identity = tuple(
        k
        for k, name in enumerate(train.names())
        if name == "clock"
        or name.startswith(("inventory/", "exo/"))
        or name.endswith(("/available", "/missing"))
    )
    hyper = {
        "gamma": 1.0,
        "observation_identity_indices": identity,
        **cfg.get("ppo", {}),
        "seed": train_seed,
    }
    agent = RecurrentPPOAgent(
        len(train.names()), train.action_transform, RecurrentPPOConfig(**hyper)
    )
    train.configure_actor(agent)
    if isinstance(cfg.get("credit"), str):
        kind = cfg["credit"]
        if kind not in ("cf", "mix"):
            raise ValueError("credit must be cf, mix, or a parameter mapping")
        cfg["credit"] = {
            **cfg.get("credit_params", {}),
            "mix": 1.0 if kind == "cf" else cfg.get("credit_weight", 0.5),
        }
    if cfg.get("credit") and (cfg.get("belief_features") or hyper["gamma"] != 1.0):
        raise ValueError(
            "counterfactual training currently uses public recurrent actors and undiscounted cohort returns"
        )
    history = []
    validation = []
    episodes = int(cfg.get("episodes_per_update", 4))
    updates = int(cfg.get("updates", 10))
    cadence = int(cfg.get("validate_every", 1))
    out = Path(cfg.get("out", "results/ppo"))
    out.mkdir(parents=True, exist_ok=True)
    best = -np.inf
    selected = 0
    checkpoint = out / "best.pt"
    rule = cfg.get("selection_rule", "deterministic")
    if rule not in ("deterministic", "sampled"):
        raise ValueError("selection_rule must be deterministic or sampled")
    if not seeds.validation:
        raise ValueError("PPO checkpoint selection needs validation seeds")
    for update in range(updates):
        returns = []
        for k in range(episodes):
            seed = seeds.train[(update * episodes + k) % len(seeds.train)]
            episode, logp, value = collect(train, agent, seed, credit=cfg.get("credit"))
            agent.add_episode(
                episode, actions_are_physical=False, behavior_log_probabilities=logp
            )
            returns.append(value)
        stats = agent.update()
        history.append(
            {"update": update + 1, "return_mean": float(np.mean(returns)), **stats}
        )
        if (update + 1) % cadence == 0 or update + 1 == updates:
            rng = torch.get_rng_state()
            row = {"update": update + 1}
            for deterministic, label in [(True, "deterministic"), (False, "sampled")]:
                torch.manual_seed(
                    StreamRegistry(root, "validation").child_seed(int(deterministic))
                    % 2**31
                )
                row[label] = float(
                    np.mean(
                        [
                            collect(
                                validation_adapter,
                                agent,
                                s,
                                deterministic=deterministic,
                            )[2]
                            for s in seeds.validation
                        ]
                    )
                )
            torch.set_rng_state(rng)
            validation.append(row)
            if row[rule] > best:
                best = row[rule]
                selected = update + 1
                save_checkpoint(
                    agent,
                    checkpoint,
                    train,
                    {
                        "selection_rule": rule,
                        "return_statistics": row,
                        "training_config": cfg,
                    },
                )
    save_checkpoint(agent, out / "last.pt", train, {"return_statistics": history[-1]})
    # Test seeds are opened only after checkpoint selection.
    chosen = load_checkpoint(checkpoint, validation_adapter)
    test = {
        "deterministic": [
            collect(validation_adapter, chosen, s, deterministic=True)[2]
            for s in seeds.test
        ]
    }
    torch.manual_seed(StreamRegistry(root, "test_policy").child_seed() % 2**31)
    test["sampled"] = [collect(validation_adapter, chosen, s)[2] for s in seeds.test]
    (out / "learning.json").write_text(
        json.dumps(
            jsonable(
                {
                    "history": history,
                    "validation": validation,
                    "test": test,
                    "performance_gate": "not certified by this budget",
                }
            ),
            indent=2,
        )
        + "\n"
    )
    append_manifest(
        str(out.parent),
        {
            "kind": "ppo",
            "archive_path": str(checkpoint.resolve()),
            "spec_hash": c.spec_hash,
            "schema_hash": train.schema_hash(),
        },
    )
    return TrainResult(
        str(checkpoint), history, validation, train.policy_class, selected
    )
