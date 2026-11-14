"""Opt-in cohort curriculum and domain sampling for native recurrent PPO.

Saved per-domain meta-strategies are frozen inputs. The adaptive rival plans
against that declared reference population, not the unknown current PPO policy.
This engineering path does not supply calibrated TR2 members or a TR5 monitor.
"""

import shutil
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from airline_market.agents.bidprice import DLPBidPriceController
from airline_market.agents.matching import CompetitorMatchingController
from airline_market.agents.rule import ThresholdLinearRuleController
from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.campaign.runner import completed_attempt
from airline_market.core.hashing import schema_hash
from airline_market.core.rng import StreamRegistry
from airline_market.game.mpc_population import EpisodeHistoryMixture
from airline_market.game.type_belief import HistoryController, HistoryPolicy
from airline_market.game.type_mpc import EpisodeTypeMPCController, TypeMPCConfig
from airline_market.learn.adapters import RecurrentEnvAdapter
from airline_market.learn.ppo import collect, load_checkpoint, save_checkpoint
from airline_market.learn.structured_train import seed_in_split
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.spec import compile_market, load_spec
from airline_market.vendor.ride_response.recurrent.ppo import (
    RecurrentPPOAgent,
    RecurrentPPOConfig,
)


def load_domains(manifest):
    """Reuse sealed game outputs without solving or refitting any source game."""
    import json

    domains = []
    for member in manifest:
        directory = Path(member["directory"])
        required = {
            "spec.json",
            "policies.json",
            "config.json",
            "results.json",
            "COMPLETE.json",
        }
        if not required <= set(member["hashes"]):
            raise ValueError("curriculum source manifest lacks required hashes")
        for name, expected in member["hashes"].items():
            if file_hash(directory / name) != expected:
                raise ValueError("changed frozen curriculum source")
        config = json.loads((directory / "config.json").read_text())
        found = completed_attempt(directory.parent, config["config_hash"])
        if found is None or found[0] != directory:
            raise ValueError("curriculum meta source lacks matching receipt")
        _, result = found
        if not result.get("certified") or result["status"] != "converged":
            raise ValueError("precursor requires completed finite-class meta source")
        compiled = compile_market(load_spec(directory / "spec.json"))
        if compiled.spec_hash != result["spec_hash"]:
            raise ValueError("curriculum source model identity differs")
        policies = json.loads((directory / "policies.json").read_text())
        if file_hash(directory / "policies.json") != result["policies_sha256"]:
            raise ValueError("curriculum source policies differ")
        carriers = tuple(c.code for c in compiled.spec.carriers)
        weights = result["history"][-1]["mixtures"]
        domains.append(
            {
                "id": member["id"],
                "role": member["role"],
                "compiled": compiled,
                "populations": {
                    c: tuple(HistoryPolicy.from_dict(p) for p in policies[c])
                    for c in carriers
                },
                "mixtures": {c: tuple(w) for c, w in zip(carriers, weights)},
                "source": deepcopy(member),
            }
        )
    validate_domains(domains)
    return domains


def validate_domains(domains):
    if not domains or len({d["id"] for d in domains}) != len(domains):
        raise ValueError("unique nonempty domain members required")
    if not any(d["role"] == "train" for d in domains):
        raise ValueError("training members required")
    for d in domains:
        if d["role"] not in ("train", "heldout"):
            raise ValueError("domain role must be train or heldout")
        c = d["compiled"]
        if len(c.spec.carriers) != 2:
            raise ValueError("this curriculum path requires two carriers")
        if c.spec.demand.mode != "single_request_bernoulli":
            raise ValueError("adaptive curriculum precursor requires Bernoulli domains")
        for carrier in c.spec.carriers:
            code = carrier.code
            grid = next(p.theta_grid for p in c.spec.pricing if p.carrier == code)
            policies, weights = d["populations"][code], d["mixtures"][code]
            if (
                not policies
                or len(policies) != len(weights)
                or any(p.grid != grid for p in policies)
                or len({p.arm for p in policies}) != 1
                or not np.isfinite(weights).all()
                or min(weights) < 0
                or abs(sum(weights) - 1) > 1e-12
            ):
                raise ValueError("domain policy/meta contract invalid")


def curriculum_stage(index, total):
    """Zero-based cohort stages, independent of update/batch boundaries."""
    if total < 1 or not 0 <= index < total:
        raise ValueError("invalid curriculum cohort")
    return 0 if 10 * index < 4 * total else 1 if 10 * index < 7 * total else 2


def choose_domain(domains, seed, training):
    seed_in_split(seed, training)
    eligible = [d for d in domains if d["role"] == "train"]
    family = "train" if training else "validation"
    rng = StreamRegistry(seed, family + "/TR/domain-v1").generator()
    return eligible[int(rng.integers(len(eligible)))]


def rival_controller(domain, focal, stage, seed, mpc):
    """Construct one real rival; only context and its permitted observations enter."""
    c = domain["compiled"]
    (rival,) = set(c.carrier_index) - {focal}
    bank = domain["populations"][rival]
    weights = domain["mixtures"][rival]
    if stage == 0:
        family = "train" if 100000 <= seed < 200000 else "validation"
        kind = int(
            StreamRegistry(seed, family + "/TR/fixed-rule-v1").generator().integers(3)
        )
        if kind == 0:
            rules = {}
            for j in c.products_of[c.carrier_index[rival]]:
                p = c.spec.products[j]
                rules[p.id] = {
                    "base": p.reference_fare,
                    "scarcity": [
                        {"leg": leg, "threshold": 1, "coefficient": 20}
                        for leg in p.legs
                    ],
                }
            controller = ThresholdLinearRuleController(rules)
        elif kind == 1:
            controller = CompetitorMatchingController(offset=0)
        else:
            controller = DLPBidPriceController(reopt_every=1, markup=0)
        controller.arm_name = bank[0].arm
        return rival, controller, ("threshold", "matching", "DLP")[kind]
    if stage == 1:
        active = [
            (p, w)
            for p, w in zip(domain["populations"][focal], domain["mixtures"][focal])
            if w > 0
        ]
        controller = EpisodeTypeMPCController(
            rival,
            bank[0].arm,
            [{focal: HistoryController(p)} for p, _ in active],
            [w for _, w in active],
            TypeMPCConfig(**mpc),
            episode_index=seed,
        )
        return rival, controller, "adaptive_MPC_reference_population_model"
    if stage == 2:
        return rival, EpisodeHistoryMixture(bank, weights), "frozen_current_meta"
    raise ValueError("unknown curriculum stage")


def make_adapter(domain, focal, stage, seed, mpc):
    rival, controller, label = rival_controller(domain, focal, stage, seed, mpc)
    arm = domain["populations"][focal][0].arm
    adapter = RecurrentEnvAdapter(
        FocalCarrierEnv(domain["compiled"], focal, arm, {rival: controller})
    )
    return adapter, controller, label


def contract(domains, cfg, anchor):
    return {
        "domains": [
            {
                "id": d["id"],
                "role": d["role"],
                "spec_hash": d["compiled"].spec_hash,
                "policies": {
                    c: [p.hash() for p in bank] for c, bank in d["populations"].items()
                },
                "mixtures": d["mixtures"],
            }
            for d in domains
        ],
        "config": {k: v for k, v in cfg.items() if not k.startswith("resume_")},
        "schema_hash": anchor.schema_hash(),
        "bounds": asdict(anchor.action_transform),
        "rival_model": "Stage2 frozen focal reference population, explicitly misspecified versus current learning PPO; strict likelihood failures propagate.",
    }


def _save(agent, path, anchor, metadata):
    temporary = path.with_suffix(".tmp.pt")
    save_checkpoint(agent, temporary, anchor, metadata)
    temporary.replace(path)


def train_curriculum(domains, cfg, directory, progress, *, stop_after=None):
    """One optimizer across sampled domains and all three cohort-count stages."""
    validate_domains(domains)
    n, batch = int(cfg["n_train"]), int(cfg["cohorts_per_update"])
    start, vstart = (
        cfg.get("train_seed_start", 100000),
        cfg.get("validation_seed_start", 200000),
    )
    validation = tuple(range(vstart, vstart + int(cfg["n_validation"])))
    if n < 1 or batch < 1 or not validation or cfg["validate_every_cohorts"] < 1:
        raise ValueError("positive training and validation budgets required")
    for seed in (start, start + n - 1):
        seed_in_split(seed, True)
    for seed in validation:
        seed_in_split(seed, False)
    if not 0 <= cfg.get("network_seed", 0) <= 9:
        raise ValueError("network initialization seed must be0–9")
    if stop_after is not None and (not 0 < stop_after <= n or stop_after % batch):
        raise ValueError("verification interruption must be an update boundary")
    focal, mpc = cfg.get("focal", "A"), cfg["mpc"]
    eligible = [d for d in domains if d["role"] == "train"]
    anchor = make_adapter(eligible[0], focal, 0, start, mpc)[0]
    for d in domains:
        other = make_adapter(d, focal, 0, start, mpc)[0]
        if (
            other.names() != anchor.names()
            or other.action_transform != anchor.action_transform
        ):
            raise ValueError("domain observation/action schemas differ")
    identity = tuple(
        k
        for k, name in enumerate(anchor.names())
        if name == "clock"
        or name.startswith(("inventory/", "exo/"))
        or name.endswith(("/available", "/missing"))
    )
    hyper = {
        "gamma": 1.0,
        **cfg.get("ppo", {}),
        "observation_identity_indices": identity,
        "seed": StreamRegistry(
            cfg.get("network_seed", 0), "train/TR/init-v1"
        ).child_seed()
        % 2**31,
    }
    if hyper["gamma"] != 1.0:
        raise ValueError("curriculum cohort objective is undiscounted")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with torch.random.fork_rng(devices=[]):
        agent = RecurrentPPOAgent(
            len(anchor.names()), anchor.action_transform, RecurrentPPOConfig(**hyper)
        )
    anchor.configure_actor(agent)
    training_contract = contract(domains, cfg, anchor)
    fingerprint = schema_hash(training_contract)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    consumed, best, history, validations, assignments = 0, None, [], [], []
    if cfg.get("resume_file"):
        path = Path(cfg["resume_file"])
        if file_hash(path) != cfg["resume_sha256"]:
            raise ValueError("changed curriculum resume checkpoint")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        metadata = payload["metadata"]
        if metadata["curriculum_fingerprint"] != fingerprint:
            raise ValueError("changed curriculum training contract")
        with torch.random.fork_rng(devices=[]):
            agent = load_checkpoint(path, anchor, restore_rng=True)
        consumed, best, history, validations, assignments = (
            deepcopy(metadata[k])
            for k in [
                "consumed",
                "best_validation",
                "history",
                "validation",
                "assignments",
            ]
        )
        prior_best = path.with_name("best.pt")
        if prior_best.exists():
            selected = torch.load(prior_best, map_location="cpu", weights_only=False)
            if selected["metadata"]["curriculum_fingerprint"] != fingerprint:
                raise ValueError("changed selected curriculum checkpoint")
            if prior_best.resolve() != (directory / "best.pt").resolve():
                shutil.copy2(prior_best, directory / "best.pt")
        elif best is not None:
            raise ValueError("selected curriculum checkpoint missing")
    limit = n if stop_after is None else stop_after
    cadence = int(cfg["validate_every_cohorts"])
    next_validation = (consumed // cadence + 1) * cadence
    started = time.perf_counter()
    while consumed < limit:
        values = []
        began = time.perf_counter()
        for k in range(min(batch, limit - consumed)):
            index, seed = consumed + k, start + consumed + k
            stage = curriculum_stage(index, n)
            d = choose_domain(domains, seed, True)
            adapter, rival, label = make_adapter(d, focal, stage, seed, mpc)
            progress(
                {
                    "stage": "training",
                    "cohorts_completed": index,
                    "cohorts_total": n,
                    "curriculum_stage": stage + 1,
                    "member": d["id"],
                    "rival": label,
                }
            )
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(
                    StreamRegistry(seed, "train/TR/policy-v1").child_seed() % 2**31
                )
                episode, logp, value = collect(adapter, agent, seed)
            agent.add_episode(
                episode, actions_are_physical=False, behavior_log_probabilities=logp
            )
            values.append(value)
            assignments.append(
                {
                    "index": index,
                    "seed": seed,
                    "stage": stage + 1,
                    "member": d["id"],
                    "rival": label,
                    "opponent_type": getattr(rival, "choice", None),
                    "MPC_decisions": deepcopy(getattr(rival, "decisions", [])),
                }
            )
        rollout_seconds = time.perf_counter() - began
        began = time.perf_counter()
        stats = agent.update()
        if not all(np.isfinite(v) for v in stats.values()):
            raise FloatingPointError("nonfinite curriculum optimizer statistics")
        consumed += len(values)
        history.append(
            {
                "cohorts": consumed,
                "return_mean": float(np.mean(values)),
                "rollout_seconds": rollout_seconds,
                "update_seconds": time.perf_counter() - began,
                **stats,
            }
        )
        selected = False
        if consumed >= next_validation or consumed == n:
            row = {"cohorts": consumed, "values": [], "assignments": []}
            for seed in validation:
                d = choose_domain(domains, seed, False)
                stage = (seed - vstart) % 3
                adapter, _, label = make_adapter(d, focal, stage, seed, mpc)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(
                        StreamRegistry(seed, "validation/TR/policy-v1").child_seed()
                        % 2**31
                    )
                    value = collect(adapter, agent, seed, deterministic=True)[2]
                row["values"].append(value)
                row["assignments"].append(
                    {
                        "seed": seed,
                        "member": d["id"],
                        "stage": stage + 1,
                        "rival": label,
                    }
                )
            row["mean"] = float(np.mean(row["values"]))
            validations.append(row)
            selected = best is None or row["mean"] > best
            best = row["mean"] if selected else best
            next_validation = (consumed // cadence + 1) * cadence
        metadata = {
            "curriculum_fingerprint": fingerprint,
            "training_contract": training_contract,
            "consumed": consumed,
            "best_validation": best,
            "history": history,
            "validation": validations,
            "assignments": assignments,
            "test_seeds_opened": False,
        }
        _save(agent, directory / "last.pt", anchor, metadata)
        if selected:
            _save(agent, directory / "best.pt", anchor, metadata)
        atomic_json(directory / "learning.json", metadata)
    return {
        "status": "completed" if consumed == n else "verification_interrupted",
        "cohorts": consumed,
        "updates": agent.updates,
        "seconds": time.perf_counter() - started,
        "stage_counts": [sum(x["stage"] == i for x in assignments) for i in (1, 2, 3)],
        "heldout_members_executed": False,
        "best_validation": best,
        "final_validation": validations[-1]["mean"] if validations else None,
        "final_budget_selected": False,
        "calibrated_TR2_completed": False,
        "TR5_monitor_completed": False,
        "gradient_variance": None,
        "test_seeds_opened": False,
        "scope": "Engineering native PPO curriculum/domain path only. Frozen per-domain meta snapshots and explicitly misspecified reference-model adaptive rival; not full TR/P2/calibrated24+8/full-budget evidence.",
    }
