"""Fresh continuation of sealed structured checkpoints, with native diagnostics."""

import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.campaign.runner import completed_attempt
from airline_market.learn.gradient_monitor import (
    MetricLog,
    MonitorState,
    gradient_repeats,
    rollout_metrics,
)
from airline_market.learn.structured_runtime import StructuredAdapter
from airline_market.learn.structured_train import (
    collect_structured,
    load_structured,
    save_structured,
)
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.spec import compile_market, load_spec


def diagnostic(agent, adapter, cfg, directory, update, attempt=0):
    """Fresh after-update on-policy batch; no mutation of actual training agent."""
    probe = deepcopy(agent)
    probe.reset_rollout()
    begin = int(cfg["diagnostic_seed_start"]) + update * int(cfg["diagnostic_pool"])
    payload = {}
    for k in range(int(cfg["diagnostic_pool"])):
        seed = begin + k
        episode, logp, _ = collect_structured(
            adapter, probe, seed, training=True, credit=cfg["credit"]
        )
        probe.add_episode(
            episode, actions_are_physical=False, behavior_log_probabilities=logp
        )
        for name, values in [
            ("observations", episode.observations),
            ("actions", episode.actions),
            ("rewards", episode.rewards),
            ("dones", episode.dones),
            ("logp", logp),
        ]:
            payload[f"{name}_{k}"] = np.asarray(values)
        if "counterfactual_advantages" in episode.metadata:
            payload[f"credit_{k}"] = np.asarray(
                episode.metadata["counterfactual_advantages"]
            )
    target = directory / f"diagnostic_{update}_attempt{attempt}"
    target.mkdir()
    np.savez_compressed(target / "rollout_pool.npz", **payload)

    def progress(row):
        with (target / "gradient_draws.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")

    measured = gradient_repeats(probe, update=update, progress=progress)
    np.savez_compressed(
        target / "gradient_repeats.npz",
        gradients=measured.pop("gradients"),
        indices=measured.pop("indices"),
        losses=measured.pop("losses"),
    )
    measured.update(
        update=update,
        seed_start=begin,
        seed_stop=begin + cfg["diagnostic_pool"],
        rollout_pool_sha256=file_hash(target / "rollout_pool.npz"),
        gradient_repeats_sha256=file_hash(target / "gradient_repeats.npz"),
    )
    atomic_json(target / "summary.json", measured)
    return measured


def continuation(agent, adapter, cfg, directory, parent_metadata, progress):
    """Keep native structured updates intact; one registered rollback/half-LR retry."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    log, monitor = MetricLog(directory), MonitorState()
    consumed = int(parent_metadata["consumed"])
    initial_consumed, initial_update = consumed, agent.updates
    batch, target = int(cfg["batch"]), int(cfg["target_update"])
    train_end = 100000 + consumed + (target - agent.updates) * batch
    if (
        target <= agent.updates
        or batch < 1
        or not 100000 <= 100000 + consumed < train_end <= 200000
    ):
        raise ValueError("invalid fresh continuation training range")
    diagnostic_seeds = {
        cfg["diagnostic_seed_start"] + u * cfg["diagnostic_pool"] + k
        for u in range(agent.updates + 1, target + 1)
        if u % 10 == 0
        for k in range(cfg["diagnostic_pool"])
    }
    if any(
        not 100000 <= s < 200000 for s in diagnostic_seeds
    ) or diagnostic_seeds & set(range(100000 + consumed, train_end)):
        raise ValueError(
            "diagnostic and optimizer-training streams overlap or leave split"
        )
    validation = range(
        cfg["validation_seed_start"], cfg["validation_seed_start"] + cfg["n_validation"]
    )
    if any(not 200000 <= s < 300000 for s in validation):
        raise ValueError("validation split invalid")
    history, diagnostics, aborted, validation_values = [], [], [], []
    executions = 0
    while agent.updates < target:
        before = deepcopy(agent)
        ordinal = agent.updates + 1
        started = time.perf_counter()
        attempted = []
        try:
            returns = []
            for k in range(batch):
                seed = 100000 + consumed + k
                progress(
                    {
                        "stage": "training",
                        "update": ordinal,
                        "cohorts_completed": consumed - initial_consumed + k,
                        "cohorts_total": (target - initial_update) * batch,
                        "attempt": monitor.restarts,
                    }
                )
                episode, logp, value = collect_structured(
                    adapter, agent, seed, training=True, credit=cfg["credit"]
                )
                attempted.append(seed)
                executions += 1
                agent.add_episode(
                    episode, actions_are_physical=False, behavior_log_probabilities=logp
                )
                returns.append(value)
            prepared = deepcopy(agent)._prepare_episodes()
            stats = agent.update()
            elapsed = time.perf_counter() - started
            metrics = rollout_metrics(agent, prepared)
            row = {
                "update": agent.updates,
                "cohorts": consumed + batch,
                "return_mean": float(np.mean(returns)),
                **metrics,
                "value_loss": stats["value_loss"],
                "learning_rate": agent.actor_optimizer.param_groups[0]["lr"],
                "cohorts_per_second": batch / elapsed,
                "gradient_variance": None,
                "validation_objective": None,
                "attempt": monitor.restarts,
            }
            reason = monitor.inspect({**row, **stats})
            if reason:
                raise FloatingPointError(reason)
            if agent.updates % 10 == 0:
                progress({"stage": "gradient_diagnostic", "update": agent.updates})
                measured = diagnostic(
                    agent, adapter, cfg, directory, agent.updates, monitor.restarts
                )
                diagnostics.append(measured)
                row["gradient_variance"] = measured["trace_covariance"]
            if agent.updates == target:
                progress({"stage": "validation", "update": agent.updates})
                validation_values = [
                    collect_structured(
                        adapter, agent, seed, training=False, deterministic=True
                    )[2]
                    for seed in validation
                ]
                row["validation_objective"] = float(np.mean(validation_values))
            log.write(row)
        except FloatingPointError as exc:
            failure = {
                "update": ordinal,
                "attempt": monitor.restarts,
                "reason": str(exc),
                "completed_rollout_seeds": attempted,
                "monitor_state": monitor.state(),
            }
            aborted.append(failure)
            atomic_json(directory / f"abort_{len(aborted)}.json", failure)
            save_structured(
                agent,
                directory / f"aborted_{len(aborted)}.pt",
                adapter,
                {"failure": failure},
            )
            # Restore the untouched pre-batch optimizer/normalizer/RNG, then halve both LRs.
            agent = before
            monitor.restart(agent)
            continue
        consumed += batch
        history.append(row)
        metadata = {
            "consumed": consumed,
            "history": history,
            "diagnostics": diagnostics,
            "monitor_state": monitor.state(),
            "aborted": aborted,
            "validation_values": validation_values,
            "parent_fingerprint": parent_metadata["training_fingerprint"],
            "test_seeds_opened": False,
        }
        save_structured(agent, directory / "last.pt", adapter, metadata)
        atomic_json(directory / "monitoring.json", metadata)
    return {
        "status": "completed",
        "start_update": initial_update,
        "end_update": agent.updates,
        "additional_optimizer_cohorts": consumed - initial_consumed,
        "training_rollout_executions": executions,
        "diagnostic_physical_cohorts": len(diagnostic_seeds),
        "diagnostics": diagnostics,
        "validation_values": validation_values,
        "restarts": monitor.restarts,
        "aborts": aborted,
        "full_D4_or_TR_completed": False,
        "scientific_RL_budget": None,
        "scope": "Fresh engineering continuation from sealed throughput weights.32minibatch diagnostics on fresh after-update on-policy pools, not a completed primary256cohort F9 variance/power study or full-budget learning claim. TensorBoard scalar wire format verified; external consumer unavailable locally.",
    }


def run(cell, context):
    p = cell["parameters"]
    source = Path(p["parent_directory"])
    for name, expected in p["parent_hashes"].items():
        if file_hash(source / name) != expected:
            raise ValueError("changed frozen continuation parent")
    parent_cfg = json.loads((source / "config.json").read_text())
    if parent_cfg["parameters"]["learner"] != p["learner"] or p["learner"] not in (
        "srl",
        "srl_c",
    ):
        raise ValueError("parent learner identity differs")
    found = completed_attempt(source.parent, parent_cfg["config_hash"])
    if found is None or found[0] != source or found[1]["status"] != "completed":
        raise ValueError("parent throughput receipt missing")
    c = compile_market(load_spec(source / "spec.json"))
    if c.spec_hash != found[1]["spec_hash"]:
        raise ValueError("parent physical model differs")
    adapter = StructuredAdapter(FocalCarrierEnv(c, "A", "direct"))
    agent, metadata = load_structured(source / "last.pt", adapter)
    if metadata["consumed"] != 256 or agent.updates != 4:
        raise ValueError("declared parent progress differs")
    cfg = {
        **p["measurement"],
        "credit": parent_cfg["parameters"]["credit"]
        if p["learner"] == "srl_c"
        else None,
    }
    latest = {"stage": "setup"}
    context.counters = lambda: latest.copy()
    result = continuation(
        agent, adapter, cfg, context.directory, metadata, latest.update
    )
    manifest = {
        "checkpoint": "last.pt",
        "sha256": file_hash(context.directory / "last.pt"),
        "parent_checkpoint_sha256": p["parent_hashes"]["last.pt"],
        "storage": "local Git-ignored weights",
        "second_machine_reproduction": False,
    }
    atomic_json(context.directory / "checkpoint_manifest.json", manifest)
    return {
        **result,
        "learner": p["learner"],
        "spec_hash": c.spec_hash,
        "parent_results_sha256": file_hash(source / "results.json"),
        "parent_weights_originally_sealed": False,
        "monitoring_sha256": file_hash(context.directory / "monitoring.json"),
        "checkpoint_manifest_sha256": file_hash(
            context.directory / "checkpoint_manifest.json"
        ),
        "metrics_csv_sha256": file_hash(context.directory / "metrics.csv"),
        "scalar_events_sha256": file_hash(
            context.directory / "events.out.tfevents.airline-scalars"
        ),
    }
