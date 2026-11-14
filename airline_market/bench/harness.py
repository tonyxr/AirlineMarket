"""Warm-up/repeat benchmarks with setup excluded from operation timing."""

from pathlib import Path
import hashlib
import json
import platform
import time

import numpy as np

from ..campaign.artifacts import atomic_json, environment
from ..campaign.instances import build_instance
from ..core.types import CoefficientAction
from ..spec import compile_market


def operation_factory(name: str, compiled, parameters: dict):
    """Return fresh equivalent operation states for each timed repeat."""
    from ..agents.fixed import FixedCoefficientController
    from ..belief import ExactBelief, ParticleBelief, FactorizedBelief
    from ..pomdp.env import FocalCarrierEnv
    from ..sim.engine import MarketEngine
    from ..sim.kernel import ExactKernel

    c = compiled
    focal = c.spec.carriers[0].code
    theta = c.spec.pricing[0].theta_grid[-1]
    action = CoefficientAction(theta)
    arm = parameters.get("arm", "linked")

    def env():
        return FocalCarrierEnv(c, focal, arm)

    if name in ("engine_step", "engine_cohort"):

        def prepare():
            engine = MarketEngine(c, {focal: FixedCoefficientController(theta)})
            engine.reset(200000)
            if name == "engine_step":
                return engine.step

            def cohort():
                while not engine.terminal():
                    engine.step()
                return c.spec.horizon.n_periods

            return cohort

        return prepare
    if name == "batch_sampler_cohort":
        from ..sim.batch import rollout_values, supports

        def prepare():
            engines = [
                env().engine for _ in range(int(parameters.get("batch_size", 16)))
            ]
            for k, engine in enumerate(engines):
                engine.reset(200000 + k)
            if not supports(c, 0, engines):
                raise ValueError(
                    "batch sampler does not support this benchmark instance"
                )
            return lambda: rollout_values(c, 0, engines, [theta], theta)

        return prepare
    if name == "kernel_transition":

        def prepare():
            e = env()
            obs, _ = e.reset(200000)
            kernel = ExactKernel(c)
            state = kernel.encode(e.engine.world)
            selected_arm = next(
                a for a in c.spec.information if a.observer == focal and a.name == arm
            )
            return lambda: list(
                kernel.transition(state, 0, action, arm=selected_arm, observation=obs)
            )

        return prepare
    if name in ("exact_belief_update", "factorized_belief_update") or name.startswith(
        "particle_belief_update_"
    ):

        def prepare():
            e = env()
            before, _ = e.reset(200000)
            if name.startswith("particle"):
                count = int(name.rsplit("_", 1)[1])
                belief = ParticleBelief(c, focal, arm, n_particles=count, seed=773)
            else:
                cls = ExactBelief if name == "exact_belief_update" else FactorizedBelief
                belief = cls(c, focal, arm)
            belief.reset(before)
            after, _, _, _ = e.step(action)
            return lambda: belief.update(before, action, after)

        return prepare
    if name == "mpc_decision":
        from ..solvers.mpc import ScenarioMPC, MPCConfig

        def prepare():
            e = env()
            obs, _ = e.reset(200000)
            belief = ExactBelief(c, focal, arm)
            belief.reset(obs)
            planner = ScenarioMPC(
                c,
                focal,
                arm,
                belief,
                config=MPCConfig(
                    evaluation="sampling",
                    n_scenarios=int(parameters.get("n_scenarios", 256)),
                ),
                tail_coefficients=theta,
            )
            return lambda: planner.choose(obs, 0)

        return prepare
    if name == "compact_dp":
        from ..solvers.exact_dp import ExactBeliefDP

        micro = compile_market(build_instance("toy_hub", {"T": 3, "actions": 2}))

        def prepare():
            dp = ExactBeliefDP(
                micro, "A", "direct", backend="compact", max_nodes=100000
            )
            return dp.solve

        return prepare
    if name == "ppo_update":
        import torch
        from ..learn.adapters import RecurrentEnvAdapter
        from ..learn.ppo import collect
        from ..vendor.ride_response.recurrent.ppo import (
            RecurrentPPOAgent,
            RecurrentPPOConfig,
        )

        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)

        def prepare():
            adapter = RecurrentEnvAdapter(env())
            agent = RecurrentPPOAgent(
                len(adapter.names()),
                adapter.action_transform,
                RecurrentPPOConfig(hidden_dim=32, update_epochs=1, seed=773),
            )
            adapter.configure_actor(agent)
            for seed in range(100000, 100002):
                episode, logp, _ = collect(adapter, agent, seed)
                agent.add_episode(
                    episode, actions_are_physical=False, behavior_log_probabilities=logp
                )
            return agent.update

        return prepare
    raise ValueError(f"unknown benchmark operation {name}")


def run_benchmarks(
    parameters: dict | None = None, out: str | Path | None = None
) -> dict:
    """Measure warmed median/IQR costs; every repeat starts at equivalent state."""
    parameters = dict(parameters or {})
    generator = parameters.get("generator", "toy_hub")
    c = compile_market(build_instance(generator, parameters.get("instance_params", {})))
    operations = parameters.get(
        "operations",
        [
            "engine_step",
            "engine_cohort",
            "batch_sampler_cohort",
            "kernel_transition",
            "exact_belief_update",
            "particle_belief_update_5000",
            "particle_belief_update_20000",
            "factorized_belief_update",
            "mpc_decision",
            "compact_dp",
            "ppo_update",
        ],
    )
    repeats = int(parameters.get("repeats", 5))
    warmup = int(parameters.get("warmup", 1))
    if repeats < 1 or warmup < 0:
        raise ValueError("invalid benchmark repeats/warmup")
    rows = []
    for name in operations:
        prepare = operation_factory(name, c, parameters)
        for _ in range(warmup):
            prepare()()
        durations, setups, counts = [], [], []
        for _ in range(repeats):
            setup_started = time.perf_counter()
            operation = prepare()
            setups.append(time.perf_counter() - setup_started)
            started = time.perf_counter()
            outcome = operation()
            durations.append(time.perf_counter() - started)
            if name == "compact_dp":
                counts.append(outcome.nodes)
        divisor = (
            int(parameters.get("batch_size", 16))
            if name == "batch_sampler_cohort"
            else 1
        )
        normalized = np.asarray(durations) / divisor
        row = {
            "operation": name,
            "repeats": repeats,
            "warmup": warmup,
            "median_seconds": float(np.median(normalized)),
            "iqr_seconds": float(np.subtract(*np.percentile(normalized, [75, 25]))),
            "raw_seconds": durations,
            "median_setup_seconds": float(np.median(setups)),
            "normalization": "per cohort" if divisor != 1 else "per operation",
        }
        if counts:
            row.update(
                nodes=counts,
                median_nodes_per_second=float(
                    np.median(np.asarray(counts) / durations)
                ),
                instance_note="toy_hub T=3, two nested actions",
            )
        rows.append(row)
        if out:
            atomic_json(
                Path(out) / "progress.json",
                {
                    "completed_operations": len(rows),
                    "total_operations": len(operations),
                    "latest": name,
                },
            )
    result = {
        "schema": "airline-benchmark/1",
        "status": "completed",
        "machine_id": hashlib.sha256(platform.node().encode()).hexdigest()[:16],
        "environment": environment(),
        "spec_hash": c.spec_hash,
        "parameters": parameters,
        "benchmarks": rows,
        "timing_scope": "Model construction/reset/collection are reported as setup, excluded from operation time. Batch normalized by its declared batch size.",
    }
    if out:
        atomic_json(Path(out) / "bench.json", result)
    return result


def compare_baseline(
    current: dict, baseline: dict, tolerance: float = 0.25
) -> list[dict]:
    """Identify actual same-machine median slowdowns above the declared 25%."""
    if current["machine_id"] != baseline["machine_id"]:
        raise ValueError("benchmark regression comparison requires the same machine")
    reference = {row["operation"]: row for row in baseline["benchmarks"]}
    failures = []
    for row in current["benchmarks"]:
        old = reference[row["operation"]]
        ratio = row["median_seconds"] / old["median_seconds"]
        if ratio > 1 + tolerance:
            failures.append({"operation": row["operation"], "slowdown_ratio": ratio})
    return failures
