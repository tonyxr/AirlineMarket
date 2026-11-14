"""F5 train-only tails and exact-Q comparison on shared and on-policy states."""

from pathlib import Path
import json
import time

import numpy as np

from ..belief import ExactBelief
from ..campaign.artifacts import atomic_json, file_hash
from ..campaign.instances import build_instance
from ..core.hashing import canonical_json, sha256_text
from ..core.rng import StreamRegistry
from ..core.types import CoefficientAction
from ..pomdp.env import FocalCarrierEnv
from ..sim.finite import FinitePricingModel
from ..sim.sparse_access import sparse_access
from ..solvers.finite_dp import ExactTailEvaluator
from ..solvers.fixed_search import episode_value, search_fixed_grid
from ..solvers.matched_mpc import MatchedScenarioMPC
from ..solvers.mpc import MPCConfig
from ..spec import compile_market
from ..vendor.ride_response.cem import CEMConfig, CEMOptimizer


def _compiled(cell):
    """Resolve the declared instance without guessed exact-scale parameters."""
    parameters = cell["parameters"]
    physical = {
        key: parameters[key]
        for key in ("instance_params", "T", "periods", "actions", "grid_indices")
        if key in parameters
    }
    return compile_market(build_instance(cell["generator"], physical))


def fit_tails(cell, context):
    """Fit fixed and three-block scheduled baselines on training cohorts only."""
    parameters = cell["parameters"]
    start, count = parameters["train_seed_start"], parameters["n_train"]
    if not 100000 <= start < start + count <= 200000:
        raise ValueError("F5 tail fitting requires a nonempty training range")
    compiled = _compiled(cell)
    focal, arm = parameters.get("focal", "A"), parameters.get("arm", "direct")
    pricing = next(p for p in compiled.spec.pricing if p.carrier == focal)
    seeds = tuple(range(start, start + count))
    factory = lambda: FocalCarrierEnv(compiled, focal, arm)
    with sparse_access():
        context.update({"stage": "fixed_grid", "training_cohorts": count})
        fixed = search_fixed_grid(factory, pricing.theta_grid, seeds)
        T = compiled.spec.horizon.n_periods
        breaks = np.linspace(0, T, 4, dtype=int)
        blocks = [(int(a), int(b)) for a, b in zip(breaks, breaks[1:]) if a < b]
        bounds = np.asarray(pricing.theta_bounds)
        evaluations = []
        env = factory()

        def schedule(x):
            """Convert bounded coordinates to a departure-period schedule."""
            return np.clip(np.asarray(x).reshape(len(blocks), len(bounds)), *bounds.T)

        def objective(x):
            """Use the same training tapes for every CEM candidate."""
            theta = schedule(x)
            values = [
                episode_value(
                    env,
                    seed,
                    lambda obs: CoefficientAction(
                        tuple(
                            theta[
                                next(
                                    k
                                    for k, (a, b) in enumerate(blocks)
                                    if a <= obs.t < b
                                )
                            ]
                        )
                    ),
                )
                for seed in seeds
            ]
            value = float(np.mean(values))
            evaluations.append({"thetas": theta.tolist(), "training_objective": value})
            context.update(
                {"stage": "schedule_cem", "candidate_evaluations": len(evaluations)}
            )
            return value

        cem = CEMConfig(**parameters["cem"])
        fitted = CEMOptimizer(len(blocks) * len(bounds), cem).optimize(
            objective,
            initial_mean=np.tile(fixed.best_theta, len(blocks)),
            initial_std=np.tile(
                np.maximum(np.ptp(bounds, axis=1) / 3, 0.01), len(blocks)
            ),
        )
        scheduled = {
            "blocks": blocks,
            "thetas": schedule(fitted["coordinates"]).tolist(),
        }
    result = {
        "status": "completed",
        "spec_hash": compiled.spec_hash,
        "focal": focal,
        "arm": arm,
        "fixed": list(fixed.best_theta),
        "fixed_training_table": fixed.table,
        "schedule": scheduled,
        "schedule_training_table": evaluations,
        "cem": parameters["cem"],
        "training_seed_start": start,
        "training_cohorts": count,
        "note": "Train-only baseline search, not RL or empirical calibration. CEM includes the best fixed policy as its initial center.",
    }
    atomic_json(context.directory / "tails.json", result)
    return {
        "status": "completed",
        "spec_hash": compiled.spec_hash,
        "tails_file": "tails.json",
        "tails_sha256": file_hash(context.directory / "tails.json"),
        "training_cohorts": count,
        "schedule_candidate_evaluations": len(evaluations),
    }


def _incumbent(tail, t):
    """Resolve the candidate appended to both exact and sampled action sets."""
    if isinstance(tail, dict):
        return tuple(
            tail["thetas"][
                next(k for k, (a, b) in enumerate(tail["blocks"]) if a <= t < b)
            ]
        )
    return tuple(tail)


def compare(cell, context):
    """Compare exact Q at identical shared states and at every on-policy decision."""
    p = cell["parameters"]
    start, count = p["seed_start"], p["n_cohorts"]
    if not 200000 <= start < start + count <= 300000:
        raise ValueError("F5 comparison requires validation cohorts only")
    tail_path = Path(p["tails_file"])
    if file_hash(tail_path) != p["tails_sha256"]:
        raise ValueError("changed fitted-tail input")
    fitted = json.loads(tail_path.read_text())
    compiled = _compiled(cell)
    if fitted["spec_hash"] != compiled.spec_hash:
        raise ValueError("tails and comparison use different physical specifications")
    focal, arm = fitted["focal"], fitted["arm"]
    pricing = next(q for q in compiled.spec.pricing if q.carrier == focal)
    objective = next(q for q in compiled.spec.objectives if q.carrier == focal)
    tail = fitted["schedule"] if p["tail"] == "schedule" else tuple(fitted["fixed"])
    config = MPCConfig(
        evaluation="sampling",
        n_scenarios=p["scenarios"],
        selection=p["selection"],
        tail_policy=tail if isinstance(tail, dict) else "fixed:best",
        protect_incumbent=p["protect_incumbent"],
        switch_standard_errors=p["k_se"],
        racing_initial_scenarios=64,
    )
    evaluator = ExactTailEvaluator(FinitePricingModel(compiled, focal, arm), tail)
    decisions, episodes = [], []
    with sparse_access():
        for index, seed in enumerate(range(start, start + count)):
            # Every factor cell visits this same fitted-fixed behavior trajectory.
            # On-policy trajectories are separate, paired physical evaluation tapes.
            row = {"seed": seed}
            for mode in ["shared_fixed_trajectory", "on_policy"]:
                env = FocalCarrierEnv(compiled, focal, arm)
                obs, _ = env.reset(seed)
                belief = ExactBelief(compiled, focal, arm)
                belief.reset(obs)
                planner = MatchedScenarioMPC(
                    compiled,
                    focal,
                    arm,
                    belief,
                    config=config,
                    tail_coefficients=fitted["fixed"],
                    plan_streams=StreamRegistry(773, "plan"),
                )
                while obs.t < compiled.spec.horizon.n_periods:
                    candidates = list(pricing.theta_grid)
                    incumbent = _incumbent(tail, obs.t)
                    if incumbent not in candidates:
                        candidates.append(incumbent)
                    reference_started = time.perf_counter()
                    q = evaluator.q_values(obs, belief, candidates)
                    reference_seconds = time.perf_counter() - reference_started
                    signature = sha256_text(
                        canonical_json(
                            {
                                "observation": obs,
                                "posterior": [
                                    (h.kernel, weight)
                                    for h, weight in sorted(belief.mass.items())
                                ],
                            }
                        )
                    )
                    began = time.perf_counter()
                    planner.draw_requests.clear()
                    chosen = planner.choose(obs, index)
                    elapsed = time.perf_counter() - began
                    selected = candidates.index(tuple(chosen.action.theta))
                    best = int(np.argmax(q))
                    decisions.append(
                        {
                            "seed": seed,
                            "mode": mode,
                            "period": obs.t,
                            "state_signature": signature,
                            "strict_argmax_agreement": selected == best,
                            "q_optimal_within_1e_12": float(q[best] - q[selected])
                            <= 1e-12,
                            "exact_objective_regret": max(
                                0.0, float(q[best] - q[selected])
                            ),
                            "regret_currency": max(0.0, float(q[best] - q[selected]))
                            * objective.profit_scale,
                            "exact_best_q": float(q[best]),
                            "exact_selected_q": float(q[selected]),
                            "exact_reference_seconds": reference_seconds,
                            "decision_seconds": elapsed,
                            "selected_candidate": selected,
                            "argmax_candidate": best,
                            "draw_requests": planner.draw_requests.copy(),
                            "diagnostics": chosen.diagnostics,
                        }
                    )
                    action = (
                        chosen.action
                        if mode == "on_policy"
                        else CoefficientAction(tuple(fitted["fixed"]))
                    )
                    after, _, _, _ = env.step(action)
                    belief.update(obs, action, after)
                    obs = after
                    context.update(
                        {
                            "stage": mode,
                            "cohort": index + 1,
                            "cohorts": count,
                            "period": obs.t,
                            "decisions_completed": len(decisions),
                        }
                    )
                metrics = env.engine.metrics()["carriers"][focal]
                if mode == "on_policy":
                    row.update(
                        value=metrics["objective"],
                        profit=metrics["profit"],
                        revenue=metrics["revenue"],
                    )
                else:
                    row.update(
                        fixed_value=metrics["objective"],
                        fixed_profit=metrics["profit"],
                        fixed_revenue=metrics["revenue"],
                    )
            row["direct_minus_fixed_profit"] = row["profit"] - row["fixed_profit"]
            episodes.append(row)
            atomic_json(context.directory / "decisions.json", decisions)
            atomic_json(context.directory / "episodes.json", episodes)
    reports = {}
    for mode in ["shared_fixed_trajectory", "on_policy"]:
        records = [d for d in decisions if d["mode"] == mode]
        regret = np.asarray([d["regret_currency"] for d in records])
        reports[mode] = {
            "decisions": len(records),
            "strict_agreement": float(
                np.mean([d["strict_argmax_agreement"] for d in records])
            ),
            "q_optimal_within_1e_12": float(
                np.mean([d["q_optimal_within_1e_12"] for d in records])
            ),
            "mean_regret_currency": float(regret.mean()),
            "p90_regret_currency": float(np.quantile(regret, 0.9)),
            "mean_seconds_per_decision": float(
                np.mean([d["decision_seconds"] for d in records])
            ),
        }
    difference = np.asarray([row["direct_minus_fixed_profit"] for row in episodes])
    se = float(np.std(difference, ddof=1) / np.sqrt(count)) if count > 1 else None
    mean = float(difference.mean())
    confidence_interval = None
    if count > 1:
        bootstrap = StreamRegistry(773, "analysis/F5/paired-bootstrap").generator()
        replicates = difference[bootstrap.integers(count, size=(2000, count))].mean(
            axis=1
        )
        confidence_interval = np.quantile(replicates, [0.025, 0.975]).tolist()
    return {
        "status": "completed",
        "spec_hash": compiled.spec_hash,
        "n_cohorts": count,
        "comparison": reports,
        "mean_direct_minus_fixed_profit": mean,
        "paired_standard_error": se,
        "paired_95_ci": confidence_interval,
        "paired_ci_method": "paired-cohort percentile bootstrap,2000 resamples; one-cohort pilot has no CI",
        "mean_cohort_revenue": float(np.mean([row["revenue"] for row in episodes])),
        "criterion_status": "pilot_only" if count < 256 else "measured_256_cohorts",
        "planning_coupling": "scenario-indexed pairs v1; identical selection prefixes across all/split/protected/racing; independent reporting family",
        "agreement_rule": "Strict first-argmax agreement reported separately from Q ties within1e-12; no metric substituted to obtain a pass",
        "regret_units": "Exact objective gap times profit_scale; includes any declared share utility as a monetary equivalent",
        "artifacts": {"decisions": "decisions.json", "episodes": "episodes.json"},
    }
