"""Independent I4 particle study on frozen exact-frontier physical inputs.

The construction-only reference kernel and SMC law are unchanged. The scale
is an established feasible lower bound, not the final largest exact scale.
"""

import json
import time
from pathlib import Path

import numpy as np

from airline_market.belief import ExactBelief
from airline_market.belief.events import AllOf, LegInventoryAtMost
from airline_market.belief.info import inventory_mass
from airline_market.belief.template_particle import (
    TemplateExactKernel,
    TemplateParticleBelief,
)
from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.rng import StreamRegistry
from airline_market.core.types import CoefficientAction
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.sim.sparse_access import sparse_access
from airline_market.solvers.matched_mpc import MatchedScenarioMPC
from airline_market.solvers.mpc import MPCConfig
from airline_market.spec import compile_market, load_spec


def frozen_model(parameters):
    """Load the exact physical bytes used by the completed history frontier."""
    path = Path(parameters["source_spec"])
    if file_hash(path) != parameters["source_spec_sha256"]:
        raise ValueError("frozen physical specification changed")
    return compile_market(load_spec(path))


def ess_record(particle, updated):
    """Keep measured pre/post values distinct, including missing initial pre-ESS."""
    return {
        "pre_resampling_ess": float(particle.diagnostics[-1]["ess"])
        if updated
        else None,
        "post_resampling_ess": float(1 / (particle.weights @ particle.weights)),
    }


def fit_fixed(cell, context):
    """Search the declared discrete grid once on common training tapes only."""
    from airline_market.solvers.fixed_search import search_fixed_grid

    p = cell["parameters"]
    start, count = p["train_seed_start"], p["n_train"]
    if not 100000 <= start < start + count <= 200000:
        raise ValueError("fixed-tail search requires nonempty training seeds only")
    compiled = frozen_model(p)
    focal, arm = p.get("focal", "A"), p.get("arm", "direct")
    pricing = next(q for q in compiled.spec.pricing if q.carrier == focal)
    with sparse_access():
        context.update(
            {
                "stage": "fixed_grid",
                "training_cohorts": count,
                "candidates": len(pricing.theta_grid),
            }
        )
        fitted = search_fixed_grid(
            lambda: FocalCarrierEnv(compiled, focal, arm),
            pricing.theta_grid,
            range(start, start + count),
        )
    result = {
        "status": "completed",
        "spec_hash": compiled.spec_hash,
        "source_spec_sha256": p["source_spec_sha256"],
        "focal": focal,
        "arm": arm,
        "fixed": list(fitted.best_theta),
        "fixed_training_table": fitted.table,
        "training_seed_start": start,
        "training_cohorts": count,
        "note": "Training-only full declared discrete-grid search, not a global continuous optimum or fitted validation policy.",
    }
    atomic_json(context.directory / "tails.json", result)
    return {
        "status": "completed",
        "spec_hash": compiled.spec_hash,
        "tails_sha256": file_hash(context.directory / "tails.json"),
        "training_cohorts": count,
        "candidates": len(pricing.theta_grid),
    }


def total_variation(left, right):
    """Compare the complete declared probability laws without dropping states."""
    return 0.5 * sum(
        abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in set(left) | set(right)
    )


def compare(cell, context):
    """Measure exact-reference error and matched-MPC decisions on both arms."""
    p = cell["parameters"]
    count, start = p["n_cohorts"], p["seed_start"]
    if not 200000 <= start < start + count <= 300000:
        raise ValueError("F4 accuracy uses a nonempty validation cohort range")
    compiled = frozen_model(p)
    path = Path(p["tails_file"])
    if file_hash(path) != p["tails_sha256"]:
        raise ValueError("fitted-tail input changed")
    tails = json.loads(path.read_text())
    if tails["spec_hash"] != compiled.spec_hash:
        raise ValueError("tail and accuracy study have different physical models")
    focal, arm = tails["focal"], p["arm"]
    event = AllOf(
        tuple(LegInventoryAtMost(p["rival"], leg, 1) for leg in p["joint_legs"])
    )
    planner_config = MPCConfig(
        evaluation="sampling",
        n_scenarios=p["planning_scenarios"],
        selection="all",
        protect_incumbent=True,
        switch_standard_errors=2.0,
    )
    rows = []
    reset_seconds = []
    with sparse_access():
        for index, seed in enumerate(range(start, start + count)):
            env = FocalCarrierEnv(compiled, focal, arm)
            obs, _ = env.reset(seed)
            exact = ExactBelief(
                compiled,
                focal,
                arm,
                kernel=TemplateExactKernel(compiled, validate=False),
            )
            exact.reset(obs)
            particle = TemplateParticleBelief(
                compiled,
                focal,
                arm,
                n_particles=p["particles"],
                seed=StreamRegistry(773, "plan/F4_I4/filter").child_seed(index),
                kernel=exact.kernel,
                conditional_sampling="systematic",
                strict=True,
            )
            particle.reset(obs)
            reset_seconds.append(particle.prior_reset_seconds[-1])
            exact_planner = MatchedScenarioMPC(
                compiled,
                focal,
                arm,
                exact,
                config=planner_config,
                tail_coefficients=tails["fixed"],
                plan_streams=StreamRegistry(773, "plan"),
            )
            particle_planner = MatchedScenarioMPC(
                compiled,
                focal,
                arm,
                particle,
                config=planner_config,
                tail_coefficients=tails["fixed"],
                plan_streams=StreamRegistry(773, "plan"),
            )
            update_seconds = None
            atomic_json(context.directory / "posterior_rows.json", rows)
            while True:
                marginal_error = 0.0
                approximate_marginals = particle.leg_marginals()
                for leg, law in exact.leg_marginals().items():
                    other = approximate_marginals[leg]
                    size = max(len(law), len(other))
                    marginal_error = max(
                        marginal_error,
                        float(
                            np.max(
                                abs(
                                    np.pad(law, (0, size - len(law)))
                                    - np.pad(other, (0, size - len(other)))
                                )
                            )
                        ),
                    )
                exact_event = exact.event_probability(event)
                particle_event = particle.event_probability(event)
                row = {
                    "seed": seed,
                    "period": obs.t,
                    "full_hidden_tv": total_variation(exact.mass, particle.mass),
                    "inventory_tv": total_variation(
                        inventory_mass(exact), inventory_mass(particle)
                    ),
                    "max_leg_marginal_error": marginal_error,
                    "exact_joint_event": exact_event,
                    "particle_joint_event": particle_event,
                    "joint_event_error": abs(exact_event - particle_event),
                    **ess_record(particle, update_seconds is not None),
                    "particle_update_seconds": update_seconds,
                    "reference_support": len(exact.mass),
                    "particle_support": len(particle.mass),
                    "strict_coefficient_agreement": None,
                    "posted_price_agreement": None,
                    "exact_planner_seconds": None,
                    "particle_planner_seconds": None,
                }
                if obs.t < compiled.spec.horizon.n_periods:
                    begin = time.perf_counter()
                    reference_action = exact_planner.choose(obs, index).action
                    row["exact_planner_seconds"] = time.perf_counter() - begin
                    begin = time.perf_counter()
                    approximate_action = particle_planner.choose(obs, index).action
                    row["particle_planner_seconds"] = time.perf_counter() - begin
                    row["strict_coefficient_agreement"] = (
                        reference_action.theta == approximate_action.theta
                    )
                    rule = env.price_rule
                    row["posted_price_agreement"] = bool(
                        np.array_equal(
                            rule.prices(reference_action.theta, obs),
                            rule.prices(approximate_action.theta, obs),
                        )
                    )
                rows.append(row)
                atomic_json(context.directory / "posterior_rows.json", rows)
                context.update(
                    {
                        "stage": "posterior_accuracy",
                        "cohort": index + 1,
                        "cohorts": count,
                        "period": obs.t,
                        "particles": p["particles"],
                        "observations_completed": len(rows),
                    }
                )
                if obs.t == compiled.spec.horizon.n_periods:
                    break
                action = CoefficientAction(tuple(tails["fixed"]))
                after, _, _, _ = env.step(action)
                exact.update(obs, action, after)
                begin = time.perf_counter()
                particle.update(obs, action, after)
                update_seconds = time.perf_counter() - begin
                obs = after
            atomic_json(context.directory / "posterior_rows.json", rows)
    decisions = [r for r in rows if r["posted_price_agreement"] is not None]
    mean_tv = float(np.mean([r["full_hidden_tv"] for r in rows]))
    joint_error = max(r["joint_event_error"] for r in rows)
    agreement = float(np.mean([r["posted_price_agreement"] for r in decisions]))
    return {
        "status": "completed",
        "spec_hash": compiled.spec_hash,
        "n_cohorts": count,
        "particles": p["particles"],
        "arm": arm,
        "observations": len(rows),
        "decisions": len(decisions),
        "mean_tv": mean_tv,
        "max_tv": max(r["full_hidden_tv"] for r in rows),
        "mean_inventory_tv": float(np.mean([r["inventory_tv"] for r in rows])),
        "max_leg_marginal_error": max(r["max_leg_marginal_error"] for r in rows),
        "max_joint_event_error": joint_error,
        "decision_agreement": agreement,
        "strict_coefficient_agreement": float(
            np.mean([r["strict_coefficient_agreement"] for r in decisions])
        ),
        "mean_update_ms": 1000
        * float(
            np.mean(
                [
                    r["particle_update_seconds"]
                    for r in rows
                    if r["particle_update_seconds"] is not None
                ]
            )
        ),
        "mean_prior_reset_seconds": float(np.mean(reset_seconds)),
        "provisional_criteria_pass": None
        if count < 64
        else mean_tv <= 0.02 and joint_error <= 0.02 and agreement >= 0.95,
        "particle_selection_status": "not_selected; final largest exact scale, hub references and final F5 planner confirmation still required",
        "source_spec_sha256": p["source_spec_sha256"],
        "initial_pre_resampling_ess": None,
        "ess_definition": "Update pre-ESS is the existing particle diagnostic before resampling; post-ESS is computed separately from retained weights. Initial pre-ESS was not retained.",
        "planner": f"provisional{p['planning_scenarios']}-scenario indexed-pair MPC,protected k2,training-fitted fixed tail",
        "decision_definition": "Identical cent-rounded posted price vector; strict coefficient equality also retained. update_every=1 in these instances.",
        "reference": "ExactBelief with construction-only TemplateExactKernel; complete hidden-state TV is primary, inventory projection separate",
        "trajectory": "Fixed training-fitted coefficients on common physical validation tapes; no oracle state used in filtering",
        "rows_sha256": file_hash(context.directory / "posterior_rows.json"),
    }
