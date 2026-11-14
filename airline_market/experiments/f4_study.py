"""F4 posterior/decision accuracy on fixed, paired validation trajectories.

I0/I3 are independent of the pending F1 exact scale. Their decision comparison
uses an explicitly provisional128-scenario protected planner; a final particle
choice still requires I4/hub studies and the selected F5 configuration.
"""

from pathlib import Path
import json
import time

import numpy as np

from ..belief import ExactBelief
from ..belief.cached_prior import CachedPriorParticleBelief
from ..belief.events import AllOf, LegInventoryAtMost
from ..belief.info import inventory_mass
from ..campaign.artifacts import atomic_json, file_hash
from ..campaign.instances import build_instance
from ..core.rng import StreamRegistry
from ..core.types import CoefficientAction
from ..pomdp.env import FocalCarrierEnv
from ..sim.sparse_access import sparse_access
from ..solvers.matched_mpc import MatchedScenarioMPC
from ..solvers.mpc import MPCConfig
from ..spec import compile_market


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
    physical = {
        key: p[key]
        for key in ("T", "periods", "actions", "instance_params")
        if key in p
    }
    compiled = compile_market(build_instance(cell["generator"], physical))
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
            exact = ExactBelief(compiled, focal, arm)
            exact.reset(obs)
            particle = CachedPriorParticleBelief(
                compiled,
                focal,
                arm,
                n_particles=p["particles"],
                seed=StreamRegistry(773, "plan/F4/filter").child_seed(index),
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
                    "ess": float(1 / (particle.weights @ particle.weights)),
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
        "particle_selection_status": "not_selected; I4/hub references and final F5 planner still required",
        "planner": f"provisional{p['planning_scenarios']}-scenario indexed-pair MPC,protected k2,training-fitted fixed tail",
        "decision_definition": "Identical cent-rounded posted price vector; strict coefficient equality also retained. update_every=1 in these instances.",
        "reference": "ExactBelief; complete hidden-state TV is primary, inventory projection separate",
        "trajectory": "Fixed training-fitted coefficients on common physical validation tapes; no oracle state used in filtering",
        "rows_sha256": file_hash(context.directory / "posterior_rows.json"),
    }
