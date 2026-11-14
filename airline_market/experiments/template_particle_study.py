"""Bounded hub SMC construction/update probes, not a particle-count selection."""

import hashlib
from pathlib import Path
from time import perf_counter

from airline_market.belief.cached_prior import CachedPriorParticleBelief
from airline_market.belief.events import AllOf, LegInventoryAtMost
from airline_market.belief.template_particle import TemplateParticleBelief
from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.errors import ZeroLikelihoodError
from airline_market.core.hashing import canonical_json
from airline_market.core.rng import StreamRegistry
from airline_market.core.types import CoefficientAction
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.spec import compile_market, load_spec


def fingerprint(belief):
    """Hash ordered hidden particles, float64 weights and filter RNG, streaming.

    HiddenState's noncomparing kernel pointer is excluded by its own repr;
    the physically known own ledger is separately hashed in observations.
    This avoids materializing a huge JSON array at100k particles.
    """
    digest = hashlib.sha256()
    for hidden in belief.particles:
        digest.update(repr(hidden).encode())
        digest.update(b"\n")
    digest.update(belief.weights.tobytes())
    digest.update(canonical_json(belief.rng.bit_generator.state).encode())
    return digest.hexdigest()


def summary(belief, event):
    """Keep actual pre/post-resampling ESS separate, including null initial pre-ESS."""
    diagnostic = belief.diagnostics[-1] if belief.diagnostics else None
    return {
        "particles": belief.n_particles,
        "support": len(belief.mass),
        "joint_event_probability": belief.event_probability(event),
        "pre_resampling_ESS": diagnostic["ess"] if diagnostic else None,
        "post_resampling_ESS": float(1 / (belief.weights @ belief.weights)),
        "resampled": diagnostic["resampled"] if diagnostic else None,
        "proposal_diagnostic": diagnostic,
        "fingerprint": fingerprint(belief),
        "leg_marginals": {
            str(leg): law.tolist() for leg, law in belief.leg_marginals().items()
        },
    }


def probe(cell, context):
    """Measure registered native prior/update work and preserve strict failures."""
    p = cell["parameters"]
    seed = int(p["validation_seed"])
    if not 200000 <= seed < 300000:
        raise ValueError("hub resource probes use validation physical streams")
    path = Path(p["source_spec"])
    if file_hash(path) != p["source_spec_sha256"]:
        raise ValueError("registered hub spec changed")
    compiled = compile_market(load_spec(path))
    focal, arm = p.get("focal", "A"), p.get("arm", "direct")
    env = FocalCarrierEnv(compiled, focal, arm)
    obs, _ = env.reset(seed)
    filter_seed = StreamRegistry(773, "plan/F4_hub/filter").child_seed(0)
    event = AllOf(
        tuple(LegInventoryAtMost(p["rival"], leg, 1) for leg in p["joint_legs"])
    )
    if p["implementation"] not in ("cached_reference", "template"):
        raise ValueError("unknown registered filter implementation")
    cls = (
        CachedPriorParticleBelief
        if p["implementation"] == "cached_reference"
        else TemplateParticleBelief
    )
    start = perf_counter()
    belief = cls(
        compiled,
        focal,
        arm,
        n_particles=int(p["particles"]),
        seed=filter_seed,
        inner_draws=int(p.get("inner_draws", 128)),
        strict=True,
    )
    context.counters = lambda: {
        "prior_world_copies": getattr(belief, "prior_world_copies", None),
        "kernel_engine_calls": getattr(belief.kernel, "engine_calls", None),
        "template_builds": getattr(belief.kernel, "template_builds", None),
        "particle_support": len(belief.mass),
    }
    measurements = {
        "spec_hash": compiled.spec_hash,
        "validation_seed": seed,
        "filter_seed": filter_seed,
        "implementation": p["implementation"],
        "particles": p["particles"],
        "operation": p["operation"],
        "constructor_seconds": perf_counter() - start,
        "initial_observation_sha256": hashlib.sha256(
            canonical_json(obs).encode()
        ).hexdigest(),
        "prior_reset_seconds": None,
        "update_seconds": None,
        "initial": None,
        "after_update": None,
    }
    artifact = context.directory / "measurements.json"
    atomic_json(artifact, measurements)
    stage = "prior_reset"
    try:
        context.update(
            {
                "stage": stage,
                "particles": p["particles"],
                "implementation": p["implementation"],
            }
        )
        start = perf_counter()
        belief.reset(obs)
        measurements["prior_reset_seconds"] = perf_counter() - start
        measurements["initial"] = summary(belief, event)
        atomic_json(artifact, measurements)
        if p["operation"] == "one_update":
            # This is a predeclared engineering trajectory, not a fitted or optimal policy.
            pricing = next(x for x in compiled.spec.pricing if x.carrier == focal)
            action = CoefficientAction(
                pricing.theta_grid[int(p.get("action_index", 0))]
            )
            after, _, _, _ = env.step(action)
            measurements["action"] = action.theta
            measurements["after_observation_sha256"] = hashlib.sha256(
                canonical_json(after).encode()
            ).hexdigest()
            stage = "one_update"
            context.update(
                {
                    "stage": stage,
                    "particles": p["particles"],
                    "implementation": p["implementation"],
                }
            )
            start = perf_counter()
            belief.update(obs, action, after)
            measurements["update_seconds"] = perf_counter() - start
            measurements["after_update"] = summary(belief, event)
        elif p["operation"] != "prior_reset":
            raise ValueError("unsupported probe operation")
    except ZeroLikelihoodError as error:
        measurements.update(
            failed_stage=stage,
            reason=str(error),
            error_type=type(error).__name__,
            failed_rng_sha256=hashlib.sha256(
                canonical_json(belief.rng.bit_generator.state).encode()
            ).hexdigest(),
        )
        atomic_json(artifact, measurements)
        return {
            "status": "failed",
            "reason": str(error),
            "error_type": type(error).__name__,
            "failed_stage": stage,
            "value": None,
            "throughput": None,
            "measurements_sha256": file_hash(artifact),
            "spec_hash": compiled.spec_hash,
        }
    atomic_json(artifact, measurements)
    return {
        "status": "completed",
        "spec_hash": compiled.spec_hash,
        "operation": p["operation"],
        "implementation": p["implementation"],
        "particles": p["particles"],
        "prior_reset_seconds": measurements["prior_reset_seconds"],
        "update_seconds": measurements["update_seconds"],
        "initial_fingerprint": measurements["initial"]["fingerprint"],
        "after_fingerprint": measurements["after_update"]["fingerprint"]
        if measurements["after_update"]
        else None,
        "measurements_sha256": file_hash(artifact),
        "N_selected": None,
        "full_F4_criterion": None,
        "scope": "One physical cohort, prior-reset or one-update engineering probe. Identical reference SMC proposals/weights/draws/failures under construction reuse. No64-cohort full-horizon sizing, final-planner agreement, exact hub reference,100k-particle accuracy certificate or global particle-count choice.",
    }
