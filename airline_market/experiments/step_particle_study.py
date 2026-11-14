"""Paired one-hypothesis cold proposal costs, not full filter accuracy or N selection."""

import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
from airline_market.belief.step_particle import StepTapeParticleBelief
from airline_market.sim.step_tape import OneStepTape, PeriodRow

from airline_market.belief.template_particle import TemplateParticleBelief
from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.hashing import canonical_json
from airline_market.core.rng import StreamRegistry
from airline_market.core.types import CoefficientAction
from airline_market.experiments.template_particle_study import fingerprint
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.sim.tape import ExogenousTape
from airline_market.spec import compile_market, load_spec


def proposal_fingerprint(mass, rng):
    """Retain every weight, ordered key and full hypothetical kernel state."""
    digest = hashlib.sha256()
    for hidden, weight in sorted(mass.items()):
        digest.update(canonical_json((hidden, hidden.kernel, weight)).encode())
        digest.update(b"\n")
    digest.update(canonical_json(rng.bit_generator.state).encode())
    return digest.hexdigest()


def array_bytes(tape):
    """Count actually stored NumPy buffers, not full Python objects or stage RSS."""
    return sum(
        value.nbytes
        if isinstance(value, np.ndarray)
        else value.value.nbytes
        if isinstance(value, PeriodRow) and isinstance(value.value, np.ndarray)
        else 0
        for value in vars(tape).values()
    )


def compare(cell, context):
    """Run unchanged paired128-draw proposals after identical prior setup."""
    p = cell["parameters"]
    path = Path(p["source_spec"])
    if file_hash(path) != p["source_spec_sha256"]:
        raise ValueError("changed step-tape physical input")
    c = compile_market(load_spec(path))
    physical_seed = p["validation_seed"]
    if not 200000 <= physical_seed < 300000:
        raise ValueError("step-tape engineering cases require validation streams")
    env = FocalCarrierEnv(c, "A", "direct")
    before, _ = env.reset(physical_seed)
    action = CoefficientAction(c.spec.pricing[0].theta_grid[0])
    after, _, _, _ = env.step(action)
    filter_seed = StreamRegistry(773, "plan/F4_hub/filter").child_seed(0)
    classes = {"original": TemplateParticleBelief, "step_tape": StepTapeParticleBelief}
    result = {
        "spec_hash": c.spec_hash,
        "validation_seed": physical_seed,
        "filter_seed": filter_seed,
        "particles": p["particles"],
        "inner_draws": p["inner_draws"],
        "cases": {},
    }
    artifact = context.directory / "measurements.json"
    outputs = {}
    priors = {}
    active = {}
    context.counters = lambda: (
        {
            "kernel_engine_calls": active["belief"].kernel.engine_calls,
            "prior_support": len(active["belief"].mass),
        }
        if active
        else {}
    )
    for label in p["execution_order"]:
        b = classes[label](
            c,
            "A",
            "direct",
            n_particles=p["particles"],
            seed=filter_seed,
            inner_draws=p["inner_draws"],
            strict=True,
        )
        active["belief"] = b
        context.update({"stage": "prior_setup", "implementation": label})
        start = perf_counter()
        b.reset(before)
        prior_seconds = perf_counter() - start
        priors[label] = fingerprint(b)
        if (
            p.get("expected_prior_fingerprint")
            and priors[label] != p["expected_prior_fingerprint"]
        ):
            raise AssertionError("prior differs from sealed engineering law")
        context.update({"stage": "one_hypothesis_proposal", "implementation": label})
        calls = b.kernel.engine_calls
        start = perf_counter()
        mass = b.conditional_successors(b.particles[0], before, action, after)
        elapsed = perf_counter() - start
        outputs[label] = (mass, canonical_json(b.rng.bit_generator.state))
        result["cases"][label] = {
            "prior_reset_seconds": prior_seconds,
            "prior_fingerprint": priors[label],
            "proposal_seconds": elapsed,
            "successor_states": len(mass),
            "estimated_joint_observation_mass": sum(mass.values()),
            "kernel_engine_calls": b.kernel.engine_calls - calls,
            "proposal_and_rng_fingerprint": proposal_fingerprint(mass, b.rng),
        }
        atomic_json(artifact, result)
    if len(set(priors.values())) != 1 or outputs["original"] != outputs["step_tape"]:
        raise AssertionError("proposal mass or RNG differs")
    if (
        result["cases"]["original"]["proposal_and_rng_fingerprint"]
        != result["cases"]["step_tape"]["proposal_and_rng_fingerprint"]
    ):
        raise AssertionError("complete hypothetical kernel state differs")
    # Separate paired buffer measurement, outside proposal timing.
    seed = StreamRegistry(773, "plan/step_tape/resource").child_seed(0)
    original = ExogenousTape.generate(c, seed)
    revised = OneStepTape.generate(c, seed, period=0)
    np.testing.assert_array_equal(original.gumbel[0], revised.gumbel[0])
    result.update(
        status="completed",
        prior_identity=True,
        proposal_full_state_mass_rng_identity=True,
        original_stored_array_bytes=array_bytes(original),
        step_stored_array_bytes=array_bytes(revised),
        original_gumbel_bytes=original.gumbel.nbytes,
        step_gumbel_bytes=revised.gumbel[0].nbytes,
        single_case_proposal_wall_ratio=result["cases"]["original"]["proposal_seconds"]
        / result["cases"]["step_tape"]["proposal_seconds"],
        update_throughput=None,
        N_selected=None,
        scope="One fixed initial hypothesis per hub, paired unchanged draws/proposal/zero estimates. Timing excludes prior setup/fingerprints/buffer audit; one case per method, no randomized throughput or full update/100k accuracy/F3 pass. Stored NumPy bytes exclude Python marks and allocator/temporary peaks. Original full-TV failure and old limits remain unchanged.",
    )
    atomic_json(artifact, result)
    return result
