"""MR1: native exact Bernoulli shopping at reachable states, other fares fixed."""

import math
from pathlib import Path

import numpy as np

from airline_market.campaign.artifacts import file_hash
from airline_market.core.hashing import canonical_json, jsonable, schema_hash
from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.sim.engine import MarketEngine
from airline_market.sim.kernel import ExactKernel, PostedPrices
from airline_market.spec import compile_market, dump_spec, load_spec


def contract(c):
    if (
        c.spec.demand.mode != "single_request_bernoulli"
        or c.spec.choice.model != "latent_class_mnl"
        or len(c.spec.carriers) < 2
        or any(t.alpha <= 0 for s in c.spec.segments for t in s.classes)
    ):
        raise ValueError(
            "positive-alpha Bernoulli latent-class MNL with rivals required"
        )
    bounds = [
        (math.ceil(100 * p.fare_floor), math.floor(100 * p.fare_cap))
        for p in c.spec.products
    ]
    if any(
        not math.isfinite(p.fare_floor) or not math.isfinite(p.fare_cap)
        for p in c.spec.products
    ) or any(hi <= lo for lo, hi in bounds):
        raise ValueError(
            "at least two finite attainable cent fares per product required"
        )
    return bounds


def cent_vector(bounds, rng):
    """Public bounds only; cap excluded so every product can be strictly raised."""
    return tuple(float(rng.integers(lo, hi)) / 100 for lo, hi in bounds)


def actions(c, prices):
    return {
        i: PriceAction({j: prices[j] for j in products})
        for i, products in enumerate(c.products_of)
    }


def parent_fingerprint(engine, kernel):
    return schema_hash(
        {
            "state": kernel.encode(engine.world),
            "tape": schema_hash(engine.tape),
            "controllers": {
                i: {"memory": ctrl.snapshot(), "rng": ctrl.rng.bit_generator.state}
                for i, ctrl in engine.controllers.items()
            },
        }
    )


def expected_new_bookings(kernel, state, prices, tolerance):
    """Full raw native Bernoulli shopping outcomes; no mass repair or resampling."""
    prices = tuple(map(float, prices))
    posted = PostedPrices(
        state,
        0,
        PriceAction({j: prices[j] for j in kernel.compiled.products_of[0]}),
        prices,
    )
    rows = kernel.period_outcomes(state, posted)
    mass = math.fsum(p for p, _ in rows)
    if any(not math.isfinite(p) or p < 0 for p, _ in rows) or abs(mass - 1) > tolerance:
        raise ValueError("raw native outcome masses invalid; no renormalization")
    values = np.zeros(len(prices))
    saved = []
    for probability, outcome in rows:
        if len(outcome.requests) != len(outcome.choices) or len(outcome.choices) > 1:
            raise ValueError("Bernoulli outcome contract changed")
        for j in outcome.choices:
            if j is not None:
                values[j] += probability
        saved.append(
            {
                "mass": probability,
                "requests": jsonable(outcome.requests),
                "choices": outcome.choices,
            }
        )
    return values, mass, saved


def compare_change(c, base_values, raised_values, changed, tolerance):
    owner = int(c.product_carrier[changed])
    rivals = [j for j in range(len(base_values)) if int(c.product_carrier[j]) != owner]
    delta = raised_values - base_values
    rival_sum = float(np.sum(delta[rivals]))
    violation = max(
        0.0,
        float(delta[changed]),
        max([-float(delta[j]) for j in rivals] or [0.0]),
        -rival_sum,
    )
    return {
        "changed_product": changed,
        "changed_owner": owner,
        "own_product_delta": float(delta[changed]),
        "rival_product_deltas": {j: float(delta[j]) for j in rivals},
        "rival_aggregate_delta": rival_sum,
        "maximum_violation": violation,
        "pass": violation <= tolerance,
    }


def run(cell, context):
    p = cell["parameters"]
    if not 200000 <= p["seed_start"] < p["seed_start"] + p["n_states"] <= 300000:
        raise ValueError("validation seed-prefix range required")
    guards = [
        (Path(p["source_spec"]), p["source_spec_sha256"]),
        (Path(p["design_registration"]), p["design_registration_sha256"]),
    ]
    for path, expected in guards:
        if file_hash(path) != expected:
            raise ValueError("frozen monotonicity input changed")
    c = compile_market(load_spec(p["source_spec"]))
    if c.spec_hash != p["compiled_spec_hash"]:
        raise ValueError("compiled saved physical source changed")
    bounds = contract(c)
    dump_spec(c.spec, context.directory / "spec.json")
    engine, kernel = MarketEngine(c), ExactKernel(c)
    registry = StreamRegistry(773, "planning/V3/MR1/native-reachable")
    current = {
        "stage": "reachable_state_monotonicity",
        "states_completed": 0,
        "states_total": p["n_states"],
        "comparisons_completed": 0,
        "native_prefix_steps": 0,
    }
    context.counters = lambda: dict(current)
    maximum, mass_error, strict_own, zero_own, failures = 0.0, 0.0, 0, 0, 0
    counts_by_period = {}
    with (context.directory / "state_comparisons.jsonl").open("x") as log:
        for index in range(p["n_states"]):
            seed = p["seed_start"] + index
            rng = registry.generator(seed)
            target = int(rng.integers(c.spec.horizon.n_periods))
            engine.reset(seed)
            prefix = []
            for _ in range(target):
                prices = cent_vector(bounds, rng)
                engine.step(actions(c, prices))
                prefix.append(prices)
                current["native_prefix_steps"] += 1
            state = kernel.encode(engine.world)
            before = parent_fingerprint(engine, kernel)
            base_prices = cent_vector(bounds, rng)
            base_values, mass, base_outcomes = expected_new_bookings(
                kernel, state, base_prices, p["tolerance"]
            )
            mass_error = max(mass_error, abs(mass - 1))
            changes = []
            for j, (_, cap) in enumerate(bounds):
                raised = list(base_prices)
                base_cent = round(base_prices[j] * 100)
                raised[j] = float(rng.integers(base_cent + 1, cap + 1)) / 100
                values, mass, outcomes = expected_new_bookings(
                    kernel, state, raised, p["tolerance"]
                )
                row = compare_change(c, base_values, values, j, p["tolerance"])
                row.update(
                    raised_prices=raised,
                    raised_expected_new_bookings=values.tolist(),
                    raw_mass=mass,
                    raw_outcomes=outcomes,
                )
                changes.append(row)
                maximum = max(maximum, row["maximum_violation"])
                mass_error = max(mass_error, abs(mass - 1))
                strict_own += row["own_product_delta"] < -p["tolerance"]
                zero_own += abs(row["own_product_delta"]) <= p["tolerance"]
                failures += not row["pass"]
                current["comparisons_completed"] += 1
                if parent_fingerprint(engine, kernel) != before:
                    raise AssertionError(
                        "exact shopping queries mutated parent state/tape/controller/RNG"
                    )
            counts_by_period[target] = counts_by_period.get(target, 0) + 1
            log.write(
                canonical_json(
                    {
                        "index": index,
                        "seed": seed,
                        "target_period": target,
                        "native_prefix_prices": prefix,
                        "kernel_state": jsonable(state),
                        "parent_before_after_sha256": before,
                        "parent_unchanged": True,
                        "tape_sha256": schema_hash(engine.tape),
                        "base_prices": base_prices,
                        "base_expected_new_bookings": base_values.tolist(),
                        "base_raw_mass": math.fsum(x["mass"] for x in base_outcomes),
                        "base_raw_outcomes": base_outcomes,
                        "changes": changes,
                    }
                )
                + "\n"
            )
            log.flush()
            current["states_completed"] = index + 1
    for path, expected in guards:
        if file_hash(path) != expected:
            raise ValueError("frozen input changed during operation")
    return {
        "status": "completed",
        "spec_hash": c.spec_hash,
        "instance": p["instance"],
        "reachable_states": p["n_states"],
        "fresh_physical_seed_prefixes": p["n_states"],
        "complete_physical_horizon_cohorts": 0,
        "native_prefix_steps": current["native_prefix_steps"],
        "comparisons": current["comparisons_completed"],
        "kernel_shopping_evaluations": p["n_states"] * (len(bounds) + 1),
        "states_by_period": counts_by_period,
        "max_raw_mass_error": mass_error,
        "maximum_monotonicity_violation": maximum,
        "violating_comparisons": failures,
        "strict_own_decreases": int(strict_own),
        "numerical_zero_own_deltas": int(zero_own),
        "MR1_saved_instance_pass": failures == 0 and mass_error <= p["tolerance"],
        "all_parent_state_tape_controller_RNG_checks_pass": True,
        "actor_forwards": 0,
        "optimizer_updates": 0,
        "full_V3_pass": None,
        "scientific_budget": None,
        "artifact_hashes": {
            name: file_hash(context.directory / name)
            for name in ["spec.json", "state_comparisons.jsonl"]
        },
        "scope": "Native Bernoulli shopping-substep new-booking monotonicity at reachable explicit-price-action states; all other fares fixed, no controller response. Not net post-cancellation bookings, strategic profit/value, Poisson/mixed-logit or full V3/V2.",
    }
