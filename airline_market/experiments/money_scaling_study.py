"""MR5 native monetary scaling: aligned-price law and same-history rounded rules."""

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np

from airline_market.agents.price_rule import AdditivePriceRule
from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.hashing import canonical_json, jsonable, schema_hash
from airline_market.core.rng import StreamRegistry
from airline_market.core.types import CoefficientAction, PriceAction
from airline_market.experiments.price_monotonicity_study import expected_new_bookings
from airline_market.sim.accounting import carrier_metrics
from airline_market.sim.engine import MarketEngine
from airline_market.sim.kernel import ExactKernel
from airline_market.sim.tape import ExogenousTape
from airline_market.spec import compile_market, load_spec

DIMENSIONLESS = {
    "intercept",
    "connection_indicator",
    "scarcity_sum",
    "scarcity_max",
    "time_to_departure",
    "product_indicator",
    "booking_pace",
    "rival_availability",
    "public_disruption",
}


def scaled_spec(data, factor):
    """Explicit dimensional transform; native cent quantum stays one cent."""
    if factor not in (2, 4):
        raise ValueError("registered integer monetary factors are2and4")
    s = copy.deepcopy(data)
    for leg in s["legs"]:
        leg["cost_per_pax"] *= factor
    for prod in s["products"]:
        for name in ["service_cost", "fare_floor", "fare_cap", "reference_fare"]:
            prod[name] *= factor
    for segment in s["segments"]:
        if segment.get("mixed"):
            raise ValueError("finite latent-class contract only")
        for taste in segment["classes"]:
            taste["alpha"] /= factor
    s["resolution"]["penalty_per_cancellation"] *= factor
    for q in s["quotes"]["requests"]:
        q["fare_bin"] *= factor
        q["fare_noise_sd"] *= factor
    for price in s["pricing"]:
        if any(f["name"] not in DIMENSIONLESS for f in price["controllable_features"]):
            raise ValueError("only declared noncurrency controllable features")
        price["base"] = {key: value * factor for key, value in price["base"].items()}
        price["time_ramp"] *= factor
        price["theta_bounds"] = [
            [v * factor for v in pair] for pair in price["theta_bounds"]
        ]
        if price["theta_grid"] is not None:
            price["theta_grid"] = [
                [v * factor for v in row] for row in price["theta_grid"]
            ]
        for feature in price["fixed_features"]:
            f, weight = feature
            if f["name"] in {"direct_rival_gap", "min_rival_gap"}:
                if "missing_value" in f["params"]:
                    f["params"]["missing_value"] *= factor
            elif f["name"] in DIMENSIONLESS:
                feature[1] = weight * factor
            else:
                raise ValueError("undeclared fixed-feature currency dimension")
    for assignment in s["controllers"]:
        if assignment["controller"] == "external":
            continue
        if assignment["controller"] != "threshold_linear_rule":
            raise ValueError("undeclared controller monetary transform")
        for rule in assignment["params"]["rules"].values():
            rule["base"] *= factor
            for term in rule.get("scarcity", []):
                term["coefficient"] *= factor
            for term in rule.get("responses", []):
                term["reference"] *= factor
    for b in s["background"]:
        for key in ["base", "scarcity_premium"]:
            if key in b["quote_rule"]:
                b["quote_rule"][key] *= factor
        b["purchase"]["slope"] /= factor
    s["timing"]["initial_previous_prices"] = {
        k: v * factor for k, v in s["timing"]["initial_previous_prices"].items()
    }
    return s


def scale_observation(obs, factor):
    if obs.privileged is not None:
        raise ValueError("shadow price rule requires nonprivileged observation")
    return replace(
        obs,
        own_bookings=tuple(replace(b, fare=b.fare * factor) for b in obs.own_bookings),
        own_cash=obs.own_cash * factor,
        own_refunds=obs.own_refunds * factor,
        own_previous_prices=tuple(x * factor for x in obs.own_previous_prices),
        own_posted_prices=tuple(x * factor for x in obs.own_posted_prices),
        quotes=tuple(
            replace(q, fare=q.fare * factor if q.fare is not None else None)
            for q in obs.quotes
        ),
    )


class ObservedPriceEngine(MarketEngine):
    """Capture each actual native pricing observation without an additional act."""

    def _action_prices(self, i, action, obs):
        result = super()._action_prices(i, action, obs)
        self.actual_price_inputs[i] = (
            action,
            obs,
            tuple(self.world.carriers[i].posted_prices),
        )
        return result

    def reset(self, seed=None, *, tape=None):
        self.actual_price_inputs = {}
        return super().reset(seed, tape=tape)


def profile_structure(engine):
    """Monetary-free realized bookings/capacity/events and request choices."""
    w = engine.world
    return {
        "t": w.t,
        "exo": w.exo,
        "events": sorted(w.events_fired),
        "carriers": [
            {
                "capacity": cs.capacity.tolist(),
                "bookings": [
                    (
                        b.record_id,
                        b.product,
                        b.period,
                        b.arrival_rank,
                        b.segment,
                        b.taste_class,
                    )
                    for b in cs.bookings
                ],
                "canceled": cs.canceled,
            }
            for cs in w.carriers
        ],
        "background_stocks": [x.stock for x in w.background],
        "requests": [
            {
                key: row[key]
                for key in ["period", "market", "segment", "chosen", "unavailable"]
            }
            for row in w.request_log
        ],
    }


def money_residual(original, scaled, factor):
    names = ["revenue", "refunds", "penalties", "service_cost", "profit"]
    return max(abs(scaled[name] - factor * original[name]) for name in names)


def run(cell, context):
    p = cell["parameters"]
    if not 200000 <= p["seed_start"] < p["seed_start"] + p["n_cohorts"] <= 300000:
        raise ValueError("validation full-cohort range required")
    guards = [
        (Path(p["source_spec"]), p["source_spec_sha256"]),
        (Path(p["design_registration"]), p["design_registration_sha256"]),
    ]
    guards += [(Path(x["source_spec"]), x["source_spec_sha256"]) for x in p["scaled"]]
    for path, expected in guards:
        if file_hash(path) != expected:
            raise ValueError("frozen monetary input changed")
    c = compile_market(load_spec(p["source_spec"]))
    if (
        c.spec_hash != p["compiled_spec_hash"]
        or c.spec.choice.model != "latent_class_mnl"
    ):
        raise ValueError("saved compiled finite-MNL source changed")
    factors = [x["factor"] for x in p["scaled"]]
    if factors != [2, 4]:
        raise ValueError("registered two-factor design required")
    scaled = []
    for variant in p["scaled"]:
        v = compile_market(load_spec(variant["source_spec"]))
        if v.spec_hash != variant["compiled_spec_hash"]:
            raise ValueError("scaled compiled source changed")
        scaled.append(v)
    base = ObservedPriceEngine(c)
    controls = [MarketEngine(v) for v in scaled]
    kernels = [ExactKernel(x) for x in [c, *scaled]]
    rules = [AdditivePriceRule(v, v.spec.pricing[0]) for v in scaled]
    grids = c.spec.pricing[0].theta_grid
    if not grids:
        raise ValueError("declared source requires a finite focal coefficient grid")
    current = {
        "stage": "paired_money_scaling",
        "cohorts_completed": 0,
        "cohorts_total": p["n_cohorts"],
        "native_profile_steps": 0,
    }
    context.counters = lambda: dict(current)
    summary = [
        {
            "factor": a,
            "price_rule_rows": 0,
            "strict_cent_price_mismatches": 0,
            "max_posted_price_scaling_error_USD": 0.0,
            "max_choice_probability_error": 0.0,
            "max_raw_mass_error": 0.0,
            "structural_mismatch_periods": 0,
            "max_currency_residual_USD": 0.0,
        }
        for a in factors
    ]
    registry = StreamRegistry(773, "planning/V3/MR5/coefficient-grid")
    with (context.directory / "cohort_rows.jsonl").open("x") as log:
        for k in range(p["n_cohorts"]):
            seed = p["seed_start"] + k
            tape = ExogenousTape.generate(c, seed)
            tape_hash = schema_hash(tape)
            base.reset(seed, tape=tape)
            for engine in controls:
                engine.reset(seed, tape=tape)
                if engine.tape is not tape:
                    raise AssertionError("actual exogenous tape object not shared")
            rng = registry.generator(seed)
            period_rows = []
            for t in range(c.spec.horizon.n_periods):
                before = [
                    kernel.encode(engine.world)
                    for kernel, engine in zip(kernels, [base, *controls])
                ]
                theta = tuple(grids[int(rng.integers(len(grids)))])
                base.step({0: CoefficientAction(theta)})
                prices = tuple(base._prices_global())
                base_prob, mass, base_outcomes = expected_new_bookings(
                    kernels[0], before[0], prices, p["probability_tolerance"]
                )
                action, observation, focal_prices = base.actual_price_inputs[0]
                if (
                    not isinstance(action, CoefficientAction)
                    or observation.privileged is not None
                ):
                    raise AssertionError(
                        "native focal coefficient/public-observation contract changed"
                    )
                variants = []
                for index, (factor, engine, v, kernel, rule) in enumerate(
                    zip(factors, controls, scaled, kernels[1:], rules)
                ):
                    lifted = scale_observation(observation, factor)
                    predicted = rule.prices(tuple(x * factor for x in theta), lifted)
                    expected_prices = np.array(focal_prices) * factor
                    errors = predicted - expected_prices
                    point = summary[index]
                    point["price_rule_rows"] += len(focal_prices)
                    point["strict_cent_price_mismatches"] += int(
                        np.sum(np.abs(errors) > p["monetary_tolerance"])
                    )
                    point["max_posted_price_scaling_error_USD"] = max(
                        point["max_posted_price_scaling_error_USD"],
                        float(np.max(np.abs(errors))),
                    )
                    aligned = tuple(float(x * factor) for x in prices)
                    control_actions = {
                        i: PriceAction({j: aligned[j] for j in prods})
                        for i, prods in enumerate(v.products_of)
                    }
                    engine.step(control_actions)
                    probs, scaled_mass, outcomes = expected_new_bookings(
                        kernel, before[index + 1], aligned, p["probability_tolerance"]
                    )
                    probability_error = float(np.max(np.abs(probs - base_prob)))
                    structure_match = profile_structure(base) == profile_structure(
                        engine
                    )
                    accounting = [
                        carrier_metrics(engine.world, v, i)
                        for i in range(len(v.spec.carriers))
                    ]
                    original = [
                        carrier_metrics(base.world, c, i)
                        for i in range(len(c.spec.carriers))
                    ]
                    residual = max(
                        money_residual(a, b, factor)
                        for a, b in zip(original, accounting)
                    )
                    point["max_choice_probability_error"] = max(
                        point["max_choice_probability_error"], probability_error
                    )
                    point["max_raw_mass_error"] = max(
                        point["max_raw_mass_error"], abs(mass - 1), abs(scaled_mass - 1)
                    )
                    point["structural_mismatch_periods"] += not structure_match
                    point["max_currency_residual_USD"] = max(
                        point["max_currency_residual_USD"], residual
                    )
                    variants.append(
                        {
                            "factor": factor,
                            "scaled_permitted_observation": jsonable(lifted),
                            "shadow_native_coefficient_prices": predicted.tolist(),
                            "factor_times_original_focal_prices": expected_prices.tolist(),
                            "posted_price_scaling_error_USD": errors.tolist(),
                            "aligned_posted_prices": aligned,
                            "control_expected_new_bookings": probs.tolist(),
                            "choice_probability_error": probability_error,
                            "raw_mass": scaled_mass,
                            "raw_outcomes": outcomes,
                            "structure_match": structure_match,
                            "base_accounting": original,
                            "control_accounting": accounting,
                            "max_currency_residual_USD": residual,
                        }
                    )
                period_rows.append(
                    {
                        "period": t,
                        "theta": theta,
                        "native_permitted_observation": jsonable(observation),
                        "native_focal_prices": focal_prices,
                        "base_posted_prices": prices,
                        "base_expected_new_bookings": base_prob.tolist(),
                        "base_raw_mass": mass,
                        "base_raw_outcomes": base_outcomes,
                        "variants": variants,
                    }
                )
                current["native_profile_steps"] += 3
            if schema_hash(tape) != tape_hash or any(
                engine.tape is not tape for engine in [base, *controls]
            ):
                raise AssertionError("shared original exogenous tape changed")
            log.write(
                canonical_json(
                    {
                        "seed": seed,
                        "tape_sha256": tape_hash,
                        "shared_actual_tape_object": True,
                        "periods": period_rows,
                    }
                )
                + "\n"
            )
            log.flush()
            current["cohorts_completed"] = k + 1
    for point in summary:
        point["native_same_history_coefficient_scaling_pass"] = (
            point["strict_cent_price_mismatches"] == 0
        )
        point["aligned_price_choice_accounting_pass"] = (
            point["structural_mismatch_periods"] == 0
            and point["max_choice_probability_error"] <= p["probability_tolerance"]
            and point["max_currency_residual_USD"] <= p["monetary_tolerance"]
        )
    for path, expected in guards:
        if file_hash(path) != expected:
            raise ValueError("frozen input changed during operation")
    atomic_json(context.directory / "scaling_summary.json", summary)
    return {
        "status": "completed",
        "spec_hash": c.spec_hash,
        "instance": p["instance"],
        "physical_cohorts": p["n_cohorts"],
        "native_profile_cohorts": p["n_cohorts"] * 3,
        "native_profile_steps": current["native_profile_steps"],
        "results_by_factor": summary,
        "full_V3_pass": None,
        "scientific_budget": None,
        "optimizer_updates": 0,
        "artifact_hashes": {
            name: file_hash(context.directory / name)
            for name in ["cohort_rows.jsonl", "scaling_summary.json"]
        },
        "scope": "Native same-history focal coefficient-rule monetary equivariance, plus two full aligned-posted-price control cohorts sharing one actual tape. No autonomous scaled-controller trajectory, tick-size amendment, full MR5/V3, learning or empirical claim.",
    }
