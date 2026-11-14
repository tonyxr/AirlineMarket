"""Exact reference-offer purchase-intensity ambiguity; no calibration or physics."""

import math
from dataclasses import replace

import numpy as np

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.sim.choice import choice_probabilities, systematic_utilities
from airline_market.spec import compile_market, load_spec


def compensating_shift(nonpurchase, arrival_multiplier):
    """Keep every purchased-category intensity fixed at one offered set."""
    p0 = float(nonpurchase)
    r = float(arrival_multiplier)
    q = 1.0 - p0
    if not 0 < p0 < 1 or not math.isfinite(r) or r <= q:
        raise ValueError(
            "finite interior nonpurchase and multiplier above conversion required"
        )
    return math.log(p0) - math.log(r - q)


def analyze(compiled, factors, price_increment, intensity_tolerance, rank_rtol):
    """Query native utilities only at declared reference/perturbed offered sets."""
    spec = compiled.spec
    if spec.demand.mode != "poisson" or spec.choice.model != "latent_class_mnl":
        raise ValueError("this analytical diagnostic requires latent-class Poisson MNL")
    if any(len(s.classes) != 1 for s in spec.segments):
        raise ValueError("one fixed native taste per observed segment required")
    if not all(r > 1 for r in factors):
        raise ValueError("registered factors must exceed one")
    weights = np.array(spec.demand.segment_weights.value_at(0), dtype=float)
    if abs(float(weights.sum()) - 1) > 1e-12:
        raise ValueError("segment masses must already sum to one; no normalization")
    reference_prices = np.array([p.reference_fare for p in spec.products], dtype=float)
    witnesses, summaries = [], []
    native_calls = 0

    def query(m, taste, prices, sellable):
        nonlocal native_calls
        u = systematic_utilities(compiled, m, taste, prices, sellable)
        p = choice_probabilities(u)
        native_calls += 1
        if abs(float(p.sum()) - 1) > 1e-12:
            raise ValueError("native choice mass is not one")
        return u, p

    for m, market in enumerate(spec.markets):
        js = list(compiled.market_products[m])
        lam = float(spec.demand.intensity[market.id].value_at(0))
        sellable = np.ones(len(spec.products), dtype=bool)
        tastes = [s.classes[0] for s in spec.segments]
        baseline = [query(m, t, reference_prices, sellable) for t in tastes]
        base_probs = np.array([p for _, p in baseline])
        shifts = {
            r: [compensating_shift(p[-1], r) for _, p in baseline] for r in factors
        }
        base_purchase = lam * (weights @ base_probs[:, :-1])
        columns = [base_purchase] + [
            lam * weights[k] * p[-1] * p[:-1] for k, (_, p) in enumerate(baseline)
        ]
        jacobian = np.column_stack(columns)
        null = np.array([1.0] + [-1.0 / p[-1] for _, p in baseline])
        singular = np.linalg.svd(jacobian, compute_uv=False)
        rank = int(np.count_nonzero(singular > rank_rtol * singular[0]))
        residual = float(np.max(np.abs(jacobian @ null)))
        if residual > intensity_tolerance:
            raise AssertionError("reference Jacobian null identity failed")
        core = [j for j in js if spec.products[j].carrier != "OUTSIDE"]
        if not core:
            raise ValueError("each diagnostic market must contain a core product")
        bumped = reference_prices.copy()
        bumped[core] += price_increment
        blocked = sellable.copy()
        blocked[core[0]] = False
        scenarios = [
            ("reference_all_offered", reference_prices, sellable, baseline),
            ("all_core_prices_plus_increment", bumped, sellable, None),
            ("first_core_product_unavailable", reference_prices, blocked, None),
        ]
        market_summary = {
            "market": market.id,
            "products": [spec.products[j].id for j in js],
            "reference_prices": reference_prices[js].tolist(),
            "segment_weights": weights.tolist(),
            "arrival_intensity": lam,
            "reference_nonpurchase": base_probs[:, -1].tolist(),
            "jacobian_columns": ["log_arrival"]
            + ["inside_shift_" + s.id for s in spec.segments],
            "jacobian": jacobian.tolist(),
            "null_direction": null.tolist(),
            "null_residual": residual,
            "singular_values": singular.tolist(),
            "rank": rank,
            "parameter_count": len(columns),
            "full_parameter_condition_number": None,
            "rank_rtol": rank_rtol,
            "intervention_product_ids": [spec.products[j].id for j in core],
            "blocked_product": spec.products[core[0]].id,
            "scenario_errors": [],
        }
        for label, prices, offered, original in scenarios:
            original = (
                original
                if original is not None
                else [query(m, t, prices, offered) for t in tastes]
            )
            old_intensities = lam * (weights @ np.array([p[:-1] for _, p in original]))
            for r in factors:
                transformed = []
                for k, taste in enumerate(tastes):
                    # Market-wide inside shift fixed ONCE using the reference offer;
                    # product offsets, alpha, connection penalties and segment mix stay fixed.
                    values = dict(taste.market_value)
                    values[market.id] = (
                        values.get(market.id, values.get("*", 0.0)) + shifts[r][k]
                    )
                    u, p = query(
                        m, replace(taste, market_value=values), prices, offered
                    )
                    transformed.append((u, p))
                new_intensities = (
                    r * lam * (weights @ np.array([p[:-1] for _, p in transformed]))
                )
                error = float(np.max(np.abs(new_intensities - old_intensities)))
                if label == "reference_all_offered" and error > intensity_tolerance:
                    raise AssertionError("reference ticket-intensity equality failed")
                market_summary["scenario_errors"].append(
                    {
                        "scenario": label,
                        "arrival_multiplier": r,
                        "max_absolute_purchase_intensity_difference": error,
                    }
                )
                witnesses.append(
                    {
                        "market": market.id,
                        "scenario": label,
                        "arrival_multiplier": r,
                        "segment_inside_shifts_fixed_at_reference": shifts[r],
                        "local_sellable": offered[js].tolist(),
                        "local_prices": prices[js].tolist(),
                        "original_utilities": [
                            [None if not math.isfinite(x) else float(x) for x in u]
                            for u, _ in original
                        ],
                        "transformed_utilities": [
                            [None if not math.isfinite(x) else float(x) for x in u]
                            for u, _ in transformed
                        ],
                        "original_probabilities": [p.tolist() for _, p in original],
                        "transformed_probabilities": [
                            p.tolist() for _, p in transformed
                        ],
                        "original_potential_intensity": lam,
                        "transformed_potential_intensity": r * lam,
                        "original_nonpurchase_intensity": lam
                        * float(weights @ np.array([p[-1] for _, p in original])),
                        "transformed_nonpurchase_intensity": r
                        * lam
                        * float(weights @ np.array([p[-1] for _, p in transformed])),
                        "original_purchase_intensities": old_intensities.tolist(),
                        "transformed_purchase_intensities": new_intensities.tolist(),
                    }
                )
        summaries.append(market_summary)
    return summaries, witnesses, native_calls


def run(cell, context):
    p = cell["parameters"]
    for n in ["source_spec", "design_registration"]:
        if file_hash(p[n]) != p[n + "_sha256"]:
            raise ValueError("frozen identification input changed: " + n)
    compiled = compile_market(load_spec(p["source_spec"]))
    summaries, witnesses, calls = analyze(
        compiled,
        p["arrival_multipliers"],
        p["core_price_increment"],
        p["intensity_tolerance"],
        p["rank_rtol"],
    )
    atomic_json(
        context.directory / "market_witnesses.json",
        {"markets": summaries, "native_choice_calls": calls},
    )
    atomic_json(context.directory / "native_probability_witnesses.json", witnesses)
    refs = [
        x["max_absolute_purchase_intensity_difference"]
        for s in summaries
        for x in s["scenario_errors"]
        if x["scenario"] == "reference_all_offered"
    ]
    summary = {
        "status": "completed",
        "markets": len(summaries),
        "products": len(compiled.spec.products),
        "segments": len(compiled.spec.segments),
        "reference_equivalences": len(refs),
        "reference_max_absolute_intensity_error": max(refs),
        "maximum_null_residual": max(s["null_residual"] for s in summaries),
        "reference_rank_counts": {
            str(rank): sum(s["rank"] == rank for s in summaries)
            for rank in sorted({s["rank"] for s in summaries})
        },
        "reference_parameter_count": len(compiled.spec.segments) + 1,
        "native_choice_probability_calls": calls,
        "new_simulation_cohorts": 0,
        "new_RNG_calls": 0,
        "parameter_fits": 0,
        "Q3_inputs_used": False,
        "calibration_at_optimum_Jacobian": None,
        "scientific_budget": None,
        "scope": "Reference fixed-price/all-offered purchase-intensity equivalence only. Offered-set changes are retained diagnostics, not required equalities. Not dynamic/capacity-constrained world equivalence or full empirical nonidentification.",
        "artifact_hashes": {
            n: file_hash(context.directory / n)
            for n in ["market_witnesses.json", "native_probability_witnesses.json"]
        },
    }
    assert file_hash(p["source_spec"]) == p["source_spec_sha256"]
    return summary
