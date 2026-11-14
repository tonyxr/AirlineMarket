"""Same total request law, different booking times: native ledger exact study."""

import json
import math
from collections import defaultdict
from fractions import Fraction
from itertools import product
from pathlib import Path

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.experiments.fixed_fare_study import (
    classical_reference,
    native_ledger_tree,
)
from airline_market.sim.fixed_fare_admission import FixedFareAdmissionEngine
from airline_market.spec import compile_market, load_spec


def total_count_law(probabilities):
    """Exact multivariate count PGF, including zero-probability count entries."""
    n = len(probabilities[0])
    law = {(0,) * n: Fraction(1)}
    for row in probabilities:
        ps = [Fraction(str(x)) for x in row]
        if len(row) != n or min(ps) < 0 or sum(ps) > 1:
            raise ValueError("invalid categorical request law")
        next_law = defaultdict(Fraction)
        for counts, mass in law.items():
            next_law[counts] += mass * (1 - sum(ps))
            for j, prob in enumerate(ps):
                new = tuple(x + int(k == j) for k, x in enumerate(counts))
                next_law[new] += mass * prob
        law = next_law
    T = len(probabilities)
    return {
        counts: law.get(counts, Fraction(0))
        for counts in product(range(T + 1), repeat=n)
    }


def independent_binomial_law(probabilities):
    """Separate product of binomial PMFs for one-class-only independent periods."""
    n = len(probabilities[0])
    active = [[] for _ in range(n)]
    for row in probabilities:
        indices = [j for j, p in enumerate(row) if p > 0]
        if len(indices) != 1:
            raise ValueError(
                "timing benchmark needs exactly one positive class per period"
            )
        j = indices[0]
        active[j].append(Fraction(str(row[j])))
    if any(not ps or len(set(ps)) != 1 for ps in active):
        raise ValueError("each class needs one common Bernoulli rate")
    T = len(probabilities)
    result = {}
    for counts in product(range(T + 1), repeat=n):
        mass = Fraction(1)
        for k, ps in zip(counts, active):
            trials, q = len(ps), ps[0]
            mass *= (
                Fraction(0)
                if k > trials
                else math.comb(trials, k) * q**k * (1 - q) ** (trials - k)
            )
        result[counts] = mass
    return result


def offline_bound(law, capacity, margins):
    """Exact best-C-fares bound with total future counts known to an analyst only."""
    value = Fraction(0)
    order = sorted(range(len(margins)), key=lambda j: margins[j], reverse=True)
    for counts, mass in law.items():
        stock, gain = capacity, Fraction(0)
        for j in order:
            admitted = min(stock, counts[j]) if margins[j] > 0 else 0
            gain += admitted * margins[j]
            stock -= admitted
        value += mass * gain
    return value


def evaluate(source, tolerance, max_nodes):
    """Solve only one new chronological native ledger problem and exact reference."""
    c = compile_market(load_spec(source["physical_spec"]))
    if len(c.spec.legs) != 1 or any(len(p.legs) != 1 for p in c.spec.products):
        raise ValueError("declared timing diagnostic is a single shared resource")
    law = total_count_law(source["class_probabilities"])
    assert (
        law == independent_binomial_law(source["class_probabilities"])
        and sum(law.values()) == 1
    )
    reference = classical_reference(source)
    bound = offline_bound(law, reference["capacities"][0], reference["margins"])
    engine = FixedFareAdmissionEngine(c, source["class_probabilities"])
    value, nodes = native_ledger_tree(engine, max_nodes)
    errors = [
        abs(
            n["native_final_value_USD"]
            - n["already_booked_net_USD"]
            - float(reference["table"][n["t"], tuple(n["stock"])])
        )
        for n in nodes
    ]
    assert max(errors) <= tolerance and value <= float(bound) + tolerance
    return {
        "case": source["case"],
        "timing_factors": source["timing_factors"],
        "native_DP_USD": value,
        "exact_DP_fraction_USD": str(reference["table"][0, reference["capacities"]]),
        "offline_total_counts_bound_fraction_USD": str(bound),
        "max_native_classical_error_USD": max(errors),
        "native_history_nodes": len(nodes),
        "joint_total_count_law": [
            {"counts": list(k), "mass": str(v)} for k, v in law.items()
        ],
        "classical_rows": reference["rows"],
        "native_nodes": nodes,
    }


def run(cell, context):
    p = cell["parameters"]
    guards = [(x["path"], x["sha256"]) for x in p["problems"]] + [
        (p["design_registration"], p["design_registration_sha256"])
    ]
    if any(file_hash(n) != h for n, h in guards):
        raise ValueError("frozen timing input changed")
    output = context.directory / "case_witnesses"
    output.mkdir()
    summaries = []
    groups = defaultdict(list)
    artifacts = {}
    physical = {}
    for item in p["problems"]:
        source = json.loads(Path(item["path"]).read_text())
        result = evaluate(source, p["tolerance"], p["max_nodes"])
        factor = source["timing_factors"]
        key = (factor["capacity"], factor["high_request_probability"])
        # Only period order changes within a registered corner; no physical or cost edits.
        if key in physical:
            assert physical[key] == source["physical_spec"]
        physical[key] = source["physical_spec"]
        destination = output / (source["case"] + ".json")
        atomic_json(destination, result)
        artifacts[str(destination.relative_to(context.directory))] = file_hash(
            destination
        )
        summary = {
            k: v
            for k, v in result.items()
            if k not in ["classical_rows", "native_nodes"]
        }
        summary.update(problem=item["path"], problem_sha256=item["sha256"])
        summaries.append(summary)
        groups[key].append(summary)
    corners = []
    for (capacity, high), rows in sorted(groups.items()):
        assert all(
            x["joint_total_count_law"] == rows[0]["joint_total_count_law"] for x in rows
        )
        assert all(
            x["offline_total_counts_bound_fraction_USD"]
            == rows[0]["offline_total_counts_bound_fraction_USD"]
            for x in rows
        )
        values = [Fraction(x["exact_DP_fraction_USD"]) for x in rows]
        corners.append(
            {
                "capacity": capacity,
                "high_request_probability": high,
                "orders": [x["timing_factors"]["order"] for x in rows],
                "exact_value_fractions_USD": {
                    x["timing_factors"]["order"]: x["exact_DP_fraction_USD"]
                    for x in rows
                },
                "min_DP_fraction_USD": str(min(values)),
                "max_DP_fraction_USD": str(max(values)),
                "range_fraction_USD": str(max(values) - min(values)),
                "common_offline_total_counts_bound_fraction_USD": rows[0][
                    "offline_total_counts_bound_fraction_USD"
                ],
            }
        )
    atomic_json(
        context.directory / "timing_summary.json",
        {"cases": summaries, "corners": corners},
    )
    artifacts["timing_summary.json"] = file_hash(
        context.directory / "timing_summary.json"
    )
    assert all(file_hash(n) == h for n, h in guards)
    return {
        "status": "completed",
        "problems": len(summaries),
        "corners": len(corners),
        "native_history_nodes": sum(x["native_history_nodes"] for x in summaries),
        "max_native_classical_error_USD": max(
            x["max_native_classical_error_USD"] for x in summaries
        ),
        "joint_count_laws_exactly_equal_within_corners": True,
        "all_DP_within_offline_count_bound": True,
        "corner_value_ranges_fraction_USD": [x["range_fraction_USD"] for x in corners],
        "new_sampled_physical_cohorts": 0,
        "RNG_calls": 0,
        "old_problem_solver_or_DLP_reruns": 0,
        "Q2_or_Q3_fit": False,
        "full_P3_or_V4_acceptance": None,
        "scientific_budget": None,
        "artifact_hashes": artifacts,
        "scope": "Synthetic timing-identification prerequisite, all declared orders retained; no estimated empirical booking curve or adoption into I9. Native admission knows current request/stock; Bellman continuation knows distributions, not future realizations.",
    }
