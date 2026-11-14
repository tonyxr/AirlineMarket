"""V4 fixed-fare native-ledger tree versus independent inventory recursion/LP."""

import json
import math
from fractions import Fraction
from itertools import product
from pathlib import Path

from airline_market.sim.fixed_fare_admission import FixedFareAdmissionEngine

from airline_market.agents.bidprice import solve_dlp
from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.hashing import jsonable
from airline_market.experiments.dlp_dual_study import exact_reference, native_comparison
from airline_market.sim.accounting import carrier_metrics
from airline_market.spec import compile_market, load_spec


def classical_reference(source):
    """Bottom-up exact rational inventory recursion; no native ledger/kernel call."""
    s = source["physical_spec"]
    ps = source["class_probabilities"]
    legs = s["legs"]
    products = s["products"]
    A = [[int(l["id"] in p["legs"]) for p in products] for l in legs]
    costs = {l["id"]: Fraction(str(l.get("cost_per_pax", 0))) for l in legs}
    margins = [
        Fraction(str(p["reference_fare"]))
        - Fraction(str(p["service_cost"]))
        - sum((costs[l] for l in p["legs"]), Fraction(0))
        for p in products
    ]
    capacities = tuple(l["capacity"]["support"][0] for l in legs)
    grid = list(product(*(range(k + 1) for k in capacities)))
    table = {(len(ps), x): Fraction(0) for x in grid}
    rows = []
    for t in reversed(range(len(ps))):
        probs = [Fraction(str(p)) for p in ps[t]]
        for x in grid:
            base = table[t + 1, x]
            v = (1 - sum(probs)) * base
            choices = []
            for j, p in enumerate(probs):
                remaining = tuple(x[i] - A[i][j] for i in range(len(x)))
                feasible = min(remaining) >= 0
                take = margins[j] + table[t + 1, remaining] if feasible else None
                best = max(base, take) if feasible else base
                choices.append(
                    {
                        "product": j,
                        "reject": str(base),
                        "accept": None if take is None else str(take),
                        "feasible": feasible,
                    }
                )
                v += p * best
            table[t, x] = v
            rows.append(
                {"t": t, "stock": list(x), "value": str(v), "class_values": choices}
            )
    return {
        "A": A,
        "margins": margins,
        "capacities": capacities,
        "table": table,
        "rows": rows,
        "grid": grid,
    }


def native_ledger_tree(engine, max_nodes=100000):
    """Full chronological accept/reject history tree using native ledger receipts.

    Return expected final native cash minus native service cost. Comparing the
    future increment subtracts the already-booked native net amount algebraically;
    neither cash timing nor the native accounting source is changed.
    """
    nodes = []

    def visit(e):
        if len(nodes) >= max_nodes:
            raise RuntimeError("declared native ledger tree node limit")
        node_id = len(nodes)
        t = e.world.t
        m = carrier_metrics(e.world, e.compiled, 0)
        node = {
            "id": node_id,
            "t": t,
            "stock": m["inventory"],
            "cash": e.world.carriers[0].cash,
            "bookings": jsonable(e.world.carriers[0].bookings),
            "already_booked_net_USD": m["profit"],
            "branches": [],
        }
        nodes.append(node)
        if t == e.horizon:
            node["native_final_value_USD"] = m["profit"]
            return m["profit"], node_id
        probs = e.probabilities[t]
        terms = []
        for j, p in [(None, 1 - math.fsum(probs)), *enumerate(probs)]:
            obs = e.observe(j)
            actions = [False, True] if obs.can_accept else [False]
            options = []
            for accept in actions:
                child = e.clone()
                child.step(j, accept)
                value, k = visit(child)
                options.append({"accept": accept, "child": k, "value_USD": value})
            best = max(x["value_USD"] for x in options)
            terms.append(p * best)
            node["branches"].append(
                {
                    "requested_product": j,
                    "mass": p,
                    "observation": jsonable(obs),
                    "actions": options,
                    "best_USD": best,
                }
            )
        value = math.fsum(terms)
        node["native_final_value_USD"] = value
        return value, node_id

    value, _ = visit(engine)
    return value, nodes


def run(cell, context):
    p = cell["parameters"]
    guards = [
        (p["problem"], p["problem_sha256"]),
        (p["design_registration"], p["design_registration_sha256"]),
    ]
    if any(file_hash(n) != h for n, h in guards):
        raise ValueError("frozen fixed-fare input changed")
    source = json.loads(Path(p["problem"]).read_text())
    c = compile_market(load_spec(source["physical_spec"]))
    engine = FixedFareAdmissionEngine(c, source["class_probabilities"])
    reference = classical_reference(source)
    value, nodes = native_ledger_tree(engine, p["max_nodes"])
    table = reference["table"]
    T = len(source["class_probabilities"])
    assert c.A[0].tolist() == reference["A"]
    errors = [
        abs(
            n["native_final_value_USD"]
            - n["already_booked_net_USD"]
            - float(table[n["t"], tuple(n["stock"])])
        )
        for n in nodes
    ]
    assert max(errors) <= p["tolerance"]
    bounds = []
    for t in range(T + 1):
        demand = [
            float(
                sum(
                    (
                        Fraction(str(row[j]))
                        for row in source["class_probabilities"][t:]
                    ),
                    Fraction(0),
                )
            )
            for j in range(len(c.spec.products))
        ]
        for stock in reference["grid"]:
            x, pi, lp = solve_dlp(
                reference["A"],
                list(stock),
                list(map(float, reference["margins"])),
                demand,
            )
            dp = float(table[t, stock])
            assert dp <= lp + p["tolerance"]
            bounds.append(
                {
                    "t": t,
                    "stock": list(stock),
                    "expected_class_counts": demand,
                    "DP_USD": dp,
                    "DLP_USD": lp,
                    "native_raw_duals": list(map(float, pi)),
                    "quantities": list(map(float, x)),
                }
            )
    root_demand = bounds[0]["expected_class_counts"]
    capacities = list(reference["capacities"])
    margins = list(map(float, reference["margins"]))
    exact = exact_reference(reference["A"], capacities, margins, root_demand)
    comparison = native_comparison(
        reference["A"], capacities, margins, root_demand, exact, p["tolerance"]
    )
    assert comparison["case_pass"]
    assert all(file_hash(n) == h for n, h in guards)
    atomic_json(
        context.directory / "classical_reference.json",
        {
            "rows": reference["rows"],
            "terminal_values": "zero on entire capacity grid",
            "A": reference["A"],
            "margins_USD": list(map(str, reference["margins"])),
            "root_value_USD": str(table[0, reference["capacities"]]),
        },
    )
    atomic_json(
        context.directory / "native_ledger_tree.json",
        {"nodes": nodes, "root_value_USD": value},
    )
    atomic_json(
        context.directory / "LP_bounds.json",
        {
            "grid": bounds,
            "root_exact_LP": exact,
            "root_native_LP_comparison": comparison,
        },
    )
    return {
        "status": "completed",
        "case": source["case"],
        "fixed_fare_native_ledger_DP_pass": True,
        "DLP_bound_pass": True,
        "root_native_value_USD": value,
        "root_classical_value_USD": float(table[0, reference["capacities"]]),
        "max_ledger_classical_error_USD": max(errors),
        "native_history_nodes": len(nodes),
        "classical_grid_rows": len(reference["rows"]),
        "LP_grid_checks": len(bounds),
        "root_LP_USD": comparison["native_objective_USD"],
        "resource_duals_unique": exact["resource_duals_unique"],
        "unique_vector_agreement": comparison["unique_vector_agreement"],
        "simulation_cohorts": 0,
        "full_V4_pass": None,
        "scientific_budget": None,
        "artifact_hashes": {
            n: file_hash(context.directory / n)
            for n in [
                "classical_reference.json",
                "native_ledger_tree.json",
                "LP_bounds.json",
            ]
        },
        "scope": "New opt-in revealed-class admission RM mode with native ledger/accounting; original MarketEngine MNL/pricing protocol unchanged. Not Littlewood100k, exponential-demand scaling or R1 learning.",
    }
