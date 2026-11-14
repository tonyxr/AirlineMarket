"""MR2/MR8: native general belief DP and independent network inventory recursion."""

import math
from itertools import product
from pathlib import Path

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.hashing import schema_hash
from airline_market.solvers.exact_dp import ExactBeliefDP
from airline_market.spec import compile_market, load_spec


def contract(c):
    s = c.spec
    if (
        len(s.carriers) != 1
        or s.carriers[0].code != "A"
        or s.capacity_priors
        or s.background
        or s.quotes.requests
        or s.capacity_process.events
        or len(s.exogenous.states) != 1
        or s.exogenous.transitions
        or s.demand.mode != "single_request_bernoulli"
        or s.choice.model != "latent_class_mnl"
        or len(s.segments) != 1
        or len(s.segments[0].classes) != 1
        or s.objectives[0].share_weight != 0
        or s.objectives[0].profit_scale != 1
    ):
        raise ValueError(
            "declared fully observed no-event one-class monetary network required"
        )
    taste, pricing = s.segments[0].classes[0], s.pricing[0]
    if (
        taste.time_penalty_per_hour
        or taste.schedule_penalty_per_hour
        or taste.connection_penalty
        or taste.product_shift
        or pricing.fixed_features
        or pricing.time_ramp
        or pricing.update_every != 1
        or pricing.sharing != "product"
        or [f.name for f in pricing.controllable_features] != ["intercept"]
        or not pricing.theta_grid
    ):
        raise ValueError("declared independent per-product finite fare grid required")
    if any(len(leg.capacity.support) != 1 or not leg.capacity.public for leg in s.legs):
        raise ValueError("deterministic public resource capacities required")
    return taste, pricing


def independent_network(c):
    """Classical remaining-inventory DP; no native kernel/solver/choice/pricer call."""
    taste, pricing = contract(c)
    s = c.spec
    capacity = tuple(leg.capacity.support[0] for leg in s.legs)
    incidence = tuple(
        tuple(int(leg.id in p.legs) for leg in s.legs) for p in s.products
    )
    prices = [
        tuple(
            round(
                min(
                    p.fare_cap,
                    max(
                        p.fare_floor,
                        pricing.base.get(p.id, p.reference_fare) + theta[j],
                    ),
                ),
                2,
            )
            for j, p in enumerate(s.products)
        )
        for theta in pricing.theta_grid
    ]
    costs = tuple(
        p.service_cost + sum(s.legs[c.leg_index[l]].cost_per_pax for l in p.legs)
        for p in s.products
    )
    states = list(product(*(range(k + 1) for k in capacity)))
    values, action_values, rows = {}, {}, []
    for state in states:
        values[(s.horizon.n_periods, state)] = 0.0
    for t in reversed(range(s.horizon.n_periods)):
        arrival = min(
            s.demand.arrival_probability_cap, s.demand.arrival_probability.value_at(t)
        )
        market_weights = s.demand.market_weights.value_at(t)
        for state in states:
            q_values = []
            evidence = []
            for fares in prices:
                probabilities = [0.0] * len(s.products)
                for m, market in enumerate(s.markets):
                    offered = [
                        j
                        for j, p in enumerate(s.products)
                        if p.market == market.id
                        and all(x >= use for x, use in zip(state, incidence[j]))
                    ]
                    utility = {
                        j: taste.market_value.get(
                            market.id, taste.market_value.get("*", 0.0)
                        )
                        + taste.carrier_pref.get("A", 0.0)
                        - taste.alpha * fares[j]
                        for j in offered
                    }
                    largest = max([0.0, *utility.values()])
                    denominator = math.exp(-largest) + math.fsum(
                        math.exp(u - largest) for u in utility.values()
                    )
                    for j in offered:
                        probabilities[j] = (
                            arrival
                            * market_weights[m]
                            * math.exp(utility[j] - largest)
                            / denominator
                        )
                no_sale = 1 - math.fsum(probabilities)
                if no_sale < -1e-12:
                    raise ValueError("independent probability mass invalid; no repair")
                future = values[(t + 1, state)]
                q = future
                for j, probability in enumerate(probabilities):
                    if probability:
                        after = tuple(x - use for x, use in zip(state, incidence[j]))
                        q += probability * (
                            fares[j] - costs[j] + values[(t + 1, after)] - future
                        )
                q_values.append(q)
                evidence.append(
                    {
                        "prices_USD": fares,
                        "product_probabilities": probabilities,
                        "no_sale_probability": no_sale,
                    }
                )
            values[(t, state)] = max(q_values)
            action_values[(t, state)] = q_values
            rows.append(
                {
                    "t": t,
                    "inventory": state,
                    "Q_net_USD": q_values,
                    "V_net_USD": max(q_values),
                    "actions": evidence,
                }
            )
    return {
        "root_value_USD": values[(0, capacity)],
        "rows": rows,
        "Q": action_values,
        "V": values,
    }


def native_reference(c, reference, max_nodes, current):
    """Unchanged ExactBeliefDP/reference; retain all cached Q and successor witnesses."""
    dp = ExactBeliefDP(
        c, "A", "direct", backend="reference", max_nodes=max_nodes, belief_merge_tol=0.0
    )
    if dp.backend != "reference" or type(dp.kernel).__name__ != "ExactKernel":
        raise AssertionError("general native reference dispatch changed")
    current["native_nodes"] = lambda: len(dp.nodes)
    result = dp.solve()
    rows = []
    largest, mass_error = 0.0, 0.0
    for key, node in dp.nodes.items():
        obs = node.observation
        pending = sum(float(c.product_cost[b.product]) for b in obs.own_bookings)
        expected = reference["Q"][(obs.t, tuple(obs.own_inventory))]
        native_q = dp.cache[key]
        errors = [abs(float(a) + pending - b) for a, b in zip(native_q, expected)]
        largest = max(largest, *errors)
        successors = []
        for action in range(len(dp.grid)):
            children = []
            for weight, cash, child in dp.children[(key, action)]:
                if child.t == c.spec.horizon.n_periods:
                    child_value = -sum(
                        float(c.product_cost[b.product])
                        for b in child.observation.own_bookings
                    )
                else:
                    child_value = float(dp.cache[dp.key(child)].max())
                children.append(
                    {
                        "mass": weight,
                        "cash_USD": cash,
                        "child_native_future_USD": child_value,
                        "child_t": child.t,
                        "child_inventory": child.observation.own_inventory,
                    }
                )
            mass = math.fsum(x["mass"] for x in children)
            mass_error = max(mass_error, abs(mass - 1))
            successors.append(children)
        rows.append(
            {
                "t": obs.t,
                "inventory": obs.own_inventory,
                "native_Q_deferred_cost_USD": native_q.tolist(),
                "existing_pending_service_cost_USD": pending,
                "native_Q_net_future_USD": (native_q + pending).tolist(),
                "independent_Q_net_USD": expected,
                "max_abs_error": max(errors),
                "raw_successors": successors,
                "public_observation_sha256": schema_hash(obs),
                "privileged": obs.privileged,
            }
        )
    return {
        "value_USD": result.value,
        "nodes": result.nodes,
        "nodes_per_period": result.nodes_per_period,
        "all_native_cached_Q_max_abs_error": largest,
        "max_raw_successor_mass_error": mass_error,
        "backend": dp.backend,
        "kernel_type": type(dp.kernel).__name__,
        "rows": rows,
    }


def run(cell, context):
    p = cell["parameters"]
    registration = Path(p["design_registration"])
    if file_hash(registration) != p["design_registration_sha256"]:
        raise ValueError("frozen network design changed")
    current = {
        "stage": "network_exact_references",
        "variants_completed": 0,
        "variants_total": len(p["variants"]),
    }

    def counters():
        return {
            key: value() if callable(value) else value for key, value in current.items()
        }

    context.counters = counters
    result_rows = []
    for index, variant in enumerate(p["variants"]):
        compiled = []
        for kind in ["monopoly", "unavailable_rival"]:
            source = variant[kind]
            if file_hash(Path(source["path"])) != source["sha256"]:
                raise ValueError("frozen network source changed")
            c = compile_market(load_spec(source["path"]))
            if c.spec_hash != source["compiled_spec_hash"]:
                raise ValueError("compiled network source changed")
            compiled.append(c)
        monopoly, parent = compiled
        current.update(
            variant=index,
            capacity=variant["capacity"],
            stage="independent_inventory_DP",
        )
        reference = independent_network(monopoly)
        atomic_json(
            context.directory / f"classical_{index}.json",
            {"root_value_USD": reference["root_value_USD"], "rows": reference["rows"]},
        )
        current["stage"] = "native_monopoly_reference"
        native = native_reference(monopoly, reference, p["max_nodes"], current)
        atomic_json(context.directory / f"native_monopoly_{index}.json", native)
        current["stage"] = "native_unavailable_rival_reference"
        removed = native_reference(parent, reference, p["max_nodes"], current)
        atomic_json(context.directory / f"native_parent_{index}.json", removed)
        maximum = max(
            abs(native["value_USD"] - reference["root_value_USD"]),
            abs(removed["value_USD"] - reference["root_value_USD"]),
            native["all_native_cached_Q_max_abs_error"],
            removed["all_native_cached_Q_max_abs_error"],
        )
        result_rows.append(
            {
                "variant": index,
                "changed_leg": variant["changed_leg"],
                "capacity": variant["capacity"],
                "classical_value_USD": reference["root_value_USD"],
                "native_monopoly_value_USD": native["value_USD"],
                "native_unavailable_rival_value_USD": removed["value_USD"],
                "all_reference_max_abs_error": maximum,
                "reference_agreement_pass": maximum <= p["value_tolerance"],
                "native_monopoly_nodes": native["nodes"],
                "native_parent_nodes": removed["nodes"],
                "native_raw_mass_max_error": max(
                    native["max_raw_successor_mass_error"],
                    removed["max_raw_successor_mass_error"],
                ),
            }
        )
        current["variants_completed"] = index + 1
        atomic_json(context.directory / "partial_variant_summary.json", result_rows)
    base = result_rows[0]["native_monopoly_value_USD"]
    for row in result_rows:
        row["capacity_increment_value_USD"] = row["native_monopoly_value_USD"] - base
        row["capacity_monotonicity_pass"] = (
            row["capacity_increment_value_USD"] >= -p["monotonicity_tolerance"]
        )
    files = [
        f"{prefix}_{i}.json"
        for i in range(len(result_rows))
        for prefix in ["classical", "native_monopoly", "native_parent"]
    ]
    atomic_json(context.directory / "network_summary.json", result_rows)
    files.append("network_summary.json")
    return {
        "status": "completed",
        "network": p["network"],
        "variants": result_rows,
        "MR8_declared_network_reference_pass": all(
            x["reference_agreement_pass"] and x["native_raw_mass_max_error"] <= 1e-12
            for x in result_rows
        ),
        "MR2_declared_capacity_edges_pass": all(
            x["capacity_monotonicity_pass"] for x in result_rows
        ),
        "physical_cohorts": 0,
        "actor_forwards": 0,
        "optimizer_updates": 0,
        "full_V3_pass": None,
        "full_V4_pass": None,
        "scientific_budget": None,
        "artifact_hashes": {
            name: file_hash(context.directory / name) for name in files
        },
        "scope": "Three declared finite coefficient-price network classes, general native belief DP versus independent inventory recursion; exact removal of public zero-capacity rival. No finite-cap rejection, fixed-fare admission DP, unrounded global optimum, learning or empirical claim.",
    }
