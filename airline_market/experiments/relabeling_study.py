"""MR3 declared label/list permutations against saved, never rerun, references."""

import copy
import json
from pathlib import Path

from airline_market.agents.price_rule import AdditivePriceRule
from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.hashing import schema_hash
from airline_market.experiments.logit_bertrand_study import NativePayoff
from airline_market.solvers.exact_dp import ExactBeliefDP
from airline_market.spec import compile_market, load_spec


def permute_spec(original):
    """Rename IDs and reverse independent lists/grid rows, preserving physical data."""
    s = copy.deepcopy(original)
    if (
        s["capacity_priors"]
        or s["background"]
        or s["quotes"]["requests"]
        or s["capacity_process"]["events"]
    ):
        raise ValueError("declared noevent/noquote/nojointprior permutation contract")
    if any(c["controller"] != "external" for c in s["controllers"]):
        raise ValueError("declared external controller assignments required")
    maps = {
        key: {x[field]: f"{prefix}{i}" for i, x in enumerate(s[key])}
        for key, field, prefix in [
            ("airports", "code", "P"),
            ("carriers", "code", "C"),
            ("legs", "id", "L"),
            ("products", "id", "J"),
            ("markets", "id", "M"),
        ]
    }
    old_products = [x["id"] for x in s["products"]]
    for a in s["airports"]:
        a["code"] = maps["airports"][a["code"]]
    for c in s["carriers"]:
        c["code"] = maps["carriers"][c["code"]]
    for l in s["legs"]:
        l["id"] = maps["legs"][l["id"]]
        l["carrier"] = maps["carriers"][l["carrier"]]
        for key in ["origin", "dest"]:
            l[key] = maps["airports"][l[key]]
    for p in s["products"]:
        p["id"] = maps["products"][p["id"]]
        p["carrier"] = maps["carriers"][p["carrier"]]
        p["market"] = maps["markets"][p["market"]]
        p["legs"] = [maps["legs"][l] for l in p["legs"]]
    for m in s["markets"]:
        m["id"] = maps["markets"][m["id"]]
        for key in ["origin", "dest"]:
            m[key] = maps["airports"][m[key]]
    for segment in s["segments"]:
        if segment["mixed"]:
            raise ValueError("finite class permutation only")
        for taste in segment["classes"]:
            taste["carrier_pref"] = {
                maps["carriers"][k]: v for k, v in taste["carrier_pref"].items()
            }
            taste["market_value"] = {
                maps["markets"].get(k, k): v for k, v in taste["market_value"].items()
            }
            taste["product_shift"] = {
                maps["products"][k]: v for k, v in taste["product_shift"].items()
            }
    weights = s["demand"]["market_weights"]
    if set(weights) == {"constant"}:
        weights["constant"] = list(reversed(weights["constant"]))
    elif set(weights) == {"breakpoints", "values"}:
        weights["values"] = [list(reversed(row)) for row in weights["values"]]
    else:
        raise ValueError("declared vector schedule representation required")
    for p, old in zip(s["pricing"], original["pricing"]):
        if (
            p["fixed_features"]
            or p["time_ramp"]
            or [x["name"] for x in p["controllable_features"]] != ["intercept"]
        ):
            raise ValueError("declared static intercept price basis required")
        p["carrier"] = maps["carriers"][p["carrier"]]
        p["base"] = {maps["products"][k]: v for k, v in p["base"].items()}
        if p["sharing"] == "product":
            p["theta_bounds"] = list(reversed(p["theta_bounds"]))
            if p["theta_grid"] is not None:
                p["theta_grid"] = [
                    list(reversed(row)) for row in reversed(p["theta_grid"])
                ]
        elif p["sharing"] == "carrier":
            if p["theta_grid"] is not None:
                p["theta_grid"] = list(reversed(p["theta_grid"]))
        else:
            raise ValueError("declared product/carrier intercept sharing only")
    for x in [*s["controllers"], *s["objectives"]]:
        x["carrier"] = maps["carriers"][x["carrier"]]
        if x.get("share_markets") is not None:
            x["share_markets"] = [maps["markets"][m] for m in x["share_markets"]]
    for arm in s["information"]:
        arm["observer"] = maps["carriers"][arm["observer"]]
    assert not s["timing"]["leaders"] and not s["timing"]["initial_previous_prices"]
    for key in [
        "airports",
        "carriers",
        "legs",
        "products",
        "markets",
        "pricing",
        "controllers",
        "objectives",
        "information",
    ]:
        s[key] = list(reversed(s[key]))
    return s, {
        "old_to_new": maps,
        "new_to_old_products": list(reversed(range(len(old_products)))),
        "new_to_old_legs": list(reversed(range(len(original["legs"])))),
        "new_to_old_carriers": list(reversed(range(len(original["carriers"])))),
    }


def guards(p):
    items = [
        (p["design_registration"], p["design_registration_sha256"]),
        (p["source_spec"], p["source_spec_sha256"]),
        (p["original_source"], p["original_source_sha256"]),
    ]
    items += [(x["path"], x["sha256"]) for x in p["saved_references"]]
    if any(file_hash(Path(path)) != h for path, h in items):
        raise ValueError("frozen relabel input/reference changed")
    original = json.loads(Path(p["original_source"]).read_text())
    expected, mapping = permute_spec(original)
    assert (
        json.loads(Path(p["source_spec"]).read_text()) == expected
        and p["mapping"] == mapping
    )
    return items


def network_case(c, p):
    """Solve ONLY permuted source; map every cached Q/price/inventory to saved table."""
    old = json.loads(Path(p["reference_results"]).read_text())
    index = p["reference_variant"]
    classical = json.loads(Path(p["reference_classical"]).read_text())
    table = {(r["t"], tuple(r["inventory"])): r for r in classical["rows"]}
    dp = ExactBeliefDP(
        c,
        c.spec.carriers[0].code,
        "direct",
        backend="reference",
        belief_merge_tol=0.0,
        max_nodes=p["max_nodes"],
    )
    answer = dp.solve()
    assert dp.backend == "reference" and type(dp.kernel).__name__ == "ExactKernel"
    rule = AdditivePriceRule(c, c.spec.pricing[0])
    mapping = p["mapping"]
    witnesses = []
    max_q = max_price = 0.0
    mass_error = 0.0
    for key, node in dp.nodes.items():
        obs = node.observation
        before = schema_hash(obs)
        assert obs.privileged is None
        stock = [0] * len(obs.own_inventory)
        for new, previous in enumerate(mapping["new_to_old_legs"]):
            stock[previous] = obs.own_inventory[new]
        saved = table[obs.t, tuple(stock)]
        pending = sum(float(c.product_cost[b.product]) for b in obs.own_bookings)
        expected = list(reversed(saved["Q_net_USD"]))
        net = (dp.cache[key] + pending).tolist()
        max_q = max(max_q, *[abs(a - b) for a, b in zip(net, expected)])
        price_rows = []
        for action, theta in enumerate(dp.grid):
            prices = rule.prices(theta, obs).tolist()
            oldprices = saved["actions"][len(dp.grid) - 1 - action]["prices_USD"]
            wanted = [oldprices[j] for j in mapping["new_to_old_products"]]
            max_price = max(max_price, *[abs(a - b) for a, b in zip(prices, wanted)])
            price_rows.append(
                {
                    "new_action": action,
                    "saved_old_action": len(dp.grid) - 1 - action,
                    "new_prices_USD": prices,
                    "mapped_saved_prices_USD": wanted,
                }
            )
            mass = sum(weight for weight, _, _ in dp.children[key, action])
            mass_error = max(mass_error, abs(mass - 1))
        assert schema_hash(obs) == before
        witnesses.append(
            {
                "t": obs.t,
                "new_inventory": obs.own_inventory,
                "mapped_old_inventory": stock,
                "native_Q_deferred_USD": dp.cache[key].tolist(),
                "pending_service_cost_USD": pending,
                "new_Q_net_USD": net,
                "mapped_saved_Q_net_USD": expected,
                "prices": price_rows,
                "privileged": obs.privileged,
            }
        )
    baseline = old["variants"][index]["native_monopoly_value_USD"]
    error = abs(answer.value - baseline)
    passed = max(error, max_q, max_price, mass_error) <= p["tolerance"]
    return {
        "kind": "network",
        "value_USD": answer.value,
        "saved_value_USD": baseline,
        "root_error_USD": error,
        "max_Q_error_USD": max_q,
        "max_price_error_USD": max_price,
        "max_raw_mass_error": mass_error,
        "nodes": answer.nodes,
        "nodes_per_period": answer.nodes_per_period,
        "backend": dp.backend,
        "rows": witnesses,
        "declared_MR3_pass": passed,
    }


def logit_case(c, p):
    """New permuted kernel queries ONLY; old five point vectors reused by hash."""
    saved = json.loads(Path(p["reference_analytical"]).read_text())
    native = NativePayoff(c)
    rows = []
    maximum = 0.0
    order = p["mapping"]["new_to_old_products"]
    original = json.loads(Path(p["original_source"]).read_text())
    oldcosts = [
        prod["service_cost"]
        + sum(
            next(l for l in original["legs"] if l["id"] == leg)["cost_per_pax"]
            for leg in prod["legs"]
        )
        for prod in original["products"]
    ]
    for point in saved["points"]:
        prices = [point["prices"][j] for j in order]
        raw = native.probabilities(prices).tolist()
        wanted = [point["native_raw_categories"][j] for j in order] + point[
            "native_raw_categories"
        ][2:]
        profits = [raw[i] * (prices[i] - float(c.product_cost[i])) for i in range(2)]
        expected = [
            point["native_raw_categories"][j] * (point["prices"][j] - oldcosts[j])
            for j in order
        ]
        maximum = max(
            maximum,
            *[abs(a - b) for a, b in zip(raw, wanted)],
            *[abs(a - b) for a, b in zip(profits, expected)],
        )
        rows.append(
            {
                "new_prices_USD": prices,
                "mapped_saved_raw_categories": wanted,
                "new_raw_categories": raw,
                "new_expected_profit_USD": profits,
                "mapped_saved_expected_profit_USD": expected,
            }
        )
    return {
        "kind": "logit",
        "rows": rows,
        "kernel_queries": native.evaluations,
        "max_probability_or_profit_error": maximum,
        "declared_MR3_pass": maximum <= p["tolerance"],
    }


def run(cell, context):
    p = cell["parameters"]
    items = guards(p)
    c = compile_market(load_spec(p["source_spec"]))
    assert c.spec_hash == p["compiled_spec_hash"]
    context.counters = lambda: {"stage": "permuted_reference", "case": p["case"]}
    result = network_case(c, p) if p["kind"] == "network" else logit_case(c, p)
    atomic_json(context.directory / "permutation_witnesses.json", result)
    assert all(file_hash(Path(path)) == h for path, h in items)
    return {
        "status": "completed",
        "case": p["case"],
        "kind": p["kind"],
        "declared_MR3_pass": result["declared_MR3_pass"],
        "physical_cohorts": 0,
        "original_source_solver_or_sampler_reruns": 0,
        "actor_forwards": 0,
        "optimizer_updates": 0,
        "full_V3_pass": None,
        "scientific_budget": None,
        "artifact_hashes": {
            "permutation_witnesses.json": file_hash(
                context.directory / "permutation_witnesses.json"
            )
        },
        "scope": "Only explicit permuted sources queried/solved, saved originals reused. Labels/list/grid action columns transformed; no same-seed unalignedGumbel bitidentity or generalquote/history/controllerrelabeling claim.",
    }
