"""MR6: zero all-demand accounting and the normalized outside-option limit."""

import json
import math
from pathlib import Path

from airline_market.agents.fixed import FixedCoefficientController
from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.hashing import canonical_json, jsonable, schema_hash
from airline_market.core.types import PriceAction
from airline_market.sim.accounting import carrier_metrics
from airline_market.sim.engine import MarketEngine
from airline_market.sim.kernel import ExactKernel, PostedPrices
from airline_market.solvers.exact_dp import ExactBeliefDP
from airline_market.spec import compile_market, load_spec


def fixed_actions(c):
    """Constant declared public reference fares; no hidden state or actor call."""
    return {
        i: PriceAction({j: c.spec.products[j].reference_fare for j in c.products_of[i]})
        for i in range(len(c.spec.carriers))
    }


def conversion_reference(c):
    """Independent two-alternative logit and native fixed-policy expectation.

    Outside utility is normalized to zero natively. Increasing the common inside
    utility by delta exactly represents outside utility -delta after subtracting
    that alternative-independent constant. Capacity>=T cannot bind before any of
    the T Bernoulli requests; this is an exact finite-horizon unlimited-stock case.
    """
    s = c.spec
    T = s.horizon.n_periods
    if (
        len(s.carriers) != 1
        or len(s.legs) != 1
        or len(s.products) != 1
        or len(s.markets) != 1
        or len(s.segments) != 1
        or len(s.segments[0].classes) != 1
        or s.background
        or s.capacity_process.events
        or s.quotes.requests
        or s.capacity_priors
        or s.demand.mode != "single_request_bernoulli"
        or s.choice.model != "latent_class_mnl"
        or len(s.exogenous.states) != 1
        or s.exogenous.transitions
        or s.demand.exogenous_multipliers
        or s.demand.holiday_multiplier != 1
        or s.objectives[0].share_weight != 0
        or s.objectives[0].profit_scale != 1
        or c.product_cost[0] != 0
    ):
        raise ValueError("conversion benchmark contract changed")
    leg = s.legs[0]
    p = s.products[0]
    taste = s.segments[0].classes[0]
    if (
        leg.capacity.support != (T,)
        or not leg.capacity.public
        or leg.capacity.probabilities != (1.0,)
    ):
        raise ValueError(
            "capacity must be public T, hence nonbinding before every request"
        )
    if (
        taste.product_shift
        or taste.connection_penalty
        or taste.time_penalty_per_hour
        or taste.schedule_penalty_per_hour
    ):
        raise ValueError("unexpected utility term")
    arrivals = [
        min(s.demand.arrival_probability_cap, s.demand.arrival_probability.value_at(t))
        for t in range(T)
    ]
    if len(set(arrivals)) != 1 or any(
        s.demand.market_weights.value_at(t) != (1.0,)
        or s.demand.segment_weights.value_at(t) != (1.0,)
        for t in range(T)
    ):
        raise ValueError("constant one-market one-class arrival contract required")
    price = p.reference_fare
    if s.pricing[0].base[p.id] != price or s.pricing[0].theta_grid != ((0.0,),):
        raise ValueError("singleton native fixed-fare class required")
    utility = (
        taste.market_value[s.markets[0].id]
        + taste.carrier_pref.get(p.carrier, 0)
        - taste.alpha * price
    )
    ratio = math.exp(-utility)
    outside = ratio / (1 + ratio)
    buy = 1 / (1 + ratio)
    engine = MarketEngine(c)
    engine.reset(0)  # analytical construction only
    kernel = ExactKernel(c, {0: FixedCoefficientController((0.0,))})
    state = kernel.encode(engine.world)
    raw = [0.0, 0.0, 0.0]
    for mass, outcome in kernel.period_outcomes(
        state, PostedPrices(state, 0, PriceAction({0: price}), (price,))
    ):
        category = 2 if not outcome.requests else 1 if outcome.choices[0] is None else 0
        raw[category] += mass
    expected = [arrivals[0] * buy, arrivals[0] * outside, 1 - arrivals[0]]
    assert (
        abs(math.fsum(raw) - 1) <= 1e-12
        and max(abs(a - b) for a, b in zip(raw, expected)) <= 1e-12
    )
    # Compare the outside tail relatively too, so float buy==1 cannot hide it.
    assert abs(raw[1] - expected[1]) <= 1e-12 * expected[1]
    dp = ExactBeliefDP(
        c,
        s.carriers[0].code,
        "direct",
        rival_controllers={},
        backend="reference",
        max_nodes=10000,
        belief_merge_tol=0.0,
    )
    answer = dp.solve()
    target = T * arrivals[0] * buy * price
    assert abs(answer.value - target) <= 1e-10
    masses = [math.fsum(x[0] for x in children) for children in dp.children.values()]
    assert max((abs(x - 1) for x in masses), default=0) <= 1e-12
    return {
        "inside_utility": utility,
        "outside_probability": outside,
        "outside_probability_upper_bound": ratio,
        "conditional_purchase_probability": buy,
        "native_raw_categories": raw,
        "independent_categories": expected,
        "native_DP_USD": answer.value,
        "independent_expected_USD": target,
        "native_DP_nodes": answer.nodes,
        "raw_DP_successor_masses": masses,
        "capacity": T,
        "max_requests": T,
        "nonbinding_stock_proof": "Before t<T, at most t seats sold, so T-t>=1 seats remain.",
        "limit_proof": "0<P(outside)=exp(-u)/(1+exp(-u))<=exp(-u), tending to0 as common inside shift tends to+infinity; no finite shift claimed to be literal infinity.",
    }


def run(cell, context):
    p = cell["parameters"]
    start = p["seed_start"]
    count = p["n_cohorts"]
    if not 200000 <= start < start + count <= 300000:
        raise ValueError("validation IDs required")
    guards = [(p["design_registration"], p["design_registration_sha256"])]
    guards += [(m["path"], m["sha256"]) for g in p["groups"] for m in g["models"]]
    guards += [
        (g["original_path"], g["original_sha256"])
        for g in p["groups"]
        if g.get("original_path")
    ]
    if any(file_hash(path) != h for path, h in guards):
        raise ValueError("frozen input changed")
    groups = []
    exact = []
    for group in p["groups"]:
        cs = [compile_market(load_spec(m["path"])) for m in group["models"]]
        assert all(c.spec_hash == m["spec_hash"] for c, m in zip(cs, group["models"]))
        if group["kind"] == "zero":
            original = json.loads(Path(group["original_path"]).read_text())
            for c, m in zip(cs, group["models"]):
                target = json.loads(json.dumps(original))
                target["demand"]["arrival_probability"] = {
                    "breakpoints": [0],
                    "values": [0.0],
                }
                if m["all_demand_zero"]:
                    for bg in target["background"]:
                        bg["purchase"]["max_prob"] = 0.0
                assert jsonable(load_spec(target)) == jsonable(c.spec)
        elif group["kind"] == "conversion":
            refs = [conversion_reference(c) for c in cs]
            assert all(
                a["outside_probability"] > b["outside_probability"]
                for a, b in zip(refs, refs[1:])
            )
            exact = refs
        else:
            raise ValueError("unknown group kind")
        groups.append((group, cs, [MarketEngine(c) for c in cs]))
    atomic_json(
        context.directory / "exact_conversion.json",
        {"references": exact, "simulation_rows_processed_yet": 0},
    )
    exact_sha = file_hash(context.directory / "exact_conversion.json")
    current = {
        "stage": "native_degenerate_demand",
        "physical_ids_completed": 0,
        "model_profiles_completed": 0,
        "native_steps": 0,
        "tapes_generated": 0,
    }
    context.counters = lambda: dict(current)
    totals = {
        m["id"]: {
            "profiles": 0,
            "purchases": 0,
            "requests": 0,
            "strategic_revenue_USD": 0.0,
            "background_revenue_USD": 0.0,
            "nonpurchase": 0,
        }
        for g in p["groups"]
        for m in g["models"]
    }
    with (context.directory / "cohort_rows.jsonl").open("x") as out:
        for group, cs, engines in groups:
            for k in range(count):
                seed = start + k
                engines[0].reset(seed)
                tape = engines[0].tape
                sha = schema_hash(tape)
                current["tapes_generated"] += 1
                profiles = []
                for index, (c, m, engine) in enumerate(
                    zip(cs, group["models"], engines)
                ):
                    if index:
                        engine.reset(seed, tape=tape)
                    actions = fixed_actions(c)
                    categories = []
                    while not engine.terminal():
                        before = len(engine.world.request_log)
                        step = engine.step(actions)
                        new = engine.world.request_log[before:]
                        assert len(new) <= 1
                        categories.append(
                            2 if not new else 1 if new[0]["chosen"] is None else 0
                        )
                        current["native_steps"] += 1
                    metrics = [
                        carrier_metrics(engine.world, c, i)
                        for i in range(len(c.spec.carriers))
                    ]
                    strategic = sum(x["revenue"] for x in metrics)
                    bg = sum(x.revenue for x in engine.world.background)
                    bookings = jsonable([x.bookings for x in engine.world.carriers])
                    requests = len(engine.world.request_log)
                    purchases = sum(len(x.bookings) for x in engine.world.carriers)
                    if group["kind"] == "zero":
                        assert requests == purchases == strategic == 0 and all(
                            state.cash == state.refunds == state.penalties == 0
                            for state in engine.world.carriers
                        )
                        if m["all_demand_zero"]:
                            assert bg == 0
                    else:
                        assert all(x >= 0 for x in metrics[0]["inventory"])
                        assert strategic == purchases * c.spec.products[
                            0
                        ].reference_fare and all(
                            x["refunds"] == x["penalties"] == 0 for x in metrics
                        )
                    assert engine.tape is tape and schema_hash(tape) == sha
                    row = {
                        "model_id": m["id"],
                        "categories": categories,
                        "requests": requests,
                        "bookings": bookings,
                        "metrics": metrics,
                        "strategic_revenue_USD": strategic,
                        "background_revenue_USD": bg,
                    }
                    profiles.append(row)
                    v = totals[m["id"]]
                    v["profiles"] += 1
                    v["purchases"] += purchases
                    v["requests"] += requests
                    v["strategic_revenue_USD"] += strategic
                    v["background_revenue_USD"] += bg
                    v["nonpurchase"] += categories.count(1)
                    current["model_profiles_completed"] += 1
                if group["kind"] == "conversion":
                    assert all(
                        a["requests"] == b["requests"]
                        for a, b in zip(profiles, profiles[1:])
                    )
                    for a, b in zip(profiles, profiles[1:]):
                        assert all(
                            x == y == 2 or x != 2 and y != 2 and y <= x
                            for x, y in zip(a["categories"], b["categories"])
                        )
                out.write(
                    canonical_json(
                        {
                            "group": group["id"],
                            "physical_seed": seed,
                            "tape_sha256": sha,
                            "profiles": profiles,
                        }
                    )
                    + "\n"
                )
                current["physical_ids_completed"] += 1
    assert file_hash(context.directory / "exact_conversion.json") == exact_sha and all(
        file_hash(path) == h for path, h in guards
    )
    return {
        "status": "completed",
        "n_common_physical_ids": count,
        "group_seed_rows": len(groups) * count,
        "model_profiles": current["model_profiles_completed"],
        "native_steps": current["native_steps"],
        "actual_tapes_generated": current["tapes_generated"],
        "totals": totals,
        "zero_all_demand_revenue_pass": True,
        "zero_main_demand_strategic_revenue_pass": True,
        "main_probability_zero_alone_implies_background_revenue_zero": all(
            totals[m["id"]]["background_revenue_USD"] == 0
            for g in p["groups"]
            if g["kind"] == "zero"
            for m in g["models"]
            if not m["all_demand_zero"]
        ),
        "conversion_declared_class_pass": True,
        "full_MR6_pass": None,
        "full_V3_pass": None,
        "scientific_budget": None,
        "artifacts": {
            "exact_conversion": {"path": "exact_conversion.json", "sha256": exact_sha},
            "rows": {
                "path": "cohort_rows.jsonl",
                "sha256": file_hash(context.directory / "cohort_rows.jsonl"),
            },
        },
        "scope": "Zero main demand is distinguished from independent background demand; finite normalized outside shifts plus an analytic tail bound, not literal finite infinity/Poisson/general validation.",
    }
