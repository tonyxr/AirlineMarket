"""MR4: exact fixed-policy symmetry and revenue-margin paired TOST, kept separate."""

import math
from pathlib import Path

import numpy as np
from scipy.stats import norm, skew
from scipy.stats import t as student_t

from airline_market.agents.fixed import FixedCoefficientController
from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.hashing import canonical_json, jsonable, schema_hash
from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.sim.accounting import carrier_metrics
from airline_market.sim.engine import MarketEngine
from airline_market.solvers.exact_dp import ExactBeliefDP
from airline_market.spec import compile_market, load_spec


def independent_moments(c, price):
    """Inventory/moment Bellman recursion using closed MNL, no native primitives."""
    s = c.spec
    assert (
        len(s.carriers) == len(s.legs) == len(s.products) == 2 and len(s.markets) == 1
    )
    assert not (
        s.capacity_priors
        or s.background
        or s.quotes.requests
        or s.capacity_process.events
    )
    assert (
        s.demand.mode == "single_request_bernoulli"
        and s.choice.model == "latent_class_mnl"
    )
    assert len(s.segments) == 1 and len(s.segments[0].classes) == 1
    taste = s.segments[0].classes[0]
    assert not (
        taste.product_shift
        or taste.connection_penalty
        or taste.schedule_penalty_per_hour
        or taste.time_penalty_per_hour
    )
    capacities = [leg.capacity.support[0] for leg in s.legs]
    assert capacities[0] == capacities[1] and all(
        len(leg.capacity.support) == 1 and leg.capacity.public for leg in s.legs
    )
    costs = [
        p.service_cost + sum(s.legs[c.leg_index[l]].cost_per_pax for l in p.legs)
        for p in s.products
    ]
    assert costs[0] == costs[1] and 0 < costs[0] < price
    utilities = [
        taste.market_value[s.markets[0].id]
        + taste.carrier_pref.get(p.carrier, 0)
        - taste.alpha * price
        for p in s.products
    ]
    assert utilities[0] == utilities[1]
    assert all(o.profit_scale == 1 and o.share_weight == 0 for o in s.objectives)
    K = capacities[0]
    T = s.horizon.n_periods
    table = {}
    rows = []
    for t in range(T, -1, -1):
        for a in range(K + 1):
            for b in range(K + 1):
                if t == T:
                    table[t, a, b] = [0.0, 0.0, 0.0]
                    continue
                arrival = min(
                    s.demand.arrival_probability_cap,
                    s.demand.arrival_probability.value_at(t),
                )
                weights = [
                    math.exp(utilities[0]) if a else 0.0,
                    math.exp(utilities[1]) if b else 0.0,
                ]
                denominator = 1 + sum(weights)
                probs = [arrival * w / denominator for w in weights]
                no_sale = 1 - math.fsum(probs)
                edges = [(no_sale, a, b, 0.0, 0.0)]
                if a:
                    edges.append((probs[0], a - 1, b, price - costs[0], 0.0))
                if b:
                    edges.append((probs[1], a, b - 1, 0.0, price - costs[1]))
                values = [0.0, 0.0, 0.0]
                for mass, x, y, ra, rb in edges:
                    ca, cb, second = table[t + 1, x, y]
                    difference = ra - rb
                    values[0] += mass * (ra + ca)
                    values[1] += mass * (rb + cb)
                    values[2] += mass * (
                        difference * difference + 2 * difference * (ca - cb) + second
                    )
                table[t, a, b] = values
                rows.append(
                    {
                        "t": t,
                        "inventory": [a, b],
                        "probabilities": [*probs, no_sale],
                        "mean_profit_USD": values[:2],
                        "second_difference_moment_USD2": values[2],
                    }
                )
    values = table[0, K, K]
    diff = values[0] - values[1]
    return {
        "root_mean_profit_USD": values[:2],
        "root_difference_USD": diff,
        "root_paired_variance_USD2": values[2] - diff**2,
        "exact_focal_revenue_USD": values[0] * price / (price - costs[0]),
        "rows": rows,
    }


def paired_statistics(differences, margin, B=2000):
    """Section2.3 paired CIs and t-TOST; zero empirical SE is inconclusive."""
    x = np.asarray(differences, dtype=float)
    n = len(x)
    if n < 2 or margin <= 0 or B != 2000:
        raise ValueError(
            "positive revenue margin, n>=2 and2000bootstrap replicates required"
        )
    avg = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / math.sqrt(n)
    draws = np.empty(B)
    registry = StreamRegistry(773, "boot")
    for k in range(B):
        rng = registry.generator(34, k)
        draws[k] = x[rng.integers(0, n, size=n)].mean()
    sample_skew = float(skew(x, bias=False)) if sd else None
    percentile = np.quantile(draws, [0.025, 0.975]).tolist()
    bca = None
    if sample_skew is not None and abs(sample_skew) > 1:
        z_fraction = (
            float(np.sum(draws < avg)) + 0.5 * float(np.sum(draws == avg))
        ) / B
        if not 0 < z_fraction < 1:
            raise ValueError("BCa bias correction undefined; no fallback")
        centered = (x - avg) / (n - 1)
        denominator = 6 * float(np.sum(centered**2)) ** 1.5
        if denominator == 0:
            raise ValueError("BCa acceleration undefined; no fallback")
        acceleration = float(np.sum(centered**3)) / denominator
        z0 = float(norm.ppf(z_fraction))
        z = norm.ppf([0.025, 0.975])
        levels = norm.cdf(z0 + (z0 + z) / (1 - acceleration * (z0 + z)))
        bca = np.quantile(draws, levels).tolist()
    if se:
        p_lower = float(student_t.sf((avg + margin) / se, n - 1))
        p_upper = float(student_t.cdf((avg - margin) / se, n - 1))
        ci90 = [
            avg - float(student_t.ppf(0.95, n - 1)) * se,
            avg + float(student_t.ppf(0.95, n - 1)) * se,
        ]
        equivalent = max(p_lower, p_upper) < 0.05
    else:
        p_lower = p_upper = None
        ci90 = [avg, avg]
        equivalent = None
    return {
        "n_cohorts": n,
        "mean_difference_USD": avg,
        "sample_SD_USD": sd,
        "SE_USD": se,
        "normal_95_CI_USD": [
            avg - 1.959963984540054 * se,
            avg + 1.959963984540054 * se,
        ],
        "sample_skewness": sample_skew,
        "percentile_cluster_bootstrap_95_USD": percentile,
        "BCa_95_USD": bca,
        "reported_bootstrap_method": "BCa" if bca is not None else "percentile",
        "TOST_margin_USD": margin,
        "TOST_p_lower": p_lower,
        "TOST_p_upper": p_upper,
        "TOST_p_max": None if se == 0 else max(p_lower, p_upper),
        "TOST_90_t_CI_USD": ci90,
        "TOST_equivalent": equivalent,
        "TOST_status": "inconclusive_zero_sample_variance"
        if se == 0
        else "equivalent"
        if equivalent
        else "equivalence_not_established",
        "bootstrap_namespace": "boot",
        "bootstrap_root": 773,
        "bootstrap_replicates": B,
        "bootstrap_means_USD": draws.tolist(),
        "strict_A_greater_B": int(np.sum(x > 0)),
        "ties": int(np.sum(x == 0)),
        "strict_A_less_B": int(np.sum(x < 0)),
    }


def native_expectations(c, max_nodes):
    """Unchanged general ExactBeliefDP with singleton0 grid and fixed0 rivals."""
    results = []
    for code in ["A", "B"]:
        dp = ExactBeliefDP(
            c,
            code,
            "direct",
            rival_controllers={i: FixedCoefficientController((0.0,)) for i in range(2)},
            backend="reference",
            max_nodes=max_nodes,
            belief_merge_tol=0.0,
        )
        assert dp.backend == "reference" and type(dp.kernel).__name__ == "ExactKernel"
        answer = dp.solve()
        masses = [
            math.fsum(weight for weight, _, _ in children)
            for children in dp.children.values()
        ]
        assert max((abs(x - 1) for x in masses), default=0) <= 1e-12
        results.append(
            {
                "focal": code,
                "value_USD": answer.value,
                "nodes": answer.nodes,
                "nodes_per_period": answer.nodes_per_period,
                "raw_successor_masses": masses,
                "backend": dp.backend,
                "kernel_type": type(dp.kernel).__name__,
            }
        )
    return results


def run(cell, context):
    p = cell["parameters"]
    if not 200000 <= p["seed_start"] < p["seed_start"] + p["n_cohorts"] <= 300000:
        raise ValueError("validation cohort range required")
    guards = [
        (p["source_spec"], p["source_spec_sha256"]),
        (p["design_registration"], p["design_registration_sha256"]),
    ]
    if any(file_hash(Path(path)) != h for path, h in guards):
        raise ValueError("frozen symmetry input changed")
    c = compile_market(load_spec(p["source_spec"]))
    assert c.spec_hash == p["compiled_spec_hash"]
    exact = independent_moments(c, p["price"])
    assert abs(exact["root_difference_USD"]) <= 1e-12
    native = native_expectations(c, p["max_nodes"])
    assert all(
        abs(x["value_USD"] - exact["root_mean_profit_USD"][i]) <= p["exact_tolerance"]
        for i, x in enumerate(native)
    )
    assert abs(native[0]["value_USD"] - native[1]["value_USD"]) <= 1e-12
    margin = p["margin_revenue_fraction"] * exact["exact_focal_revenue_USD"]
    assert p["margin_revenue_fraction"] == 0.001 and p["equivalence_alpha"] == 0.05
    exact.update(
        native=native,
        TOST_margin_USD=margin,
        nominal_90_normal_halfwidth_USD=1.6448536269514722
        * math.sqrt(exact["root_paired_variance_USD2"] / p["n_cohorts"]),
        simulation_rows_processed_yet=0,
    )
    atomic_json(context.directory / "exact_reference.json", exact)
    exact_sha = file_hash(context.directory / "exact_reference.json")
    engine = MarketEngine(c)
    T = c.spec.horizon.n_periods
    actions = {i: PriceAction({i: p["price"]}) for i in range(2)}
    current = {
        "stage": "native_symmetric_policy",
        "cohorts_completed": 0,
        "cohorts_total": p["n_cohorts"],
        "native_period_steps": 0,
    }
    context.counters = lambda: dict(current)
    differences = []
    histogram = {}
    profits = np.zeros(2)
    revenues = np.zeros(2)
    with (context.directory / "native_cohorts.jsonl").open("x") as out:
        for k in range(p["n_cohorts"]):
            seed = p["seed_start"] + k
            engine.reset(seed)
            tape = engine.tape
            tape_sha = schema_hash(tape)
            categories = []
            while not engine.terminal():
                count = len(engine.world.request_log)
                step = engine.step(actions)
                new = engine.world.request_log[count:]
                assert (
                    len(new) <= 1
                    and not step.resolutions
                    and all(
                        float(cs.posted_prices[0]) == p["price"]
                        for cs in engine.world.carriers
                    )
                )
                categories.append(
                    3
                    if not new
                    else 2
                    if new[0]["chosen"] is None
                    else new[0]["chosen"]
                )
                current["native_period_steps"] += 1
            metrics = [carrier_metrics(engine.world, c, i) for i in range(2)]
            counts = [len(cs.bookings) for cs in engine.world.carriers]
            for i, m in enumerate(metrics):
                assert (
                    m["refunds"] == m["penalties"] == 0
                    and m["profit"]
                    == counts[i] * (p["price"] - float(c.product_cost[i]))
                    and m["revenue"] == counts[i] * p["price"]
                )
                profits[i] += m["profit"]
                revenues[i] += m["revenue"]
            d = metrics[0]["profit"] - metrics[1]["profit"]
            differences.append(d)
            histogram[str(d)] = histogram.get(str(d), 0) + 1
            assert (
                len(categories) == T
                and engine.tape is tape
                and schema_hash(tape) == tape_sha
            )
            out.write(
                canonical_json(
                    {
                        "physical_seed": seed,
                        "categories": categories,
                        "bookings": jsonable(
                            [cs.bookings for cs in engine.world.carriers]
                        ),
                        "profits_USD": [m["profit"] for m in metrics],
                        "revenues_USD": [m["revenue"] for m in metrics],
                        "paired_difference_USD": d,
                        "tape_sha256": tape_sha,
                    }
                )
                + "\n"
            )
            current["cohorts_completed"] = k + 1
    current["stage"] = "paired_statistics"
    statistics = paired_statistics(differences, margin, p["bootstrap_replicates"])
    atomic_json(context.directory / "paired_statistics.json", statistics)
    assert file_hash(context.directory / "exact_reference.json") == exact_sha and all(
        file_hash(Path(path)) == h for path, h in guards
    )
    return {
        "status": "completed",
        "n_cohorts": p["n_cohorts"],
        "native_period_steps": current["native_period_steps"],
        "exact_mean_profit_USD": exact["root_mean_profit_USD"],
        "exact_symmetry_pass": True,
        "TOST_equivalent": statistics["TOST_equivalent"],
        "TOST_status": statistics["TOST_status"],
        "mean_profit_USD": (profits / p["n_cohorts"]).tolist(),
        "mean_revenue_USD": (revenues / p["n_cohorts"]).tolist(),
        "mean_difference_USD": statistics["mean_difference_USD"],
        "mean_difference_percent_exact_focal_revenue": 100
        * statistics["mean_difference_USD"]
        / exact["exact_focal_revenue_USD"],
        "oracle_gap_closed": None,
        "oracle_gap_note": "No registered positive oracle gap in singleton constantpolicy class; never divide by zero.",
        "difference_histogram": histogram,
        "actor_forwards": 0,
        "optimizer_updates": 0,
        "full_V3_pass": None,
        "scientific_budget": None,
        "artifact_hashes": {
            name: file_hash(context.directory / name)
            for name in [
                "exact_reference.json",
                "native_cohorts.jsonl",
                "paired_statistics.json",
            ]
        },
        "scope": "Exact symmetry and statistical TOST are separate. One full native episode yields both carrier profits; no swaps/antithetic averaging/forcedchoices/countcuts/retries. Margin .1percentexactfocalrevenue, notprofit. Fixedpolicyclass notequilibrium/generalMR4.",
    }
