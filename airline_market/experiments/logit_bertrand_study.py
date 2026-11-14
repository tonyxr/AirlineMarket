"""One-active-period logit game: native cents versus an independent continuous FOC.

Native specs require T>=2. The opening period has exactly zero demand and no
events; the final period is the economic game. No existing runtime is changed.
"""

import itertools
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar, root
from scipy.stats import chisquare

from airline_market.agents.fixed import FixedCoefficientController
from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.hashing import canonical_json, schema_hash
from airline_market.core.types import PriceAction
from airline_market.sim.accounting import carrier_metrics
from airline_market.sim.engine import MarketEngine
from airline_market.sim.kernel import ExactKernel, PostedPrices
from airline_market.spec import compile_market, dump_spec, load_spec


def contract(c):
    s = c.spec
    if (
        s.horizon.n_periods != 2
        or len(s.carriers) != 2
        or len(s.legs) != 2
        or len(s.products) != 2
        or len(s.markets) != 1
        or len(s.segments) != 1
        or len(s.segments[0].classes) != 1
        or len(s.exogenous.states) != 1
        or s.exogenous.transitions
        or s.capacity_process.events
        or s.background
        or s.capacity_priors
        or s.quotes.requests
        or s.timing.mode != "simultaneous"
        or s.demand.mode != "single_request_bernoulli"
        or s.choice.model != "latent_class_mnl"
        or s.demand.holiday_multiplier != 1
        or s.demand.exogenous_multipliers
        or s.demand.arrival_probability.value_at(0) != 0
    ):
        raise ValueError("one-active-period slack logit contract changed")
    taste = s.segments[0].classes[0]
    if (
        taste.product_shift
        or taste.time_penalty_per_hour
        or taste.schedule_penalty_per_hour
        or taste.connection_penalty
        or taste.alpha <= 0
        or any(o.share_weight != 0 for o in s.objectives)
        or any(len(c.products_of[i]) != 1 or len(c.legs_of[i]) != 1 for i in range(2))
        or any(
            p.legs != (s.legs[i].id,) or p.carrier != s.carriers[i].code
            for i, p in enumerate(s.products)
        )
        or any(
            len(l.capacity.support) != 1
            or l.capacity.support[0] < 2
            or l.capacity.probabilities != (1.0,)
            or not l.capacity.public
            for l in s.legs
        )
        or any(
            s.demand.market_weights.value_at(t) != (1.0,)
            or s.demand.segment_weights.value_at(t) != (1.0,)
            for t in range(2)
        )
    ):
        raise ValueError(
            "single-product public slack-capacity utility contract changed"
        )
    bounds = np.array([[p.fare_floor, p.fare_cap] for p in s.products])
    costs = np.asarray(c.product_cost)
    if np.any(bounds[:, 0] > costs) or np.any(costs >= bounds[:, 1]):
        raise ValueError("benchmark requires costs inside the price bounds")
    return {
        "alpha": taste.alpha,
        "values": np.array(
            [
                taste.market_value[s.markets[0].id]
                + taste.carrier_pref.get(p.carrier, 0)
                for p in s.products
            ]
        ),
        "costs": costs,
        "bounds": bounds,
        "arrival": min(
            s.demand.arrival_probability_cap, s.demand.arrival_probability.value_at(1)
        ),
    }


def closed_shares(q, prices):
    """Independent three-alternative linear-utility softmax, conditional on arrival."""
    utilities = [q["values"][i] - q["alpha"] * float(prices[i]) for i in range(2)] + [
        0.0
    ]
    maximum = max(utilities)
    weights = [math.exp(x - maximum) for x in utilities]
    return np.array([x / sum(weights) for x in weights])


def continuous_equilibrium(q):
    """Two simultaneous closed-form FOCs; independent of native profit maximization."""

    def equations(prices):
        shares = closed_shares(q, prices)[:2]
        return 1 - q["alpha"] * (prices - q["costs"]) * (1 - shares)

    answer = root(equations, q["costs"] + 1 / q["alpha"], method="hybr", tol=1e-11)
    residual = float(np.max(np.abs(equations(answer.x))))
    if not answer.success or residual > 1e-10:
        raise RuntimeError("independent continuous equilibrium root failed")
    if np.any(answer.x <= q["bounds"][:, 0]) or np.any(answer.x >= q["bounds"][:, 1]):
        raise ValueError(
            "continuous target is not interior; no boundary target substitution"
        )
    return {"prices": answer.x.tolist(), "FOC_max_abs": residual}


class NativePayoff:
    """General kernel outcome probabilities; continuous prices are auxiliary only."""

    def __init__(self, c):
        self.c, self.q = c, contract(c)
        controllers = {i: FixedCoefficientController((0.0,)) for i in range(2)}
        self.kernel = ExactKernel(c, controllers)
        engine = MarketEngine(c)
        engine.reset(0)  # model-generated construction, not a validation cohort
        step = engine.step({i: PriceAction({i: 100.0}) for i in range(2)})
        if (
            step.purchases
            or engine.world.request_log
            or any(cs.cash for cs in engine.world.carriers)
        ):
            raise AssertionError("empty opening period changed economic state")
        self.state = self.kernel.encode(engine.world)
        self.evaluations = 0
        self.objective_log = []

    def probabilities(self, prices):
        prices = tuple(map(float, prices))
        posted = PostedPrices(self.state, 0, PriceAction({0: prices[0]}), prices)
        rows = self.kernel.period_outcomes(self.state, posted)
        values = np.zeros(4)
        for mass, outcome in rows:
            if not outcome.requests:
                category = 3
            else:
                if len(outcome.requests) != 1 or len(outcome.choices) != 1:
                    raise AssertionError("benchmark has more than one request")
                category = 2 if outcome.choices[0] is None else outcome.choices[0]
            values[category] += mass
        if np.any(values < 0) or abs(values.sum() - 1) > 1e-12:
            raise ValueError(
                "native raw outcome masses do not sum to one; no renormalization"
            )
        self.evaluations += 1
        return values

    def profit(self, prices, player):
        return float(
            self.probabilities(prices)[player]
            * (prices[player] - self.q["costs"][player])
        )

    def best_response(self, prices, player):
        """Native unrounded objective optimizer, then certify its cent-price maximum.

        For p>=cost, alpha*(p-cost)*(1-s_i) is strictly increasing, so the
        continuous optimum is unique. Below cost profit is negative. Thus a
        global cent maximum is at the floor/ceiling of that root or a bound.
        The root certifies the optimizer; it never changes a failed optimizer.
        """
        q, fixed = self.q, list(map(float, prices))
        lo, hi = q["bounds"][player]

        def objective(price):
            point = fixed.copy()
            point[player] = float(price)
            payoff = self.profit(point, player)
            self.objective_log.append(
                {
                    "player": player,
                    "prices": point,
                    "USD_profit": payoff,
                    "scope": "unrounded auxiliary kernel objective",
                }
            )
            return -payoff

        answer = minimize_scalar(
            objective,
            bounds=(q["costs"][player], hi),
            method="bounded",
            options={"xatol": 1e-9},
        )
        if not answer.success:
            raise RuntimeError("native-profit scalar optimizer failed")

        def foc(price):
            point = fixed.copy()
            point[player] = price
            return 1 - q["alpha"] * (price - q["costs"][player]) * (
                1 - closed_shares(q, point)[player]
            )

        optimum = (
            hi if foc(hi) >= 0 else brentq(foc, q["costs"][player], hi, xtol=1e-11)
        )
        if abs(answer.x - optimum) > 2e-4:
            raise AssertionError("native optimizer disagrees with unique global root")

        def candidates(value):
            return sorted(
                {
                    float(lo),
                    float(hi),
                    max(lo, min(hi, math.floor(value * 100) / 100)),
                    max(lo, min(hi, math.ceil(value * 100) / 100)),
                }
            )

        def evaluate(price):
            point = fixed.copy()
            point[player] = price
            return self.profit(point, player)

        options = [(x, evaluate(x)) for x in candidates(answer.x)]
        selected = max(options, key=lambda row: row[1])
        certificate = [(x, evaluate(x)) for x in candidates(optimum)]
        gap = max(x[1] for x in certificate) - selected[1]
        if gap > 1e-10:
            raise AssertionError(
                "native cent candidate misses the certified global maximum"
            )
        return {
            "price": selected[0],
            "USD_profit": selected[1],
            "continuous_auxiliary_optimizer": float(answer.x),
            "independent_unique_root": float(optimum),
            "certificate_gap_USD": gap,
        }


def analytical(native, parameters):
    q = native.q
    points = []
    for price in parameters["analytical_points"]:
        raw = native.probabilities(price)
        expected = np.r_[q["arrival"] * closed_shares(q, price), 1 - q["arrival"]]
        slopes = []
        for i in range(2):
            plus, minus = list(price), list(price)
            plus[i] += 0.01
            minus[i] -= 0.01
            slope = (
                native.probabilities(plus)[i] - native.probabilities(minus)[i]
            ) / 0.02
            observed = slope * price[i] / raw[i]
            theory = -q["alpha"] * price[i] * (1 - closed_shares(q, price)[i])
            slopes.append(
                {
                    "player": i,
                    "native_cent_difference_elasticity": observed,
                    "continuous_closed_elasticity": theory,
                    "abs_error": abs(observed - theory),
                }
            )
        points.append(
            {
                "prices": price,
                "native_raw_categories": raw.tolist(),
                "closed_raw_categories": expected.tolist(),
                "share_max_abs_error": float(np.max(np.abs(raw - expected))),
                "elasticities": slopes,
            }
        )
    target = continuous_equilibrium(q)
    paths = []
    for start in itertools.product(*q["bounds"]):
        for order in [(0, 1), (1, 0)]:
            price, seen, rows, converged = list(start), set(), [], False
            for iteration in range(parameters["max_BR_iterations"]):
                before = tuple(price)
                if before in seen:
                    break
                seen.add(before)
                responses = []
                for i in order:
                    response = native.best_response(price, i)
                    price[i] = response["price"]
                    responses.append(response)
                rows.append(
                    {
                        "iteration": iteration + 1,
                        "prices": price.copy(),
                        "responses": responses,
                    }
                )
                if tuple(price) == before:
                    converged = True
                    break
            deviations = []
            for i in range(2):
                best = native.best_response(price, i)
                deviations.append(best["USD_profit"] - native.profit(price, i))
            error = float(np.max(np.abs(np.asarray(price) - target["prices"])))
            paths.append(
                {
                    "start": list(start),
                    "order": order,
                    "converged": converged,
                    "iterations": rows,
                    "final_prices": price,
                    "continuous_target_max_price_error": error,
                    "cent_deviation_gains_USD": deviations,
                }
            )
    passed = (
        all(x["share_max_abs_error"] <= 1e-12 for x in points)
        and all(v["abs_error"] <= 1e-5 for x in points for v in x["elasticities"])
        and all(
            x["converged"]
            and x["continuous_target_max_price_error"] <= 0.01 + 1e-12
            and max(x["cent_deviation_gains_USD"]) <= 1e-10
            for x in paths
        )
    )
    return {
        "points": points,
        "continuous_FOC_target": target,
        "native_cent_BR_paths": paths,
        "analytical_point_pass": passed,
        "scope": "One active period in native T2 with empty opening period; declared slack two-product examples and cent price class, not a general dynamic game certificate.",
    }


def sample(c, p, context):
    start, n = p["validation_seed_start"], p["n_validation"]
    if not 200000 <= start < start + n <= 300000:
        raise ValueError("validation seed range is outside the permitted split")
    engine = MarketEngine(c)
    prices = p["sampling_prices"]
    actions = {i: PriceAction({i: float(prices[i])}) for i in range(2)}
    counts, profits = np.zeros(4, dtype=np.int64), np.zeros(2)
    progress = {"stage": "native_sampler", "cohorts_completed": 0, "cohorts_total": n}
    context.counters = lambda: dict(progress)
    with (context.directory / "native_cohorts.jsonl").open("x") as handle:
        for seed in range(start, start + n):
            engine.reset(seed)
            empty = engine.step(actions)
            if empty.purchases or engine.world.request_log:
                raise AssertionError("zero-demand opening period generated an arrival")
            step = engine.step(actions)
            log = engine.world.request_log
            category = (
                3 if not log else 2 if log[0]["chosen"] is None else log[0]["chosen"]
            )
            if len(log) > 1 or len(step.purchases) > 1 or step.resolutions:
                raise AssertionError(
                    "sampler violated the slack one-active-period contract"
                )
            if any(
                float(engine.world.carriers[i].posted_prices[0]) != prices[i]
                for i in range(2)
            ):
                raise AssertionError("native executed prices changed")
            values = [carrier_metrics(engine.world, c, i)["profit"] for i in range(2)]
            expected = [
                prices[i] - c.product_cost[i] if category == i else 0.0
                for i in range(2)
            ]
            if not np.allclose(values, expected, rtol=0, atol=1e-12):
                raise AssertionError("native terminal net-dollar accounting mismatch")
            counts[category] += 1
            profits += values
            handle.write(
                canonical_json(
                    {
                        "seed": seed,
                        "category": int(category),
                        "profits_USD": values,
                        "tape_sha256": schema_hash(engine.tape),
                    }
                )
                + "\n"
            )
            progress["cohorts_completed"] = seed - start + 1
    q = contract(c)
    expected = np.r_[q["arrival"] * closed_shares(q, prices), 1 - q["arrival"]]
    test = chisquare(counts, n * expected)
    return {
        "cohorts": n,
        "validation_seed_start": start,
        "counts": counts.tolist(),
        "closed_probabilities": expected.tolist(),
        "chi2": float(test.statistic),
        "raw_p": float(test.pvalue),
        "mean_profit_USD": (profits / n).tolist(),
        "joint_Holm_decision": None,
        "row_sha256": file_hash(context.directory / "native_cohorts.jsonl"),
    }


def run(cell, context):
    p = cell["parameters"]
    source = Path(p["source_spec"])
    if file_hash(source) != p["source_spec_sha256"]:
        raise ValueError("frozen logit benchmark source changed")
    c = compile_market(load_spec(source))
    if c.spec_hash != p["compiled_spec_hash"]:
        raise ValueError("logit benchmark compiled hash changed")
    dump_spec(c.spec, context.directory / "spec.json")
    context.counters = lambda: {"stage": "analytical_comparisons"}
    native = NativePayoff(c)
    checks = analytical(native, p)
    atomic_json(context.directory / "analytical.json", checks)
    atomic_json(
        context.directory / "native_profit_objective_evaluations.json",
        native.objective_log,
    )
    measured = sample(c, p, context)
    result = {
        "status": "completed",
        "spec_hash": c.spec_hash,
        "analytical_point_pass": checks["analytical_point_pass"],
        "sampler": measured,
        "full_V4_pass": None,
        "scientific_budget": None,
        "test_seeds_opened": False,
        "scope": "V4 logit/Bertrand subblock; joint Holm decision only after both registered cells seal. No R1 learning or general-game/F3/F9/empirical acceptance.",
    }
    names = [
        "spec.json",
        "analytical.json",
        "native_profit_objective_evaluations.json",
        "native_cohorts.jsonl",
    ]
    result["artifact_hashes"] = {
        name: file_hash(context.directory / name) for name in names
    }
    atomic_json(context.directory / "logit_bertrand.json", result)
    return {
        **result,
        "summary_sha256": file_hash(context.directory / "logit_bertrand.json"),
    }
