"""Vectorized no-disruption Bernoulli rollouts, checked against MarketEngine.

This is a declared capability specialization, not a change of model. Unsupported
features use the reference engine in callers. Currency still uses Python round.
"""

import numpy as np
from .exogenous import transition_row
from .arrivals import period_requests

FEATURES = {
    "intercept",
    "connection_indicator",
    "scarcity_sum",
    "product_indicator",
    "time_to_departure",
    "direct_rival_gap",
}


def supports(compiled, focal, initial):
    """
    Check whether the vectorized specialization preserves this declared model (Eqs.
    3–16).
    """
    s = compiled.spec
    if (
        not initial
        or s.demand.mode != "single_request_bernoulli"
        or s.choice.model != "latent_class_mnl"
    ):
        return False
    if s.timing.mode != "leader_follower" or set(s.timing.leaders) != {
        c.code for i, c in enumerate(s.carriers) if i != focal
    }:
        return False
    p = next(p for p in s.pricing if p.carrier == s.carriers[focal].code)
    if (
        p.update_every != 1
        or p.sharing != "carrier"
        or any(
            f.name not in FEATURES
            for f in (*p.controllable_features, *[f for f, w in p.fixed_features])
        )
    ):
        return False
    if any(
        a.controller != "threshold_linear_rule"
        for a in s.controllers
        if a.carrier != s.carriers[focal].code
    ):
        return False
    if any(
        q.fare_noise_sd or q.missing_probability or q.fare_bin or q.every_k_periods != 1
        for q in s.quotes.requests
    ):
        return False
    if any(
        q.lag_periods != 1
        for q in s.quotes.requests
        if q.observer != s.carriers[focal].code
    ):
        return False
    if any(
        q.lag_periods != 0
        for q in s.quotes.requests
        if q.observer == s.carriers[focal].code
    ):
        return False
    for e in initial:
        x = e.world.exo
        if any(
            transition_row(compiled, x, t)[x] != 1.0
            for t in range(e.world.t, s.horizon.n_periods)
        ):
            return False
        if any(
            "hazard" in event.trigger or "period" in event.trigger
            for event in s.capacity_process.events
        ):
            return False
    return True


def rounded(x):
    """
    Apply Python cent rounding elementwise for exact reference-price identity (Eqs.
    3–16).
    """
    a = np.asarray(x)
    return np.fromiter(
        (round(float(v), 2) for v in a.flat), float, count=a.size
    ).reshape(a.shape)


def rollout_values(
    compiled, focal, initial, candidates, tail, *, hold=1, schedules=False
):
    """
    Evaluate coefficient candidates on common scenario tapes without disruptions (Eqs.
    3–16).
    """
    c = compiled
    s = c.spec
    S = len(initial)
    K = len(candidates)
    P = len(s.products)
    T = s.horizon.n_periods
    start = initial[0].world.t
    # One row per (candidate, scenario), sharing the scenario columns exactly.
    counts = np.zeros((K, S, P), int)
    cash = np.empty((K, S))
    previous = np.zeros((K, S, P))
    capacity = np.array(
        [
            np.concatenate(
                [e.world.carriers[i].capacity for i in range(len(s.carriers))]
            )
            for e in initial
        ]
    )
    # Global legs need not be grouped by carrier in the specification.
    capacity = np.zeros((S, len(s.legs)), int)
    for u, e in enumerate(initial):
        for i, cs in enumerate(e.world.carriers):
            capacity[u, list(c.legs_of[i])] = cs.capacity
            previous[:, u, list(c.products_of[i])] = cs.previous_prices
            for b in cs.bookings:
                counts[:, u, b.product] += 1
        cash[:, u] = e.world.carriers[focal].cash
    own = list(c.products_of[focal])
    ownlegs = list(c.legs_of[focal])
    price = next(p for p in s.pricing if p.carrier == s.carriers[focal].code)
    objective = next(o for o in s.objectives if o.carrier == s.carriers[focal].code)
    requests = [
        [period_requests(c, e.tape, t, e.world.exo) for t in range(start, T)]
        for e in initial
    ]

    def feature(f, j, inventory, prices, available, t):
        """
        Evaluate a supported observable coefficient feature across rollout rows (Eqs.
        3–16).
        """
        p = s.products[j]
        name = f.name
        if name == "intercept":
            return np.ones((K, S))
        if name == "connection_indicator":
            return np.full((K, S), len(p.legs) > 1, dtype=float)
        if name == "product_indicator":
            return np.full((K, S), p.id in f.params.get("products", ()), dtype=float)
        if name == "time_to_departure":
            return np.full((K, S), (T - t) / T)
        if name == "scarcity_sum":
            return sum(
                (float(c.A_global[l, j]) / (1 + inventory[:, :, l]) for l in ownlegs),
                np.zeros((K, S)),
            )
        if name == "direct_rival_gap":
            target = f.params.get("targets", {}).get(p.id)
            j2 = (
                c.product_index[target]
                if target
                else next(
                    (
                        x
                        for x in c.market_products[int(c.product_market[j])]
                        if c.product_carrier[x] != focal
                    ),
                    None,
                )
            )
            return (
                np.where(
                    available[:, :, j2],
                    prices[:, :, j2] - s.products[j2].reference_fare,
                    f.params.get("missing_value", 0.0),
                )
                if j2 is not None
                else np.zeros((K, S))
            )
        raise ValueError(name)

    for t in range(start, T):
        inventory = capacity[None, :, :] - np.einsum("ksp,lp->ksl", counts, c.A_global)
        available = np.all(
            inventory[:, :, :, None] >= c.A_global[None, None, :, :], axis=2
        )
        prices = previous.copy()
        for a in s.controllers:
            i = c.carrier_index[a.carrier]
            if i == focal:
                continue
            for j in c.products_of[i]:
                rule = a.params.get("rules", {}).get(
                    s.products[j].id, {"base": s.products[j].reference_fare}
                )
                fare = np.full(
                    (K, S), rule.get("base", s.products[j].reference_fare), dtype=float
                )
                for term in rule.get("scarcity", ()):
                    fare += term["coefficient"] * (
                        inventory[:, :, c.leg_index[term["leg"]]] <= term["threshold"]
                    )
                for term in rule.get("responses", ()):
                    fare += term["coefficient"] * (
                        previous[:, :, c.product_index[term["product"]]]
                        - term["reference"]
                    )
                prices[:, :, j] = rounded(fare)
        if schedules:
            theta = np.asarray([x[min(t - start, len(x) - 1)] for x in candidates])
        elif t - start < hold:
            theta = np.asarray(candidates)
        elif isinstance(tail, dict):
            theta = np.repeat(
                np.asarray(
                    tail["thetas"][
                        next(k for k, (a, b) in enumerate(tail["blocks"]) if a <= t < b)
                    ]
                )[None, :],
                K,
                axis=0,
            )
        else:
            theta = np.repeat(np.asarray(tail)[None, :], K, axis=0)
        for j in own:
            product = s.products[j]
            base = np.full(
                (K, S), price.base.get(product.id, product.reference_fare), dtype=float
            )
            base += price.time_ramp * t / (T - 1)
            for f, w in price.fixed_features:
                base += w * feature(f, j, inventory, prices, available, t)
            adjustment = np.zeros((K, S))
            for k, f in enumerate(price.controllable_features):
                adjustment += theta[:, k, None] * feature(
                    f, j, inventory, prices, available, t
                )
            prices[:, :, j] = rounded(
                np.clip(base + adjustment, product.fare_floor, product.fare_cap)
            )
        for u in range(S):
            if not requests[u][t - start]:
                continue
            r = requests[u][t - start][0]
            taste = s.segments[r.segment].classes[r.taste_class]
            market = s.markets[r.market].id
            utility = np.full((K, P + 1), -np.inf)
            utility[:, P] = 0.0
            for j in c.market_products[r.market]:
                p = s.products[j]
                penalty = (
                    s.metadata.get("connection_penalties", {}).get(market, 0.0)
                    if s.choice.connection_penalty_rule == "legacy_market_od"
                    else taste.connection_penalty * max(0, len(p.legs) - 1)
                )
                v = (
                    taste.market_value.get(market, taste.market_value.get("*", 0.0))
                    + taste.carrier_pref.get(p.carrier, 0.0)
                    - taste.alpha * prices[:, u, j]
                    - penalty
                )
                v -= taste.time_penalty_per_hour * p.attributes.get("duration_h", 0.0)
                if taste.ideal_dep_hour is not None:
                    v -= taste.schedule_penalty_per_hour * abs(
                        p.attributes.get("dep_hour", taste.ideal_dep_hour)
                        - taste.ideal_dep_hour
                    )
                v += taste.product_shift.get(p.id, 0.0)
                utility[:, j] = np.where(available[:, u, j], v, -np.inf)
            chosen = np.argmax(utility + initial[u].tape.gumbel[t, 0], axis=1)
            for k, j in enumerate(chosen):
                if j < P:
                    counts[k, u, j] += 1
                    if c.product_carrier[j] == focal:
                        cash[k, u] += prices[k, u, j]
        previous = prices
    profit = cash - np.einsum("ksp,p->ks", counts[:, :, own], c.product_cost[own])
    markets = set(objective.share_markets or [m.id for m in s.markets if m.in_segment])
    selected = [j for j, p in enumerate(s.products) if p.market in markets]
    mine = [j for j in selected if c.product_carrier[j] == focal]
    num = counts[:, :, mine].sum(2)
    den = counts[:, :, selected].sum(2)
    share = np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)
    return profit / objective.profit_scale + objective.share_weight * share
