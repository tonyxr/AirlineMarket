"""R5/A.3: exact physical diagnostics of a permitted optimal pricing policy.

Diagnostics traverse the full reachable compact posterior, independently of
value-cache reductions. Carrier load factor is E[occupied leg seats] divided
by E[initial leg seats], over all physical legs. Fare bounds count sellable
focal product-offer decisions; spill counts blocked first choices under the
same full offer set (before substitution), with all signs/masses retained.
"""

from collections import defaultdict
from functools import lru_cache

import numpy as np

from airline_market.belief.info import total_correlation
from airline_market.sim.arrivals import bernoulli_probability
from airline_market.sim.choice import choice_probabilities, systematic_utilities
from airline_market.sim.tape import segment_weights
from airline_market.solvers.exact_dp import argmax_first

NAMES = (
    "remaining_A",
    "remaining_B",
    "rival_sold_out",
    "revenue_A",
    "revenue_B",
    "tickets_A",
    "tickets_B",
    "offered_fare_sum",
    "offer_count",
    "fare_bound_count",
    "spill",
)


def period_stats(model, node, index):
    """Integrate pre-action-state offer and sales diagnostics; never select on truth."""
    m = model
    t, capacity, own, _journeys, previous, signal, belief = node
    c = m.compiled
    prices = m.prices(m._diagnostic_grid[index], t, capacity, own, signal)
    stats = np.zeros(len(NAMES) + m.T + 1)
    local_active = [all(x >= u for x, u in zip(own, use)) for use in m.own_uses]
    for active, fare, j in zip(local_active, prices, m.own_products):
        if active:
            p = c.spec.products[j]
            stats[7] += fare
            stats[8] += 1
            stats[9] += int(fare == p.fare_floor or fare == p.fare_cap)
    stock_law = defaultdict(float)
    for hidden, weight in belief:
        stock_law[hidden[0]] += weight
        probabilities = m.outcomes(t, own, hidden[0], prices, previous)
        rivals = m.rival_quotes(t, hidden[0], previous)
        stats[3] += weight * sum(
            probabilities[j] * fare for j, fare in zip(m.own_products, prices)
        )
        stats[4] += weight * sum(
            probabilities[j] * quote[0] for j, quote in zip(m.rival_products, rivals)
        )
        stats[5] += weight * sum(probabilities[j] for j in m.own_products)
        stats[6] += weight * sum(probabilities[j] for j in m.rival_products)
        global_prices = np.zeros(len(c.spec.products))
        global_prices[list(m.own_products)] = prices
        global_prices[list(m.rival_products)] = [x[0] for x in rivals]
        active = np.zeros(len(c.spec.products), dtype=bool)
        active[list(m.own_products)] = local_active
        active[list(m.rival_products)] = [x[1] for x in rivals]
        full = np.ones(len(active), dtype=bool)
        arrival = bernoulli_probability(c, t, m.exos[t])
        for market, mw in enumerate(c.spec.demand.market_weights.value_at(t)):
            for seg, sw in enumerate(segment_weights(c, t, market)):
                for taste in c.spec.segments[seg].classes:
                    probs = choice_probabilities(
                        systematic_utilities(c, market, taste, global_prices, full)
                    )
                    blocked = sum(
                        probs[k]
                        for k, j in enumerate(c.market_products[market])
                        if not active[j]
                    )
                    stats[10] += weight * arrival * mw * sw * taste.weight * blocked
    stats[len(NAMES) + t] = total_correlation(stock_law)
    return stats


def exact_diagnostics(dp, *, policy_sink=None):
    """Forward policy evaluation under the true joint posterior, including after own sellout."""
    m = dp.finite.model
    m._diagnostic_grid = dp.grid
    roots = [(p, m.project(n.observation, n.belief)) for p, n in dp.roots()]
    initial_A = sum(p * sum(n[1]) for p, n in roots)
    initial_B = sum(p * sum(w * sum(h[0]) for h, w in n[-1]) for p, n in roots)

    @lru_cache(None)
    def evaluate(node):
        t, _capacity, own, _journeys, _previous, _signal, belief = node
        if t == m.T:
            result = np.zeros(len(NAMES) + m.T + 1)
            result[0] = sum(own)
            result[1] = sum(w * sum(h[0]) for h, w in belief)
            result[2] = sum(
                w
                * int(
                    not any(
                        all(x >= u for x, u in zip(h[0], use)) for use in m.rival_uses
                    )
                )
                for h, w in belief
            )
            return result
        # The choice reads only node observation and posterior, never a sampled hidden state.
        index = argmax_first(dp.finite.q_values(node))
        if policy_sink is not None:
            policy_sink(node, index)
        result = period_stats(m, node, index)
        for p, _, child in dp.finite.successors(node, index):
            result += p * evaluate(child)
        return result

    total = sum((p * evaluate(n) for p, n in roots), np.zeros(len(NAMES) + m.T + 1))
    result = dict(zip(NAMES, map(float, total[: len(NAMES)])))
    result.update(
        initial_seat_capacity_A=initial_A,
        initial_seat_capacity_B=initial_B,
        load_factor_A=1 - total[0] / initial_A if initial_A else None,
        load_factor_B=1 - total[1] / initial_B if initial_B else None,
        mean_offered_fare=total[7] / total[8] if total[8] else None,
        share_fare_at_bounds=total[9] / total[8] if total[8] else None,
        mean_paid_fare_A=total[3] / total[5] if total[5] else None,
        posterior_total_correlation_by_period=list(map(float, total[len(NAMES) :])),
        diagnostic_states=evaluate.cache_info().currsize,
    )
    result["plausible"] = bool(
        all(
            result["load_factor_" + x] is not None
            and 0.70 <= result["load_factor_" + x] <= 0.95
            for x in ("A", "B")
        )
        and result["rival_sold_out"] <= 0.6
        and result["share_fare_at_bounds"] is not None
        and result["share_fare_at_bounds"] < 0.8
    )
    return result
