"""Known-demand reference-price forecasts from permitted observations only."""

import numpy as np
from ..sim.choice import class_mixture_probabilities
from ..sim.exogenous import transition_row
from ..sim.tape import segment_weights
from ..sim.ledger import sellable
from .price_rule import visible


def reference_offers(compiled, carrier: int, obs):
    """
    Combine reference fares and visible rival quotes without hidden inventory (Eqs.
    18–19).
    """
    prices = np.array([p.reference_fare for p in compiled.spec.products], dtype=float)
    active = np.ones(len(prices), dtype=bool)
    for q in obs.quotes:
        if q.product >= len(prices) or q.missing:
            continue
        if q.fare is not None:
            prices[q.product] = q.fare
        if q.available is not None:
            active[q.product] = q.available
        elif q.seats is not None:
            active[q.product] = q.seats > 0
    active[list(compiled.products_of[carrier])] = sellable(
        obs.own_inventory, compiled.A[carrier]
    )
    return prices, active


def remaining_demand(compiled, carrier: int, obs, prices=None):
    """
    Forecast remaining product demand at reference offers under the known model (Eqs.
    18–19).
    """
    c = compiled
    s = c.spec
    reference, active = reference_offers(c, carrier, obs)
    if prices is not None:
        reference[list(c.products_of[carrier])] = prices
    result = np.zeros(len(c.products_of[carrier]))
    position = {j: k for k, j in enumerate(c.products_of[carrier])}
    exo = np.array(s.exogenous.initial, dtype=float)
    known = [
        k
        for k, e in enumerate(s.exogenous.states)
        if obs.public_exo.get("state:" + e.name, 0)
    ]
    if known:
        exo = np.eye(len(exo))[known[0]]
    for t in range(obs.t, s.horizon.n_periods):
        for m, market in enumerate(s.markets):
            expected = 0.0
            for state, p in enumerate(exo):
                multiplier = (
                    s.demand.holiday_multiplier
                    * s.demand.exogenous_multipliers.get(
                        s.exogenous.states[state].name, 1.0
                    )
                )
                if s.demand.mode == "poisson":
                    value = s.demand.intensity[market.id].value_at(t) * multiplier
                else:
                    value = (
                        min(
                            s.demand.arrival_probability_cap,
                            s.demand.arrival_probability.value_at(t) * multiplier,
                        )
                        * s.demand.market_weights.value_at(t)[m]
                    )
                expected += p * value
            for g, w in enumerate(segment_weights(c, t, m)):
                probabilities = class_mixture_probabilities(c, m, g, reference, active)
                for local, j in enumerate(c.market_products[m]):
                    if j in position:
                        result[position[j]] += expected * w * probabilities[local]
        exo = sum(
            (p * np.array(transition_row(c, k, t)) for k, p in enumerate(exo)),
            np.zeros(len(exo)),
        )
    return result
