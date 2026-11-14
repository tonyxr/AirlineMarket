"""Myopic coefficient response under observed/reference competitive offers."""

import numpy as np
from ..core.types import CoefficientAction
from ..sim.choice import class_mixture_probabilities
from ..sim.arrivals import bernoulli_probability
from ..sim.tape import segment_weights
from .base import StatelessController
from .price_rule import AdditivePriceRule
from .forecast import reference_offers


class MyopicBestResponseController(StatelessController):
    """
    One-period coefficient response under a declared point rival estimate (Eqs. 18–19).
    """

    def __init__(self, theta_grid=None, rival_counts=None):
        """
        Bind declared inputs and initialize MyopicBestResponseController bookkeeping
        (Eqs. 18–19).
        """
        self.grid = theta_grid
        self.rival_counts = rival_counts or {}

    def act(self, obs):
        """
        Return the controller's action using only its permitted observation (Eqs.
        18–19).
        """
        c = self.ctx.compiled
        i = self.carrier
        grid = self.grid or self.ctx.pricing.theta_grid
        rule = AdditivePriceRule(c, self.ctx.pricing, self.ctx.arm)
        values = []
        objective = self.ctx.objective
        share_markets = set(
            objective.share_markets or [m.id for m in c.spec.markets if m.in_segment]
        )
        own_n = sum(
            c.spec.products[b.product].market in share_markets for b in obs.own_bookings
        )
        other_n = sum(
            n
            for product, n in self.rival_counts.items()
            if c.spec.products[c.product_index[product]].market in share_markets
        )
        base_share = own_n / (own_n + other_n) if own_n + other_n else 0.0
        exo = next(
            (
                k
                for k, e in enumerate(c.spec.exogenous.states)
                if obs.public_exo.get("state:" + e.name, 0.0)
            ),
            0,
        )
        for theta in grid:
            prices, active = reference_offers(c, i, obs)
            own = rule.prices(theta, obs)
            prices[list(c.products_of[i])] = own
            value = 0.0
            for m, market in enumerate(c.spec.markets):
                multiplier = (
                    c.spec.demand.holiday_multiplier
                    * c.spec.demand.exogenous_multipliers.get(
                        c.spec.exogenous.states[exo].name, 1.0
                    )
                )
                if c.spec.demand.mode == "poisson":
                    arrivals = (
                        c.spec.demand.intensity[market.id].value_at(obs.t) * multiplier
                    )
                else:
                    arrivals = (
                        bernoulli_probability(c, obs.t, exo)
                        * c.spec.demand.market_weights.value_at(obs.t)[m]
                    )
                for g, w in enumerate(segment_weights(c, obs.t, m)):
                    p = class_mixture_probabilities(c, m, g, prices, active)
                    for local, j in enumerate(c.market_products[m]):
                        mine = c.product_carrier[j] == i
                        if mine:
                            value += (
                                arrivals
                                * w
                                * p[local]
                                * (prices[j] - c.product_cost[j])
                                / objective.profit_scale
                            )
                        if market.id in share_markets:
                            after = (own_n + int(mine)) / (own_n + other_n + 1)
                            value += (
                                arrivals
                                * w
                                * p[local]
                                * objective.share_weight
                                * (after - base_share)
                            )
            values.append(value)
        return CoefficientAction(tuple(grid[int(np.argmax(values))]))
