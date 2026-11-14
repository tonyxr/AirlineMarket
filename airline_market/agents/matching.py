"""Observed same-market fare matching, with an explicit reference rule."""

from ..core.types import PriceAction
from ..core.money import round_fare, clip_fare
from .base import StatelessController
from .price_rule import visible


class CompetitorMatchingController(StatelessController):
    """Price response to visible same-market rival offers (Eqs. 18–19)."""

    policy_class = "price"

    def __init__(self, offset: float = 0.0, fallback: float | None = None):
        """
        Bind declared inputs and initialize CompetitorMatchingController bookkeeping
        (Eqs. 18–19).
        """
        self.offset = offset
        self.fallback = fallback

    def act(self, obs):
        """
        Return the controller's action using only its permitted observation (Eqs.
        18–19).
        """
        c = self.ctx.compiled
        prices = {}
        for j in c.products_of[self.carrier]:
            product = c.spec.products[j]
            fares = [
                q.fare
                for q in obs.quotes
                if q.product < len(c.spec.products)
                and c.product_market[q.product] == c.product_market[j]
                and visible(q)
            ]
            fare = (
                min(fares) + self.offset
                if fares
                else self.fallback
                if self.fallback is not None
                else product.reference_fare
            )
            prices[j] = round_fare(
                clip_fare(fare, product.fare_floor, product.fare_cap)
            )
        return PriceAction(prices)
