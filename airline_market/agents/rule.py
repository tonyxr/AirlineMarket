"""Inventory thresholds and lagged competitor-price responses."""

from ..core.types import PriceAction
from ..core.money import round_fare
from .base import StatelessController


class ThresholdLinearRuleController(StatelessController):
    """
    Inventory thresholds plus declared lagged competitor-price responses (Eqs. 18–19).
    """

    policy_class = "price"

    def __init__(self, rules):
        """
        Bind declared inputs and initialize ThresholdLinearRuleController bookkeeping
        (Eqs. 18–19).
        """
        self.rules = rules

    def act(self, obs):
        """
        Return the controller's action using only its permitted observation (Eqs.
        18–19).
        """
        c = self.ctx.compiled
        quote = {q.product: q for q in obs.quotes}
        prices = {}
        for j in c.products_of[self.carrier]:
            p = c.spec.products[j]
            rule = self.rules.get(p.id, {"base": p.reference_fare})
            fare = rule.get("base", p.reference_fare)
            for term in rule.get("scarcity", ()):
                local = c.legs_of[self.carrier].index(c.leg_index[term["leg"]])
                fare += term["coefficient"] * (
                    obs.own_inventory[local] <= term["threshold"]
                )
            for term in rule.get("responses", ()):
                q = quote.get(c.product_index[term["product"]])
                # An explicitly supplied reference is the rule's initial-price convention.
                previous = q.fare if q and q.fare is not None else term["reference"]
                fare += term["coefficient"] * (previous - term["reference"])
            prices[j] = round_fare(fare)
        return PriceAction(prices)
