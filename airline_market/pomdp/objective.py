"""Per-period cash plus terminal service/share reward (Eqs. 20–22, 27)."""

from ..sim.accounting import carrier_metrics


class ObjectiveEvaluator:
    """Cash increments plus terminal service/share reward decomposition (Eqs. 24–27)."""

    def __init__(self, objective_spec, compiled):
        """
        Bind declared inputs and initialize ObjectiveEvaluator bookkeeping (Eqs. 24–27).
        """
        self.spec = objective_spec
        self.compiled = compiled

    def period_reward(self, carrier: int, before, after) -> float:
        """
        Return the focal cash increment in the objective's declared scale (Eqs. 24–27).
        """
        return (
            after.carriers[carrier].cash - before.carriers[carrier].cash
        ) / self.spec.profit_scale

    def terminal_reward(self, world) -> float:
        """
        Settle active service cost and realized competitive journey share (Eqs. 24–27).
        """
        i = self.compiled.carrier_index[self.spec.carrier]
        m = carrier_metrics(world, self.compiled, i)
        return (
            -m["service_cost"] / self.spec.profit_scale
            + self.spec.share_weight * m["share"]
        )
