"""Exact sparse access to immutable itinerary incidence in isolated workers.

The reference evaluator finds each product's tuple position and scans every
carrier leg for each scalar feature and quote. Cache the static positions and
visit only used resources. No random draws, actor inputs or price arithmetic
are changed. The original source and active exact campaigns stay untouched.
"""

from contextlib import contextmanager
import sys
import numpy as np

from ..agents import price_rule
from . import quotes
from .count_ledger import counted_inventory, counted_ledger

ReferenceEvaluator = price_rule.FeatureEvaluator
reference_posted_quotes = quotes.posted_quotes
_layouts = {}


def layout(compiled):
    """Cache static local product/resource positions for one compiled identity."""
    key = id(compiled)
    if key not in _layouts:
        rows = []
        for j, product in enumerate(compiled.spec.products):
            carrier = int(compiled.product_carrier[j])
            local = compiled.products_of[carrier].index(j)
            used = tuple(int(k) for k in np.flatnonzero(compiled.A[carrier][:, local]))
            rows.append((carrier, local, used))
        _layouts[key] = (compiled, tuple(rows))
    return _layouts[key][1]


class SparseFeatureEvaluator(ReferenceEvaluator):
    """Evaluate common own-network features in itinerary length rather than hub size."""

    def __init__(self, compiled, carrier):
        """Build immutable incidence positions; retain the original quote evaluator."""
        super().__init__(compiled, carrier)
        self.layout = layout(compiled)

    def __call__(self, feature, j, obs):
        """Preserve scalar summation order and use only the permitted observation."""
        name = feature.name
        product = self.c.spec.products[j]
        if name == "intercept":
            return 1.0
        if name == "product_indicator":
            return float(product.id in feature.params.get("products", ()))
        if name == "connection_indicator":
            return float(len(product.legs) > 1)
        if name == "time_to_departure":
            return (
                self.c.spec.horizon.n_periods - obs.t
            ) / self.c.spec.horizon.n_periods
        if name == "scarcity_sum":
            return sum(1.0 / (1 + obs.own_inventory[k]) for k in self.layout[j][2])
        if name == "scarcity_max":
            return max(
                (
                    obs.own_capacity[k] / (1 + obs.own_inventory[k])
                    for k in self.layout[j][2]
                ),
                default=0.0,
            )
        return super().__call__(feature, j, obs)


def sparse_posted_quotes(world, compiled):
    """Construct the same ordered true offers using only each itinerary's legs."""
    values = []
    inventories = [
        counted_inventory(state, compiled, i) for i, state in enumerate(world.carriers)
    ]
    for j, (carrier, local, used) in enumerate(layout(compiled)):
        seats = min(int(inventories[carrier][k]) for k in used)
        values.append(
            quotes.TrueQuote(
                j,
                world.t,
                float(world.carriers[carrier].posted_prices[local]),
                seats > 0,
                max(0, seats),
            )
        )
    for k, (state, spec) in enumerate(zip(world.background, compiled.spec.background)):
        fare, active, seats = quotes.background_quote(spec, state.stock)
        values.append(
            quotes.TrueQuote(
                len(compiled.spec.products) + k, world.t, fare, active, seats
            )
        )
    return tuple(values)


@contextmanager
def sparse_access():
    """Install exact primitives in this dedicated worker, restoring imported aliases."""
    if price_rule.FeatureEvaluator is SparseFeatureEvaluator:
        yield
        return
    replacements = {
        ReferenceEvaluator: SparseFeatureEvaluator,
        reference_posted_quotes: sparse_posted_quotes,
    }

    def modules():
        """Include maintained aliases loaded before or during the worker context."""
        return [
            module
            for name, module in list(sys.modules.items())
            if module is not None
            and name.startswith("airline_market.")
            and name != __name__
            and not name.startswith(
                ("airline_market.vendor.", "airline_market.legacy_toy.")
            )
        ]

    with counted_ledger():
        for module in modules():
            for name, value in list(vars(module).items()):
                for original, replacement in replacements.items():
                    if value is original:
                        setattr(module, name, replacement)
        try:
            yield
        finally:
            for module in modules():
                for name, value in list(vars(module).items()):
                    for original, replacement in replacements.items():
                        if value is replacement:
                            setattr(module, name, original)
            _layouts.clear()
