"""Explicit opt-in net objective under declared fully refundable pass-through fees."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

from airline_market.sim.accounting import carrier_metrics


@dataclass(frozen=True)
class RefundableFee:
    """Declared affine fee clipped to the actual booked all-in fare, in USD."""

    rate: float
    flat_USD: float

    def __post_init__(self):
        if not math.isfinite(self.rate) or not 0 <= self.rate <= 1:
            raise ValueError("fee rate must be finite and in[0,1]")
        if not math.isfinite(self.flat_USD) or self.flat_USD < 0:
            raise ValueError("flat fee must be finite and nonnegative")

    def amount(self, booked_fare):
        """Use the original transaction fare, never a current quote or price."""
        p = float(booked_fare)
        if not math.isfinite(p) or p < 0:
            raise ValueError("booked fare must be finite and nonnegative")
        return min(p, max(0.0, self.rate * p + self.flat_USD))


def net_contribution_metrics(
    world,
    compiled,
    carrier,
    rules: Mapping[str, RefundableFee],
    *,
    fees_fully_refundable,
):
    """Evaluate an explicit objective without mutating native cash or observations.

    The model contract assumes fees are paid out on booking and fully returned
    when the native original-fare refund occurs. Service cost and cancellation
    penalties are exactly the existing native accounting terms. Other fee
    ownership/refund contracts are deliberately not inferred from public fares.
    """
    if (
        fees_fully_refundable is not True
        or compiled.spec.resolution.refund != "full_original_fare"
    ):
        raise ValueError(
            "this objective requires explicit fully refundable fees and original-fare refunds"
        )
    own = compiled.products_of[carrier]
    if any(compiled.spec.products[j].id not in rules for j in own):
        raise ValueError("an explicit fee rule is required for every own product")
    if any(
        not isinstance(rules[compiled.spec.products[j].id], RefundableFee) for j in own
    ):
        raise TypeError("fee rules must be declared RefundableFee values")
    native = carrier_metrics(world, compiled, carrier)
    fee = math.fsum(
        rules[compiled.spec.products[b.product].id].amount(b.fare)
        for b in world.carriers[carrier].bookings
    )
    net_cash = world.carriers[carrier].cash - fee
    net = net_cash - native["service_cost"]
    code = compiled.spec.carriers[carrier].code
    objective = next(o for o in compiled.spec.objectives if o.carrier == code)
    return {
        "native_gross_contribution_USD": native["profit"],
        "active_refundable_fee_liability_USD": fee,
        "net_cash_USD": net_cash,
        "native_service_cost_USD": native["service_cost"],
        "native_penalties_USD": world.carriers[carrier].penalties,
        "declared_net_contribution_USD": net,
        "native_share": native["share"],
        "declared_net_objective": net / objective.profit_scale
        + objective.share_weight * native["share"],
        "native_default_objective": native["objective"],
        "contract": "Fully refundable declared per-product fee; native full original-fare refund/active service-cost law unchanged.",
        "empirical_net_profit_identified": False,
        "default_engine_objective_changed": False,
    }
