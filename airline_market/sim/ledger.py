"""Shared seat and cash accounting for sampler/kernel (Eqs. 12–15)."""

from dataclasses import dataclass
import numpy as np
from .state import Booking


def counts(carrier_state, compiled, i: int) -> np.ndarray:
    """Count active bookings in the carrier's compiled product order (Eqs. 3–16)."""
    return np.array(
        [
            sum(b.product == j for b in carrier_state.bookings)
            for j in compiled.products_of[i]
        ],
        dtype=np.int64,
    )


def inventory(carrier_state, compiled, i: int) -> np.ndarray:
    """Subtract the itinerary incidence use from leg capacity (Eqs. 3–16)."""
    return carrier_state.capacity - compiled.A[i] @ counts(carrier_state, compiled, i)


def sellable(inv, A_i) -> np.ndarray:
    """Test that every leg of each itinerary has a remaining seat (Eqs. 3–16)."""
    return np.all(A_i <= np.asarray(inv)[:, None], axis=0)


def book(world, carrier: int, product: int, fare: float, request, compiled) -> Booking:
    """
    Reserve all itinerary legs and append the corresponding fare receipt (Eqs. 3–16).
    """
    state = world.carriers[carrier]
    local = compiled.products_of[carrier].index(product)
    if not sellable(inventory(state, compiled, carrier), compiled.A[carrier])[local]:
        raise ValueError("booking exceeds available leg inventory")
    b = Booking(
        world.next_record_id,
        carrier,
        product,
        float(fare),
        world.t,
        request.rank,
        request.segment,
        request.taste_class,
    )
    world.next_record_id += 1
    state.bookings.append(b)
    state.cash += fare
    state.revenue += fare
    return b


@dataclass(frozen=True)
class ResolutionOutcome:
    """
    Whole-itinerary cancellations with original-fare refunds and penalties (Eqs. 3–16).
    """

    canceled: tuple[Booking, ...]
    refund: float
    penalty: float


def resolve(world, carrier: int, compiled, resolution=None) -> ResolutionOutcome:
    """Cancel/refund whole bookings in the declared fixed priority order (Eqs. 3–16)."""
    rule = resolution or compiled.spec.resolution
    state = world.carriers[carrier]
    remaining = state.capacity.copy()
    kept = []
    removed = []
    order = {j: k for k, j in enumerate(compiled.priority_order[carrier])}
    for b in sorted(state.bookings, key=lambda b: (order[b.product], b.record_id)):
        use = compiled.A_global[list(compiled.legs_of[carrier]), b.product]
        if np.all(use <= remaining):
            kept.append(b)
            remaining -= use
        else:
            state.cash -= b.fare + rule.penalty_per_cancellation
            state.refunds += b.fare
            state.penalties += rule.penalty_per_cancellation
            state.canceled += 1
            removed.append(b)
    state.bookings = kept
    return ResolutionOutcome(
        tuple(removed),
        sum(b.fare for b in removed),
        len(removed) * rule.penalty_per_cancellation,
    )
