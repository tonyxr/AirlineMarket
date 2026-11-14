"""Opt-in fixed-fare request/admission mode using native ledger Eq.12–15.

A request's class is observed before its accept/reject decision. This is a
separate RM mode; the existing MarketEngine MNL/pricing protocol is unchanged.
"""

import math
from dataclasses import dataclass

import numpy as np

from airline_market.core.types import PeriodPhase
from airline_market.sim.arrivals import Request
from airline_market.sim.ledger import book, inventory, sellable
from airline_market.sim.quotes import QuoteHistory
from airline_market.sim.state import CarrierState, WorldState


@dataclass(frozen=True)
class AdmissionObservation:
    """Permitted public time, own stock and the currently revealed request class."""

    period: int
    own_inventory: tuple[int, ...]
    requested_product: int | None
    fixed_fare_USD: float | None
    can_accept: bool


class FixedFareAdmissionEngine:
    """Explicit accept/reject at fixed fares, with native booking/cash primitives."""

    def __init__(self, compiled, probabilities):
        self.compiled = compiled
        s = compiled.spec
        T = s.horizon.n_periods
        if (
            len(s.carriers) != 1
            or s.background
            or s.capacity_priors
            or s.capacity_process.events
            or s.quotes.requests
            or len(s.exogenous.states) != 1
            or s.exogenous.transitions
            or s.objectives[0].share_weight != 0
            or s.objectives[0].profit_scale != 1
        ):
            raise ValueError(
                "fixed-fare mode requires one public carrier with no events/signals/share bonus"
            )
        if any(
            not l.capacity.public
            or len(l.capacity.support) != 1
            or l.capacity.probabilities != (1.0,)
            for l in s.legs
        ):
            raise ValueError("deterministic public capacity required")
        if len(probabilities) != T or any(
            len(row) != len(s.products) for row in probabilities
        ):
            raise ValueError(
                "one probability per period and fixed product class required"
            )
        if any(
            not math.isfinite(x) or x < 0 for row in probabilities for x in row
        ) or any(math.fsum(row) > 1 for row in probabilities):
            raise ValueError(
                "class probabilities must be finite, nonnegative and sum to at most1"
            )
        self.probabilities = tuple(tuple(map(float, row)) for row in probabilities)
        self.fares = tuple(float(p.reference_fare) for p in s.products)
        self.horizon = T
        self.reset()

    def reset(self):
        """Initialize deterministic physical stock and native empty ledgers; no RNG."""
        c = self.compiled
        capacity = np.asarray(
            [l.capacity.support[0] for l in c.spec.legs], dtype=np.int64
        )
        prices = np.asarray(self.fares, dtype=np.float64)
        state = CarrierState(
            capacity.copy(), [], 0.0, 0.0, 0.0, 0, prices.copy(), prices.copy()
        )
        self.world = WorldState(
            0,
            PeriodPhase.PRE_DECISION,
            0,
            [state],
            [],
            QuoteHistory(),
            initial_capacity=tuple(map(int, capacity)),
        )

    def observe(self, requested_product):
        """Reveal only this request class and own public inventory before action."""
        if self.world.t >= self.horizon:
            raise ValueError("episode is terminal")
        if requested_product is not None and (
            type(requested_product) is not int
            or not 0 <= requested_product < len(self.fares)
        ):
            raise ValueError("unknown request class")
        stock = inventory(self.world.carriers[0], self.compiled, 0)
        feasible = (
            False
            if requested_product is None
            else bool(sellable(stock, self.compiled.A[0])[requested_product])
        )
        return AdmissionObservation(
            self.world.t,
            tuple(map(int, stock)),
            requested_product,
            None if requested_product is None else self.fares[requested_product],
            feasible,
        )

    def step(self, requested_product, accept):
        """Reject without changing receipts/stock, or book all native itinerary legs."""
        obs = self.observe(requested_product)
        if type(accept) is not bool or (requested_product is None and accept):
            raise ValueError("boolean admission for a revealed request required")
        booking = None
        if accept:
            request = Request(
                0, int(self.compiled.product_market[requested_product]), 0, 0, 0, 0
            )
            booking = book(
                self.world,
                0,
                requested_product,
                obs.fixed_fare_USD,
                request,
                self.compiled,
            )
        if requested_product is not None:
            self.world.request_log.append(
                {
                    "period": obs.period,
                    "market": int(self.compiled.product_market[requested_product]),
                    "requested_product": requested_product,
                    "chosen": requested_product if accept else None,
                    "admitted": accept,
                    "segment": 0,
                    "taste_class": 0,
                }
            )
        self.world.t += 1
        self.world.phase = (
            PeriodPhase.TERMINAL
            if self.world.t == self.horizon
            else PeriodPhase.PRE_DECISION
        )
        return booking

    def clone(self):
        """Clone mutable native ledgers; share immutable compiled inputs only."""
        other = object.__new__(type(self))
        other.compiled, other.probabilities, other.fares, other.horizon = (
            self.compiled,
            self.probabilities,
            self.fares,
            self.horizon,
        )
        other.world = self.world.clone()
        return other
