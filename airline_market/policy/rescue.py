"""A.14/B0/V9: credentialled rescue-fare options for native stranded passengers.

Native-period eligibility only; no legal dated booking/PNR or planner claim.
A cheaper ordinary offer remains available, and active caps still apply.
"""

import copy
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from itertools import pairwise

from airline_market.policy.regulator import _interval
from airline_market.sim.arrivals import Request
from airline_market.sim.choice import sampled_choice, systematic_utilities
from airline_market.sim.ledger import book
from airline_market.sim.rebooking import RebookingMarketEngine
from airline_market.sim.state import Booking


def _positive(value, *, cent=False):
    try:
        number = Decimal(str(value))
        valid = number.is_finite() and number > 0
        if cent:
            valid = valid and number == number.quantize(Decimal("0.01"))
        if not valid:
            raise ValueError(
                "positive finite cent fare" if cent else "positive finite distance"
            )
        return number
    except (InvalidOperation, TypeError) as error:
        raise ValueError("invalid fare or distance") from error


@dataclass(frozen=True)
class RescueFareMenu:
    """One declared design on [start, stop); distance is explicit O&D km input.

    two_tier: two upper-bound/fare pairs, with final bound None.
    counter_three_tier: three pairs, sold only on a proved counter visit.
    same_route: exact compiled-market IDs mapped to fixed fare options.
    No distance, window or price is inferred from outcomes.
    """

    start: int
    stop: int
    design: str
    distance_tiers: tuple[tuple[object, object], ...] = ()
    market_distances_km: tuple[tuple[str, object], ...] = ()
    route_fares: tuple[tuple[str, object], ...] = ()

    def __post_init__(self):
        _interval(self.start, self.stop)
        if self.design not in {"two_tier", "counter_three_tier", "same_route"}:
            raise ValueError("unknown rescue menu design")
        distances = tuple(
            sorted((key, _positive(value)) for key, value in self.market_distances_km)
        )
        fares = tuple(
            sorted(
                (key, _positive(value, cent=True)) for key, value in self.route_fares
            )
        )
        for rows in [distances, fares]:
            if len({key for key, _ in rows}) != len(rows) or any(
                not isinstance(key, str) or not key for key, _ in rows
            ):
                raise ValueError("unique nonempty compiled market keys required")
        tiers = tuple(
            (None if upper is None else _positive(upper), _positive(fare, cent=True))
            for upper, fare in self.distance_tiers
        )
        if self.design == "same_route":
            if not fares or tiers or distances:
                raise ValueError("same-route design requires route fares only")
        else:
            required = 2 if self.design == "two_tier" else 3
            if (
                len(tiers) != required
                or not distances
                or fares
                or tiers[-1][0] is not None
            ):
                raise ValueError(
                    "distance menu requires its declared tiers and explicit distances"
                )
            bounds = [bound for bound, _ in tiers[:-1]]
            if any(b is None for b in bounds) or any(
                a >= b for a, b in pairwise(bounds)
            ):
                raise ValueError(
                    "distance boundaries must strictly increase, with final None"
                )
        object.__setattr__(self, "distance_tiers", tiers)
        object.__setattr__(self, "market_distances_km", distances)
        object.__setattr__(self, "route_fares", fares)

    @property
    def counter_only(self):
        return self.design == "counter_three_tier"

    def fare_for(self, market, period):
        """Return a fixed option, never a price based on hidden inventory."""
        if not self.start <= period < self.stop:
            return None
        if self.design == "same_route":
            return dict(self.route_fares).get(market)
        distance = dict(self.market_distances_km).get(market)
        if distance is None:
            return None
        for upper, fare in self.distance_tiers:
            if upper is None or distance <= upper:
                return fare
        raise AssertionError("validated distance menu has no final tier")


@dataclass(frozen=True)
class PresentedItinerary:
    original_record: int
    period: int
    airport_counter: bool


@dataclass(frozen=True)
class RescueSnapshot:
    native: object
    proofs: tuple[PresentedItinerary, ...]


class RescueMarketEngine(RebookingMarketEngine):
    """Private rescue option on a genuine native cancellation lineage.

    Ordinary passengers never enter this offer path. The same product/taste
    innovation serves ordinary and rescue fare options, so the cheaper fare
    dominates without creating a second physical alternative or passenger.
    """

    def __init__(self, compiled, controllers=None, *, rescue_menu=None, **kwargs):
        if rescue_menu is not None and not isinstance(rescue_menu, RescueFareMenu):
            raise TypeError("explicit RescueFareMenu required")
        super().__init__(compiled, controllers, **kwargs)
        if rescue_menu is not None:
            targets = {
                key
                for key, _ in (
                    *rescue_menu.route_fares,
                    *rescue_menu.market_distances_km,
                )
            }
            if targets - set(compiled.market_index):
                raise ValueError("rescue menu names an unknown compiled market")
        self.rescue_menu = rescue_menu
        self.presented_itineraries = ()
        self._presented = set()
        self._counter_visits = set()

    def reset(self, seed=None, *, tape=None):
        self.presented_itineraries = ()
        self._presented = set()
        self._counter_visits = set()
        super().reset(seed, tape=tape)

    def present_cancelled_itinerary(self, booking, *, airport_counter=False):
        """Operator supplies a native cancelled ticket; never trust an actor claim.

        A counter proof is valid for this purchase period only. External legal
        identity/PNR authentication and dated departure validity are not modeled.
        """
        if type(airport_counter) is not bool:
            raise TypeError("counter evidence must be an explicit boolean")
        if not isinstance(booking, Booking):
            raise TypeError("native cancelled itinerary required")
        passenger = self._passengers.get(booking.record_id)
        if passenger is None or passenger.original != booking:
            raise ValueError("itinerary is not an original native cancellation")
        proof = PresentedItinerary(booking.record_id, self.world.t, airport_counter)
        if proof not in self.presented_itineraries:
            self.presented_itineraries += (proof,)
        self._presented.add(booking.record_id)
        if airport_counter:
            self._counter_visits.add((booking.record_id, self.world.t))
        return proof

    def _rescue_prices(self, root, passenger, prices):
        t = self.world.t
        menu = self.rescue_menu
        market = int(self.compiled.product_market[passenger.original.product])
        market_id = self.compiled.spec.markets[market].id
        proved = root in self._presented
        counter = (root, t) in self._counter_visits
        fare = (
            menu.fare_for(market_id, t)
            if proved and (not menu.counter_only or counter)
            else None
        )
        effective = prices
        if fare is not None:
            effective = prices.copy()
            for j in self.compiled.market_products[market]:
                # Posted prices already include active caps. A private rescue
                # offer cannot raise a cheaper regular fare or exceed a cap.
                effective[j] = float(min(Decimal(str(prices[j])), fare))
        return effective, {
            "rescue_eligible": fare is not None,
            "rescue_option_fare": None if fare is None else float(fare),
            "cancelled_itinerary_presented": proved,
            "airport_counter_proved_this_period": counter,
            "rescue_design": menu.design,
        }

    def _reenter(self, prices):
        if self.rescue_menu is None or self.rebooking is None:
            return super()._reenter(prices)
        c, t = self.compiled, self.world.t
        purchases = []
        ordered = sorted(
            self._passengers.items(),
            key=lambda pair: (
                pair[1].first_stranded_at,
                pair[1].original.period,
                pair[1].original.arrival_rank,
            ),
        )
        for root, passenger in ordered:
            if (
                passenger.active_record is not None
                or t >= passenger.deadline
                or (root, t) in self._attempts
            ):
                continue
            self._attempts.add((root, t))
            b = passenger.original
            market = int(c.product_market[b.product])
            private_prices, rescue = self._rescue_prices(root, passenger, prices)
            taste, gumbels = self._marks(passenger, t)
            columns = (*c.market_products[market], len(c.spec.products))
            stages = [b.carrier] + ([None] if self.rebooking.interline else [])
            chosen, stage = None, None
            for carrier in stages:
                active = self._sellable_global().copy()
                for j in c.market_products[market]:
                    active[j] &= (
                        int(c.product_carrier[j]) == carrier
                        if carrier is not None
                        else int(c.product_carrier[j]) != b.carrier
                    )
                utilities = systematic_utilities(
                    c, market, taste, private_prices, active
                )
                selected = sampled_choice(utilities, gumbels[list(columns)])
                if selected < len(columns) - 1:
                    chosen = columns[selected]
                    stage = "own" if carrier is not None else "interline"
                    break
            record = {
                "period": t,
                "original_record": root,
                "original_request": (b.period, b.arrival_rank),
                "market": market,
                "deadline": passenger.deadline,
                "chosen": chosen,
                "stage": stage,
                "alpha": taste.alpha,
                "replacement_record": None,
                "fare": None,
                "rescue_discount_used": False,
                **rescue,
            }
            if chosen is not None:
                request = Request(-1 - root, market, b.segment, b.taste_class, -1, -1)
                new = book(
                    self.world,
                    int(c.product_carrier[chosen]),
                    chosen,
                    float(private_prices[chosen]),
                    request,
                    c,
                )
                self._replacement_roots[new.record_id] = root
                self._passengers[root] = replace(passenger, active_record=new.record_id)
                record.update(
                    replacement_record=new.record_id,
                    fare=new.fare,
                    rescue_discount_used=bool(private_prices[chosen] < prices[chosen]),
                )
                purchases.append(new)
            self.rebooking_history.append(record)
        return tuple(purchases)

    def snapshot(self):
        return RescueSnapshot(super().snapshot(), self.presented_itineraries)

    def restore(self, snapshot):
        if not isinstance(snapshot, RescueSnapshot):
            raise TypeError("rescue snapshot required")
        super().restore(snapshot.native)
        self.presented_itineraries = snapshot.proofs
        self._presented = {p.original_record for p in snapshot.proofs}
        self._counter_visits = {
            (p.original_record, p.period) for p in snapshot.proofs if p.airport_counter
        }

    def branch(self):
        other = RescueMarketEngine(
            self.compiled,
            {i: copy.deepcopy(ctrl) for i, ctrl in self.controllers.items()},
            rescue_menu=self.rescue_menu,
            rebooking=self.rebooking,
            exits=self.exits,
            cap_policy=self.cap_policy,
            check_invariants=self.check_invariants,
        )
        other.restore(self.snapshot())
        return other
