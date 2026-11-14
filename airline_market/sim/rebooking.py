"""A.5/B0/V9: finite-window high-WTP re-entry of stranded native passengers.

Opt-in physical execution only. Rebooking periods are native clock periods, not
hours or legal dated itineraries. No belief/planner kernel is changed here.
"""

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PeriodPhase
from airline_market.sim.arrivals import Request, period_requests
from airline_market.sim.background import step as background_step
from airline_market.sim.capacity import apply_events
from airline_market.sim.choice import (
    sampled_choice,
    sampled_taste,
    systematic_utilities,
)
from airline_market.sim.disruption import DisruptedMarketEngine
from airline_market.sim.engine import StepResult
from airline_market.sim.exogenous import advance
from airline_market.sim.invariants import check_world
from airline_market.sim.ledger import book, inventory
from airline_market.sim.state import Booking


@dataclass(frozen=True)
class RebookingPolicy:
    """Explicit synthetic re-entry assumptions; no empirical parameters adopted."""

    valid_periods: int
    wtp_multiplier: float
    utility_bonus: float = 0.0
    interline: bool = False

    def __post_init__(self):
        if type(self.valid_periods) is not int or self.valid_periods < 1:
            raise ValueError("positive integer rebooking window required")
        if not math.isfinite(self.wtp_multiplier) or self.wtp_multiplier <= 1:
            raise ValueError("high-WTP multiplier must exceed one")
        if not math.isfinite(self.utility_bonus) or self.utility_bonus < 0:
            raise ValueError("nonnegative finite rebooking utility bonus required")
        if type(self.interline) is not bool:
            raise TypeError("interline must be an explicit boolean")


@dataclass(frozen=True)
class RebookingPassenger:
    original: Booking
    first_stranded_at: int
    deadline: int
    active_record: int | None = None


@dataclass(frozen=True)
class RebookingSnapshot:
    native: object
    passengers: tuple[tuple[int, RebookingPassenger], ...]
    replacement_roots: tuple[tuple[int, int], ...]
    attempts: frozenset[tuple[int, int]]
    history: tuple


class RebookingMarketEngine(DisruptedMarketEngine):
    """Post once; serve queued stranded requests before ordinary native arrivals.

    Own-carrier offers have priority. If none is selected, optional interline
    shopping considers other carriers in the SAME compiled O&D. A passenger may
    book at most one active replacement, and repeated cancellation never extends
    their original rebooking deadline. Controller inputs are unchanged.
    """

    def __init__(self, compiled, controllers=None, *, rebooking=None, **kwargs):
        if rebooking is not None and not isinstance(rebooking, RebookingPolicy):
            raise TypeError("explicit RebookingPolicy required")
        super().__init__(compiled, controllers, **kwargs)
        self.rebooking = rebooking
        self._passengers = {}
        self._replacement_roots = {}
        self._attempts = set()
        self.rebooking_history = []

    def reset(self, seed=None, *, tape=None):
        self._passengers = {}
        self._replacement_roots = {}
        self._attempts = set()
        self.rebooking_history = []
        super().reset(seed, tape=tape)

    def _record_stranded(self, bookings, period, cause):
        bookings = tuple(bookings)
        super()._record_stranded(bookings, period, cause)
        if self.rebooking is None:
            return
        for b in bookings:
            root = self._replacement_roots.get(b.record_id, b.record_id)
            old = self._passengers.get(root)
            if old is None:
                if b.segment < 0 or b.taste_class < 0 or b.arrival_rank < 0:
                    raise ValueError(
                        "re-entry requires original native passenger marks"
                    )
                self._passengers[root] = RebookingPassenger(
                    b,
                    period,
                    min(
                        self.compiled.spec.horizon.n_periods,
                        period + self.rebooking.valid_periods,
                    ),
                )
            else:
                self._passengers[root] = replace(old, active_record=None)

    def _marks(self, passenger, period):
        """CRN keys use original potential-request coordinates, never ticket IDs."""
        b = passenger.original
        registry = StreamRegistry(self.tape.seed, "rebooking")
        coordinates = (
            b.period,
            b.arrival_rank,
            int(self.compiled.product_market[b.product]),
        )
        draws = registry.generator(1, *coordinates).normal(
            size=len(self.compiled.spec.carriers) + 1
        )
        taste = sampled_taste(self.compiled, b.segment, b.taste_class, draws)
        taste = replace(
            taste,
            alpha=taste.alpha / self.rebooking.wtp_multiplier,
            market_value={
                k: v + self.rebooking.utility_bonus
                for k, v in {"*": 0.0, **taste.market_value}.items()
            },
        )
        gumbels = registry.generator(2, *coordinates, period).gumbel(
            size=len(self.compiled.spec.products) + 1
        )
        return taste, gumbels

    def _reenter(self, prices):
        if self.rebooking is None:
            return ()
        c, t = self.compiled, self.world.t
        purchases = []
        # Stable original physical-request order, independent of new ticket IDs.
        ordered = sorted(
            self._passengers.items(),
            key=lambda pair: (
                pair[1].first_stranded_at,
                pair[1].original.period,
                pair[1].original.arrival_rank,
            ),
        )
        for root, passenger in ordered:
            if passenger.active_record is not None or t >= passenger.deadline:
                continue
            if (root, t) in self._attempts:
                continue
            self._attempts.add((root, t))
            b = passenger.original
            market = int(c.product_market[b.product])
            taste, gumbels = self._marks(passenger, t)
            columns = (*c.market_products[market], len(c.spec.products))
            stages = [b.carrier]
            if self.rebooking.interline:
                stages.append(None)
            chosen, stage = None, None
            for carrier in stages:
                active = self._sellable_global().copy()
                for j in c.market_products[market]:
                    active[j] &= (
                        int(c.product_carrier[j]) == carrier
                        if carrier is not None
                        else int(c.product_carrier[j]) != b.carrier
                    )
                utilities = systematic_utilities(c, market, taste, prices, active)
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
            }
            if chosen is not None:
                request = Request(-1 - root, market, b.segment, b.taste_class, -1, -1)
                new = book(
                    self.world,
                    int(c.product_carrier[chosen]),
                    chosen,
                    float(prices[chosen]),
                    request,
                    c,
                )
                self._replacement_roots[new.record_id] = root
                self._passengers[root] = replace(passenger, active_record=new.record_id)
                record.update(replacement_record=new.record_id, fare=new.fare)
                purchases.append(new)
            self.rebooking_history.append(record)
            # Re-entry audit is separate from ordinary-demand CS: repeated search
            # opportunities are not independent extra welfare passengers.
        return tuple(purchases)

    def conservation(self):
        if self.rebooking is None:
            return super().conservation()
        active = [
            self._replacement_roots.get(b.record_id, b.record_id)
            for cs in self.world.carriers
            for b in cs.bookings
        ]
        pending = [
            root
            for root, p in self._passengers.items()
            if p.active_record is None and self.world.t < p.deadline
        ]
        unserved = [
            root
            for root, p in self._passengers.items()
            if p.active_record is None and self.world.t >= p.deadline
        ]
        return {
            "original_booked_passengers": self.world.next_record_id
            - len(self._replacement_roots),
            "active_passengers": len(active),
            "pending_stranded": len(pending),
            "unserved_stranded": len(unserved),
            "active_plus_pending_plus_unserved": len(active)
            + len(pending)
            + len(unserved),
            "unique_active_passengers": len(set(active)),
            "replacement_tickets_ever": len(self._replacement_roots),
            "active_rebooked_passengers": sum(
                p.active_record is not None for p in self._passengers.values()
            ),
            "native_ticket_count": self.world.next_record_id,
            "clock_unit": "native periods",
        }

    def snapshot(self):
        return RebookingSnapshot(
            super().snapshot(),
            tuple(self._passengers.items()),
            tuple(self._replacement_roots.items()),
            frozenset(self._attempts),
            tuple(copy.deepcopy(self.rebooking_history)),
        )

    def restore(self, snapshot):
        if not isinstance(snapshot, RebookingSnapshot):
            raise TypeError("rebooking snapshot required")
        super().restore(snapshot.native)
        self._passengers = dict(snapshot.passengers)
        self._replacement_roots = dict(snapshot.replacement_roots)
        self._attempts = set(snapshot.attempts)
        self.rebooking_history = list(copy.deepcopy(snapshot.history))

    def branch(self):
        other = RebookingMarketEngine(
            self.compiled,
            {i: copy.deepcopy(ctrl) for i, ctrl in self.controllers.items()},
            rebooking=self.rebooking,
            exits=self.exits,
            cap_policy=self.cap_policy,
            check_invariants=self.check_invariants,
        )
        other.restore(self.snapshot())
        return other

    # Frozen DisruptedMarketEngine.step with re-entry inserted after posting prices.

    def step(
        self, actions: Mapping | None = None, *, _requests=None, _choices=None
    ) -> StepResult:
        """
        Advance one period using the supplied action and declared event timing (Eqs.
        3–16).
        """
        if self.rebooking is None:
            return super().step(actions, _requests=_requests, _choices=_choices)
        c = self.compiled
        world = self.world
        if world is None:
            raise RuntimeError("reset first")
        t = world.t
        self._open_period()
        before = (
            list(self._opening_cash)
            if self._pending_events
            else [cs.cash for cs in world.carriers]
        )
        before_bookings = tuple(b for cs in world.carriers for b in cs.bookings)
        prices = self.post_actions(actions)
        purchases = list(self._reenter(prices))
        requests = (
            period_requests(c, self.tape, t, world.exo)
            if _requests is None
            else _requests
        )
        for index, request in enumerate(requests):
            active = self._sellable_global()
            taste = sampled_taste(
                c,
                request.segment,
                request.taste_class,
                self.tape.mixed_draws[t, request.mixed_row],
            )
            utilities = systematic_utilities(c, request.market, taste, prices, active)
            columns = (*c.market_products[request.market], len(c.spec.products))
            gumbels = self.tape.gumbel[t, request.gumbel_row, list(columns)]
            chosen_local = (
                sampled_choice(utilities, gumbels)
                if _choices is None
                else (
                    len(columns) - 1
                    if _choices[index] is None
                    else columns.index(_choices[index])
                )
            )
            chosen = None
            if chosen_local < len(columns) - 1:
                chosen = columns[chosen_local]
                i = int(c.product_carrier[chosen])
                purchases.append(
                    book(world, i, chosen, float(prices[chosen]), request, c)
                )
            max_u = float(np.max(utilities))
            logsum = max_u + float(np.log(np.exp(utilities - max_u).sum()))
            world.request_log.append(
                {
                    "period": t,
                    "market": request.market,
                    "segment": request.segment,
                    "chosen": chosen,
                    "unavailable": not all(
                        active[list(c.market_products[request.market])]
                    ),
                    "consumer_surplus": logsum / taste.alpha,
                    "realized_surplus": float(
                        utilities[chosen_local] + gumbels[chosen_local] - gumbels[-1]
                    )
                    / taste.alpha,
                }
            )
            for i, cs in enumerate(world.carriers):
                for l, x in zip(c.legs_of[i], inventory(cs, c, i)):
                    if x == 0:
                        world.sellout_period.setdefault(l, t)
        for index, (bg, bs) in enumerate(zip(world.background, c.spec.background)):
            background_step(bg, bs, float(self.tape.background_uniforms[t, index]))
        world.last_purchases = tuple(purchases)
        world.posted_leaders.clear()
        for cs in world.carriers:
            cs.previous_prices = cs.posted_prices.copy()
        old = world.exo
        world.t += 1
        world.exo = advance(c, old, t, float(self.tape.exo_uniform[t]))
        resolutions = apply_events(world, c, old, world.exo, t, self.tape)
        self._capture_native_cancellations((*before_bookings, *purchases))
        resolutions[:0] = [event.record() for event in self._pending_events]
        self._pending_events = ()
        resolutions.extend(event.record() for event in self._enforce_exits(world.t))
        world.phase = (
            PeriodPhase.TERMINAL if self.terminal() else PeriodPhase.PRE_DECISION
        )
        for i, ctrl in self.controllers.items():
            ctrl.observe_outcome(
                {
                    "purchases": tuple(b for b in purchases if b.carrier == i),
                    "cash_change": world.carriers[i].cash - before[i],
                }
            )
            world.carriers[i].controller_memory = ctrl.memory()
        if self.check_invariants:
            check_world(world, c)
        self._trace.append(
            {
                "period": t,
                "prices": prices.tolist(),
                "purchases": tuple(purchases),
                "events": resolutions,
            }
        )
        self._opening_period = None
        self._opening_cash = ()
        return StepResult(
            t,
            tuple(purchases),
            tuple(resolutions),
            self.terminal(),
            tuple(cs.cash - x for cs, x in zip(world.carriers, before)),
        )
