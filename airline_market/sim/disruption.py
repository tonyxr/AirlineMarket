"""A.5/B0/V9: permanent rival exit with native refunds and a stranded ledger.

The native step is preserved below with explicit boundary hooks. This opt-in
executor is not yet an exit-aware belief kernel or rebooking-demand model.
All active bookings represent not-yet-departed native cohort itineraries.
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from airline_market.core.types import PeriodPhase
from airline_market.policy.regulator import NO_CAP, RegulatedMarketEngine
from airline_market.sim.arrivals import period_requests
from airline_market.sim.background import step as background_step
from airline_market.sim.capacity import apply_events
from airline_market.sim.choice import (
    sampled_choice,
    sampled_taste,
    systematic_utilities,
)
from airline_market.sim.engine import EngineSnapshot, StepResult
from airline_market.sim.exogenous import advance
from airline_market.sim.invariants import check_world
from airline_market.sim.ledger import book, inventory, resolve
from airline_market.sim.state import Booking


@dataclass(frozen=True)
class RivalExit:
    """Remove all target capacity from this native-period boundary onward."""

    carrier: str
    reveal_period: int

    def __post_init__(self):
        if (
            not self.carrier
            or type(self.reveal_period) is not int
            or self.reveal_period < 0
        ):
            raise ValueError(
                "exit requires a carrier and nonnegative integer reveal period"
            )


@dataclass(frozen=True)
class StrandedPassenger:
    """One original ticket, refunded and retained for future rebooking demand."""

    booking: Booking
    canceled_at: int
    cause: str


@dataclass(frozen=True)
class ExitEvent:
    plan: int
    carrier: int
    reveal_period: int
    period: int
    removed_capacity: tuple[int, ...]
    canceled_ids: tuple[int, ...]
    refunds: float
    penalties: float
    recovery_blocked: bool

    def record(self):
        return {"disruption": "rival_exit", **vars(self)}


@dataclass(frozen=True)
class DisruptionSnapshot:
    native: EngineSnapshot
    stranded: tuple[StrandedPassenger, ...]
    activated: frozenset[int]
    opening_period: int | None
    opening_cash: tuple[float, ...]
    pending_events: tuple[ExitEvent, ...]


class DisruptedMarketEngine(RegulatedMarketEngine):
    """Opt-in physical exit executor; actors retain native permitted observations."""

    def __init__(
        self,
        compiled,
        controllers=None,
        *,
        exits=(),
        cap_policy=NO_CAP,
        check_invariants=True,
    ):
        super().__init__(
            compiled,
            controllers,
            cap_policy=cap_policy,
            check_invariants=check_invariants,
        )
        self.exits = tuple(exits)
        for plan in self.exits:
            if not isinstance(plan, RivalExit):
                raise TypeError("explicit RivalExit records required")
            if (
                plan.carrier not in compiled.carrier_index
                or plan.reveal_period >= compiled.spec.horizon.n_periods
            ):
                raise ValueError("exit carrier or reveal period is outside the model")
        self.stranded = ()
        self._stranded_ids = set()
        self._activated = set()
        self._opening_period = None
        self._opening_cash = ()
        self._pending_events = ()

    def _open_period(self):
        if self._opening_period != self.world.t:
            self._opening_period = self.world.t
            self._opening_cash = tuple(cs.cash for cs in self.world.carriers)

    def _record_stranded(self, bookings, period, cause):
        for booking in bookings:
            if booking.record_id not in self._stranded_ids:
                self.stranded += (StrandedPassenger(booking, period, cause),)
                self._stranded_ids.add(booking.record_id)

    def _enforce_exits(self, period):
        events = []
        for k, plan in enumerate(self.exits):
            if period < plan.reveal_period:
                continue
            i = self.compiled.carrier_index[plan.carrier]
            state = self.world.carriers[i]
            first = k not in self._activated
            if not first and not np.any(state.capacity) and not state.bookings:
                continue
            capacity = tuple(map(int, state.capacity))
            state.capacity[:] = 0
            outcome = resolve(self.world, i, self.compiled)
            self._record_stranded(outcome.canceled, period, "rival_exit")
            events.append(
                ExitEvent(
                    k,
                    i,
                    plan.reveal_period,
                    period,
                    capacity,
                    tuple(b.record_id for b in outcome.canceled),
                    outcome.refund,
                    outcome.penalty,
                    not first,
                )
            )
            self._activated.add(k)
        return tuple(events)

    def reset(self, seed=None, *, tape=None):
        super().reset(seed, tape=tape)
        self.stranded = ()
        self._stranded_ids = set()
        self._activated = set()
        self._opening_period = None
        self._opening_cash = ()
        self._pending_events = ()
        if any(p.reveal_period == 0 for p in self.exits):
            self._open_period()
            self._pending_events += self._enforce_exits(0)

    def begin_period(self, actions=None):
        if self.world is not None and not self.terminal():
            self._open_period()
            self._pending_events += self._enforce_exits(self.world.t)
        return super().begin_period(actions)

    def _capture_native_cancellations(self, candidates):
        active = {b.record_id for cs in self.world.carriers for b in cs.bookings}
        self._record_stranded(
            (b for b in candidates if b.record_id not in active),
            self.world.t,
            "native_capacity_event",
        )

    def conservation(self):
        """Analyst-only counts; pending stranded passengers have not re-entered."""
        active = sum(len(cs.bookings) for cs in self.world.carriers)
        return {
            "active_tickets": active,
            "pending_stranded": len(self.stranded),
            "ever_booked_tickets": self.world.next_record_id,
            "active_plus_stranded": active + len(self.stranded),
            "rebooked_passengers": None,
            "reentry_status": "not implemented",
        }

    def snapshot(self):
        return DisruptionSnapshot(
            super().snapshot(),
            self.stranded,
            frozenset(self._activated),
            self._opening_period,
            self._opening_cash,
            self._pending_events,
        )

    def restore(self, snapshot):
        if not isinstance(snapshot, DisruptionSnapshot):
            raise TypeError(
                "disruption snapshot required; do not discard the stranded ledger"
            )
        super().restore(snapshot.native)
        self.stranded = snapshot.stranded
        self._stranded_ids = {s.booking.record_id for s in self.stranded}
        self._activated = set(snapshot.activated)
        self._opening_period = snapshot.opening_period
        self._opening_cash = snapshot.opening_cash
        self._pending_events = snapshot.pending_events

    def branch(self):
        other = DisruptedMarketEngine(
            self.compiled,
            {i: copy.deepcopy(ctrl) for i, ctrl in self.controllers.items()},
            exits=self.exits,
            cap_policy=self.cap_policy,
            check_invariants=self.check_invariants,
        )
        other.restore(self.snapshot())
        return other

    # Native MarketEngine.step, with the boundary hooks recorded in the source manifest.

    def step(
        self, actions: Mapping | None = None, *, _requests=None, _choices=None
    ) -> StepResult:
        """
        Advance one period using the supplied action and declared event timing (Eqs.
        3–16).
        """
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
        purchases = []
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
