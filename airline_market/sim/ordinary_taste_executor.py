"""A.10/A.5/A.14/B0/V9: opt-in real ordinary executor with taste-return hooks.

Frozen bodies are unchanged. Remove hooks/name/delegation adaptations to recover
their ASTs. This executor runs the actual step, not an observer replay.
"""

from collections.abc import Mapping

import numpy as np

from airline_market.core.types import PeriodPhase
from airline_market.sim.arrivals import period_requests
from airline_market.sim.background import step as background_step
from airline_market.sim.capacity import apply_events
from airline_market.sim.choice import (
    sampled_choice,
    sampled_taste,
    systematic_utilities,
)
from airline_market.sim.engine import StepResult
from airline_market.sim.exogenous import advance
from airline_market.sim.invariants import check_world
from airline_market.sim.ledger import book, inventory
from airline_market.sim.rebooking import RebookingMarketEngine


class _OrdinaryTasteExecutor(RebookingMarketEngine):
    def step(
        self, actions: Mapping | None = None, *, _requests=None, _choices=None
    ) -> StepResult:
        """
        Advance one period using the supplied action and declared event timing (Eqs.
        3–16).
        """
        if self.rebooking is None:
            return self._ordinary_disrupted_step(
                actions, _requests=_requests, _choices=_choices
            )
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
            self._capture_ordinary_taste(request, t, taste, "rebooking")
            utilities = systematic_utilities(c, request.market, taste, prices, active)
            columns = (*c.market_products[request.market], len(c.spec.products))
            gumbels = self.tape.gumbel[t, request.gumbel_row, list(columns)]
            chosen_local = (
                sampled_choice(utilities, gumbels)
                if _choices is None
                else len(columns) - 1
                if _choices[index] is None
                else columns.index(_choices[index])
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
            tuple(cs.cash - x for (cs, x) in zip(world.carriers, before)),
        )

    def _ordinary_disrupted_step(
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
            self._capture_ordinary_taste(request, t, taste, "disrupted")
            utilities = systematic_utilities(c, request.market, taste, prices, active)
            columns = (*c.market_products[request.market], len(c.spec.products))
            gumbels = self.tape.gumbel[t, request.gumbel_row, list(columns)]
            chosen_local = (
                sampled_choice(utilities, gumbels)
                if _choices is None
                else len(columns) - 1
                if _choices[index] is None
                else columns.index(_choices[index])
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
            tuple(cs.cash - x for (cs, x) in zip(world.carriers, before)),
        )
