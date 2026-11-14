"""Generic departure-cohort sampler with explicit event timing (model §6)."""

from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np
from ..core.types import CoefficientAction, PriceAction, HoldAction, PeriodPhase
from ..core.rng import StreamRegistry
from ..agents import make_controller
from ..agents.base import ControllerContext
from ..agents.external import ExternalController
from ..agents.price_rule import AdditivePriceRule
from ..spec.model import InformationArmSpec
from ..pomdp.observation import ObservationBuilder
from .state import WorldState, CarrierState, BackgroundState
from .tape import ExogenousTape
from .exogenous import categorical, advance
from .ledger import inventory, sellable, book
from .quotes import QuoteHistory, posted_quotes
from .arrivals import period_requests
from .choice import systematic_utilities, sampled_choice, sampled_taste
from .capacity import apply_events
from .background import step as background_step
from .accounting import metrics
from .invariants import check_world


@dataclass(frozen=True)
class EngineSnapshot:
    """
    Replayable mutable-world/controller state plus immutable exogenous tape (Eqs. 3–16).
    """

    world: WorldState
    controllers: tuple
    tape: ExogenousTape


@dataclass(frozen=True)
class StepResult:
    """
    Purchases, resolutions, cash changes and termination for one booking period (Eqs.
    3–16).
    """

    period: int
    purchases: tuple
    resolutions: tuple
    terminal: bool
    cash_changes: tuple[float, ...]


class MarketEngine:
    """
    Generic shared-leg cohort simulator implementing the complete event sequence (Eqs.
    3–16).
    """

    def __init__(
        self,
        compiled,
        controllers: Mapping | None = None,
        *,
        check_invariants: bool = True,
    ):
        """Bind declared inputs and initialize MarketEngine bookkeeping (Eqs. 3–16)."""
        self.compiled = compiled
        self.check_invariants = check_invariants
        self.assignments = {
            compiled.carrier_index[a.carrier]: a for a in compiled.spec.controllers
        }
        self.controllers = {i: make_controller(a) for i, a in self.assignments.items()}
        for key, value in (controllers or {}).items():
            self.controllers[
                compiled.carrier_index[key] if isinstance(key, str) else key
            ] = value
        self.pricing = {
            compiled.carrier_index[p.carrier]: p for p in compiled.spec.pricing
        }
        self.rules = {
            i: AdditivePriceRule(compiled, p) for i, p in self.pricing.items()
        }
        self.world = None
        self.tape = None
        self._trace = []
        self._seed = 0

    def reset(self, seed: int | None = None, *, tape: ExogenousTape | None = None):
        """Start a new cohort and clear per-episode memory (Eqs. 3–16)."""
        c = self.compiled
        s = c.spec
        self._seed = int(seed or 0)
        self.tape = tape or ExogenousTape.generate(c, self._seed)
        K = [
            l.capacity.support[categorical(float(u), l.capacity.probabilities)]
            for l, u in zip(s.legs, self.tape.initial_capacity_uniform)
        ]
        for prior in s.capacity_priors:
            u = self.tape.initial_capacity_uniform[c.leg_index[prior.legs[0]]]
            vector = prior.support[categorical(float(u), prior.probabilities)]
            for leg, k in zip(prior.legs, vector):
                K[c.leg_index[leg]] = k
        carriers = []
        for i in range(len(s.carriers)):
            prices = np.array(
                [
                    s.timing.initial_previous_prices.get(
                        s.products[j].id, s.products[j].reference_fare
                    )
                    for j in c.products_of[i]
                ],
                dtype=np.float64,
            )
            carriers.append(
                CarrierState(
                    np.array([K[l] for l in c.legs_of[i]], dtype=np.int64),
                    [],
                    0.0,
                    0.0,
                    0.0,
                    0,
                    prices.copy(),
                    prices.copy(),
                )
            )
        world = WorldState(
            0,
            PeriodPhase.PRE_DECISION,
            categorical(self.tape.initial_exo_uniform, s.exogenous.initial),
            carriers,
            [BackgroundState(int(k)) for k in self.tape.background_initial],
            QuoteHistory(),
            initial_capacity=tuple(K),
        )
        world.quote_history.initial = posted_quotes(world, c)
        self.world = world
        self._trace = []
        for i, ctrl in self.controllers.items():
            objective = next(o for o in s.objectives if o.carrier == s.carriers[i].code)
            arm = self.controller_arm(i)
            ctrl.reset(
                ControllerContext(c, i, self.pricing.get(i), objective, arm),
                StreamRegistry(self._seed, "ctrl/" + s.carriers[i].code).generator(),
            )

    def controller_arm(self, i: int):
        """Select the controller's declared shopping-information scope (Eqs. 3–16)."""
        c = self.compiled
        code = c.spec.carriers[i].code
        requested = getattr(self.controllers[i], "arm_name", None) or (
            self.assignments[i].params.get("arm") if i in self.assignments else None
        )
        arms = [a for a in c.spec.information if a.observer == code]
        return next(
            (a for a in arms if a.name == requested),
            arms[0]
            if arms
            else InformationArmSpec(name="own", observer=code, requests=()),
        )

    def _action_prices(self, i: int, action, obs):
        """
        Apply coefficient/direct actions and preserve controller/update memory (Eqs.
        3–16).
        """
        c = self.compiled
        state = self.world.carriers[i]
        if isinstance(action, HoldAction):
            if state.theta is None:
                raise ValueError("HoldAction requires a previous coefficient action")
            action = CoefficientAction(state.theta)
        if not isinstance(action, (CoefficientAction, PriceAction)):
            action = CoefficientAction(tuple(map(float, action)))
        if isinstance(action, CoefficientAction):
            if state.theta is not None and self.world.t % self.pricing[i].update_every:
                action = CoefficientAction(state.theta)
            state.theta = action.theta
            prices = self.rules[i].prices(action.theta, obs)
        else:
            prices = np.array(
                [action.prices[j] for j in c.products_of[i]], dtype=np.float64
            )
            if not np.isfinite(prices).all():
                raise ValueError("nonfinite direct prices")
        state.posted_prices = prices
        state.controller_memory = self.controllers[i].memory()

    def begin_period(self, actions: Mapping | None = None):
        """
        Post leaders before followers under the declared timing protocol (Eqs. 3–16).
        """
        if self.world is None:
            raise RuntimeError("reset first")
        if self.terminal() or self.world.phase != PeriodPhase.PRE_DECISION:
            return
        c = self.compiled
        world = self.world
        actions = self._normalize_actions(actions)
        world.quote_history.records[world.t] = posted_quotes(world, c)
        if c.spec.timing.mode == "leader_follower":
            for code in c.spec.timing.leaders:
                i = c.carrier_index[code]
                ctrl = self.controllers[i]
                if i in world.posted_leaders:
                    continue
                if isinstance(ctrl, ExternalController) and i not in actions:
                    continue
                obs = ObservationBuilder(c, self.controller_arm(i)).build(
                    world, world.quote_history, self.tape
                )
                self._action_prices(
                    i, actions[i] if i in actions else ctrl.act(obs), obs
                )
                world.posted_leaders.add(i)
                world.quote_history.records[world.t] = posted_quotes(world, c)
            if len(world.posted_leaders) == len(c.spec.timing.leaders):
                world.phase = PeriodPhase.LEADERS_POSTED
        else:
            world.phase = PeriodPhase.POSTED

    def observe(self, carrier, arm):
        """
        Construct the permitted own-ledger and shopping observation after leader posting
        (Eqs. 3–16).
        """
        if self.world is None:
            raise RuntimeError("reset first")
        self.begin_period()
        if self.terminal() and self.world.t not in self.world.quote_history.records:
            # Terminal shopping is a diagnostic of current offers, with no new
            # passenger/pricing decision by external players.
            self.world.quote_history.records[self.world.t] = posted_quotes(
                self.world, self.compiled
            )
            for code in self.compiled.spec.timing.leaders:
                k = self.compiled.carrier_index[code]
                ctrl = self.controllers[k]
                if not isinstance(ctrl, ExternalController):
                    obs = ObservationBuilder(
                        self.compiled, self.controller_arm(k)
                    ).build(self.world, self.world.quote_history, self.tape)
                    self._action_prices(k, ctrl.act(obs), obs)
            self.world.quote_history.records[self.world.t] = posted_quotes(
                self.world, self.compiled
            )
        i = (
            self.compiled.carrier_index[carrier]
            if isinstance(carrier, str)
            else carrier
        )
        if isinstance(arm, str):
            arm = next(
                a
                for a in self.compiled.spec.information
                if a.name == arm and a.observer == self.compiled.spec.carriers[i].code
            )
        return ObservationBuilder(self.compiled, arm).build(
            self.world, self.world.quote_history, self.tape
        )

    def _normalize_actions(self, actions):
        """Translate carrier names to stable compiled indices (Eqs. 3–16)."""
        return {
            self.compiled.carrier_index[k] if isinstance(k, str) else k: v
            for k, v in (actions or {}).items()
        }

    def _sellable_global(self):
        """
        Return capacity-feasible product availability in global product order (Eqs.
        3–16).
        """
        c = self.compiled
        active = np.zeros(len(c.spec.products), dtype=bool)
        for i, cs in enumerate(self.world.carriers):
            active[list(c.products_of[i])] = sellable(inventory(cs, c, i), c.A[i])
        return active

    def _prices_global(self):
        """Return posted fares in global product order (Eqs. 3–16)."""
        c = self.compiled
        prices = np.zeros(len(c.spec.products))
        for i, cs in enumerate(self.world.carriers):
            prices[list(c.products_of[i])] = cs.posted_prices
        return prices

    def post_actions(self, actions: Mapping | None = None):
        """
        Post all supplied/fixed-controller actions before customer choice (Eqs. 3–16).
        """
        if self.world is None:
            raise RuntimeError("reset first")
        if self.terminal():
            raise ValueError("cannot step terminal cohort")
        actions = self._normalize_actions(actions)
        self.begin_period(actions)
        c = self.compiled
        world = self.world
        leaders = (
            {c.carrier_index[x] for x in c.spec.timing.leaders}
            if c.spec.timing.mode == "leader_follower"
            else set()
        )
        for i, ctrl in self.controllers.items():
            if i in leaders:
                if i not in world.posted_leaders:
                    raise KeyError(f"missing external leader action for {i}")
                continue
            obs = ObservationBuilder(c, self.controller_arm(i)).build(
                world, world.quote_history, self.tape
            )
            if isinstance(ctrl, ExternalController) and i not in actions:
                raise KeyError(f"missing external action for {i}")
            self._action_prices(i, actions[i] if i in actions else ctrl.act(obs), obs)
        world.quote_history.records[world.t] = posted_quotes(world, c)
        return self._prices_global()

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
        before = [cs.cash for cs in world.carriers]
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
        return StepResult(
            t,
            tuple(purchases),
            tuple(resolutions),
            self.terminal(),
            tuple(cs.cash - x for cs, x in zip(world.carriers, before)),
        )

    def terminal(self):
        """Test whether the cohort booking horizon has ended (Eqs. 3–16)."""
        return (
            self.world is not None
            and self.world.t >= self.compiled.spec.horizon.n_periods
        )

    def metrics(self):
        """Return analyst-only cohort accounting and welfare summaries (Eqs. 3–16)."""
        return metrics(self.world, self.compiled)

    def trace(self):
        """
        Return recorded period prices, purchases and resolution events (Eqs. 3–16).
        """
        return list(self._trace)

    def snapshot(self):
        """
        Copy mutable execution/filter state for reproducible branching (Eqs. 3–16).
        """
        return EngineSnapshot(
            self.world.clone(),
            tuple(ctrl.snapshot() for ctrl in self.controllers.values()),
            self.tape,
        )

    def restore(self, snapshot):
        """Restore a previously captured execution/filter snapshot (Eqs. 3–16)."""
        self.world = snapshot.world.clone()
        self.tape = snapshot.tape
        for ctrl, state in zip(self.controllers.values(), snapshot.controllers):
            ctrl.restore(state)

    def branch(self):
        """
        Clone mutable state while sharing immutable model data and random tapes (Eqs.
        3–16).
        """
        other = MarketEngine(
            self.compiled,
            {i: copy.deepcopy(ctrl) for i, ctrl in self.controllers.items()},
            check_invariants=self.check_invariants,
        )
        other.restore(self.snapshot())
        return other
