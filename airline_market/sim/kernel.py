"""Enumerated transitions sharing the sampler's primitives (Eqs. 25–27)."""

from __future__ import annotations
from dataclasses import dataclass, replace
from itertools import product
from collections import defaultdict
import math
import copy
import numpy as np
from scipy.stats import poisson
from scipy.special import ndtr
from ..core.errors import KernelLimitError
from ..core.types import PeriodPhase, CoefficientAction
from ..spec.validate import validate_exact_compatible
from .engine import MarketEngine
from .state import Booking
from .ledger import inventory, sellable, book
from .choice import systematic_utilities, choice_probabilities
from .arrivals import bernoulli_probability, Request
from .tape import ExogenousTape, segment_weights
from .exogenous import transition_row, demand_multiplier
from .background import purchase_probability
from .quotes import posted_quotes


@dataclass(frozen=True, order=True)
class KernelCarrierState:
    """
    Finite capacity/count ledger and controller memory for one carrier (Eqs. 3–16).
    """

    capacity: tuple[int, ...]
    counts: tuple[int, ...]
    fares: tuple[tuple[float, ...], ...] | None
    previous_prices: tuple[float, ...]
    controller_memory: object = ()
    cash: float = 0.0
    refunds: float = 0.0
    penalties: float = 0.0
    canceled: int = 0
    theta: tuple[float, ...] | None = None
    booking_order: tuple[int, ...] = ()
    periods: tuple[tuple[int, ...], ...] = ()
    posted_prices: tuple[float, ...] = ()


@dataclass(frozen=True, order=True)
class KernelState:
    """
    Finite exact-transition state, including declared lagged quote history (Eqs. 3–16).
    """

    t: int
    exo: int
    carriers: tuple[KernelCarrierState, ...]
    background: tuple[int, ...]
    phase: str = "pre_decision"
    history: tuple = ()
    initial_quotes: tuple = ()
    events_fired: tuple[int, ...] = ()
    posted_leaders: tuple[int, ...] = ()


@dataclass(frozen=True)
class PeriodOutcome:
    """Enumerated chronological requests and product-choice outcomes (Eqs. 3–16)."""

    requests: tuple[Request, ...]
    choices: tuple[int | None, ...]


@dataclass(frozen=True)
class PostedPrices:
    """Focal action and all posted product fares for one transition (Eqs. 3–16)."""

    state: KernelState
    focal: int
    action: object
    values: tuple[float, ...]


@dataclass(frozen=True)
class FocalSignal:
    """
    Observable own purchases, cash/capacity changes and next shopping results (Eqs.
    3–16).
    """

    purchases: tuple
    own_capacity: tuple
    public_exo: tuple
    cash_delta: float
    observation: object

    def key(self):
        """
        Return the deterministic memoization key for the represented state (Eqs. 3–16).
        """
        o = self.observation
        return (
            self.purchases,
            self.own_capacity,
            self.public_exo,
            self.cash_delta,
            tuple((b.product, b.fare, b.period) for b in o.own_bookings),
            o.own_previous_prices,
            o.quotes,
        )


class ExactKernel:
    """
    General engine-backed exact reference transition and observation law (Eqs. 3–16).
    """

    def __init__(
        self,
        compiled,
        controllers_spec=None,
        *,
        max_requests: int | None = None,
        max_outcomes: int = 1_000_000,
        prob_tol: float = 1e-12,
        prob_tol_tail: float = 1e-6,
        validate=True,
    ):
        """Bind declared inputs and initialize ExactKernel bookkeeping (Eqs. 3–16)."""
        if validate:
            validate_exact_compatible(compiled.spec)
        if (
            validate
            and controllers_spec
            and any(
                not getattr(ctrl, "exact_compatible", False)
                for ctrl in controllers_spec.values()
            )
        ):
            raise ValueError(
                "exact kernels require exact-compatible rival controllers; episode-mixture types need an explicit joint type belief"
            )
        self.compiled = compiled
        self.controllers_spec = controllers_spec
        self.max_outcomes = max_outcomes
        self.max_requests = (
            max_requests
            if max_requests is not None
            else compiled.spec.demand.poisson_truncation
        )
        self.prob_tol = prob_tol
        self.prob_tol_tail = prob_tol_tail
        self.tail_mass = 0.0
        self.base_tape = ExogenousTape.generate(compiled, 0)
        self.arm = None

    def encode(self, world, focal: int = 0) -> KernelState:
        """
        Project private world state to the finite exact-kernel representation (Eqs.
        3–16).
        """
        c = self.compiled
        carriers = []
        for i, cs in enumerate(world.carriers):
            groups = [
                [b for b in cs.bookings if b.product == j] for j in c.products_of[i]
            ]
            fares = (
                tuple(tuple(b.fare for b in group) for group in groups)
                if i == focal
                else None
            )
            cash = (
                cs.cash
                if i == focal
                else sum(
                    len(group) * c.spec.products[j].reference_fare
                    for j, group in zip(c.products_of[i], groups)
                )
            )
            carriers.append(
                KernelCarrierState(
                    tuple(map(int, cs.capacity)),
                    tuple(map(len, groups)),
                    fares,
                    tuple(map(float, cs.previous_prices)),
                    cs.controller_memory,
                    cash,
                    cs.refunds if i == focal else 0.0,
                    cs.penalties if i == focal else 0.0,
                    cs.canceled if i == focal else 0,
                    cs.theta,
                    tuple(c.products_of[i].index(b.product) for b in cs.bookings)
                    if i == focal
                    else tuple(k for k, group in enumerate(groups) for _ in group),
                    tuple(
                        tuple(b.period if i == focal else -1 for b in group)
                        for group in groups
                    ),
                    tuple(map(float, cs.posted_prices)),
                )
            )
        lag = max([q.lag_periods for q in c.spec.quotes.requests] or [1])
        history = tuple(
            (t, quotes)
            for t, quotes in sorted(world.quote_history.records.items())
            if t >= world.t - lag
        )
        state = KernelState(
            world.t,
            world.exo,
            tuple(carriers),
            tuple(b.stock for b in world.background),
            world.phase.value,
            history,
            world.quote_history.initial,
            tuple(sorted(world.events_fired)),
            tuple(sorted(world.posted_leaders)),
        )
        return self.canonical_state(state, focal)

    def canonical_state(self, state: KernelState, focal: int):
        """
        Discard unused historical fares while retaining all declared lagged inputs (Eqs.
        3–16).
        """
        # Registered controllers use own inventory/counts and declared lagged
        # quotes, not unqueried historical own fares. Keep all required history.
        c = self.compiled
        needed = set()
        for request in c.spec.quotes.requests:
            if request.lag_periods:
                for target in request.target_products:
                    if target in c.product_index:
                        needed.add(c.product_index[target])
                    else:
                        needed.add(
                            len(c.spec.products)
                            + next(
                                k
                                for k, b in enumerate(c.spec.background)
                                if b.id == target
                            )
                        )

        def history_row(row):
            """
            Preserve queried historical quotes and canonicalize unused fields (Eqs.
            3–16).
            """
            return tuple(
                q
                if q.product in needed
                else replace(q, fare=0.0, available=False, seats=0)
                for q in row
            )

        rivals = []
        for i, cs in enumerate(state.carriers):
            if i != focal and state.phase == PeriodPhase.PRE_DECISION.value:
                refs = tuple(
                    c.spec.products[j].reference_fare for j in c.products_of[i]
                )
                cs = replace(cs, previous_prices=refs, posted_prices=refs)
            rivals.append(cs)
        return replace(
            state,
            carriers=tuple(rivals),
            history=tuple((t, history_row(row)) for t, row in state.history),
            initial_quotes=history_row(state.initial_quotes),
        )

    def engine(self, ks: KernelState, focal: int = 0) -> MarketEngine:
        """
        Reconstruct a hypothetical reference engine from a model-generated state (Eqs.
        3–16).
        """
        c = self.compiled
        engine = MarketEngine(
            c, copy.deepcopy(self.controllers_spec), check_invariants=False
        )
        engine.reset(tape=self.base_tape)
        w = engine.world
        w.t = ks.t
        w.exo = ks.exo
        w.phase = PeriodPhase(ks.phase)
        w.events_fired = set(ks.events_fired)
        w.posted_leaders = set(ks.posted_leaders)
        w.quote_history.records = dict(ks.history)
        w.quote_history.initial = ks.initial_quotes or w.quote_history.initial
        next_id = 0
        for i, (source, target) in enumerate(zip(ks.carriers, w.carriers)):
            target.capacity = np.array(source.capacity, dtype=np.int64)
            target.bookings = []
            groups = []
            for local, j in enumerate(c.products_of[i]):
                fares = (
                    source.fares[local]
                    if source.fares is not None
                    else (c.spec.products[j].reference_fare,) * source.counts[local]
                )
                periods = (
                    source.periods[local]
                    if source.periods
                    else (-1,) * source.counts[local]
                )
                groups.append(list(zip(fares, periods)))
            order = source.booking_order or tuple(
                k for k, n in enumerate(source.counts) for _ in range(n)
            )
            for local in order:
                fare, period = groups[local].pop(0)
                target.bookings.append(
                    Booking(next_id, i, c.products_of[i][local], fare, period, 0)
                )
                next_id += 1
            target.cash = source.cash
            target.refunds = source.refunds
            target.penalties = source.penalties
            target.canceled = source.canceled
            target.revenue = source.cash + source.refunds + source.penalties
            target.theta = source.theta
            target.previous_prices = np.array(source.previous_prices)
            target.posted_prices = np.array(
                source.posted_prices or source.previous_prices
            )
            target.controller_memory = source.controller_memory
            engine.controllers[i].restore(source.controller_memory)
        w.next_record_id = next_id
        for b, x in zip(w.background, ks.background):
            b.stock = x
        return engine

    def initial_distribution(self, focal: int = 0):
        """Enumerate initial capacity, background and exogenous priors (Eqs. 3–16)."""
        c = self.compiled
        s = c.spec
        components = []
        covered = set()
        for prior in s.capacity_priors:
            components.append(
                [
                    (p, tuple(zip(prior.legs, vector)))
                    for p, vector in zip(prior.probabilities, prior.support)
                    if p
                ]
            )
            covered.update(prior.legs)
        for leg in s.legs:
            if leg.id not in covered:
                components.append(
                    [
                        (p, ((leg.id, k),))
                        for p, k in zip(
                            leg.capacity.probabilities, leg.capacity.support
                        )
                        if p
                    ]
                )
        capacity_options = []
        for combo in product(*components):
            prob = math.prod(x[0] for x in combo)
            values = {leg: k for _, pairs in combo for leg, k in pairs}
            capacity_options.append((prob, values))
        background = [
            [(p, k) for p, k in zip(b.capacity.probabilities, b.capacity.support) if p]
            for b in s.background
        ]
        result = []
        for pc, values in capacity_options:
            for bg in product(*background):
                for exo, pe in enumerate(s.exogenous.initial):
                    if not pe:
                        continue
                    engine = MarketEngine(
                        c, copy.deepcopy(self.controllers_spec), check_invariants=False
                    )
                    engine.reset(tape=self.base_tape)
                    w = engine.world
                    w.exo = exo
                    for i, cs in enumerate(w.carriers):
                        cs.capacity = np.array(
                            [values[s.legs[l].id] for l in c.legs_of[i]], dtype=np.int64
                        )
                    for state, (_, k) in zip(w.background, bg):
                        state.stock = k
                    w.initial_capacity = tuple(values[l.id] for l in s.legs)
                    w.quote_history.initial = posted_quotes(w, c)
                    result.append(
                        (pc * math.prod(x[0] for x in bg) * pe, self.encode(w, focal))
                    )
                    if len(result) > self.max_outcomes:
                        raise KernelLimitError(
                            "initial capacity support exceeds max_outcomes"
                        )
        return result

    def posted_prices(
        self, ks: KernelState, focal: int, focal_action, observation=None
    ):
        """Post endogenous rival offers and the supplied focal action (Eqs. 3–16)."""
        engine = self.engine(ks, focal)
        engine.begin_period()
        values = engine.post_actions({focal: focal_action})
        if observation is not None and isinstance(focal_action, CoefficientAction):
            # Use the effective (possibly held) coefficients with the observed,
            # not the true/noiseless, focal quote features.
            prices = engine.rules[focal].prices(
                engine.world.carriers[focal].theta, observation
            )
            values[list(self.compiled.products_of[focal])] = prices
        return PostedPrices(ks, focal, focal_action, tuple(map(float, values)))

    def _request_options(self, world, prices):
        """
        Integrate markets, segments, classes and available alternatives (Eqs. 3–16).
        """
        c = self.compiled
        s = c.spec
        d = s.demand
        t = world.t
        if d.mode == "single_request_bernoulli":
            weights = d.market_weights.value_at(t)
        else:
            intensities = [d.intensity[m.id].value_at(t) for m in s.markets]
            total = sum(intensities)
            weights = (
                [x / total for x in intensities] if total else [0.0] * len(s.markets)
            )
        active = np.zeros(len(s.products), dtype=bool)
        for i, cs in enumerate(world.carriers):
            active[list(c.products_of[i])] = sellable(inventory(cs, c, i), c.A[i])
        for m, wm in enumerate(weights):
            if not wm:
                continue
            for g, wg in enumerate(segment_weights(c, t, m)):
                for k, taste in enumerate(s.segments[g].classes):
                    weight = wm * wg * taste.weight
                    if not weight:
                        continue
                    probabilities = choice_probabilities(
                        systematic_utilities(c, m, taste, prices, active)
                    )
                    for local, p in enumerate(probabilities):
                        if p:
                            j = (
                                c.market_products[m][local]
                                if local < len(c.market_products[m])
                                else None
                            )
                            yield weight * float(p), (m, g, k, j)

    def observation_distribution(
        self, ks: KernelState, focal: int, arm, observed_quotes=None
    ):
        """Enumerate discrete focal shopping signals, or evaluate their likelihood."""
        from ..pomdp.observation import QuoteObs
        from .quotes import quote_likelihood

        engine = self.engine(ks, focal)
        template = engine.observe(focal, arm)
        c = self.compiled
        requests = {q.id: q for q in c.spec.quotes.requests}
        names = {
            **c.product_index,
            **{b.id: len(c.spec.products) + i for i, b in enumerate(c.spec.background)},
        }
        combos = [(1.0, ())]
        position = 0
        for request_id in arm.requests:
            request = requests[request_id]
            if ks.t % request.every_k_periods:
                continue
            period = ks.t - request.lag_periods
            truth = engine.world.quote_history.get(period)
            for target in request.target_products:
                j = names[target]
                q = truth[j]
                if observed_quotes is not None:
                    if position >= len(observed_quotes):
                        return []
                    observed = observed_quotes[position]
                    position += 1
                    if observed.product != j or observed.period_posted != period:
                        return []
                    options = [(quote_likelihood(observed, q, request), observed)]
                else:
                    options = []
                    if request.missing_probability:
                        options.append(
                            (
                                request.missing_probability,
                                QuoteObs(j, period, None, None, None, True),
                            )
                        )
                    active = q.available if request.availability == "flag" else None
                    seats = (
                        min(q.seats, request.seats_cap)
                        if request.availability == "seats_capped"
                        else None
                    )
                    if request.fare_noise_sd:
                        width = request.fare_bin
                        sd = request.fare_noise_sd
                        first = math.floor((q.fare - 10 * sd) / width) - 1
                        last = math.ceil((q.fare + 10 * sd) / width) + 1
                        if last - first > self.max_outcomes:
                            raise KernelLimitError("too many discrete quote bins")
                        for index in range(first, last + 1):
                            fare = round(index * width, 2)
                            probability = float(
                                ndtr((fare + width / 2 - q.fare) / sd)
                                - ndtr((fare - width / 2 - q.fare) / sd)
                            )
                            if probability:
                                options.append(
                                    (
                                        (1 - request.missing_probability) * probability,
                                        QuoteObs(j, period, fare, active, seats, False),
                                    )
                                )
                        if abs(sum(p for p, _ in options) - 1) > self.prob_tol:
                            raise KernelLimitError(
                                "enumerated Gaussian quote tail exceeds probability tolerance"
                            )
                    else:
                        fare = (
                            round(
                                round(q.fare / request.fare_bin) * request.fare_bin, 2
                            )
                            if request.fare_bin
                            else round(q.fare, 2)
                        )
                        if request.missing_probability < 1:
                            options.append(
                                (
                                    1 - request.missing_probability,
                                    QuoteObs(j, period, fare, active, seats, False),
                                )
                            )
                combos = [
                    (p * qprob, (*quotes, quote))
                    for p, quotes in combos
                    for qprob, quote in options
                    if p * qprob > 0
                ]
                if len(combos) > self.max_outcomes:
                    raise KernelLimitError("quote combinations exceed max_outcomes")
        if observed_quotes is not None and position != len(observed_quotes):
            return []
        return [(p, replace(template, quotes=quotes)) for p, quotes in combos]

    def period_outcomes(self, ks: KernelState, posted: PostedPrices):
        """
        Enumerate a Bernoulli request or a declared bounded Poisson period (Eqs. 3–16).
        """
        c = self.compiled
        s = c.spec
        world = self.engine(ks, posted.focal).world
        t = ks.t
        if s.demand.mode == "single_request_bernoulli":
            arrival = bernoulli_probability(c, t, ks.exo)
            counts = [(1 - arrival, 0), (arrival, 1)]
        else:
            lam = sum(
                sc.value_at(t) for sc in s.demand.intensity.values()
            ) * demand_multiplier(c, ks.exo)
            n = self.max_requests
            self.tail_mass = float(poisson.sf(n, lam))
            if self.tail_mass > self.prob_tol_tail:
                raise KernelLimitError(
                    f"Poisson tail mass {self.tail_mass:.3g} exceeds {self.prob_tol_tail}"
                )
            counts = [(float(poisson.pmf(k, lam)), k) for k in range(n)] + [
                (float(poisson.sf(n - 1, lam)), n)
            ]
        result = []

        def expand(prob, remaining, w, requests, choices):
            """
            Enumerate chronological outcomes up to the declared period limit (Eqs.
            3–16).
            """
            if not remaining:
                result.append((prob, PeriodOutcome(tuple(requests), tuple(choices))))
                if len(result) > self.max_outcomes:
                    raise KernelLimitError("period enumeration exceeds max_outcomes")
                return
            for p, (m, g, k, j) in self._request_options(w, posted.values):
                next_w = w.clone()
                rank = len(requests)
                request = Request(rank, m, g, k, rank, rank)
                if j is not None:
                    book(
                        next_w,
                        int(c.product_carrier[j]),
                        j,
                        posted.values[j],
                        request,
                        c,
                    )
                expand(
                    prob * p, remaining - 1, next_w, requests + [request], choices + [j]
                )

        for p, n in counts:
            if p:
                expand(p, n, world, [], [])
        return result

    def transition(
        self,
        ks: KernelState,
        focal: int,
        focal_action,
        *,
        arm=None,
        observation=None,
        observed_quotes=None,
    ):
        """
        Enumerate purchases, background draws, external shocks and next signals (Eqs.
        3–16).
        """
        c = self.compiled
        arm = arm or self.arm
        if arm is None:
            arm = next(
                a
                for a in c.spec.information
                if a.observer == c.spec.carriers[focal].code
            )
        posted = self.posted_prices(ks, focal, focal_action, observation)
        rows = self.period_outcomes(ks, posted)
        results = []
        base = self.engine(ks, focal)
        bg_options = [
            [(1 - p, False), (p, True)]
            for p, b in [
                (purchase_probability(spec, x), spec)
                for spec, x in zip(c.spec.background, ks.background)
            ]
        ]
        hazards = []
        for event in c.spec.capacity_process.events:
            hazards.append(event.trigger.get("hazard"))
        for p, outcome in rows:
            for bg in product(*bg_options):
                pbg = math.prod(x[0] for x in bg)
                if not pbg:
                    continue
                for exo, pe in enumerate(transition_row(c, ks.exo, ks.t)):
                    if not pe:
                        continue
                    hazard_options = []
                    for hazard in hazards:
                        ph = (
                            hazard.get(
                                c.spec.exogenous.states[exo].name,
                                hazard.get("probability", 0.0),
                            )
                            if hazard
                            else 0.0
                        )
                        hazard_options.append([(1 - ph, False), (ph, True)])
                    for hs in product(*hazard_options):
                        ph = math.prod(x[0] for x in hs)
                        if not ph:
                            continue
                        engine = base.branch()
                        tape = self.base_tape
                        exo_u = np.array(tape.exo_uniform)
                        row = transition_row(c, ks.exo, ks.t)
                        exo_u[ks.t] = sum(row[:exo]) + pe / 2
                        bgu = np.array(tape.background_uniforms)
                        for k, (_, buy) in enumerate(bg):
                            bgu[ks.t, k] = 0.0 if buy else 1.0
                        capu = np.array(tape.capacity_uniforms)
                        for k, (_, fire) in enumerate(hs):
                            capu[ks.t, k] = 0.0 if fire else 1.0
                        n = max(1, len(outcome.requests))
                        g = np.zeros(
                            (c.spec.horizon.n_periods, n, len(c.spec.products) + 1)
                        )
                        mixed = np.zeros(
                            (c.spec.horizon.n_periods, n, len(c.spec.carriers) + 1)
                        )
                        engine.tape = replace(
                            tape,
                            exo_uniform=exo_u,
                            background_uniforms=bgu,
                            capacity_uniforms=capu,
                            gumbel=g,
                            mixed_draws=mixed,
                        )
                        action = focal_action
                        if observation is not None and isinstance(
                            action, CoefficientAction
                        ):
                            from ..core.types import PriceAction

                            old_theta = engine.world.carriers[focal].theta
                            update = next(
                                p.update_every
                                for p in self.compiled.spec.pricing
                                if p.carrier == self.compiled.spec.carriers[focal].code
                            )
                            engine.world.carriers[focal].theta = (
                                old_theta
                                if old_theta is not None and ks.t % update
                                else action.theta
                            )
                            action = PriceAction(
                                {j: posted.values[j] for j in c.products_of[focal]}
                            )
                        before = engine.world.carriers[focal].cash
                        step = engine.step(
                            {focal: action},
                            _requests=outcome.requests,
                            _choices=outcome.choices,
                        )
                        next_state = self.encode(engine.world, focal)
                        for pobs, obs in self.observation_distribution(
                            next_state, focal, arm, observed_quotes
                        ):
                            signal = FocalSignal(
                                tuple(
                                    (b.product, b.fare)
                                    for b in step.purchases
                                    if b.carrier == focal
                                ),
                                obs.own_capacity,
                                tuple(sorted(obs.public_exo.items())),
                                engine.world.carriers[focal].cash - before,
                                obs,
                            )
                            results.append(
                                (p * pbg * pe * ph * pobs, next_state, signal)
                            )
                        if len(results) > self.max_outcomes:
                            raise KernelLimitError(
                                "transition enumeration exceeds max_outcomes"
                            )
        return results
