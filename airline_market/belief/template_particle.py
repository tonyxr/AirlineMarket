"""Reuse seed-zero construction state without changing SMC draws or proposals.

New opt-in revision. Existing kernels, particle implementations and campaigns
remain unchanged. Templates contain only model-generated seed-zero state;
physical evaluation tapes are never inputs to this cache.
"""

import copy
from time import perf_counter

import numpy as np

from airline_market.belief import particle as particle_module
from airline_market.belief.particle import ParticleBelief
from airline_market.core.types import PeriodPhase
from airline_market.sim.engine import MarketEngine
from airline_market.sim.kernel import ExactKernel, KernelState
from airline_market.sim.state import Booking


def fork_initial_engine(template):
    """Clone mutable world/controllers while sharing only immutable model inputs.

    This is equivalent to constructing/resetting the identical seed-zero
    engine. Unlike a generic deepcopy, controller contexts retain the same
    compiled model and read-only tape; each controller/RNG is independent.
    """
    engine = copy.copy(template)
    memo = {id(template.compiled): template.compiled, id(template.tape): template.tape}
    engine.controllers = {
        i: copy.deepcopy(ctrl, memo) for i, ctrl in template.controllers.items()
    }
    engine.rules = {i: copy.deepcopy(rule, memo) for i, rule in template.rules.items()}
    engine.world = template.world.clone()
    engine._trace = list(template._trace)
    return engine


class TemplateExactKernel(ExactKernel):
    """Keep reference transitions/decode exactly, caching construction only."""

    def __init__(self, *args, **kwargs):
        """Initialize the reference kernel and an empty hypothetical-engine cache."""
        super().__init__(*args, **kwargs)
        self._template = None
        self.engine_calls = 0
        self.template_builds = 0

    def initial_engine(self):
        """Construct one independent seed-zero engine for this controller mapping."""
        if self._template is None:
            self._template = MarketEngine(
                self.compiled,
                copy.deepcopy(
                    self.controllers_spec, {id(self.compiled): self.compiled}
                ),
                check_invariants=False,
            )
            self._template.reset(tape=self.base_tape)
            self.template_builds += 1
        self.engine_calls += 1
        return fork_initial_engine(self._template)

    def engine(self, ks: KernelState, focal: int = 0) -> MarketEngine:
        """
        Reconstruct a hypothetical reference engine from a model-generated state (Eqs.
        3–16).
        """
        c = self.compiled
        engine = self.initial_engine()
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


class _PriorWorld:
    """The reset/world-only interface used by the unchanged particle prior loop."""

    def __init__(self, template):
        """Clone the exact post-reset world, including initial controller memory."""
        self.world = template.world.clone()

    def reset(self, seed=None, *, tape=None):
        """Replay only the already constructed seed-zero reset used in that loop."""
        if seed != 0 or tape is not None:
            raise ValueError(
                "particle prior replay is only the reference seed-zero reset"
            )


class TemplateParticleBelief(ParticleBelief):
    """Reference SMC with shared construction templates and unchanged strict failures."""

    def __init__(self, compiled, focal, arm, **kwargs):
        """Select the opt-in reference kernel; retain every original SMC setting."""
        if kwargs.get("kernel") is None:
            kwargs["kernel"] = TemplateExactKernel(compiled, validate=False)
        super().__init__(compiled, focal, arm, **kwargs)
        self._prior_template = None
        self.prior_reset_seconds = []
        self.prior_world_copies = 0

    def reset(self, first_obs):
        """Reuse the identical initial world, without enumerating or replacing priors.

        The original reset implementation calls its module-local engine alias
        only as constructor -> reset(0) -> world. Bind that alias solely during
        this single-worker reset and restore it on every failure path. All prior
        sampling, likelihoods, ordering and RNG consumption remain original.
        """
        started = perf_counter()
        if self._prior_template is None:
            self._prior_template = MarketEngine(self.compiled, check_invariants=False)
            self._prior_template.reset(tape=self.kernel.base_tape)
        original = particle_module.MarketEngine

        def initial_world(compiled, *, check_invariants=False):
            """Return exactly the model prior's post-reset seed-zero world."""
            if compiled is not self.compiled or check_invariants:
                raise ValueError("unexpected particle prior construction interface")
            self.prior_world_copies += 1
            return _PriorWorld(self._prior_template)

        particle_module.MarketEngine = initial_world
        try:
            return super().reset(first_obs)
        finally:
            particle_module.MarketEngine = original
            self.prior_reset_seconds.append(perf_counter() - started)
