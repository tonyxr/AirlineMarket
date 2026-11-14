"""Reuse the immutable dummy construction tape in particle-prior initialization.

The reference reset constructs an engine at seed0 for every distinct sampled
capacity vector, then replaces its capacity with that model-prior draw. The
construction tape is identical each time. Reusing that read-only object
preserves all draws, particle weights and failures, without accessing the
physical evaluation tape. The temporary method binding is worker-local.
"""

from time import perf_counter

from .particle import ParticleBelief
from ..sim.engine import MarketEngine
from ..sim.tape import ExogenousTape


class CachedPriorParticleBelief(ParticleBelief):
    """Preserve reference SMC while avoiding repeated identical tape generation."""

    def __init__(self, *args, **kwargs):
        """Create the reference filter and empty construction-tape cache."""
        super().__init__(*args, **kwargs)
        self._construction_tape = None
        self.prior_reset_seconds = []
        self.prior_tape_seconds = 0.0

    def reset(self, first_obs):
        """Reuse only seed0 construction draws; restore the method even on failure."""
        started = perf_counter()
        if self._construction_tape is None:
            begin = perf_counter()
            self._construction_tape = ExogenousTape.generate(self.compiled, 0)
            self.prior_tape_seconds = perf_counter() - begin
        original = MarketEngine.reset
        compiled = self.compiled

        def reset(engine, seed=None, *, tape=None):
            """Apply the same immutable tape only to matching dummy engine resets."""
            construction = (
                self._construction_tape
                if engine.compiled is compiled and seed == 0 and tape is None
                else tape
            )
            return original(engine, seed, tape=construction)

        MarketEngine.reset = reset
        try:
            return super().reset(first_obs)
        finally:
            MarketEngine.reset = original
            self.prior_reset_seconds.append(perf_counter() - started)
