"""Identity-only legacy sampling scheme, executed by the generic engine."""

from dataclasses import dataclass
import numpy as np
from ..core.types import CoefficientAction
from ..sim.kernel import ExactKernel
from ..sim.tape import LegacyToyTapeAdapter
from ..legacy_toy.model import NoiseTape


@dataclass(frozen=True)
class Decision:
    """Legacy-compatible selected coefficients and forecast objective (Eqs. 26–32)."""

    coefficients: tuple
    expected_objective: float


def legacy_sort_key(h):
    """
    Keep the prototype's rival-capacity/count posterior sampling order (Eqs. 26–32).
    """
    return (h.rivals[0].capacity, h.rivals[0].counts)


def planning_seed(master, episode_index, period):
    """
    Derive the frozen legacy forecast seed independently of evaluation noise (Eqs.
    26–32).
    """
    return int(
        np.random.SeedSequence([master, episode_index, period, 0xA17]).generate_state(
            1
        )[0]
    )


class LegacyRolloutPlanner:
    """
    Frozen posterior-sampling rollout scheme executed by the new engine (Eqs. 26–32).
    """

    def __init__(self, compiled, tail_coefficients, scenarios=12, horizon=None):
        """
        Bind declared inputs and initialize LegacyRolloutPlanner bookkeeping (Eqs.
        26–32).
        """
        self.compiled = compiled
        self.kernel = ExactKernel(compiled)
        self.tail = tuple(tail_coefficients)
        self.scenarios = scenarios
        self.horizon = horizon
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.grid = next(
            p.theta_grid for p in compiled.spec.pricing if p.carrier == "A"
        )

    def choose(self, observation, belief, seed):
        """
        Select a coefficient action from the permitted observation and belief (Eqs.
        26–32).
        """
        obs = observation
        mass = {}
        for h, p in belief.mass.items():
            k = legacy_sort_key(h)
            if k not in mass:
                mass[k] = [h, 0.0]
            mass[k][1] += p
        states, weights = zip(*(mass[k] for k in sorted(mass)))
        key = (
            obs.t,
            obs.own_capacity,
            tuple((b.product, b.fare) for b in obs.own_bookings),
            obs.own_cash,
            obs.own_previous_prices,
            tuple(sorted(obs.public_exo.items())),
            tuple((k, mass[k][1]) for k in sorted(mass)),
            int(seed),
        )
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        self.cache_misses += 1
        rng = np.random.default_rng(seed)
        cdf = np.cumsum(weights)
        selected = [
            states[min(int(np.searchsorted(cdf, u, side="right")), len(states) - 1)]
            for u in rng.random(self.scenarios)
        ]
        tapes = [
            LegacyToyTapeAdapter.from_legacy(
                NoiseTape.from_seed(int(s), self.compiled.spec.horizon.n_periods),
                self.compiled,
            )
            for s in rng.integers(0, 2**32, size=self.scenarios)
        ]
        initial = []
        for h, tape in zip(selected, tapes):
            engine = self.kernel.engine(h.kernel, obs.carrier)
            engine.tape = tape
            initial.append(engine)
        end = (
            self.compiled.spec.horizon.n_periods
            if self.horizon is None
            else min(self.compiled.spec.horizon.n_periods, obs.t + self.horizon)
        )
        from ..sim.batch import supports, rollout_values

        if self.horizon is None and supports(self.compiled, obs.carrier, initial):
            scores = rollout_values(
                self.compiled, obs.carrier, initial, self.grid, self.tail
            ).mean(1)
            best = int(np.argmax(scores))
            decision = Decision(tuple(self.grid[best]), float(scores[best]))
            self.cache[key] = decision
            return decision
        scores = []
        for action in self.grid:
            values = []
            for start in initial:
                e = start.branch()
                e.step({obs.carrier: CoefficientAction(action)})
                while e.world.t < end:
                    e.step({obs.carrier: CoefficientAction(self.tail)})
                values.append(e.metrics()["objective"])
            scores.append(float(np.mean(values)))
        best = int(np.argmax(scores))
        decision = Decision(tuple(self.grid[best]), scores[best])
        self.cache[key] = decision
        return decision
