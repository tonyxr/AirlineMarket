"""Matched coefficient-grid rollout; no access to the evaluation noise tape."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .belief import Belief
from .model import COEFFICIENT_GRID, Model, NoiseTape, Observation


@dataclass(frozen=True)
class Decision:
    coefficients: tuple[float, float, float]
    expected_objective: float | None


class RolloutPlanner:
    """One candidate action followed by a common tuned coefficient policy.

    Each candidate uses the same posterior quantiles and independent forecast
    tapes. Future B offers are recomputed on every branch. This is a finite
    simulation rollout approximation, not an exact optimal policy/upper bound.
    """

    def __init__(self, model: Model, tail_coefficients, scenarios: int = 12,
                 horizon: int | None = None):
        if scenarios < 1 or horizon is not None and horizon < 1:
            raise ValueError("scenarios and any rollout horizon must be positive")
        self.model = model
        self.tail = tuple(tail_coefficients)
        self.scenarios = scenarios
        self.horizon = horizon
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def choose(self, observation: Observation, belief: Belief, seed: int) -> Decision:
        # Ignore the additional raw signal after filtering; all arms use the
        # same posterior-to-action map, own state, and direct-price basis.
        key = (observation.time, observation.own_capacity, observation.own_bookings,
               observation.own_cash, observation.previous_own_prices,
               observation.weather_event, belief.key(), int(seed))
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        self.cache_misses += 1
        rng = np.random.default_rng(seed)
        states, weights = zip(*sorted(belief.mass.items()))
        cdf = np.cumsum(weights)
        quantiles = rng.random(self.scenarios)
        selected = [states[min(int(np.searchsorted(cdf, u, side="right")), len(states) - 1)] for u in quantiles]
        tapes = [NoiseTape.from_seed(int(s), self.model.config.horizon)
                 for s in rng.integers(0, 2**32, size=self.scenarios)]
        initial = [self.model.planning_state(observation, state) for state in selected]
        end = self.model.config.horizon
        if self.horizon is not None:
            end = min(end, observation.time + self.horizon)
        scores = []
        for action in COEFFICIENT_GRID:
            values = []
            for start, tape in zip(initial, tapes):
                state = start.clone()
                self.model.step(state, action, tape)
                while state.time < end:
                    self.model.step(state, self.tail, tape)
                values.append(self.model.metrics(state)["objective"])
            scores.append(float(np.mean(values)))
        # Stable deterministic tie breaking for genuinely matched negative controls.
        best = int(np.argmax(scores))
        result = Decision(tuple(COEFFICIENT_GRID[best]), scores[best])
        self.cache[key] = result
        return result
