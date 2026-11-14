"""Exact penultimate-period conditional vectors for a certified final period.

When every competing current offer is visible and there is no terminal share
term, the final-period optimum depends on the next observation rather than
its hidden posterior. For each penultimate action/physical hypothesis, cache
its expected reward plus that final optimum. Mixing these vectors with the
current posterior is linear and avoids a separate Bellman expansion for each
distinct penultimate belief. No probabilities or beliefs are rounded.
"""

from functools import lru_cache
import numpy as np

from ..core.errors import KernelLimitError
from .finite_dp import FiniteBeliefSolver


class ConditionalTailSolver(FiniteBeliefSolver):
    """Retain the original exact recursion outside the certified two-period tail."""

    def __init__(self, model, grid, max_nodes=2_000_000, *, tail_cache_size=250000):
        """Keep a bounded exact-vector cache and explicit independent work counters."""
        super().__init__(model, grid, max_nodes)
        self.tail_enabled = bool(
            model._stock_only_last_step and not model.track_journeys
        )
        self.tail_belief_queries = 0
        self.tail_vector_evaluations = 0
        self._tail_live = 0
        self.tail_vectors = lru_cache(maxsize=tail_cache_size)(self._tail_vector)

    def memo_entries(self):
        """Count retained prefix/final Bellman entries and conditional vectors."""
        return (
            len(self.cache) + self.tail_vectors.cache_info().currsize + self._tail_live
        )

    def _require_room(self):
        """Apply the declared node budget to all live/retained exact memo entries."""
        if self.memo_entries() >= self.max_nodes:
            raise KernelLimitError(
                f"conditional-tail exact DP exceeds max_nodes={self.max_nodes}"
            )

    def _tail_vector(self, t, capacity, own, journeys, previous, signal, hidden):
        """Integrate the final observed optimum conditional on one current hypothesis."""
        self._require_room()
        self._tail_live += 1
        self.tail_vector_evaluations += 1
        model = self.model
        try:
            values = []
            for theta in self.grid:
                prices = model.prices(theta, t, capacity, own, signal)
                total = 0.0
                for (
                    probability,
                    new_own,
                    new_journeys,
                    new_hidden,
                    reward,
                    _,
                ) in model.step(t, own, journeys, hidden, prices, previous):
                    next_signal = model.signal(t + 1, new_hidden, prices)
                    child = (
                        t + 1,
                        capacity,
                        new_own,
                        new_journeys,
                        prices,
                        next_signal,
                        ((new_hidden, 1.0),),
                    )
                    # The certificate makes this value identical across all
                    # hypotheses with the same next permitted signal. This
                    # point mass computes a conditional coefficient, not an
                    # oracle action for the penultimate decision.
                    total += probability * (reward + float(self.q_values(child).max()))
                values.append(total)
            result = np.asarray(values)
            result.setflags(write=False)
            return result
        finally:
            self._tail_live -= 1

    def q_values(self, node):
        """Mix exact conditional vectors only where the last-step certificate applies."""
        t, capacity, own, journeys, previous, signal, belief = node
        model = self.model
        if t >= model.T or not model.track_journeys and model.sold_out(own):
            return super().q_values(node)
        if self.tail_enabled and t == model.T - 2:
            self.tail_belief_queries += 1
            values = np.zeros(len(self.grid))
            for hidden, weight in belief:
                values += weight * self.tail_vectors(
                    t, capacity, own, journeys, previous, signal, hidden
                )
            return values
        key = self.key(node)
        if key not in self.cache:
            self._require_room()
        return super().q_values(node)
