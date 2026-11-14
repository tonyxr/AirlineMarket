"""Compact Bellman and exact fixed-tail evaluation (Eqs. 26–29).

Cash is settled in edge rewards. Booking dates and cumulative cash therefore
never enter these no-refund decision states. No belief masses are rounded.
"""

from collections import defaultdict
from functools import lru_cache
import numpy as np
from ..core.errors import KernelLimitError


class FiniteBeliefSolver:
    """Memoized optimal control over sufficient inventory posteriors."""

    def __init__(self, model, grid, max_nodes=2_000_000):
        """
        Bind declared inputs and initialize FiniteBeliefSolver bookkeeping (Eqs. 26–32).
        """
        self.model = model
        self.grid = tuple(tuple(x) for x in grid)
        self.max_nodes = max_nodes
        self.cache = {}
        self.counts = defaultdict(int)

    def key(self, node):
        """Discard last-step hidden uncertainty already resolved by offers."""
        t, capacity, own, journeys, previous, signal, belief = node
        if self.model.monopoly(own, belief):
            return t, capacity, own, "monopoly"
        if t == self.model.T - 1 and self.model._stock_only_last_step:
            return t, capacity, own, journeys, self.model.immediate_signal(own, signal)
        return (
            t,
            capacity,
            own,
            journeys,
            None if self.model.previous_encoded_by_signal else previous,
            signal,
            belief,
        )

    def successors(self, node, index):
        """Integrate shared requests and condition on the next visible signal."""
        m = self.model
        t, capacity, own, journeys, previous, signal, belief = node
        prices = m.prices(self.grid[index], t, capacity, own, signal)
        groups = {}
        for hidden, weight in belief:
            for p, new_own, new_journeys, new_hidden, reward, sale in m.step(
                t, own, journeys, hidden, prices, previous
            ):
                observation = m.signal(t + 1, new_hidden, prices)
                key = new_own, new_journeys, observation, sale
                if key not in groups:
                    groups[key] = [0.0, 0.0, defaultdict(float)]
                group = groups[key]
                probability = weight * p
                group[0] += probability
                group[1] += probability * reward
                group[2][new_hidden] += probability
        for (new_own, new_journeys, observation, sale), (
            probability,
            reward,
            mass,
        ) in groups.items():
            posterior = tuple(sorted((h, p / probability) for h, p in mass.items()))
            yield (
                probability,
                reward / probability,
                (
                    t + 1,
                    capacity,
                    new_own,
                    new_journeys,
                    prices,
                    observation,
                    posterior,
                ),
            )

    def q_values(self, node):
        """Return exact future net-profit/share values for every coefficient."""
        m = self.model
        t, capacity, own, journeys, previous, signal, belief = node
        if t >= m.T:
            return np.full(
                len(self.grid), sum(p * m.terminal(journeys, h) for h, p in belief)
            )
        if not m.track_journeys and m.sold_out(own):
            return np.zeros(len(self.grid))
        key = self.key(node)
        if key in self.cache:
            if self.cache[key] is None:
                raise KernelLimitError(
                    "this DP node is incomplete after a previous enumeration limit"
                )
            return self.cache[key]
        if len(self.cache) >= self.max_nodes:
            raise KernelLimitError(
                f"compact exact DP exceeds max_nodes={self.max_nodes}; counts={dict(self.counts)}"
            )
        # Reserve before descending, so the limit counts live and solved nodes.
        self.cache[key] = None
        self.counts[t] += 1
        if t == m.T - 1:
            result = []
            for theta in self.grid:
                prices = m.prices(theta, t, capacity, own, signal)
                result.append(
                    sum(
                        weight * p * (reward + m.terminal(new_journeys, new_hidden))
                        for hidden, weight in belief
                        for p, _, new_journeys, new_hidden, reward, _ in m.step(
                            t, own, journeys, hidden, prices, previous
                        )
                    )
                )
            q = np.asarray(result)
        else:
            q = np.array(
                [
                    sum(
                        p * (reward + float(self.q_values(child).max()))
                        for p, reward, child in self.successors(node, index)
                    )
                    for index in range(len(self.grid))
                ]
            )
        self.cache[key] = q
        return q


class ExactTailEvaluator:
    """Exact one-step-then-tail values using a fully observed policy evaluator.

    The fixed tail does not adapt coefficients to a future belief. Its expected
    value is linear in the current posterior, so full belief trees are avoided.
    Prices and rival reactions are still recomputed in every successor state.
    """

    def __init__(self, model, tail):
        """
        Bind declared inputs and initialize ExactTailEvaluator bookkeeping (Eqs. 26–32).
        """
        self.model = model
        self.tail = tuple(tail) if not isinstance(tail, dict) else tail
        self.states = 0
        self._value = lru_cache(maxsize=1_000_000)(self._value)

    def theta(self, t):
        """Resolve a fixed or piecewise-constant coefficient schedule."""
        if isinstance(self.tail, dict):
            return tuple(
                self.tail["thetas"][
                    next(
                        k for k, (a, b) in enumerate(self.tail["blocks"]) if a <= t < b
                    )
                ]
            )
        return self.tail

    def _value(self, t, capacity, own, journeys, hidden, previous):
        """Expected future net objective of the declared tail (Eq. 27)."""
        m = self.model
        self.states += 1
        if t >= m.T:
            return m.terminal(journeys, hidden)
        if not m.track_journeys and m.sold_out(own):
            return 0.0
        signal = m.signal(t, hidden, previous)
        prices = m.prices(self.theta(t), t, capacity, own, signal)
        return sum(
            p
            * (
                reward
                + self._value(
                    t + 1, capacity, new_own, new_journeys, new_hidden, prices
                )
            )
            for p, new_own, new_journeys, new_hidden, reward, _ in m.step(
                t, own, journeys, hidden, prices, previous
            )
        )

    def q_values(self, obs, belief, grid):
        """Expected complete-cohort objectives, conditional on observed history."""
        m = self.model
        t, capacity, own, journeys, previous, signal, mass = m.project(obs, belief)
        sunk = obs.own_cash / m.objective.profit_scale - m.existing_service_cost(obs)
        return np.asarray(
            [
                sunk
                + sum(
                    weight
                    * p
                    * (
                        reward
                        + self._value(
                            t + 1, capacity, new_own, new_journeys, new_hidden, prices
                        )
                    )
                    for hidden, weight in mass
                    for p, new_own, new_journeys, new_hidden, reward, _ in m.step(
                        t, own, journeys, hidden, prices, previous
                    )
                )
                for theta in grid
                for prices in [m.prices(tuple(theta), t, capacity, own, signal)]
            ]
        )

    def root_value(self, roots):
        """Average the fixed tail over the original initial observation law."""
        total = 0.0
        for probability, node in roots:
            t, capacity, own, journeys, previous, _, mass = self.model.project(
                node.observation, node.belief
            )
            total += probability * sum(
                p * self._value(t, capacity, own, journeys, h, previous)
                for h, p in mass
            )
        return total
