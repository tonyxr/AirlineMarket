"""Exact finite-state filtering from permitted quotes and own sales only."""

from __future__ import annotations

from collections import defaultdict
import math

from .model import Model, Observation, RivalState


class Belief:
    def __init__(self, model: Model):
        self.model = model
        self.mass = model.prior()

    def normalize(self, weights: dict[RivalState, float]) -> None:
        total = sum(weights.values())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("observation has zero model likelihood; refusing a hidden-state fallback")
        self.mass = {state: weight / total for state, weight in weights.items() if weight > 0}

    def condition(self, observation: Observation) -> None:
        if observation.privileged_rival is not None:
            self.mass = {observation.privileged_rival: 1.0}
            return
        weights = {}
        for state, weight in self.mass.items():
            predicted = {q.product: q for q in self.model.rival_quotes(state, observation.previous_own_prices)}
            # XY has independent seats and demand. Its likelihood cancels.
            if all(q.product == "XY" or predicted[q.product] == q for q in observation.quotes):
                weights[state] = weight
        self.normalize(weights)

    def advance(self, before: Observation, coefficients, own_sale: int | None,
                after: Observation) -> None:
        prices = self.model.prices(coefficients, before.time, before.own_inventory, before.quotes)
        weights = defaultdict(float)
        for state, weight in self.mass.items():
            outcomes = self.model.outcomes(before.time, before.own_inventory, state.inventory,
                                           prices, before.previous_own_prices, before.weather_event)
            if own_sale is not None:
                weights[state] += weight * outcomes[own_sale]
            else:
                weights[state] += weight * outcomes[5]
                for j in range(3):
                    if outcomes[j + 2] > 0:
                        weights[state.purchase(j)] += weight * outcomes[j + 2]
        if after.weather_event and not before.weather_event:
            transformed = defaultdict(float)
            for state, weight in weights.items():
                transformed[state.capacity_loss()] += weight
            weights = transformed
        self.normalize(weights)
        self.condition(after)

    def entropy(self) -> float:
        return -sum(p * math.log(p) for p in self.mass.values())

    def mean_inventory(self) -> tuple[float, float]:
        return tuple(sum(state.inventory[l] * p for state, p in self.mass.items()) for l in range(2))

    def key(self):
        return tuple(sorted(self.mass.items()))
