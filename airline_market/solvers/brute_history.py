"""Explicit policy enumeration for the binary-sale-observation micro family.

Every action map on the full finite own-sale history tree is evaluated by
forward physical-state probabilities. There is no Bellman maximization,
posterior optimization, or hidden-state action selection in the enumerator.
"""

from collections import defaultdict
from functools import lru_cache
from itertools import product

import numpy as np

from ..core.hashing import canonical_json
from ..core.types import CoefficientAction
from ..sim.kernel import ExactKernel
from ..spec import load_spec


def binary_sale_micro(parameters):
    """Declare an inventory-uncertain duopoly with one binary own-sale signal."""
    T = int(parameters.get("T", 3))
    a_capacity = int(parameters.get("own_capacity", 2))
    support = parameters.get("rival_support", [1, 2])
    return load_spec({"schema_version": "airline-market-spec/1", "name": "binary_sale_micro",
        "horizon": {"n_periods": T}, "airports": [{"code": "O"}, {"code": "D"}],
        "carriers": [{"code": "A"}, {"code": "B"}],
        "legs": [{"id": "A_OD", "carrier": "A", "origin": "O", "dest": "D",
                  "capacity": {"support": [a_capacity], "probabilities": [1.0]}},
                 {"id": "B_OD", "carrier": "B", "origin": "O", "dest": "D",
                  "capacity": {"support": support,
                               "probabilities": [1 / len(support)] * len(support)}}],
        "products": [{"id": c + "_p", "carrier": c, "market": "OD", "legs": [c + "_OD"],
                      "reference_fare": 100, "fare_floor": 20, "fare_cap": 300,
                      "service_cost": 10, "priority": i} for i, c in enumerate(["A", "B"])],
        "markets": [{"id": "OD", "origin": "O", "dest": "D", "in_segment": True}],
        "segments": [{"id": "single", "classes": [{"weight": 1.0, "alpha": 0.02,
            "carrier_pref": {"A": 0, "B": 0.1}, "market_value": {"*": 3.5}}]}],
        "demand": {"mode": "single_request_bernoulli",
                   "arrival_probability": {"constant": parameters.get("arrival", 0.7)},
                   "market_weights": {"constant": [1]},
                   "segment_weights": {"constant": [1]}},
        "choice": {}, "exogenous": {"states": [{"name": "normal"}], "initial": [1],
                                     "transitions": {}, "default_transition": [[1]]},
        "capacity_process": {}, "resolution": {}, "quotes": {"requests": []},
        "information": [{"name": "direct", "observer": "A", "requests": []}],
        "pricing": [{"carrier": "A", "base": {"A_p": 100},
                     "controllable_features": [{"name": "intercept"}],
                     "theta_bounds": [[-40, 40]], "theta_grid": [[-40], [40]]},
                    {"carrier": "B", "base": {"B_p": 100}, "theta_grid": [[]]}],
        "controllers": [{"carrier": "A", "controller": "external"},
                        {"carrier": "B", "controller": "threshold_linear_rule",
                         "params": {"rules": {"B_p": {"base": 130, "scarcity": [
                             {"leg": "B_OD", "threshold": 1, "coefficient": 60}]}}}}],
        "objectives": [{"carrier": c, "profit_scale": 1000,
                        "share_weight": parameters.get("share_weight", 0.0)} for c in ["A", "B"]],
        "timing": {"mode": "leader_follower", "leaders": ["B"]},
        "metadata": {"focal": "A", "verification_family": "binary_sale_history",
                     "parameters": parameters}})


class BruteHistoryEnumerator:
    """Evaluate every binary-history policy without invoking an optimized DP."""

    def __init__(self, compiled):
        """Check the micro observation tree and prepare only physical transition caches."""
        self.compiled = compiled
        s = compiled.spec
        self.T = s.horizon.n_periods
        self.focal = compiled.carrier_index["A"]
        self.arm = next(a for a in s.information if a.observer == "A" and a.name == "direct")
        if (self.T not in (2, 3) or self.arm.requests or self.arm.privileged
                or len(compiled.products_of[self.focal]) != 1
                or s.demand.mode != "single_request_bernoulli"
                or len(s.markets) != 1 or len(s.segments) != 1
                or len(s.segments[0].classes) != 1 or s.capacity_process.events or s.background):
            raise ValueError("enumeration certificate requires the binary-sale micro observation family")
        self.grid = next(p.theta_grid for p in s.pricing if p.carrier == "A")
        if len(self.grid) != 2:
            raise ValueError("binary policy enumeration requires exactly two actions")
        self.kernel = ExactKernel(compiled)
        self.objective = next(o for o in s.objectives if o.carrier == "A")
        self.observations = {}
        self.initial = []
        for prior, state in self.kernel.initial_distribution(self.focal):
            for probability, obs in self.kernel.observation_distribution(state, self.focal, self.arm):
                key = canonical_json(obs)
                self.observations[key] = obs
                self.initial.append((state, key, prior * probability))
        if len(self.observations) != 1:
            raise ValueError("initial observable branching needs a larger policy-history alphabet")
        self.transitions = lru_cache(maxsize=None)(self._transitions)

    def _transitions(self, state, observation_key, action):
        """Cache physical paths only, retaining the actual permitted outcome signal."""
        rows = []
        for probability, after, signal in self.kernel.transition(
                state, self.focal, CoefficientAction(self.grid[action]),
                arm=self.arm, observation=self.observations[observation_key]):
            key = canonical_json(signal.observation)
            self.observations[key] = signal.observation
            symbol = int(bool(signal.purchases))
            rows.append((probability, after, key, symbol))
        return tuple(rows)

    def terminal_objective(self, state):
        """Settle complete path cash/service/share from physical ledgers, without a posterior."""
        c = self.compiled
        own = state.carriers[self.focal]
        service = sum(c.product_cost[j] * n for j, n in zip(c.products_of[self.focal], own.counts))
        journeys = [sum(cs.counts) for cs in state.carriers]
        share = journeys[self.focal] / sum(journeys) if sum(journeys) else 0.0
        return (own.cash - service) / self.objective.profit_scale + self.objective.share_weight * share

    def value(self, actions):
        """Forward-evaluate a complete action map while checking observation sufficiency."""
        if len(actions) != 2**self.T - 1:
            raise ValueError("one action is required for every binary prefix")
        frontier = {(state, obs, ()): p for state, obs, p in self.initial}
        for _ in range(self.T):
            observed_by_history = defaultdict(set)
            following = defaultdict(float)
            for (state, obs, history), probability in frontier.items():
                observed_by_history[history].add(obs)
                index = 2**len(history) - 1 + sum(bit * 2**(len(history) - k - 1)
                                                for k, bit in enumerate(history))
                action = int(actions[index])
                if action not in (0, 1):
                    raise ValueError("policy actions must belong to the two-action grid")
                for p, after, key, symbol in self.transitions(state, obs, action):
                    following[after, key, (*history, symbol)] += probability * p
            if any(len(keys) != 1 for keys in observed_by_history.values()):
                raise ValueError("binary sale history omits a stochastic permitted observation")
            frontier = following
        total = sum(frontier.values())
        if abs(total - 1) > 1e-10:
            raise ValueError("enumerated physical probability mass was not conserved")
        return sum(p * self.terminal_objective(state)
                   for (state, _, _), p in frontier.items())

    def enumerate(self):
        """Return every explicit policy value; maximize only after complete enumeration."""
        policies = list(product((0, 1), repeat=2**self.T - 1))
        values = np.array([self.value(policy) for policy in policies])
        return {"policies": policies, "values": values.tolist(),
                "maximum": float(values.max()), "first_maximizer": int(values.argmax()),
                "transition_cache_entries": self.transitions.cache_info().currsize,
                "scope": "All deterministic policies on the complete binary own-sale history tree; observation sufficiency checked for every evaluated policy. No Bellman optimization inside enumeration."}
