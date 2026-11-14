"""Exact episode-type best responses with full ledgers and observable histories.

An opponent type is drawn once, independently of physical priors, and remains
fixed. The posterior is joint over (type, physical state), never a per-period
redraw or an average of separately optimized, type-revealed values.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, fields

import numpy as np

from ..agents.base import StatelessController
from ..agents.external import ExternalController
from ..core.errors import KernelLimitError
from ..core.hashing import canonical_json, spec_hash
from ..core.types import CoefficientAction
from ..sim.kernel import ExactKernel, KernelState
from ..sim.state import Booking
from ..spec.validate import validate_exact_compatible


@dataclass
class HistoryPolicy:
    """Total deterministic coefficient policy; default is part of its definition.

    Keys alternate permitted observations and the carrier's own action indexes.
    Histories absent from the fitted tree use the declared default action. This
    completes a policy on deviations' off-support histories, without a truth read.
    """

    grid: tuple
    table: dict
    default: int = 0
    arm: str = "own"

    def __post_init__(self):
        """Validate every executable action, including the off-tree definition."""
        self.grid = tuple(tuple(x) for x in self.grid)
        self.table = {tuple(k): int(v) for k, v in self.table.items()}
        if not self.grid or any(
            i < 0 or i >= len(self.grid) for i in [self.default, *self.table.values()]
        ):
            raise ValueError("history policy action outside declared grid")

    def action(self, path):
        """Select solely from the observable own-history key."""
        return self.table.get(tuple(path), self.default)

    def to_dict(self):
        """Serialize executable rows without lossy tuple-to-JSON-key coercion."""
        return {
            "grid": self.grid,
            "table": [[k, v] for k, v in sorted(self.table.items())],
            "default": self.default,
            "arm": self.arm,
        }

    @classmethod
    def from_dict(cls, data):
        """Load the complete policy, including its off-support action."""
        return cls(
            tuple(tuple(x) for x in data["grid"]),
            {tuple(k): v for k, v in data["table"]},
            data["default"],
            data["arm"],
        )

    def hash(self):
        """Identify the complete policy, including off-support behavior and arm."""
        rows = sorted((k, v) for k, v in self.table.items() if v != self.default)
        return spec_hash((self.grid, rows, self.default, self.arm))


class HistoryController(StatelessController):
    """Execute a finite own-history policy under native or full-ledger kernels.

    The old general kernel deliberately projects rival cash and booking dates;
    it cannot certify these policies. Only FullLedgerKernel accepts them here.
    """

    exact_compatible = False
    policy_class = "finite_observable_history"

    def __init__(self, policy):
        """Bind an immutable-in-use total action map and its permitted arm."""
        self.policy = policy
        self.arm_name = policy.arm

    def reset(self, ctx, rng):
        """Clear only the carrier's observable action/observation prefix."""
        super().reset(ctx, rng)
        self.prefix = ()

    def act(self, obs):
        """Append the permitted observation and selected own action to memory."""
        path = (*self.prefix, canonical_json(obs))
        index = self.policy.action(path)
        self.prefix = (*path, str(index))
        return CoefficientAction(self.policy.grid[index])

    def memory(self):
        """Expose hashable execution memory to exact hypothetical-state storage."""
        return self.prefix

    def snapshot(self):
        """Capture the observable prefix for exact replay."""
        return self.prefix

    def restore(self, state):
        """Restore that prefix without accessing an engine or hidden tape."""
        self.prefix = tuple(state)


@dataclass(frozen=True, order=True)
class FullLedgerState(KernelState):
    """Full physical ledger plus the last-purchase signal available to each actor."""

    recent_purchases: tuple = ()


class FullLedgerKernel(ExactKernel):
    """Reference physical kernel retaining every carrier's fares, cash and history.

    This intentionally avoids the general reference kernel's rival projections:
    a rival history policy can depend on its own past fares, dates and cash.
    """

    def __init__(self, compiled, controllers, **kwargs):
        """Validate the physical finite law and the expanded controller contract."""
        validate_exact_compatible(compiled.spec)
        if any(
            q.fare_noise_sd or q.missing_probability
            for q in compiled.spec.quotes.requests
        ):
            raise ValueError(
                "full-ledger history kernel requires deterministic shopping channels"
            )
        for controller in controllers.values():
            if not (
                isinstance(controller, HistoryController)
                or getattr(controller, "exact_compatible", False)
            ):
                raise ValueError(
                    "full-ledger kernel needs deterministic finite controllers"
                )
        super().__init__(compiled, controllers, validate=False, **kwargs)

    def canonical_state(self, state, focal):
        """Retain the full physical state and finite rival execution memory."""
        return state

    def encode(self, world, focal=0):
        """Encode each carrier as an own ledger, rather than a redacted rival."""
        state = super().encode(world, focal)
        carriers = tuple(
            super(FullLedgerKernel, self).encode(world, i).carriers[i]
            for i in range(len(world.carriers))
        )
        values = {f.name: getattr(state, f.name) for f in fields(KernelState)}
        values["carriers"] = carriers
        return FullLedgerState(
            **values,
            recent_purchases=tuple(
                (b.carrier, b.product, b.fare) for b in world.last_purchases
            ),
        )

    def engine(self, state, focal=0):
        """Restore recent own-sale observations along with all carrier ledgers."""
        engine = super().engine(state, focal)
        engine.world.last_purchases = tuple(
            Booking(-1, i, j, fare, state.t - 1, 0)
            for i, j, fare in state.recent_purchases
        )
        return engine


@dataclass
class TypeNode:
    """One permitted decision history and its normalized joint type/state law."""

    observation: object
    mass: dict
    prefix: tuple = ()

    @property
    def path(self):
        """Return the information-set key used by an executable policy."""
        return (*self.prefix, canonical_json(self.observation))

    def type_probabilities(self, n_types):
        """Marginalize the joint posterior; do not reset the episode-type prior."""
        result = np.zeros(n_types)
        for (kind, _), probability in self.mass.items():
            result[kind] += probability
        return result


class EpisodeTypeDP:
    """Exact best response over all own-history policies on the declared grid."""

    def __init__(
        self, compiled, focal, arm, opponent_types, probabilities, *, max_nodes=100_000
    ):
        """Bind the finite type bank and one focal information/action class."""
        self.compiled = compiled
        self.focal = compiled.carrier_index[focal]
        self.arm = next(
            a
            for a in compiled.spec.information
            if a.observer == focal and a.name == arm
        )
        if self.arm.privileged:
            raise ValueError("episode-type study requires a nonprivileged arm")
        self.grid = next(
            p.theta_grid for p in compiled.spec.pricing if p.carrier == focal
        )
        self.objective = next(o for o in compiled.spec.objectives if o.carrier == focal)
        self.probabilities = tuple(float(p) for p in probabilities)
        if (
            not opponent_types
            or len(opponent_types) != len(self.probabilities)
            or any(not math.isfinite(p) or p < 0 for p in self.probabilities)
            or abs(sum(self.probabilities) - 1) > 1e-12
        ):
            raise ValueError("episode-type prior must be a probability vector")
        expected = set(compiled.carrier_index) - {focal}
        if any(set(t) != expected for t in opponent_types):
            raise ValueError(
                "each episode type must specify every rival, never the focal"
            )
        self.kernels = tuple(
            FullLedgerKernel(compiled, {**t, focal: ExternalController()})
            for t in opponent_types
        )
        self.max_nodes = max_nodes
        self.cache = {}
        self.children = {}
        self.nodes = {}

    def key(self, node):
        """Preserve type identity, full physical support, posterior and own actions."""
        return node.path, tuple(sorted(node.mass.items()))

    def roots(self):
        """Condition the product type/physical prior on the initial observation."""
        groups = {}
        for kind, (prior, kernel) in enumerate(zip(self.probabilities, self.kernels)):
            if not prior:
                continue
            for p, state in kernel.initial_distribution(self.focal):
                for q, obs in kernel.observation_distribution(
                    state, self.focal, self.arm
                ):
                    if p * q:
                        key = canonical_json(obs)
                        if key not in groups:
                            groups[key] = (obs, defaultdict(float))
                        groups[key][1][kind, state] += prior * p * q
        return [
            (
                sum(mass.values()),
                TypeNode(obs, {s: p / sum(mass.values()) for s, p in mass.items()}),
            )
            for obs, mass in groups.values()
        ]

    def successors(self, node, index):
        """Bayes-group next signals while the same hidden type persists."""
        key = self.key(node), index
        if key in self.children:
            return self.children[key]
        groups = {}
        for (kind, state), probability in node.mass.items():
            for q, after, signal in self.kernels[kind].transition(
                state,
                self.focal,
                CoefficientAction(self.grid[index]),
                arm=self.arm,
                observation=node.observation,
            ):
                if not q:
                    continue
                k = canonical_json(signal.observation)
                if k not in groups:
                    groups[k] = (
                        signal.observation,
                        signal.cash_delta,
                        defaultdict(float),
                    )
                obs, cash, mass = groups[k]
                if abs(cash - signal.cash_delta) > 1e-10:
                    raise ValueError(
                        "cash reward is not measurable in the decision history"
                    )
                mass[kind, after] += probability * q
        rows = []
        for obs, cash, mass in groups.values():
            weight = sum(mass.values())
            child = TypeNode(
                obs, {s: p / weight for s, p in mass.items()}, (*node.path, str(index))
            )
            rows.append((weight, cash / self.objective.profit_scale, child))
        self.children[key] = rows
        return rows

    def terminal(self, node):
        """Settle service cost and the declared segment market-share objective."""
        c, o = self.compiled, self.objective
        markets = set(o.share_markets or [m.id for m in c.spec.markets if m.in_segment])
        result = 0.0
        for (_, state), p in node.mass.items():
            journeys = [
                sum(
                    n
                    for j, n in zip(c.products_of[i], cs.counts)
                    if c.spec.products[j].market in markets
                )
                for i, cs in enumerate(state.carriers)
            ]
            cost = sum(
                c.product_cost[j] * n
                for j, n in zip(
                    c.products_of[self.focal], state.carriers[self.focal].counts
                )
            )
            result += p * (
                -cost / o.profit_scale
                + (
                    o.share_weight * journeys[self.focal] / sum(journeys)
                    if sum(journeys)
                    else 0
                )
            )
        return result

    def q_values(self, node):
        """Integrate type-weighted action values before taking a common maximum."""
        if node.observation.t >= self.compiled.spec.horizon.n_periods:
            return np.full(len(self.grid), self.terminal(node))
        key = self.key(node)
        if key not in self.cache:
            if len(self.nodes) >= self.max_nodes:
                raise KernelLimitError("episode-type DP exceeds max_nodes")
            self.nodes[key] = node
            self.cache[key] = np.array(
                [
                    sum(
                        p * (r + self.q_values(child).max())
                        for p, r, child in self.successors(node, a)
                    )
                    for a in range(len(self.grid))
                ]
            )
        return self.cache[key]

    def solve(self):
        """Return the optimized expectation before any hidden type is revealed."""
        return float(sum(p * self.q_values(node).max() for p, node in self.roots()))

    def policy(self):
        """Export a total observable policy with explicit default action zero."""
        table = {}
        frontier = [node for _, node in self.roots()]
        while frontier:
            node = frontier.pop()
            if node.observation.t >= self.compiled.spec.horizon.n_periods:
                continue
            index = int(self.q_values(node).argmax())
            table[node.path] = index
            frontier.extend(child for p, _, child in self.successors(node, index) if p)
        return HistoryPolicy(self.grid, table, arm=self.arm.name)

    def evaluate(self, policy):
        """Evaluate a complete deterministic policy, without Bellman optimization."""
        memo = {}
        visited = set()

        def value(node):
            """Integrate the supplied total policy on its actual observable tree."""
            key = self.key(node)
            if key not in memo:
                visited.add(key)
                if len(visited) > self.max_nodes:
                    raise KernelLimitError(
                        "episode-type policy evaluation exceeds max_nodes"
                    )
                memo[key] = (
                    self.terminal(node)
                    if node.observation.t >= self.compiled.spec.horizon.n_periods
                    else sum(
                        p * (r + value(child))
                        for p, r, child in self.successors(
                            node, policy.action(node.path)
                        )
                    )
                )
            return memo[key]

        return float(sum(p * value(node) for p, node in self.roots()))
