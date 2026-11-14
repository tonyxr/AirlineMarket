"""Finite-horizon belief-tree optimization. No sampling or hidden-state oracle."""

from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from ..belief import ExactBelief, factorize
from ..core.hashing import canonical_json
from ..core.errors import KernelLimitError
from ..core.types import CoefficientAction
from ..pomdp.objective import ObjectiveEvaluator
from ..sim.kernel import ExactKernel


def argmax_first(values, tol=1e-10):
    """
    Choose the lowest-index maximizing action under the declared tie tolerance (Eqs.
    26–32).
    """
    a = np.asarray(values)
    return int(np.flatnonzero(a >= a.max() - tol)[0])


@dataclass
class DPNode:
    """
    Observable decision prefix and its conditional finite hidden-state posterior (Eqs.
    26–32).
    """

    observation: object
    belief: ExactBelief

    @property
    def t(self):
        """Return the booking period of this decision node (Eqs. 26–32)."""
        return self.observation.t


@dataclass(frozen=True)
class DPResult:
    """Exact root value and actual enumerated decision-node counts (Eqs. 26–32)."""

    value: float
    nodes: int
    nodes_per_period: dict
    exact: bool = True


@dataclass(frozen=True)
class InfoValueReport:
    """Two optimized arm values and their signed difference (Eqs. 26–32)."""

    V_a: float
    V_b: float
    delta: float


class ExactBeliefDP:
    """
    Finite-horizon exact coefficient control with sufficient-statistic memoization (Eqs.
    26–32).
    """

    def __init__(
        self,
        compiled,
        focal,
        arm,
        rival_controllers=None,
        theta_grid=None,
        *,
        max_nodes=2_000_000,
        node_key="sufficient",
        belief_merge_tol=0.0,
        backend="auto",
    ):
        """
        Bind declared inputs and initialize ExactBeliefDP bookkeeping (Eqs. 26–32).
        """
        if belief_merge_tol:
            raise ValueError("approximate belief merging is not an exact result")
        if node_key not in ("history", "belief", "sufficient"):
            raise ValueError("unknown node key")
        self.compiled = compiled
        self.focal = compiled.carrier_index[focal]
        self.code = focal
        self.arm = next(
            a
            for a in compiled.spec.information
            if a.name == arm and a.observer == focal
        )
        self.kernel = ExactKernel(compiled, rival_controllers)
        self.max_nodes = max_nodes
        self.node_key = node_key
        self.grid = tuple(
            theta_grid
            or next(p.theta_grid for p in compiled.spec.pricing if p.carrier == focal)
        )
        self.objective = ObjectiveEvaluator(
            next(o for o in compiled.spec.objectives if o.carrier == focal), compiled
        )
        self.cache = {}
        self.nodes = {}
        self.children = {}
        self.counts = defaultdict(int)
        from ..sim.capacity import refunds_possible

        self.refunds_possible = refunds_possible(compiled)
        self.finite = None
        self.backend = "reference"
        self.backend_reason = None
        if backend not in ("auto", "reference", "compact"):
            raise ValueError("unknown DP backend")
        if backend != "reference" and node_key != "history":
            from ..sim.finite import FinitePricingModel, UnsupportedFiniteModel
            from .finite_dp import FiniteBeliefSolver

            try:
                self.finite = FiniteBeliefSolver(
                    FinitePricingModel(compiled, focal, self.arm, rival_controllers),
                    self.grid,
                    max_nodes,
                )
                self.backend = "compact"
            except UnsupportedFiniteModel as error:
                self.backend_reason = str(error)
                if backend == "compact":
                    raise

    def key(self, node):
        """
        Keep future-relevant counts/fares, quotes, memory and posterior; exclude sunk
        cash (Eqs. 26–32).
        """
        obs = node.observation
        if self.node_key == "history":
            return canonical_json(obs), node.belief.key(), repr(node.belief.history)
        refunds = self.refunds_possible
        products = self.compiled.products_of[self.focal]
        ledger = tuple(
            tuple(b.fare for b in obs.own_bookings if b.product == j)
            if refunds
            else sum(b.product == j for b in obs.own_bookings)
            for j in products
        )

        def state_key(ks):
            """
            Project kernel state onto future-relevant ledger, memory and history fields
            (Eqs. 26–32).
            """
            carriers = tuple(
                (
                    cs.capacity,
                    cs.counts,
                    cs.fares if refunds and i == self.focal else None,
                    cs.previous_prices,
                    cs.controller_memory,
                    cs.theta,
                )
                for i, cs in enumerate(ks.carriers)
            )
            return (
                ks.t,
                ks.exo,
                carriers,
                ks.background,
                ks.phase,
                ks.history,
                ks.initial_quotes,
                ks.events_fired,
                ks.posted_leaders,
            )

        return (
            obs.t,
            obs.own_capacity,
            ledger,
            obs.own_previous_prices,
            tuple(sorted(obs.public_exo.items())),
            obs.quotes,
            tuple(
                (state_key(h.kernel), float(p))
                for h, p in sorted(node.belief.mass.items())
            ),
        )

    def belief(self, mass, obs):
        """
        Construct a normalized finite posterior for a hypothetical observation (Eqs.
        26–32).
        """
        b = ExactBelief(self.compiled, self.code, self.arm.name, self.kernel)
        b._normalize(mass)
        b.last_observation = obs
        return b

    def roots(self):
        """
        Integrate initial capacity priors and the first observation law (Eqs. 26–32).
        """
        groups = {}
        for prior, ks in self.kernel.initial_distribution(self.focal):
            for p, obs in self.kernel.observation_distribution(
                ks, self.focal, self.arm
            ):
                if not p:
                    continue
                key = canonical_json(obs)
                if key not in groups:
                    groups[key] = (obs, defaultdict(float))
                b = ExactBelief(self.compiled, self.code, self.arm.name, self.kernel)
                groups[key][1][b._hidden(ks)] += prior * p
        return [
            (sum(m.values()), DPNode(obs, self.belief(m, obs)))
            for obs, m in groups.values()
        ]

    def terminal(self, node):
        """
        Integrate terminal service/share reward directly from active count ledgers (Eqs.
        26–32).
        """
        c = self.compiled
        o = self.objective.spec
        markets = set(o.share_markets or [m.id for m in c.spec.markets if m.in_segment])
        total = 0.0
        for h, p in node.belief.mass.items():
            journeys = [
                sum(
                    n
                    for j, n in zip(c.products_of[i], cs.counts)
                    if c.spec.products[j].market in markets
                )
                for i, cs in enumerate(h.kernel.carriers)
            ]
            service = sum(
                c.product_cost[j] * n
                for j, n in zip(
                    c.products_of[self.focal], h.kernel.carriers[self.focal].counts
                )
            )
            total += p * (
                -service / o.profit_scale
                + o.share_weight * journeys[self.focal] / sum(journeys)
                if sum(journeys)
                else -service / o.profit_scale
            )
        return total

    def successors(self, node, index):
        """
        Group exact successors by observable outcome and normalize child beliefs (Eqs.
        26–32).
        """
        key = (self.key(node), index)
        if key in self.children:
            return self.children[key]
        action = CoefficientAction(self.grid[index])
        groups = {}
        for h, p in node.belief.mass.items():
            for q, ks, signal in self.kernel.transition(
                h.kernel, self.focal, action, arm=self.arm, observation=node.observation
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
                groups[k][2][node.belief._hidden(ks)] += p * q
        rows = []
        for obs, cash, mass in groups.values():
            weight = sum(mass.values())
            b = self.belief(mass, obs)
            if self.node_key == "history":
                b.history = node.belief.history + [(node.observation, action, obs)]
            rows.append(
                (weight, cash / self.objective.spec.profit_scale, DPNode(obs, b))
            )
        self.children[key] = rows
        return rows

    def q_values(self, node):
        """
        Compute exact candidate continuation values, without a hidden-state action read
        (Eqs. 26–32).
        """
        if self.finite is not None:
            return self.finite.q_values(
                self.finite.model.project(node.observation, node.belief)
            ) - self.finite.model.existing_service_cost(node.observation)
        if node.t >= self.compiled.spec.horizon.n_periods:
            return np.full(len(self.grid), self.terminal(node))
        key = self.key(node)
        if key in self.cache:
            return self.cache[key]
        if len(self.nodes) >= self.max_nodes:
            raise KernelLimitError(
                f"exact DP exceeds max_nodes={self.max_nodes}; counts={dict(self.counts)}"
            )
        self.nodes[key] = node
        self.counts[node.t] += 1
        if node.t >= self.compiled.spec.horizon.n_periods:
            q = np.full(len(self.grid), self.terminal(node))
        else:
            q = np.array(
                [
                    sum(
                        p * (r + self.value_of(child))
                        for p, r, child in self.successors(node, a)
                    )
                    for a in range(len(self.grid))
                ]
            )
        self.cache[key] = q
        return q

    def value_of(self, node):
        """
        Return the optimized continuation value of one observable belief node (Eqs.
        26–32).
        """
        return float(self.q_values(node).max())

    def solve(self):
        """
        Integrate optimized root values and report actual enumeration counts (Eqs.
        26–32).
        """
        value = sum(p * self.value_of(node) for p, node in self.roots())
        return DPResult(
            value,
            len(self.finite.cache) if self.finite else len(self.nodes),
            dict(self.finite.counts if self.finite else self.counts),
        )

    def policy(self):
        """
        Return an observation-to-action selector using the declared planning state (Eqs.
        26–32).
        """
        return lambda obs, belief: argmax_first(self.q_values(DPNode(obs, belief)))

    def evaluate_policy(self, policy_fn):
        """Integrate the exact expected objective of a supplied policy (Eqs. 26–32)."""
        memo = {}

        def evaluate(node):
            """
            Evaluate the supplied model candidate under the declared objective (Eqs.
            26–32).
            """
            k = self.key(node)
            if k in memo:
                return memo[k]
            if node.t >= self.compiled.spec.horizon.n_periods:
                v = self.terminal(node)
            else:
                v = sum(
                    p * (r + evaluate(child))
                    for p, r, child in self.successors(
                        node, int(policy_fn(node.observation, node.belief))
                    )
                )
            memo[k] = v
            if len(memo) > self.max_nodes:
                raise KernelLimitError("policy evaluation exceeds max_nodes")
            return v

        return sum(p * evaluate(node) for p, node in self.roots())

    def reachable_nodes(self, policy="optimal"):
        """
        Yield decision nodes and their reach probability under the given policy (Eqs.
        26–32).
        """
        frontier = self.roots()
        while frontier:
            next_nodes = []
            for probability, node in frontier:
                if node.t >= self.compiled.spec.horizon.n_periods:
                    continue
                yield node, probability
                action = (
                    argmax_first(self.q_values(node))
                    if policy == "optimal"
                    else int(policy(node.observation, node.belief))
                )
                next_nodes.extend(
                    (probability * p, child)
                    for p, _, child in self.successors(node, action)
                    if probability * p > 0
                )
            frontier = next_nodes


def information_value(dp_factory, arm_a, arm_b):
    """Compare two separately optimized information-arm root values (Eqs. 26–32)."""
    a = dp_factory(arm_a).solve().value
    b = dp_factory(arm_b).solve().value
    return InfoValueReport(a, b, b - a)


def factorization_loss(dp):
    """
    Measure optimal decision loss from repeated feasible marginal compression (Eqs.
    26–32).
    """
    rows = []
    for node, p in dp.reachable_nodes():
        q = dp.q_values(node)
        compressed = DPNode(node.observation, factorize(node.belief))
        fq = dp.q_values(compressed)
        ties = np.flatnonzero(fq >= fq.max() - 1e-9)
        loss = q.max() - q[ties]
        rows.append(
            {
                "t": node.t,
                "reach_probability": p,
                "loss": float(loss[0]),
                "min_over_ties": float(loss.min()),
                "max_over_ties": float(loss.max()),
                "truncated_mass": compressed.belief.truncated_mass,
                "max_marginal_deviation": compressed.belief.max_marginal_deviation,
            }
        )
    total = sum(r["reach_probability"] for r in rows)
    return {
        "mean": sum(r["reach_probability"] * r["loss"] for r in rows) / total
        if total
        else 0.0,
        "max": max((r["loss"] for r in rows), default=0.0),
        "rows": rows,
        "repeated_compression_value": dp.evaluate_policy(
            lambda obs, b: argmax_first(dp.q_values(DPNode(obs, factorize(b))))
        ),
    }


def minimax_regret_marginal_rule(Qc, Qx):
    """Solve the two-prior marginal-only randomized regret diagnostic (Eqs. 26–32)."""
    from scipy.optimize import linprog

    q = np.asarray([Qc, Qx])
    regret = q.max(1)[:, None] - q
    result = linprog(
        [0.0, 0.0, 1.0],
        A_ub=np.c_[regret, -np.ones(2)],
        b_ub=np.zeros(2),
        A_eq=[[1.0, 1.0, 0.0]],
        b_eq=[1.0],
        bounds=[(0, 1), (0, 1), (0, None)],
        method="highs",
    )
    return {"regret": float(result.fun), "mixing_probability": float(result.x[0])}
