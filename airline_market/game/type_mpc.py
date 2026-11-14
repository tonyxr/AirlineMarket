"""Persistent joint-type filtering and sampled fixed-tail MPC (Eqs. 26–29).

This new path uses the unchanged full-ledger type kernel for Bayesian updates.
It does not make old kernels compatible with history/mixture controllers.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from airline_market.core.errors import KernelLimitError, ZeroLikelihoodError
from airline_market.core.hashing import canonical_json
from airline_market.core.rng import StreamRegistry
from airline_market.core.types import CoefficientAction, PriceAction
from airline_market.game.type_psro_keyed import EpisodeTypeDP
from airline_market.sim.accounting import carrier_metrics
from airline_market.sim.tape import ExogenousTape
from airline_market.solvers.mpc import MPCDecision


class JointTypeFilter:
    """Condition every positive joint (episode type, full state) mass, Eq. 29."""

    def __init__(self, compiled, focal, arm, types, probabilities, max_support=20000):
        if compiled.spec.demand.mode != "single_request_bernoulli":
            raise ValueError("this joint-type MPC precursor requires Bernoulli demand")
        self.model = EpisodeTypeDP(compiled, focal, arm, types, probabilities)
        self.max_support = int(max_support)
        if self.max_support < 1:
            raise ValueError("joint support limit must be positive")
        self.node = None

    def _accept(self, nodes):
        if len(nodes) != 1:
            raise ZeroLikelihoodError(
                "observation has no unique positive joint-type law"
            )
        node = nodes[0]
        if len(node.mass) > self.max_support:
            raise KernelLimitError("joint-type filtering exceeds max_support")
        self.node = node
        # Cached successor trees are disposable; every current posterior row stays.
        self.model.children.clear()

    def reset(self, observation):
        if observation.t != 0:
            raise ValueError("joint-type reset requires the initial observation")
        key = canonical_json(observation)
        self._accept(
            [
                n
                for p, n in self.model.roots()
                if p and canonical_json(n.observation) == key
            ]
        )

    def update(self, action_index, observation):
        if self.node is None or observation.t != self.node.observation.t + 1:
            raise ValueError("joint-type observations must be consecutive")
        if not 0 <= action_index < len(self.model.grid):
            raise ValueError("action outside registered grid")
        key = canonical_json(observation)
        self._accept(
            [
                n
                for p, _, n in self.model.successors(self.node, action_index)
                if p and canonical_json(n.observation) == key
            ]
        )

    def snapshot(self):
        return deepcopy(self.node)

    def restore(self, node):
        self.node = deepcopy(node)
        self.model.children.clear()


@dataclass(frozen=True)
class TypeMPCConfig:
    """Declared sampled one-step/fixed-tail class, not an exact BR certificate."""

    n_scenarios: int = 32
    n_reporting: int = 32
    tail_index: int = 0
    switch_standard_errors: float = 2.0
    switch_absolute_margin: float = 0.0

    def __post_init__(self):
        if self.n_scenarios < 2 or self.n_reporting < 2:
            raise ValueError(
                "paired MPC needs at least two selection/reporting scenarios"
            )
        if self.switch_standard_errors < 0 or self.switch_absolute_margin < 0:
            raise ValueError("switching margins must be nonnegative")


@dataclass
class TypeScenario:
    """Analyst planning object; type identity never becomes an actor input."""

    kind: int
    engine: object
    forecast_seed: int


class EpisodeTypeMPC:
    """Average candidate values under the joint posterior before maximization."""

    def __init__(self, belief, config=None):
        self.belief = belief
        self.model = belief.model
        self.compiled = self.model.compiled
        self.focal = self.model.focal
        self.config = config or TypeMPCConfig()
        self.pricing = next(
            p
            for p in self.compiled.spec.pricing
            if p.carrier == self.compiled.spec.carriers[self.focal].code
        )
        if not 0 <= self.config.tail_index < len(self.model.grid):
            raise ValueError("tail action outside registered grid")
        self.last_decision = None

    def scenarios(self, count, episode_index=0, family=0):
        """Indexed CRN pairs; one type remains attached to its complete forecast."""
        node = self.belief.node
        if node is None:
            raise ValueError("joint posterior has not been initialized")
        rows = list(node.mass)
        weights = np.array(list(node.mass.values()), dtype=float)
        weights /= weights.sum()
        result = []
        for index in range(count):
            rng = StreamRegistry(773, "plan/F7/joint-type-mpc-v1").generator(
                episode_index, node.observation.t, family, index
            )
            kind, state = rows[int(rng.choice(len(rows), p=weights))]
            seed = int(rng.integers(2**63))
            engine = self.model.kernels[kind].engine(state, self.focal)
            engine.tape = ExogenousTape.generate(self.compiled, seed)
            result.append(TypeScenario(kind, engine, seed))
        return result

    def scores(self, indices, scenarios):
        """Native branches with endogenous rival reactions and common first action."""
        obs = self.belief.node.observation
        values = np.empty((len(indices), len(scenarios)))
        for row, index in enumerate(indices):
            for column, scenario in enumerate(scenarios):
                engine = scenario.engine.branch()
                while not engine.terminal():
                    first = engine.world.t == obs.t
                    current = (
                        obs if first else engine.observe(self.focal, self.model.arm)
                    )
                    action = index if first else self.config.tail_index
                    old = engine.world.carriers[self.focal].theta
                    theta = (
                        old
                        if old is not None and current.t % self.pricing.update_every
                        else self.model.grid[action]
                    )
                    # Explicit own observation also fixes the first-stage quote map.
                    prices = engine.rules[self.focal].prices(theta, current)
                    engine.world.carriers[self.focal].theta = theta
                    engine.step(
                        {
                            self.focal: PriceAction(
                                dict(zip(self.compiled.products_of[self.focal], prices))
                            )
                        }
                    )
                values[row, column] = carrier_metrics(
                    engine.world, self.compiled, self.focal
                )["objective"]
        return values

    def choose(self, episode_index=0):
        """Protected common-scenario choice with independent reporting, Eq. 26."""
        if self.belief.node.observation.t >= self.compiled.spec.horizon.n_periods:
            raise ValueError("no MPC action at a terminal observation")
        cfg = self.config
        values = self.scores(
            range(len(self.model.grid)),
            self.scenarios(cfg.n_scenarios, episode_index, 0),
        )
        means = values.mean(axis=1)
        best = int(means.argmax())
        difference = values[best] - values[cfg.tail_index]
        paired_se = float(difference.std(ddof=1) / np.sqrt(len(difference)))
        threshold = cfg.switch_absolute_margin + cfg.switch_standard_errors * paired_se
        if float(difference.mean()) <= threshold:
            best = cfg.tail_index
        held = self.scores(
            [best, cfg.tail_index], self.scenarios(cfg.n_reporting, episode_index, 1)
        )
        self.last_decision = MPCDecision(
            CoefficientAction(self.model.grid[best]),
            float(held[0].mean()),
            float(held[0].std(ddof=1) / np.sqrt(cfg.n_reporting)),
            tuple(map(float, means)),
            cfg.n_scenarios,
            {
                "selected_index": best,
                "tail_index": cfg.tail_index,
                "paired_selection_se": paired_se,
                "switch_threshold": threshold,
                "independent_selected_minus_tail": float((held[0] - held[1]).mean()),
                "reporting_scenarios": cfg.n_reporting,
                "type_probabilities": self.belief.node.type_probabilities(
                    len(self.model.kernels)
                ).tolist(),
                "scope": "Persistent joint-type one-step/fixed-tail sampled MPC; paired switch is heuristic, not a BR/Nash certificate.",
            },
        )
        return self.last_decision


class EpisodeTypeMPCController:
    """Online observation-only MPC response to a declared persistent type bank."""

    exact_compatible = False
    policy_class = "persistent_joint_type_sampled_mpc_v1"

    def __init__(
        self,
        focal,
        arm,
        types,
        probabilities,
        config=None,
        *,
        episode_index=0,
        max_support=20000,
    ):
        self.focal_code, self.arm_name = focal, arm
        self.types, self.probabilities = types, probabilities
        self.config, self.episode_index, self.max_support = (
            config,
            episode_index,
            max_support,
        )

    def reset(self, ctx, rng):
        if ctx.compiled.spec.carriers[ctx.carrier].code != self.focal_code:
            raise ValueError("MPC controller bound to the wrong carrier")
        self.belief = JointTypeFilter(
            ctx.compiled,
            self.focal_code,
            self.arm_name,
            self.types,
            self.probabilities,
            self.max_support,
        )
        self.planner = EpisodeTypeMPC(self.belief, self.config)
        self.previous_index = None
        self.decisions = []

    def act(self, obs):
        if self.belief.node is None:
            self.belief.reset(obs)
        else:
            self.belief.update(self.previous_index, obs)
        decision = self.planner.choose(self.episode_index)
        self.previous_index = decision.diagnostics["selected_index"]
        self.decisions.append(
            {
                "period": obs.t,
                "posterior_support": len(self.belief.node.mass),
                **decision.diagnostics,
            }
        )
        return decision.action

    def observe_outcome(self, *args):
        pass

    def memory(self):
        return self.previous_index, self.belief.model.key(
            self.belief.node
        ) if self.belief.node else None

    def snapshot(self):
        return (
            self.belief.snapshot(),
            self.previous_index,
            deepcopy(self.decisions),
            deepcopy(self.planner.last_decision),
        )

    def restore(self, state):
        node, self.previous_index, self.decisions, self.planner.last_decision = (
            deepcopy(state)
        )
        self.belief.restore(node)
