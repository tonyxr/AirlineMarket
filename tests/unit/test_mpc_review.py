"""Exact Q agreement and sampling incumbent protection regression checks."""

from dataclasses import replace
import numpy as np
from airline_market.spec import compile_market
from airline_market.spec.library_builders import toy_hub
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.belief import ExactBelief
from airline_market.solvers.mpc import ScenarioMPC, MPCConfig
from airline_market.core.types import CoefficientAction
from airline_market.core.rng import StreamRegistry


def test_exact_rollout_choices_agree_with_exact_q_at_every_reached_period():
    c = compile_market(toy_hub())
    tail = tuple(c.spec.pricing[0].theta_grid[-1])
    agreements = []
    for seed in range(8):
        env = FocalCarrierEnv(c, "A", "direct")
        obs, _ = env.reset(seed)
        b = ExactBelief(c, "A", "direct")
        b.reset(obs)
        p = ScenarioMPC(
            c,
            "A",
            "direct",
            b,
            config=MPCConfig(evaluation="exact"),
            tail_coefficients=tail,
        )
        while obs.t < c.spec.horizon.n_periods:
            q = p.exact_evaluator.q_values(obs, b, c.spec.pricing[0].theta_grid)
            decision = p.choose(obs, seed)
            agreements.append(
                decision.action.theta
                == tuple(c.spec.pricing[0].theta_grid[int(np.argmax(q))])
            )
            after, _, _, _ = env.step(decision.action)
            b.update(obs, decision.action, after)
            obs = after
            assert decision.std_error == 0.0 and decision.n_scenarios == 0
    assert all(agreements)


def test_protection_keeps_actual_incumbent_and_uses_independent_report(monkeypatch):
    c = compile_market(toy_hub())
    env = FocalCarrierEnv(c, "A", "direct")
    obs, _ = env.reset(3)
    b = ExactBelief(c, "A", "direct")
    b.reset(obs)
    tail = tuple(c.spec.pricing[0].theta_grid[-1])
    p = ScenarioMPC(
        c,
        "A",
        "direct",
        b,
        config=MPCConfig(
            evaluation="sampling", n_scenarios=4, n_estimation_scenarios=3
        ),
        tail_coefficients=tail,
    )
    stages = []

    def scenarios(o, i, n, stage):
        stages.append(stage)
        return [stage] * n

    def scores(candidates, initial, o, i):
        if initial[0] == 0:
            a = np.zeros((len(candidates), 4))
            a[0] = [-1.0, 1.0, -1.0, 1.1]
            return a
        return np.full((len(candidates), len(initial)), 7.0)

    monkeypatch.setattr(p, "_scenarios", scenarios)
    monkeypatch.setattr(p, "_scores", scores)
    d = p.choose(obs, 0)
    assert d.action.theta == tail and d.estimate == 7.0 and stages == [0, 1]
    assert d.diagnostics["incumbent_retained"]


def test_successive_halving_keeps_paired_incumbent_and_fresh_report(monkeypatch):
    c = compile_market(toy_hub())
    env = FocalCarrierEnv(c, "A", "direct")
    obs, _ = env.reset(200000)
    belief = ExactBelief(c, "A", "direct")
    belief.reset(obs)
    grid = list(c.spec.pricing[0].theta_grid)
    planner = ScenarioMPC(
        c,
        "A",
        "direct",
        belief,
        config=MPCConfig(
            evaluation="sampling",
            selection="racing",
            n_scenarios=256,
            racing_initial_scenarios=64,
            n_estimation_scenarios=17,
        ),
        tail_coefficients=grid[-1],
    )
    stages = []

    def scenarios(o, i, n, stage):
        stages.append((stage, n))
        return [stage] * n

    def scores(candidates, initial, o, i):
        if initial[0] == 1:
            return np.full((len(candidates), len(initial)), 7.0)
        assert grid[-1] in candidates
        return np.asarray(
            [
                np.full(len(initial), 2.0 if theta == grid[0] else 0.0)
                for theta in candidates
            ]
        )

    monkeypatch.setattr(planner, "_scenarios", scenarios)
    monkeypatch.setattr(planner, "_scores", scores)
    decision = planner.choose(obs, 0)
    assert decision.action.theta == grid[0]
    assert decision.estimate == 7.0
    assert stages == [(2, 64), (3, 64), (4, 128), (1, 17)]
    racing = decision.diagnostics["racing"]
    assert racing["candidate_scenario_evaluations"] < len(grid) * 256
    assert (
        racing["samples_by_candidate"][0] == racing["samples_by_candidate"][-1] == 256
    )
