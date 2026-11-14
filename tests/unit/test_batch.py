import numpy as np
from airline_market.spec import compile_market
from airline_market.spec.library_builders import toy_hub
from airline_market.sim.engine import MarketEngine
from airline_market.sim.tape import ExogenousTape
from airline_market.sim.batch import rollout_values, supports
from airline_market.core.types import CoefficientAction


def test_batch_reference_every_candidate_scenario():
    c = compile_market(toy_hub())
    grid = c.spec.pricing[0].theta_grid
    tail = grid[5]
    for t in (0, 3, 8):
        initial = []
        for seed in range(30):
            e = MarketEngine(c)
            e.reset(seed)
            for _ in range(t):
                e.step({"A": CoefficientAction(grid[8])})
            e.tape = ExogenousTape.generate(c, seed + 1000)
            initial.append(e)
        assert supports(c, 0, initial)
        values = rollout_values(c, 0, initial, grid, tail)
        for k, theta in enumerate(grid):
            for u, start in enumerate(initial):
                e = start.branch()
                e.step({"A": CoefficientAction(theta)})
                while not e.terminal():
                    e.step({"A": CoefficientAction(tail)})
                assert values[k, u] == e.metrics()["objective"]
