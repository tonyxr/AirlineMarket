from dataclasses import replace
from itertools import product
import numpy as np
import pytest
from airline_market.spec import compile_market
from airline_market.spec.library_builders import toy_hub
from airline_market.spec.model import InformationArmSpec, ControllerAssignment
from airline_market.core.errors import InformationLeakError
from airline_market.core.types import CoefficientAction
from airline_market.legacy_toy.model import Model, RivalState, COEFFICIENT_GRID, Quote
from airline_market.sim.engine import MarketEngine
from airline_market.agents.price_rule import AdditivePriceRule
from airline_market.agents.bidprice import solve_dlp, DLPBidPriceController
from airline_market.agents.emsr import emsr_b_protection
from airline_market.agents import make_controller


def test_price_basis_matches_legacy_grid():
    c = compile_market(toy_hub())
    engine = MarketEngine(c)
    engine.reset(3)
    obs = engine.observe("A", "direct")
    rule = AdditivePriceRule(
        c, c.spec.pricing[0], next(a for a in c.spec.information if a.name == "direct")
    )
    old = Model()
    for t in range(10):
        for stock in product(range(3), repeat=2):
            view = replace(obs, t=t, own_inventory=stock)
            quotes = [
                Quote(c.spec.products[q.product].market, q.fare, q.available)
                for q in view.quotes
            ]
            for theta in COEFFICIENT_GRID:
                assert tuple(rule.prices(theta, view)) == old.prices(
                    theta, t, stock, quotes
                )


def test_rule_matches_legacy_reachable_states():
    c = compile_market(toy_hub())
    engine = MarketEngine(c)
    engine.reset(7)
    old = Model()
    for hd in (1, 3):
        for n in product(range(3), range(hd + 1), range(3)):
            state = RivalState((2, hd), n)
            if min(state.inventory) < 0:
                continue
            obs = engine.observe("B", "B_direct")
            for previous in ((70.0, 125.0), (110.0, 185.0), (160.0, 250.0)):
                q = list(obs.quotes)
                q = [replace(x, fare=previous[k]) for k, x in enumerate(q)]
                view = replace(obs, own_inventory=state.inventory, quotes=tuple(q))
                prices = engine.controllers[1].act(view).prices
                assert tuple(prices[j] for j in c.products_of[1]) == tuple(
                    q.fare for q in old.rival_quotes(state, previous)
                )


def test_feature_information_and_baseline_solutions():
    c = compile_market(toy_hub())
    with pytest.raises(InformationLeakError):
        AdditivePriceRule(
            c,
            c.spec.pricing[0],
            InformationArmSpec(name="empty", observer="A", requests=()),
        )
    assert emsr_b_protection([100.0, 50.0], [10.0, 10.0], [2.0, 2.0]) == (0.0, 10.0)
    quantities, duals, value = solve_dlp([[1, 1]], [1], [100, 50], [2, 2])
    assert abs(value - 100) < 1e-9 and abs(duals[0] - 100) < 1e-9


@pytest.mark.parametrize(
    "key,params",
    [
        ("fixed", {"theta": [0, 0, 15]}),
        ("schedule", {"thetas_by_period": [[0, 0, 0], [30, 30, 15]]}),
        ("matching", {"offset": -1}),
        ("emsr_b", {}),
        ("bid_price_dlp", {}),
        ("myopic_br", {}),
    ],
)
def test_controllers_run_from_observations(key, params):
    s = toy_hub()
    assignment = ControllerAssignment(carrier="A", controller=key, params=params)
    s = replace(s, controllers=(assignment, s.controllers[1]))
    e = MarketEngine(compile_market(s))
    e.reset(8)
    while not e.terminal():
        e.step()
    assert np.isfinite(e.metrics()["profit"])
    e.reset(8)
    if key == "bid_price_dlp":
        assert e.controllers[0].cached is None
