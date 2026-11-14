import numpy as np
import pytest
from airline_market.core.types import CoefficientAction
from airline_market.legacy_toy.model import (
    AirlineEnv,
    Config,
    Model,
    NoiseTape,
    COEFFICIENT_GRID,
)
from airline_market.spec import compile_market
from airline_market.spec.library_builders import toy_hub
from airline_market.sim.engine import MarketEngine
from airline_market.sim.tape import LegacyToyTapeAdapter


@pytest.mark.parametrize("shock", [0.0, 0.5, 1.0])
@pytest.mark.parametrize("holiday", [1.0, 1.1])
def test_engine_matches_legacy_500_seeds(shock, holiday):
    cfg = Config(capacity_shock_probability=shock, holiday_multiplier=holiday)
    compiled = compile_market(toy_hub(cfg))
    new = MarketEngine(compiled)
    old = AirlineEnv(Model(cfg))
    actions = np.random.default_rng(892).integers(18, size=(500, 10))
    for seed in range(500):
        old.reset(seed)
        new.reset(
            tape=LegacyToyTapeAdapter.from_legacy(
                NoiseTape.from_seed(seed, 10), compiled
            )
        )
        for index in actions[seed]:
            a = COEFFICIENT_GRID[int(index)]
            old.step(a)
            new.step({"A": CoefficientAction(a)})
            for i in (0, 1):
                state = new.world.carriers[i]
                assert [
                    (b.product - (2 if i else 0), b.fare) for b in state.bookings
                ] == [(b.product, b.fare) for b in old.state.bookings[i]]
                assert state.cash == old.state.cash[i]
                assert tuple(state.capacity) == old.state.capacity[i]
                assert state.refunds == old.state.refunds[i]
                assert state.canceled == old.state.canceled[i]
            assert new.world.exo == int(old.state.weather_event)
            assert new.world.background[0].stock == old.state.reference_inventory
        expected = old.model.metrics(old.state)
        actual = new.metrics()
        for key, value in expected.items():
            assert actual[key] == value
