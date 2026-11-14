from dataclasses import replace
import numpy as np
import pytest
from airline_market.spec import compile_market, load_spec
from airline_market.spec.library_builders import toy_hub
from airline_market.core.hashing import jsonable
from airline_market.core.errors import InformationLeakError
from airline_market.core.types import CoefficientAction
from airline_market.sim.engine import MarketEngine
from airline_market.sim.tape import ExogenousTape
from airline_market.sim.arrivals import period_requests
from airline_market.sim.choice import systematic_utilities, sampled_choice


def test_snapshot_and_actor_view():
    c = compile_market(toy_hub())
    engine = MarketEngine(c)
    engine.reset(77)
    obs = engine.observe("A", "linked")
    assert obs.privileged is None
    with pytest.raises(InformationLeakError):
        obs.require_privileged()
    with pytest.raises(InformationLeakError):
        getattr(obs, "world")
    action = {"A": CoefficientAction((0.0, 0.0, 15.0))}
    saved = engine.snapshot()
    engine.step(action)
    result = engine.metrics()
    engine.restore(saved)
    engine.step(action)
    assert engine.metrics() == result
    assert saved.world.t == 0
    assert engine.branch().tape is engine.tape
    with pytest.raises(ValueError):
        engine.tape.gumbel[0, 0, 0] = 0
    assert not any(
        b.taste_class != -1 or b.segment != -1
        for b in engine.observe("A", "direct").own_bookings
    )


def test_local_and_global_alternatives_agree():
    c = compile_market(toy_hub())
    rng = np.random.default_rng(1)
    for m in range(3):
        u = systematic_utilities(
            c,
            m,
            c.spec.segments[0].classes[0],
            np.full(5, 150.0),
            np.ones(5, dtype=bool),
        )
        columns = (*c.market_products[m], 5)
        for g in rng.gumbel(size=(1000, 6)):
            full = np.full(6, -np.inf)
            full[list(columns)] = u
            assert columns[sampled_choice(u, g[list(columns)])] == int(
                np.argmax(full + g)
            )


def test_poisson_thinning_mean():
    data = jsonable(toy_hub())
    data["horizon"]["n_periods"] = 20000
    data["demand"]["mode"] = "poisson"
    data["demand"]["intensity"] = {m: {"constant": 2.0} for m in ("OH", "HD", "OD")}
    data["exogenous"]["states"][1]["attributes"]["disrupted"] = 1
    c = compile_market(load_spec(data))
    tape = ExogenousTape.generate(c, 919)
    values = np.array([len(period_requests(c, tape, t, 1)) for t in range(20000)])
    expected = 6 * 0.7
    assert abs(values.mean() - expected) < 3 * np.sqrt(expected / len(values))


def test_realized_surplus_uses_outside_option_as_baseline():
    from dataclasses import replace
    from airline_market.spec import compile_market
    from airline_market.spec.library_builders import toy_hub
    from airline_market.spec.model import CapacityDistribution
    from airline_market.sim.engine import MarketEngine
    from airline_market.core.types import CoefficientAction

    s = toy_hub()
    zero = CapacityDistribution(support=(0,), probabilities=(1.0,))
    s = replace(s, legs=tuple(replace(l, capacity=zero) for l in s.legs))
    e = MarketEngine(compile_market(s))
    e.reset(17)
    while not e.terminal():
        e.step({"A": CoefficientAction((0.0, 0.0, 0.0))})
    assert e.metrics()["consumer_surplus"] == 0.0
    assert e.metrics()["realized_consumer_surplus"] == 0.0
