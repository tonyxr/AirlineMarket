"""Poisson oligopoly construction, actor scope and replay checks for F3."""

from dataclasses import replace
import numpy as np
import pytest

from airline_market.pipeline.scaling import scaling_network
from airline_market.agents.random_coefficient import RandomCoefficientController
from airline_market.spec import compile_market, load_spec
from airline_market.core.hashing import jsonable, spec_hash
from airline_market.sim.engine import MarketEngine


@pytest.mark.parametrize("carriers", [2, 3, 4])
def test_poisson_oligopoly_incidence_intensity_and_scoped_requests(carriers):
    spec = scaling_network({"n_spokes": 2, "n_carriers": carriers, "periods": 3})
    c = compile_market(spec)
    assert len(spec.carriers) == carriers
    assert len(spec.legs) == 4 * carriers
    assert len(spec.products) == 6 + 8 * (carriers - 1)
    assert np.array_equal(c.A_global.sum(axis=0), [len(p.legs) for p in spec.products])
    assert spec.demand.mode == "poisson"
    expected = (
        sum(schedule.value_at(0) for schedule in spec.demand.intensity.values()) * 3
    )
    assert expected == pytest.approx(
        spec.metadata["assumptions"]["expected_requests_per_cohort"]
    )
    requests = {q.id: q for q in spec.quotes.requests}
    assert len(requests) == len(spec.quotes.requests)
    for arm in spec.information:
        assert all(requests[q].observer == arm.observer for q in arm.requests)
        assert all(requests[q].lag_periods == 1 for q in arm.requests)
    focal = {a.name: a for a in spec.information if a.observer == "A"}
    assert len(focal["linked"].requests) - len(focal["direct"].requests) == 2 * (
        carriers - 1
    )
    assert len(focal["unrelated"].requests) == len(focal["linked"].requests)
    assert all(
        value == 0 for value in spec.metadata["placebo_shortfall_by_observer"].values()
    )
    assert spec_hash(load_spec(jsonable(spec))) == spec_hash(spec)


def test_random_coefficients_replay_and_remain_in_observable_bounds():
    c = compile_market(
        scaling_network(
            {
                "n_spokes": 2,
                "n_carriers": 3,
                "periods": 4,
                "controller_mode": "external_all",
            }
        )
    )
    engine = MarketEngine(
        c, {carrier.code: RandomCoefficientController() for carrier in c.spec.carriers}
    )
    engine.reset(200000)
    engine.step()
    snapshot = engine.snapshot()
    first = engine.step()
    prices = [state.posted_prices.copy() for state in engine.world.carriers]
    metrics = engine.metrics()
    engine.restore(snapshot)
    second = engine.step()
    assert first == second and engine.metrics() == metrics
    for actual, expected in zip(engine.world.carriers, prices):
        np.testing.assert_array_equal(actual.posted_prices, expected)
    for carrier, pricing in enumerate(c.spec.pricing):
        theta = engine.world.carriers[carrier].theta
        assert all(
            lo <= value <= hi for value, (lo, hi) in zip(theta, pricing.theta_bounds)
        )


def test_invalid_scaling_design_is_not_silently_changed():
    with pytest.raises(ValueError, match="2–4 carriers"):
        scaling_network({"n_carriers": 1})
    with pytest.raises(ValueError, match="unknown scaling"):
        scaling_network({"carrires": 3})
