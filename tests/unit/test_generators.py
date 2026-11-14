import time
from airline_market.pipeline.generators import (
    hub_feeder_family,
    independent_legs,
    grid_network,
)
from airline_market.spec import compile_market
from airline_market.core.hashing import spec_hash
from airline_market.sim.engine import MarketEngine
from airline_market.core.types import CoefficientAction


def test_generator_factor_levels_and_scale():
    base = hub_feeder_family()
    assert spec_hash(base) == spec_hash(hub_feeder_family())
    for params in (
        [{"connecting_share": x} for x in [0.1, 0.3, 0.5, 0.7]]
        + [{"bottleneck": x} for x in ["focal", "rival", "both"]]
        + [{"lag_periods": x} for x in [0, 1, 2]]
        + [{"fare_bin": x} for x in [0, 10, 25]]
        + [{"rival_controller": x} for x in ["matching", "bid_price_dlp"]]
        + [{"availability": "seats_capped"}, {"demand_factor": "latent"}]
    ):
        compile_market(hub_feeder_family(params))
    s = independent_legs()
    arms = {a.name: a for a in s.information}
    assert arms["direct"].requests == arms["linked"].requests
    c = compile_market(grid_network(20, periods=3))
    e = MarketEngine(c)
    e.reset(1)
    started = time.monotonic()
    while not e.terminal():
        e.step({"A": CoefficientAction((0.0, 0.0, 0.0))})
    assert time.monotonic() - started < 2.0
