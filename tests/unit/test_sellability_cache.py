"""Availability-law, mutation ownership, shock and restore comparisons."""

import numpy as np
import pytest

from airline_market.sim import ledger
from airline_market.sim.sellability_cache import BinarySellabilityCache, cached_sellability
from airline_market.core.rng import StreamRegistry
from airline_market.campaign.instances import build_instance
from airline_market.spec import compile_market
from airline_market.sim.engine import MarketEngine
from airline_market.agents.fixed import FixedCoefficientController


def test_binary_cache_matches_dense_values_and_returns_fresh_arrays():
    rng = StreamRegistry(200010, "validation/sellability_cache").generator()
    matrix = rng.integers(0, 2, size=(7, 17), dtype=np.int8)
    matrix.setflags(write=False)
    cache = BinarySellabilityCache(max_entries=8)
    for _ in range(100):
        stock = rng.choice(np.array([-1, 0, 0.5, 1, 3]), size=7)
        expected = ledger.sellable(stock, matrix)
        actual = cache(stock, matrix)
        np.testing.assert_array_equal(actual, expected)
        actual[:] = False
        np.testing.assert_array_equal(cache(stock, matrix), expected)
    assert cache.info()["entries"] <= 8


def test_mutable_or_nonbinary_incidence_is_not_cached():
    cache = BinarySellabilityCache()
    with pytest.raises(ValueError, match="immutable binary"):
        cache([2], np.array([[1]], dtype=np.int8))
    invalid = np.array([[2]], dtype=np.int8)
    invalid.setflags(write=False)
    with pytest.raises(ValueError, match="immutable binary"):
        cache([2], invalid)


def test_cache_preserves_complete_paths_after_capacity_shock_and_restore():
    c = compile_market(build_instance("hub_feeder_family", {
        "T": 4, "n_spokes": 2, "capacity": 2, "capacity_support": [1, 2]}))
    dim = len(c.spec.pricing[0].controllable_features)
    def engine():
        """Use the same permitted fixed coefficient controller in both executions."""
        return MarketEngine(c, {"A": FixedCoefficientController((0,) * dim)})
    def finish(e):
        """Apply the same observed capacity intervention, then settle whole itineraries."""
        e.world.carriers[1].capacity[0] = 0
        ledger.resolve(e.world, 1, c)
        while not e.terminal():
            e.step()
        return e.metrics()
    a, b = engine(), engine()
    a.reset(200011); a.step()
    expected = finish(a)
    with cached_sellability() as cache:
        b.reset(200011); b.step()
        snapshot = b.snapshot()
        assert finish(b) == expected
        assert b.trace() == a.trace()
        b.restore(snapshot)
        assert finish(b) == expected
        assert cache.info()["hits"] > 0


def test_process_local_aliases_restore_on_failure():
    original = ledger.sellable
    with pytest.raises(RuntimeError):
        with cached_sellability():
            assert ledger.sellable is not original
            raise RuntimeError("declared failure")
    assert ledger.sellable is original
