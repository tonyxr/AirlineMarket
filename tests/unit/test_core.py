import copy
import json
import numpy as np
from airline_market.core.hashing import canonical_json, spec_hash
from airline_market.core.money import round_fare, clip_fare
from airline_market.core.rng import StreamRegistry
from airline_market.core.snapshot import deep_freeze
from airline_market.vendor.ride_response.cem import CEMConfig, CEMOptimizer
from airline_market.vendor.ride_response.strategic import solve_tariff_game


def test_money_identity():
    for x in np.random.default_rng(71).uniform(-1000, 1000, 10000):
        assert round_fare(x) == round(float(x), 2)
    assert clip_fare(2, 3, 5) == 3


def test_hash_and_freeze():
    obj = deep_freeze({"b": [1, 2], "a": {"z": 3}})
    assert canonical_json(obj) == '{"a":{"z":3},"b":[1,2]}'
    assert spec_hash(obj) == spec_hash(json.loads(canonical_json(obj)))
    clone = copy.deepcopy(obj)
    assert clone["b"] == (1, 2)
    try:
        clone["a"]["z"] = 4
        raise AssertionError("spec mapping was mutable")
    except TypeError:
        pass


def test_named_streams():
    a = StreamRegistry(11, "exo").generator(2).random(1000)
    b = StreamRegistry(11, "exo").generator(2).random(1000)
    c = StreamRegistry(11, "plan").generator(2).random(1000)
    assert np.array_equal(a, b)
    assert not np.any(a == c)
    assert StreamRegistry(11, "plan").child_seed(2) == StreamRegistry(
        11, "plan"
    ).child_seed(2)


def test_vendored_cem_quadratic():
    opt = CEMOptimizer(2, CEMConfig(population=64, iterations=15, seed=4))
    result = opt.optimize(lambda x: -float(((x - [1, -1]) ** 2).sum()))
    assert np.max(np.abs(result["coordinates"] - [1, -1])) < 0.03


def test_vendored_game():
    first = [[3, 1, 0], [4, 2, 1], [5, 3, 2]]
    second = [[3, 4, 5], [1, 2, 3], [0, 1, 2]]
    result = solve_tariff_game([0, 1, 2], first, second)
    assert result["selected_static_nash"]["firm1_multiplier"] == 2
    assert result["selected_static_nash"]["firm2_multiplier"] == 2
