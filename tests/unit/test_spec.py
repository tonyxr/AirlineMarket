import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from airline_market.core.errors import SpecError
from airline_market.core.hashing import spec_hash, jsonable
from airline_market.spec import load_spec, dump_spec, compile_market, apply_overrides
from airline_market.spec.library_builders import toy_hub, cx_example
from airline_market.spec.information import derive_information_arms
from airline_market.spec.schedule import PeriodSchedule
from airline_market.spec.validate import validate_exact_compatible


def test_spec_roundtrip_and_indices(tmp_path):
    s = toy_hub()
    path = tmp_path / "spec.json"
    dump_spec(s, path)
    assert spec_hash(load_spec(path)) == spec_hash(s)
    c = compile_market(s)
    np.testing.assert_array_equal(c.A[0], [[1, 1], [0, 1]])
    np.testing.assert_array_equal(c.A[1], [[1, 0, 1], [0, 1, 1]])
    assert not c.A_global.flags.writeable
    assert tuple(c.product_index) == ("A_OH", "A_OD", "B_OH", "B_HD", "B_OD")
    validate_exact_compatible(s)
    with pytest.raises(TypeError):
        s.metadata["new"] = 1


def test_schedule():
    s = PeriodSchedule.parse({"piecewise": [[0, [0.8, 0.2]], [4, [0.3, 0.7]]]})
    assert s.value_at(3) == (0.8, 0.2)
    assert s.value_at(4) == (0.3, 0.7)
    assert s.as_array(6).shape == (6, 2)
    with pytest.raises(ValueError):
        PeriodSchedule((0, 0), (1, 2))


def test_derived_arms():
    s = toy_hub()
    arms = derive_information_arms(compile_market(s), "A")
    for name, arm in arms.items():
        original = next(a for a in s.information if a.name == name)
        assert arm.requests == original.requests
        assert arm.privileged == original.privileged


def test_config_overrides():
    cases = dict(
        horizon=12,
        own_capacity=(3, 4),
        rival_oh_capacity=3,
        rival_hd_capacities=(1, 2, 3),
        arrival_probability=0.8,
        holiday_multiplier=1.1,
        capacity_shock_probability=0.3,
        shock_step=3,
        disruption_penalty=17.0,
        feeder_premium=20.0,
        rival_response=0.2,
        share_weight=0.25,
        profit_scale=500.0,
    )
    for key, value in cases.items():
        s = apply_overrides(toy_hub(), {key: value})
        assert s.metadata["legacy_config"][key] == value
    s = apply_overrides(toy_hub(), {"objectives.0.share_weight": 0.5})
    assert s.objectives[0].share_weight == 0.5
    with pytest.raises(SpecError):
        apply_overrides(s, {"not.a.path": 1})


@pytest.mark.parametrize("case", range(13))
def test_validation_paths(case):
    s = jsonable(toy_hub())
    if case == 0:
        s["legs"][1]["id"] = s["legs"][0]["id"]
    elif case == 1:
        s["products"][0]["legs"] = ["absent"]
    elif case == 2:
        s["products"][1]["legs"] = ["A_HD", "A_OH"]
    elif case == 3:
        s["legs"][0]["capacity"]["probabilities"] = [0.3]
    elif case == 4:
        s["products"][0]["reference_fare"] = 99999
    elif case == 5:
        s["demand"]["market_weights"]["values"][0] = [0.4, 0.4]
    elif case == 6:
        s["segments"][0]["classes"][0]["alpha"] = 0
    elif case == 7:
        s["exogenous"]["initial"] = [0.5, 0.2]
    elif case == 8:
        s["capacity_process"]["events"][0]["effects"] = [["absent", -1]]
    elif case == 9:
        s["timing"]["mode"] = "simultaneous"
    elif case == 10:
        s["pricing"][0]["theta_grid"] = [[999, 999, 999]]
    elif case == 11:
        s["objectives"][0]["profit_scale"] = 0
    elif case == 12:
        s["information"][0]["requests"] = ["absent"]
    with pytest.raises(SpecError) as exc:
        load_spec(s)
    assert str(exc.value).startswith("$.")


def test_cx_priors():
    c, x = cx_example("C"), cx_example("X")
    assert c.capacity_priors[0].support == ((1, 1), (2, 2))
    assert x.capacity_priors[0].support == ((1, 2), (2, 1))
    assert c.demand.arrival_probability_cap == 1
