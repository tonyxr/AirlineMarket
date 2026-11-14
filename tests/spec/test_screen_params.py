"""R5: compatibility and physical/utility semantics, independent local fixtures."""

import importlib.util
import os

import numpy as np
import pytest

from airline_market.core.hashing import canonical_json
from airline_market.pipeline.generators import hub_feeder_family as original
from airline_market.sim.choice import systematic_utilities
from airline_market.spec import compile_market

path = os.environ.get("R5_GENERATOR_MODULE")
if path:
    s = importlib.util.spec_from_file_location("screen_generators_staged", path)
    module = importlib.util.module_from_spec(s)
    s.loader.exec_module(module)
else:
    from airline_market.pipeline import screen_generators as module


@pytest.mark.parametrize(
    "p",
    [
        {},
        {"periods": 5, "n_spokes": 2, "capacity": 3, "capacity_support": [2, 3]},
        {"periods": 4, "capacity": 2, "bottleneck": "both", "lag_periods": 1},
        {
            "periods": 6,
            "capacity": 4,
            "capacity_support": [1, 2, 3],
            "connecting_share": 0.7,
        },
    ],
)
def test_default_bytes_identical(p):
    expected = canonical_json(original(p))
    explicit = {
        **p,
        "price_scale": 1.0,
        "outside_utility_shift": 0.0,
        "capacity_focal": p.get("capacity", 3),
        "capacity_rival": p.get("capacity", 3),
    }
    assert canonical_json(module.hub_feeder_family(p)) == expected
    assert canonical_json(module.hub_feeder_family(explicit)) == expected


def test_price_and_outside_change_native_utilities_as_declared():
    a = compile_market(original({"n_spokes": 1}))
    b = compile_market(
        module.hub_feeder_family(
            {"n_spokes": 1, "price_scale": 2.0, "outside_utility_shift": 1.3}
        )
    )
    prices = np.full(len(a.spec.products), 100.0)
    active = np.ones(len(prices), dtype=bool)
    for old, new in zip(a.spec.segments, b.spec.segments):
        x, y = old.classes[0], new.classes[0]
        u = systematic_utilities(a, 0, x, prices, active)
        v = systematic_utilities(b, 0, y, prices, active)
        np.testing.assert_allclose(
            v[:-1] - u[:-1], -1.3 - x.alpha * 100, rtol=0, atol=1e-14
        )
        assert u[-1] == v[-1] == 0
    assert a.spec.pricing == b.spec.pricing


def test_capacity_overrides_leave_random_prior_explicit():
    spec = module.hub_feeder_family(
        {
            "n_spokes": 1,
            "capacity": 3,
            "capacity_support": [1, 3],
            "capacity_focal": 5,
            "capacity_rival": 6,
        }
    )
    assert {leg.capacity.support for leg in spec.legs if leg.carrier == "A"} == {(5,)}
    assert {leg.capacity.support for leg in spec.legs if leg.carrier == "B"} == {
        (6,),
        (1, 3),
    }


@pytest.mark.parametrize(
    "p",
    [
        {"price_scale": 0},
        {"price_scale": float("inf")},
        {"outside_utility_shift": float("nan")},
        {"capacity_focal": 2.5},
    ],
)
def test_invalid_parameters_fail(p):
    with pytest.raises(ValueError):
        module.hub_feeder_family(p)


def test_screen_grid_is_original_nested_order():
    x = module.screen_spec({"T": 5, "actions": 4, "n_spokes": 1})
    assert x.horizon.n_periods == 5
    assert len(x.pricing[0].theta_grid) == 4


def test_diagnostics_zero_arrivals_have_zero_load_and_spill():
    path = os.environ.get("R5_DIAGNOSTICS_MODULE")
    if path:
        s = importlib.util.spec_from_file_location("screen_diagnostics_staged", path)
        diag = importlib.util.module_from_spec(s)
        s.loader.exec_module(diag)
    else:
        from airline_market.experiments import screen_diagnostics as diag
    from airline_market.solvers.lagged_dp import history_dp

    dp = history_dp(
        compile_market(
            module.screen_spec(
                {
                    "n_spokes": 1,
                    "T": 2,
                    "actions": 2,
                    "arrival_probability": 0.0,
                    "price_scale": 0.8,
                }
            )
        ),
        "A",
        "direct",
        conditional_tail=False,
    )
    dp.solve()
    r = diag.exact_diagnostics(dp)
    assert r["load_factor_A"] == r["load_factor_B"] == 0
    assert r["spill"] == r["revenue_A"] == r["revenue_B"] == 0
    assert not r["plausible"]
    records = []
    same = diag.exact_diagnostics(
        dp, policy_sink=lambda node, index: records.append((node, index))
    )
    assert same == r
    assert records and all(node[0] < 2 for node, _ in records)


def test_dsd_linear_effects_are_orthogonal_to_second_order_terms():
    if os.environ.get("R5_DESIGN_MODULE"):
        s = importlib.util.spec_from_file_location(
            "screen_design", os.environ["R5_DESIGN_MODULE"]
        )
        d = importlib.util.module_from_spec(s)
        s.loader.exec_module(d)
    else:
        from airline_market.experiments import screen_design as d
    x = d.dsd_matrix()
    assert x.shape == (13, 5)
    np.testing.assert_array_equal(x.T @ x, 10 * np.eye(5, dtype=int))
    for i in range(5):
        for j in range(i, 5):
            np.testing.assert_array_equal(
                x.T @ (x[:, i] * x[:, j]), np.zeros(5, dtype=int)
            )
    rows = d.round_one()
    assert len(rows) == 52 and len({r["setting_id"] for r in rows}) == 52
