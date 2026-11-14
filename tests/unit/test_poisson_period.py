"""A.10/B0/V9: NEW deterministic finite Poisson expectation references, no MC."""

import importlib.util
import json
import math
import os
import pickle
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from airline_market.sim.choice import choice_probabilities, systematic_utilities
from airline_market.sim.segmented_welfare import conditional_logit_surplus
from airline_market.sim.tape import ExogenousTape
from airline_market.spec import compile_market
from tests.unit import test_rationing_native as native

if os.environ.get("POISSON_PERIOD_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_poisson_period", os.environ["POISSON_PERIOD_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.sim import poisson_period as module


def fixture(*, stock=1, rate=2, price=100, segment="leisure"):
    c = native.fixture().compiled
    prices = tuple((p.id, price) for p in c.spec.products)
    capacity = tuple((l.id, stock if l.id == "A_OH0" else 0) for l in c.spec.legs)
    rates = tuple(
        ((m.id, s.id), rate if (m.id, s.id) == ("OH0", segment) else 0)
        for m in c.spec.markets
        for s in c.spec.segments
    )
    return module.NativePoissonPeriod(
        c,
        prices,
        capacity,
        rates,
        "unit declared conditional forecast, no hidden-state inference",
    )


def initial(problem, market="OH0", segment="leisure"):
    c = problem.compiled
    active = np.array(
        [
            all(dict(problem.free_capacity)[k] > 0 for k in p.legs)
            for p in c.spec.products
        ]
    )
    taste = next(s for s in c.spec.segments if s.id == segment).classes[0]
    m = c.market_index[market]
    u = systematic_utilities(
        c,
        m,
        taste,
        np.array([dict(problem.prices)[p.id] for p in c.spec.products]),
        active,
    )
    return choice_probabilities(u), conditional_logit_surplus(u, taste.alpha)


@pytest.mark.parametrize("capacity", [1, 2])
@pytest.mark.parametrize("rate", [0.3, 2, 50])
def test_single_product_full_poisson_closed_form_capacity_depletion_and_CS(
    capacity, rate
):
    p = fixture(stock=capacity, rate=rate)
    q, cs = initial(p)
    q = q[0]
    hazard = rate * q
    expected = (
        -math.expm1(-hazard) if capacity == 1 else 2 - math.exp(-hazard) * (2 + hazard)
    )
    r = module.poisson_period_expectation(p, state_limit=100)
    assert dict(r["expected_product_bookings"])["A_OH0"] == pytest.approx(
        expected, abs=1e-12
    )
    assert dict(r["expected_ordinary_conditional_CS"])[
        ("OH0", "leisure")
    ] == pytest.approx(cs * expected / q, abs=1e-11)
    assert dict(r["expected_gross_ticket_revenue"])["A"] == pytest.approx(
        expected * 100, abs=1e-10
    )
    assert r["expected_ordinary_outside_choices"] == pytest.approx(
        rate - expected, abs=1e-12
    )
    dist = dict(zip(r["states"], r["transition_probabilities"]))
    assert dist[tuple(v for k, v in p.free_capacity)] == pytest.approx(
        math.exp(-hazard), abs=1e-12
    )
    assert abs(r["numerical_probability_mass_residual"]) < 1e-12
    assert all(abs(v) < 1e-11 for k, v in r["numerical_leg_stock_identity_residual"])


def test_two_segments_use_own_alpha_and_share_capacity_not_average_alpha_or_MC():
    p = fixture()
    rates = tuple(
        (
            key,
            2
            if key == ("OH0", "leisure")
            else 0.7
            if key == ("OH0", "business")
            else 0,
        )
        for key, v in p.market_segment_intensities
    )
    p = replace(p, market_segment_intensities=rates)
    ql, cl = initial(p)
    qb, cb = initial(p, segment="business")
    h = 2 * ql[0] + 0.7 * qb[0]
    occupancy = -math.expm1(-h) / h
    r = module.poisson_period_expectation(p, state_limit=20)
    cs = dict(r["expected_ordinary_conditional_CS"])
    assert dict(r["expected_product_bookings"])["A_OH0"] == pytest.approx(
        -math.expm1(-h), abs=1e-12
    )
    assert cs[("OH0", "leisure")] == pytest.approx(2 * cl * occupancy, abs=1e-11)
    assert cs[("OH0", "business")] == pytest.approx(0.7 * cb * occupancy, abs=1e-11)


def test_shared_leg_connection_vs_local_analytic_race_retains_full_transition_law():
    p = fixture()
    c = p.compiled
    p = replace(
        p,
        free_capacity=tuple((l.id, 1 if l.carrier == "A" else 0) for l in c.spec.legs),
        market_segment_intensities=tuple(
            (
                key,
                1
                if key == ("OH0", "leisure")
                else 2
                if key == ("OD0_0", "leisure")
                else 0,
            )
            for key, v in p.market_segment_intensities
        ),
    )
    q1, _c1 = initial(p)
    q2, c2 = initial(p, market="OD0_0")
    h1 = q1[0]
    h2 = 2 * q2[0]
    h = h1 + h2
    integral = -math.expm1(-h) / h
    r = module.poisson_period_expectation(p, state_limit=20)
    b = dict(r["expected_product_bookings"])
    assert b["A_OH0"] == pytest.approx(h1 * integral, abs=1e-12)
    assert b["A_OD0_0"] == pytest.approx(h2 * integral, abs=1e-12)
    assert sum(b.values()) == pytest.approx(-math.expm1(-h), abs=1e-12)
    assert dict(r["expected_ordinary_conditional_CS"])[
        ("OD0_0", "leisure")
    ] == pytest.approx(2 * c2 * integral, abs=1e-11)


@pytest.mark.parametrize("empty", ["demand", "stock"])
def test_explicit_zero_demand_or_stock_has_full_zero_outputs_not_empirical_exposure_inference(
    empty,
):
    p = fixture(rate=0 if empty == "demand" else 2, stock=0 if empty == "stock" else 1)
    r = module.poisson_period_expectation(p, state_limit=1)
    assert r["state_count"] == 1 and r["transition_probabilities"] == (1.0,)
    assert all(v == 0 for k, v in r["expected_product_bookings"]) and all(
        v == 0 for k, v in r["expected_ordinary_conditional_CS"]
    )
    assert r["expected_ordinary_outside_choices"] == (0 if empty == "demand" else 2)


def test_signed_negative_fares_not_clipped_and_no_world_tape_actor_or_global_randomness(
    monkeypatch,
):
    p = fixture(price=-1)
    before = pickle.dumps(np.random.get_state())
    spec = pickle.dumps(p.compiled.spec)

    def forbidden(*args, **kwargs):
        raise AssertionError("no sampler/global random draw")

    monkeypatch.setattr(ExogenousTape, "generate", forbidden)
    for name in [
        "rand",
        "randn",
        "randint",
        "random",
        "normal",
        "gumbel",
        "default_rng",
    ]:
        monkeypatch.setattr(np.random, name, forbidden)
    r = module.poisson_period_expectation(p, state_limit=10)
    assert dict(r["expected_gross_ticket_revenue"])["A"] < 0
    assert (
        pickle.dumps(np.random.get_state()) == before
        and pickle.dumps(p.compiled.spec) == spec
    )


@pytest.mark.parametrize("limit", [0, -1, True, None])
def test_explicit_positive_state_limit_required(limit):
    with pytest.raises(ValueError, match="positive state limit"):
        module.poisson_period_expectation(fixture(), state_limit=limit)


def test_resource_limits_return_no_partial_bound_transition_or_cap_reduction():
    p = fixture(stock=2)
    with pytest.raises(module.PoissonPeriodLimitError, match="state limit; no partial"):
        module.poisson_period_expectation(p, state_limit=1)
    p = fixture(stock=250)
    with pytest.raises(
        module.PoissonPeriodLimitError, match="dimension limit; no partial"
    ):
        module.poisson_period_expectation(p, state_limit=1000)
    assert dict(p.free_capacity)["A_OH0"] == 250


@pytest.mark.parametrize(
    "field", ["prices", "free_capacity", "market_segment_intensities"]
)
def test_no_missing_support_zero_fill(field):
    p = fixture()
    with pytest.raises(ValueError, match="complete"):
        replace(p, **{field: getattr(p, field)[:-1]})


def test_nonfinite_and_noncanonical_or_mixed_inputs_refused_without_fallback():
    p = fixture()
    with pytest.raises(ValueError, match="finite fixed"):
        replace(p, prices=tuple((k, float("nan")) for k, v in p.prices))
    with pytest.raises(ValueError, match="intensities"):
        replace(
            p,
            market_segment_intensities=tuple(
                (k, -1) for k, v in p.market_segment_intensities
            ),
        )
    c = p.compiled
    seg = c.spec.segments[0]
    spec = replace(
        c.spec,
        segments=(
            replace(seg, classes=(replace(seg.classes[0], weight=0.9),)),
            *c.spec.segments[1:],
        ),
    )
    with pytest.raises(ValueError, match="no normalization"):
        replace(p, compiled=SimpleNamespace(spec=spec))
    spec = replace(
        c.spec,
        segments=(replace(seg, mixed={"log_alpha_sd": 0.1}), *c.spec.segments[1:]),
    )
    with pytest.raises(ValueError, match="no mixed-logit fallback"):
        replace(p, compiled=SimpleNamespace(spec=spec))


def test_complete_immutable_owned_records_and_component_scope_only():
    r = module.poisson_period_expectation(fixture(), state_limit=10)
    saved = module.poisson_period_record(r)
    json.dumps(saved, allow_nan=False)
    saved["states"].clear()
    assert r["states"]
    with pytest.raises(TypeError):
        r["state_count"] = 999
    assert (
        not r["monte_carlo_or_poisson_count_truncation"]
        and not r["numerical_clipping_or_normalization"]
    )
    assert (
        not r["reentry_shock_rationing_or_planner_adopted"]
        and not r["primary_B4_welfare_adopted"]
    )
    assert (
        r["lifecycle_consumer_surplus"]
        is r["denied_requests"]
        is r["misallocation"]
        is r["scientific_budget"]
        is None
    )


def test_multiple_finite_classes_integrate_own_alpha_and_canonical_weight():
    p = fixture()
    c = p.compiled
    seg = c.spec.segments[0]
    first = replace(seg.classes[0], weight=0.25)
    second = replace(seg.classes[0], weight=0.75, alpha=0.04)
    c = compile_market(
        replace(
            c.spec,
            segments=(replace(seg, classes=(first, second)), *c.spec.segments[1:]),
        )
    )
    p = replace(p, compiled=c)
    active = np.array(
        [
            all(dict(p.free_capacity)[k] > 0 for k in product.legs)
            for product in c.spec.products
        ]
    )
    h = g = 0
    for taste in (first, second):
        u = systematic_utilities(
            c,
            c.market_index["OH0"],
            taste,
            np.array([dict(p.prices)[x.id] for x in c.spec.products]),
            active,
        )
        h += 2 * taste.weight * choice_probabilities(u)[0]
        g += 2 * taste.weight * conditional_logit_surplus(u, taste.alpha)
    r = module.poisson_period_expectation(p, state_limit=10)
    assert dict(r["expected_product_bookings"])["A_OH0"] == pytest.approx(
        -math.expm1(-h), abs=1e-12
    )
    assert dict(r["expected_ordinary_conditional_CS"])[
        ("OH0", "leisure")
    ] == pytest.approx(g * (-math.expm1(-h)) / h, abs=1e-11)


def test_float_underflow_keeps_structural_zero_probability_state_without_true_zero_claim():
    p = fixture(price=1e6)
    r = module.poisson_period_expectation(p, state_limit=10)
    assert r["state_count"] == 2 and 0 in r["transition_probabilities"]
    assert not r["symbolic_exactness_or_rigorous_error_interval_certified"]
    assert not r["numerical_clipping_or_normalization"]
