"""A.10/B0/V9: NEW finite forecast/depletion references, not sampled cohorts."""

import importlib.util
import json
import math
import os
import pickle
import sys
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from airline_market.sim.choice import choice_probabilities, systematic_utilities
from airline_market.sim.poisson_period import (
    NativePoissonPeriod,
    PoissonPeriodLimitError,
)
from airline_market.sim.segmented_welfare import conditional_logit_surplus
from airline_market.sim.tape import ExogenousTape
from airline_market.spec import compile_market
from tests.unit import test_rationing_native as native

if os.environ.get("POISSON_FORECAST_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_poisson_forecast", os.environ["POISSON_FORECAST_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.sim import poisson_forecast as module


def period(*, rate=1, stock=1, price=100):
    c = native.fixture().compiled
    return NativePoissonPeriod(
        c,
        tuple((p.id, price) for p in c.spec.products),
        tuple((l.id, stock if l.id == "A_OH0" else 0) for l in c.spec.legs),
        tuple(
            ((m.id, s.id), rate if (m.id, s.id) == ("OH0", "leisure") else 0)
            for m in c.spec.markets
            for s in c.spec.segments
        ),
        "unit explicit conditional ordinary forecast",
    )


def path(name, weight, *periods):
    return module.PoissonForecastPath(name, Fraction(weight), periods)


def forecast(*paths):
    return module.FinitePoissonForecast(paths, 0, "unit complete joint path law")


def run(f, **kwargs):
    return module.poisson_forecast_expectation(
        f, state_limit=20, conditional_limit=100, **kwargs
    )


def q_cs(p):
    c = p.compiled
    active = np.array([p.id == "A_OH0" for p in c.spec.products])
    taste = c.spec.segments[c.segment_index["leisure"]].classes[0]
    u = systematic_utilities(
        c,
        c.market_index["OH0"],
        taste,
        np.array([dict(p.prices)[x.id] for x in c.spec.products]),
        active,
    )
    return choice_probabilities(u)[0], conditional_logit_surplus(u, taste.alpha)


@pytest.mark.parametrize("stock", [1, 2])
def test_two_period_full_stock_propagation_matches_nonhomogeneous_poisson_closed_form(
    stock,
):
    p = period(stock=stock, rate=0.4)
    second = replace(
        p,
        market_segment_intensities=tuple(
            (k, 2 if v else 0) for k, v in p.market_segment_intensities
        ),
    )
    q, cs = q_cs(p)
    h = 2.4 * q
    expected = -math.expm1(-h) if stock == 1 else 2 - (2 + h) * math.exp(-h)
    r = run(forecast(path("only", 1, p, second)))
    assert dict(r["expected_product_bookings"])["A_OH0"] == pytest.approx(
        expected, abs=1e-12
    )
    assert dict(r["expected_ordinary_conditional_CS"])[
        ("OH0", "leisure")
    ] == pytest.approx(cs * expected / q, abs=1e-11)
    assert dict(r["expected_gross_ticket_revenue"])["A"] == pytest.approx(
        100 * expected, abs=1e-10
    )
    assert r["expected_ordinary_arrivals"] == pytest.approx(2.4)
    assert r["expected_ordinary_outside_choices"] == pytest.approx(
        2.4 - expected, abs=1e-12
    )
    assert all(abs(v) < 1e-11 for k, v in r["numerical_leg_stock_identity_residual"])
    assert abs(r["numerical_terminal_mass_residual"]) < 1e-12
    assert [p["native_period"] for p in r["paths"][0]["periods"]] == [0, 1]


def test_price_schedule_integrates_period_specific_rewards_without_stock_reset():
    first = period(rate=0.8, price=20)
    second = replace(first, prices=tuple((k, -2) for k, v in first.prices))
    q1, cs1 = q_cs(first)
    q2, cs2 = q_cs(second)
    h1, h2 = 0.8 * q1, 0.8 * q2
    b1, b2 = -math.expm1(-h1), math.exp(-h1) * -math.expm1(-h2)
    r = run(forecast(path("declared prices", 1, first, second)))
    assert dict(r["expected_gross_ticket_revenue"])["A"] == pytest.approx(
        20 * b1 - 2 * b2, abs=1e-11
    )
    assert dict(r["expected_ordinary_conditional_CS"])[
        ("OH0", "leisure")
    ] == pytest.approx(cs1 * b1 / q1 + cs2 * b2 / q2, abs=1e-11)
    assert dict(r["paths"][0]["periods"][1]["expected_gross_ticket_revenue"])["A"] < 0


def test_persistent_joint_regime_mixture_is_not_mean_intensity_or_independent_redraw():
    low = period(rate=0.2)
    high = replace(
        low,
        market_segment_intensities=tuple(
            (k, 4 if v else 0) for k, v in low.market_segment_intensities
        ),
    )
    f = forecast(
        path("low persistent", Fraction(1, 2), low, low),
        path("high persistent", Fraction(1, 2), high, high),
    )
    r = run(f)
    q, _ = q_cs(low)
    expected = 1 - (math.exp(-0.4 * q) + math.exp(-8 * q)) / 2
    mean_intensity_result = -math.expm1(-4.2 * q)
    independent_regime_result = 1 - ((math.exp(-0.2 * q) + math.exp(-4 * q)) / 2) ** 2
    actual = dict(r["expected_product_bookings"])["A_OH0"]
    assert actual == pytest.approx(expected, abs=1e-12)
    assert abs(actual - mean_intensity_result) > 0.01
    assert abs(actual - independent_regime_result) > 0.01
    assert {p["name"] for p in r["paths"]} == {p.name for p in f.paths}


def test_joint_initial_stock_law_is_not_expected_stock_substitution():
    empty, two = period(stock=0, rate=2), period(stock=2, rate=2)
    r = run(
        forecast(path("zero", Fraction(1, 2), empty), path("two", Fraction(1, 2), two))
    )
    q, _ = q_cs(two)
    h = 2 * q
    expected = (2 - (2 + h) * math.exp(-h)) / 2
    actual = dict(r["expected_product_bookings"])["A_OH0"]
    assert actual == pytest.approx(expected, abs=1e-12)
    assert abs(actual - -math.expm1(-h)) > 0.01
    assert dict(r["expected_initial_net_free_capacity"])["A_OH0"] == 1


def test_all_zero_weight_paths_zero_probability_stock_states_and_empty_demand_retained(
    monkeypatch,
):
    p = period(price=1e6)
    empty = replace(
        p,
        market_segment_intensities=tuple(
            (k, 0) for k, v in p.market_segment_intensities
        ),
    )
    original = module.poisson_period_expectation
    calls = []

    def spy(problem, **kwargs):
        calls.append(problem.free_capacity)
        return original(problem, **kwargs)

    monkeypatch.setattr(module, "poisson_period_expectation", spy)
    r = run(forecast(path("underflow", 1, p, p), path("zero weight", 0, empty, empty)))
    assert len(r["paths"]) == 2
    selected = next(x for x in r["paths"] if x["name"] == "underflow")
    assert any(
        row["incoming_probability"] == 0
        for row in selected["periods"][1]["conditionals"]
    )
    assert len(calls) == r["conditional_operator_calls"] == 5
    assert len(selected["terminal_stock_distribution"]) == 2
    assert not r["symbolic_exactness_or_rigorous_error_interval"]


def test_no_sampling_global_rng_or_input_mutation_and_canonical_path_order(monkeypatch):
    p = period()
    a, b = path("a", Fraction(1, 3), p, p), path("b", Fraction(2, 3), p, p)
    f = forecast(b, a)
    before = pickle.dumps(f)
    rng = pickle.dumps(np.random.get_state())

    def forbidden(*args, **kwargs):
        raise AssertionError("no new physical draw")

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
    r = run(f)
    assert [p["name"] for p in r["paths"]] == ["a", "b"]
    assert pickle.dumps(f) == before and pickle.dumps(np.random.get_state()) == rng
    assert r == run(forecast(a, b))


def test_record_owned_immutable_complete_input_paths_and_unadopted_scope():
    p = period()
    r = run(forecast(path("only", 1, p, p)))
    record = module.poisson_forecast_record(r)
    json.dumps(record, allow_nan=False)
    record["paths"].clear()
    assert r["paths"]
    with pytest.raises(TypeError):
        r["paths"][0]["name"] = "changed"
    for k in [
        "forecast_observability_or_joint_law_authenticated",
        "physical_reentry_rationing_or_lifecycle_adopted",
        "state_contingent_policy_or_inference_adopted",
        "primary_default_empirical_production_adopted",
    ]:
        assert not r[k]
    assert (
        r["lifecycle_consumer_surplus"]
        is r["denied_requests"]
        is r["misallocation"]
        is r["scientific_budget"]
        is None
    )


@pytest.mark.parametrize("value", [0, -1, True, None])
def test_explicit_infrastructure_limits_required(value):
    f = forecast(path("only", 1, period()))
    with pytest.raises(ValueError, match="positive"):
        module.poisson_forecast_expectation(f, state_limit=20, conditional_limit=value)


def test_limit_or_component_failure_propagates_no_partial_or_fallback(monkeypatch):
    p = period()
    f = forecast(path("only", 1, p, p))
    with pytest.raises(
        module.PoissonForecastLimitError, match="conditional-call limit"
    ):
        module.poisson_forecast_expectation(f, state_limit=20, conditional_limit=1)
    with pytest.raises(PoissonPeriodLimitError, match="state limit"):
        module.poisson_forecast_expectation(f, state_limit=1, conditional_limit=100)

    def fail(*args, **kwargs):
        raise RuntimeError("new injected backend failure")

    monkeypatch.setattr(module, "poisson_period_expectation", fail)
    with pytest.raises(RuntimeError, match="injected backend"):
        run(f)


def test_complete_rational_joint_law_and_common_support_required_without_normalization():
    p = period()
    with pytest.raises(ValueError, match="rational path weight"):
        module.PoissonForecastPath("float", 0.5, (p,))
    with pytest.raises(ValueError, match="sum to one"):
        forecast(path("a", Fraction(1, 3), p), path("b", Fraction(1, 3), p))
    with pytest.raises(ValueError, match="unique"):
        forecast(path("same", Fraction(1, 2), p), path("same", Fraction(1, 2), p))
    with pytest.raises(ValueError, match="full period support"):
        forecast(path("a", Fraction(1, 2), p), path("b", Fraction(1, 2), p, p))
    with pytest.raises(ValueError, match="no stock reset"):
        path(
            "reset",
            1,
            p,
            replace(p, free_capacity=tuple((k, 0) for k, v in p.free_capacity)),
        )
    changed = replace(
        p,
        compiled=compile_market(
            replace(p.compiled.spec, name="different native model")
        ),
    )
    with pytest.raises(ValueError, match="common model"):
        forecast(path("a", Fraction(1, 2), p), path("b", Fraction(1, 2), changed))
    with pytest.raises(ValueError, match="period range"):
        module.FinitePoissonForecast(
            (path("a", 1, p, p),), p.compiled.spec.horizon.n_periods - 1, "unit"
        )
    with pytest.raises(ValueError, match="conditioning"):
        module.FinitePoissonForecast((path("a", 1, p),), 0, "")


def test_shared_leg_two_market_races_across_declared_price_and_rate_periods():
    first = period()
    c = first.compiled
    first = replace(
        first,
        free_capacity=tuple((l.id, 1 if l.carrier == "A" else 0) for l in c.spec.legs),
        market_segment_intensities=tuple(
            (
                k,
                0.6
                if k == ("OH0", "leisure")
                else 1.3
                if k == ("OD0_0", "leisure")
                else 0,
            )
            for k, v in first.market_segment_intensities
        ),
    )
    second = replace(
        first,
        prices=tuple((k, 50) for k, v in first.prices),
        market_segment_intensities=tuple(
            (
                k,
                1.8
                if k == ("OH0", "leisure")
                else 0.4
                if k == ("OD0_0", "leisure")
                else 0,
            )
            for k, v in first.market_segment_intensities
        ),
    )
    taste = c.spec.segments[c.segment_index["leisure"]].classes[0]
    active = np.array([p.carrier == "A" for p in c.spec.products])
    expected = {"A_OH0": 0.0, "A_OD0_0": 0.0}
    unsold = 1.0
    for p in (first, second):
        rates = dict(p.market_segment_intensities)
        hazards = {}
        for market, product in [("OH0", "A_OH0"), ("OD0_0", "A_OD0_0")]:
            u = systematic_utilities(
                c,
                c.market_index[market],
                taste,
                np.array([dict(p.prices)[x.id] for x in c.spec.products]),
                active,
            )
            hazards[product] = rates[(market, "leisure")] * choice_probabilities(u)[0]
        h = sum(hazards.values())
        for product, value in hazards.items():
            expected[product] += unsold * value * -math.expm1(-h) / h
        unsold *= math.exp(-h)
    r = run(forecast(path("shared-leg path", 1, first, second)))
    bookings = dict(r["expected_product_bookings"])
    for k, v in expected.items():
        assert bookings[k] == pytest.approx(v, abs=1e-12)
    assert sum(bookings.values()) == pytest.approx(1 - unsold, abs=1e-12)
    assert all(abs(v) < 1e-11 for k, v in r["numerical_leg_stock_identity_residual"])


def test_union_of_propagated_states_has_own_limit_without_discarding_zeroes():
    first = period()
    c = first.compiled
    first = replace(
        first,
        free_capacity=tuple((l.id, 1 if l.carrier == "A" else 0) for l in c.spec.legs),
    )
    second = replace(
        first,
        market_segment_intensities=tuple(
            (k, 1 if k == ("OD0_0", "leisure") else 0)
            for k, v in first.market_segment_intensities
        ),
    )
    f = forecast(path("successive local/connection", 1, first, second))
    with pytest.raises(
        module.PoissonForecastLimitError, match="propagated-state limit"
    ):
        module.poisson_forecast_expectation(f, state_limit=2, conditional_limit=100)


def test_rational_weight_underflow_retains_full_zero_float_path_and_exact_input_weight():
    p = period()
    tiny = Fraction(1, 10**400)
    r = run(forecast(path("ordinary", 1 - tiny, p), path("tiny", tiny, p)))
    small = next(x for x in r["paths"] if x["name"] == "tiny")
    assert small["weight"] == (1, 10**400) and small["float_weight"] == 0
    assert small["periods"] and r["conditional_operator_calls"] == 2
    assert not r["symbolic_exactness_or_rigorous_error_interval"]
