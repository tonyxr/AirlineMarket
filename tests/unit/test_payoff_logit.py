"""A.10/B0/V9: NEW exact payoff declarations/logit timing references, no cohorts."""

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

from airline_market.sim.segmented_welfare import conditional_logit_surplus

if os.environ.get("PAYOFF_LOGIT_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_payoff_logit", os.environ["PAYOFF_LOGIT_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.sim import payoff_logit as module


def payoff(value=0, *, payment=0, waiting=0, cancellation=0):
    return module.ContingentPayoff(
        *(
            None if v is None else Fraction(v)
            for v in (value, payment, waiting, cancellation)
        )
    )


def state(name, weight, **values):
    return module.PayoffState(
        name,
        Fraction(weight),
        tuple(
            (k, v if type(v) is module.ContingentPayoff else payoff(v))
            for k, v in values.items()
        ),
    )


def problem(*states, alpha=1, offsets=None):
    return module.SingleCommitmentLogit(
        tuple(
            (k, Fraction(v))
            for k, v in (offsets or {k: 0 for k, v in states[0].payoffs}).items()
        ),
        "outside",
        Fraction(alpha),
        states,
        "unit declared USD-equivalent",
        "unit risk-neutral one-commitment law, no delivered outcomes authenticated",
    )


def run(p):
    return module.payoff_information_surplus(p)


def ratio(x):
    return None if x is None else Fraction(int(x["numerator"]), int(x["denominator"]))


def test_one_state_reduces_to_frozen_static_logit_with_all_payoff_components():
    p = problem(
        state(
            "only",
            1,
            A=payoff(30, payment=7, waiting=2, cancellation=1),
            B=-3,
            outside=5,
        ),
        alpha=Fraction(1, 2),
        offsets={"A": 1, "B": -1, "outside": 0},
    )
    r = run(p)
    expected = conditional_logit_surplus([1 + 0.5 * 20, -1 + 0.5 * -3, 0.5 * 5], 0.5)
    assert r["single_commitment_CS_choose_before_payoff_state"] == pytest.approx(
        expected, abs=1e-12
    )
    assert r[
        "single_commitment_CS_observe_payoff_state_before_choice"
    ] == pytest.approx(expected, abs=1e-12)
    assert abs(r["information_timing_difference"]) < 1e-12
    assert dict((k, ratio(v)) for k, v in r["expected_net_values"])["A"] == 20


def test_information_before_choice_is_not_logsum_of_average_outcomes():
    p = problem(
        state("left", Fraction(1, 2), A=10, B=-10, outside=0),
        state("right", Fraction(1, 2), A=-10, B=10, outside=0),
    )
    r = run(p)
    assert r["single_commitment_CS_choose_before_payoff_state"] == pytest.approx(
        math.log(3), abs=1e-12
    )
    assert r[
        "single_commitment_CS_observe_payoff_state_before_choice"
    ] == pytest.approx(math.log(math.exp(10) + math.exp(-10) + 1), abs=1e-12)
    assert r["information_timing_difference"] > 8


def test_common_state_shifts_and_extreme_common_payoffs_cancel_before_float_conversion():
    base = problem(
        state("one", Fraction(1, 3), A=2, outside=0),
        state("two", Fraction(2, 3), A=-1, outside=0),
    )
    huge = Fraction(10**500)
    shifted = problem(
        state("one", Fraction(1, 3), A=huge + 2, outside=huge),
        state("two", Fraction(2, 3), A=-huge - 1, outside=-huge),
    )
    a, b = run(base), run(shifted)
    for k in [
        "single_commitment_CS_choose_before_payoff_state",
        "single_commitment_CS_observe_payoff_state_before_choice",
        "information_timing_difference",
    ]:
        assert a[k] == b[k]


def test_signed_fares_losses_and_exact_fractional_expected_values_not_clipped():
    p = problem(
        state(
            "rebate",
            Fraction(1, 3),
            A=payoff(1, payment=-2, waiting=Fraction(1, 7), cancellation=-1),
            outside=0,
        ),
        state(
            "loss",
            Fraction(2, 3),
            A=payoff(-2, payment=3, waiting=4, cancellation=5),
            outside=0,
        ),
    )
    r = run(p)
    expected = Fraction(1, 3) * (4 - Fraction(1, 7)) + Fraction(2, 3) * -14
    assert dict((k, ratio(v)) for k, v in r["expected_net_values"])["A"] == expected
    assert ratio(dict(r["states"][1]["payoffs"])["A"]["net_value"]) == 4 - Fraction(
        1, 7
    )
    assert not r["gap_clipped_or_probabilities_normalized"]


def test_unknown_positive_payoff_retains_every_state_and_nulls_both_timing_values():
    p = problem(
        state("known", Fraction(1, 2), A=3, outside=0),
        state("unknown", Fraction(1, 2), A=payoff(3, waiting=None), outside=0),
    )
    r = run(p)
    assert r["positive_probability_unknown_input"]
    assert (
        len(r["states"]) == 2
        and r["states"][0]["conditional_surplus_if_state_revealed"] is not None
    )
    assert (
        r["single_commitment_CS_choose_before_payoff_state"]
        is r["single_commitment_CS_observe_payoff_state_before_choice"]
        is r["information_timing_difference"]
        is None
    )
    assert dict(r["states"][1]["payoffs"])["A"]["waiting_loss"] is None


def test_exact_zero_probability_unknown_state_kept_without_complete_case_filtering():
    p = problem(
        state("known", 1, A=2, outside=0), state("zero", 0, A=payoff(None), outside=0)
    )
    r = run(p)
    assert len(r["states"]) == 2
    assert r["states"][1]["status"] == "declared_zero_probability_not_evaluated"
    assert r["states"][1]["conditional_surplus_if_state_revealed"] is None
    assert r["single_commitment_CS_choose_before_payoff_state"] == pytest.approx(
        math.log1p(math.exp(2)), abs=1e-12
    )


def test_tiny_positive_unknown_probability_never_becomes_declared_zero():
    tiny = Fraction(1, 10**400)
    p = problem(
        state("known", 1 - tiny, A=0, outside=0),
        state("tiny", tiny, A=payoff(1, waiting=None), outside=0),
    )
    r = run(p)
    assert r["positive_probability_unknown_input"]
    assert r["single_commitment_CS_choose_before_payoff_state"] is None
    assert ratio(r["states"][1]["probability"]) == tiny


def test_exact_probability_weighting_does_not_underflow_before_multiplication():
    tiny = Fraction(1, 10**400)
    p = problem(
        state("base", 1 - tiny, A=0, outside=0),
        state("tiny", tiny, A=10**300, outside=0),
    )
    r = run(p)
    states = r["states"]
    expected = sum(
        (
            ratio(s["probability"])
            * Fraction.from_float(s["conditional_surplus_if_state_revealed"])
            for s in states
        ),
        Fraction(),
    )
    assert ratio(r["exact_weighted_binary64_state_surplus"]) == expected
    assert float(tiny) == 0
    assert expected - (1 - tiny) * Fraction.from_float(
        states[0]["conditional_surplus_if_state_revealed"]
    ) == tiny * Fraction.from_float(states[1]["conditional_surplus_if_state_revealed"])
    assert r["single_commitment_CS_observe_payoff_state_before_choice"] == float(
        expected
    )


def test_one_outside_option_is_zero_surplus_in_both_information_timings():
    r = run(
        problem(
            state("a", Fraction(1, 2), outside=100),
            state("b", Fraction(1, 2), outside=-100),
        )
    )
    assert (
        r["single_commitment_CS_choose_before_payoff_state"]
        == r["single_commitment_CS_observe_payoff_state_before_choice"]
        == 0
    )


def test_input_global_rng_and_canonical_order_immutable_owned_json(monkeypatch):
    p = problem(
        state("b", Fraction(1, 3), A=4, outside=0),
        state("a", Fraction(2, 3), A=-2, outside=0),
    )
    before = pickle.dumps(p)
    rng = pickle.dumps(np.random.get_state())

    def forbidden(*args, **kwargs):
        raise AssertionError("no sampling")

    for k in ["random", "rand", "randint", "gumbel", "default_rng"]:
        monkeypatch.setattr(np.random, k, forbidden)
    r = run(p)
    assert pickle.dumps(p) == before and pickle.dumps(np.random.get_state()) == rng
    assert r == run(
        replace(
            p,
            states=tuple(reversed(p.states)),
            alternatives=tuple(reversed(p.alternatives)),
        )
    )
    saved = module.payoff_logit_record(r)
    json.dumps(saved, allow_nan=False)
    saved["states"].clear()
    assert r["states"]
    with pytest.raises(TypeError):
        r["scientific_budget"] = 1
    assert (
        r["native_lifecycle_consumer_surplus"]
        is r["delivered_travel"]
        is r["denied_requests"]
        is r["misallocation"]
        is r["scientific_budget"]
        is None
    )
    assert (
        not r["primary_default_empirical_production_adopted"]
        and not r["state_law_preferences_or_delivered_outcomes_authenticated"]
    )


@pytest.mark.parametrize("value", [0, -1, 0.1, None, True])
def test_explicit_positive_rational_alpha_required(value):
    p = problem(state("one", 1, A=1, outside=0))
    with pytest.raises(ValueError, match="alpha"):
        replace(p, alpha=value if type(value) is not int else Fraction(value))


def test_every_joint_state_alternative_outside_required_without_renormalization():
    p = problem(state("one", 1, A=1, outside=0))
    with pytest.raises(ValueError, match="sum to one"):
        replace(p, states=(replace(p.states[0], probability=Fraction(1, 2)),))
    with pytest.raises(ValueError, match="every state"):
        replace(p, states=(replace(p.states[0], payoffs=(("A", payoff(1)),)),))
    with pytest.raises(ValueError, match="outside"):
        replace(p, outside="missing")
    with pytest.raises(ValueError, match="distinct finite"):
        replace(p, states=(p.states[0], p.states[0]))
    with pytest.raises(ValueError, match="definition"):
        replace(p, definition="")
    with pytest.raises(TypeError, match="rational payoff"):
        module.ContingentPayoff(1.0, Fraction(), Fraction(), Fraction())


def test_nonrepresentable_relative_utility_has_no_partial_numerical_result():
    p = problem(state("huge", 1, A=10**500, outside=0))
    with pytest.raises(OverflowError):
        run(p)
