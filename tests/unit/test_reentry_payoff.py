"""A.10/A.5/B0/V9: NEW one-person finite policy reward definitions."""

import importlib.util
import json
import math
import os
import pickle
import sys
from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest

from airline_market.sim.ordered_reentry_logit import OrderedReentryLogit
from airline_market.sim.payoff_logit import ContingentPayoff

if os.environ.get("REENTRY_PAYOFF_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_reentry_payoff", os.environ["REENTRY_PAYOFF_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.sim import reentry_payoff as module


def pay(b=0, p=0, w=0, c=0):
    # Explicit zeroes are NEW fixture declarations, no production payoff defaults.
    return ContingentPayoff(*(None if x is None else F(x) for x in (b, p, w, c)))


def terminal(name, period, value):
    return module.ReentryPayoffState(name, period, None, (), pay(b=value))


def edge(name, target, reward, probability=F(1)):
    return module.PayoffTransition(name, probability, target, reward)


def fixed_choice(u=0.0, other=(), interline=False, alpha=1.0):
    return OrderedReentryLogit(
        (("buy", u),),
        other,
        "wait",
        0.0,
        alpha,
        interline,
        "NEW current myopic fixed type menu",
    )


def path(
    name="type",
    weight=F(1),
    u=0.0,
    *,
    benefit=30,
    payment=10,
    waiting=2,
    terminal_unknown=False,
):
    q = fixed_choice(u)
    yes = terminal("yes", 2, benefit)
    no = terminal("no", 2, None if terminal_unknown else 0)
    second = module.ReentryPayoffState(
        "second",
        1,
        q,
        (
            ("buy", (edge("purchase", "yes", pay(p=payment)),)),
            ("wait", (edge("no purchase", "no", pay(w=waiting)),)),
        ),
        None,
    )
    first = module.ReentryPayoffState(
        "first",
        0,
        q,
        (
            ("buy", (edge("purchase", "yes", pay(p=payment)),)),
            ("wait", (edge("retry", "second", pay(w=waiting)),)),
        ),
        None,
    )
    return module.ReentryPayoffType(name, weight, "first", (yes, second, no, first))


def law(types):
    return module.SinglePersonReentryPayoff(
        (0, 7, 0),
        tuple(types),
        2,
        "declared payoff units",
        "NEW conditional original-booked person; complete action-conditional finite law, not native lifecycle adoption",
    )


def run(problem, **kwargs):
    return module.reentry_policy_payoff(
        problem,
        node_limit=kwargs.get("node_limit", 100),
        edge_limit=kwargs.get("edge_limit", 100),
    )


def fraction(x):
    return None if x is None else F(int(x["numerator"]), int(x["denominator"]))


def test_two_attempt_original_person_explicit_payment_waiting_terminal_payoff():
    result = run(law((path(),)))
    # P(served)=3/4; expected net service-payment=15; expected waiting loss=1.5.
    assert fraction(result["expected_declared_policy_payoff"]) == F(27, 2)
    assert result["declared_original_people"] == 1
    states = {s["state"]: s for s in result["types"][0]["states"]}
    assert (
        fraction(states["second"]["value_exact_weighting_of_binary64_probabilities"])
        == 9
    )
    assert not result["per_search_expected_gains_or_CS_added"]
    assert (
        result["native_lifecycle_consumer_surplus"]
        is result["delivered_travel"]
        is None
    )


def test_persistent_type_weighted_once_differs_from_independent_redraw_or_mean_utility():
    high = path("high", F(1, 2), math.log(3), benefit=1, payment=0, waiting=0)
    low = path("low", F(1, 2), -math.log(3), benefit=1, payment=0, waiting=0)
    result = run(law((high, low)))
    assert float(fraction(result["expected_declared_policy_payoff"])) == pytest.approx(
        11 / 16
    )
    mean = run(law((path(benefit=1, payment=0, waiting=0),)))
    assert float(fraction(mean["expected_declared_policy_payoff"])) == pytest.approx(
        0.75
    )
    assert result["declared_original_people"] == 1 and len(result["types"]) == 2
    assert not result["types_independently_redrawn_between_attempts"]


def test_actual_per_attempt_gain_not_added_or_used_to_optimize_future_reward(
    monkeypatch,
):
    problem = law((path(),))
    baseline = run(problem)
    old = module.ordered_reentry_expectation

    def altered(q):
        r = dict(old(q))
        r["expected_gain_divided_by_declared_alpha"] = 1e200
        return r

    monkeypatch.setattr(module, "ordered_reentry_expectation", altered)
    assert (
        run(problem)["expected_declared_policy_payoff"]
        == baseline["expected_declared_policy_payoff"]
    )
    assert not baseline["future_payoff_optimized_or_fed_back_into_choice"]


def test_declared_replacement_cancellation_refund_loss_and_retry_count_once():
    t = path()
    first = next(s for s in t.states if s.name == "first")
    branches = (
        edge("remains reserved", "yes", pay(p=10), F(3, 4)),
        edge("replacement cancelled, full refund", "second", pay(p=0, c=4), F(1, 4)),
    )
    first = replace(
        first,
        transitions=(("buy", branches), ("wait", (edge("retry", "second", pay(w=2)),))),
    )
    changed = replace(
        t, states=tuple(first if s.name == "first" else s for s in t.states)
    )
    result = run(law((changed,)))
    # first buy: .75*20+.25*(-4+9); first wait: -2+9.
    assert fraction(result["expected_declared_policy_payoff"]) == F(93, 8)
    assert result["declared_original_people"] == 1


def test_positive_unknown_terminal_keeps_full_policy_value_null():
    r = run(law((path(terminal_unknown=True),)))
    assert (
        r["expected_declared_policy_payoff"] is None
        and r["positive_support_unknown_or_choice_underflow"]
    )
    assert len(r["types"][0]["states"]) == 4


def test_exact_zero_type_retained_and_tiny_positive_unknown_not_float_underflowed():
    unknown = path("unknown", F(0), terminal_unknown=True)
    known = path("known", F(1))
    r = run(law((known, unknown)))
    assert (
        fraction(r["expected_declared_policy_payoff"]) == F(27, 2)
        and len(r["types"]) == 2
    )
    tiny = F(1, 10**1000)
    r = run(
        law((replace(known, probability=1 - tiny), replace(unknown, probability=tiny)))
    )
    assert r["expected_declared_policy_payoff"] is None


def test_exact_type_weight_times_huge_terminal_value_avoids_early_conversion():
    huge = 10**1000
    small = F(1, huge)
    types = (
        module.ReentryPayoffType("huge", small, "done", (terminal("done", 0, huge),)),
        module.ReentryPayoffType("zero", 1 - small, "done", (terminal("done", 0, 0),)),
    )
    assert fraction(run(law(types))["expected_declared_policy_payoff"]) == 1


def test_zero_branch_unknown_and_inactive_other_remain_without_zero_fill():
    t = path()
    first = next(s for s in t.states if s.name == "first")
    unknown = terminal("unknown", 2, None)
    q = replace(first.choice, other=(("other", 0.0),))
    first = replace(
        first,
        choice=q,
        transitions=(
            (
                "buy",
                (
                    edge("known", "yes", pay(p=10)),
                    edge("zero unknown", "unknown", pay(b=None), F(0)),
                ),
            ),
            ("wait", dict(first.transitions)["wait"]),
            ("other", (edge("inactive unknown", "unknown", pay(b=None)),)),
        ),
    )
    t = replace(
        t,
        states=tuple(first if s.name == "first" else s for s in t.states) + (unknown,),
    )
    r = run(law((t,)))
    assert fraction(r["expected_declared_policy_payoff"]) == F(27, 2)
    assert len(r["types"][0]["states"]) == 5
    enabled = replace(first, choice=replace(q, interline=True))
    t = replace(t, states=tuple(enabled if s.name == "first" else s for s in t.states))
    assert run(law((t,)))["expected_declared_policy_payoff"] is None


def test_positive_choice_underflow_yields_null_even_when_all_payoffs_known():
    r = run(law((path(u=-1000.0),)))
    assert r["expected_declared_policy_payoff"] is None
    first = next(s for s in r["types"][0]["states"] if s["state"] == "first")
    assert "positive_choice_probability_underflow" in first["null_reasons"]
    assert dict(first["ordered_choice"]["choice_probabilities"])["buy"] == 0
    # Explicit -infinity is declared unavailability, separate from underflow.
    r = run(law((path(u=-math.inf),)))
    assert fraction(r["expected_declared_policy_payoff"]) == -4


def test_owned_input_export_canonical_order_and_global_rng_identity():
    problem = law((path(),))
    before = pickle.dumps((problem, np.random.get_state()))
    result = run(problem)
    assert pickle.dumps((problem, np.random.get_state())) == before
    record = module.reentry_payoff_record(result)
    json.dumps(record, allow_nan=False)
    record["types"][0]["states"].clear()
    assert len(result["types"][0]["states"]) == 4
    with pytest.raises(TypeError):
        result["expected_declared_policy_payoff"] = 0
    assert run(replace(problem, types=tuple(reversed(problem.types)))) == result


@pytest.mark.parametrize(
    "change",
    [
        "missing action",
        "wrong sum",
        "duplicate branch",
        "missing child",
        "cycle",
        "alpha redraw",
        "unattached state",
        "beyond horizon",
        "float weight",
        "unknown terminal omitted",
    ],
)
def test_incomplete_noncausal_or_inconsistent_laws_refused(change):
    t = path()
    first = next(s for s in t.states if s.name == "first")
    with pytest.raises((ValueError, TypeError)):
        if change == "missing action":
            replace(first, transitions=first.transitions[:1])
        elif change == "wrong sum":
            replace(
                first,
                transitions=(
                    ("buy", (edge("wrong", "yes", pay(), F(1, 2)),)),
                    ("wait", dict(first.transitions)["wait"]),
                ),
            )
        elif change == "duplicate branch":
            replace(
                first,
                transitions=(
                    (
                        "buy",
                        (
                            edge("same", "yes", pay(), F(1, 2)),
                            edge("same", "yes", pay(), F(1, 2)),
                        ),
                    ),
                    ("wait", dict(first.transitions)["wait"]),
                ),
            )
        elif change == "missing child":
            replace(t, states=tuple(s for s in t.states if s.name != "yes"))
        elif change == "cycle":
            replace(
                t,
                states=tuple(
                    replace(s, period=0) if s.name == "second" else s for s in t.states
                ),
            )
        elif change == "alpha redraw":
            replace(
                t,
                states=tuple(
                    replace(s, choice=replace(s.choice, alpha=2))
                    if s.name == "second"
                    else s
                    for s in t.states
                ),
            )
        elif change == "unattached state":
            replace(t, states=(*t.states, terminal("unattached", 2, 0)))
        elif change == "beyond horizon":
            replace(law((t,)), horizon=1)
        elif change == "float weight":
            replace(t, probability=1.0)
        elif change == "unknown terminal omitted":
            module.ReentryPayoffState("missing", 2, None, (), None)


def test_no_partial_infrastructure_limits_include_zero_weight_type_support():
    problem = law((path("known", F(1)), path("zero", F(0))))
    with pytest.raises(module.ReentryPayoffLimitError):
        run(problem, node_limit=7)
    with pytest.raises(module.ReentryPayoffLimitError):
        run(problem, edge_limit=7)
    assert run(problem)["nodes_evaluated"] == 8
    with pytest.raises(ValueError):
        run(problem, node_limit=0)


def test_exact_binary64_mass_residual_retained_without_renormalizing_payoff():
    q = fixed_choice(other=(("other", 0.0),), interline=True)
    node = module.ReentryPayoffState(
        "start",
        0,
        q,
        tuple((a, (edge("end", "end", pay()),)) for a in ("buy", "other", "wait")),
        None,
    )
    kind = module.ReentryPayoffType(
        "single", F(1), "start", (node, terminal("end", 1, 1))
    )
    r = run(law((kind,)))
    state = r["types"][0]["states"][0]
    mass = sum(
        (F.from_float(p) for k, p in state["ordered_choice"]["choice_probabilities"]),
        F(),
    )
    assert fraction(state["exact_binary64_choice_probability_mass"]) == mass
    assert fraction(state["exact_binary64_choice_mass_residual"]) == mass - 1
    assert fraction(r["expected_declared_policy_payoff"]) == mass


def test_signed_and_cancelling_declared_payoffs_not_logged_clamped_or_repaired():
    negative = module.ReentryPayoffState("end", 0, None, (), pay(b=-2, p=7, w=-1, c=3))
    positive = terminal("end", 0, 11)
    a = module.ReentryPayoffType("negative", F(1, 2), "end", (negative,))
    b = module.ReentryPayoffType("positive", F(1, 2), "end", (positive,))
    r = run(law((a, b)))
    assert fraction(r["types"][0]["initial_value"]) == -11
    assert fraction(r["expected_declared_policy_payoff"]) == 0
    assert r["native_lifecycle_consumer_surplus"] is None
