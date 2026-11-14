"""A.8/B0/V9: NEW tiny order-bound witnesses, never old scientific replay."""

import importlib.util
import itertools
import json
import os
import pickle
import sys
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from airline_market.policy.allocation_comparison import compare_completed_choices
from airline_market.policy.rationing import (
    AllocationProblem,
    Applicant,
    Offer,
)
from airline_market.policy.rationing_native import NativeOrdinaryPool
from airline_market.policy.regulator import RegulatedMarketEngine
from tests.unit import test_allocation_comparison as fixtures
from tests.unit import test_rationing_native as native

if os.environ.get("NATIVE_ORDER_BOUND_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_native_order", os.environ["NATIVE_ORDER_BOUND_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.policy import native_order_bound as module


def problem(people, capacities=(("L", 1),), products=(("A", ("L",)), ("B", ("L",)))):
    return AllocationProblem(
        0,
        capacities,
        products,
        tuple(
            Applicant((0, i, 0), tuple(Offer(*o) for o in offers))
            for i, offers in enumerate(people)
        ),
    )


def completed(p, order=None, limit=10000):
    keys = tuple(a.key for a in p.applicants) if order is None else order
    actual = module.native_order_allocation(p, keys)
    pool = NativeOrdinaryPool(
        p, (), (), (), "unit declared exact score, not welfare", "local-declaration", 31
    )
    return compare_completed_choices(pool, actual["assignments"], state_limit=limit)


def brute(p):
    """Independent direct stock/argmax exhaustive permutation reference."""
    products = dict(p.products)
    best = None
    for people in itertools.permutations(p.applicants):
        stock = dict(p.capacities)
        score = Fraction(0)
        assignments = []
        for person in people:
            feasible = [
                o
                for o in person.offers
                if all(stock[k] > 0 for k in products[o.product])
            ]
            options = [
                (o.choice_value, o.product, o.allocation_value) for o in feasible
            ] + [(Fraction(0), None, Fraction(0))]
            _, product, value = max(options, key=lambda x: x[0])
            if product is not None:
                for leg in products[product]:
                    stock[leg] -= 1
            score += value
            assignments.append((person.key, product))
        if best is None or score > best[0]:
            best = score, tuple(a.key for a in people), tuple(assignments)
    return best


@pytest.mark.parametrize("stock", [0, 1, 2])
@pytest.mark.parametrize("variant", [0, 1, 2])
def test_independent_exhaustive_signed_multiple_products_and_canonical_ties(
    stock, variant
):
    p = problem(
        [
            [("A", 2, "-1.00000000000000000001"), ("B", 1, 4)],
            [("B", 1, 3), ("A", 1, variant)],
            [("A", -1, 100), ("B", 0, -2)],
            [],
        ],
        capacities=(("L", stock), ("R", 1)),
        products=(("A", ("L",)), ("B", ("L", "R"))),
    )
    expected = brute(p)
    before = pickle.dumps(p), pickle.dumps(np.random.get_state())
    result = module.exact_native_order_bound(p, state_limit=10000)
    assert (
        result["allocation_value"],
        result["priority_order"],
        result["assignments"],
    ) == expected
    assert len(result["assignments"]) == 4
    assert all(x["initial"] == x["used"] + x["remaining"] for x in result["capacity"])
    assert before == (pickle.dumps(p), pickle.dumps(np.random.get_state()))
    assert result == module.exact_native_order_bound(
        replace(p, applicants=tuple(reversed(p.applicants))), state_limit=10000
    )


def test_ordering_improvement_connection_competes_with_two_locals():
    p = fixtures.synthetic_pool().problem
    row = module.compare_native_order(completed(p), state_limit=100)
    assert (
        row.completed.actual_score == 1
        and row.native_order_bound["allocation_value"] == 2
    )
    assert row.ordering_score_gap == 1 and row.alternative_rationing_relaxation_gap == 0
    assert len(row.native_order_bound["assignments"]) == 3


def test_assignment_relaxation_can_ration_preferred_alternative_but_order_cannot():
    p = problem(
        [[("A", 2, 0), ("B", 1, 10)]],
        capacities=(("L", 1), ("R", 1)),
        products=(("A", ("L",)), ("B", ("R",))),
    )
    row = module.compare_native_order(completed(p), state_limit=10)
    assert row.native_order_bound["allocation_value"] == row.ordering_score_gap == 0
    assert row.alternative_rationing_relaxation_gap == 10
    assert row.completed.bound["assignments"][0][1] == "B"


def test_native_inside_outside_tie_forces_exact_negative_optimum():
    p = problem([[("A", 0, "-1.0000000000000000000000001")]])
    row = module.compare_native_order(completed(p), state_limit=10)
    assert row.native_order_bound["allocation_value"] == Fraction(
        "-1.0000000000000000000000001"
    )
    assert row.ordering_score_gap == 0 and row.alternative_rationing_relaxation_gap > 1


def test_empty_problem_has_complete_empty_witness_and_exact_score():
    row = module.compare_native_order(completed(problem([])), state_limit=1)
    assert (
        row.native_order_bound["assignments"] == ()
        and row.native_order_bound["allocation_value"] == 0
    )
    assert row.native_order_bound["visited_states"] == 1


@pytest.mark.parametrize("limit", [0, -1, True, None, 1.5])
def test_explicit_infrastructure_limit_required(limit):
    with pytest.raises(ValueError, match="positive state limit"):
        module.exact_native_order_bound(problem([]), state_limit=limit)


def test_new_limit_retains_actual_and_prior_ir_but_returns_no_partial_order():
    p = problem([[("A", 1, 1)], [("B", 1, 2)]])
    original = completed(p)
    with pytest.raises(module.NativeOrderLimitError, match="no partial bound"):
        module.exact_native_order_bound(p, state_limit=1)
    row = module.compare_native_order(original, state_limit=1)
    assert row.completed is original and row.completed.bound is not None
    assert (
        row.native_order_bound
        is row.ordering_score_gap
        is row.alternative_rationing_relaxation_gap
        is None
    )
    assert row.status == "state_limit_no_order_bound"


def test_old_ir_limit_preserved_without_solver_replay(monkeypatch):
    p = problem([[("A", 1, 1)], [("B", 1, 2)]])
    old = completed(p, limit=1)
    assert old.bound is None

    def forbidden(*args, **kwargs):
        raise AssertionError("no repeat IR solver")

    monkeypatch.setattr(
        "airline_market.policy.allocation_comparison.exact_assignment_bound", forbidden
    )
    row = module.compare_native_order(old, state_limit=100)
    assert (
        row.native_order_bound is not None
        and row.alternative_rationing_relaxation_gap is None
    )
    assert row.completed.bound is None


@pytest.mark.parametrize("order", [(), ((0, 0, 0), (0, 0, 0)), ((0, 9, 0), (0, 1, 0))])
def test_full_original_pool_cannot_be_filtered_duplicated_or_replaced(order):
    with pytest.raises(ValueError, match="EVERY"):
        module.native_order_allocation(problem([[], []]), order)


def test_feasible_ir_assignment_is_not_necessarily_a_native_order_witness():
    p = problem(
        [[("A", 2, 0), ("B", 1, 10)]],
        capacities=(("L", 1), ("R", 1)),
        products=(("A", ("L",)), ("B", ("R",))),
    )
    old = completed(p)
    forged = replace(
        old,
        actual_choices=old.bound["assignments"],
        actual_score=Fraction(10),
        actual_capacity=old.bound["capacity"],
    )
    with pytest.raises(ValueError, match="native choice order"):
        module.compare_native_order(forged, state_limit=100)


def test_lossless_json_owns_exports_and_no_welfare_or_policy_adoption():
    row = module.compare_native_order(
        completed(problem([[("A", 0, "-0.000000000000000000001")]])), state_limit=10
    )
    saved = json.loads(json.dumps(module.native_order_comparison_record(row)))
    assert saved["native_order_bound"]["allocation_value"] == {
        "numerator": "-1",
        "denominator": "1000000000000000000000",
    }
    assert (
        not saved["native_efficient_assignment_booked"]
        and not saved["primary_B4_adopted"]
    )
    assert (
        not saved["lifecycle_welfare_denial_or_misallocation_identified"]
        and saved["scientific_budget"] is None
    )
    saved["native_order_bound"]["assignments"].clear()
    assert row.native_order_bound["assignments"]
    with pytest.raises(TypeError):
        row.native_order_bound["capacity"][0]["remaining"] = 90


def test_new_native_world_executes_computed_full_order_with_unmodified_choices(
    monkeypatch,
):
    e, _ = fixtures.fixture()
    parent = RegulatedMarketEngine(e.compiled, cap_policy=e.cap_policy)
    parent.reset(tape=e.tape)
    e.step(native.actions(e))
    row = e.allocation_comparisons()[0]
    comparison = module.compare_native_order(row, state_limit=100000)
    result = comparison.native_order_bound
    lookup = {
        (row.pool.problem.period, r.gumbel_row, r.market): r for r in row.pool.requests
    }
    # Frozen dated lineage refuses forced hooks. This NEW control-world witness
    # uses the base regulated native engine's explicit unit hook, all requests,
    # and genuine native choices. It is not an efficient dated booking policy.

    def forbidden(*args, **kwargs):
        raise AssertionError("no tape generation while projecting or stepping")

    monkeypatch.setattr("airline_market.sim.tape.ExogenousTape.generate", forbidden)
    parent.step(
        native.actions(parent), _requests=[lookup[k] for k in result["priority_order"]]
    )
    actual = tuple(
        (
            key,
            None
            if x["chosen"] is None
            else parent.compiled.spec.products[x["chosen"]].id,
        )
        for key, x in zip(result["priority_order"], parent.world.request_log)
    )
    assert actual == result["assignments"] and len(actual) == len(lookup)
    assert result["allocation_value"] >= row.actual_score
