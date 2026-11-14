"""A.8/B0/V9: static rationing lottery, exact score bound and physical incidence."""

import importlib.util
import itertools
import json
import os
import pickle
import sys
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.sim.arrivals import Request
from airline_market.sim.choice import sampled_choice
from airline_market.sim.engine import MarketEngine
from airline_market.sim.ledger import book, inventory
from airline_market.sim.tape import ExogenousTape
from airline_market.spec import compile_market

if os.environ.get("RATIONING_SOURCE"):
    s = importlib.util.spec_from_file_location(
        "staged_rationing", os.environ["RATIONING_SOURCE"]
    )
    module = importlib.util.module_from_spec(s)
    sys.modules[s.name] = module
    s.loader.exec_module(module)
else:
    from airline_market.policy import rationing as module


def person(rank, offers, period=0):
    return module.Applicant((period, rank, 0), tuple(module.Offer(*x) for x in offers))


def fixture():
    return module.AllocationProblem(
        0,
        (("a", 1), ("b", 1)),
        (("ab", ("a", "b")), ("a", ("a",)), ("b", ("b",))),
        (
            person(0, (("ab", 10, 100),)),
            person(1, (("a", 10, 60),)),
            person(2, (("b", 10, 60),)),
        ),
    )


def rng():
    return StreamRegistry(26974, "unit/V9_rationing")


def feasible(problem, result):
    stock = dict(problem.capacities)
    used = {k: 0 for k in stock}
    people = {p.key: p for p in problem.applicants}
    graph = dict(problem.products)
    assert len(result["assignments"]) == len(people) and len(
        dict(result["assignments"])
    ) == len(people)
    value = Fraction(0)
    for key, p in result["assignments"]:
        if p is None:
            continue
        offer = next(o for o in people[key].offers if o.product == p)
        assert offer.choice_value >= 0
        value += offer.allocation_value
        for leg in graph[p]:
            used[leg] += 1
    assert value == result["allocation_value"]
    for row in result["capacity"]:
        assert (
            row["initial"] == stock[row["leg"]]
            and row["used"] == used[row["leg"]]
            and row["remaining"] >= 0
            and row["initial"] == row["used"] + row["remaining"]
        )


def exhaustive(problem):
    people = problem.applicants
    options = [
        (None,) + tuple(o.product for o in p.offers if o.choice_value >= 0)
        for p in people
    ]
    graph = dict(problem.products)
    stock = dict(problem.capacities)
    best = Fraction(0)
    for picks in itertools.product(*options):
        use = {k: 0 for k in stock}
        score = Fraction(0)
        for p, chosen in zip(people, picks):
            if chosen is None:
                continue
            score += next(o.allocation_value for o in p.offers if o.product == chosen)
            for leg in graph[chosen]:
                use[leg] += 1
        if all(use[k] <= stock[k] for k in stock):
            best = max(best, score)
    return best


def test_connection_can_lose_to_two_local_assignments_in_exact_bound():
    p = fixture()
    ordered = module.priority_allocation(p, binding_cap=False)
    bound = module.exact_assignment_bound(p, state_limit=1000)
    assert ordered["allocation_value"] == 100 and bound["allocation_value"] == 120
    assert dict(bound["assignments"]) == {
        (0, 0, 0): None,
        (0, 1, 0): "a",
        (0, 2, 0): "b",
    }
    feasible(p, ordered)
    feasible(p, bound)


def test_exact_bound_matches_independent_exhaustive_multi_product_assignments():
    p = module.AllocationProblem(
        0,
        (("a", 2), ("b", 1)),
        (("a", ("a",)), ("b", ("b",)), ("ab", ("a", "b"))),
        (
            person(0, (("ab", 4, "10.01"), ("a", 2, 4))),
            person(1, (("a", 3, 7), ("b", 1, 8))),
            person(2, (("ab", -1, 100), ("b", 2, 5))),
            person(3, (("a", 0, 1),)),
        ),
    )
    out = module.exact_assignment_bound(p, state_limit=10000)
    assert out["allocation_value"] == exhaustive(p)
    feasible(p, out)


def test_exact_declared_decimal_scores_are_not_cent_or_context_rounded():
    low = Decimal(".100000000000000000000000000000001")
    high = Decimal(".100000000000000000000000000000002")
    p = module.AllocationProblem(
        0,
        (("a", 1),),
        (("a", ("a",)),),
        (person(0, (("a", 1, low),)), person(1, (("a", 1, high),))),
    )
    out = module.exact_assignment_bound(p, state_limit=100)
    assert (
        out["allocation_value"] == Fraction(high)
        and dict(out["assignments"])[(0, 1, 0)] == "a"
    )
    feasible(p, out)


def test_lottery_replays_same_pool_seed_independent_of_input_order():
    p = fixture()
    a = module.priority_allocation(p, binding_cap=True, registry=rng())
    other = module.AllocationProblem(
        p.period, p.capacities, p.products, tuple(reversed(p.applicants))
    )
    b = module.priority_allocation(other, binding_cap=True, registry=rng())
    assert a == b and sorted(a["priority_order"]) == [x.key for x in p.applicants]
    assert a["priority_namespace"] == "unit/V9_rationing/A8_priority_v1"
    feasible(p, a)


def test_lottery_does_not_mutate_numpy_global_random_state_or_problem():
    p = fixture()
    before = pickle.dumps(p)
    state = np.random.get_state()
    module.priority_allocation(p, binding_cap=True, registry=rng())
    after = np.random.get_state()
    assert (
        before == pickle.dumps(p)
        and state[0] == after[0]
        and np.array_equal(state[1], after[1])
        and state[2:] == after[2:]
    )


def test_nonbinding_preserves_original_chronology_and_touches_no_RNG(monkeypatch):
    def fail(*a, **kw):
        raise AssertionError("unexpected random draw")

    monkeypatch.setattr(StreamRegistry, "generator", fail)
    p = fixture()
    out = module.priority_allocation(p, binding_cap=False)
    assert (
        out["priority_order"] == tuple(x.key for x in p.applicants)
        and out["priority_namespace"] is None
    )
    feasible(p, out)


def test_native_product_before_outside_tie_and_negative_choice_rules():
    p = module.AllocationProblem(
        0,
        (("a", 1), ("b", 1)),
        (("a", ("a",)), ("b", ("b",))),
        (person(0, (("b", 0, 0), ("a", 0, 0))), person(1, (("a", -1, 100),))),
    )
    out = module.priority_allocation(p, binding_cap=False)
    assert dict(out["assignments"]) == {(0, 0, 0): "b", (0, 1, 0): None}
    assert sampled_choice([0, 0, 0], [0, 0, 0]) == 0
    bound = module.exact_assignment_bound(p, state_limit=100)
    assert bound["allocation_value"] == 0 and all(
        x is None for k, x in bound["assignments"]
    )
    feasible(p, out)
    feasible(p, bound)


def test_signed_priority_objective_is_retained_with_outside_bound():
    p = module.AllocationProblem(
        0, (("a", 1),), (("a", ("a",)),), (person(0, (("a", 1, -5),)),)
    )
    ordered = module.priority_allocation(p, binding_cap=False)
    bound = module.exact_assignment_bound(p, state_limit=100)
    assert ordered["allocation_value"] == -5 and bound["allocation_value"] == 0
    feasible(p, ordered)
    feasible(p, bound)


def test_individually_unwilling_high_objective_is_not_forced():
    p = module.AllocationProblem(
        0, (("a", 1),), (("a", ("a",)),), (person(0, (("a", -1, 1000),)),)
    )
    out = module.exact_assignment_bound(p, state_limit=100)
    assert out["allocation_value"] == 0 and out["unassigned_requests"] == 1


def test_zero_pool_and_zero_stock_preserve_full_supplied_denominators():
    empty = module.AllocationProblem(0, (("a", 0),), (("a", ("a",)),), ())
    out = module.exact_assignment_bound(empty, state_limit=1)
    assert (
        out["allocation_value"] == 0
        and out["supplied_requests"] == 0
        and out["assignments"] == ()
    )
    blocked = module.AllocationProblem(
        0, (("a", 0),), (("a", ("a",)),), (person(0, (("a", 10, 10),)), person(1, ()))
    )
    out = module.priority_allocation(blocked, binding_cap=True, registry=rng())
    assert out["supplied_requests"] == out["unassigned_requests"] == 2
    feasible(blocked, out)


def test_state_limit_returns_no_partial_bound_and_exact_path_calls_no_RNG(monkeypatch):
    def fail(*a, **kw):
        raise AssertionError("unexpected RNG")

    monkeypatch.setattr(StreamRegistry, "generator", fail)
    with pytest.raises(module.AllocationLimitError, match="no partial"):
        module.exact_assignment_bound(fixture(), state_limit=1)
    out = module.exact_assignment_bound(fixture(), state_limit=1000)
    assert out["allocation_value"] == 120


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_caller_state_limit_is_explicit(value):
    with pytest.raises(ValueError):
        module.exact_assignment_bound(fixture(), state_limit=value)


@pytest.mark.parametrize("value", ["NaN", "Infinity", True])
def test_nonfinite_or_boolean_scores_are_not_repaired(value):
    with pytest.raises(ValueError):
        module.Offer("p", 1, value)


def test_duplicate_request_or_physical_offer_cannot_create_extra_passenger():
    with pytest.raises(ValueError):
        module.Applicant((0, 0, 0), (module.Offer("a", 1, 1), module.Offer("a", 2, 2)))
    with pytest.raises(ValueError):
        module.AllocationProblem(
            0,
            (("a", 1),),
            (("a", ("a",)),),
            (person(0, (("a", 1, 1),)), module.Applicant((0, 0, 1), ())),
        )


@pytest.mark.parametrize("stock", [-1, True, 1.5])
def test_unknown_or_fractional_stock_is_not_zero_filled(stock):
    with pytest.raises(ValueError):
        module.AllocationProblem(0, (("a", stock),), (("a", ("a",)),), ())


def test_unknown_incidence_and_wrong_period_are_explicit():
    with pytest.raises(ValueError):
        module.AllocationProblem(0, (("a", 1),), (("p", ("missing",)),), ())
    with pytest.raises(ValueError):
        module.AllocationProblem(0, (("a", 1),), (("p", ("a", "a")),), ())
    with pytest.raises(ValueError):
        module.AllocationProblem(
            0, (("a", 1),), (("p", ("a",)),), (person(0, (), period=1),)
        )
    with pytest.raises(TypeError):
        module.priority_allocation(fixture(), binding_cap=True)


def test_native_incidence_witness_books_without_overselling_shared_legs():
    c = compile_market(
        screen_spec(
            {
                "T": 5,
                "actions": 2,
                "n_spokes": 1,
                "capacity": 20,
                "capacity_support": [3, 4],
            }
        )
    )
    e = MarketEngine(c)
    e.reset(
        tape=ExogenousTape.generate(
            c, 29, registry=StreamRegistry(26975, "unit/V9_rationing_native")
        )
    )
    for state in e.world.carriers:
        state.capacity[:] = 1
    A = c.products_of[0]
    connection = max(A, key=lambda j: len(c.spec.products[j].legs))
    local = min(A, key=lambda j: len(c.spec.products[j].legs))
    requests = (
        person(0, ((c.spec.products[connection].id, 10, 100),)),
        person(1, ((c.spec.products[local].id, 10, 60),)),
    )
    free = {
        c.spec.legs[l].id: np.int64(x)
        for i, state in enumerate(e.world.carriers)
        for l, x in zip(c.legs_of[i], inventory(state, c, i))
    }
    p = module.from_compiled(c, 0, free, requests)
    before = pickle.dumps(e.world)
    out = module.exact_assignment_bound(p, state_limit=1000)
    assert pickle.dumps(e.world) == before
    feasible(p, out)
    for key, product in out["assignments"]:
        if product is None:
            continue
        j = c.product_index[product]
        book(
            e.world,
            int(c.product_carrier[j]),
            j,
            80,
            Request(key[1], int(c.product_market[j]), 0, 0, 0, 0),
            c,
        )
    assert sum(len(s.bookings) for s in e.world.carriers) == 1
    assert all(
        np.all(inventory(state, c, i) >= 0) for i, state in enumerate(e.world.carriers)
    )
    assert e.world.carriers[0].revenue == e.world.carriers[0].cash == 80
    with pytest.raises(ValueError):
        module.from_compiled(c, 0, {}, requests)


def test_receipt_record_serializes_signed_and_fractional_scores_losslessly():
    for value in ("-.100000000000000000000000000000001", "0", "1.25"):
        p = module.AllocationProblem(
            0, (("a", 1),), (("a", ("a",)),), (person(0, (("a", 1, value),)),)
        )
        out = module.priority_allocation(p, binding_cap=False)
        record = json.loads(json.dumps(module.result_record(out)))
        exact = record["allocation_value"]
        assert (
            Fraction(int(exact["numerator"]), int(exact["denominator"]))
            == out["allocation_value"]
        )
        assert (
            not record["native_engine_adopted"]
            and not record["dynamic_or_equilibrium_bound"]
        )
