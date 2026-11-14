"""A.8/A.5/B0/V9: NEW native dated unit comparison, no efficient booking adoption."""

import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.policy.dated_demand import (
    DatedDemandMarketEngine,
    compile_dated_demand,
)
from airline_market.policy.rationed_native import WholePeriodPriority
from airline_market.policy.rationing import AllocationProblem, Applicant, Offer
from airline_market.policy.rationing_native import NativeOrdinaryPool
from airline_market.policy.regulator import CapPolicy, CapWindow
from airline_market.sim.tape import ExogenousTape
from tests.unit import test_dated_demand as dated
from tests.unit import test_rationing_native as native

if os.environ.get("ALLOCATION_COMPARISON_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_alloc_comparison", os.environ["ALLOCATION_COMPARISON_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.policy import allocation_comparison as module


def fixture(
    *, mode="native_choice_gain", limit=100000, enabled=True, cap=100, score="1"
):
    c, clock, dep = dated.inputs()
    scenario = compile_dated_demand(c, clock, dep)
    values = (
        ()
        if mode == "native_choice_gain"
        else tuple((p.id, score) for p in c.spec.products)
    )
    obj = (
        module.AllocationObjective(
            mode, "unit explicit score; not CS/dollars/lifecycle welfare", values
        )
        if enabled
        else None
    )
    kwargs = {
        "ordinary_priority": WholePeriodPriority("unit/V9_allocation_comparison"),
        "cap_policy": CapPolicy(
            (CapWindow(0, 5, tuple((p.id, cap) for p in c.spec.products)),)
        ),
    }
    e = module.DatedAllocationComparisonEngine(
        scenario, allocation_objective=obj, allocation_state_limit=limit, **kwargs
    )
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(42974, "unit/V9_allocation_comparison")
    )
    e.reset(tape=tape)
    parent = DatedDemandMarketEngine(scenario, **kwargs)
    parent.reset(tape=tape)
    return e, parent


def finish(e):
    while not e.terminal():
        e.step(native.actions(e))
    return e


def test_static_score_comparison_preserves_every_native_choice_world_trace_cash_and_rng():
    e, parent = fixture()
    before = pickle.dumps(np.random.get_state())
    while not e.terminal():
        e.step(native.actions(e))
        parent.step(native.actions(parent))
        assert pickle.dumps(e.world) == pickle.dumps(parent.world)
        assert pickle.dumps(e.trace()) == pickle.dumps(parent.trace())
        row = e.allocation_comparisons()[-1]
        receipt = e.admission_records()[-1]
        assert row.actual_choices == receipt.actual_choices
        assert set(dict(row.actual_choices)) == {
            a.key for a in row.pool.problem.applicants
        }
        assert row.bound["allocation_value"] >= row.actual_score and row.gap >= 0
        assert all(
            x["initial"] == x["used"] + x["remaining"] for x in row.actual_capacity
        )
    assert (
        e.metrics() == parent.metrics()
        and pickle.dumps(np.random.get_state()) == before
    )
    assert len(e.allocation_comparisons()) == 5


def test_disabled_comparison_delegates_complete_native_path_without_records():
    e, parent = fixture(enabled=False)
    finish(e)
    finish(parent)
    assert e.allocation_comparisons() == () and e.metrics() == parent.metrics()
    assert pickle.dumps(e.world) == pickle.dumps(parent.world) and pickle.dumps(
        e.trace()
    ) == pickle.dumps(parent.trace())


def test_explicit_limit_retains_null_bound_and_gap_but_native_period_still_closes():
    e, parent = fixture(limit=1)
    e.step(native.actions(e))
    parent.step(native.actions(parent))
    row = e.allocation_comparisons()[0]
    assert row.status == "state_limit_no_bound" and row.bound is row.gap is None
    assert isinstance(row.actual_score, Fraction) and row.reason
    assert e.world.t == 1 and pickle.dumps(e.world) == pickle.dumps(parent.world)
    assert e.admission_records()[0].actual_choices == row.actual_choices


def test_signed_fine_product_scores_not_rounded_or_clipped_to_zero():
    e, _ = fixture(mode="product_scores", score="-1.0000000000000000000000001")
    e.step(native.actions(e))
    row = e.allocation_comparisons()[0]
    n = sum(p is not None for k, p in row.actual_choices)
    assert n > 0
    expected = -n * Fraction("1.0000000000000000000000001")
    assert (
        row.actual_score == expected
        and row.bound["allocation_value"] == 0
        and row.gap == -expected
    )


def test_json_is_lossless_and_not_a_welfare_or_denial_result():
    e, _ = fixture(mode="product_scores", score="0.0000000000000000000000001")
    e.step(native.actions(e))
    row = e.allocation_comparisons()[0]
    record = module.comparison_record(row)
    saved = json.loads(json.dumps(record))
    assert saved["actual_declared_score"] == {
        "numerator": str(row.actual_score.numerator),
        "denominator": str(row.actual_score.denominator),
    }
    assert saved["objective"]["mode"] == "product_scores"
    assert (
        not saved["native_efficient_assignment_booked"]
        and not saved["lifecycle_welfare_denial_or_misallocation_identified"]
    )
    assert (
        not saved["dynamic_or_equilibrium_bound"] and saved["scientific_budget"] is None
    )
    saved["input_pool"]["applicants"].clear()
    assert row.pool.problem.applicants


def synthetic_pool():
    p = AllocationProblem(
        0,
        (("L1", 1), ("L2", 1)),
        (("C", ("L1", "L2")), ("A", ("L1",)), ("B", ("L2",))),
        (
            Applicant((0, 0, 0), (Offer("C", 1, 1),)),
            Applicant((0, 1, 1), (Offer("A", 1, 1),)),
            Applicant((0, 2, 2), (Offer("B", 1, 1),)),
        ),
    )
    return NativeOrdinaryPool(
        p, (), (), (), "unit synthetic explicit scores", "declared", 31
    )


def test_shared_leg_assignment_gap_independent_of_native_choice_rationing():
    pool = synthetic_pool()
    actual = (((0, 0, 0), "C"), ((0, 1, 1), None), ((0, 2, 2), None))
    row = module.compare_completed_choices(pool, actual, state_limit=100)
    assert row.actual_score == 1 and row.bound["allocation_value"] == 2 and row.gap == 1
    assert dict(row.bound["assignments"]) == {
        (0, 0, 0): None,
        (0, 1, 1): "A",
        (0, 2, 2): "B",
    }


@pytest.mark.parametrize(
    "change", ["missing", "duplicate", "unoffered", "oversold", "unwilling"]
)
def test_actual_witness_must_retain_full_pool_and_willing_capacity_feasible_assignments(
    change,
):
    pool = synthetic_pool()
    a = [((0, 0, 0), "C"), ((0, 1, 1), None), ((0, 2, 2), None)]
    if change == "missing":
        a.pop()
    elif change == "duplicate":
        a[-1] = a[0]
    elif change == "unoffered":
        a[0] = ((0, 0, 0), "A")
    elif change == "oversold":
        a[1] = ((0, 1, 1), "A")
    else:
        first = pool.problem.applicants[0]
        bad = replace(first, offers=(Offer("C", -1, 1),))
        pool = replace(
            pool,
            problem=replace(
                pool.problem, applicants=(bad, *pool.problem.applicants[1:])
            ),
        )
    with pytest.raises(ValueError):
        module.compare_completed_choices(pool, a, state_limit=100)


def test_snapshot_branch_reset_and_immutable_analysis_histories():
    e, _ = fixture()
    e.step(native.actions(e))
    snap = e.snapshot()
    branch = e.branch()
    branch.step(native.actions(branch))
    assert type(branch) is module.DatedAllocationComparisonEngine
    assert (
        len(e.allocation_comparisons()) == 1
        and len(branch.allocation_comparisons()) == 2
    )
    branch.restore(snap)
    assert pickle.dumps(branch.snapshot()) == pickle.dumps(snap)
    with pytest.raises(TypeError):
        branch.allocation_comparisons()[0].bound["allocation_value"] = 0
    branch.reset(tape=branch.tape)
    assert branch.allocation_comparisons() == () and e.allocation_comparisons()
    other, _ = fixture(mode="product_scores")
    with pytest.raises(ValueError, match="objective/state-limit"):
        other.restore(snap)


def test_state_limit_contract_mismatch_refused_before_native_mutation():
    e, _ = fixture()
    e.step(native.actions(e))
    other, _ = fixture(limit=1)
    before = pickle.dumps(other.world)
    with pytest.raises(ValueError, match="objective/state-limit"):
        other.restore(e.snapshot())
    assert pickle.dumps(other.world) == before


def test_empty_full_pool_is_not_dropped_or_reported_as_empirical_exposure():
    e, _ = fixture()
    e.tape = replace(e.tape, arrival_marks=tuple(() for _ in e.tape.arrival_marks))
    e.step(native.actions(e))
    row = e.allocation_comparisons()[0]
    assert (
        row.actual_choices == ()
        and row.actual_score == row.bound["allocation_value"] == row.gap == 0
    )
    assert module.comparison_record(row)["input_pool"]["requests"] == []


def test_no_second_actor_price_or_outcome_forward_and_no_new_tape_generation(
    monkeypatch,
):
    e, parent = fixture()
    calls = []
    original = e.post_actions

    def post(actions):
        calls.append(1)
        return original(actions)

    e.post_actions = post
    outcomes = {}
    for i, controller in e.controllers.items():

        def outcome(row, i=i):
            outcomes[i] = outcomes.get(i, 0) + 1

        controller.observe_outcome = outcome

    def forbidden(*_a, **_k):
        raise AssertionError("comparison cannot generate physical tape")

    monkeypatch.setattr(ExogenousTape, "generate", forbidden)
    e.step(native.actions(e))
    parent.step(native.actions(parent))
    assert calls == [1] and pickle.dumps(e.world) == pickle.dumps(parent.world)
    assert outcomes == {i: 1 for i in e.controllers}


def test_complete_explicit_product_table_and_priority_are_required():
    c, clock, dep = dated.inputs()
    s = compile_dated_demand(c, clock, dep)
    obj = module.AllocationObjective(
        "product_scores", "declared", ((c.spec.products[0].id, 1),)
    )
    with pytest.raises(ValueError, match="EVERY physical product"):
        module.DatedAllocationComparisonEngine(
            s,
            ordinary_priority=WholePeriodPriority("unit/x"),
            allocation_objective=obj,
            allocation_state_limit=100,
        )
    with pytest.raises(ValueError, match="whole-period"):
        module.DatedAllocationComparisonEngine(
            s,
            allocation_objective=module.AllocationObjective(
                "native_choice_gain", "declared"
            ),
            allocation_state_limit=100,
        )


def test_dated_exit_reentry_and_real_hold_use_precede_scored_ordinary_pool():
    from airline_market.policy.dated_b1_family import (
        DatedB1Declaration,
        compile_dated_b1_member,
    )
    from airline_market.policy.dated_rescue import (
        DatedRebookingPolicy,
        DatedRescueMenu,
        UtcWindow,
    )
    from airline_market.policy.dated_reserve import DatedSeatReserve
    from airline_market.policy.dated_shocks import B1DatedShock

    c, clock, dep = dated.inputs()
    s = compile_dated_b1_member(
        c,
        clock,
        dep,
        DatedB1Declaration(
            "exit", B1DatedShock("exit", "rival_exit", "B_OH0", 3, rival_carrier="B")
        ),
    )
    hold = DatedSeatReserve(
        dated.hour(24), ".10", tuple(l.id for l in c.spec.legs if l.carrier == "A"), dep
    )
    menu = DatedRescueMenu(
        UtcWindow(dated.hour(0), dated.hour(100)),
        "same_route",
        route_fares=tuple((m.id, 99) for m in c.spec.markets),
        cancelled_departures=UtcWindow(dated.hour(90), dated.hour(100)),
    )
    e = module.DatedAllocationComparisonEngine(
        s,
        ordinary_priority=WholePeriodPriority("unit/V9_allocation_held"),
        allocation_objective=module.AllocationObjective(
            "native_choice_gain", "current native utility, not lifecycle welfare"
        ),
        allocation_state_limit=100000,
        dated_rebooking=DatedRebookingPolicy(72, 20, interline=True),
        dated_menu=menu,
        dated_reserve=hold,
    )
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(42974, "unit/V9_allocation_held")
    )
    g = tape.gumbel.copy()
    for t in (0, 1):
        for slot, mark in enumerate(tape.arrival_marks[t]):
            products = [
                j
                for j in c.market_products[mark.market]
                if c.spec.products[j].carrier == "B"
            ]
            if products:
                g[t, slot, :] = -1e9
                g[t, slot, products[0]] = 1e9
    g.setflags(write=False)
    e.reset(tape=replace(tape, gumbel=g))
    e.step(native.actions(e))
    e.step(native.actions(e))
    assert e._passengers
    for passenger in e._passengers.values():
        e.present_cancelled_itinerary(passenger.original)
    expected = {}
    calls = []
    original = e._reenter

    def reenter(prices):
        calls.append(1)
        result = original(prices)
        expected.update(
            {
                k: n - e._reserve_ledger.get(k, {}).get("held", 0)
                for k, n in e._remaining().items()
            }
        )
        return result

    e._reenter = reenter
    e.step(native.actions(e))
    row = e.allocation_comparisons()[-1]
    assert calls == [1] and dict(row.pool.problem.capacities) == expected
    assert e.rebooking_history and any(r["used"] > 0 for r in e.reserve_balance())
    assert (
        e.conservation()["original_booked_passengers"]
        == e.conservation()["active_plus_pending_plus_unserved"]
    )


def test_physical_comparison_needs_explicit_dated_reentry_policy():
    from airline_market.policy.dated_b1_family import (
        DatedB1Declaration,
        compile_dated_b1_member,
    )
    from airline_market.policy.dated_shocks import B1DatedShock

    c, clock, dep = dated.inputs()
    s = compile_dated_b1_member(
        c,
        clock,
        dep,
        DatedB1Declaration(
            "exit", B1DatedShock("x", "rival_exit", "B_OH0", 3, rival_carrier="B")
        ),
    )
    with pytest.raises(ValueError, match="explicit stranded re-entry"):
        module.DatedAllocationComparisonEngine(s)
