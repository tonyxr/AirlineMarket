"""A.8/A.5/B0/V9: NEW dated order-benchmark native composition fixtures."""

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
from airline_market.policy.allocation_comparison import AllocationObjective
from airline_market.policy.dated_b1_family import (
    DatedB1Declaration,
    compile_dated_b1_member,
)
from airline_market.policy.dated_demand import (
    DatedDemandMarketEngine,
    compile_dated_demand,
)
from airline_market.policy.dated_rescue import (
    DatedRebookingPolicy,
    DatedRescueMenu,
    UtcWindow,
)
from airline_market.policy.dated_reserve import DatedSeatReserve
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.policy.native_order_bound import NativeOrderLimitError
from airline_market.policy.rationed_native import WholePeriodPriority
from airline_market.policy.regulator import CapPolicy, CapWindow
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from tests.unit import test_dated_demand as dated
from tests.unit import test_rationing_native as native

if os.environ.get("DATED_ORDER_ADMISSION_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_dated_order", os.environ["DATED_ORDER_ADMISSION_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.policy import dated_order_admission as module


def fixture(*, enabled=True, cap=100, limit=100000, score="1", physical=False):
    c, clock, dep = dated.inputs()
    scenario = (
        compile_dated_b1_member(
            c,
            clock,
            dep,
            DatedB1Declaration(
                "exit",
                B1DatedShock("exit", "rival_exit", "B_OH0", 3, rival_carrier="B"),
            ),
        )
        if physical
        else compile_dated_demand(c, clock, dep)
    )
    policy = (
        module.EfficientNativeOrder(
            AllocationObjective(
                "product_scores",
                "unit fixed exact score; not CS/dollars/lifecycle",
                tuple((p.id, score) for p in c.spec.products),
            ),
            limit,
        )
        if enabled
        else None
    )
    kwargs = {
        "cap_policy": CapPolicy(
            (CapWindow(0, 5, tuple((p.id, cap) for p in c.spec.products)),)
        )
    }
    if physical:
        kwargs["dated_rebooking"] = DatedRebookingPolicy(72, 20, interline=True)
    e = module.DatedOrderBenchmarkEngine(scenario, order_policy=policy, **kwargs)
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(45974, "unit/V9_dated_order_admission")
    )
    e.reset(tape=tape)
    parent = DatedDemandMarketEngine(scenario, **kwargs)
    parent.reset(tape=tape)
    return e, parent


@pytest.mark.parametrize("enabled,cap", [(False, 100), (True, 10000)])
def test_disabled_and_no_strict_clip_preserve_complete_parent_world_trace_metrics(
    enabled, cap, monkeypatch
):
    e, parent = fixture(enabled=enabled, cap=cap, limit=1)

    def forbidden(*args, **kwargs):
        raise AssertionError("no order solve without strict clipping")

    monkeypatch.setattr(module, "exact_native_order_bound", forbidden)
    before = pickle.dumps(np.random.get_state())
    while not e.terminal():
        e.step(native.actions(e))
        parent.step(native.actions(parent))
        assert pickle.dumps(e.world) == pickle.dumps(parent.world)
        assert pickle.dumps(e.trace()) == pickle.dumps(parent.trace())
        assert e.coordinate_records() == parent.coordinate_records()
    assert (
        e.metrics() == parent.metrics()
        and pickle.dumps(np.random.get_state()) == before
    )
    assert bool(e.order_admissions()) == enabled
    if enabled:
        assert all(not r.optimal_order_applied for r in e.order_admissions())


def test_binding_executes_full_optimal_order_and_preserves_native_coordinates_no_new_draw(
    monkeypatch,
):
    e, _ = fixture()

    def forbidden(*args, **kwargs):
        raise AssertionError("no new tape or random priority stream during benchmark")

    monkeypatch.setattr(ExogenousTape, "generate", forbidden)
    monkeypatch.setattr(StreamRegistry, "generator", forbidden)
    before = pickle.dumps(np.random.get_state())
    e.step(native.actions(e))
    row = e.order_admissions()[0]
    assert (
        row.optimal_order_applied and row.actual_choices == row.selection["assignments"]
    )
    assert {k for k, p in row.actual_choices} == {
        a.key for a in row.pool.problem.applicants
    }
    by_request = {(0, r.gumbel_row, r.market): r for r in row.pool.requests}
    for cs in e.world.carriers:
        for b in cs.bookings:
            coord = e.request_coordinate(b.record_id)
            request = by_request[coord.stream_key]
            assert b.arrival_rank == coord.native_rank == request.rank
    assert pickle.dumps(np.random.get_state()) == before
    assert all(
        x["initial"] == x["used"] + x["remaining"] for x in row.selection["capacity"]
    )


def test_shared_leg_local_pair_is_actually_booked_before_connection_at_same_posting():
    e, parent = fixture()
    marks = (
        ArrivalMark(0.1, 2, 0, 0, 0),
        ArrivalMark(0.2, 0, 0, 0, 0),
        ArrivalMark(0.3, 1, 0, 0, 0),
    )
    g = e.tape.gumbel.copy()
    g[0, :3, :] = -1e9
    g[0, :3, -1] = 0  # Outside beats every non-target alternative.
    for slot, product in enumerate(["B_OD0_0", "B_OH0", "B_HD0"]):
        g[0, slot, e.compiled.product_index[product]] = 1e9
    g.setflags(write=False)
    tape = replace(e.tape, arrival_marks=(marks, *e.tape.arrival_marks[1:]), gumbel=g)
    for engine in (e, parent):
        engine.reset(tape=tape)
        engine.world.carriers[1].capacity[:] = 1
    e.step(native.actions(e))
    parent.step(native.actions(parent))
    row = e.order_admissions()[0]
    assert row.selection["allocation_value"] == 2
    assert sum(x["chosen"] is not None for x in parent.world.request_log) == 1
    assert {p for k, p in row.actual_choices if p is not None} == {"B_OH0", "B_HD0"}
    assert len(row.actual_choices) == 3 and {r.rank for r in row.pool.requests} == {
        0,
        1,
        2,
    }


def test_signed_fine_negative_scores_are_not_rationed_outside():
    e, _ = fixture(score="-1.0000000000000000000000001")
    e.step(native.actions(e))
    row = e.order_admissions()[0]
    sold = sum(p is not None for k, p in row.actual_choices)
    assert sold > 0 and row.selection["allocation_value"] == -sold * Fraction(
        "1.0000000000000000000000001"
    )


def test_limit_stops_before_ordinary_booking_without_fallback_partial_receipt_or_retry():
    e, _ = fixture(limit=1)
    with pytest.raises(NativeOrderLimitError, match="no partial bound"):
        e.step(native.actions(e))
    assert e.world.t == 0 and e.world.request_log == [] and e.order_admissions() == ()
    assert e.cap_postings()  # native posting happened; no rollback promise


def test_single_posting_outcome_callback_and_no_request_choice_hook(monkeypatch):
    e, _ = fixture()
    posts = []
    outcomes = {}
    post = e.post_actions

    def posting(actions):
        posts.append(1)
        return post(actions)

    e.post_actions = posting
    for i, c in e.controllers.items():

        def outcome(row, i=i):
            outcomes[i] = outcomes.get(i, 0) + 1

        c.observe_outcome = outcome
    with pytest.raises(ValueError, match="genuine full native"):
        e.step(native.actions(e), _requests=[])
    with pytest.raises(ValueError, match="genuine full native"):
        e.step(native.actions(e), _choices=[])
    e.step(native.actions(e))
    assert posts == [1] and outcomes == {i: 1 for i in e.controllers}


def test_empty_binding_pool_retains_complete_empty_record():
    e, _ = fixture()
    e.tape = replace(e.tape, arrival_marks=tuple(() for _ in e.tape.arrival_marks))
    e.step(native.actions(e))
    r = e.order_admissions()[0]
    assert (
        r.pool.requests == ()
        and r.actual_choices == ()
        and r.selection["allocation_value"] == 0
    )
    assert r.selection["visited_states"] == 1 and r.optimal_order_applied


def test_snapshot_branch_reset_and_distinct_objective_limit_contracts():
    e, _ = fixture()
    e.step(native.actions(e))
    snapshot = e.snapshot()
    other = e.branch()
    e.step(native.actions(e))
    assert len(other.order_admissions()) == 1
    e.restore(snapshot)
    assert pickle.dumps(e.world) == pickle.dumps(other.world)
    with pytest.raises(TypeError):
        other.order_admissions()[0].selection["capacity"][0]["remaining"] = 77
    incompatible, _ = fixture(limit=123)
    with pytest.raises(ValueError, match="order objective/state-limit"):
        incompatible.restore(snapshot)
    different, _ = fixture(score="2")
    with pytest.raises(ValueError, match="order objective/state-limit"):
        different.restore(snapshot)
    e.reset(tape=e.tape)
    assert e.order_admissions() == () and e.coordinate_records() == ()
    assert len(other.order_admissions()) == 1


def test_dated_exit_actual_refunds_root_lineage_reentry_and_held_stock_before_capture():
    e, _ = fixture(physical=True)
    dep = e.demand_scenario.departures
    # Construct the NEW composition with native proof-gated held seats.
    hold = DatedSeatReserve(
        dated.hour(24),
        ".10",
        tuple(l.id for l in e.compiled.spec.legs if l.carrier == "A"),
        dep,
    )
    menu = DatedRescueMenu(
        UtcWindow(dated.hour(0), dated.hour(100)),
        "same_route",
        route_fares=tuple((m.id, 99) for m in e.compiled.spec.markets),
        cancelled_departures=UtcWindow(dated.hour(90), dated.hour(100)),
    )
    world = module.DatedOrderBenchmarkEngine(
        e.demand_scenario,
        order_policy=e.order_policy,
        dated_rebooking=e.dated_rebooking,
        dated_menu=menu,
        dated_reserve=hold,
        cap_policy=e.cap_policy,
    )
    g = e.tape.gumbel.copy()
    for t in (0, 1):
        for slot, mark in enumerate(e.tape.arrival_marks[t]):
            products = [
                j
                for j in e.compiled.market_products[mark.market]
                if e.compiled.spec.products[j].carrier == "B"
            ]
            if products:
                g[t, slot, :] = -1e9
                g[t, slot, products[0]] = 1e9
    g.setflags(write=False)
    world.reset(tape=replace(e.tape, gumbel=g))
    world.step(native.actions(world))
    world.step(native.actions(world))
    assert world._passengers
    for passenger in world._passengers.values():
        coord = world.request_coordinate(passenger.original.record_id)
        assert coord.native_rank == passenger.original.arrival_rank
        world.present_cancelled_itinerary(passenger.original)
    expected = {}
    calls = []
    reenter = world._reenter

    def reentry(prices):
        calls.append(1)
        out = reenter(prices)
        expected.update(
            {
                k: n - world._reserve_ledger.get(k, {}).get("held", 0)
                for k, n in world._remaining().items()
            }
        )
        return out

    world._reenter = reentry
    world.step(native.actions(world))
    r = world.order_admissions()[-1]
    assert calls == [1] and dict(r.pool.problem.capacities) == expected
    assert world.rebooking_history and any(
        r["used"] > 0 for r in world.reserve_balance()
    )
    assert (
        world.conservation()["original_booked_passengers"]
        == world.conservation()["active_plus_pending_plus_unserved"]
    )
    branch = world.branch()
    assert pickle.dumps(branch.world) == pickle.dumps(world.world)
    assert branch.coordinate_records() == world.coordinate_records()
    assert all("original_potential_request" in r for r in world.rebooking_history)


def test_lossless_record_true_execution_separate_from_adoption_and_owned_json():
    e, _ = fixture(score=".0000000000000000000000001")
    e.step(native.actions(e))
    row = e.order_admissions()[0]
    saved = json.loads(json.dumps(module.dated_order_record(row)))
    assert (
        saved["native_order_benchmark_executed"]
        and not saved["static_IR_assignment_booked"]
    )
    assert saved["optimal_native_order_applied"] and not saved["random_lottery_applied"]
    assert not saved["primary_B4_adopted"] and saved["scientific_budget"] is None
    assert not saved["lifecycle_welfare_denial_or_misallocation_identified"]
    saved["input_pool"]["requests"].clear()
    assert row.pool.requests


@pytest.mark.parametrize("limit", [0, -1, True, None])
def test_explicit_positive_infrastructure_limit(limit):
    with pytest.raises(ValueError, match="positive order"):
        module.EfficientNativeOrder(
            AllocationObjective("native_choice_gain", "unit"), limit
        )


def test_physical_policy_and_competing_random_priority_and_full_score_support_required():
    e, _ = fixture(physical=True)
    with pytest.raises(ValueError, match="explicit stranded re-entry"):
        module.DatedOrderBenchmarkEngine(e.demand_scenario)
    c, _ = fixture()
    with pytest.raises(ValueError, match="separate contracts"):
        module.DatedOrderBenchmarkEngine(
            c.demand_scenario, ordinary_priority=WholePeriodPriority("unit/no")
        )
    obj = AllocationObjective(
        "product_scores", "unit incomplete", ((c.compiled.spec.products[0].id, 1),)
    )
    with pytest.raises(ValueError, match="EVERY physical product"):
        module.DatedOrderBenchmarkEngine(
            c.demand_scenario, order_policy=module.EfficientNativeOrder(obj, 100)
        )


def test_hurricane_native_choice_objective_recovery_and_parent_date_contract():
    scenario = dated.scenarios(composite=True)["hurricane"]
    obj = AllocationObjective(
        "native_choice_gain", "unit native utility gains, not CS or lifecycle"
    )
    policy = module.EfficientNativeOrder(obj, 100000)
    caps = CapPolicy(
        (CapWindow(0, 5, tuple((p.id, 100) for p in scenario.compiled.spec.products)),)
    )
    e = module.DatedOrderBenchmarkEngine(
        scenario,
        order_policy=policy,
        dated_rebooking=DatedRebookingPolicy(72, 20, interline=True),
        cap_policy=caps,
    )
    e.reset(
        tape=ExogenousTape.generate(
            scenario.compiled,
            31,
            registry=StreamRegistry(45974, "unit/V9_dated_order_hurricane"),
        )
    )
    while not e.terminal():
        e.step(native.actions(e))
    assert len(e.order_admissions()) == 5
    assert e._loss_started and e._loss_finished
    assert all(
        sum(n for k, n in e._loss_removed[i])
        == sum(n for k, n in e._loss_restored[i])
        + sum(n for k, n in e._loss_blocked[i])
        for i in e._loss_started
    )
    assert all(
        r.actual_choices == r.selection["assignments"] for r in e.order_admissions()
    )
    branch = e.branch()
    assert pickle.dumps(branch.world) == pickle.dumps(e.world)
    control, _ = fixture()
    control.order_policy = (
        policy  # Same order declaration cannot authorize different physical dates.
    )
    with pytest.raises(ValueError, match="demand/date/composite"):
        control.restore(e.snapshot())
