"""A.8/B0/V9: whole-period native priority and inherited booking closure."""

import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace

import numpy as np
import pytest

from airline_market.agents.fixed import FixedCoefficientController
from airline_market.core.rng import StreamRegistry
from airline_market.policy.cap_evidence import CapEvidenceMarketEngine
from airline_market.policy.regulator import CapPolicy, CapWindow
from airline_market.sim.arrivals import period_requests
from tests.unit import test_cap_evidence as dated
from tests.unit import test_rationing_native as ordinary
from tests.unit import test_triggered_dated as physical

if os.environ.get("RATIONED_NATIVE_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_rationed_native", os.environ["RATIONED_NATIVE_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.policy import rationed_native as module


def policy():
    return module.WholePeriodPriority("unit/V9_native_admission")


def from_base(old, priority):
    e = module.RationedNativeMarketEngine(
        old.compiled,
        ordinary_priority=priority,
        trigger_rule=old.trigger.rule if old.trigger else None,
        mechanical_benchmark=old.trigger.benchmark if old.trigger else None,
        booking_clock=old.booking_clock,
        leg_departures=old.leg_departures,
        dated_rebooking=old.dated_rebooking,
        dated_menu=old.dated_menu,
        dated_reserve=old.dated_reserve,
        exits=old.exits,
        cap_policy=old.cap_policy,
    )
    e.reset(tape=old.tape)
    return e


def poisson(limit=100, priority=True):
    plain = ordinary.fixture()
    refs = tuple((p.id, limit) for p in plain.compiled.spec.products)
    cap = CapPolicy((CapWindow(0, 5, refs),))
    old = CapEvidenceMarketEngine(plain.compiled, cap_policy=cap)
    old.reset(tape=plain.tape)
    return from_base(old, policy() if priority else None), old


@pytest.mark.parametrize("limit,enabled", [(100, False), (1000000, True)])
def test_disabled_or_nonbinding_policy_is_native_world_trace_metrics_identity(
    limit, enabled
):
    e, old = poisson(limit, enabled)
    for _ in range(5):
        e.step(ordinary.actions(e))
        old.step(ordinary.actions(old))
    assert pickle.dumps(e.world) == pickle.dumps(old.world)
    assert (
        pickle.dumps(e.trace()) == pickle.dumps(old.trace())
        and e.metrics() == old.metrics()
    )
    if enabled:
        assert all(
            not r.lottery_applied and r.priority_namespace is None
            for r in e.admission_records()
        )
    else:
        assert e.admission_records() == ()


def test_binding_uses_independent_named_uniform_order_of_full_physical_pool():
    e, _ = poisson()
    requests = tuple(period_requests(e.compiled, e.tape, 0, e.world.exo))
    keys = tuple((0, r.gumbel_row, r.market) for r in requests)
    expected = (
        StreamRegistry(e.tape.seed, policy().namespace + "/A8_priority_v1")
        .generator(0)
        .permutation(len(keys))
    )
    before = pickle.dumps(np.random.get_state())
    e.step(ordinary.actions(e))
    r = e.admission_records()[0]
    assert r.keys == keys and r.priority_order == tuple(keys[int(i)] for i in expected)
    assert r.lottery_applied and r.actual_choices == r.predicted_choices
    assert pickle.dumps(np.random.get_state()) == before
    assert len(r.actual_choices) == len(requests) > 1


def test_native_commit_matches_independent_parent_processing_same_request_order():
    e, old = poisson()
    requests = tuple(period_requests(e.compiled, e.tape, 0, e.world.exo))
    order = (
        StreamRegistry(e.tape.seed, policy().namespace + "/A8_priority_v1")
        .generator(0)
        .permutation(len(requests))
    )
    old.step(ordinary.actions(old), _requests=[requests[int(i)] for i in order])
    e.step(ordinary.actions(e))
    assert pickle.dumps(e.world) == pickle.dumps(old.world)
    assert (
        pickle.dumps(e.trace()) == pickle.dumps(old.trace())
        and e.metrics() == old.metrics()
    )


def test_full_pool_is_retained_across_binding_and_nonbinding_cap_contrasts():
    a, _ = poisson(100)
    b, _ = poisson(1000000)
    a.step(ordinary.actions(a))
    b.step(ordinary.actions(b))
    ra, rb = a.admission_records()[0], b.admission_records()[0]
    assert ra.keys == rb.keys and ra.request_coordinates == rb.request_coordinates
    assert ra.lottery_applied and not rb.lottery_applied
    assert len(ra.keys) == len(ra.actual_choices) == len(rb.actual_choices)


def test_outside_preferring_and_zero_stock_applicants_are_not_prefiltered():
    e, _ = poisson()
    requests = tuple(period_requests(e.compiled, e.tape, 0, e.world.exo))
    for cs in e.world.carriers:
        cs.capacity[:] = 0
    marks = e.tape.gumbel.copy()
    for r in requests:
        marks[0, r.gumbel_row, -1] = 1e9
    marks.setflags(write=False)
    e.tape = replace(e.tape, gumbel=marks)
    e.step(ordinary.actions(e))
    r = e.admission_records()[0]
    assert len(r.keys) == len(requests) > 1 and all(
        p is None for _, p in r.actual_choices
    )
    assert all(n == 0 for _, n in r.free_capacity_before)


def test_empty_native_bernoulli_pool_still_closes_native_period():
    base, _, _ = dated.fixture(trigger=False)
    e = from_base(base, policy())
    q = e.tape.request_uniforms.copy()
    q[0, 0] = 1
    q.setflags(write=False)
    e.tape = replace(e.tape, request_uniforms=q)
    result = e.step(physical.actions(e))
    r = e.admission_records()[0]
    assert (
        result.period == 0
        and e.world.t == 1
        and r.keys == r.priority_order == r.actual_choices == ()
    )


def test_forced_request_choice_hooks_refused_before_pricing_or_booking():
    e, _ = poisson()
    before = pickle.dumps(e.world)
    with pytest.raises(ValueError, match="full physical"):
        e.step(ordinary.actions(e), _requests=[])
    with pytest.raises(ValueError, match="full physical"):
        e.step(ordinary.actions(e), _choices=[])
    assert pickle.dumps(e.world) == before and e.cap_postings() == ()


def test_native_posting_and_outcome_callbacks_happen_once_per_period():
    e, _ = poisson()
    acts = {}
    outcomes = {}
    rule = {}
    for i in e.controllers:
        ctrl = FixedCoefficientController((0,) * len(e.rules[i].spec.theta_bounds))
        original = ctrl.act

        def act(obs, i=i, original=original):
            acts[i] = acts.get(i, 0) + 1
            return original(obs)

        def outcome(row, i=i):
            outcomes[i] = outcomes.get(i, 0) + 1

        ctrl.act = act
        ctrl.observe_outcome = outcome
        e.controllers[i] = ctrl
        native = e.rules[i].prices

        def prices(theta, obs, i=i, native=native):
            rule[i] = rule.get(i, 0) + 1
            return native(theta, obs)

        e.rules[i].prices = prices
    e.step()
    assert acts == rule == outcomes == {0: 1, 1: 1} and len(e.cap_postings()) == 1


def test_genuine_reentry_and_held_stock_run_before_ordinary_pool_capture():
    base, reports, _ = dated.fixture(start=1)
    e = from_base(base, policy())
    q = e.tape.request_uniforms.copy()
    q[0, 0] = 0
    q[1, 0] = 0
    q.setflags(write=False)
    e.tape = replace(e.tape, request_uniforms=q)
    request = period_requests(e.compiled, e.tape, 0, e.world.exo)[0]
    j = next(
        j
        for j in e.compiled.market_products[request.market]
        if int(e.compiled.product_carrier[j]) == 1
    )
    marks = e.tape.gumbel.copy()
    marks[0, request.gumbel_row, :] = -1e9
    marks[0, request.gumbel_row, j] = 1e9
    marks.setflags(write=False)
    e.tape = replace(e.tape, gumbel=marks)
    e.step(physical.actions(e))
    assert len(e._passengers) == 1
    e.present_cancelled_itinerary(next(iter(e._passengers.values())).original)
    physical.publish(e, reports)
    expected = {}
    calls = []
    native = e._reenter

    def reenter(prices):
        calls.append(1)
        purchases = native(prices)
        expected.update(
            {
                k: n - e._reserve_ledger.get(k, {}).get("held", 0)
                for k, n in e._remaining().items()
            }
        )
        return purchases

    e._reenter = reenter
    e.step(physical.actions(e))
    r = e.admission_records()[-1]
    assert len(calls) == 1 and dict(r.free_capacity_before) == expected
    assert e.rebooking_history and any(x["used"] for x in e.reserve_balance())
    assert (
        e.conservation()["original_booked_passengers"]
        == e.conservation()["active_plus_pending_plus_unserved"]
    )
    for cs in e.world.carriers:
        assert cs.cash == pytest.approx(
            cs.revenue - cs.refunds - cs.penalties, abs=1e-10
        )


def test_native_bookings_keep_original_admission_rank_not_lottery_position():
    e, _ = poisson()
    e.step(ordinary.actions(e))
    r = e.admission_records()[0]
    by_key = {(0, x[4], x[1]): x for x in r.request_coordinates}
    chosen = [(key, p) for key, p in r.actual_choices if p is not None]
    all_books = [b for cs in e.world.carriers for b in cs.bookings]
    assert chosen and len(chosen) == len(all_books)
    for (key, p), b in zip(chosen, sorted(all_books, key=lambda b: b.record_id)):
        assert (
            e.compiled.spec.products[b.product].id == p
            and b.arrival_rank == by_key[key][0]
        )


def test_snapshot_branch_restore_and_reset_replay_same_priority_without_shared_history():
    e, _ = poisson()
    e.step(ordinary.actions(e))
    snap = e.snapshot()
    other = e.branch()
    assert type(other) is module.RationedNativeMarketEngine
    a = e.step(ordinary.actions(e))
    b = other.step(ordinary.actions(other))
    assert a == b and e.admission_records() == other.admission_records()
    other.restore(snap)
    assert len(other.admission_records()) == 1 and len(e.admission_records()) == 2
    other.step(ordinary.actions(other))
    assert e.admission_records() == other.admission_records()
    other.reset(tape=other.tape)
    assert other.admission_records() == ()


def test_different_priority_contract_restore_refused_before_native_mutation():
    e, _ = poisson()
    e.step(ordinary.actions(e))
    other, _ = poisson()
    other.ordinary_priority = module.WholePeriodPriority("unit/other")
    before = pickle.dumps(other.world)
    with pytest.raises(ValueError, match="priority contract"):
        other.restore(e.snapshot())
    assert pickle.dumps(other.world) == before
    with pytest.raises(TypeError):
        e.restore(e.snapshot().native)


def test_failure_before_complete_posting_does_not_publish_admission_record():
    base, _, _ = dated.fixture(start=0)
    e = from_base(base, policy())
    before = e.world.t
    with pytest.raises(ValueError, match="public regulator report"):
        e.step(physical.actions(e))
    assert e.world.t == before and e.admission_records() == ()


def test_admission_json_is_analyst_only_and_has_no_primary_welfare_claim():
    e, _ = poisson()
    e.step(ordinary.actions(e))
    x = json.loads(json.dumps(module.admission_record(e.admission_records()[0])))
    assert (
        len(x["keys"]) == len(x["actual_choices"])
        and x["cap_posting"]["any_price_constraint_binding"]
    )
    assert (
        not x["primary_B4_policy_adopted"]
        and not x["excess_demand_measured"]
        and not x["efficient_assignment_adopted"]
    )
    with pytest.raises(TypeError):
        module.admission_record({})


def test_empty_namespace_or_wrong_policy_type_is_not_silently_selected():
    with pytest.raises(ValueError):
        module.WholePeriodPriority(" ")
    e, _ = poisson()
    with pytest.raises(TypeError):
        module.RationedNativeMarketEngine(e.compiled, ordinary_priority=True)


def test_uninitialized_and_terminal_steps_do_not_create_partial_admission_receipts():
    e, _ = poisson()
    e.world = None
    with pytest.raises(ValueError, match="nonterminal"):
        e.step(ordinary.actions(e))
    e.reset(tape=e.tape)
    for _ in range(5):
        e.step(ordinary.actions(e))
    records = e.admission_records()
    with pytest.raises(ValueError, match="nonterminal"):
        e.step(ordinary.actions(e))
    assert e.admission_records() == records
