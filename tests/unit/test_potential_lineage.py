"""A.5/A.8/B0/V9: physical potential slots through native booking/cancellation."""

import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.policy.rationed_native import WholePeriodPriority
from airline_market.sim.rebooking import RebookingMarketEngine
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule
from tests.unit import test_cap_evidence as dated
from tests.unit import test_rationed_native as admission
from tests.unit import test_rationing_native as ordinary
from tests.unit import test_triggered_dated as physical

if os.environ.get("POTENTIAL_LINEAGE_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_potential_lineage", os.environ["POTENTIAL_LINEAGE_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.policy import potential_lineage as module


def from_base(old):
    e = module.PotentialLineageMarketEngine(
        old.compiled,
        ordinary_priority=old.ordinary_priority,
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


def controlled(*, high=False, priority=False, extra_exit=False):
    old, reports, _ = dated.fixture(trigger=False, extra_exit=extra_exit)
    base = old.compiled.spec
    normal = base.exogenous.states[0]
    exo = replace(
        base.exogenous,
        states=(replace(normal, name="low"), replace(normal, name="high")),
        initial=(1.0, 0.0),
    )
    demand = replace(
        base.demand,
        mode="poisson",
        arrival_probability=None,
        market_weights=None,
        intensity={m.id: PeriodSchedule.parse({"constant": 2.0}) for m in base.markets},
        exogenous_multipliers={"low": 0.25, "high": 1.0},
    )
    c = compile_market(replace(base, exogenous=exo, demand=demand))
    e = module.PotentialLineageMarketEngine(
        c,
        ordinary_priority=WholePeriodPriority("unit/V9_potential_priority")
        if priority
        else None,
        booking_clock=old.booking_clock,
        leg_departures=old.leg_departures,
        dated_rebooking=old.dated_rebooking,
        dated_menu=old.dated_menu,
        dated_reserve=old.dated_reserve,
        exits=old.exits,
    )
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(30974, "unit/V9_potential_lineage")
    )
    m = c.market_index["OD0_0"]
    j = c.product_index["B_OD0_0"]
    marks = (ArrivalMark(0.1, m, 0, 0.5, 0.1), ArrivalMark(0.2, m, 0, 0.0, 0.1))
    g = tape.gumbel.copy()
    g[0, :, :] = -1e9
    g[0, :, j] = 1e9
    g.setflags(write=False)
    tape = replace(tape, arrival_marks=(marks, (), (), (), ()), gumbel=g)
    e.reset(tape=tape)
    e.world.exo = int(high)
    return e, reports


def target(e):
    return next(
        (k, c)
        for k, c in e.coordinate_records()
        if c.stream_key == (0, 1, e.compiled.market_index["OD0_0"])
    )


@pytest.mark.parametrize("priority", [False, True])
def test_new_no_shock_world_trace_metrics_and_booking_rank_identity(priority):
    old, _ = admission.poisson(priority=priority)
    e = from_base(old)
    for _ in range(5):
        e.step(ordinary.actions(e))
        old.step(ordinary.actions(old))
    assert pickle.dumps(e.world) == pickle.dumps(old.world)
    assert (
        pickle.dumps(e.trace()) == pickle.dumps(old.trace())
        and e.metrics() == old.metrics()
    )
    assert len(e.coordinate_records()) == e.world.next_record_id > 0
    assert e.admission_records() == old.admission_records()


def test_sidecar_is_available_before_first_end_period_native_cancellation():
    e, _ = controlled()
    seen = []
    native = e._record_stranded

    def stranded(bookings, period, cause):
        bookings = tuple(bookings)
        seen.extend(
            (b.record_id, e.request_coordinate(b.record_id).stream_key)
            for b in bookings
        )
        return native(bookings, period, cause)

    e._record_stranded = stranded
    e.step(physical.actions(e))
    root, c = target(e)
    p = e._passengers[root]
    assert (
        seen == [(root, c.stream_key)]
        and p.original.arrival_rank == 0
        and c.potential_slot == 1
    )
    assert e.world.carriers[1].refunds == p.original.fare


def test_cross_thinning_common_customer_uses_identical_native_reentry_marks():
    a, _ = controlled(high=False)
    b, _ = controlled(high=True)
    a.step(physical.actions(a))
    b.step(physical.actions(b))
    ia, ca = target(a)
    ib, cb = target(b)
    assert (
        ia != ib
        and ca.stream_key == cb.stream_key
        and ca.native_rank == 0
        and cb.native_rank == 1
    )
    ta, ga = a._marks(a._passengers[ia], 1)
    tb, gb = b._marks(b._passengers[ib], 1)
    assert pickle.dumps(ta) == pickle.dumps(tb) and np.array_equal(ga, gb)
    # New unit counterexample documents why the frozen rank-key scope is limited.
    _, old_a = RebookingMarketEngine._marks(a, a._passengers[ia], 1)
    _, old_b = RebookingMarketEngine._marks(b, b._passengers[ib], 1)
    assert not np.array_equal(old_a, old_b)
    assert (
        a._passengers[ia].original.arrival_rank == 0
        and b._passengers[ib].original.arrival_rank == 1
    )


def test_native_marks_are_delegated_once_with_private_view_only(monkeypatch):
    e, _ = controlled()
    e.step(physical.actions(e))
    root, c = target(e)
    passenger = e._passengers[root]
    original = passenger.original
    native = RebookingMarketEngine._marks
    calls = []

    def marks(self, p, t):
        calls.append(p.original.arrival_rank)
        return native(self, p, t)

    monkeypatch.setattr(RebookingMarketEngine, "_marks", marks)
    before = pickle.dumps(e.world)
    global_before = pickle.dumps(np.random.get_state())
    e._marks(passenger, 1)
    assert (
        calls == [c.potential_slot]
        and passenger.original is original
        and original.arrival_rank == c.native_rank
    )
    assert (
        pickle.dumps(e.world) == before
        and pickle.dumps(np.random.get_state()) == global_before
    )


def test_replacement_and_repeated_cancellation_resolve_to_same_original_coordinate():
    e, _ = controlled(extra_exit=True)
    e.step(physical.actions(e))
    root, c = target(e)
    e.present_cancelled_itinerary(e._passengers[root].original)
    e.step(physical.actions(e))
    replacements = tuple(k for k, v in e._replacement_roots.items() if v == root)
    assert replacements and all(e.request_coordinate(k) == c for k in replacements)
    assert (
        e._passengers[root].original.record_id == root
        and e._passengers[root].active_record is None
    )
    row = e.rebooking_history[0]
    assert (
        row["original_potential_request"] == c.stream_key
        and row["native_original_rank"] == c.native_rank
    )
    assert (
        e.conservation()["original_booked_passengers"]
        == e.conservation()["active_plus_pending_plus_unserved"]
    )
    for cs in e.world.carriers:
        assert cs.cash == pytest.approx(
            cs.revenue - cs.refunds - cs.penalties, abs=1e-10
        )


def test_lineage_composes_with_whole_period_native_priority():
    e, _ = controlled(high=True, priority=True)
    e.step(physical.actions(e))
    assert len(e.coordinate_records()) == 2 and len(e.admission_records()) == 1
    assert (
        e.admission_records()[0].actual_choices
        == e.admission_records()[0].predicted_choices
    )
    assert [x.native_rank for _, x in e.coordinate_records()] == [0, 1]


def test_unknown_preexisting_ticket_refused_without_rank_fallback():
    e, _ = controlled()
    e.step(physical.actions(e))
    root, _ = target(e)
    p = e._passengers[root]
    del e._physical_coordinates[root]
    with pytest.raises(ValueError, match="no rank fallback"):
        e._marks(p, 1)
    with pytest.raises(ValueError, match="no rank fallback"):
        e.request_coordinate(99999)


def test_mutated_native_original_is_not_authenticated_by_coordinate_only():
    e, _ = controlled()
    e.step(physical.actions(e))
    root, _ = target(e)
    p = e._passengers[root]
    with pytest.raises(ValueError, match="native booking differs"):
        e._marks(replace(p, original=replace(p.original, segment=1)), 1)


def test_forced_hooks_refused_before_native_posting():
    old, _ = admission.poisson(priority=False)
    e = from_base(old)
    before = pickle.dumps(e.world)
    with pytest.raises(ValueError, match="genuine native"):
        e.step(ordinary.actions(e), _requests=[])
    with pytest.raises(ValueError, match="genuine native"):
        e.step(ordinary.actions(e), _choices=[])
    assert pickle.dumps(e.world) == before and e.coordinate_records() == ()


def test_empty_and_outside_requests_do_not_fabricate_coordinates():
    e, _ = controlled()
    t = e.tape
    g = t.gumbel.copy()
    g[0, :, :] = -1e9
    g[0, :, -1] = 1e9
    g.setflags(write=False)
    e.tape = replace(t, gumbel=g)
    e.step(physical.actions(e))
    assert e.world.next_record_id == 0 and e.coordinate_records() == ()
    e.step(physical.actions(e))
    assert e.coordinate_records() == () and e.world.t == 2


def test_snapshot_branch_restore_reset_own_sidecars_and_reentry_law():
    e, _ = controlled()
    e.step(physical.actions(e))
    root, c = target(e)
    e.present_cancelled_itinerary(e._passengers[root].original)
    snap = e.snapshot()
    other = e.branch()
    assert type(other) is module.PotentialLineageMarketEngine
    a = e.step(physical.actions(e))
    b = other.step(physical.actions(other))
    assert a == b and pickle.dumps(e.world) == pickle.dumps(other.world)
    assert e.coordinate_records() == other.coordinate_records()
    other._physical_coordinates.clear()
    assert e.request_coordinate(root) == c
    other.restore(snap)
    assert other.request_coordinate(root) == c
    other.reset(tape=other.tape)
    assert other.coordinate_records() == () and e.coordinate_records()
    with pytest.raises(TypeError):
        e.restore(snap.native)
    with pytest.raises(ValueError, match="coordinate rule"):
        e.restore(replace(snap, coordinate_rule="other"))


def test_analyst_json_preserves_both_coordinates_without_pnr_claim():
    e, _ = controlled()
    e.step(physical.actions(e))
    x = json.loads(json.dumps(module.coordinate_record(e)))
    row = x["ordinary_tickets"][0]
    assert row["native_rank"] == 0 and row["potential_slot"] == 1
    assert row["stream_key"] == [0, 1, e.compiled.market_index["OD0_0"]]
    assert not x["old_receipts_rewritten"] and not x["empirical_identity_authenticated"]
