"""A.5/B0/B1/V9: native temporary installed loss, refunds, lineage and recovery."""

import importlib.util
import os
import pickle
import sys
from dataclasses import replace

import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.policy.potential_lineage import PotentialLineageMarketEngine
from airline_market.policy.rationed_native import WholePeriodPriority
from airline_market.policy.regulator import CapPolicy, CapWindow
from airline_market.sim.disruption import RivalExit
from airline_market.sim.tape import ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.model import CapacityEventSpec, CapacityProcessSpec
from tests.unit import test_cap_evidence as dated
from tests.unit import test_rationing_native as ordinary
from tests.unit import test_triggered_dated as physical

if os.environ.get("CAPACITY_WINDOWS_V2_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_capacity_windows_v2", os.environ["CAPACITY_WINDOWS_V2_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.policy import capacity_windows_v2 as module


def fixture(windows=(), *, dated_policy=False, exits=(), priority=False):
    if dated_policy:
        old, _, _ = dated.fixture(trigger=False)
        kwargs = {
            "booking_clock": old.booking_clock,
            "leg_departures": old.leg_departures,
            "dated_rebooking": old.dated_rebooking,
            "dated_menu": old.dated_menu,
            "dated_reserve": old.dated_reserve,
        }
        c, tape = old.compiled, old.tape
    else:
        old = ordinary.fixture()
        c, tape = old.compiled, old.tape
        kwargs = {}
    kwargs.update(
        exits=exits,
        ordinary_priority=WholePeriodPriority("unit/V9_capacity_windows")
        if priority
        else None,
    )
    e = module.CapacityWindowMarketEngine(c, capacity_windows=windows, **kwargs)
    e.reset(tape=tape)
    baseline = PotentialLineageMarketEngine(c, **kwargs)
    baseline.reset(tape=tape)
    return e, baseline


def window(start=1, stop=3, share=".3", legs=("B_OH0", "B_HD0")):
    return module.CapacityLossWindow(start, stop, share, legs)


def installed(e, leg):
    l = e.compiled.leg_index[leg]
    i = int(e.compiled.leg_carrier[l])
    return int(e.world.carriers[i].capacity[e.compiled.legs_of[i].index(l)])


def original_on_B(e):
    q = e.tape.request_uniforms.copy()
    q[:, 0] = 1
    q[0, 0] = 0
    q[0, 1] = 0.999
    q[0, 2] = 0
    q.setflags(write=False)
    g = e.tape.gumbel.copy()
    g[0, 0, :] = -1e9
    g[0, 0, e.compiled.product_index["B_OD0_0"]] = 1e9
    g.setflags(write=False)
    e.tape = replace(e.tape, request_uniforms=q, gumbel=g)
    e.step(physical.actions(e))
    return (
        next(iter(e._passengers.values())).original
        if e._passengers
        else next(b for cs in e.world.carriers for b in cs.bookings)
    )


@pytest.mark.parametrize(
    "windows", [(), (window(0, 3, 0),), (window(0, 3, ".1", ("B_HD0",)),)]
)
def test_no_policy_zero_share_or_rounded_zero_native_world_trace_metrics_identity(
    windows,
):
    e, b = fixture(windows)
    for _ in range(5):
        e.step(ordinary.actions(e))
        b.step(ordinary.actions(b))
    assert pickle.dumps(e.world) == pickle.dumps(b.world)
    assert (
        pickle.dumps(e.trace()) == pickle.dumps(b.trace())
        and e.metrics() == b.metrics()
    )
    assert all(r["removed"] == 0 for r in e.capacity_window_balance())


def test_floor_loss_is_from_installed_capacity_not_remaining_bookable_stock():
    e, _ = fixture((window(1, 3, ".3"),), dated_policy=True)
    original_on_B(e)
    rows = {r["leg"]: r for r in e.capacity_window_balance()}
    assert rows["B_OH0"]["installed_at_reveal"] == 20 and rows["B_OH0"]["removed"] == 6
    assert installed(e, "B_OH0") == 14
    assert rows["B_HD0"]["removed"] == int(rows["B_HD0"]["installed_at_reveal"] * 0.3)


def test_whole_itinerary_refund_and_stranding_before_next_period_reentry():
    e, _ = fixture((window(1, 3, 1),), dated_policy=True)
    b = original_on_B(e)
    p = e._passengers[b.record_id]
    assert p.original == b and p.first_stranded_at == 1 and p.active_record is None
    assert e.request_coordinate(b.record_id).stream_key == (
        0,
        0,
        int(e.compiled.product_market[b.product]),
    )
    assert e.world.carriers[1].refunds == b.fare
    event = e.capacity_window_events()[0]
    assert event.canceled_ids == (b.record_id,) and event.refunds == b.fare
    assert installed(e, "B_OH0") == installed(e, "B_HD0") == 0


def test_native_reentry_is_not_reset_by_duplicate_boundary_cancellation():
    e, _ = fixture((window(1, 3, 1),), dated_policy=True)
    b = original_on_B(e)
    e.present_cancelled_itinerary(b)
    e.step(physical.actions(e))
    active = e._passengers[b.record_id].active_record
    assert active is not None and any(
        x.record_id == active for cs in e.world.carriers for x in cs.bookings
    )
    assert (
        len(e.rebooking_history) == 1
        and e._passengers[b.record_id].first_stranded_at == 1
    )
    assert (
        e.conservation()["original_booked_passengers"]
        == e.conservation()["active_plus_pending_plus_unserved"]
    )
    assert any(x["used"] for x in e.reserve_balance())


def test_recovery_restores_installed_debit_not_cancelled_tickets_or_passenger_deadline():
    e, _ = fixture((window(1, 3, 1),), dated_policy=True)
    b = original_on_B(e)
    deadline = e._passengers[b.record_id].deadline
    e.step(physical.actions(e))
    e.step(physical.actions(e))
    assert (
        e.world.t == 3
        and installed(e, "B_OH0") == 20
        and installed(e, "B_HD0") in (3, 4)
    )
    assert not any(
        x.record_id == b.record_id for cs in e.world.carriers for x in cs.bookings
    )
    assert e._passengers[b.record_id].deadline == deadline
    assert all(
        r["removed"] == r["restored"] + r["blocked_by_exit"] + r["outstanding"]
        and r["finished"]
        for r in e.capacity_window_balance()
    )
    assert all(r["outstanding"] == 0 for r in e.capacity_window_balance())


def test_repeated_begin_observe_does_not_repeat_debit_or_refund():
    e, _ = fixture((window(1, 3, 1),), dated_policy=True)
    b = original_on_B(e)
    before = (e.world.carriers[1].refunds, len(e.capacity_window_events()))
    e.begin_period(physical.actions(e))
    e.begin_period(physical.actions(e))
    assert (e.world.carriers[1].refunds, len(e.capacity_window_events())) == before
    assert e._passengers[b.record_id].first_stranded_at == 1


def test_permanent_exit_dominates_expiry_without_capacity_reopening():
    e, _ = fixture((window(1, 3, 1),), dated_policy=True, exits=(RivalExit("B", 2),))
    b = original_on_B(e)
    e.step(physical.actions(e))
    e.step(physical.actions(e))
    assert installed(e, "B_OH0") == installed(e, "B_HD0") == 0
    rows = e.capacity_window_balance()
    assert all(
        r["restored"] == 0 and r["blocked_by_exit"] == r["removed"] for r in rows
    )
    assert e.world.carriers[1].refunds == b.fare


def test_beyond_horizon_stop_retains_outstanding_debit():
    e, _ = fixture((window(1, 8, ".3"),))
    for _ in range(5):
        e.step(ordinary.actions(e))
    assert all(
        r["outstanding"] == r["removed"] and not r["finished"]
        for r in e.capacity_window_balance()
    )
    assert installed(e, "B_OH0") == 14


def test_adjacent_windows_restore_before_fresh_reveal():
    e, _ = fixture((window(0, 2, ".3"), window(2, 4, ".5")))
    e.step(ordinary.actions(e))
    e.step(ordinary.actions(e))
    second = next(
        r
        for r in e.capacity_window_balance()
        if r["window"] == 1 and r["leg"] == "B_OH0"
    )
    assert (
        second["installed_at_reveal"] == 20
        and second["removed"] == 10
        and installed(e, "B_OH0") == 10
    )
    assert [x.kind for x in e.capacity_window_events()[:3]] == [
        "remove",
        "restore",
        "remove",
    ]


def test_both_carrier_multi_leg_loss_uses_native_whole_booking_resolution_once():
    plain = ordinary.fixture()
    legs = tuple(l.id for l in plain.compiled.spec.legs)
    e, _ = fixture((window(1, 3, 1, legs),))
    e.step(ordinary.actions(e))
    events = e.capacity_window_events()
    assert len(events) == 1 and len(events[0].canceled_ids) == len(
        set(events[0].canceled_ids)
    )
    assert all(installed(e, l) == 0 for l in legs)
    for cs in e.world.carriers:
        assert cs.cash == pytest.approx(
            cs.revenue - cs.refunds - cs.penalties, abs=1e-10
        )
    assert (
        e.conservation()["ever_booked_tickets"]
        == e.conservation()["active_plus_stranded"]
    )


def test_returning_capacity_never_refills_reserve_quota_lost_during_outage():
    e, _ = fixture((window(2, 3, 1, ("A_OH0", "A_HD0")),), dated_policy=True)
    q = e.tape.request_uniforms.copy()
    q[:, 0] = 1
    q.setflags(write=False)
    e.tape = replace(e.tape, request_uniforms=q)
    e.step(physical.actions(e))
    e.step(physical.actions(e))
    lost = e.reserve_balance()
    assert all(r["capacity_lost"] == r["initial"] and r["held"] == 0 for r in lost)
    e.step(physical.actions(e))
    restored = e.reserve_balance()
    assert all(r["held"] == 0 and r["capacity_lost"] == r["initial"] for r in restored)
    assert installed(e, "A_OH0") == 20


def test_snapshot_branch_restore_reset_own_windows_ledgers_and_contracts():
    e, _ = fixture((window(1, 3, ".3"),))
    e.step(ordinary.actions(e))
    snap = e.snapshot()
    other = e.branch()
    assert type(other) is module.CapacityWindowMarketEngine
    a = e.step(ordinary.actions(e))
    b = other.step(ordinary.actions(other))
    assert a == b
    assert e.capacity_window_balance() == other.capacity_window_balance()
    other._loss_removed.clear()
    assert e.capacity_window_balance()
    other.restore(snap)
    assert len(other.capacity_window_events()) == 1
    other.reset(tape=other.tape)
    assert other.capacity_window_balance() == ()
    with pytest.raises(TypeError):
        e.restore(snap.native)
    with pytest.raises(ValueError, match="contract"):
        e.restore(replace(snap, native=replace(snap.native, windows=())))


def test_invalid_and_overlapping_windows_are_scoped_refusals():
    with pytest.raises(ValueError):
        window(1, 1)
    with pytest.raises(ValueError):
        window(1, 2, 1.1)
    with pytest.raises(ValueError):
        window(1, 2, ".3", ("B_OH0", "B_OH0"))
    with pytest.raises(ValueError, match="overlapping"):
        fixture((window(0, 3), window(2, 4)))
    with pytest.raises(ValueError, match="outside"):
        fixture((window(5, 7),))
    with pytest.raises(ValueError, match="outside"):
        fixture((window(1, 3, ".3", ("absent",)),))


def test_native_event_on_target_leg_refused_before_running_incompatible_overlay():
    c = ordinary.fixture().compiled
    event = CapacityEventSpec(
        trigger={"hazard": {"probability": 1.0}}, effects=(("B_OH0", -1),)
    )
    c = compile_market(
        replace(c.spec, capacity_process=CapacityProcessSpec(events=(event,)))
    )
    with pytest.raises(ValueError, match="native capacity events"):
        module.CapacityWindowMarketEngine(c, capacity_windows=(window(),))


def test_native_event_on_other_leg_remains_native_and_not_restored_with_overlay():
    c = ordinary.fixture().compiled
    event = CapacityEventSpec(
        trigger={"hazard": {"probability": 1.0}}, effects=(("A_OH0", -1),)
    )
    c = compile_market(
        replace(c.spec, capacity_process=CapacityProcessSpec(events=(event,)))
    )
    e = module.CapacityWindowMarketEngine(c, capacity_windows=(window(),))
    b = PotentialLineageMarketEngine(c)
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(31974, "unit/V9_capacity_events")
    )
    e.reset(tape=tape)
    b.reset(tape=tape)
    for _ in range(3):
        e.step(ordinary.actions(e))
        b.step(ordinary.actions(b))
    assert installed(e, "A_OH0") == installed(b, "A_OH0") == 19
    assert installed(e, "B_OH0") == 20 and all(
        r["outstanding"] == 0 for r in e.capacity_window_balance()
    )


def test_whole_period_priority_and_fixed_pool_witness_use_post_loss_stock():
    c = ordinary.fixture().compiled
    cap = CapPolicy((CapWindow(0, 5, tuple((p.id, 100) for p in c.spec.products)),))
    e = module.CapacityWindowMarketEngine(
        c,
        capacity_windows=(window(0, 2, ".3"),),
        ordinary_priority=WholePeriodPriority("unit/V9_loss_priority"),
        cap_policy=cap,
    )
    e.reset(tape=ordinary.fixture().tape)
    e.step(ordinary.actions(e))
    r = e.admission_records()[0]
    assert r.lottery_applied and r.actual_choices == r.predicted_choices
    assert dict(r.free_capacity_before)["B_OH0"] == 14


def test_permanent_exit_before_window_keeps_zero_loss_and_zero_recovery():
    e, b = fixture((window(),), exits=(RivalExit("B", 0),))
    for _ in range(5):
        e.step(ordinary.actions(e))
        b.step(ordinary.actions(b))
    assert pickle.dumps(e.world) == pickle.dumps(b.world) and pickle.dumps(
        e.trace()
    ) == pickle.dumps(b.trace())
    assert all(
        r["installed_at_reveal"] == r["removed"] == r["restored"] == 0
        for r in e.capacity_window_balance()
    )


def test_snapshot_exit_contract_refused_before_native_restore():
    e, _ = fixture((window(),))
    e.step(ordinary.actions(e))
    snap = e.snapshot()
    before = pickle.dumps(e.world)
    with pytest.raises(ValueError, match="contract"):
        e.restore(
            replace(snap, native=replace(snap.native, exits=(RivalExit("B", 0),)))
        )
    assert pickle.dumps(e.world) == before


def test_installed_loss_floor_is_exact_even_under_low_decimal_context():
    from decimal import localcontext

    with localcontext() as context:
        context.prec = 6
        e, _ = fixture(
            (window(0, 2, ".29999999999999999999999999999999999999", ("B_OH0",)),)
        )
    row = e.capacity_window_balance()[0]
    assert (
        row["installed_at_reveal"] == 20
        and row["removed"] == 5
        and installed(e, "B_OH0") == 15
    )


def test_simultaneous_leg_losses_resolve_once_without_sequential_overcancellation():
    from airline_market.policy.capacity_windows import CapacityWindowMarketEngine as V1
    from airline_market.sim.arrivals import Request
    from airline_market.sim.ledger import book

    c = ordinary.fixture().compiled
    # Native priority is declared in the fixture: connection first, then local.
    specs = tuple(
        replace(p, priority=0 if p.id == "B_OD0_0" else 1) for p in c.spec.products
    )
    c = compile_market(replace(c.spec, products=specs))
    wins = (window(1, 3, ".5", ("B_OH0",)), window(1, 3, 1, ("B_HD0",)))
    new = module.CapacityWindowMarketEngine(c, capacity_windows=wins)
    old = V1(c, capacity_windows=wins)
    tape = ordinary.fixture().tape
    for e in (new, old):
        e.reset(tape=tape)
        state = e.world.carriers[1]
        state.capacity[:] = 2
        for rank, pid in enumerate(("B_OD0_0", "B_OH0")):
            j = c.product_index[pid]
            book(
                e.world,
                1,
                j,
                100,
                Request(rank, int(c.product_market[j]), 0, 0, rank, rank),
                c,
            )
        e._enforce_exits(1)
    assert [c.spec.products[b.product].id for b in new.world.carriers[1].bookings] == [
        "B_OH0"
    ]
    assert old.world.carriers[1].bookings == []
    assert new.world.carriers[1].refunds == 100 and old.world.carriers[1].refunds == 200
    assert new.capacity_window_events()[-1].kind == "remove_batch"
