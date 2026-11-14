"""A.10/A.5/B0/V9: NEW prospective reservation point histories, no waiting inference."""

import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.policy.allocation_comparison import AllocationObjective
from airline_market.policy.dated_demand import compile_dated_demand
from airline_market.policy.dated_order_admission import EfficientNativeOrder
from airline_market.policy.dated_rescue import DatedRebookingPolicy
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.policy.rationed_native import WholePeriodPriority
from airline_market.policy.regulator import CapPolicy, CapWindow
from airline_market.sim.ordinary_cohort import (
    OrdinaryCohortMarketEngine,
    OrdinaryCohortOrderEngine,
)
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.model import CapacityEventSpec, CapacityProcessSpec
from tests.unit import test_dated_demand as dated
from tests.unit import test_rationing_native as native

if os.environ.get("RESERVATION_CENSUS_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_reservation_census", os.environ["RESERVATION_CENSUS_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.sim import reservation_census as module


def fixture(mode="chronological", *, repeated=False, outside=False, empty=False):
    c, clock, dep = dated.inputs()
    scenario = compile_dated_demand(c, clock, dep)
    kwargs = {
        "cap_policy": CapPolicy(
            (CapWindow(0, 5, tuple((p.id, 100) for p in c.spec.products)),)
        )
    }
    if repeated:
        scenario = replace(
            scenario,
            physical_shocks=(
                B1DatedShock("new exit", "rival_exit", "B_OH0", 3, rival_carrier="B"),
                B1DatedShock("new weather", "weather", "B_HD0", 3, ".6", 1),
            ),
        )
        kwargs["dated_rebooking"] = DatedRebookingPolicy(72, 20, 100, interline=True)
    if mode == "efficient":
        kwargs["order_policy"] = EfficientNativeOrder(
            AllocationObjective("native_choice_gain", "unit census private benchmark"),
            100000,
        )
        cls, parent = module.ReservationCensusOrderEngine, OrdinaryCohortOrderEngine
    else:
        kwargs["ordinary_priority"] = (
            WholePeriodPriority("unit/V9_reservation_census")
            if mode == "random"
            else None
        )
        cls, parent = module.ReservationCensusMarketEngine, OrdinaryCohortMarketEngine
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(54974, "unit/V9_reservation_census")
    )
    if repeated:
        marks = tuple(ArrivalMark((i + 1) / 4, 0, 0, 0, 0) for i in range(3))
        g = tape.gumbel.copy()
        g[:2, :3, :] = -1e9
        g[:2, :3, -1] = 0
        g[:2, :3, c.product_index["B_OH0"]] = 1e9
        g.setflags(write=False)
        tape = replace(
            tape, arrival_marks=(marks, marks, *tape.arrival_marks[2:]), gumbel=g
        )
    if outside:
        g = tape.gumbel.copy()
        g[:, :, :] = -1e9
        g[:, :, -1] = 1e9
        g.setflags(write=False)
        tape = replace(tape, gumbel=g)
    if empty:
        tape = replace(tape, arrival_marks=tuple(() for _ in tape.arrival_marks))
    e, p = cls(scenario, **kwargs), parent(scenario, **kwargs)
    for engine in (e, p):
        engine.reset(tape=tape)
        if repeated:
            for carrier in engine.world.carriers:
                carrier.capacity[:] = 3
    return e, p


def summary(e):
    return module.reservation_census_summary(e.reservation_boundaries())


@pytest.mark.parametrize("mode", ["chronological", "random", "efficient"])
def test_full_parent_identity_and_each_actual_closing_ledger_join(mode, monkeypatch):
    e, p = fixture(mode)
    before = pickle.dumps(np.random.get_state())

    def forbidden(*args, **kwargs):
        raise AssertionError("no generation during step or query")

    monkeypatch.setattr(ExogenousTape, "generate", forbidden)
    while not e.terminal():
        e.step(native.actions(e))
        p.step(native.actions(p))
        assert pickle.dumps(e.world) == pickle.dumps(p.world)
        assert pickle.dumps(e.trace()) == pickle.dumps(p.trace())
        assert (
            e.ordinary_periods() == p.ordinary_periods()
            and e.coordinate_records() == p.coordinate_records()
        )
        boundary = e.reservation_boundaries()[-1]
        active = {b.record_id for cs in e.world.carriers for b in cs.bookings}
        assert {
            r.active_record for r in boundary.rows if r.active_record is not None
        } == active
        assert {r.key for r in boundary.rows} == {
            r.key for period in e.ordinary_periods() for r in period.rows
        }
        state = pickle.dumps(e.snapshot())
        result = summary(e)
        assert pickle.dumps(e.snapshot()) == state
        assert result["unique_original_people"] == len(e.world.request_log)
    assert e.metrics() == p.metrics() and pickle.dumps(np.random.get_state()) == before


@pytest.mark.parametrize("mode", ["chronological", "random", "efficient"])
def test_real_exit_interline_repeated_cancellation_keeps_original_point_histories(mode):
    e, _ = fixture(mode, repeated=True)
    for _ in range(4):
        e.step(native.actions(e))
    before_summary = pickle.dumps(e.snapshot())
    result = summary(e)
    assert e._replacement_roots and any(
        r["replacement_record"] is not None for r in e.rebooking_history
    )
    histories = result["persons"]
    assert any(
        any(r.reservation_state == "active_replacement" for t, r in p["observations"])
        for p in histories
    )
    assert any(
        any(
            r.reservation_state == "no_active_reservation" for t, r in p["observations"]
        )
        for p in histories
    )
    assert result["unique_original_people"] == len(e.world.request_log)
    assert (
        sum(len(p["observations"]) for p in histories)
        > result["unique_original_people"]
    )
    assert all(
        p["first_actual_replacement_booking_time"]
        is p["elapsed_waiting_hours"]
        is p["delivered_travel"]
        is p["lifecycle_consumer_surplus"]
        is None
        for p in histories
    )
    # Native first-stranding/deadlines and every actual ticket/search remain in parent.
    assert pickle.dumps(e.snapshot()) == before_summary


def test_outside_and_empty_periods_keep_full_support_without_invented_tickets_or_people():
    e, _ = fixture(outside=True)
    for _ in range(3):
        e.step(native.actions(e))
    r = summary(e)
    assert r["unique_original_people"] > 0
    assert all(
        row.reservation_state == "ordinary_outside"
        and row.active_record is row.original_record is None
        for b in e.reservation_boundaries()
        for row in b.rows
    )
    assert not r["boundary_observations_are_additional_people"]
    empty, _ = fixture(empty=True)
    for _ in range(3):
        empty.step(native.actions(empty))
    r = summary(empty)
    assert r["unique_original_people"] == 0 and len(r["by_boundary"]) == 3
    assert all(b["arrived_original_people"] == 0 for b in r["by_boundary"])


def test_branch_reset_immutable_snapshots_and_all_parent_contracts():
    e, _ = fixture("efficient", repeated=True)
    e.step(native.actions(e))
    snap = e.snapshot()
    branch = e.branch()
    assert pickle.dumps(branch.snapshot()) == pickle.dumps(snap)
    branch.step(native.actions(branch))
    assert (
        len(branch.reservation_boundaries()) == 2
        and len(e.reservation_boundaries()) == 1
    )
    with pytest.raises(ValueError, match="capture rule"):
        e.restore(replace(snap, capture_rule="other"))
    with pytest.raises(TypeError, match="combined"):
        e.restore(snap.native)
    e.restore(snap)
    assert len(e.reservation_boundaries()) == 1
    tape = e.tape
    e.reset(tape=tape)
    assert (
        e.reservation_boundaries() == () and summary(e)["unique_original_people"] == 0
    )
    assert snap.boundaries[0].rows


def test_failure_after_successful_parent_close_retains_mutations_without_positive_census(
    monkeypatch,
):
    e, _ = fixture()

    def fail(*args):
        raise RuntimeError("new capture failure")

    monkeypatch.setattr(e, "_capture_reservation_boundary", fail)
    with pytest.raises(RuntimeError, match="capture failure"):
        e.step(native.actions(e))
    assert e.world.t == 1 and len(e.trace()) == 1 and e._pending_reservation == 0
    with pytest.raises(ValueError, match="no partial positive"):
        e.reservation_boundaries()
    with pytest.raises(ValueError, match="unfinished"):
        e.step(native.actions(e))
    branch = e.branch()
    assert branch._pending_reservation == 0
    with pytest.raises(ValueError, match="no partial positive"):
        branch.reservation_boundaries()


def test_one_native_post_and_callback_no_account_or_timing_report_replay(monkeypatch):
    e, _ = fixture("efficient")
    calls = []
    original = e.post_actions

    def post(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(e, "post_actions", post)
    outcomes = []
    for controller in e.controllers.values():
        old = controller.observe_outcome

        def cb(x, old=old):
            outcomes.append(1)
            return old(x)

        monkeypatch.setattr(controller, "observe_outcome", cb)
    e.step(native.actions(e))
    summary(e)
    assert len(calls) == 1 and len(outcomes) == len(e.controllers)


def receipts():
    row = module.ReservationObservation(
        (0, 2, 0), "leisure", 0, "OH0", 0, 7, 7, "B_OH0", "B", "active_original"
    )
    outside = module.ReservationObservation(
        (0, 3, 0), "business", 0, "OH0", 1, None, None, None, None, "ordinary_outside"
    )
    return (
        module.ReservationBoundary(0, 1, (row, outside)),
        module.ReservationBoundary(
            1,
            2,
            (
                replace(
                    row,
                    active_record=None,
                    active_product=None,
                    active_carrier=None,
                    reservation_state="no_active_reservation",
                ),
                outside,
            ),
        ),
    )


def test_pure_point_summary_counts_each_original_once_not_elapsed_intervals_or_first_booking():
    data = receipts()
    r = module.reservation_census_summary(data)
    assert r["unique_original_people"] == 2 and r["completed_native_periods"] == 2
    person = r["persons"][0]
    assert (
        person["active_reservation_observations"]
        == person["unreserved_booked_person_observations"]
        == 1
    )
    assert person["elapsed_waiting_hours"] is None
    record = module.reservation_census_record(r)
    json.dumps(record, allow_nan=False)
    record["persons"].clear()
    assert r["persons"]
    with pytest.raises(TypeError):
        r["persons"][0]["elapsed_waiting_hours"] = 0
    assert (
        not r["same_producer_bank_or_original_capture_authenticated"]
        and not r["primary_default_empirical_production_adopted"]
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing boundary",
        "missing old person",
        "late person",
        "duplicate key",
        "identity",
        "ticket fields",
        "kind",
        "negative id",
    ],
)
def test_incomplete_or_inconsistent_point_receipts_are_refused(change):
    data = list(receipts())
    first, second = data
    if change == "missing boundary":
        data = [second]
    elif change == "missing old person":
        data[1] = replace(second, rows=second.rows[:1])
    elif change == "late person":
        data[0] = replace(first, rows=first.rows[:1])
    elif change == "duplicate key":
        data[0] = replace(first, rows=(first.rows[0], first.rows[0]))
    elif change == "identity":
        data[1] = replace(
            second, rows=(replace(second.rows[0], segment="changed"), second.rows[1])
        )
    elif change == "ticket fields":
        data[0] = replace(
            first, rows=(replace(first.rows[0], active_product=None), first.rows[1])
        )
    elif change == "kind":
        data[0] = replace(
            first,
            rows=(
                replace(first.rows[0], reservation_state="active_replacement"),
                first.rows[1],
            ),
        )
    elif change == "negative id":
        data[0] = replace(
            first, rows=(replace(first.rows[0], active_record=-1), first.rows[1])
        )
    with pytest.raises(ValueError):
        module.reservation_census_summary(data)


def test_forced_native_hooks_refused_before_any_capture_or_mutation():
    e, _ = fixture()
    before = pickle.dumps(e.snapshot())
    with pytest.raises(ValueError, match="genuine parent"):
        e.step(native.actions(e), _requests=())
    assert pickle.dumps(e.snapshot()) == before


def test_genuine_purchase_and_native_end_cancellation_within_one_step_is_not_zero_waiting():
    c, clock, dep = dated.inputs()
    event = CapacityEventSpec(
        trigger={"hazard": {"probability": 1}},
        effects=tuple((l.id, -20) for l in c.spec.legs),
        once=True,
    )
    c = compile_market(
        replace(c.spec, capacity_process=CapacityProcessSpec(events=(event,)))
    )
    e = module.ReservationCensusMarketEngine(compile_dated_demand(c, clock, dep))
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(54974, "unit/V9_reservation_transient")
    )
    g = tape.gumbel.copy()
    g[0, 0, :] = -1e9
    g[0, 0, -1] = 0
    g[0, 0, c.product_index["A_OH0"]] = 1e9
    g.setflags(write=False)
    tape = replace(
        tape,
        arrival_marks=((ArrivalMark(0.25, 0, 0, 0, 0),), *(() for _ in range(4))),
        gumbel=g,
    )
    e.reset(tape=tape)
    e.step(native.actions(e))
    assert e.world.next_record_id == 1 and len(e.stranded) == 1
    r = summary(e)
    person = r["persons"][0]
    assert person["active_reservation_observations"] == 0
    assert person["unreserved_booked_person_observations"] == 1
    assert (
        person["elapsed_waiting_hours"] is person["lifecycle_consumer_surplus"] is None
    )
    assert person["identity"][-1] == e.stranded[0].booking.record_id
    first = e.reservation_boundaries()[0]
    saved = pickle.dumps(first)
    e.step(native.actions(e))
    assert pickle.dumps(first) == saved


def test_receipts_own_mutable_input_sequences_and_refuse_mutable_identity_values():
    key = [0, 1, 0]
    row = module.ReservationObservation(
        key, "business", 0, "OH0", 0, None, None, None, None, "ordinary_outside"
    )
    rows = [row]
    boundary = module.ReservationBoundary(0, 1, rows)
    key[0] = 99
    rows.clear()
    assert boundary.rows == (row,) and row.key == (0, 1, 0)
    with pytest.raises(ValueError, match="immutable native"):
        replace(row, segment=["business"])
