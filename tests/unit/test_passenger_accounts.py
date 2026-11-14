"""A.10/A.5/B0/V9: NEW original-passenger account fixtures, no welfare adoption."""

import importlib.util
import json
import os
import pickle
import struct
import sys
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PeriodPhase, PriceAction
from airline_market.policy.allocation_comparison import AllocationObjective
from airline_market.policy.dated_demand import (
    DatedDemandMarketEngine,
    compile_dated_demand,
)
from airline_market.policy.dated_order_admission import (
    DatedOrderBenchmarkEngine,
    EfficientNativeOrder,
)
from airline_market.policy.dated_rescue import DatedRebookingPolicy
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.policy.potential_lineage import PotentialLineageMarketEngine
from airline_market.policy.rationed_native import WholePeriodPriority
from airline_market.policy.regulator import CapPolicy, CapWindow
from airline_market.sim.disruption import RivalExit
from airline_market.sim.engine import MarketEngine
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from tests.unit import test_dated_demand as dated
from tests.unit import test_rationing_native as native

if os.environ.get("PASSENGER_ACCOUNTS_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_passenger_accounts", os.environ["PASSENGER_ACCOUNTS_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.sim import passenger_accounts as module


def fixture(*, priority=None, efficient=False, repeated=False, penalty=None):
    c, clock, dep = dated.inputs()
    if penalty is not None:
        c = compile_market(
            replace(
                c.spec,
                resolution=replace(c.spec.resolution, penalty_per_cancellation=penalty),
            )
        )
    s = compile_dated_demand(c, clock, dep)
    kwargs = {}
    if repeated:
        s = replace(
            s,
            physical_shocks=(
                B1DatedShock("exit", "rival_exit", "B_OH0", 3, rival_carrier="B"),
                B1DatedShock("weather", "weather", "B_HD0", 3, ".6", 1),
            ),
        )
        kwargs["dated_rebooking"] = DatedRebookingPolicy(72, 20, 100, interline=True)
    if efficient:
        kwargs.update(
            order_policy=EfficientNativeOrder(
                AllocationObjective(
                    "native_choice_gain", "unit explicit gain, not lifecycle welfare"
                ),
                100000,
            ),
            cap_policy=CapPolicy(
                (CapWindow(0, 5, tuple((p.id, 100) for p in c.spec.products)),)
            ),
        )
        e = DatedOrderBenchmarkEngine(s, **kwargs)
    else:
        e = DatedDemandMarketEngine(s, ordinary_priority=priority, **kwargs)
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(48974, "unit/V9_passenger_accounts")
    )
    if repeated:
        marks = tuple(ArrivalMark((k + 1) / 4, 0, 0, 0, 0) for k in range(3))
        g = tape.gumbel.copy()
        g[:2, :3, :] = -1e9
        g[:2, :3, -1] = 0
        g[:2, :3, c.product_index["B_OH0"]] = 1e9
        g.setflags(write=False)
        tape = replace(
            tape, arrival_marks=(marks, marks, *tape.arrival_marks[2:]), gumbel=g
        )
    e.reset(tape=tape)
    if repeated:
        e.world.carriers[0].capacity[:] = 3
        e.world.carriers[1].capacity[:] = 3
    return e


def rational(row):
    return (
        None
        if row is None
        else Fraction(int(row["numerator"]), int(row["denominator"]))
    )


def report(e):
    return module.passenger_accounts_record(module.original_passenger_accounts(e))


def test_empty_completed_prefix_has_zero_booked_population_without_zero_demand_inference():
    saved = report(fixture())
    assert (
        saved["original_booked_lineages"] == saved["native_tickets_ever"] == 0
        and saved["lineages"] == []
    )
    assert (
        "booked" in saved["population"]
        and saved["stranded_lifecycle_consumer_surplus"] is None
    )


@pytest.mark.parametrize("mode", ["chronological", "priority", "efficient"])
def test_new_native_path_accounts_all_originals_and_is_read_only_without_actor_tape_or_solver(
    mode, monkeypatch
):
    e = fixture(
        priority=WholePeriodPriority("unit/V9_passenger_accounts")
        if mode == "priority"
        else None,
        efficient=mode == "efficient",
    )
    e.step(native.actions(e))
    e.step(native.actions(e))
    before = pickle.dumps(e.snapshot()), pickle.dumps(np.random.get_state())

    def forbidden(*args, **kwargs):
        raise AssertionError("accounting cannot execute actors/tape/solver/native path")

    for name in ["step", "post_actions", "_reenter", "_marks"]:
        monkeypatch.setattr(e, name, forbidden)
    monkeypatch.setattr(ExogenousTape, "generate", forbidden)
    monkeypatch.setattr(StreamRegistry, "generator", forbidden)
    saved = report(e)
    assert (
        saved["native_tickets_ever"]
        == saved["original_booked_lineages"]
        == len(e._physical_coordinates)
    )
    assert saved["replacement_tickets_ever"] == 0
    assert before == (pickle.dumps(e.snapshot()), pickle.dumps(np.random.get_state()))
    assert all(r["native_original_rank"] >= 0 for r in saved["lineages"])
    assert all(
        rational(r["net_ticket_fare_paid"])
        == sum(
            rational(t["fare"]["exact"])
            for t in r["tickets"]
            if t["active_reservation"]
        )
        for r in saved["lineages"]
    )


def test_repeated_actual_cancellations_keep_one_root_all_ticket_refunds_searches_and_current_state():
    e = fixture(repeated=True)
    for _ in range(3):
        e.step(native.actions(e))
    saved = report(e)
    roots = [r for r in saved["lineages"] if r["reentry_queue_present"]]
    assert len(roots) == 3 and saved["replacement_tickets_ever"] == 3
    assert any(
        sum(t["cancellation"] is not None for t in r["tickets"]) == 2 for r in roots
    )
    assert any(r["current_reservation_state"] == "no_active_reservation" for r in roots)
    assert (
        sum(len(r["reentry_attempts"]) for r in saved["lineages"])
        == saved["reentry_search_opportunities"]
    )
    assert not saved["searches_are_additional_people"]
    for r in roots:
        gross = sum(rational(t["fare"]["exact"]) for t in r["tickets"])
        refunds = sum(
            rational(t["fare"]["exact"])
            for t in r["tickets"]
            if t["cancellation"] is not None
        )
        assert (
            rational(r["gross_ticket_fares"]) == gross
            and rational(r["fare_refunds"]) == refunds
        )
        assert rational(r["net_ticket_fare_paid"]) == gross - refunds
        assert (
            r["carrier_penalties_allocated_to_passenger"] is None
            and r["delivered_service_or_lifecycle_utility"] is None
        )
    assert all(rational(c["cash_arithmetic_residual"]) == 0 for c in saved["carriers"])
    before = {
        r["original_record"]: r["native_reentry_deadline_exclusive"] for r in roots
    }
    e.step(native.actions(e))
    later = report(e)
    assert (
        later["original_booked_lineages"] == 3
        and later["reentry_search_opportunities"] > 3
    )
    assert any(
        a["replacement_record"] is None
        for r in later["lineages"]
        for a in r["reentry_attempts"]
    )
    assert before == {
        r["original_record"]: r["native_reentry_deadline_exclusive"]
        for r in later["lineages"]
    }


def test_binary64_accumulation_residual_is_retained_exactly_without_rounding_repair():
    e = fixture()
    actions = {
        x.code: PriceAction({j: 0.1 for j in e.compiled.products_of[i]})
        for i, x in enumerate(e.compiled.spec.carriers)
    }
    for _ in range(3):
        e.step(actions)
    saved = report(e)
    for row, state in zip(saved["carriers"], e.world.carriers):
        expected = Fraction.from_float(state.cash) - rational(row["ticket_fares_exact"])
        assert rational(row["cash_arithmetic_residual"]) == expected
    assert any(
        rational(r["revenue_arithmetic_residual"]) != 0 for r in saved["carriers"]
    )
    assert not saved["native_arithmetic_repaired_or_tolerance_applied"]


def test_signed_zero_nonfinite_native_accumulators_and_negative_values_remain_explicit():
    e = fixture()
    e.world.carriers[0].cash = -0.0
    e.world.carriers[0].refunds = -2.5
    e.world.carriers[1].cash = struct.unpack(">d", bytes.fromhex("7ff8000000000042"))[0]
    saved = report(e)
    json.dumps(saved, allow_nan=False)
    first, second = saved["carriers"]
    assert first["native_accumulators"]["cash"]["binary64_bits"] == "8000000000000000"
    assert rational(first["refund_arithmetic_residual"]) == Fraction(-5, 2)
    assert second["native_accumulators"]["cash"]["binary64_bits"] == "7ff8000000000042"
    assert second["cash_arithmetic_residual"] is None


@pytest.mark.parametrize(
    "damage",
    [
        "missing_coordinate",
        "bad_rank",
        "missing_cancel",
        "missing_history",
        "wrong_root",
        "duplicate_active",
        "missing_trace",
        "pending",
    ],
)
def test_incomplete_or_inconsistent_lineage_is_refused_without_reconstruction_or_partial_account(
    damage,
):
    e = fixture(repeated=True)
    for _ in range(3):
        e.step(native.actions(e))
    if damage == "missing_coordinate":
        e._physical_coordinates.pop(next(iter(e._physical_coordinates)))
    elif damage == "bad_rank":
        key = next(iter(e._physical_coordinates))
        coord = e._physical_coordinates[key]
        e._physical_coordinates[key] = replace(coord, native_rank=999)
    elif damage == "missing_cancel":
        e.stranded = e.stranded[1:]
    elif damage == "missing_history":
        e.rebooking_history = []
    elif damage == "wrong_root":
        e._replacement_roots[next(iter(e._replacement_roots))] = 999
    elif damage == "duplicate_active":
        cs = next(cs for cs in e.world.carriers if cs.bookings)
        cs.bookings.append(cs.bookings[0])
    elif damage == "missing_trace":
        e._trace.pop()
    else:
        e._pending_order = object()
    with pytest.raises(ValueError):
        report(e)


def test_old_engine_and_partial_posting_are_not_reconstructed():
    e = fixture()
    old = MarketEngine(e.compiled)
    old.reset(tape=e.tape)
    with pytest.raises(TypeError, match="no old-ticket reconstruction"):
        report(old)
    e.world.phase = PeriodPhase.LEADERS_POSTED
    with pytest.raises(ValueError, match="completed native prefix"):
        report(e)


def test_owned_immutable_snapshot_export_no_lifecycle_or_primary_adoption():
    e = fixture(repeated=True)
    for _ in range(3):
        e.step(native.actions(e))
    row = module.original_passenger_accounts(e)
    saved = module.passenger_accounts_record(row)
    json.dumps(saved, allow_nan=False)
    saved["lineages"].clear()
    assert row.data["lineages"]
    with pytest.raises(TypeError):
        row.data["lineages"][0]["original_record"] = 999
    assert (
        row.data["denied_requests"]
        is row.data["misallocation"]
        is row.data["total_lifecycle_surplus"]
        is None
    )
    assert (
        not row.data["primary_B4_welfare_adopted"]
        and row.data["scientific_budget"] is None
    )


def test_outside_requests_are_retained_as_prefix_counts_not_invented_ticketed_people():
    e = fixture()
    g = e.tape.gumbel.copy()
    g[:, :, :] = -1e9
    g[:, :, -1] = 1e9
    g.setflags(write=False)
    e.tape = replace(e.tape, gumbel=g)
    e.step(native.actions(e))
    saved = report(e)
    assert (
        saved["original_booked_lineages"] == 0
        and saved["ordinary_outside_choices_in_prefix"] > 0
    )
    assert saved["ordinary_outside_choices_in_prefix"] == len(e.world.request_log)


def test_actual_native_cancellation_penalties_stay_carrier_totals_without_passenger_allocation():
    e = fixture(repeated=True, penalty=7)
    for _ in range(3):
        e.step(native.actions(e))
    saved = report(e)
    assert (
        sum(
            rational(r["native_accumulators"]["penalties"]["exact"])
            for r in saved["carriers"]
        )
        > 0
    )
    assert all(
        rational(r["cash_arithmetic_residual"]) == 0
        and r["cancelled_counter_residual"] == 0
        for r in saved["carriers"]
    )
    assert all(
        r["carrier_penalties_allocated_to_passenger"] is None for r in saved["lineages"]
    )


def test_first_actual_cancellation_is_known_even_without_reentry_policy_or_queue():
    base = fixture()
    e = PotentialLineageMarketEngine(base.compiled, exits=(RivalExit("B", 1),))
    g = base.tape.gumbel.copy()
    g[0, :, :] = -1e9
    g[0, :, -1] = 0
    for slot, mark in enumerate(base.tape.arrival_marks[0]):
        choices = [
            j
            for j in e.compiled.market_products[mark.market]
            if e.compiled.spec.products[j].carrier == "B"
        ]
        if choices:
            g[0, slot, choices[0]] = 1e9
    g.setflags(write=False)
    e.reset(tape=replace(base.tape, gumbel=g))
    e.step(native.actions(e))
    saved = report(e)
    assert saved["lineages"]
    assert all(
        r["first_stranded_native_period"] == 1 and not r["reentry_queue_present"]
        for r in saved["lineages"]
    )
    assert all(
        r["current_reservation_state"] == "no_active_reservation"
        for r in saved["lineages"]
    )
