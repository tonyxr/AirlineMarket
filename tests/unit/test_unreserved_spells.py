"""A.10/A.5/B0/V9: NEW exact declared-clock unreserved spans, no lifecycle utility."""

import copy
import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.core.snapshot import deep_freeze
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
from airline_market.policy.dated_reserve import UtcBookingClock
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.policy.rationed_native import WholePeriodPriority
from airline_market.policy.regulator import CapPolicy, CapWindow
from airline_market.sim.passenger_accounts import (
    OriginalPassengerAccounts,
    original_passenger_accounts,
)
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from tests.unit import test_dated_demand as dated
from tests.unit import test_rationing_native as native

if os.environ.get("UNRESERVED_SPELLS_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_unreserved_spells", os.environ["UNRESERVED_SPELLS_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.sim import unreserved_spells as module


def fraction(record):
    return Fraction(int(record["numerator"]), int(record["denominator"]))


def supplied(*, dated_rows=True, cancelled_last=True, zero=False):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    clock = UtcBookingClock(
        tuple(start + timedelta(microseconds=x) for x in (0, 1, 3, 6, 10, 15))
    )
    expiry = start + timedelta(microseconds=8)

    def ticket(rid, booked, cancelled):
        return {
            "record_id": rid,
            "native_period": booked,
            "carrier": "B" if rid == 0 else "A",
            "product": "B_OH0" if rid == 0 else "A_OH0",
            "active_reservation": cancelled is None,
            "cancellation": None
            if cancelled is None
            else {"native_period": cancelled, "cause": "new declared native cause"},
            "fare": {"binary64_bits": "8000000000000000", "repr": "-0.0"},
        }

    replacement_period = 1 if zero else 2
    attempts = [
        {"original_record": 0, "period": replacement_period, "replacement_record": 1}
    ]
    if dated_rows:
        attempts[0].update(
            attempt_utc=clock.at(replacement_period).isoformat(),
            first_stranded_utc=clock.at(1).isoformat(),
            reentry_deadline_utc=expiry.isoformat(),
        )
    account = {
        "original_record": 0,
        "potential_coordinate": (0, 2, 0),
        "segment": "leisure",
        "request_market": "OH0",
        "tickets": (
            ticket(0, 0, 1),
            ticket(1, replacement_period, 3 if cancelled_last else None),
        ),
        "first_stranded_native_period": 1,
        "native_reentry_deadline_exclusive": 4,
        "reentry_attempts": attempts,
        "current_active_record": None if cancelled_last else 1,
    }
    data = {
        "completed_native_periods": 5,
        "original_booked_lineages": 1,
        "native_tickets_ever": 2,
        "ordinary_outside_choices_in_prefix": 7,
        "lineages": (account,),
    }
    return OriginalPassengerAccounts(deep_freeze(data)), clock


def test_exact_recurrent_spans_keep_one_original_person_and_after_expiry_followup():
    accounts, clock = supplied()
    result = module.unreserved_spells(accounts, clock)
    person = result["persons"][0]
    assert (
        result["original_booked_people"] == 1
        and result["outside_ordinary_choices_in_prefix"] == 7
    )
    assert (
        person["cancelled_ticket_spells"] == 2
        and person["closed_spells"] == 1
        and person["open_at_prefix"]
    )
    assert [fraction(s["declared_clock_elapsed_hours"]) for s in person["spells"]] == [
        Fraction(2, 3_600_000_000),
        Fraction(9, 3_600_000_000),
    ]
    assert fraction(result["observed_unreserved_opportunity_person_hours"]) == Fraction(
        11, 3_600_000_000
    )
    assert person["spells"][1]["extends_after_first_reentry_deadline"]
    assert (
        person["native_utc_attempt_rows_checked"] == 1
        and person["source_account"] == accounts.data["lineages"][0]
    )
    assert (
        person["full_lifecycle_waiting_hours"]
        is person["lifecycle_consumer_surplus"]
        is result["continuous_waiting_hours"]
        is None
    )
    assert (
        not result["closing_boundary_states_reconstructed"]
        and not result["producer_original_capture_or_native_clock_authentication"]
    )


def test_same_opportunity_zero_span_is_retained_without_zero_real_waiting_claim():
    accounts, clock = supplied(zero=True)
    r = module.unreserved_spells(accounts, clock)
    assert len(r["persons"][0]["spells"]) == 2
    assert fraction(r["persons"][0]["spells"][0]["declared_clock_elapsed_hours"]) == 0
    assert (
        r["persons"][0]["spells"][0]["continuous_waiting_or_delivered_outcome"] is None
    )


def test_active_replacement_closes_only_observed_spell_and_undated_clock_is_declaration():
    accounts, clock = supplied(dated_rows=False, cancelled_last=False)
    p = module.unreserved_spells(accounts, clock)["persons"][0]
    assert (
        p["closed_spells"] == 1
        and not p["open_at_prefix"]
        and p["current_active_record"] == 1
    )
    assert (
        p["attempt_rows_without_native_UTC"] == 1
        and p["exact_reentry_deadline_utc"] is None
    )
    assert p["full_lifecycle_waiting_hours"] is None


def test_no_reentry_queue_retains_known_first_cancellation_and_entire_open_observation():
    accounts, clock = supplied()
    data = module.unreserved_spells_record(accounts.data)
    p = data["lineages"][0]
    p["tickets"] = p["tickets"][:1]
    p["reentry_attempts"] = []
    p["native_reentry_deadline_exclusive"] = None
    data["native_tickets_ever"] = 1
    r = module.unreserved_spells(OriginalPassengerAccounts(deep_freeze(data)), clock)
    assert fraction(r["observed_unreserved_opportunity_person_hours"]) == Fraction(
        14, 3_600_000_000
    )
    assert r["persons"][0]["exact_reentry_deadline_utc"] is None


@pytest.mark.parametrize(
    "change",
    [
        "partial UTC",
        "wrong attempt",
        "wrong first",
        "extended expiry",
        "naive UTC",
        "missing success",
        "wrong success period",
        "reversed cancellation",
        "missing ticket",
        "current mismatch",
        "first cancellation",
        "native deadline",
        "duplicate original",
    ],
)
def test_inconsistent_new_input_is_refused_without_partial_result_or_repair(change):
    accounts, clock = supplied()
    data = module.unreserved_spells_record(accounts.data)
    p = data["lineages"][0]
    row = p["reentry_attempts"][0]
    if change == "partial UTC":
        del row["attempt_utc"]
    elif change == "wrong attempt":
        row["attempt_utc"] = clock.at(1).isoformat()
    elif change == "wrong first":
        row["first_stranded_utc"] = clock.at(0).isoformat()
    elif change == "extended expiry":
        row["reentry_deadline_utc"] = clock.at(5).isoformat()
    elif change == "naive UTC":
        row["attempt_utc"] = "2026-01-01T00:00:00"
    elif change == "missing success":
        p["reentry_attempts"] = []
    elif change == "wrong success period":
        row["period"] = 3
    elif change == "reversed cancellation":
        p["tickets"][1]["cancellation"]["native_period"] = 1
    elif change == "missing ticket":
        data["native_tickets_ever"] = 3
    elif change == "current mismatch":
        p["current_active_record"] = 1
    elif change == "first cancellation":
        p["first_stranded_native_period"] = 2
    elif change == "native deadline":
        p["native_reentry_deadline_exclusive"] = 5
    elif change == "duplicate original":
        data["lineages"].append(copy.deepcopy(p))
    with pytest.raises(ValueError):
        module.unreserved_spells(OriginalPassengerAccounts(deep_freeze(data)), clock)


def test_empty_outside_only_context_no_invented_people_or_mean_and_full_clock_required():
    accounts, clock = supplied()
    data = dict(accounts.data)
    data.update(lineages=(), native_tickets_ever=0, original_booked_lineages=0)
    r = module.unreserved_spells(OriginalPassengerAccounts(deep_freeze(data)), clock)
    assert (
        not r["persons"] and r["mean_observed_hours_per_original_booked_person"] is None
    )
    assert r["outside_ordinary_choices_in_prefix"] == 7
    with pytest.raises(ValueError):
        module.unreserved_spells(accounts, UtcBookingClock(clock.boundaries[:4]))
    with pytest.raises(TypeError):
        module.unreserved_spells(accounts, None)


def test_owned_immutable_json_input_and_global_rng_identity():
    accounts, clock = supplied()
    before = pickle.dumps((accounts, clock, np.random.get_state()))
    result = module.unreserved_spells(accounts, clock)
    assert pickle.dumps((accounts, clock, np.random.get_state())) == before
    exported = module.unreserved_spells_record(result)
    json.dumps(exported, allow_nan=False)
    exported["persons"][0]["source_account"]["tickets"][0]["fare"].clear()
    assert (
        result["persons"][0]["source_account"]["tickets"][0]["fare"]["repr"] == "-0.0"
    )
    with pytest.raises(TypeError):
        result["persons"][0]["open_at_prefix"] = False


def new_world(mode):
    c, clock, dep = dated.inputs()
    scenario = replace(
        compile_dated_demand(c, clock, dep),
        physical_shocks=(
            B1DatedShock("new span exit", "rival_exit", "B_OH0", 3, rival_carrier="B"),
            B1DatedShock("new span weather", "weather", "B_HD0", 3, ".6", 1),
        ),
    )
    kwargs = {
        "dated_rebooking": DatedRebookingPolicy(72, 20, 100, interline=True),
        "cap_policy": CapPolicy(
            (CapWindow(0, 5, tuple((p.id, 100) for p in c.spec.products)),)
        ),
    }
    if mode == "efficient":
        kwargs["order_policy"] = EfficientNativeOrder(
            AllocationObjective(
                "native_choice_gain", "NEW unit unreserved spans private order"
            ),
            100000,
        )
        cls = DatedOrderBenchmarkEngine
    else:
        cls = DatedDemandMarketEngine
        kwargs["ordinary_priority"] = (
            WholePeriodPriority("unit/V9_unreserved_spells")
            if mode == "random"
            else None
        )
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(57974, "unit/V9_unreserved_spells")
    )
    marks = tuple(ArrivalMark((i + 1) / 4, 0, 0, 0, 0) for i in range(3))
    g = tape.gumbel.copy()
    g[:2, :3, :] = -1e9
    g[:2, :3, -1] = 0
    g[:2, :3, c.product_index["B_OH0"]] = 1e9
    g.setflags(write=False)
    tape = replace(
        tape, arrival_marks=(marks, marks, *tape.arrival_marks[2:]), gumbel=g
    )
    e = cls(scenario, **kwargs)
    e.reset(tape=tape)
    for carrier in e.world.carriers:
        carrier.capacity[:] = 3
    for _ in range(4):
        e.step(native.actions(e))
    return e, clock


@pytest.mark.parametrize("mode", ["chronological", "random", "efficient"])
def test_new_native_exit_replacements_repeated_loss_and_failed_search_readonly(
    mode, monkeypatch
):
    e, clock = new_world(mode)
    accounts = original_passenger_accounts(e)
    assert e._replacement_roots and any(
        r["replacement_record"] is None for r in e.rebooking_history
    )
    before = pickle.dumps((e.snapshot(), e.trace(), e.metrics(), np.random.get_state()))

    def forbidden(*args, **kwargs):
        raise AssertionError("no producer, timing, tape or physical replay")

    monkeypatch.setattr(e, "step", forbidden)
    monkeypatch.setattr(e, "timed_rebooking_metrics", forbidden)
    monkeypatch.setattr(ExogenousTape, "generate", forbidden)
    result = module.unreserved_spells(accounts, clock)
    assert (
        pickle.dumps((e.snapshot(), e.trace(), e.metrics(), np.random.get_state()))
        == before
    )
    assert result["original_booked_people"] == accounts.data["original_booked_lineages"]
    assert result["native_utc_attempt_rows_checked"] == len(e.rebooking_history)
    assert any(p["cancelled_ticket_spells"] > 1 for p in result["persons"])
    assert any(p["open_at_prefix"] for p in result["persons"])
    assert sum(p["cancelled_ticket_spells"] for p in result["persons"]) > sum(
        p["cancelled_ticket_spells"] > 0 for p in result["persons"]
    )
    assert (
        result["lifecycle_consumer_surplus"]
        is result["denied_requests"]
        is result["misallocation"]
        is None
    )


def test_repeated_failed_search_cannot_extend_first_expiry_even_with_same_native_index():
    accounts, clock = supplied()
    data = module.unreserved_spells_record(accounts.data)
    row = copy.deepcopy(data["lineages"][0]["reentry_attempts"][0])
    row.update(period=3, replacement_record=None, attempt_utc=clock.at(3).isoformat())
    data["lineages"][0]["reentry_attempts"].append(row)
    good = module.unreserved_spells(OriginalPassengerAccounts(deep_freeze(data)), clock)
    assert good["native_utc_attempt_rows_checked"] == 2
    row["reentry_deadline_utc"] = (clock.at(3) + timedelta(microseconds=3)).isoformat()
    with pytest.raises(ValueError, match="cannot extend"):
        module.unreserved_spells(OriginalPassengerAccounts(deep_freeze(data)), clock)


def test_never_cancelled_person_retained_with_zero_observed_spells_not_zero_lifecycle():
    accounts, clock = supplied()
    data = module.unreserved_spells_record(accounts.data)
    other = copy.deepcopy(data["lineages"][0])
    other.update(
        original_record=2,
        potential_coordinate=(0, 9, 0),
        first_stranded_native_period=None,
        native_reentry_deadline_exclusive=None,
        reentry_attempts=[],
        current_active_record=2,
    )
    other["tickets"] = [
        dict(
            other["tickets"][0], record_id=2, active_reservation=True, cancellation=None
        )
    ]
    data["lineages"].append(other)
    data.update(original_booked_lineages=2, native_tickets_ever=3)
    result = module.unreserved_spells(
        OriginalPassengerAccounts(deep_freeze(data)), clock
    )
    assert (
        result["original_booked_people"] == 2
        and len(result["persons"][1]["spells"]) == 0
    )
    assert fraction(
        result["mean_observed_hours_per_original_booked_person"]
    ) == Fraction(11, 7_200_000_000)
    assert result["persons"][1]["full_lifecycle_waiting_hours"] is None
