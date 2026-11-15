"""A.10/A.5/B0/V9: NEW genuine native presentation-call source witnesses."""

import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.dated_demand import compile_dated_demand
from airline_market.policy.dated_rescue import (
    DatedRebookingPolicy,
    DatedRescueMenu,
    UtcWindow,
)
from airline_market.policy.dated_reserve import UtcBookingClock
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.policy.potential_lineage import PotentialLineageMarketEngine
from airline_market.sim.own_history_information import OwnHistoryInformationMarketEngine
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

if os.environ.get("PROOF_INFORMATION_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_proof_information", os.environ["PROOF_INFORMATION_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.sim import proof_information as module


def fixture(*, kind="exit", count=3, interline=True, outside=False, counter_menu=False):
    base = screen_spec(
        {
            "T": 5,
            "actions": 2,
            "n_spokes": 1,
            "capacity": 3 if kind == "recurrent" else 20,
            "capacity_support": [3, 4],
        }
    )
    demand = replace(
        base.demand,
        mode="poisson",
        arrival_probability=None,
        market_weights=None,
        intensity={m.id: PeriodSchedule.parse({"constant": 2.0}) for m in base.markets},
    )
    c = compile_market(replace(base, demand=demand))
    start = datetime(2026, 11, 22, tzinfo=timezone.utc)
    clock = UtcBookingClock(
        tuple(start + timedelta(hours=h) for h in (0, 12, 24, 36, 48, 60))
    )
    departures = tuple(
        (
            l.id,
            start
            + timedelta(
                hours=84 if l.id == "B_OH0" else 120 if l.id == "B_HD0" else 108
            ),
        )
        for l in c.spec.legs
    )
    scenario = compile_dated_demand(c, clock, departures)
    if kind == "exit":
        shocks = (B1DatedShock("exit", "rival_exit", "B_OH0", 3, rival_carrier="B"),)
    elif kind == "recurrent":
        shocks = (
            B1DatedShock("weather", "weather", "B_OH0", 3, ".6", 1),
            B1DatedShock("later_exit", "rival_exit", "B_HD0", 3, rival_carrier="B"),
        )
    else:
        shocks = ()
    scenario = replace(scenario, physical_shocks=shocks)
    kwargs = {"dated_rebooking": DatedRebookingPolicy(72, 20, 100, interline=interline)}
    if counter_menu:
        kwargs["dated_menu"] = DatedRescueMenu(
            UtcWindow(clock.at(0), clock.at(5)),
            "counter_three_tier",
            ((500, 99), (1000, 199), (None, 299)),
            tuple((m.id, 400) for m in c.spec.markets),
        )
    e = module.ProofInformationMarketEngine(scenario, **kwargs)
    parent = OwnHistoryInformationMarketEngine(scenario, **kwargs)
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(77974, "unit/V9_proof_information")
    )
    marks = tuple(ArrivalMark((i + 1) / (count + 1), 0, 0, 0, 0) for i in range(count))
    g = tape.gumbel.copy()
    g[:] = -1e9
    g[:, :, -1] = 1e9 if outside else 0
    if not outside:
        g[:, :, c.product_index["B_OH0"]] = 1e9
    g.setflags(write=False)
    tape = replace(
        tape, arrival_marks=tuple(marks for _ in tape.arrival_marks), gumbel=g
    )
    for x in (e, parent):
        x.reset(tape=tape)
    return e, parent


def actions(e):
    return {
        carrier.code: PriceAction(
            {j: 200 if i == 0 else 300 for j in e.compiled.products_of[i]}
        )
        for i, carrier in enumerate(e.compiled.spec.carriers)
    }


def advance(e, parent, n=1):
    for _ in range(n):
        e.step(actions(e))
        parent.step(actions(parent))
        assert pickle.dumps(e.world) == pickle.dumps(parent.world)
        assert pickle.dumps(e.trace()) == pickle.dumps(parent.trace())
        assert e.rebooking_history == parent.rebooking_history
        assert e.own_information_periods() == parent.own_information_periods()
        assert e.presented_itineraries == parent.presented_itineraries
        assert e.metrics() == parent.metrics()


def original(e, root=0):
    return e._passengers[root].original


def present_pair(e, parent, *, root=0, counter=False):
    proof = e.present_cancelled_itinerary(original(e, root), airport_counter=counter)
    other = parent.present_cancelled_itinerary(
        original(parent, root), airport_counter=counter
    )
    assert proof == other
    return proof


def test_duplicates_preserved_and_counter_expires_without_new_people():
    e, parent = fixture(interline=False)
    advance(e, parent, 2)
    for counter in (False, True, True):
        present_pair(e, parent, counter=counter)
    calls = e.presentation_calls()
    assert [r.fact.person_call for r in calls] == [0, 1, 2]
    assert all(r.fact.person_ticket == 0 for r in calls)
    assert len(e.presented_itineraries) == 2
    assert len(e._passengers) == 3
    advance(e, parent)
    row = e.proof_information_periods()[-1].reentry_rows[0]
    assert row.view.native_counter_this_period and row.view.native_proof_on_file
    assert row.view.earlier_presentation_calls == tuple(r.fact for r in calls)
    advance(e, parent)
    later = e.proof_information_periods()[-1].reentry_rows[0]
    assert later.view.native_proof_on_file and not later.view.native_counter_this_period
    assert row.view.native_counter_this_period  # owned earlier point unchanged
    assert row.view.earlier_presentation_calls == later.view.earlier_presentation_calls


def test_context_precedes_single_native_marks_call_and_no_private_read(monkeypatch):
    e, parent = fixture(interline=False)
    native = PotentialLineageMarketEngine._marks
    counts = {id(e): 0, id(parent): 0}

    def checked(self, passenger, period):
        counts[id(self)] += 1
        if self is e:
            row = self._pending_proof[-1]
            assert row.analyst_root == passenger.original.record_id
            assert row.view.native_period == period
        return native(self, passenger, period)

    monkeypatch.setattr(PotentialLineageMarketEngine, "_marks", checked)
    before = pickle.dumps(np.random.get_state())
    advance(e, parent, 2)
    present_pair(e, parent, counter=True)
    advance(e, parent, 3)
    assert counts[id(e)] == counts[id(parent)] == len(e.rebooking_history)
    assert pickle.dumps(np.random.get_state()) == before


@pytest.mark.parametrize("counter", [0, None, "yes"])
def test_invalid_native_counter_no_capture_or_mutation(counter):
    e, parent = fixture(interline=False)
    advance(e, parent, 2)
    saved = pickle.dumps(e.snapshot())
    with pytest.raises(TypeError):
        e.present_cancelled_itinerary(original(e), airport_counter=counter)
    assert pickle.dumps(e.snapshot()) == saved


def test_wrong_or_forged_or_active_ticket_no_capture():
    e, parent = fixture(interline=False)
    advance(e, parent)
    control, cp = fixture(kind="control", interline=False)
    advance(control, cp)
    active = next(b for cs in control.world.carriers for b in cs.bookings)
    with pytest.raises(ValueError):
        control.present_cancelled_itinerary(active)
    advance(e, parent)
    saved = pickle.dumps(e.snapshot())
    with pytest.raises(TypeError):
        e.present_cancelled_itinerary("cancelled")
    with pytest.raises(ValueError):
        e.present_cancelled_itinerary(replace(original(e), fare=12345.0))
    assert pickle.dumps(e.snapshot()) == saved


def test_other_person_calls_excluded_and_local_keys():
    e, parent = fixture(interline=False)
    advance(e, parent, 2)
    present_pair(e, parent, root=0, counter=True)
    present_pair(e, parent, root=1, counter=True)
    advance(e, parent)
    rows = e.proof_information_periods()[-1].reentry_rows
    assert module.proof_information_key(rows[0]) == module.proof_information_key(
        rows[1]
    )
    assert len(rows[2].view.earlier_presentation_calls) == 0
    assert not rows[2].view.native_proof_on_file
    view = module.proof_information_view(rows[0])
    encoded = json.dumps(view, allow_nan=False)
    for forbidden in (
        "analyst",
        "class",
        "gumbel",
        "stock",
        "fare",
        "outcome",
        "root",
        "original_key",
    ):
        assert forbidden not in encoded
    view["earlier_presentation_calls"][0]["airport_counter"] = False
    assert (
        len(module.proof_information_view(rows[0])["earlier_presentation_calls"]) == 1
    )


def test_owned_branch_reset_restore_contract():
    e, parent = fixture(interline=False)
    advance(e, parent, 2)
    present_pair(e, parent, counter=True)
    saved = e.snapshot()
    branch = e.branch()
    assert pickle.dumps(branch.snapshot()) == pickle.dumps(saved)
    branch.present_cancelled_itinerary(original(branch))
    assert len(branch.presentation_calls()) == 2 and len(e.presentation_calls()) == 1
    branch.step(actions(branch))
    assert len(branch.proof_information_periods()) == 3
    assert len(e.proof_information_periods()) == 2
    with pytest.raises(ValueError):
        e.restore(replace(saved, rule="other"))
    with pytest.raises(ValueError):
        e.restore(replace(saved, spec_hash="other"))
    with pytest.raises(TypeError):
        e.restore(saved.native)
    e.reset(tape=e.tape)
    assert e.presentation_calls() == () and e.proof_information_periods() == ()
    e.restore(saved)
    assert pickle.dumps(e.snapshot()) == pickle.dumps(saved)


@pytest.mark.parametrize("count,outside", [(0, False), (2, True)])
def test_empty_outside_all_parent_support_retained(count, outside):
    e, parent = fixture(count=count, outside=outside)
    advance(e, parent, 5)
    assert len(e.proof_information_periods()) == 5
    assert all(p.reentry_rows == () for p in e.proof_information_periods())
    assert e.presentation_calls() == ()
    assert sum(len(p.rows) for p in e.information_periods()) == count * 5
    assert module.proof_information_record(e)["scientific_budget"] is None


def test_recurrent_cancellation_keeps_original_ticket_proof_first_expiry():
    e, parent = fixture(kind="recurrent", interline=False)
    advance(e, parent, 2)
    roots = tuple(e._passengers)
    for root in roots:
        present_pair(e, parent, root=root, counter=True)
    early = e.proof_information_periods()[1].reentry_rows
    advance(e, parent, 3)
    late = e.proof_information_periods()[-1].reentry_rows
    for row in late:
        if row.analyst_root in roots:
            assert row.view.native_proof_on_file
            assert row.view.earlier_presentation_calls[0].person_ticket == 0
    assert all(not r.view.native_proof_on_file for r in early)
    for root in roots:
        assert len(e._own_tickets[root]) == 2
        own = [
            r.view
            for p in e.own_information_periods()
            for r in p.reentry_rows
            if r.analyst_root == root
        ]
        assert len({v.native_exact_expiry_utc for v in own}) == 1
    assert any(r["replacement_record"] is not None for r in e.rebooking_history)


def test_capture_failure_retains_native_proof_without_positive_journal(monkeypatch):
    e, parent = fixture(interline=False)
    advance(e, parent, 2)
    before = e.proof_information_periods()

    def fail(*args):
        raise RuntimeError("new injected presentation capture failure")

    monkeypatch.setattr(e, "_capture_presentation", fail)
    with pytest.raises(RuntimeError, match="injected"):
        e.present_cancelled_itinerary(original(e), airport_counter=True)
    assert 0 in e._presented and (0, 2) in e._counter_visits
    saved = e.snapshot()
    assert saved.presentation_capture_failed and saved.periods == before
    for call in (
        e.presentation_calls,
        e.proof_information_periods,
        lambda: e.step(actions(e)),
        lambda: e.present_cancelled_itinerary(original(e)),
        lambda: module.proof_information_record(e),
    ):
        with pytest.raises(ValueError, match="incomplete"):
            call()
    branch = e.branch()
    assert pickle.dumps(branch.snapshot()) == pickle.dumps(saved)
    with pytest.raises(ValueError, match="incomplete"):
        branch.presentation_calls()


def test_native_marks_failure_keeps_pending_and_no_retry(monkeypatch):
    e, parent = fixture(interline=False)
    advance(e, parent)

    def fail(*args):
        raise RuntimeError("new native marks failure")

    monkeypatch.setattr(PotentialLineageMarketEngine, "_marks", fail)
    with pytest.raises(RuntimeError, match="marks failure"):
        e.step(actions(e))
    saved = e.snapshot()
    assert saved.pending is not None and len(saved.periods) == 1
    with pytest.raises(ValueError, match="incomplete"):
        e.proof_information_periods()
    with pytest.raises(ValueError, match="incomplete"):
        e.step(actions(e))


def test_postclose_capture_failure_native_mutations_retained(monkeypatch):
    e, parent = fixture(interline=False)
    advance(e, parent)
    original_query = e.own_information_periods

    def fail():
        raise RuntimeError("new postclose proof join failure")

    monkeypatch.setattr(e, "own_information_periods", fail)
    with pytest.raises(RuntimeError, match="postclose"):
        e.step(actions(e))
    assert e.world.t == 2 and len(original_query()) == 2
    assert len(e.snapshot().periods) == 1 and e.snapshot().pending is not None
    with pytest.raises(ValueError, match="incomplete"):
        module.proof_information_record(e)


def test_refusals_before_native_mutation():
    e, _ = fixture()
    before = pickle.dumps(e.snapshot())
    with pytest.raises(ValueError):
        e.step(actions(e), _requests=[])
    with pytest.raises(ValueError):
        e.step(actions(e), _choices=[])
    assert pickle.dumps(e.snapshot()) == before
    e.booking_clock = UtcBookingClock(
        tuple(t + timedelta(minutes=1) for t in e.booking_clock.boundaries)
    )
    with pytest.raises(ValueError, match="clocks"):
        e.step(actions(e))
    assert e.snapshot().pending is None


def test_record_owned_separate_analyst_links_no_producer_replay(monkeypatch):
    e, parent = fixture(interline=False)
    advance(e, parent, 2)
    present_pair(e, parent, counter=True)
    advance(e, parent)

    def fail(*args, **kwargs):
        raise AssertionError("reader called native producer")

    monkeypatch.setattr(e, "step", fail)
    monkeypatch.setattr(e, "_marks", fail)
    out = module.proof_information_record(e)
    assert out["native_presentation_calls_captured"]
    assert not out[
        "actual_received_disclosure_complete_knowledge_or_legal_PNR_authenticated"
    ]
    assert "analyst_root" in json.dumps(out["analyst_context"])
    json.dumps(out, allow_nan=False)
    out["proof_views_by_period"][-1]["views"][0]["earlier_presentation_calls"][0][
        "airport_counter"
    ] = False
    assert (
        len(
            module.proof_information_record(e)["proof_views_by_period"][-1]["views"][0][
                "earlier_presentation_calls"
            ]
        )
        == 1
    )
    with pytest.raises(TypeError):
        module.proof_information_record(parent)
    with pytest.raises(TypeError):
        module.proof_information_view("row")


def test_same_call_native_counter_menu_status_and_no_repeated_forward():
    e, parent = fixture(interline=False, counter_menu=True)
    advance(e, parent, 2)
    present_pair(e, parent, counter=True)
    advance(e, parent)
    rows = [r for r in e.rebooking_history if r["period"] == 2]
    assert rows[0]["rescue_eligible"] and rows[0]["airport_counter_proved_this_period"]
    assert rows[0]["rescue_option_fare"] == 99
    assert all(not r["rescue_eligible"] for r in rows[1:])
    advance(e, parent)
    rows = [r for r in e.rebooking_history if r["period"] == 3]
    assert not rows[0]["rescue_eligible"]
    assert not rows[0]["airport_counter_proved_this_period"]


def test_successful_presentation_at_terminal_retained_no_new_attempt():
    e, parent = fixture(interline=False)
    advance(e, parent, 5)
    present_pair(e, parent, counter=True)
    assert e.presentation_calls()[0].fact.native_period == 5
    assert len(e.proof_information_periods()) == 5
    assert all(
        not r.view.native_proof_on_file
        for p in e.proof_information_periods()
        for r in p.reentry_rows
    )


def test_original_proof_after_interline_replacement_no_new_attempt_or_ticket():
    e, parent = fixture(interline=True)
    advance(e, parent, 2)
    assert all(p.active_record is not None for p in e._passengers.values())
    kept = next(b for cs in e.world.carriers for b in cs.bookings)
    assert kept.record_id not in e._passengers
    saved = pickle.dumps(e.snapshot())
    with pytest.raises(ValueError):
        e.present_cancelled_itinerary(kept)
    assert pickle.dumps(e.snapshot()) == saved
    present_pair(e, parent, counter=True)
    attempts = len(e.rebooking_history)
    advance(e, parent, 3)
    assert len(e.rebooking_history) == attempts
    assert len(e.presentation_calls()) == 1
    assert e.presentation_calls()[0].fact.person_ticket == 0


def test_parent_proof_capture_divergence_refused_without_repair():
    e, parent = fixture(interline=False)
    advance(e, parent, 2)
    # Explicit corrupt NEW local source state, not a native-produced proof.
    e._presented.add(0)
    with pytest.raises(RuntimeError, match="proof status"):
        e.step(actions(e))
    assert e.snapshot().pending == () and 0 in e._presented
    with pytest.raises(ValueError, match="incomplete"):
        e.presentation_calls()
