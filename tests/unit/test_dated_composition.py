"""A.5/A.14/B0/V9: integrated UTC eligibility, held seats and lineage ownership."""

import importlib.util
import os
import pickle
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from airline_market.agents.fixed import FixedCoefficientController
from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.dated_rescue import (
    DatedRebookingPolicy,
    DatedRescueMenu,
    UtcWindow,
)
from airline_market.policy.dated_reserve import (
    DatedSeatReserve,
    UtcBookingClock,
)
from airline_market.policy.regulator import CapPolicy, CapWindow
from airline_market.sim.arrivals import Request
from airline_market.sim.disruption import RivalExit
from airline_market.sim.engine import MarketEngine
from airline_market.sim.tape import ExogenousTape
from airline_market.spec import compile_market

if os.environ.get("DATED_COMPOSITION_SOURCE"):
    spec = importlib.util.spec_from_file_location(
        "staged_dated_composition", os.environ["DATED_COMPOSITION_SOURCE"]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
else:
    from airline_market.policy import dated_composition as module

BASE = datetime(2026, 5, 1, tzinfo=timezone.utc)


def hour(n):
    return BASE + timedelta(hours=n)


def fixture(
    *,
    window=(12, 40),
    original=(90, 100),
    valid_hours=72,
    share=".10",
    reserve=True,
    extra_exit=False,
    design="same_route",
    **kwargs,
):
    c = compile_market(
        screen_spec(
            {
                "T": 5,
                "actions": 2,
                "n_spokes": 1,
                "capacity": 20,
                "capacity_support": [3, 4],
                "arrival_probability": 0.741,
            }
        )
    )
    legs = tuple(l.id for l in c.spec.legs if l.carrier == "A")
    dep = {l.id: hour(96) for l in c.spec.legs}
    dep.update({l: hour(d) for l, d in zip(legs, (48, 60))})
    clock = UtcBookingClock(tuple(hour(x) for x in (0, 12, 24, 30, 36, 40)))
    policy = (
        DatedSeatReserve(hour(12), share, legs, tuple(dep.items())) if reserve else None
    )
    fares = (
        {"route_fares": tuple((m.id, 99) for m in c.spec.markets)}
        if design == "same_route"
        else {
            "distance_tiers": ((500, 200), (1000, 300), (None, 400)),
            "market_distances_km": tuple((m.id, 300) for m in c.spec.markets),
        }
    )
    menu = DatedRescueMenu(
        UtcWindow(*(hour(x) for x in window)),
        design,
        cancelled_departures=None
        if original is None
        else UtcWindow(*(hour(x) for x in original)),
        **fares,
    )
    e = module.DatedRescueReserveMarketEngine(
        c,
        booking_clock=clock,
        leg_departures=tuple(dep.items()),
        dated_rebooking=DatedRebookingPolicy(valid_hours, 2, 100, True),
        dated_menu=menu,
        dated_reserve=policy,
        exits=(RivalExit("B", 1),) + ((RivalExit("A", 2),) if extra_exit else ()),
        **kwargs,
    )
    e.reset(
        tape=ExogenousTape.generate(
            c, 23, registry=StreamRegistry(23974, "unit/V9_dated_composition")
        )
    )
    return e


def actions(e):
    return {
        x.code: PriceAction(
            {j: 400 if i == 0 else 300 for j in e.compiled.products_of[i]}
        )
        for i, x in enumerate(e.compiled.spec.carriers)
    }


def empty(e):
    return e.step(actions(e), _requests=[])


def strand(e, n=1):
    c = e.compiled
    j = max(c.products_of[1], key=lambda j: len(c.spec.products[j].legs))
    e.step(
        actions(e),
        _requests=[Request(i, int(c.product_market[j]), 0, 0, 0, 0) for i in range(n)],
        _choices=[j] * n,
    )
    e.begin_period()
    return tuple(p.original for p in e._passengers.values())


def cash(e):
    for s in e.world.carriers:
        assert s.cash == pytest.approx(
            s.revenue - s.refunds - s.penalties, rel=0, abs=1e-10
        )
    p = e.conservation()
    assert p["original_booked_passengers"] == p["active_plus_pending_plus_unserved"]
    for r in e.reserve_balance():
        assert (
            r["initial"] == r["held"] + r["used"] + r["released"] + r["capacity_lost"]
        )


def only_held(e):
    # Explicit unit capacity-loss setup after original rival sales; A has no tickets.
    e.world.carriers[0].capacity[:] = 2
    assert all(r["held"] == r["physical_remaining"] == 2 for r in e.reserve_balance())


def test_one_engine_consumes_held_stock_with_dated_proof_and_timestamps():
    e = fixture()
    (b,) = strand(e)
    e.present_cancelled_itinerary(b)
    empty(e)
    r = e.rebooking_history[-1]
    assert (
        r["fare"] == 99
        and r["reserve_eligible"]
        and r["dated_menu_active"]
        and r["dated_cancelled_departure_eligible"]
    )
    assert (
        r["attempt_utc"] == hour(12).isoformat()
        and r["reentry_deadline_utc"] == hour(84).isoformat()
    )
    assert len(r["held_seats_used"]) == 2 and all(
        x["used"] == 1 for x in e.reserve_balance()
    )
    cash(e)


@pytest.mark.parametrize(
    "window,original,proved",
    [
        ((24, 40), (90, 100), True),
        ((0, 12), (90, 100), True),
        ((12, 40), (0, 96), True),
        ((12, 40), (90, 100), False),
    ],
)
def test_ineligible_lineage_cannot_consume_held_seats(window, original, proved):
    e = fixture(window=window, original=original)
    (b,) = strand(e)
    only_held(e)
    if proved:
        e.present_cancelled_itinerary(b)
    empty(e)
    r = e.rebooking_history[-1]
    assert (
        r["replacement_record"] is None
        and not r["reserve_eligible"]
        and not r["held_seats_used"]
    )
    assert all(x["used"] == 0 for x in e.reserve_balance())
    cash(e)


def test_counter_eligibility_is_checked_on_same_native_UTC_opportunity():
    e = fixture(design="counter_three_tier")
    (b,) = strand(e)
    only_held(e)
    e.present_cancelled_itinerary(b)
    empty(e)
    assert e.rebooking_history[-1]["replacement_record"] is None
    # First leg releases at 24h; the second remains held until 36h.
    e.present_cancelled_itinerary(b, airport_counter=True)
    empty(e)
    r = e.rebooking_history[-1]
    assert r["fare"] == 200 and r["reserve_eligible"] and len(r["held_seats_used"]) == 1
    cash(e)


def test_releases_are_per_leg_exact_UTC_and_do_not_refill_used_quota():
    e = fixture()
    (b,) = strand(e)
    e.present_cancelled_itinerary(b)
    empty(e)
    rows = {r["leg"]: r for r in e.reserve_balance()}
    legs = e.dated_reserve.affected_legs
    assert (
        rows[legs[0]]["released"] == 1
        and rows[legs[0]]["used"] == 1
        and rows[legs[1]]["held"] == 1
    )
    empty(e)
    empty(e)
    releases = [x for x in e.reserve_events if x["kind"] == "release"]
    assert len(releases) == 2
    assert all(x["observed_utc"] == x["release_due_utc"] for x in releases)
    assert all(x["released"] == 1 and x["used"] == 1 for x in e.reserve_balance())
    cash(e)


def test_ordinary_native_request_cannot_buy_only_held_stock():
    e = fixture()
    strand(e)
    only_held(e)
    c = e.compiled
    j = max(c.products_of[0], key=lambda j: len(c.spec.products[j].legs))
    market = int(c.product_market[j])
    assert not any(e._sellable_global()[list(c.products_of[0])])
    e.step(actions(e), _requests=[Request(0, market, 0, 0, 0, 0)])
    assert not e.world.carriers[0].bookings and all(
        x["used"] == 0 for x in e.reserve_balance()
    )
    cash(e)


def test_forged_cancelled_booking_cannot_unlock_held_seats():
    e = fixture()
    (b,) = strand(e)
    with pytest.raises(ValueError):
        e.present_cancelled_itinerary(replace(b, fare=b.fare + 1))
    cash(e)


def test_active_cap_is_cheaper_than_rescue_fare_on_held_sale():
    probe = fixture()
    j = max(
        probe.compiled.products_of[0],
        key=lambda j: len(probe.compiled.spec.products[j].legs),
    )
    pid = probe.compiled.spec.products[j].id
    e = fixture(cap_policy=CapPolicy((CapWindow(1, 5, ((pid, 80),), 1),)))
    (b,) = strand(e)
    e.present_cancelled_itinerary(b)
    empty(e)
    r = e.rebooking_history[-1]
    assert r["fare"] == 80 and r["reserve_eligible"] and not r["rescue_discount_used"]
    cash(e)


def test_expiry_between_opportunities_does_not_spend_still_held_stock():
    e = fixture(valid_hours=17)
    (b,) = strand(e)
    only_held(e)
    empty(e)
    empty(e)
    e.present_cancelled_itinerary(b)
    before = len(e.rebooking_history)
    empty(e)
    assert e.world.t == 4 and len(e.rebooking_history) == before
    assert (
        e.reentry_deadline_utc(b.record_id) == hour(29)
        and e._passengers[b.record_id].deadline == 3
    )
    assert all(x["used"] == 0 for x in e.reserve_balance())
    cash(e)


def test_repeated_cancellation_preserves_first_deadline_no_quota_refill():
    e = fixture(extra_exit=True, valid_hours=18)
    (b,) = strand(e)
    e.present_cancelled_itinerary(b)
    empty(e)
    p = e._passengers[b.record_id]
    assert p.active_record is None and p.first_stranded_at == 1 and p.deadline == 3
    assert e.reentry_deadline_utc(b.record_id) == hour(30)
    assert len(e._passengers) == 1 and all(
        x["initial"] == 2 and x["used"] == 1 for x in e.reserve_balance()
    )
    cash(e)


def test_timing_distinguishes_cancelled_first_success_from_current_service():
    e = fixture(extra_exit=True)
    (b,) = strand(e)
    e.present_cancelled_itinerary(b)
    empty(e)
    row = e.timed_rebooking_metrics()["rows"][0]
    assert (
        row["first_rebooked_within_hours"] == 1
        and row["active_replacement_booked_within_hours"] == 0
        and row["stranded_lineages"] == 1
    )
    assert (
        row["first_rebooked_rate"] == 1
        and not e.timed_rebooking_metrics()["primary_B4_metric_adopted"]
    )
    cash(e)


def test_unserved_timing_keeps_censored_denominator_and_null_rate():
    e = fixture()
    strand(e)
    row = e.timed_rebooking_metrics()["rows"][1]
    assert (
        row["stranded_lineages"] == 1
        and row["first_rebooked_rate"] is None
        and row["right_censored_without_first_success"] == 1
    )
    assert (
        row["first_rebooked_rate_lower"] == 0 and row["first_rebooked_rate_upper"] == 1
    )


def test_nested_snapshot_and_branch_own_both_mutable_ledgers():
    e = fixture()
    (b,) = strand(e)
    e.present_cancelled_itinerary(b)
    snap = e.snapshot()
    before = pickle.dumps(snap)
    other = e.branch()
    empty(other)
    assert (
        pickle.dumps(e.snapshot()) == before
        and other.reserve_events is not e.reserve_events
    )
    empty(e)
    assert pickle.dumps(e.snapshot()) == pickle.dumps(other.snapshot())
    e.restore(snap)
    assert pickle.dumps(e.snapshot()) == before
    other.rebooking_history[0]["fare"] = -1
    assert not e.rebooking_history
    cash(e)


def test_restore_rejects_different_public_departures():
    e = fixture()
    other = e.branch()
    other.leg_departures = tuple(
        (k, v + timedelta(hours=1)) for k, v in other.leg_departures
    )
    with pytest.raises(ValueError, match="dates/policy"):
        other.restore(e.snapshot())


def test_forced_choice_hook_cannot_bypass_hold_after_dated_gate_closes():
    e = fixture(window=(0, 12))
    strand(e)
    with pytest.raises(ValueError, match="forced-choice"):
        e.step(actions(e), _requests=[], _choices=[])


def test_reserve_and_rescue_departure_banks_must_agree():
    e = fixture()
    p = replace(
        e.dated_reserve,
        leg_departures=tuple(
            (k, v + timedelta(hours=1)) for k, v in e.dated_reserve.leg_departures
        ),
    )
    with pytest.raises(ValueError):
        module.DatedRescueReserveMarketEngine(
            e.compiled,
            booking_clock=e.booking_clock,
            leg_departures=e.leg_departures,
            dated_rebooking=e.dated_rebooking,
            dated_menu=e.dated_menu,
            dated_reserve=p,
            exits=e.exits,
        )


def test_no_reserve_matches_dated_rescue_native_sales():
    e = fixture(reserve=False)
    (b,) = strand(e)
    e.present_cancelled_itinerary(b)
    empty(e)
    assert (
        e.rebooking_history[-1]["fare"] == 99
        and "reserve_eligible" not in e.rebooking_history[-1]
    )
    cash(e)


def test_no_policy_is_byte_identical_to_native_arrival_tape():
    c = fixture().compiled
    e = module.DatedRescueReserveMarketEngine(c)
    native = MarketEngine(c)
    tape = ExogenousTape.generate(
        c, 23, registry=StreamRegistry(23975, "unit/V9_dated_composition_none")
    )
    e.reset(tape=tape)
    native.reset(tape=tape)
    for _ in range(5):
        e.step(actions(e))
        native.step(actions(native))
        assert pickle.dumps(e.world) == pickle.dumps(native.world)
        assert pickle.dumps(e.trace()) == pickle.dumps(native.trace())
    assert e.metrics() == native.metrics()


def test_single_controller_forward_and_callback_keep_native_information(monkeypatch):
    e = fixture()
    ctrl = FixedCoefficientController((0, 0, 0))
    e.controllers[0] = ctrl
    act, observe = ctrl.act, ctrl.observe_outcome
    calls = []
    callbacks = []

    def spy(obs):
        assert obs.privileged is None
        calls.append(obs)
        return act(obs)

    def done(x):
        callbacks.append(x)
        return observe(x)

    monkeypatch.setattr(ctrl, "act", spy)
    monkeypatch.setattr(ctrl, "observe_outcome", done)
    e.step({"B": actions(e)["B"]}, _requests=[])
    assert len(calls) == len(callbacks) == 1


def test_reset_clears_both_ledgers_and_keeps_public_contract():
    e = fixture()
    (b,) = strand(e)
    e.present_cancelled_itinerary(b)
    empty(e)
    clock, policy, menu, reserve = (
        e.booking_clock,
        e.dated_rebooking,
        e.dated_menu,
        e.dated_reserve,
    )
    e.reset(tape=e.tape)
    assert (
        not e._passengers
        and not e.rebooking_history
        and not e.reserve_events
        and not e._reserve_ledger
        and not e._dated_released
    )
    assert (e.booking_clock, e.dated_rebooking, e.dated_menu, e.dated_reserve) == (
        clock,
        policy,
        menu,
        reserve,
    )


def test_future_release_is_not_rounded_to_terminal_by_composition():
    template = fixture()
    dep = tuple((k, hour(96)) for k, v in template.leg_departures)
    p = replace(template.dated_reserve, leg_departures=dep)
    e = module.DatedRescueReserveMarketEngine(
        template.compiled,
        booking_clock=template.booking_clock,
        leg_departures=dep,
        dated_rebooking=template.dated_rebooking,
        dated_menu=template.dated_menu,
        dated_reserve=p,
    )
    e.reset(tape=template.tape)
    while not e.terminal():
        empty(e)
    assert all(x["held"] == 2 and x["released"] == 0 for x in e.reserve_balance())
    assert not any(x["kind"] == "release" for x in e.reserve_events)


def test_typed_reserve_policy_is_required_before_public_bank_comparison():
    e = fixture()
    with pytest.raises(TypeError, match="DatedSeatReserve"):
        module.DatedRescueReserveMarketEngine(
            e.compiled,
            booking_clock=e.booking_clock,
            leg_departures=e.leg_departures,
            dated_rebooking=e.dated_rebooking,
            dated_menu=e.dated_menu,
            dated_reserve=object(),
        )
