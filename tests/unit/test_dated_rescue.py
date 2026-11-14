"""A.5/A.14/B0/V9: dated native eligibility, lineages and censored timing metrics."""

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
from airline_market.policy.dated_reserve import UtcBookingClock
from airline_market.policy.regulator import CapPolicy, CapWindow
from airline_market.sim.arrivals import Request
from airline_market.sim.disruption import RivalExit
from airline_market.sim.engine import MarketEngine
from airline_market.sim.tape import ExogenousTape
from airline_market.spec import compile_market

if os.environ.get("DATED_RESCUE_SOURCE"):
    s = importlib.util.spec_from_file_location(
        "staged_dated_rescue", os.environ["DATED_RESCUE_SOURCE"]
    )
    m = importlib.util.module_from_spec(s)
    sys.modules[s.name] = m
    s.loader.exec_module(m)
else:
    from airline_market.policy import dated_rescue as m

BASE = datetime(2026, 5, 1, tzinfo=timezone.utc)


def hour(n):
    return BASE + timedelta(hours=n)


def fixture(
    *,
    hours=(0, 12, 24, 30, 36, 40),
    valid_hours=18,
    menu_hours=(12, 30),
    original_window=None,
    capacity=20,
    design="same_route",
    extra_exit=False,
    all_exit=False,
    with_menu=True,
    **kwargs,
):
    c = compile_market(
        screen_spec(
            dict(
                T=5,
                actions=2,
                n_spokes=1,
                capacity=capacity,
                capacity_support=[3, 4],
                arrival_probability=0.741,
            )
        )
    )
    departures = tuple(
        (leg.id, hour(96 if leg.carrier == "B" else 48)) for leg in c.spec.legs
    )
    clock = UtcBookingClock(tuple(hour(x) for x in hours))
    policy = m.DatedRebookingPolicy(valid_hours, 2, 100, True)
    fares = (
        dict(route_fares=tuple((x.id, 99) for x in c.spec.markets))
        if design == "same_route"
        else dict(
            distance_tiers=((500, 200), (1000, 300), (None, 400)),
            market_distances_km=tuple((x.id, 300) for x in c.spec.markets),
        )
    )
    menu = (
        m.DatedRescueMenu(
            m.UtcWindow(*(hour(x) for x in menu_hours)),
            design,
            cancelled_departures=None
            if original_window is None
            else m.UtcWindow(*(hour(x) for x in original_window)),
            **fares,
        )
        if with_menu
        else None
    )
    exits = [RivalExit("B", 1)]
    if extra_exit:
        exits.append(RivalExit("A", 2))
    if all_exit:
        exits.append(RivalExit("A", 1))
    e = m.DatedRescueMarketEngine(
        c,
        booking_clock=clock,
        leg_departures=departures,
        dated_rebooking=policy,
        dated_menu=menu,
        exits=tuple(exits),
        **kwargs,
    )
    e.reset(
        tape=ExogenousTape.generate(
            c, 23, registry=StreamRegistry(22974, "unit/V9_dated_rescue")
        )
    )
    return e


def actions(e):
    return {
        c.code: PriceAction(
            {j: 400 if i == 0 else 300 for j in e.compiled.products_of[i]}
        )
        for i, c in enumerate(e.compiled.spec.carriers)
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
    return tuple(p.original for p in e._passengers.values())


def cash(e):
    for s in e.world.carriers:
        assert s.cash == pytest.approx(
            s.revenue - s.refunds - s.penalties, abs=1e-10, rel=0
        )
    p = e.conservation()
    assert p["original_booked_passengers"] == p["active_plus_pending_plus_unserved"]


@pytest.mark.parametrize("a,b", [(12, 12), (12, 0)])
def test_window_must_be_ordered(a, b):
    with pytest.raises(ValueError):
        m.UtcWindow(hour(a), hour(b))


def test_window_is_start_inclusive_stop_exclusive_and_zoned():
    w = m.UtcWindow(hour(12), hour(30))
    assert w.contains(hour(12)) and not w.contains(hour(30))
    with pytest.raises(ValueError):
        m.UtcWindow(datetime(2026, 5, 1), hour(30))


@pytest.mark.parametrize("h", [0, -1, True, 1.5])
def test_hours_are_explicit_positive_integers(h):
    with pytest.raises(ValueError):
        m.DatedRebookingPolicy(h, 2)


def test_purchase_window_allows_only_eligible_native_cancelled_proof():
    e = fixture(original_window=(90, 100))
    (b,) = strand(e)
    e.present_cancelled_itinerary(b)
    empty(e)
    r = e.rebooking_history[-1]
    assert (
        r["fare"] == 99
        and r["rescue_eligible"]
        and r["dated_menu_active"]
        and r["dated_cancelled_departure_eligible"]
    )
    assert (
        r["attempt_utc"] == hour(12).isoformat()
        and r["reentry_deadline_utc"] == hour(30).isoformat()
    )
    assert e._passengers[b.record_id].active_record == r["replacement_record"]
    cash(e)


@pytest.mark.parametrize("window", [(24, 36), (0, 12)])
def test_outside_purchase_window_keeps_regular_fare(window):
    e = fixture(menu_hours=window)
    (b,) = strand(e)
    e.present_cancelled_itinerary(b)
    empty(e)
    r = e.rebooking_history[-1]
    assert r["fare"] == 400 and not r["rescue_eligible"] and not r["dated_menu_active"]
    cash(e)


def test_original_departure_window_is_explicit_and_exclusive():
    e = fixture(original_window=(0, 96))
    (b,) = strand(e)
    e.present_cancelled_itinerary(b)
    empty(e)
    r = e.rebooking_history[-1]
    assert r["fare"] == 400 and not r["dated_cancelled_departure_eligible"]


def test_unproved_or_forged_itinerary_cannot_buy_private_fare():
    e = fixture()
    (b,) = strand(e)
    with pytest.raises(ValueError):
        e.present_cancelled_itinerary(replace(b, fare=b.fare + 1))
    empty(e)
    assert (
        e.rebooking_history[-1]["fare"] == 400
        and not e.rebooking_history[-1]["rescue_eligible"]
    )


def test_counter_proof_is_required_on_the_actual_attempt_instant():
    e = fixture(design="counter_three_tier")
    (b,) = strand(e)
    e.present_cancelled_itinerary(b, airport_counter=True)
    empty(e)
    assert (
        e.rebooking_history[-1]["fare"] == 200
        and e.rebooking_history[-1]["airport_counter_proved_this_period"]
    )
    cash(e)


def test_no_counter_visit_keeps_regular_fare():
    e = fixture(design="counter_three_tier")
    (b,) = strand(e)
    e.present_cancelled_itinerary(b)
    empty(e)
    assert (
        e.rebooking_history[-1]["fare"] == 400
        and not e.rebooking_history[-1]["rescue_eligible"]
    )


def test_UTC_expiry_between_opportunities_inserts_no_attempt():
    e = fixture(valid_hours=17, all_exit=True)
    strand(e)
    while not e.terminal():
        empty(e)
    assert [r["attempt_utc"] for r in e.rebooking_history] == [
        hour(12).isoformat(),
        hour(24).isoformat(),
    ]
    assert all(
        r["reentry_deadline_utc"] == hour(29).isoformat() for r in e.rebooking_history
    )
    cash(e)


def test_exact_expiry_is_not_an_eligible_attempt():
    e = fixture(valid_hours=18, all_exit=True)
    strand(e)
    while not e.terminal():
        empty(e)
    assert [r["period"] for r in e.rebooking_history] == [1, 2]
    assert not any(
        r["attempt_utc"] == hour(30).isoformat() for r in e.rebooking_history
    )


def test_repeated_cancellation_does_not_extend_first_deadline_or_duplicate_passenger():
    e = fixture(extra_exit=True)
    (b,) = strand(e)
    e.present_cancelled_itinerary(b)
    empty(e)
    assert e._passengers[b.record_id].active_record is None and len(e._passengers) == 1
    assert (
        e.reentry_deadline_utc(b.record_id) == hour(30)
        and e._passengers[b.record_id].deadline == 3
    )
    metric = e.timed_rebooking_metrics((24,))["rows"][0]
    assert (
        metric["stranded_lineages"] == 1
        and metric["first_rebooked_within_hours"] == 1
        and metric["active_replacement_booked_within_hours"] == 0
    )
    assert (
        metric["first_rebooked_rate"] == 1
        and metric["active_replacement_rate_at_current_time"] == 0
    )
    cash(e)


def test_right_censored_unserved_lineage_keeps_full_denominator_and_null_point():
    e = fixture(all_exit=True, valid_hours=72)
    strand(e)
    while not e.terminal():
        empty(e)
    rows = {r["hours"]: r for r in e.timed_rebooking_metrics()["rows"]}
    assert (
        rows[24]["first_rebooked_rate"] == 0
        and rows[24]["followup_at_least_hours"] == 1
    )
    for h in [48, 72]:
        assert (
            rows[h]["stranded_lineages"] == 1 and rows[h]["first_rebooked_rate"] is None
        )
        assert (
            rows[h]["first_rebooked_rate_lower"] == 0
            and rows[h]["first_rebooked_rate_upper"] == 1
        )
        assert rows[h]["right_censored_without_first_success"] == 1


def test_partial_success_does_not_drop_censored_passenger():
    e = fixture(valid_hours=72)
    bs = strand(e, n=2)
    # Explicit unit capacity-loss setup after original sales; no ledger repair.
    e.world.carriers[0].capacity[:] = 1
    for b in bs:
        e.present_cancelled_itinerary(b)
    empty(e)
    r = e.timed_rebooking_metrics((48,))["rows"][0]
    assert r["stranded_lineages"] == 2 and r["first_rebooked_within_hours"] == 1
    assert (
        r["first_rebooked_rate"] is None
        and r["first_rebooked_rate_lower"] == 0.5
        and r["first_rebooked_rate_upper"] == 1
    )


def test_no_stranded_denominator_is_undefined_not_zero():
    e = fixture()
    r = e.timed_rebooking_metrics()["rows"][0]
    assert (
        r["stranded_lineages"] == 0
        and r["first_rebooked_rate"]
        is r["first_rebooked_rate_lower"]
        is r["first_rebooked_rate_upper"]
        is None
    )


def test_rebooking_without_private_menu_uses_same_dated_expiry():
    e = fixture(with_menu=False, all_exit=True, valid_hours=18)
    strand(e)
    while not e.terminal():
        empty(e)
    assert len(e.rebooking_history) == 2 and all(
        r["reentry_deadline_utc"] == hour(30).isoformat() for r in e.rebooking_history
    )


def test_branch_restore_and_public_date_ownership():
    e = fixture()
    (b,) = strand(e)
    e.present_cancelled_itinerary(b)
    snap = e.snapshot()
    branch = e.branch()
    assert branch.booking_clock is e.booking_clock and branch.dated_menu is e.dated_menu
    empty(branch)
    e.restore(snap)
    empty(e)
    assert pickle.dumps(
        (e.world, e.rebooking_history, e.timed_rebooking_metrics())
    ) == pickle.dumps(
        (branch.world, branch.rebooking_history, branch.timed_rebooking_metrics())
    )
    assert (
        e._passengers is not branch._passengers and e._attempts is not branch._attempts
    )
    other = fixture(menu_hours=(12, 36))
    with pytest.raises(ValueError, match="public dates/policy"):
        other.restore(snap)
    e.reset(tape=e.tape)
    assert not e._passengers and not e.rebooking_history


def test_no_policy_matches_actual_native_arrival_tape():
    raw = fixture()
    c = raw.compiled
    a = m.DatedRescueMarketEngine(c)
    b = MarketEngine(c)
    for e in [a, b]:
        e.reset(tape=raw.tape)
        while not e.terminal():
            e.step(actions(e))
    assert a.world.request_log and pickle.dumps(
        (a.world, a.trace(), a.metrics())
    ) == pickle.dumps((b.world, b.trace(), b.metrics()))


def test_active_cap_and_cheaper_regular_price_preserved():
    raw = fixture()
    cap = CapPolicy(
        (
            CapWindow(
                1, 3, tuple((p.id, 50) for p in raw.compiled.spec.products), "1.30"
            ),
        )
    )
    e = fixture(cap_policy=cap)
    (b,) = strand(e)
    e.present_cancelled_itinerary(b)
    empty(e)
    assert (
        e.rebooking_history[-1]["fare"] == 65
        and e.rebooking_history[-1]["rescue_eligible"]
    )
    cash(e)


def test_controller_observations_and_one_callback_are_native(monkeypatch):
    e = fixture()
    e.controllers[0] = FixedCoefficientController((0, 0, 0))
    ctrl = e.controllers[0]
    act = ctrl.act
    outcome = ctrl.observe_outcome
    calls = []
    callbacks = []

    def spy(obs):
        assert obs.privileged is None
        calls.append(obs)
        return act(obs)

    def done(x):
        callbacks.append(x)
        return outcome(x)

    monkeypatch.setattr(ctrl, "act", spy)
    monkeypatch.setattr(ctrl, "observe_outcome", done)
    e.step({"B": actions(e)["B"]}, _requests=[])
    assert len(calls) == len(callbacks) == 1


def test_public_clock_and_departures_are_required_without_new_period_mapping():
    e = fixture()
    with pytest.raises(ValueError, match="every compiled leg"):
        m.DatedRescueMarketEngine(
            e.compiled,
            booking_clock=e.booking_clock,
            leg_departures=e.leg_departures[:-1],
            dated_rebooking=e.dated_rebooking,
        )
    with pytest.raises(ValueError, match="pre-departure"):
        m.DatedRescueMarketEngine(
            e.compiled,
            booking_clock=e.booking_clock,
            leg_departures=tuple((leg, hour(40)) for leg, v in e.leg_departures),
            dated_rebooking=e.dated_rebooking,
        )
    with pytest.raises(ValueError, match="explicit dated"):
        m.DatedRescueMarketEngine(e.compiled, booking_clock=e.booking_clock)
    with pytest.raises(ValueError):
        m.DatedRescueMenu(m.UtcWindow(hour(1), hour(2)), "unknown")


def test_duplicate_metric_horizons_do_not_change_denominators():
    e = fixture()
    with pytest.raises(ValueError):
        e.timed_rebooking_metrics((24, 24))
