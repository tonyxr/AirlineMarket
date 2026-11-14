"""A.14/B0/V9: exact elapsed-clock release, native sales and snapshot ownership."""

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
from airline_market.policy.regulator import CapPolicy, CapWindow
from airline_market.policy.rescue import RescueFareMenu
from airline_market.sim.arrivals import Request
from airline_market.sim.disruption import RivalExit
from airline_market.sim.engine import MarketEngine
from airline_market.sim.rebooking import RebookingPolicy
from airline_market.sim.tape import ExogenousTape
from airline_market.spec import compile_market

if os.environ.get("DATED_RESERVE_SOURCE"):
    s = importlib.util.spec_from_file_location(
        "staged_dated_reserve", os.environ["DATED_RESERVE_SOURCE"]
    )
    m = importlib.util.module_from_spec(s)
    sys.modules[s.name] = m
    s.loader.exec_module(m)
else:
    from airline_market.policy import dated_reserve as m

BASE = datetime(2026, 5, 1, tzinfo=timezone.utc)


def hour(n):
    return BASE + timedelta(hours=n)


def fixture(
    *,
    hours=(0, 12, 24, 30, 36, 40),
    reveal=12,
    cutoffs=(24, 30),
    share=".10",
    exits=True,
    **kwargs,
):
    c = compile_market(
        screen_spec(
            dict(
                T=5,
                actions=2,
                n_spokes=1,
                capacity=20,
                capacity_support=[3, 4],
                arrival_probability=0.741,
            )
        )
    )
    legs = tuple(leg.id for leg in c.spec.legs if leg.carrier == "A")
    assert len(legs) == 2
    dep = {leg.id: hour(96) for leg in c.spec.legs}
    dep.update({leg: hour(cutoff + 24) for leg, cutoff in zip(legs, cutoffs)})
    clock = m.UtcBookingClock(tuple(hour(x) for x in hours))
    p = m.DatedSeatReserve(hour(reveal), share, legs, tuple(dep.items()))
    menu = RescueFareMenu(
        1, 5, "same_route", route_fares=tuple((x.id, 99) for x in c.spec.markets)
    )
    e = m.DatedReserveMarketEngine(
        c,
        booking_clock=clock,
        dated_reserve=p,
        rebooking=RebookingPolicy(4, 2, 100, True),
        rescue_menu=menu,
        exits=(RivalExit("B", 1),) if exits else (),
        **kwargs,
    )
    e.reset(
        tape=ExogenousTape.generate(
            c, 23, registry=StreamRegistry(21974, "unit/V9_dated_reserve")
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


def strand(e):
    c = e.compiled
    j = max(c.products_of[1], key=lambda j: len(c.spec.products[j].legs))
    e.step(
        actions(e),
        _requests=[Request(0, int(c.product_market[j]), 0, 0, 0, 0)],
        _choices=[j],
    )
    return e.stranded[0].booking


def cash(e):
    for x in e.world.carriers:
        assert x.cash == pytest.approx(
            x.revenue - x.refunds - x.penalties, abs=1e-10, rel=0
        )
    p = e.conservation()
    assert p["original_booked_passengers"] == p["active_plus_pending_plus_unserved"]


@pytest.mark.parametrize(
    "values",
    [
        (hour(0),),
        (hour(1), hour(0)),
        (hour(0), hour(0)),
        (datetime(2026, 5, 1), hour(1)),
    ],
)
def test_invalid_clock_is_not_inferred(values):
    with pytest.raises(ValueError):
        m.UtcBookingClock(values)


@pytest.mark.parametrize("period", [-1, 6, True, 1.5])
def test_clock_period_access_is_explicit(period):
    c = m.UtcBookingClock(tuple(hour(x) for x in range(6)))
    with pytest.raises(ValueError):
        c.at(period)


def test_offset_normalization_and_24_elapsed_hours():
    p = m.DatedSeatReserve(
        "2026-10-30T00:00:00Z", ".02", ("L",), (("L", "2026-11-01T12:00:00-05:00"),)
    )
    due = p.release_at("L")
    assert due == m.utc("2026-10-31T13:00:00-04:00")
    assert dict(p.leg_departures)["L"] - due == timedelta(hours=24)
    c = m.UtcBookingClock(("2026-10-30T00:00:00Z", "2026-10-31T13:00:00-04:00"))
    assert c.index(due) == 1 and c.boundaries[1].tzinfo == timezone.utc


@pytest.mark.parametrize(
    "kw,message",
    [
        (dict(hours=(0, 12, 24, 30, 36)), "boundary per native"),
        (dict(hours=(0, 12, 24, 30, 36, 48)), "before every flight"),
        (dict(reveal=-1), "reveal before cohort"),
        (dict(reveal=11), "exact declared UTC boundary"),
        (dict(cutoffs=(25, 30)), "exact declared UTC boundary"),
    ],
)
def test_model_and_calendar_must_name_exact_supported_events(kw, message):
    with pytest.raises(ValueError, match=message):
        fixture(**kw)


def test_per_leg_release_is_exact_and_idempotent():
    e = fixture()
    empty(e)
    e.begin_period()
    legs = e.dated_reserve.affected_legs
    assert all(x["initial"] == x["held"] == 2 for x in e.reserve_balance())
    empty(e)
    rows = {x["leg"]: x for x in e.reserve_balance()}
    assert (
        rows[legs[0]]["held"] == 0
        and rows[legs[0]]["released"] == 2
        and rows[legs[1]]["held"] == 2
    )
    empty(e)
    assert all(x["released"] == 2 and x["held"] == 0 for x in e.reserve_balance())
    released = [x for x in e.reserve_events if x["kind"] == "release"]
    assert len(released) == 2
    for x in released:
        assert (
            m.utc(x["observed_utc"])
            == m.utc(x["release_due_utc"])
            == e.dated_reserve.release_at(x["leg"])
        )
    before = pickle.dumps(e.reserve_events)
    e.reserve_balance()
    e.begin_period()
    assert pickle.dumps(e.reserve_events) == before


def test_original_cancelled_passenger_uses_held_stock_at_native_fare():
    e = fixture()
    b = strand(e)
    e.present_cancelled_itinerary(b)
    empty(e)
    r = e.rebooking_history[-1]
    assert (
        r["fare"] == 99
        and r["reserve_eligible"]
        and r["stage"] == "interline"
        and len(r["held_seats_used"]) == 2
    )
    legs = e.dated_reserve.affected_legs
    rows = {x["leg"]: x for x in e.reserve_balance()}
    assert rows[legs[0]]["used"] == rows[legs[1]]["used"] == 1
    assert (
        rows[legs[0]]["released"] == 1
        and rows[legs[0]]["held"] == 0
        and rows[legs[1]]["held"] == 1
    )
    cash(e)


def test_capacity_loss_writeoff_does_not_refill_quota():
    e = fixture()
    empty(e)
    e.begin_period()
    e.world.carriers[0].capacity[:] = 1
    assert all(x["capacity_lost"] == 1 and x["held"] == 1 for x in e.reserve_balance())
    e.world.carriers[0].capacity[:] = 20
    assert all(x["initial"] == 2 and x["held"] == 1 for x in e.reserve_balance())
    empty(e)
    cash(e)


def test_cutoffs_after_terminal_stay_held_not_rounded_to_terminal():
    e = fixture(cutoffs=(48, 54), exits=False)
    while not e.terminal():
        empty(e)
    assert all(x["held"] == 2 and x["released"] == 0 for x in e.reserve_balance())
    assert (
        not any(x["kind"] == "release" for x in e.reserve_events)
        and not e._reserve_released
    )


def test_cutoff_on_terminal_releases_at_that_exact_boundary():
    e = fixture(cutoffs=(40, 40), exits=False)
    while not e.terminal():
        empty(e)
    assert all(x["held"] == 0 and x["released"] == 2 for x in e.reserve_balance())
    assert all(
        x["observed_utc"] == hour(40).isoformat()
        for x in e.reserve_events
        if x["kind"] == "release"
    )


def test_reveal_at_cutoff_holds_zero_without_requiring_past_clock_boundary():
    e = fixture(reveal=12, cutoffs=(11, 12), hours=(0, 12, 14, 18, 24, 30), exits=False)
    empty(e)
    e.begin_period()
    assert all(x["initial"] == x["held"] == 0 for x in e.reserve_balance())
    assert all(x["kind"] == "hold_skipped_after_release" for x in e.reserve_events)


def test_future_reveal_does_not_activate_within_booking_horizon():
    e = fixture(reveal=50, cutoffs=(60, 66), exits=False)
    while not e.terminal():
        empty(e)
    assert not e.reserve_events and not e.reserve_balance()


def test_branch_restore_keep_independent_released_leg_sets_and_native_world():
    e = fixture()
    b = strand(e)
    e.present_cancelled_itinerary(b)
    e.begin_period()
    snap = e.snapshot()
    branch = e.branch()
    assert (
        branch.booking_clock is e.booking_clock
        and branch.dated_reserve is e.dated_reserve
    )
    assert (
        branch._dated_released is not e._dated_released
        and branch._reserve_ledger is not e._reserve_ledger
    )
    empty(branch)
    assert branch._dated_released != e._dated_released
    e.restore(snap)
    empty(e)
    assert pickle.dumps(
        (e.world, e.reserve_events, e.rebooking_history)
    ) == pickle.dumps((branch.world, branch.reserve_events, branch.rebooking_history))
    e.reset(tape=e.tape)
    assert not e._dated_released and not e.reserve_events


def test_different_public_clock_snapshot_cannot_change_future_cutoff():
    e = fixture()
    other = fixture(hours=(0, 12, 24, 30, 37, 41))
    with pytest.raises(ValueError, match="clock/policy differs"):
        other.restore(e.snapshot())


def test_no_policy_path_is_byte_identical_to_base_native_engine():
    raw = fixture(exits=False)
    a = m.DatedReserveMarketEngine(raw.compiled)
    b = MarketEngine(raw.compiled)
    for e in [a, b]:
        e.reset(tape=raw.tape)
        while not e.terminal():
            e.step(actions(e), _requests=[])
    assert pickle.dumps((a.world, a.trace(), a.metrics())) == pickle.dumps(
        (b.world, b.trace(), b.metrics())
    )


def test_cheaper_cap_still_limits_held_seat_native_sale():
    raw = fixture()
    cap = CapPolicy(
        (
            CapWindow(
                1, 3, tuple((p.id, 50) for p in raw.compiled.spec.products), "1.30"
            ),
        )
    )
    e = fixture(cap_policy=cap)
    b = strand(e)
    e.present_cancelled_itinerary(b)
    empty(e)
    assert e.rebooking_history[-1]["fare"] == 65
    cash(e)


def test_actual_controller_called_once_with_unchanged_permitted_observation(
    monkeypatch,
):
    e = fixture(exits=False)
    e.controllers[0] = FixedCoefficientController((0, 0, 0))
    ctrl = e.controllers[0]
    act = ctrl.act
    observe = ctrl.observe_outcome
    calls = []
    outcomes = []

    def spy(obs):
        assert obs.privileged is None
        calls.append(obs)
        return act(obs)

    def done(x):
        outcomes.append(x)
        return observe(x)

    monkeypatch.setattr(ctrl, "act", spy)
    monkeypatch.setattr(ctrl, "observe_outcome", done)
    e.step({"B": actions(e)["B"]}, _requests=[])
    assert len(calls) == len(outcomes) == 1


def test_explicit_departure_map_and_share_are_required():
    e = fixture()
    p = e.dated_reserve
    with pytest.raises(ValueError, match="every compiled leg"):
        m.DatedReserveMarketEngine(
            e.compiled,
            booking_clock=e.booking_clock,
            dated_reserve=replace(p, leg_departures=p.leg_departures[:-1]),
        )
    with pytest.raises(ValueError, match="share"):
        replace(p, share=".03")
    with pytest.raises(ValueError, match="distinct"):
        replace(p, leg_departures=p.leg_departures + (p.leg_departures[0],))
    with pytest.raises(ValueError, match="together"):
        m.DatedReserveMarketEngine(e.compiled, booking_clock=e.booking_clock)


def test_forced_choice_cannot_bypass_active_hold():
    e = fixture()
    empty(e)
    with pytest.raises(ValueError, match="forced-choice"):
        e.step(actions(e), _requests=[], _choices=[])
