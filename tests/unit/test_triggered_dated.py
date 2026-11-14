"""A.5/A.14/B0/V9: public trigger, dated eligibility, native fares and held stock."""

import hashlib
import importlib.util
import os
import pickle
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from airline_market.agents.fixed import FixedCoefficientController
from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.dated_composition import DatedRescueReserveMarketEngine
from airline_market.policy.dated_rescue import (
    DatedRebookingPolicy,
    DatedRescueMenu,
    UtcWindow,
)
from airline_market.policy.dated_reserve import DatedSeatReserve, UtcBookingClock
from airline_market.policy.regulator import CapPolicy, CapWindow
from airline_market.policy.trigger import (
    MechanicalFareRow,
    PublicMechanicalTable,
    PublicTriggerReport,
    TriggerRule,
)
from airline_market.sim.arrivals import Request
from airline_market.sim.disruption import RivalExit
from airline_market.sim.engine import MarketEngine
from airline_market.sim.tape import ExogenousTape
from airline_market.spec import compile_market

if os.environ.get("TRIGGERED_DATED_SOURCE"):
    spec = importlib.util.spec_from_file_location(
        "staged_triggered_dated", os.environ["TRIGGERED_DATED_SOURCE"]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
else:
    from airline_market.policy import triggered_dated as module

BASE = datetime(2026, 5, 1, tzinfo=timezone.utc)


def hour(n):
    return BASE + timedelta(hours=n)


def fixture(
    *,
    tau=0,
    kappa=0,
    window=2,
    start=1,
    stop=4,
    loss=True,
    trigger=True,
    menu_window=(12, 40),
    counter=False,
    extra_exit=False,
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
    dep.update({l: hour(h) for l, h in zip(legs, (48, 60))})
    clock = UtcBookingClock(tuple(hour(x) for x in (0, 12, 24, 30, 36, 40)))
    hold = DatedSeatReserve(hour(12), ".10", legs, tuple(dep.items()))
    fares = (
        {"route_fares": tuple((m.id, 99) for m in c.spec.markets)}
        if not counter
        else {
            "distance_tiers": ((500, 200), (1000, 300), (None, 400)),
            "market_distances_km": tuple((m.id, 300) for m in c.spec.markets),
        }
    )
    menu = DatedRescueMenu(
        UtcWindow(*(hour(x) for x in menu_window)),
        "counter_three_tier" if counter else "same_route",
        cancelled_departures=UtcWindow(hour(90), hour(100)),
        **fares,
    )
    refs = tuple((p.id, 100) for p in c.spec.products)
    reports = tuple(
        PublicTriggerReport(
            t, ((legs[0], 20),), ((legs[0], 10 if loss else 20),), ((legs[0], 9),), refs
        )
        for t in range(5)
    )
    bench = PublicMechanicalTable(
        tuple(
            MechanicalFareRow(
                r.period,
                r.remaining_seats,
                r.reference_fares,
                tuple((p.id, 80) for p in c.spec.products),
            )
            for r in reports
        ),
        hashlib.sha256(b"unit pre-shock demand").hexdigest(),
        hashlib.sha256(b"unit frozen rival").hexdigest(),
    )
    rule = TriggerRule(start, stop, kappa, tau, window)
    e = module.TriggeredDatedMarketEngine(
        c,
        booking_clock=clock,
        leg_departures=tuple(dep.items()),
        dated_rebooking=DatedRebookingPolicy(72, 2, 100, True),
        dated_menu=menu,
        dated_reserve=hold,
        exits=(RivalExit("B", 1),) + ((RivalExit("A", 2),) if extra_exit else ()),
        trigger_rule=rule if trigger else None,
        mechanical_benchmark=bench if trigger else None,
        **kwargs,
    )
    e.reset(
        tape=ExogenousTape.generate(
            c, 23, registry=StreamRegistry(24974, "unit/V9_triggered_dated")
        )
    )
    return e, reports


def actions(e, fare=400):
    return {
        x.code: PriceAction(
            {j: fare if i == 0 else 300 for j in e.compiled.products_of[i]}
        )
        for i, x in enumerate(e.compiled.spec.carriers)
    }


def publish(e, reports):
    return e.publish_regulator_report_at(
        e.booking_clock.at(e.world.t), reports[e.world.t]
    )


def empty(e, reports):
    publish(e, reports)
    return e.step(actions(e), _requests=[])


def strand(e):
    c = e.compiled
    j = max(c.products_of[1], key=lambda j: len(c.spec.products[j].legs))
    e.step(
        actions(e),
        _requests=[Request(0, int(c.product_market[j]), 0, 0, 0, 0)],
        _choices=[j],
    )
    e.reserve_balance()
    return next(iter(e._passengers.values())).original


def cash(e):
    for c in e.world.carriers:
        assert c.cash == pytest.approx(
            c.revenue - c.refunds - c.penalties, abs=1e-10, rel=0
        )
    p = e.conservation()
    assert p["original_booked_passengers"] == p["active_plus_pending_plus_unserved"]
    for r in e.reserve_balance():
        assert (
            r["initial"] == r["held"] + r["used"] + r["released"] + r["capacity_lost"]
        )


def held_only(e):
    e.world.carriers[0].capacity[:] = 2
    assert all(x["held"] == x["physical_remaining"] == 2 for x in e.reserve_balance())


def test_trigger_caps_private_held_sale_before_native_booking_and_quotes():
    e, rs = fixture()
    b = strand(e)
    e.present_cancelled_itinerary(b)
    empty(e, rs)
    r = e.rebooking_history[-1]
    assert (
        r["fare"] == 80
        and r["reserve_eligible"]
        and r["dated_menu_active"]
        and len(r["held_seats_used"]) == 2
        and not r["rescue_discount_used"]
    )
    assert (
        r["attempt_utc"] == hour(12).isoformat()
        and r["reentry_deadline_utc"] == hour(84).isoformat()
    )
    assert max(e.trace()[-1]["prices"]) == 80 and e.trigger.active_until
    assert e.world.carriers[0].revenue == 80 and e.world.carriers[1].refunds == 300
    cash(e)


@pytest.mark.parametrize(
    "proved,window", [(False, (12, 40)), (True, (24, 40)), (True, (0, 12))]
)
def test_cheap_trigger_price_never_unlocks_held_stock_for_ineligible(proved, window):
    e, rs = fixture(menu_window=window)
    b = strand(e)
    held_only(e)
    if proved:
        e.present_cancelled_itinerary(b)
    empty(e, rs)
    r = e.rebooking_history[-1]
    assert (
        r["replacement_record"] is None
        and not r["reserve_eligible"]
        and not r["held_seats_used"]
    )
    assert (
        all(x["used"] == 0 for x in e.reserve_balance())
        and max(e.trace()[-1]["prices"]) == 80
    )
    cash(e)


def test_counter_proof_stays_current_opportunity_under_trigger():
    e, rs = fixture(counter=True)
    b = strand(e)
    held_only(e)
    e.present_cancelled_itinerary(b)
    empty(e, rs)
    assert e.rebooking_history[-1]["replacement_record"] is None
    e.present_cancelled_itinerary(b, airport_counter=True)
    empty(e, rs)
    r = e.rebooking_history[-1]
    assert r["fare"] == 80 and r["reserve_eligible"] and len(r["held_seats_used"]) == 1
    cash(e)


def test_release_clock_survives_trigger_expiry_and_rearming():
    e, rs = fixture()
    b = strand(e)
    e.present_cancelled_itinerary(b)
    empty(e, rs)
    legs = e.dated_reserve.affected_legs
    rows = {x["leg"]: x for x in e.reserve_balance()}
    assert rows[legs[0]]["released"] == 1 and rows[legs[1]]["held"] == 1
    first = dict(e.trigger.active_until)
    empty(e, rs)
    assert e.trigger.active_until == first
    empty(e, rs)
    assert all(x["released"] == 1 and x["used"] == 1 for x in e.reserve_balance())
    assert all(
        x["observed_utc"] == x["release_due_utc"]
        for x in e.reserve_events
        if x["kind"] == "release"
    )
    cash(e)


def test_trigger_window_uses_explicit_native_boundaries_not_hour_duration():
    e, rs = fixture()
    strand(e)
    empty(e, rs)
    assert set(e.trigger.active_until.values()) == {3}
    assert e.booking_clock.at(1) == hour(12) and e.booking_clock.at(3) == hour(30)
    # Two native opportunities span 18 elapsed hours on this declared clock.
    assert e.booking_clock.at(3) - e.booking_clock.at(1) == timedelta(hours=18)


def test_report_UTC_timestamp_and_native_period_are_both_explicit():
    e, rs = fixture()
    strand(e)
    with pytest.raises(ValueError, match="UTC instant"):
        e.publish_regulator_report_at(hour(13), rs[1])
    with pytest.raises(ValueError, match="timezone-aware"):
        e.publish_regulator_report_at(datetime(2026, 5, 1, 12), rs[1])
    assert not e._regulator_reports
    publish(e, rs)
    assert e._regulator_reports[1] is rs[1]
    with pytest.raises(ValueError, match="immutable"):
        e.publish_regulator_report_at(hour(12), replace(rs[1], declared_emergency=True))


def test_missing_report_prevents_trigger_enabled_sale_without_private_fallback():
    e, _rs = fixture()
    strand(e)
    before = [s.posted_prices.copy() for s in e.world.carriers]
    with pytest.raises(ValueError, match="fresh public"):
        e.step(actions(e), _requests=[])
    assert (
        all(
            np.array_equal(s.posted_prices, p) for s, p in zip(e.world.carriers, before)
        )
        and not e.rebooking_history
    )


def test_unrepresented_public_state_is_not_replaced_by_private_inventory():
    e, rs = fixture()
    strand(e)
    bad = replace(rs[1], remaining_seats=((e.dated_reserve.affected_legs[0], 8),))
    e.publish_regulator_report_at(hour(12), bad)
    before = [s.posted_prices.copy() for s in e.world.carriers]
    with pytest.raises(KeyError, match="absent"):
        e.step(actions(e), _requests=[])
    assert (
        all(
            np.array_equal(s.posted_prices, p) for s, p in zip(e.world.carriers, before)
        )
        and not e.rebooking_history
    )


def test_benchmark_receives_only_supplied_public_report(monkeypatch):
    e, rs = fixture()
    b = strand(e)
    e.present_cancelled_itinerary(b)
    old = PublicMechanicalTable.fares
    seen = []

    def observe(self, report):
        assert type(report) is PublicTriggerReport and report is rs[1]
        seen.append(report)
        return old(self, report)

    monkeypatch.setattr(PublicMechanicalTable, "fares", observe)
    empty(e, rs)
    assert seen and all(x is rs[1] for x in seen)
    # Deliberately distinct unit public report and private remaining stock;
    # no claim of truthful upstream measurement is made by this physical adapter.
    assert dict(rs[1].remaining_seats)[e.dated_reserve.affected_legs[0]] == 9


def test_no_capacity_loss_does_not_fire_despite_booking_depletion():
    e, rs = fixture(kappa=".3", loss=False)
    b = strand(e)
    e.present_cancelled_itinerary(b)
    empty(e, rs)
    assert not e.trigger.active_until and e.rebooking_history[-1]["fare"] == 99
    cash(e)


def test_declaration_can_trigger_without_capacity_loss():
    e, rs = fixture(kappa=".9", loss=False)
    b = strand(e)
    e.present_cancelled_itinerary(b)
    rs = list(rs)
    rs[1] = replace(rs[1], declared_emergency=True)
    empty(e, rs)
    assert e.rebooking_history[-1]["fare"] == 80 and e.trigger.active_until
    cash(e)


def test_cheaper_static_cap_still_dominates_public_trigger_and_rescue():
    raw, _ = fixture()
    cap = CapPolicy(
        (CapWindow(1, 5, tuple((p.id, 50) for p in raw.compiled.spec.products), 1),)
    )
    e, rs = fixture(cap_policy=cap)
    b = strand(e)
    e.present_cancelled_itinerary(b)
    empty(e, rs)
    assert e.rebooking_history[-1]["fare"] == 50 and max(e.trace()[-1]["prices"]) == 50
    cash(e)


def test_replacement_refund_is_its_trigger_capped_fare_without_lineage_refill():
    e, rs = fixture(extra_exit=True)
    b = strand(e)
    e.present_cancelled_itinerary(b)
    empty(e, rs)
    assert e.world.carriers[0].revenue == e.world.carriers[0].refunds == 80
    assert e._passengers[b.record_id].active_record is None and len(e._passengers) == 1
    assert all(x["initial"] == 2 and x["used"] == 1 for x in e.reserve_balance())
    row = e.timed_rebooking_metrics()["rows"][0]
    assert (
        row["first_rebooked_within_hours"] == 1
        and row["active_replacement_booked_within_hours"] == 0
    )
    cash(e)


def test_branch_and_restore_own_trigger_and_both_dated_ledgers():
    e, rs = fixture()
    b = strand(e)
    e.present_cancelled_itinerary(b)
    empty(e, rs)
    snapshot = e.snapshot()
    before = pickle.dumps(snapshot)
    other = e.branch()
    assert type(other) is module.TriggeredDatedMarketEngine
    assert (
        other.trigger.history is not e.trigger.history
        and other._regulator_reports is not e._regulator_reports
    )
    assert (
        other._reserve_ledger is not e._reserve_ledger
        and other._dated_released is not e._dated_released
    )
    empty(other, rs)
    assert pickle.dumps(e.snapshot()) == before
    empty(e, rs)
    assert pickle.dumps(e.snapshot()) == pickle.dumps(other.snapshot())
    e.restore(snapshot)
    assert pickle.dumps(e.snapshot()) == before


@pytest.mark.parametrize("change", ["rule", "benchmark"])
def test_restore_refuses_different_immutable_trigger_contract(change):
    e, _rs = fixture()
    other, _ = fixture(tau=".1" if change == "rule" else 0)
    if change == "benchmark":
        other.trigger.benchmark = replace(
            other.trigger.benchmark, pre_shock_demand_sha256="c" * 64
        )
    with pytest.raises(ValueError, match="rule/benchmark"):
        other.restore(e.snapshot())


def test_restore_refuses_different_immutable_public_clock():
    e, _rs = fixture()
    other = e.branch()
    other.booking_clock = UtcBookingClock(
        tuple(hour(x) for x in (0, 12, 24, 30, 37, 41))
    )
    with pytest.raises(ValueError, match="dates/policy|clock/policy"):
        other.restore(e.snapshot())


def test_reset_clears_reports_trigger_and_both_lineage_quota_ledgers():
    e, rs = fixture()
    b = strand(e)
    e.present_cancelled_itinerary(b)
    empty(e, rs)
    clock = e.booking_clock
    bench = e.trigger.benchmark
    e.reset(tape=e.tape)
    assert (
        not e.trigger.active_until
        and not e.trigger.history
        and not e._regulator_reports
        and not e.rebooking_history
        and not e._passengers
        and not e.reserve_events
        and not e._reserve_ledger
    )
    assert e.booking_clock is clock and e.trigger.benchmark is bench


def test_single_controller_forward_and_callback_keep_native_observations(monkeypatch):
    e, rs = fixture()
    b = strand(e)
    e.present_cancelled_itinerary(b)
    ctrl = FixedCoefficientController((0, 0, 0))
    e.controllers[0] = ctrl
    act, done = ctrl.act, ctrl.observe_outcome
    seen = []
    callbacks = []

    def spy(obs):
        assert obs.privileged is None
        seen.append(obs)
        return act(obs)

    def observe(x):
        callbacks.append(x)
        return done(x)

    monkeypatch.setattr(ctrl, "act", spy)
    monkeypatch.setattr(ctrl, "observe_outcome", observe)
    publish(e, rs)
    e.step({"B": actions(e)["B"]}, _requests=[])
    assert len(seen) == len(callbacks) == 1
    cash(e)


def test_infinite_tau_matches_dated_pair_on_same_physical_tape():
    e, _rs = fixture(tau="Infinity")
    b = strand(e)
    e.present_cancelled_itinerary(b)
    reference = DatedRescueReserveMarketEngine(
        e.compiled,
        booking_clock=e.booking_clock,
        leg_departures=e.leg_departures,
        dated_rebooking=e.dated_rebooking,
        dated_menu=e.dated_menu,
        dated_reserve=e.dated_reserve,
        exits=e.exits,
    )
    reference.reset(tape=e.tape)
    c = reference.compiled
    j = max(c.products_of[1], key=lambda j: len(c.spec.products[j].legs))
    reference.step(
        actions(reference),
        _requests=[Request(0, int(c.product_market[j]), 0, 0, 0, 0)],
        _choices=[j],
    )
    reference.reserve_balance()
    reference.present_cancelled_itinerary(b)
    while not e.terminal():
        e.step(actions(e), _requests=[])
        reference.step(actions(reference), _requests=[])
    assert pickle.dumps(
        (e.world, e.trace(), e.metrics(), e.reserve_balance())
    ) == pickle.dumps(
        (
            reference.world,
            reference.trace(),
            reference.metrics(),
            reference.reserve_balance(),
        )
    )
    assert not e.trigger.history and not e.trigger.active_until


def test_no_policy_matches_native_arrivals_and_requires_clock_for_UTC_reports():
    c = fixture()[0].compiled
    e = module.TriggeredDatedMarketEngine(c)
    native = MarketEngine(c)
    tape = ExogenousTape.generate(
        c, 25, registry=StreamRegistry(24975, "unit/V9_triggered_dated_none")
    )
    e.reset(tape=tape)
    native.reset(tape=tape)
    for _ in range(5):
        e.step(actions(e))
        native.step(actions(native))
    assert pickle.dumps((e.world, e.trace(), e.metrics())) == pickle.dumps(
        (native.world, native.trace(), native.metrics())
    )
    _, rs = fixture()
    with pytest.raises(ValueError, match="UTC clock"):
        e.publish_regulator_report_at(hour(0), rs[0])


def test_disabled_trigger_keeps_dated_sales_and_branch_type():
    e, _rs = fixture(trigger=False)
    b = strand(e)
    e.present_cancelled_itinerary(b)
    e.step(actions(e), _requests=[])
    assert e.trigger is None and e.rebooking_history[-1]["fare"] == 99
    other = e.branch()
    assert type(other) is module.TriggeredDatedMarketEngine and other.trigger is None
    assert pickle.dumps(other.snapshot()) == pickle.dumps(e.snapshot())
    cash(e)


def test_native_trigger_outer_stop_releases_only_price_ceiling():
    e, rs = fixture()
    strand(e)
    for _ in range(4):
        empty(e, rs)
        if e.world.t <= 4:
            assert max(e.trace()[-1]["prices"]) == 80
    assert e.world.t == 5 and max(e.trace()[-1]["prices"]) == 400
    assert all(x["held"] == 0 for x in e.reserve_balance())
    cash(e)
