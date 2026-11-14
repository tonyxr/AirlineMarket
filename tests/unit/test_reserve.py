"""A.14/B0/V9: native reserve admission, cash, release and branch ownership."""

import importlib.util
import os
import pickle
import sys
from decimal import Decimal

import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.regulator import CapPolicy, CapWindow
from airline_market.policy.rescue import RescueFareMenu, RescueMarketEngine
from airline_market.sim.arrivals import Request
from airline_market.sim.disruption import RivalExit
from airline_market.sim.rebooking import RebookingPolicy
from airline_market.sim.tape import ExogenousTape
from airline_market.spec import compile_market

if os.environ.get("RESERVE_SOURCE"):
    spec = importlib.util.spec_from_file_location(
        "staged_reserve", os.environ["RESERVE_SOURCE"]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
else:
    from airline_market.policy import reserve as module


def fixture(k=20, share=".10", release=3, **kwargs):
    c = compile_market(
        screen_spec(
            {
                "T": 5,
                "actions": 2,
                "n_spokes": 1,
                "capacity": k,
                "capacity_support": [3, 4],
                "arrival_probability": 0.741,
            }
        )
    )
    menu = RescueFareMenu(
        1, 5, "same_route", route_fares=tuple((m.id, 99) for m in c.spec.markets)
    )
    plan = module.SeatReserve(
        1, release, Decimal(share), tuple(l.id for l in c.spec.legs if l.carrier == "A")
    )
    e = module.ReserveMarketEngine(
        c,
        seat_reserve=plan,
        rescue_menu=menu,
        rebooking=RebookingPolicy(4, 2, 100, True),
        exits=(RivalExit("B", 1),),
        **kwargs,
    )
    e.reset(
        tape=ExogenousTape.generate(
            c, 23, registry=StreamRegistry(21874, "unit/V9_reserve")
        )
    )
    return e


def actions(e):
    c = e.compiled
    return {
        carrier.code: PriceAction({j: 400 if i == 0 else 300 for j in c.products_of[i]})
        for i, carrier in enumerate(c.spec.carriers)
    }


def strand(e, n=1):
    c = e.compiled
    j = max(c.products_of[1], key=lambda j: len(c.spec.products[j].legs))
    e.step(
        actions(e),
        _requests=[Request(i, int(c.product_market[j]), 0, 0, 0, 0) for i in range(n)],
        _choices=[j] * n,
    )
    return tuple(x.booking for x in e.stranded)


def empty(e):
    return e.step(actions(e), _requests=[])


def cash(e):
    for s in e.world.carriers:
        assert s.cash == pytest.approx(
            s.revenue - s.refunds - s.penalties, rel=0, abs=1e-10
        )
    p = e.conservation()
    assert p["original_booked_passengers"] == p["active_plus_pending_plus_unserved"]


@pytest.mark.parametrize(
    "share,k,expected",
    [(".02", 19, 0), (".02", 51, 1), (".05", 19, 0), (".05", 20, 1), (".10", 20, 2)],
)
def test_reveal_floor_uses_remaining_not_nominal_capacity(share, k, expected):
    e = fixture(k, share)
    strand(e)
    e.begin_period()
    rows = e.reserve_balance()
    assert rows and all(r["initial"] == expected for r in rows)
    for r in rows:
        assert r["ordinary_available"] + r["held"] == r["physical_remaining"] == k
    assert all(
        ev["rounding"] == "floor" for ev in e.reserve_events if ev["kind"] == "hold"
    )


@pytest.mark.parametrize(
    "args",
    [
        (1, 1, ".1", ("a",)),
        (-1, 2, ".1", ("a",)),
        (1, 3, ".03", ("a",)),
        (1, 3, "NaN", ("a",)),
        (1, 3, ".1", ("a", "a")),
    ],
)
def test_invalid_policy_rejected(args):
    with pytest.raises(ValueError):
        module.SeatReserve(*args)


def test_mandate_requires_menu_coverage_and_explicit_clock():
    e = fixture()
    c = e.compiled
    with pytest.raises(ValueError, match="menu"):
        module.ReserveMarketEngine(
            c, seat_reserve=e.seat_reserve, rebooking=e.rebooking
        )
    with pytest.raises(ValueError, match="every affected"):
        module.ReserveMarketEngine(
            c,
            seat_reserve=e.seat_reserve,
            rebooking=e.rebooking,
            rescue_menu=RescueFareMenu(
                1, 2, "same_route", route_fares=((c.spec.markets[0].id, 99),)
            ),
        )
    with pytest.raises(ValueError, match="unknown affected"):
        module.ReserveMarketEngine(
            c,
            seat_reserve=module.SeatReserve(1, 3, ".1", ("unknown",)),
            rebooking=e.rebooking,
            rescue_menu=e.rescue_menu,
        )


def test_eligible_connection_uses_native_booking_one_seat_on_each_held_leg():
    e = fixture()
    (b,) = strand(e)
    e.present_cancelled_itinerary(b)
    before = e.world.carriers[0].cash
    empty(e)
    record = e.rebooking_history[-1]
    assert (
        record["reserve_eligible"]
        and record["fare"] == 99
        and record["stage"] == "interline"
    )
    assert len(record["held_seats_used"]) == 2 and set(
        record["held_seats_used"].values()
    ) == {1}
    assert e.world.carriers[0].cash - before == 99
    rows = {r["leg"]: r for r in e.reserve_balance()}
    for leg in record["held_seats_used"]:
        assert (
            rows[leg]["used"] == 1
            and rows[leg]["held"] == 1
            and rows[leg]["physical_remaining"] == 19
        )
    assert e._reserve_eligible is False
    cash(e)


def test_unproved_passenger_keeps_regular_price_and_cannot_access_held_only_stock():
    e = fixture()
    strand(e)
    e.begin_period()
    # Native setup contains ordinary tickets consuming only unheld capacity.
    c = e.compiled
    for i, cs in enumerate(e.world.carriers):
        if i == 0:
            cs.capacity[:] = 2
    e._sync_reserve()
    assert not e._sellable_global()[list(c.products_of[0])].any()
    empty(e)
    assert e.rebooking_history[-1]["chosen"] is None
    assert all(r["used"] == 0 and r["held"] == 2 for r in e.reserve_balance())
    cash(e)


def test_proof_cannot_be_forged():
    from dataclasses import replace

    e = fixture()
    (b,) = strand(e)
    with pytest.raises(ValueError):
        e.present_cancelled_itinerary(replace(b, fare=b.fare + 1))


def test_native_ordinary_choice_cannot_consume_held_stock_and_release_restores_sales():
    e = fixture()
    strand(e)
    e.begin_period()
    c = e.compiled
    for cs in e.world.carriers[:1]:
        cs.capacity[:] = 2
    e._sync_reserve()
    j = max(c.products_of[0], key=lambda j: len(c.spec.products[j].legs))
    m = int(c.product_market[j])
    requests = [Request(0, m, 0, 0, 0, 0)]
    e.step(actions(e), _requests=requests)
    assert (
        e.world.request_log[-1]["chosen"] is None and not e.world.carriers[0].bookings
    )
    e.step(actions(e), _requests=[])
    assert e.world.t == 3 and all(
        r["released"] == 2 and r["held"] == 0 for r in e.reserve_balance()
    )
    assert e._sellable_global()[j]
    e.step(actions(e), _requests=requests, _choices=[j])
    # The original pending passenger may shop after release, before ordinary demand.
    assert len(e.world.carriers[0].bookings) == 2
    cash(e)


def test_unused_release_is_once_only_and_restores_exact_free_inventory():
    e = fixture()
    strand(e)
    e.begin_period()
    empty(e)
    empty(e)
    rows = e.reserve_balance()
    assert all(
        r["released"] == 2
        and r["held"] == 0
        and r["ordinary_available"] == r["physical_remaining"]
        for r in rows
    )
    events = tuple(e.reserve_events)
    e.begin_period()
    e.reserve_balance()
    assert tuple(e.reserve_events) == events


def test_capacity_loss_is_explicit_writeoff_without_negative_free_or_replenishment():
    e = fixture()
    strand(e)
    e.begin_period()
    e.world.carriers[0].capacity[:] = 1
    rows = e.reserve_balance()
    assert all(
        r["capacity_lost"] == 1 and r["held"] == 1 and r["ordinary_available"] == 0
        for r in rows
    )
    e.world.carriers[0].capacity[:] = 20
    assert all(r["held"] == 1 and r["initial"] == 2 for r in e.reserve_balance())


def test_snapshot_branch_reset_preserve_independent_reserves_proof_and_native_state():
    e = fixture()
    (b,) = strand(e)
    e.present_cancelled_itinerary(b)
    e.begin_period()
    snapshot = e.snapshot()
    branch = e.branch()
    assert branch.seat_reserve is e.seat_reserve and branch.rescue_menu is e.rescue_menu
    assert branch._reserve_ledger is not e._reserve_ledger
    empty(branch)
    assert branch.reserve_balance() != e.reserve_balance()
    e.restore(snapshot)
    empty(e)
    assert pickle.dumps(
        (e.world, e.reserve_events, e.rebooking_history)
    ) == pickle.dumps((branch.world, branch.reserve_events, branch.rebooking_history))
    e.reset(tape=e.tape)
    assert (
        not e.reserve_events and not e.reserve_balance() and not e.presented_itineraries
    )


def test_inactive_overlay_is_byte_identical_to_rescue_engine():
    base = fixture()
    c = base.compiled
    e = module.ReserveMarketEngine(
        c, rebooking=base.rebooking, rescue_menu=base.rescue_menu, exits=base.exits
    )
    r = RescueMarketEngine(
        c, rebooking=base.rebooking, rescue_menu=base.rescue_menu, exits=base.exits
    )
    for x in [e, r]:
        x.reset(tape=base.tape)
        strand(x)
        empty(x)
    assert pickle.dumps((e.world, e.trace(), e.metrics())) == pickle.dumps(
        (r.world, r.trace(), r.metrics())
    )


def test_effective_reserved_fare_respects_active_cap():
    raw = fixture()
    p = CapPolicy(
        (
            CapWindow(
                1, 3, tuple((x.id, 50) for x in raw.compiled.spec.products), "1.30"
            ),
        )
    )
    e = fixture(cap_policy=p)
    (b,) = strand(e)
    e.present_cancelled_itinerary(b)
    empty(e)
    assert (
        e.rebooking_history[-1]["fare"] == 65
        and e.rebooking_history[-1]["held_seats_used"]
    )
    cash(e)


def test_exception_during_private_choice_restores_ordinary_admission(monkeypatch):
    e = fixture()
    (b,) = strand(e)
    e.present_cancelled_itinerary(b)

    def fail(*_):
        raise RuntimeError("injected choice failure")

    monkeypatch.setattr(module, "sampled_choice", fail)
    with pytest.raises(RuntimeError, match="injected"):
        empty(e)
    assert not e._reserve_eligible


def test_forced_choice_test_hook_cannot_bypass_hold():
    e = fixture()
    strand(e)
    with pytest.raises(ValueError, match="bypasses reserve admission"):
        e.step(actions(e), _requests=[], _choices=[])


def test_several_stranded_passengers_use_each_held_seat_once_then_regular_fare():
    e = fixture()
    bookings = strand(e, 3)
    for b in bookings:
        e.present_cancelled_itinerary(b)
    empty(e)
    assert [r["fare"] for r in e.rebooking_history] == [99, 99, 400]
    assert [bool(r["held_seats_used"]) for r in e.rebooking_history] == [
        True,
        True,
        False,
    ]
    assert all(
        r["used"] == r["initial"] == 2 and r["held"] == 0 for r in e.reserve_balance()
    )
    cash(e)


def test_counter_only_fare_requires_same_period_visit_for_reserve_access():
    e = fixture()
    c = e.compiled
    e.rescue_menu = RescueFareMenu(
        1,
        5,
        "counter_three_tier",
        ((500, 99), (1000, 199), (None, 299)),
        tuple((m.id, 400) for m in c.spec.markets),
    )
    (b,) = strand(e)
    e.present_cancelled_itinerary(b)
    e.begin_period()
    e.world.carriers[0].capacity[:] = 2
    empty(e)
    assert e.rebooking_history[-1]["chosen"] is None
    e.present_cancelled_itinerary(b, airport_counter=True)
    empty(e)
    assert (
        e.rebooking_history[-1]["fare"] == 99
        and e.rebooking_history[-1]["held_seats_used"]
    )
    cash(e)


def test_reveal_subtracts_existing_native_itinerary_incidence():
    e = fixture()
    c = e.compiled
    a = max(c.products_of[0], key=lambda j: len(c.spec.products[j].legs))
    b = max(c.products_of[1], key=lambda j: len(c.spec.products[j].legs))
    e.step(
        actions(e),
        _requests=[
            Request(k, int(c.product_market[j]), 0, 0, 0, 0)
            for k, j in enumerate([a, b])
        ],
        _choices=[a, b],
    )
    e.begin_period()
    rows = {r["leg"]: r for r in e.reserve_balance()}
    for leg in c.spec.products[a].legs:
        assert rows[leg]["physical_remaining"] == 19 and rows[leg]["initial"] == 1
    cash(e)
