"""A.14/B0/V9: rescue eligibility, menus, windows, privacy and native money."""

import importlib.util
import os
import pickle
from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.regulator import CapPolicy, CapWindow
from airline_market.sim.arrivals import Request
from airline_market.sim.disruption import RivalExit
from airline_market.sim.rebooking import RebookingMarketEngine, RebookingPolicy
from airline_market.sim.tape import ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.model import CapacityEventSpec, CapacityProcessSpec

if os.environ.get("RESCUE_SOURCE"):
    spec = importlib.util.spec_from_file_location(
        "staged_rescue", os.environ["RESCUE_SOURCE"]
    )
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
else:
    from airline_market.policy import rescue as module


@pytest.mark.parametrize(
    "distance,expected", [(999.99, 199), (1000, 199), (1000.01, 299)]
)
def test_two_distance_tiers_use_explicit_od_distance_and_inclusive_boundary(
    distance, expected
):
    menu = module.RescueFareMenu(
        1, 3, "two_tier", ((1000, 199), (None, 299)), (("route", distance),)
    )
    assert menu.fare_for("route", 1) == Decimal(expected)
    assert (
        menu.fare_for("route", 0)
        is menu.fare_for("route", 3)
        is menu.fare_for("unknown", 1)
        is None
    )


def test_counter_menu_has_three_tiers_and_same_route_is_exact():
    menu = module.RescueFareMenu(
        0,
        4,
        "counter_three_tier",
        ((500, 200), (1000, 300), (None, 400)),
        (("short", 500), ("medium", 700), ("long", 1001)),
    )
    assert menu.counter_only and [
        menu.fare_for(k, 2) for k in ["short", "medium", "long"]
    ] == list(map(Decimal, [200, 300, 400]))
    route = module.RescueFareMenu(0, 1, "same_route", route_fares=(("O-D", 99),))
    assert route.fare_for("O-D", 0) == 99 and route.fare_for("D-O", 0) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"design": "unknown"},
        {"design": "same_route", "route_fares": (("m", 99.001),)},
        {"design": "same_route", "route_fares": (("m", float("nan")),)},
        {"design": "same_route", "route_fares": (("m", 99), ("m", 100))},
        {
            "design": "two_tier",
            "distance_tiers": ((None, 199),),
            "market_distances_km": (("m", 1),),
        },
        {
            "design": "counter_three_tier",
            "distance_tiers": ((500, 200), (500, 300), (None, 400)),
            "market_distances_km": (("m", 1),),
        },
    ],
)
def test_invalid_menu_has_no_silent_rounding_or_missing_distance_imputation(kwargs):
    with pytest.raises(ValueError):
        module.RescueFareMenu(0, 4, **kwargs)


def model(twice=False):
    s = screen_spec(
        {
            "T": 5,
            "actions": 2,
            "n_spokes": 1,
            "capacity": 5,
            "capacity_support": [3, 4],
            "price_scale": 1.311,
            "outside_utility_shift": 0.173,
            "arrival_probability": 0.741,
        }
    )
    if twice:
        events = tuple(
            CapacityEventSpec(
                trigger={"hazard": {"probability": 0.5}},
                effects=tuple((l.id, delta) for l in s.legs if l.carrier == "A"),
                once=True,
            )
            for delta in [-20, 5]
        )
        s = replace(s, capacity_process=CapacityProcessSpec(events=events))
    return compile_market(s)


def shared_tape(c, twice=False):
    tape = ExogenousTape.generate(
        c, 19, registry=StreamRegistry(21873, "unit/V9_rescue")
    )
    if twice:
        u = np.ones_like(tape.capacity_uniforms)
        u[1, :] = 0
        u.flags.writeable = False
        tape = replace(tape, capacity_uniforms=u)
    return tape


def actions(c, a=400):
    return {
        carrier.code: PriceAction({j: a if i == 0 else 300 for j in c.products_of[i]})
        for i, carrier in enumerate(c.spec.carriers)
    }


def connect(c):
    return max(c.products_of[1], key=lambda j: len(c.spec.products[j].legs))


def route_id(c):
    return c.spec.markets[int(c.product_market[connect(c)])].id


def route_menu(c, start=1, stop=5, fare=99):
    return module.RescueFareMenu(
        start, stop, "same_route", route_fares=((route_id(c), fare),)
    )


def engine(c, menu=None, **kwargs):
    return module.RescueMarketEngine(
        c,
        exits=(RivalExit("B", 1),),
        rebooking=RebookingPolicy(3, 2, 100, True),
        rescue_menu=menu,
        **kwargs,
    )


def original_sale(e):
    c = e.compiled
    j = connect(c)
    e.step(
        actions(c),
        _requests=[Request(0, int(c.product_market[j]), 0, 0, 0, 0)],
        _choices=[j],
    )
    return e.stranded[0].booking


def empty(e, a=400):
    return e.step(actions(e.compiled, a), _requests=[], _choices=[])


def conserved(e):
    r = e.conservation()
    assert r["original_booked_passengers"] == r["active_plus_pending_plus_unserved"]
    assert r["active_passengers"] == r["unique_active_passengers"]
    for s in e.world.carriers:
        assert s.cash == pytest.approx(
            s.revenue - s.refunds - s.penalties, rel=0, abs=1e-10
        )
    return r


def test_no_rescue_policy_preserves_frozen_rebooking_world_trace_and_metrics():
    c = model()
    base = RebookingMarketEngine(
        c, exits=(RivalExit("B", 1),), rebooking=RebookingPolicy(3, 2, 100, True)
    )
    overlay = engine(c)
    for e in [base, overlay]:
        e.reset(tape=shared_tape(c))
        original_sale(e)
        empty(e)
    assert pickle.dumps(
        (base.world, base.trace(), base.metrics()), protocol=4
    ) == pickle.dumps((overlay.world, overlay.trace(), overlay.metrics()), protocol=4)


def test_no_shock_active_menu_is_byte_identical_and_does_not_draw_rescue_randomness(
    monkeypatch,
):
    c = model()
    base = RebookingMarketEngine(c, rebooking=RebookingPolicy(3, 2, 100, True))
    e = module.RescueMarketEngine(
        c, rebooking=base.rebooking, rescue_menu=route_menu(c)
    )

    def forbidden(*args):
        raise AssertionError("no stranded passenger may draw re-entry marks")

    monkeypatch.setattr(e, "_marks", forbidden)
    for item in [base, e]:
        item.reset(tape=shared_tape(c))
        while not item.terminal():
            item.step(actions(c))
    assert pickle.dumps(
        (base.world, base.trace(), base.metrics()), protocol=4
    ) == pickle.dumps((e.world, e.trace(), e.metrics()), protocol=4)


def test_real_cancelled_ticket_required_and_ordinary_passenger_never_gets_rescue_fare():
    c = model()
    e = engine(c, route_menu(c))
    e.reset(tape=shared_tape(c))
    b = original_sale(e)
    with pytest.raises(ValueError, match="cancellation"):
        e.present_cancelled_itinerary(replace(b, fare=b.fare + 1))
    with pytest.raises(TypeError, match="native"):
        e.present_cancelled_itinerary("forged external claim")
    e.present_cancelled_itinerary(b)
    j = next(
        j
        for j in c.market_products[int(c.product_market[b.product])]
        if c.product_carrier[j] == 0
    )
    result = e.step(
        actions(c),
        _requests=[Request(0, int(c.product_market[j]), 0, 0, 0, 0)],
        _choices=[j],
    )
    assert [p.fare for p in result.purchases] == [99, 400]
    assert e.rebooking_history[-1]["rescue_discount_used"] is True
    assert all(e.world.carriers[0].posted_prices == 400)
    assert conserved(e)["original_booked_passengers"] == 2


@pytest.mark.parametrize("presented,expected", [(False, 400), (True, 99)])
def test_only_presented_genuine_cancellation_gets_fixed_route_option(
    presented, expected
):
    c = model()
    e = engine(c, route_menu(c))
    e.reset(tape=shared_tape(c))
    b = original_sale(e)
    if presented:
        e.present_cancelled_itinerary(b)
    assert empty(e).purchases[0].fare == expected
    assert e.rebooking_history[-1]["rescue_discount_used"] is presented
    conserved(e)


@pytest.mark.parametrize("counter,expected", [(False, 400), (True, 200)])
def test_three_tier_menu_requires_same_period_airport_counter_evidence(
    counter, expected
):
    c = model()
    menu = module.RescueFareMenu(
        1,
        5,
        "counter_three_tier",
        ((500, 200), (1000, 300), (None, 400)),
        ((route_id(c), 500),),
    )
    e = engine(c, menu)
    e.reset(tape=shared_tape(c))
    b = original_sale(e)
    e.present_cancelled_itinerary(b, airport_counter=counter)
    assert empty(e).purchases[0].fare == expected
    conserved(e)


@pytest.mark.parametrize("start,stop,expected", [(0, 1, 400), (1, 2, 99), (2, 5, 400)])
def test_sale_enforces_start_inclusive_stop_exclusive_menu_window(
    start, stop, expected
):
    c = model()
    e = engine(c, route_menu(c, start, stop))
    e.reset(tape=shared_tape(c))
    b = original_sale(e)
    e.present_cancelled_itinerary(b)
    assert empty(e).purchases[0].fare == expected
    conserved(e)


def test_cheaper_ordinary_and_active_cap_remain_available_without_duplicate_alternatives():
    c = model()
    for ordinary, rescue, expected, discount in [
        (50, 99, 50, False),
        (400, 20, 20, True),
        (400, 99, 26, False),
    ]:
        refs = tuple((c.spec.products[j].id, 20) for j in c.products_of[0])
        cap = CapPolicy((CapWindow(0, 5, refs),)) if ordinary == 400 else CapPolicy()
        e = engine(c, route_menu(c, fare=rescue), cap_policy=cap)
        e.reset(tape=shared_tape(c))
        b = original_sale(e)
        e.present_cancelled_itinerary(b)
        assert empty(e, ordinary).purchases[0].fare == expected
        assert e.rebooking_history[-1]["rescue_discount_used"] is discount
        assert (
            len(e.rebooking_history) == 1
            and conserved(e)["original_booked_passengers"] == 1
        )


@pytest.mark.parametrize("fare", [99, 99.99])
def test_repeated_cancel_refunds_actual_rescue_fare_without_duplicating_passenger(fare):
    c = model(True)
    e = engine(c, route_menu(c, fare=fare))
    e.reset(tape=shared_tape(c, True))
    b = original_sale(e)
    e.present_cancelled_itinerary(b)
    assert empty(e).purchases[0].fare == fare
    assert e.world.carriers[0].revenue == e.world.carriers[0].refunds == fare
    assert conserved(e)["pending_stranded"] == 1
    assert empty(e).purchases[0].fare == fare
    r = conserved(e)
    assert (
        r["original_booked_passengers"] == r["active_passengers"] == 1
        and r["replacement_tickets_ever"] == 2
    )


def test_counter_visit_is_not_carried_forward_after_a_replacement_cancellation():
    c = model(True)
    menu = module.RescueFareMenu(
        1,
        5,
        "counter_three_tier",
        ((500, 200), (1000, 300), (None, 400)),
        ((route_id(c), 500),),
    )
    e = engine(c, menu)
    e.reset(tape=shared_tape(c, True))
    b = original_sale(e)
    e.present_cancelled_itinerary(b, airport_counter=True)
    assert empty(e).purchases[0].fare == 200
    assert e.world.carriers[0].refunds == 200
    assert empty(e).purchases[0].fare == 400
    assert e.rebooking_history[-1]["airport_counter_proved_this_period"] is False
    conserved(e)


def test_snapshot_branch_restore_and_reset_keep_credentials_independent():
    c = model()
    e = engine(c, route_menu(c))
    e.reset(tape=shared_tape(c))
    b = original_sale(e)
    without = e.snapshot()
    e.present_cancelled_itinerary(b)
    branch = e.branch()
    assert branch.tape is e.tape and branch._presented is not e._presented
    assert empty(branch).purchases[0].fare == 99
    e.restore(without)
    assert not e.presented_itineraries
    assert empty(e).purchases[0].fare == 400
    assert len(branch.presented_itineraries) == 1
    e.reset(tape=shared_tape(c))
    assert not e.presented_itineraries
    with pytest.raises(ValueError, match="cancellation"):
        e.present_cancelled_itinerary(b)


def test_actor_gets_one_forward_and_native_callback_with_actual_private_fare():
    class Spy:
        def reset(self, context, rng):
            self.products = context.compiled.products_of[context.carrier]
            self.calls = 0
            self.outcomes = []

        def act(self, obs):
            assert obs.privileged is None and not hasattr(obs, "presented_itineraries")
            self.calls += 1
            return PriceAction({j: 400 for j in self.products})

        def observe_outcome(self, o):
            self.outcomes.append(o)

        def memory(self):
            return ()

    c = model()
    spy = Spy()
    e = engine(c, route_menu(c), controllers={"A": spy})
    e.reset(tape=shared_tape(c))
    j = connect(c)
    b_only = {"B": actions(c)["B"]}
    e.step(
        b_only,
        _requests=[Request(0, int(c.product_market[j]), 0, 0, 0, 0)],
        _choices=[j],
    )
    e.present_cancelled_itinerary(e.stranded[0].booking)
    e.step(b_only, _requests=[], _choices=[])
    assert spy.calls == len(spy.outcomes) == 2 and spy.outcomes[-1]["cash_change"] == 99
    assert spy.outcomes[-1]["purchases"][0].fare == 99
    conserved(e)
