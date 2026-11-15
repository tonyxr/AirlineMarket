"""A.10/A.5/B0/V9: NEW SAME-call native ordinary information witnesses."""

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
from airline_market.policy.dated_reserve import DatedSeatReserve, UtcBookingClock
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.policy.rationed_native import WholePeriodPriority
from airline_market.policy.reserve import ReserveMarketEngine
from airline_market.sim.ordinary_menu import OrdinaryMenuMarketEngine
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

if os.environ.get("ORDINARY_INFORMATION_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_ordinary_information", os.environ["ORDINARY_INFORMATION_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.sim import ordinary_information as module


def fixture(*, count=3, shock=False, outside=False, reserve=False):
    # NEW unit source, not an imported old fixture or scientific bank.
    base = screen_spec(
        {
            "T": 5,
            "actions": 2,
            "n_spokes": 1,
            "capacity": 20,
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
    start = datetime(2026, 11, 20, tzinfo=timezone.utc)
    clock = UtcBookingClock(
        tuple(start + timedelta(hours=h) for h in (0, 12, 24, 48, 60, 72))
    )
    departures = tuple(
        (l.id, start + timedelta(hours=96 if l.carrier == "B" else 108))
        for l in c.spec.legs
    )
    scenario = compile_dated_demand(c, clock, departures)
    kwargs = {}
    if shock:
        scenario = replace(
            scenario,
            physical_shocks=(
                B1DatedShock("exit", "rival_exit", "B_OH0", 3, rival_carrier="B"),
            ),
        )
        kwargs["dated_rebooking"] = DatedRebookingPolicy(72, 20, 100, interline=True)
    if reserve:
        kwargs["dated_rebooking"] = DatedRebookingPolicy(72, 20, 100, interline=True)
        kwargs["dated_menu"] = DatedRescueMenu(
            UtcWindow(clock.at(0), clock.at(5)),
            "same_route",
            route_fares=tuple((m.id, 99) for m in c.spec.markets),
        )
        kwargs["dated_reserve"] = DatedSeatReserve(
            clock.at(0),
            ".1",
            tuple(l.id for l in c.spec.legs if l.carrier == "A"),
            departures,
        )
    e = module.OrdinaryInformationMarketEngine(scenario, **kwargs)
    parent = OrdinaryMenuMarketEngine(scenario, **kwargs)
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(75974, "unit/V9_ordinary_information")
    )
    marks = tuple(
        ArrivalMark((i + 1) / (count + 1), 0, 0, 0, 0 if i % 2 == 0 else 0.999)
        for i in range(count)
    )
    g = tape.gumbel.copy()
    g[:] = -1e9
    g[:, :, -1] = 1e9 if outside else 0
    if not outside:
        g[:, :, c.product_index["B_OH0"]] = 1e9
    g.setflags(write=False)
    tape = replace(
        tape, gumbel=g, arrival_marks=tuple(marks for _ in tape.arrival_marks)
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


def test_same_native_call_before_taste_and_full_inherited_identity(monkeypatch):
    e, parent = fixture(shock=True)
    import airline_market.sim.rebooking as core

    original_taste, original_available = (
        core.sampled_taste,
        ReserveMarketEngine._sellable_global,
    )
    active = {"engine": None}
    counts = {id(e): 0, id(parent): 0}
    seen = []

    def taste(*args, **kwargs):
        x = active["engine"]
        if (
            x is e
            and e._pending_menus is not None
            and len(e._pending_menus) > len(seen)
        ):
            assert len(e._pending_information) == len(e._pending_menus)
            key, index, view = e._pending_information[-1]
            source = e._pending_menus[-1]
            assert (key, index) == source[:2]
            assert tuple(v.fare for v in view.offers) == tuple(
                source[2][j] for j in source[4][:-1]
            )
            assert tuple(v.offered for v in view.offers) == tuple(
                source[3][j] for j in source[4][:-1]
            )
            seen.append(view)
        return original_taste(*args, **kwargs)

    def available(self):
        if id(self) in counts:
            counts[id(self)] += 1
        return original_available(self)

    monkeypatch.setattr(core, "sampled_taste", taste)
    monkeypatch.setattr(ReserveMarketEngine, "_sellable_global", available)
    rng = pickle.dumps(np.random.get_state())
    while not e.terminal():
        seen.clear()
        active["engine"] = e
        e.step(actions(e))
        assert len(seen) == len(e.information_periods()[-1].rows)
        active["engine"] = parent
        parent.step(actions(parent))
        assert counts[id(e)] == counts[id(parent)]
        assert pickle.dumps(e.world) == pickle.dumps(parent.world)
        assert pickle.dumps(e.trace()) == pickle.dumps(parent.trace())
        assert e.menu_periods() == parent.menu_periods()
        assert e.rebooking_history == parent.rebooking_history
        assert len(e.information_periods()[-1].rows) == len(
            e.ordinary_periods()[-1].rows
        )
    assert e.metrics() == parent.metrics()
    assert any(r["replacement_record"] is not None for r in e.rebooking_history)
    assert pickle.dumps(np.random.get_state()) == rng


def test_actual_depletion_and_all_unavailable_outside_rows_retained():
    e, parent = fixture()
    for x in (e, parent):
        x.world.carriers[1].capacity[:] = 1
        x.step(actions(x))
    rows = e.information_periods()[0].rows
    offers = [
        next(v for v in r.menu_view.offers if v.product_id == "B_OH0") for r in rows
    ]
    assert [v.offered for v in offers] == [True, False, False]
    assert len(rows) == 3 and [
        r.analyst_person.original_record is not None for r in rows
    ] == [True, False, False]
    assert len(rows[0].menu_view.offers) == len(e.compiled.market_products[0])
    assert pickle.dumps(e.world) == pickle.dumps(parent.world)


def test_view_excludes_analyst_identity_class_outcome_and_other_markets():
    e, _ = fixture()
    e.step(actions(e))
    rows = e.information_periods()[0].rows
    view = module.ordinary_information_view(rows[0])
    assert set(view) == {
        "market_id",
        "native_period",
        "compiled_opportunity_utc",
        "offers",
        "outside_available",
        "scope",
    }
    assert {v["product_id"] for v in view["offers"]} == {
        e.compiled.spec.products[j].id for j in e.compiled.market_products[0]
    }
    assert all(set(v) == {"product_id", "fare", "offered"} for v in view["offers"])
    changed = replace(
        rows[0],
        analyst_person=replace(
            rows[0].analyst_person, taste_class=999, native_log_index=999
        ),
    )
    assert module.ordinary_information_key(changed) == module.ordinary_information_key(
        rows[0]
    )
    assert len({r.analyst_person.key for r in rows}) == 3
    assert len({module.ordinary_information_key(r) for r in rows}) == 1


def test_different_hidden_stock_same_menu_view_key():
    e, _ = fixture(count=1)
    other = e.branch()
    other.world.carriers[1].capacity[:] -= 1
    assert pickle.dumps(e.world) != pickle.dumps(other.world)
    e.step(actions(e))
    other.step(actions(other))
    a, b = e.information_periods()[0].rows[0], other.information_periods()[0].rows[0]
    assert module.ordinary_information_view(a) == module.ordinary_information_view(b)
    assert module.ordinary_information_key(a) == module.ordinary_information_key(b)


@pytest.mark.parametrize("count,outside", [(0, False), (2, True)])
def test_empty_and_native_outside_support(count, outside):
    e, _ = fixture(count=count, outside=outside)
    e.step(actions(e))
    e.step(actions(e))
    assert len(e.information_periods()) == 2
    assert all(len(p.rows) == count for p in e.information_periods())
    assert all(
        r.analyst_person.original_record is None
        for p in e.information_periods()
        for r in p.rows
    )


def test_view_key_retains_exact_binary64_signed_zero_and_unknown_free_export():
    e, _ = fixture(count=1)
    e.step(actions(e))
    row = e.information_periods()[0].rows[0]
    first = row.menu_view.offers[0]
    a = replace(
        row, menu_view=replace(row.menu_view, offers=(replace(first, fare=0.0),))
    )
    b = replace(
        row, menu_view=replace(row.menu_view, offers=(replace(first, fare=-0.0),))
    )
    assert module.ordinary_information_key(a) != module.ordinary_information_key(b)
    for value in (float("inf"), float("nan"), -5.25):
        declared = replace(
            row, menu_view=replace(row.menu_view, offers=(replace(first, fare=value),))
        )
        json.dumps(module.ordinary_information_view(declared), allow_nan=False)
    # These synthetic view envelopes are not native-produced nonfinite fares.


def test_owned_record_and_independent_nested_snapshot_branch_reset():
    e, _ = fixture()
    e.step(actions(e))
    before = pickle.dumps(e.snapshot())
    saved = module.ordinary_information_record(e)
    json.dumps(saved, allow_nan=False)
    saved["menu_views_by_period"][0]["views"][0]["offers"] = ()
    saved["analyst_context"]["identity_by_period"] = ()
    assert pickle.dumps(e.snapshot()) == before
    assert module.ordinary_information_record(e)["menu_views_by_period"][0]["views"][0][
        "offers"
    ]
    branch = e.branch()
    branch.step(actions(branch))
    assert len(e.information_periods()) == 1 and len(branch.information_periods()) == 2
    e.reset(tape=e.tape)
    assert e.information_periods() == ()


@pytest.mark.parametrize("field", ["rule", "spec_hash"])
def test_snapshot_contract_refused_before_mutation(field):
    e, _ = fixture()
    before = pickle.dumps(e.snapshot())
    with pytest.raises(ValueError, match="contract"):
        e.restore(replace(e.snapshot(), **{field: "different"}))
    with pytest.raises(TypeError, match="combined"):
        e.restore(e.snapshot().native)
    assert pickle.dumps(e.snapshot()) == before


@pytest.mark.parametrize("hook", ["_requests", "_choices"])
def test_forced_choice_request_hooks_refused_before_mutation(hook):
    e, _ = fixture()
    before = pickle.dumps(e.snapshot())
    with pytest.raises(ValueError, match="genuine"):
        e.step(actions(e), **{hook: []})
    assert pickle.dumps(e.snapshot()) == before


@pytest.mark.parametrize("name", ["booking_clock", "shock_clock"])
def test_exposed_native_clock_mismatch_refused(name):
    e, _ = fixture()
    clock = e.demand_scenario.clock
    setattr(
        e,
        name,
        UtcBookingClock(tuple(t + timedelta(minutes=1) for t in clock.boundaries)),
    )
    before = pickle.dumps(e.snapshot())
    with pytest.raises(ValueError, match="clock"):
        e.step(actions(e))
    assert pickle.dumps(e.snapshot()) == before


def test_random_priority_refused():
    e, _ = fixture()
    priority = WholePeriodPriority("unit/V9_ordinary_information/refused")
    with pytest.raises(ValueError, match="chronological"):
        module.OrdinaryInformationMarketEngine(
            e.demand_scenario, ordinary_priority=priority
        )
    e.ordinary_priority = priority
    before = pickle.dumps(e.snapshot())
    with pytest.raises(ValueError, match="chronological"):
        e.step(actions(e))
    assert pickle.dumps(e.snapshot()) == before


def test_native_failure_keeps_pending_no_positive_query_or_continuation():
    e, _ = fixture()

    def fail(_):
        raise RuntimeError("NEW information callback failure")

    e.controllers[0].observe_outcome = fail
    with pytest.raises(RuntimeError, match="callback"):
        e.step(actions(e))
    assert e.world.t == 1 and e.snapshot().pending is not None
    with pytest.raises(ValueError, match="partial"):
        module.ordinary_information_record(e)
    with pytest.raises(ValueError, match="continuation"):
        e.step(actions(e))
    assert e.branch().snapshot().pending == e.snapshot().pending


def test_projection_failure_before_taste_and_prior_complete_receipt(monkeypatch):
    e, _ = fixture()
    e.step(actions(e))
    first = e.information_periods()[0]

    def fail(_):
        raise RuntimeError("NEW projection failure")

    monkeypatch.setattr(e, "_make_information", fail)
    with pytest.raises(RuntimeError, match="projection"):
        e.step(actions(e))
    snap = e.snapshot()
    assert snap.periods == (first,) and snap.pending == () and e.world.t == 1
    assert len(snap.native.pending) == 1  # exact SAME-call parent menu retained
    with pytest.raises(ValueError, match="partial"):
        e.information_periods()


def test_post_close_failure_keeps_native_mutations_and_prior_receipt(monkeypatch):
    e, _ = fixture()
    e.step(actions(e))
    first = e.information_periods()[0]
    original = OrdinaryMenuMarketEngine.menu_periods

    def wrong(self):
        p = original(self)
        return (*p[:-1], replace(p[-1], period=999))

    monkeypatch.setattr(OrdinaryMenuMarketEngine, "menu_periods", wrong)
    with pytest.raises(RuntimeError, match="menus and information"):
        e.step(actions(e))
    assert (
        e.world.t == 2
        and e.snapshot().periods == (first,)
        and e.snapshot().pending is not None
    )
    with pytest.raises(ValueError, match="partial"):
        e.information_periods()


def test_positive_held_seats_keep_actual_net_menu_projection():
    e, parent = fixture(reserve=True)
    e.step(actions(e))
    parent.step(actions(parent))
    assert any(r["kind"] == "hold" and r["seats"] > 0 for r in e.reserve_events)
    assert e.reserve_events == parent.reserve_events
    assert pickle.dumps(e.world) == pickle.dumps(parent.world)
    for row, source in zip(
        e.information_periods()[0].rows, parent.menu_periods()[0].rows
    ):
        assert tuple(o.offered for o in row.menu_view.offers) == tuple(
            source.net_sellable[j] for j in source.native_columns[:-1]
        )
        assert row.menu_view.scope == "current-market-menu-before-taste-utility-choice"


def test_unrevealed_future_program_never_enters_narrow_menu_key():
    e, _ = fixture(shock=True, count=1)
    original = e.demand_scenario.physical_shocks[0]
    other_scenario = replace(
        e.demand_scenario, physical_shocks=(replace(original, reveal_days=1),)
    )
    other = module.OrdinaryInformationMarketEngine(
        other_scenario, dated_rebooking=e.dated_rebooking
    )
    other.reset(tape=e.tape)
    assert e.demand_scenario != other.demand_scenario
    e.step(actions(e))
    other.step(actions(other))
    a, b = e.information_periods()[0].rows[0], other.information_periods()[0].rows[0]
    assert module.ordinary_information_view(a) == module.ordinary_information_view(b)
    assert module.ordinary_information_key(a) == module.ordinary_information_key(b)
    # Equality of this menu does not establish equality of full passenger knowledge.
