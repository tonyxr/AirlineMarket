"""A.10/A.5/B0/V9: NEW prospective menu/native lifecycle witnesses."""

import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.policy.dated_demand import compile_dated_demand
from airline_market.policy.dated_rescue import (
    DatedRebookingPolicy,
    DatedRescueMenu,
    UtcWindow,
)
from airline_market.policy.dated_reserve import DatedSeatReserve
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.policy.rationed_native import WholePeriodPriority
from airline_market.policy.reserve import ReserveMarketEngine
from airline_market.sim.ordinary_cohort import OrdinaryCohortMarketEngine
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from tests.unit import test_dated_demand as dated
from tests.unit import test_rationing_native as native

if os.environ.get("ORDINARY_MENU_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_ordinary_menu", os.environ["ORDINARY_MENU_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.sim import ordinary_menu as module


def fixture(*, shock=False, marks=None, reserve=False):
    c, clock, dep = dated.inputs()
    scenario = compile_dated_demand(c, clock, dep)
    kwargs = {}
    if shock:
        scenario = replace(
            scenario,
            physical_shocks=(
                B1DatedShock("exit", "rival_exit", "B_OH0", 3, rival_carrier="B"),
                B1DatedShock("weather", "weather", "B_HD0", 3, ".6", 1),
            ),
        )
        kwargs["dated_rebooking"] = DatedRebookingPolicy(72, 20, 100, interline=True)
    if reserve:
        kwargs["dated_menu"] = DatedRescueMenu(
            UtcWindow(clock.at(0), clock.at(5)),
            "same_route",
            route_fares=tuple((m.id, 99) for m in c.spec.markets),
        )
        kwargs["dated_reserve"] = DatedSeatReserve(
            clock.at(0), ".1", tuple(l.id for l in c.spec.legs if l.carrier == "A"), dep
        )
    e = module.OrdinaryMenuMarketEngine(scenario, **kwargs)
    parent = OrdinaryCohortMarketEngine(scenario, **kwargs)
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(66974, "unit/V9_ordinary_menu")
    )
    if marks is not None:
        tape = replace(tape, arrival_marks=tuple(marks for _ in tape.arrival_marks))
    for x in (e, parent):
        x.reset(tape=tape)
    return e, parent


def test_full_native_identity_and_each_input_is_actual_choice_input(monkeypatch):
    e, parent = fixture(shock=True)
    import airline_market.sim.rebooking as core

    utility = core.systematic_utilities
    actual = []

    def observe(c, market, taste, prices, mask):
        if (
            e._menu_inflight is None
            and e._pending_menus is not None
            and len(e._pending_menus) > len(actual)
        ):
            actual.append(
                (tuple(float(p) for p in prices), tuple(bool(v) for v in mask))
            )
        return utility(c, market, taste, prices, mask)

    rng = pickle.dumps(np.random.get_state())
    monkeypatch.setattr(core, "systematic_utilities", observe)
    while not e.terminal():
        actual.clear()
        e.step(native.actions(e))
        rows = e.menu_periods()[-1].rows
        assert [(r.prices, r.net_sellable) for r in rows] == actual
        assert len(actual) == len(rows)  # capture performs no utility forward
        parent.step(native.actions(parent))
        assert pickle.dumps(e.world) == pickle.dumps(parent.world)
        assert pickle.dumps(e.trace()) == pickle.dumps(parent.trace())
        assert e.ordinary_periods() == parent.ordinary_periods()
        assert e.coordinate_records() == parent.coordinate_records()
    assert e.metrics() == parent.metrics()
    assert pickle.dumps(np.random.get_state()) == rng


def test_same_period_depletion_is_captured_before_next_choice():
    marks = tuple(ArrivalMark((i + 1) / 4, 0, 0, 0, 0) for i in range(3))
    e, parent = fixture(marks=marks)
    j = e.compiled.product_index["B_OH0"]
    g = e.tape.gumbel.copy()
    g[:] = -1e9
    g[:, :, -1] = 0
    g[:, :, j] = 1e9
    g.setflags(write=False)
    for x in (e, parent):
        x.tape = replace(x.tape, gumbel=g)
        x.world.carriers[1].capacity[:] = 1
        x.step(native.actions(x))
    rows = e.menu_periods()[0].rows
    assert len(rows) == 3
    assert [r.net_sellable[j] for r in rows] == [True, False, False]
    assert [r.person.native_row["chosen"] for r in rows] == [j, None, None]
    assert pickle.dumps(e.world) == pickle.dumps(parent.world)


def test_empty_periods_and_outside_requests_retained():
    e, _ = fixture(marks=())
    for _ in range(2):
        e.step(native.actions(e))
    assert [(p.period, p.rows) for p in e.menu_periods()] == [(0, ()), (1, ())]
    e, _ = fixture(marks=(ArrivalMark(0.1, 0, 0, 0, 0),))
    g = e.tape.gumbel.copy()
    g[:] = -1e9
    g[:, :, -1] = 1e9
    g.setflags(write=False)
    e.tape = replace(e.tape, gumbel=g)
    e.step(native.actions(e))
    row = e.menu_periods()[0].rows[0]
    assert row.person.original_record is None
    assert row.person.native_row["chosen"] is None
    assert len(row.prices) == len(e.compiled.spec.products)
    assert row.native_columns[-1] == len(e.compiled.spec.products)


def test_branch_reset_owned_export_and_capture_contract():
    e, _ = fixture()
    e.step(native.actions(e))
    before = pickle.dumps(e.snapshot())
    saved = module.ordinary_menu_record(e)
    json.dumps(saved, allow_nan=False)
    assert pickle.dumps(e.snapshot()) == before
    saved["periods"][0]["rows"] = ()
    assert module.ordinary_menu_record(e)["periods"][0]["rows"]
    branch = e.branch()
    assert pickle.dumps(branch.snapshot()) == before
    branch.step(native.actions(branch))
    assert len(e.menu_periods()) == 1 and len(branch.menu_periods()) == 2
    for bad in (
        replace(e.snapshot(), rule="different"),
        replace(e.snapshot(), spec_hash="different"),
    ):
        with pytest.raises(ValueError, match="contract"):
            e.restore(bad)
        assert pickle.dumps(e.snapshot()) == before
    with pytest.raises(TypeError):
        e.restore(e.snapshot().native)
    e.reset(tape=e.tape)
    assert e.menu_periods() == ()


def test_native_failure_retains_pending_no_receipt_query_or_continuation():
    e, _ = fixture()

    def fail(outcome):
        raise RuntimeError("NEW callback failure")

    e.controllers[0].observe_outcome = fail
    with pytest.raises(RuntimeError, match="callback"):
        e.step(native.actions(e))
    snap = e.snapshot()
    assert snap.pending is not None and snap.periods == ()
    assert e.world.t == 1  # native mutations retained, no rollback
    with pytest.raises(ValueError, match="partial"):
        e.menu_periods()
    with pytest.raises(ValueError, match="continuation"):
        e.step(native.actions(e))
    branch = e.branch()
    assert branch.snapshot().pending == snap.pending
    with pytest.raises(ValueError, match="partial"):
        module.ordinary_menu_record(branch)


@pytest.mark.parametrize("hook", ["_requests", "_choices"])
def test_forced_hooks_refused_before_mutation(hook):
    e, _ = fixture()
    before = pickle.dumps(e.snapshot())
    with pytest.raises(ValueError, match="genuine"):
        e.step(native.actions(e), **{hook: []})
    assert pickle.dumps(e.snapshot()) == before


def test_priority_refused_at_construction_and_before_step():
    e, _ = fixture()
    with pytest.raises(ValueError, match="chronological"):
        module.OrdinaryMenuMarketEngine(
            e.demand_scenario,
            ordinary_priority=WholePeriodPriority("unit/menu_refused"),
        )
    e.ordinary_priority = WholePeriodPriority("unit/menu_refused")
    before = pickle.dumps(e.snapshot())
    with pytest.raises(ValueError, match="chronological"):
        e.step(native.actions(e))
    assert pickle.dumps(e.snapshot()) == before


def test_held_seat_and_actual_reentry_lifecycle_no_extra_availability_call(monkeypatch):
    marks = tuple(ArrivalMark((i + 1) / 4, 0, 0, 0, 0) for i in range(3))
    e, parent = fixture(shock=True, reserve=True, marks=marks)
    g = e.tape.gumbel.copy()
    g[:2, :, :] = -1e9
    g[:2, :, -1] = 0
    g[:2, :, e.compiled.product_index["B_OH0"]] = 1e9
    g.setflags(write=False)
    for x in (e, parent):
        x.tape = replace(x.tape, gumbel=g)
    method = ReserveMarketEngine._sellable_global
    counts = {id(e): 0, id(parent): 0}

    def counted(self):
        counts[id(self)] += 1
        return method(self)

    monkeypatch.setattr(ReserveMarketEngine, "_sellable_global", counted)
    while not e.terminal():
        e.step(native.actions(e))
        parent.step(native.actions(parent))
        assert counts[id(e)] == counts[id(parent)]
        assert e.reserve_events == parent.reserve_events
        assert e.rebooking_history == parent.rebooking_history
        assert pickle.dumps(e.world) == pickle.dumps(parent.world)
    assert any(r["kind"] == "hold" and r["seats"] > 0 for r in e.reserve_events)
    assert any(r["replacement_record"] is not None for r in e.rebooking_history)
    assert len(e.menu_periods()) == e.world.t


def test_failure_before_mask_retains_inflight_request_in_snapshot(monkeypatch):
    e, _ = fixture(marks=(ArrivalMark(0.1, 0, 0, 0, 0),))
    original = ReserveMarketEngine._sellable_global

    def fail(self):
        if self is e and e._menu_inflight is not None:
            raise RuntimeError("NEW availability failure")
        return original(self)

    monkeypatch.setattr(ReserveMarketEngine, "_sellable_global", fail)
    with pytest.raises(RuntimeError, match="availability"):
        e.step(native.actions(e))
    snap = e.snapshot()
    assert snap.inflight is not None and snap.pending == () and snap.periods == ()
    assert e.branch().snapshot().inflight == snap.inflight
    with pytest.raises(ValueError, match="partial"):
        e.menu_periods()


def test_post_close_capture_error_retains_prior_complete_receipt(monkeypatch):
    e, _ = fixture()
    e.step(native.actions(e))
    first = e.menu_periods()[0]
    original = OrdinaryCohortMarketEngine.ordinary_periods

    def wrong(self):
        p = original(self)
        return (*p[:-1], replace(p[-1], period=999))

    monkeypatch.setattr(OrdinaryCohortMarketEngine, "ordinary_periods", wrong)
    with pytest.raises(RuntimeError, match="cohort and menus"):
        e.step(native.actions(e))
    assert e.world.t == 2 and e.snapshot().periods == (first,)
    assert e.snapshot().pending is not None
    with pytest.raises(ValueError, match="partial"):
        e.menu_periods()
