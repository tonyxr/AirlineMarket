"""A.10/A.5/B0/V9: NEW all-ordinary identity/cohort join, no lifecycle adoption."""

import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
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
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.policy.rationed_native import WholePeriodPriority
from airline_market.policy.regulator import CapPolicy, CapWindow
from airline_market.sim.passenger_accounts import original_passenger_accounts
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from tests.unit import test_dated_demand as dated
from tests.unit import test_rationing_native as native

if os.environ.get("ORDINARY_COHORT_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_ordinary_cohort", os.environ["ORDINARY_COHORT_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.sim import ordinary_cohort as module


def fixture(mode="chronological", *, repeated=False):
    c, clock, dep = dated.inputs()
    scenario = compile_dated_demand(c, clock, dep)
    kwargs = {}
    if repeated:
        scenario = replace(
            scenario,
            physical_shocks=(
                B1DatedShock("exit", "rival_exit", "B_OH0", 3, rival_carrier="B"),
                B1DatedShock("weather", "weather", "B_HD0", 3, ".6", 1),
            ),
        )
        kwargs["dated_rebooking"] = DatedRebookingPolicy(72, 20, 100, interline=True)
    if mode == "efficient":
        kwargs.update(
            order_policy=EfficientNativeOrder(
                AllocationObjective("native_choice_gain", "unit gain only"), 100000
            ),
            cap_policy=CapPolicy(
                (CapWindow(0, 5, tuple((p.id, 100) for p in c.spec.products)),)
            ),
        )
        cls, parent = module.OrdinaryCohortOrderEngine, DatedOrderBenchmarkEngine
    else:
        kwargs["ordinary_priority"] = (
            WholePeriodPriority("unit/V9_ordinary_cohort") if mode == "random" else None
        )
        kwargs["cap_policy"] = CapPolicy(
            (CapWindow(0, 5, tuple((p.id, 100) for p in c.spec.products)),)
        )
        cls, parent = module.OrdinaryCohortMarketEngine, DatedDemandMarketEngine
    e = cls(scenario, **kwargs)
    p = parent(scenario, **kwargs)
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(51974, "unit/V9_ordinary_cohort")
    )
    if repeated:
        marks = tuple(ArrivalMark((i + 1) / 4, 0, 0, 0, 0) for i in range(3))
        g = tape.gumbel.copy()
        g[:2, :3, :] = -1e9
        g[:2, :3, -1] = 0
        g[:2, :3, c.product_index["B_OH0"]] = 1e9
        g.setflags(write=False)
        tape = replace(
            tape, arrival_marks=(marks, marks, *tape.arrival_marks[2:]), gumbel=g
        )
    for x in (e, p):
        x.reset(tape=tape)
        if repeated:
            x.world.carriers[0].capacity[:] = 3
            x.world.carriers[1].capacity[:] = 3
    return e, p


def panel(e):
    accounts = original_passenger_accounts(
        e
    )  # ONE NEW current producer; no saved-account replay.
    return module.ordinary_panel_record(
        module.ordinary_person_panel(e.ordinary_periods(), accounts)
    )


@pytest.mark.parametrize("mode", ["chronological", "random", "efficient"])
def test_capture_preserves_full_native_world_trace_metrics_lineage_and_global_rng(
    mode, monkeypatch
):
    e, p = fixture(mode)
    before = pickle.dumps(np.random.get_state())

    def forbidden(*args, **kwargs):
        raise AssertionError("no tape generation during capture/step")

    monkeypatch.setattr(ExogenousTape, "generate", forbidden)
    while not e.terminal():
        e.step(native.actions(e))
        p.step(native.actions(p))
        assert pickle.dumps(e.world) == pickle.dumps(p.world) and pickle.dumps(
            e.trace()
        ) == pickle.dumps(p.trace())
        assert e.coordinate_records() == p.coordinate_records()
        rows = [r for period in e.ordinary_periods() for r in period.rows]
        assert [dict(r.native_row) for r in rows] == e.world.request_log
    assert e.metrics() == p.metrics() and pickle.dumps(np.random.get_state()) == before
    saved = panel(e)
    assert saved["ordinary_original_requests"] == len(e.world.request_log)
    assert (
        saved["originally_booked"] + saved["ordinary_outside"]
        == saved["ordinary_original_requests"]
    )
    assert saved["period_request_counts"] == [len(r.rows) for r in e.ordinary_periods()]


def test_outside_request_has_physical_identity_and_original_CS_without_invented_ticket():
    e, _ = fixture()
    g = e.tape.gumbel.copy()
    g[:, :, :] = -1e9
    g[:, :, -1] = 1e9
    g.setflags(write=False)
    e.tape = replace(e.tape, gumbel=g)
    e.step(native.actions(e))
    saved = panel(e)
    assert saved["ordinary_original_requests"] > 0 and saved["originally_booked"] == 0
    assert all(
        r["original_record"] is None and r["booked_passenger_account"] is None
        for r in saved["persons"]
    )
    assert (
        len({tuple(r["potential_coordinate"]) for r in saved["persons"]})
        == saved["ordinary_outside"]
    )
    assert all(
        r["original_ordinary_log"]["consumer_surplus"]["exact"] is not None
        for r in saved["persons"]
    )
    assert not saved["empirical_exposure_or_travellers_inferred"]


@pytest.mark.parametrize("mode", ["chronological", "random", "efficient"])
def test_actual_reentry_and_repeat_cancellation_join_to_original_CS_row_without_new_people(
    mode,
):
    e, _ = fixture(mode, repeated=True)
    for _ in range(4):
        e.step(native.actions(e))
    accounts = original_passenger_accounts(e)
    saved = module.ordinary_panel_record(
        module.ordinary_person_panel(e.ordinary_periods(), accounts)
    )
    assert accounts.data["replacement_tickets_ever"] > 0
    assert saved["ordinary_original_requests"] == len(e.world.request_log)
    assert (
        sum(r["booked_passenger_account"] is not None for r in saved["persons"])
        == accounts.data["original_booked_lineages"]
    )
    assert any(
        r["booked_passenger_account"]
        and len(r["booked_passenger_account"]["tickets"]) > 1
        for r in saved["persons"]
    )
    assert not saved["reentry_searches_are_additional_people"]
    assert not saved["original_ordinary_CS_reclassified_after_stranding"]
    for r in saved["persons"]:
        assert r["lifecycle_consumer_surplus"] is None
    assert (
        saved["total_lifecycle_surplus"]
        is saved["denied_requests"]
        is saved["misallocation"]
        is None
    )


def test_empty_periods_are_retained_in_complete_prefix():
    e, _ = fixture()
    e.tape = replace(e.tape, arrival_marks=tuple(() for _ in e.tape.arrival_marks))
    for _ in range(2):
        e.step(native.actions(e))
    saved = panel(e)
    assert (
        saved["period_request_counts"] == [0, 0]
        and saved["ordinary_original_requests"] == 0
    )


def test_one_posting_callback_and_original_parent_coordinate_tracking(monkeypatch):
    e, _ = fixture("efficient")
    calls = []
    outcomes = {}
    post = e.post_actions

    def posting(actions):
        calls.append(1)
        return post(actions)

    e.post_actions = posting
    for i, c in e.controllers.items():

        def outcome(row, i=i):
            outcomes[i] = outcomes.get(i, 0) + 1

        c.observe_outcome = outcome
    e.step(native.actions(e))
    assert calls == [1] and outcomes == {i: 1 for i in e.controllers}
    assert all(
        e._physical_coordinates[r.original_record].stream_key == r.key
        for r in e.ordinary_periods()[0].rows
        if r.original_record is not None
    )


def test_failed_native_callback_keeps_partial_capture_without_positive_period_or_panel():
    e, _ = fixture()

    def failure(row):
        raise RuntimeError("NEW unit callback failure")

    e.controllers[0].observe_outcome = failure
    with pytest.raises(RuntimeError, match="NEW unit callback"):
        e.step(native.actions(e))
    assert e._ordinary_periods == [] and e._pending_ordinary is not None
    with pytest.raises(ValueError, match="incomplete ordinary capture"):
        e.ordinary_periods()
    with pytest.raises(ValueError):
        original_passenger_accounts(e)


@pytest.mark.parametrize("mode", ["chronological", "random", "efficient"])
def test_independent_snapshot_branch_reset_and_capture_rule_parent_contract(mode):
    e, _ = fixture(mode)
    e.step(native.actions(e))
    snap = e.snapshot()
    b = e.branch()
    e.step(native.actions(e))
    assert len(b.ordinary_periods()) == 1
    e.restore(snap)
    assert pickle.dumps(e.world) == pickle.dumps(b.world)
    with pytest.raises(ValueError, match="capture rule"):
        e.restore(replace(snap, capture_rule="other"))
    e.reset(tape=e.tape)
    assert e.ordinary_periods() == () and len(b.ordinary_periods()) == 1


@pytest.mark.parametrize(
    "damage",
    [
        "period_gap",
        "row_gap",
        "duplicate",
        "wrong_account",
        "outside_support",
        "missing_booked",
    ],
)
def test_structurally_incompatible_supplied_panels_refused_without_filling(damage):
    e, _ = fixture()
    e.step(native.actions(e))
    ps = list(e.ordinary_periods())
    a = original_passenger_accounts(e)
    if damage == "period_gap":
        ps = []
    elif damage == "row_gap":
        ps[0] = replace(ps[0], log_start=1)
    elif damage == "duplicate":
        rows = list(ps[0].rows)
        rows[-1] = replace(rows[-1], key=rows[0].key)
        ps[0] = replace(ps[0], rows=tuple(rows))
    elif damage == "wrong_account":
        rows = list(ps[0].rows)
        i = next(i for i, r in enumerate(rows) if r.original_record is not None)
        rows[i] = replace(rows[i], native_rank=999)
        ps[0] = replace(ps[0], rows=tuple(rows))
    elif damage == "outside_support":
        a = replace(a, data={**a.data, "ordinary_outside_choices_in_prefix": 999})
    else:
        rows = list(ps[0].rows)
        i = next(i for i, r in enumerate(rows) if r.original_record is not None)
        rows[i] = replace(rows[i], original_record=None)
        ps[0] = replace(ps[0], rows=tuple(rows))
    with pytest.raises(ValueError):
        module.ordinary_person_panel(ps, a)


def test_immutable_owned_export_forced_hook_refusal_and_no_adoption():
    e, _ = fixture()
    with pytest.raises(ValueError, match="genuine native"):
        e.step(native.actions(e), _requests=[])
    e.step(native.actions(e))
    a = original_passenger_accounts(e)
    r = module.ordinary_person_panel(e.ordinary_periods(), a)
    saved = module.ordinary_panel_record(r)
    json.dumps(saved, allow_nan=False)
    saved["persons"].clear()
    assert r["persons"]
    with pytest.raises(TypeError):
        e.ordinary_periods()[0].rows[0].native_row["chosen"] = 999
    assert not r["primary_B4_welfare_adopted"] and r["scientific_budget"] is None


def test_shared_tape_outside_customer_keeps_potential_key_when_prior_thinning_changes_native_rank():
    first, _ = fixture()
    second, _ = fixture()
    base = first.tape
    g = base.gumbel.copy()
    g[0, :, :] = -1e9
    g[0, :, -1] = 1e9
    g.setflags(write=False)
    common = ArrivalMark(0.2, 0, 0, 0, 0)
    for engine, earlier in [(first, 0), (second, 1)]:
        marks = (ArrivalMark(0.1, 0, 0, earlier, 0), common)
        engine.reset(
            tape=replace(base, arrival_marks=(marks, *base.arrival_marks[1:]), gumbel=g)
        )
        engine.step(native.actions(engine))
    a = next(r for r in first.ordinary_periods()[0].rows if r.key == (0, 1, 0))
    b = next(r for r in second.ordinary_periods()[0].rows if r.key == (0, 1, 0))
    assert a.native_rank == 1 and b.native_rank == 0
    assert a.original_record is b.original_record is None
    assert a.native_row == b.native_row
    assert a.native_log_index != b.native_log_index
