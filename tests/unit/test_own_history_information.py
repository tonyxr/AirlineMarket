"""A.10/A.5/B0/V9: NEW genuine own-history/re-entry source witnesses."""

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
from airline_market.policy.dated_rescue import DatedRebookingPolicy
from airline_market.policy.dated_reserve import UtcBookingClock
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.policy.potential_lineage import PotentialLineageMarketEngine
from airline_market.policy.rationed_native import WholePeriodPriority
from airline_market.sim.ordinary_information import OrdinaryInformationMarketEngine
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

if os.environ.get("OWN_HISTORY_INFORMATION_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_own_history_information", os.environ["OWN_HISTORY_INFORMATION_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.sim import own_history_information as module


def fixture(*, kind="exit", count=3, interline=True, outside=False):
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
    start = datetime(2026, 11, 21, tzinfo=timezone.utc)
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
    e = module.OwnHistoryInformationMarketEngine(scenario, **kwargs)
    parent = OrdinaryInformationMarketEngine(scenario, **kwargs)
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(76974, "unit/V9_own_history_information")
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


def run_pair(e, parent):
    while not e.terminal():
        e.step(actions(e))
        parent.step(actions(parent))
        assert pickle.dumps(e.world) == pickle.dumps(parent.world)
        assert pickle.dumps(e.trace()) == pickle.dumps(parent.trace())
        assert e.information_periods() == parent.information_periods()
        assert e.rebooking_history == parent.rebooking_history
    assert e.metrics() == parent.metrics()


def test_context_captured_before_one_native_marks_call_and_full_identity(monkeypatch):
    e, parent = fixture()
    original = PotentialLineageMarketEngine._marks
    counts = {id(e): 0, id(parent): 0}
    captures = []

    def observed(self, passenger, period):
        counts[id(self)] += 1
        if self is e:
            context = e._pending_own[-1]
            assert context.analyst_root == passenger.original.record_id
            assert context.view.own_facts == tuple(
                e._own_histories[context.analyst_root]
            )
            assert (
                context.view.own_facts[-1].kind == "native_ticket_cancelled"
                or context.view.own_facts[-1].native_period < period
            )
            captures.append(context)
        return original(self, passenger, period)

    monkeypatch.setattr(PotentialLineageMarketEngine, "_marks", observed)
    rng = pickle.dumps(np.random.get_state())
    run_pair(e, parent)
    assert captures and counts[id(e)] == counts[id(parent)] == len(captures)
    assert any(r["stage"] == "interline" for r in e.rebooking_history)
    assert pickle.dumps(np.random.get_state()) == rng


def test_failed_attempts_are_earlier_own_facts_not_new_people_or_future_outcomes():
    e, parent = fixture(interline=False)
    run_pair(e, parent)
    rows = [r for p in e.own_information_periods() for r in p.reentry_rows]
    assert len(rows) > len(e._own_histories)
    root = rows[0].analyst_root
    own = [r for r in rows if r.analyst_root == root]
    assert len(own) >= 2
    assert [len(r.view.own_facts) for r in own] == list(range(2, 2 + len(own)))
    assert all(
        r.view.own_facts[-1].native_period < r.view.native_period
        or r.view.own_facts[-1].kind == "native_ticket_cancelled"
        for r in own
    )
    assert all(f.kind == "native_reentry_outside" for f in own[-1].view.own_facts[2:])
    assert len({r.analyst_original_key for r in own}) == 1
    assert all(
        r.view.native_exact_expiry_utc == own[0].view.native_exact_expiry_utc
        for r in own
    )


def test_recovery_own_purchase_then_repeated_cancellation_keeps_first_expiry():
    e, parent = fixture(kind="recurrent", interline=False)
    run_pair(e, parent)
    rows = [r for p in e.own_information_periods() for r in p.reentry_rows]
    assert any(r["stage"] == "own" for r in e.rebooking_history)
    histories = e._own_histories
    recurrent = [
        root
        for root, facts in histories.items()
        if sum(f.kind == "native_ticket_cancelled" for f in facts) >= 2
    ]
    assert recurrent
    for root in recurrent:
        contexts = [r for r in rows if r.analyst_root == root]
        assert len({r.view.first_stranded_period for r in contexts}) == 1
        assert len({r.view.native_exact_expiry_utc for r in contexts}) == 1
        assert any(
            f.person_ticket == 1 and f.kind == "native_ticket_cancelled"
            for f in contexts[-1].view.own_facts
        )
        assert [f.ordinal for f in histories[root]] == list(range(len(histories[root])))


def test_native_own_view_excludes_global_keys_private_class_future_stock_and_current_outcome():
    e, parent = fixture()
    run_pair(e, parent)
    rows = [r for p in e.own_information_periods() for r in p.reentry_rows]
    view = module.own_reentry_information_view(rows[0])
    assert set(view) == {
        "native_period",
        "compiled_opportunity_utc",
        "original_market",
        "original_carrier",
        "first_stranded_period",
        "exclusive_deadline_period",
        "native_exact_expiry_utc",
        "interline_enabled",
        "own_facts",
        "scope",
    }
    assert all(
        set(f)
        == {
            "ordinal",
            "kind",
            "native_period",
            "person_ticket",
            "product_id",
            "fare",
            "stage",
        }
        for f in view["own_facts"]
    )
    assert [f["kind"] for f in view["own_facts"]] == [
        "native_ordinary_purchase",
        "native_ticket_cancelled",
    ]
    changed = replace(rows[0], analyst_root=9999, analyst_original_key=(9, 9999, 9))
    assert module.own_reentry_information_key(
        changed
    ) == module.own_reentry_information_key(rows[0])
    first_period = next(p for p in e.own_information_periods() if p.reentry_rows)
    assert (
        len({module.own_reentry_information_key(r) for r in first_period.reentry_rows})
        == 1
    )
    assert len({r.analyst_root for r in first_period.reentry_rows}) > 1


@pytest.mark.parametrize(
    "count,outside,kind", [(0, False, "exit"), (2, True, "exit"), (2, False, "control")]
)
def test_no_search_empty_outside_unstranded_support_keeps_all_parent_periods(
    count, outside, kind
):
    e, parent = fixture(count=count, outside=outside, kind=kind)
    run_pair(e, parent)
    assert len(e.own_information_periods()) == 5
    assert not any(p.reentry_rows for p in e.own_information_periods())
    assert all(len(p.rows) == count for p in e.information_periods())
    if outside or count == 0:
        assert e._own_histories == {}
    else:
        assert e._own_histories and all(len(h) == 1 for h in e._own_histories.values())


def test_nested_snapshot_branch_reset_and_owned_exports():
    e, _ = fixture(interline=False)
    e.step(actions(e))
    e.step(actions(e))
    before = pickle.dumps(e.snapshot())
    saved = module.own_information_record(e)
    json.dumps(saved, allow_nan=False)
    saved["analyst_context"]["original_roots"] = ()
    saved["reentry_views_by_period"][1]["views"][0]["own_facts"] = ()
    assert pickle.dumps(e.snapshot()) == before
    branch = e.branch()
    branch.step(actions(branch))
    assert (
        len(e.own_information_periods()) == 2
        and len(branch.own_information_periods()) == 3
    )
    assert len(branch._own_histories[0]) > len(e._own_histories[0])
    assert module.own_information_record(e)["analyst_context"]["original_roots"]
    e.reset(tape=e.tape)
    assert e.own_information_periods() == () and e._own_histories == {}


def test_lossless_own_fact_envelopes_signed_zero_unknown_and_nonfinite_are_declared_only():
    e, parent = fixture()
    run_pair(e, parent)
    row = next(r for p in e.own_information_periods() for r in p.reentry_rows)
    fact = row.view.own_facts[0]
    a = replace(row, view=replace(row.view, own_facts=(replace(fact, fare=0.0),)))
    b = replace(row, view=replace(row.view, own_facts=(replace(fact, fare=-0.0),)))
    assert module.own_reentry_information_key(a) != module.own_reentry_information_key(
        b
    )
    for value in (None, -3.125, float("inf"), float("nan")):
        declared = replace(
            row, view=replace(row.view, own_facts=(replace(fact, fare=value),))
        )
        json.dumps(module.own_reentry_information_view(declared), allow_nan=False)


@pytest.mark.parametrize("field", ["rule", "spec_hash"])
def test_snapshot_scope_refused_before_mutation(field):
    e, _ = fixture()
    before = pickle.dumps(e.snapshot())
    with pytest.raises(ValueError, match="contract"):
        e.restore(replace(e.snapshot(), **{field: "wrong"}))
    with pytest.raises(TypeError, match="combined"):
        e.restore(e.snapshot().native)
    assert pickle.dumps(e.snapshot()) == before


@pytest.mark.parametrize("hook", ["_requests", "_choices"])
def test_forced_hooks_refused_before_mutation(hook):
    e, _ = fixture()
    before = pickle.dumps(e.snapshot())
    with pytest.raises(ValueError, match="genuine"):
        e.step(actions(e), **{hook: []})
    assert pickle.dumps(e.snapshot()) == before


@pytest.mark.parametrize("name", ["booking_clock", "shock_clock"])
def test_native_clock_mismatch_refused(name):
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


def test_random_admission_refused():
    e, _ = fixture()
    p = WholePeriodPriority("unit/V9_own_history_information/refused")
    with pytest.raises(ValueError, match="chronological"):
        module.OwnHistoryInformationMarketEngine(e.demand_scenario, ordinary_priority=p)
    e.ordinary_priority = p
    before = pickle.dumps(e.snapshot())
    with pytest.raises(ValueError, match="chronological"):
        e.step(actions(e))
    assert pickle.dumps(e.snapshot()) == before


def test_native_callback_failure_retains_pending_facts_no_positive_query_or_retry():
    e, _ = fixture()

    def fail(_):
        raise RuntimeError("NEW own callback failure")

    e.controllers[0].observe_outcome = fail
    with pytest.raises(RuntimeError, match="callback"):
        e.step(actions(e))
    assert (
        e.world.t == 1 and e.snapshot().pending is not None and e.snapshot().histories
    )
    with pytest.raises(ValueError, match="partial"):
        module.own_information_record(e)
    with pytest.raises(ValueError, match="continuation"):
        e.step(actions(e))
    assert e.branch().snapshot().histories == e.snapshot().histories


def test_native_marks_failure_keeps_prospective_context_and_prior_receipt(monkeypatch):
    e, _ = fixture()
    e.step(actions(e))
    first = e.own_information_periods()[0]

    def fail(self, passenger, period):
        raise RuntimeError("NEW native marks failure")

    monkeypatch.setattr(PotentialLineageMarketEngine, "_marks", fail)
    with pytest.raises(RuntimeError, match="marks"):
        e.step(actions(e))
    assert (
        e.snapshot().periods == (first,)
        and len(e.snapshot().pending) == 1
        and e.world.t == 1
    )
    with pytest.raises(ValueError, match="partial"):
        e.own_information_periods()


def test_post_close_consistency_failure_preserves_native_mutations_and_prior_receipt():
    e, _ = fixture()
    e.step(actions(e))
    first = e.own_information_periods()[0]
    original = e.controllers[0].observe_outcome

    def corrupt(outcome):
        original(outcome)
        e._pending_own.append("NEW corrupt context")

    e.controllers[0].observe_outcome = corrupt
    with pytest.raises(RuntimeError, match="attempts and own contexts"):
        e.step(actions(e))
    assert (
        e.world.t == 2
        and e.snapshot().periods == (first,)
        and e.snapshot().pending is not None
    )
    with pytest.raises(ValueError, match="partial"):
        e.own_information_periods()
