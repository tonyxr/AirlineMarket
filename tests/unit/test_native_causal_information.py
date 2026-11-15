"""A.10/A.5/A.14/B0/V9: NEW genuine native causal-source units only."""

import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from itertools import pairwise
from pathlib import Path

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
from airline_market.sim.causal_information import (
    CausalInformationLimit,
    causal_decision_view,
)
from airline_market.sim.received_information import DisclosureField
from airline_market.sim.reentry_stage_information import (
    ReentryStageInformationMarketEngine,
)
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

if os.environ.get("NATIVE_CAUSAL_SOURCE_DIR"):
    folder = Path(os.environ["NATIVE_CAUSAL_SOURCE_DIR"])
    sp = importlib.util.spec_from_file_location(
        "airline_market.sim.native_causal_information",
        folder / "native_causal_information.py",
    )
    loaded = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = loaded
    sp.loader.exec_module(loaded)
from airline_market.sim import native_causal_information as module


def source(
    *,
    route="base",
    interline=True,
    weather=False,
    outside=False,
    count=2,
    row_limit=1000000,
    connection=False,
    missing_type=False,
):
    # Locally adapted constructor, NEW namespace/date/marks; no old fixture call.
    spec = screen_spec(
        {
            "T": 5,
            "actions": 2,
            "n_spokes": 1,
            "capacity": 20,
            "capacity_support": [3, 4],
        }
    )
    c = compile_market(
        replace(
            spec,
            demand=replace(
                spec.demand,
                mode="poisson",
                arrival_probability=None,
                market_weights=None,
                intensity={
                    m.id: PeriodSchedule.parse({"constant": 2.0}) for m in spec.markets
                },
            ),
        )
    )
    start = datetime(2027, 1, 18, tzinfo=timezone.utc)
    clock = UtcBookingClock(tuple(start + timedelta(hours=12 * t) for t in range(6)))
    departures = tuple(
        (l.id, start + timedelta(hours=84 if l.id == "B_OH0" else 108))
        for l in c.spec.legs
    )
    scenario = compile_dated_demand(c, clock, departures)
    shock = (
        B1DatedShock("fresh_weather", "weather", "B_OH0", 3, ".6", 1)
        if weather
        else B1DatedShock("fresh_exit", "rival_exit", "B_OH0", 3, rival_carrier="B")
    )
    scenario = replace(scenario, physical_shocks=(shock,))
    kwargs = {"dated_rebooking": DatedRebookingPolicy(72, 20, 100, interline=interline)}
    if route != "base":
        kwargs["dated_menu"] = DatedRescueMenu(
            UtcWindow(clock.at(0), clock.at(5)),
            "same_route",
            route_fares=tuple((m.id, 99) for m in c.spec.markets),
        )
    if route in {"reserve", "noheld", "zero"}:
        leg = "A_HD0" if route == "noheld" else "A_OH0"
        kwargs["dated_reserve"] = DatedSeatReserve(
            clock.at(1), Decimal("0" if route == "zero" else ".10"), (leg,), departures
        )
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(86974, "unit/V9_native_causal_information")
    )
    target = c.product_index["B_OD0_0" if connection else "B_OH0"]
    market = int(c.product_market[target])
    marks = tuple(
        ArrivalMark((i + 1) / (count + 1), market, 0, 0, 0) for i in range(count)
    )
    g = tape.gumbel.copy()
    g[:] = -1e9
    g[:, :, -1] = 0
    g[:, :, target] = 1e9
    if outside:
        g[:, :, -1] = 2e9
    elif count:
        g[:, count - 1, -1] = 2e9
    g.setflags(write=False)
    tape = replace(tape, arrival_marks=(marks, (), (), (), ()), gumbel=g)
    keys = tuple((0, i, market) for i in range(count))
    types = tuple(
        module.DeclaredKnownType(
            key, None if i else "declared-known", "explicit witness type", "unit-type"
        )
        for i, key in enumerate(keys)
        if not (missing_type and i == 0)
    )
    deliveries = tuple(
        module.DeclaredInputDelivery(
            key,
            t,
            stage,
            f"notice-{i}-{t}-{stage}",
            "notice",
            f"declared-{stage}",
            "unit-declared-message",
            module.TIMING,
        )
        for i, key in enumerate(keys)
        for t in range(5)
        for stage in ("ordinary", "own", "interline")
        if stage != "ordinary" or t == 0
    )
    e = module.NativeCausalInformationMarketEngine(
        scenario,
        causal_fields=(DisclosureField("notice", "text", "declared notice", "text"),),
        causal_known_types=types,
        causal_deliveries=deliveries,
        causal_definition="declared source-order witness",
        causal_unobserved_definition="no inferred knowledge",
        causal_row_limit=row_limit,
        **kwargs,
    )
    parent = ReentryStageInformationMarketEngine(scenario, **kwargs)
    for engine in [e, parent]:
        engine.reset(tape=tape)
    actions = {
        carrier.code: PriceAction({j: 200 if i == 0 else 300 for j in c.products_of[i]})
        for i, carrier in enumerate(c.spec.carriers)
    }
    return e, parent, actions


def advance(pair, steps=5, proof=True):
    e, parent, actions = pair
    for t in range(steps):
        for engine in (parent, e):
            if t == 1 and proof:
                for p in engine._passengers.values():
                    engine.present_cancelled_itinerary(p.original, airport_counter=True)
                    engine.present_cancelled_itinerary(
                        p.original, airport_counter=False
                    )
            engine.step(actions)
    return e, parent


@pytest.mark.parametrize("route", ["base", "rescue", "reserve", "noheld", "zero"])
def test_new_all_routes_native_identity_and_actual_source_order(route):
    rng = pickle.dumps(np.random.get_state())
    e, p = advance(source(route=route))
    assert pickle.dumps(e.snapshot().native) == pickle.dumps(p.snapshot())
    assert pickle.dumps(e.trace()) == pickle.dumps(p.trace())
    assert pickle.dumps(e.metrics()) == pickle.dumps(p.metrics())
    assert pickle.dumps(np.random.get_state()) == rng
    rows = e.causal_periods()
    assert len(rows) == 5 and len(rows[0].inputs) == 2
    assert [r.kind for r in rows[1].inputs] == ["own", "interline"]
    own, interline = rows[1].inputs
    view = causal_decision_view(interline.receipt)
    history = view["earlier_person_history"]
    assert any(x.get("kind") == "earlier_choice" and x.get("outside") for x in history)
    assert own.receipt.source_journal.events[-1].payload.stage == "own"
    assert interline.computation_input["earlier_within_attempt_results"][0]["outside"]
    events = e.snapshot().journal.events
    assert len([x for x in events if type(x.payload) is module.RegisterPerson]) == 2
    assert (
        len(
            [
                x
                for x in events
                if type(x.payload) is module.OwnAction
                and x.payload.kind == "presentation"
            ]
        )
        == 2
    )
    assert any(
        type(x.payload) is module.OwnAction
        and x.payload.kind == "cancellation"
        and x.position.anchor_utc == e.demand_scenario.clock.at(1)
        for x in events
    )
    for a, b in pairwise(events):
        assert b.position > a.position
        if a.position.anchor_utc == b.position.anchor_utc:
            assert b.position.ordinal == a.position.ordinal + 1
    out = module.native_causal_information_record(e)
    assert out["native_parent_steps_completed"] == 5
    assert not out["declared_message_schedule_is_actual_external_reception"]
    assert out["parent_source"]["source_context"]["scenario"]
    json.dumps(out, allow_nan=False)


@pytest.mark.parametrize(
    "kwargs",
    [{"count": 0}, {"outside": True}, {"interline": False}, {"connection": True}],
)
def test_new_full_empty_outside_disabled_connection_support(kwargs):
    e, p = advance(source(**kwargs))
    assert pickle.dumps(e.snapshot().native) == pickle.dumps(p.snapshot())
    if kwargs.get("count") == 0:
        assert not module.native_causal_information_record(e)[
            "native_decision_input_rows_captured"
        ]
        assert e.snapshot().journal.events == ()
    if kwargs.get("interline") is False:
        assert all(
            x.kind != "interline"
            for period in e.causal_periods()
            for x in period.inputs
        )
    assert len(e.causal_periods()) == 5


def test_new_earlier_receipt_owned_privacy_and_later_result_excluded():
    e, p, actions = source()
    for x in (p, e):
        x.step(actions)
    row = e.causal_periods()[0].inputs[0]
    before = pickle.dumps(row)
    view = causal_decision_view(row.receipt)
    text = json.dumps(view)
    for private in (
        "analyst",
        "person_key",
        "source_ref",
        "ordinal",
        "gumbel",
        "known_class",
        "current_result",
    ):
        assert private not in text
    assert row.receipt.source_journal.events[-1].payload.stage == "ordinary"
    assert not any(
        type(x.payload) is module.DecisionResult
        for x in row.receipt.source_journal.events
    )
    for x in (p, e):
        x.step(actions)
    assert pickle.dumps(row) == before
    out = module.native_causal_information_record(e)
    out["programme"].clear()
    assert module.native_causal_information_record(e)["programme"]
    with pytest.raises(TypeError):
        row.computation_input["new"] = 1


def test_new_missing_explicit_type_is_pending_failure_before_choice():
    e, p, actions = source(missing_type=True)
    with pytest.raises(ValueError, match="explicit known or unknown"):
        e.step(actions)
    assert e.world.t == 0 and not e.world.request_log
    assert e.snapshot().failed and e.snapshot().pending == ()
    with pytest.raises(ValueError, match="incomplete"):
        e.causal_periods()
    with pytest.raises(ValueError, match="incomplete"):
        e.step(actions)
    assert p.world.t == 0


@pytest.mark.parametrize("hook", ["_input", "_result", "_finish_causal_period"])
def test_new_injected_failure_retains_native_pending_and_no_positive_query(
    monkeypatch, hook
):
    e, _p, actions = source()

    def fail(*args, **kwargs):
        raise RuntimeError("new-injected-causal-failure")

    monkeypatch.setattr(e, hook, fail)
    with pytest.raises(RuntimeError, match="new-injected"):
        e.step(actions)
    snap = e.snapshot()
    assert snap.failed and snap.periods == () and snap.pending is not None
    if hook == "_finish_causal_period":
        assert e.world.t == 1 and len(e.stage_periods()) == 1
    with pytest.raises(ValueError):
        module.native_causal_information_record(e)
    with pytest.raises(ValueError):
        e.step(actions)


def test_new_proposed_completion_limit_is_before_append(monkeypatch):
    e, _p, actions = source()
    original = e._check_causal_limit

    def check(proposed=None):
        if type(proposed) is module.NativeCausalPeriod:
            raise module.NativeCausalInformationLimit("new-complete-limit")
        return original(proposed)

    monkeypatch.setattr(e, "_check_causal_limit", check)
    with pytest.raises(module.NativeCausalInformationLimit):
        e.step(actions)
    assert e.world.t == 1 and e.snapshot().periods == ()
    assert e.snapshot().pending


def test_new_branch_reset_and_programme_snapshot_binding():
    e, _p, actions = source()
    e.step(actions)
    snap = e.snapshot()
    branch = e.branch()
    assert module.native_causal_information_record(
        branch
    ) == module.native_causal_information_record(e)
    branch.step(actions)
    assert len(e.causal_periods()) == 1 and len(branch.causal_periods()) == 2
    with pytest.raises(ValueError, match="programme"):
        branch.restore(replace(snap, contract_hash="changed"))
    e.restore(snap)
    assert len(e.causal_periods()) == 1
    e.reset(tape=e.tape)
    assert e.causal_periods() == () and not e.snapshot().journal.events


def test_new_no_forced_hooks_or_implicit_scalar_timing():
    e, _p, actions = source()
    with pytest.raises(ValueError, match="genuine"):
        e.step(actions, _requests=())
    assert e.world.t == 0 and not e.snapshot().journal.events
    row = e.causal_deliveries[0]
    with pytest.raises(ValueError):
        replace(row, timing="assumed-reception")
    with pytest.raises(ValueError):
        replace(row, person_key=(True, 0, 0))


def test_new_same_call_counts_no_extra_native_probability_or_choice(monkeypatch):
    from airline_market.sim import reentry_stage_executor as rebooking

    e, p, actions = source()
    counts = {"new": {}, "parent": {}}
    current = ["parent"]
    for name in ("systematic_utilities", "sampled_choice"):
        original = getattr(rebooking, name)

        def wrap(*args, _f=original, _n=name, **kwargs):
            d = counts[current[0]]
            d[_n] = d.get(_n, 0) + 1
            return _f(*args, **kwargs)

        monkeypatch.setattr(rebooking, name, wrap)
    for t in range(5):
        for label, engine in (("parent", p), ("new", e)):
            current[0] = label
            engine.step(actions)
    assert counts["parent"] == counts["new"]
    assert counts["new"]["sampled_choice"] > 0


def test_new_native_utility_observes_already_saved_input_and_same_arguments(
    monkeypatch,
):
    from airline_market.sim import reentry_stage_executor as rebooking

    e, p, actions = source()
    for engine in (p, e):
        engine.step(actions)
    original = rebooking.systematic_utilities
    observed = []

    def witness(compiled, market, taste, prices, active):
        row = e._causal_pending[-1]
        native = e._pending_stage_inputs[-1]
        assert row.kind == native.view["stage"]
        assert e._causal_decision is not None
        assert row.receipt.source_journal.events[-1].payload.stage == row.kind
        assert tuple(float(x) for x in prices) == native.prices
        assert tuple(bool(x) for x in active) == native.offered
        observed.append(row.kind)
        return original(compiled, market, taste, prices, active)

    monkeypatch.setattr(rebooking, "systematic_utilities", witness)
    e.step(actions)
    assert observed == ["own", "interline"]


def test_new_presentation_capture_failure_retains_actual_proof_and_prior_period(
    monkeypatch,
):
    e, _p, actions = source()
    e.step(actions)
    passenger = next(iter(e._passengers.values()))

    def fail(*args):
        raise RuntimeError("presentation-source-failure")

    monkeypatch.setattr(e, "_emit", fail)
    with pytest.raises(RuntimeError, match="presentation-source"):
        e.present_cancelled_itinerary(passenger.original, airport_counter=True)
    assert passenger.original.record_id in e._presented
    snap = e.snapshot()
    assert snap.failed and len(snap.periods) == 1
    assert len(e._proof_calls) == 1
    with pytest.raises(ValueError):
        e.causal_periods()
    with pytest.raises(ValueError):
        e.step(actions)


def test_new_full_parent_input_support_checked_after_native_close(monkeypatch):
    e, _p, actions = source()
    original = e._finish_causal_period

    def omit(period):
        e._causal_pending.pop()
        return original(period)

    monkeypatch.setattr(e, "_finish_causal_period", omit)
    with pytest.raises(RuntimeError, match="ALL actual parent"):
        e.step(actions)
    assert e.world.t == 1 and not e.snapshot().periods


def test_new_initial_source_limit_and_terminal_guards():
    with pytest.raises((module.NativeCausalInformationLimit, CausalInformationLimit)):
        source(row_limit=1)
    e, _p, actions = source(count=0)
    for _ in range(5):
        e.step(actions)
    before = pickle.dumps(e.snapshot())
    with pytest.raises(ValueError, match="nonterminal"):
        e.step(actions)
    assert pickle.dumps(e.snapshot()) == before
