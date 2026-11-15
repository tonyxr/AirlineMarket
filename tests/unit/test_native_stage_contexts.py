"""A.10/A.5/A.14/B0/V9: NEW native stage/attempt/source binding references."""

import copy
import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
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
from airline_market.sim.native_information_context_v2 import native_information_contexts
from airline_market.sim.received_information import (
    DisclosureField,
    received_information_journal,
)
from airline_market.sim.reentry_stage_information import (
    ReentryStageInformationMarketEngine,
    reentry_stage_record,
)
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

if os.environ.get("NATIVE_STAGE_SOURCE_DIR"):
    sp = importlib.util.spec_from_file_location(
        "airline_market.sim.native_stage_contexts",
        Path(os.environ["NATIVE_STAGE_SOURCE_DIR"]) / "native_stage_contexts.py",
    )
    loaded = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = loaded
    sp.loader.exec_module(loaded)

from airline_market.sim import native_stage_contexts as module


def source(
    *,
    route="base",
    interline=True,
    weather=False,
    outside=False,
    count=2,
    row_limit=1000000,
    connection=False,
):
    # Locally adapted constructor, NEW namespace/date/marks; no old fixture call.
    spec = screen_spec(
        {
            "T": 5,
            "actions": 2,
            "n_spokes": 1,
            "capacity": 10 if weather else 20,
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
    start = datetime(2026, 12, 27, tzinfo=timezone.utc)
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
    e = ReentryStageInformationMarketEngine(
        scenario, stage_row_limit=row_limit, **kwargs
    )
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(84974, "unit/V9_native_stage_contexts")
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
    e.reset(tape=tape)
    actions = {
        carrier.code: PriceAction({j: 200 if i == 0 else 300 for j in c.products_of[i]})
        for i, carrier in enumerate(c.spec.carriers)
    }
    return e, actions


def inputs(*, steps=5, proof=True, **kwargs):
    e, actions = source(**kwargs)
    for t in range(steps):
        if t == 1 and proof:
            for passenger in e._passengers.values():
                e.present_cancelled_itinerary(passenger.original, airport_counter=True)
        e.step(actions)
    clock = e.demand_scenario.clock
    journal = received_information_journal(
        e.compiled,
        e.information_periods(),
        clock,
        (DisclosureField("notice", "text", "Explicit declared notice schema", "text"),),
        definition="NEW genuine complete native-stage source",
        unobserved_definition="No received knowledge inferred",
        row_limit=1000000,
    )
    native = native_information_contexts(
        journal,
        e.own_information_periods(),
        e.proof_information_periods(),
        e.snapshot().native.native.histories,
        e.presentation_calls(),
        definition="NEW prospective native own/proof contexts",
        row_limit=1000000,
    )
    return e, native, reentry_stage_record(e)


def bind(native, stage, limit=2000000):
    return module.bind_native_stage_contexts(
        native,
        stage,
        definition="NEW complete native computation contexts, knowledge/law open",
        row_limit=limit,
    )


@pytest.mark.parametrize("route", ["base", "rescue", "reserve", "noheld", "zero"])
def test_new_complete_native_routes_attempt_stage_join(route):
    e, native, stage = inputs(route=route)
    before = pickle.dumps(e.snapshot())
    rng = pickle.dumps(np.random.get_state())
    result = bind(native, stage)
    assert len(result.people) == 2  # genuine booked person AND ordinary outside
    assert len(result.periods) == 5
    rows = result.periods[1]["reentry_stage_contexts"]
    assert len(rows) == 2
    assert rows[0]["person_decision"] == rows[1]["person_decision"] == 1
    assert rows[0]["view"]["current_native_stage_choice_inputs"]["stage"] == "own"
    assert rows[1]["view"]["current_native_stage_choice_inputs"]["stage"] == "interline"
    assert rows[1]["view"]["current_native_stage_choice_inputs"][
        "earlier_within_attempt_results"
    ] == ({"stage": "own", "outside": True},)
    assert (
        rows[0]["view"]["earlier_native_attempt_context"]
        == rows[1]["view"]["earlier_native_attempt_context"]
    )
    assert pickle.dumps(e.snapshot()) == before
    assert pickle.dumps(np.random.get_state()) == rng
    record = module.native_stage_context_record(result)
    json.dumps(record, allow_nan=False)
    assert record["lifecycle_utility_CS_service_or_payoff"] is None
    assert not record[
        "actual_received_information_type_law_preferences_or_utility_authenticated"
    ]


@pytest.mark.parametrize(
    "kwargs",
    [{"count": 0}, {"outside": True}, {"interline": False}, {"steps": 0}, {"steps": 1}],
)
def test_new_empty_outside_disabled_zero_partial_prefix_support(kwargs):
    _e, native, stage = inputs(**kwargs)
    result = bind(native, stage)
    assert len(result.people) == len(native.native.people)
    assert len(result.periods) == len(native.native.periods)
    if kwargs.get("interline") is False:
        rows = [r for p in result.periods for r in p["reentry_stage_contexts"]]
        assert rows and all(r["stage_ordinal"] == 0 for r in rows)
    else:
        assert not any(p["reentry_stage_contexts"] for p in result.periods)


def test_new_connection_and_own_purchase_stop_on_same_original_people():
    _e, native, stage = inputs(connection=True, route="reserve")
    result = bind(native, stage)
    assert len(result.periods[1]["reentry_stage_contexts"]) == 2
    _e, native, stage = inputs(weather=True, interline=False, count=6)
    result = bind(native, stage)
    rows = result.periods[3]["reentry_stage_contexts"]
    assert rows and all(r["stage_ordinal"] == 0 for r in rows)
    assert len(result.people) == 6


@pytest.mark.parametrize(
    "change",
    [
        "clock",
        "ordinary",
        "proof",
        "fact",
        "ticket",
        "key",
        "model",
        "period",
        "pending",
    ],
)
def test_new_full_source_association_refusals(change):
    _e, native, stage = inputs()
    changed = copy.deepcopy(stage)
    src = changed["source_context"]
    if change == "clock":
        src["scenario"]["clock_utc"] = src["scenario"]["clock_utc"][:-1]
    elif change == "ordinary":
        src["ordinary_information"] = src["ordinary_information"][1:]
    elif change == "proof":
        src["proof_information"] = src["proof_information"][:-1]
    elif change in {"fact", "ticket", "key"}:
        name = {
            "fact": "final_own_histories",
            "ticket": "own_tickets",
            "key": "own_keys",
        }[change]
        src[name] = {}
    elif change == "model":
        src["native_spec"]["products"][0]["carrier"] = "different"
    elif change == "period":
        changed["periods"] = changed["periods"][:-1]
    else:
        changed["pending_results"] = ()
    before = copy.deepcopy(changed)
    with pytest.raises(ValueError):
        bind(native, changed)
    assert changed == before


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "reverse",
        "root",
        "key",
        "ordinal",
        "mask",
        "fare",
        "result",
        "result_index",
        "private",
        "earlier_outcome",
    ],
)
def test_new_stage_reach_same_call_and_privacy_refusals(change):
    _e, native, stage = inputs()
    changed = copy.deepcopy(stage)
    period = changed["periods"][1]
    rows, results = list(period["inputs"]), list(period["results"])
    if change == "missing":
        rows, results = rows[:-1], results[:-1]
    elif change == "duplicate":
        rows.append(copy.deepcopy(rows[-1]))
        results.append(copy.deepcopy(results[-1]))
    elif change == "reverse":
        rows.reverse()
        results.reverse()
    elif change == "root":
        rows[0]["analyst_root"] += 1
    elif change == "key":
        rows[0]["analyst_original_key"] = (0, 999, 0)
    elif change == "ordinal":
        rows[0]["stage_ordinal"] = 1
    elif change == "mask":
        flags = list(rows[0]["offered"])
        # Original B product must never be offered in the interline stage.
        j = next(
            i
            for i, p in enumerate(changed["source_context"]["native_spec"]["products"])
            if p["id"] == "B_OH0"
        )
        flags = list(rows[1]["offered"])
        flags[j] = True
        rows[1]["offered"] = tuple(flags)
    elif change == "fare":
        rows[0]["view"]["offers"][0]["fare"]["repr"] = "999"
    elif change == "result":
        results[0]["selected_product"] = results[1]["selected_product"]
        results[0]["outside"] = False
    elif change == "result_index":
        results[1]["input_index"] = 0
    elif change == "private":
        rows[0]["view"]["private_class"] = 0
    else:
        rows[0]["view"]["earlier_within_attempt_results"] = (
            {"stage": "own", "outside": True},
        )
    period["inputs"], period["results"] = tuple(rows), tuple(results)
    with pytest.raises(ValueError):
        bind(native, changed)


def test_new_owned_views_limits_and_no_native_or_upstream_calls(monkeypatch):
    e, native, stage = inputs(route="reserve")
    from airline_market.sim import (
        native_information_context_v2,
        received_information,
        reentry_stage_information,
    )

    def refused(*_args, **_kwargs):
        raise AssertionError("pure binder must not replay a producer")

    for target, name in [
        (native_information_context_v2, "native_information_contexts"),
        (reentry_stage_information, "reentry_stage_record"),
        (received_information, "received_information_journal"),
    ]:
        monkeypatch.setattr(target, name, refused)
    monkeypatch.setattr(e, "step", refused)
    result = bind(native, stage)
    export = module.native_stage_context_record(result)
    export["result"]["definition"] = "changed external copy"
    assert result.definition != "changed external copy"
    with pytest.raises(TypeError):
        result.periods[1]["reentry_stage_contexts"][0]["view"]["new"] = "bad"
    before = pickle.dumps((native, stage))
    with pytest.raises(module.NativeStageContextLimit):
        bind(native, stage, limit=1)
    # Full supplied sources fit, but proposed combined output must ALSO fit.
    source_size = module._units(result.analyst_context)
    with pytest.raises(module.NativeStageContextLimit):
        bind(native, stage, limit=source_size)
    assert pickle.dumps((native, stage)) == before
    assert "selected_product" not in json.dumps(
        module._encode(result.periods[1]["reentry_stage_contexts"][0]["view"])
    )


@pytest.mark.parametrize(
    "change",
    [
        "flag",
        "contract",
        "route",
        "outcome_fare",
        "boolean_integer",
        "parent_people",
        "parent_view",
    ],
)
def test_new_full_metadata_parent_context_and_actual_outcome_refusals(change):
    _e, native, stage = inputs(route="reserve")
    if change == "flag":
        stage["native_reentry_stage_inputs_captured"] = False
    elif change == "contract":
        del stage["source_context"]["dated_reserve"]
    elif change == "route":
        stage["periods"][1]["inputs"][0]["route"] = "base"
    elif change == "outcome_fare":
        row = stage["periods"][1]["inputs"][1]
        j = stage["periods"][1]["results"][1]["selected_product"]
        prices = list(row["prices"])
        prices[j] = module._binary(101.0)
        row["prices"] = tuple(prices)
        pid = stage["source_context"]["native_spec"]["products"][j]["id"]
        for offer in row["view"]["offers"]:
            if offer["product_id"] == pid:
                offer["fare"] = prices[j]
    elif change == "boolean_integer":
        stage["periods"][1]["inputs"][0]["view"]["outside_available"] = 1
    elif change == "parent_people":
        native = replace(native, native=replace(native.native, people=()))
    else:
        periods = list(native.native.periods)
        attempts = list(periods[1].reentry_contexts)
        view = module._encode(attempts[0].view)
        view["known_private_type"] = "injected private class"
        attempts[0] = replace(attempts[0], view=view)
        periods[1] = replace(periods[1], reentry_contexts=tuple(attempts))
        native = replace(native, native=replace(native.native, periods=tuple(periods)))
    with pytest.raises(ValueError):
        bind(native, stage)
