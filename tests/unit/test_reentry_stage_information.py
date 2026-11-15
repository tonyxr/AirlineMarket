"""A.10/A.5/A.14/B0/V9: NEW genuine stage-source identity and reach checks."""

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
from airline_market.sim.proof_information import ProofInformationMarketEngine
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

if os.environ.get("STAGE_SOURCE_DIR"):
    folder = Path(os.environ["STAGE_SOURCE_DIR"])
    for name in ["reentry_stage_executor", "reentry_stage_information"]:
        sp = importlib.util.spec_from_file_location(
            "airline_market.sim." + name, folder / (name + ".py")
        )
        loaded = importlib.util.module_from_spec(sp)
        sys.modules[sp.name] = loaded
        sp.loader.exec_module(loaded)

from airline_market.sim import reentry_stage_executor as executor
from airline_market.sim import reentry_stage_information as module


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
    start = datetime(2026, 12, 20, tzinfo=timezone.utc)
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
    e = module.ReentryStageInformationMarketEngine(
        scenario, stage_row_limit=row_limit, **kwargs
    )
    parent = ProofInformationMarketEngine(scenario, **kwargs)
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(83974, "unit/V9_reentry_stage_information")
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
    for engine in [e, parent]:
        engine.reset(tape=tape)
    actions = {
        carrier.code: PriceAction({j: 200 if i == 0 else 300 for j in c.products_of[i]})
        for i, carrier in enumerate(c.spec.carriers)
    }
    return e, parent, actions


def advance(pair, *, proof=True, steps=5):
    e, parent, actions = pair
    for t in range(steps):
        for engine in [parent, e]:
            if t == 1 and proof:
                for passenger in engine._passengers.values():
                    engine.present_cancelled_itinerary(
                        passenger.original, airport_counter=True
                    )
            engine.step(actions)
    return e, parent


@pytest.mark.parametrize(
    "route,expected",
    [
        ("base", "base"),
        ("rescue", "rescue"),
        ("reserve", "reserve"),
        ("noheld", "reserve"),
        ("zero", "rescue"),
    ],
)
def test_new_native_routes_full_identity_and_final_price_restore(route, expected):
    rng = pickle.dumps(np.random.get_state())
    e, parent = advance(source(route=route))
    assert pickle.dumps(e.snapshot().native) == pickle.dumps(parent.snapshot())
    assert pickle.dumps(e.trace()) == pickle.dumps(parent.trace())
    assert pickle.dumps(e.metrics()) == pickle.dumps(parent.metrics())
    assert pickle.dumps(np.random.get_state()) == rng
    stages = e.stage_periods()[1]
    assert len(stages.inputs) == 2 and all(r.route == expected for r in stages.inputs)
    own, interline = stages.inputs
    assert own.view["stage"] == "own" and interline.view["stage"] == "interline"
    assert own.view["earlier_within_attempt_results"] == ()
    assert interline.view["earlier_within_attempt_results"][0]["outside"]
    assert stages.results[0].outside and not stages.results[1].outside
    fare = next(
        o["fare"] for o in interline.view["offers"] if o["product_id"] == "A_OH0"
    )
    assert fare == (200 if route in {"base", "noheld"} else 99)
    assert parent.rebooking_history[0]["fare"] == fare
    assert len(e.stage_periods()) == 5 and not e.stage_periods()[0].inputs
    json.dumps(module.reentry_stage_record(e), allow_nan=False)


def test_new_own_purchase_stops_without_counterfactual_interline_stage():
    e, parent = advance(source(weather=True, interline=False, count=10))
    period = e.stage_periods()[3]
    assert period.inputs and all(r.view["stage"] == "own" for r in period.inputs)
    assert all(not r.outside for r in period.results)
    assert pickle.dumps(e.snapshot().native) == pickle.dumps(parent.snapshot())


@pytest.mark.parametrize(
    "kwargs", [{"count": 0}, {"outside": True}, {"interline": False}]
)
def test_new_empty_outside_and_disabled_stage_support(kwargs):
    e, _parent = advance(source(**kwargs))
    if kwargs.get("interline") is False:
        assert all(
            row.view["stage"] == "own" for p in e.stage_periods() for row in p.inputs
        )
        assert all(r.outside for p in e.stage_periods() for r in p.results)
    else:
        assert not any(p.inputs for p in e.stage_periods())
    record = module.reentry_stage_record(e)
    assert len(record["source_context"]["ordinary_information"]) == 5


def test_new_same_call_array_arguments_no_extra_native_choice_or_availability(
    monkeypatch,
):
    e, parent, actions = source(route="reserve")
    counts = {"parent": {}, "new": {}}
    current = ["parent"]
    from airline_market.policy import rescue, reserve
    from airline_market.sim import rebooking

    for name in ["systematic_utilities", "sampled_choice"]:
        original = getattr(executor, name)

        def spy(*args, _name=name, _original=original):
            d = counts[current[0]]
            d[_name] = d.get(_name, 0) + 1
            if (
                current[0] == "new"
                and _name == "systematic_utilities"
                and e._pending_stage_inputs
            ):
                row = e._pending_stage_inputs[-1]
                if len(e._pending_stage_inputs) > len(e._pending_stage_results):
                    assert row.prices == tuple(float(x) for x in args[3])
                    assert row.offered == tuple(bool(x) for x in args[4])
            return _original(*args)

        for target in [executor, rescue, reserve, rebooking]:
            monkeypatch.setattr(target, name, spy)
    for engine, label in [(parent, "parent"), (e, "new")]:
        for name in ["_sellable_global", "_marks", "_rescue_prices"]:
            original = getattr(engine, name)

            def spy(*args, _name=name, _original=original):
                d = counts[current[0]]
                d[_name] = d.get(_name, 0) + 1
                return _original(*args)

            monkeypatch.setattr(engine, name, spy)
        for t in range(3):
            current[0] = label
            if t == 1:
                for passenger in engine._passengers.values():
                    engine.present_cancelled_itinerary(passenger.original)
            engine.step(actions)
    assert counts["new"] == counts["parent"]
    assert pickle.dumps(e.snapshot().native) == pickle.dumps(parent.snapshot())


def test_new_narrow_views_owned_snapshots_branches_reset_and_contract():
    pair = source(route="reserve")
    e, _parent = advance(pair, steps=2)
    row = e.stage_periods()[1].inputs[1]
    view = module.reentry_stage_view(row)
    assert not {
        "analyst_root",
        "analyst_original_key",
        "prices",
        "offered",
        "taste",
        "gumbels",
        "class",
        "selected_product",
    } & set(view)
    view["offers"][0]["fare"] = 999
    assert module.reentry_stage_view(row)["offers"][0]["fare"] != 999
    saved = pickle.dumps(e.snapshot())
    b = e.branch()
    assert pickle.dumps(b.snapshot()) == saved
    b.step(pair[2])
    assert pickle.dumps(e.snapshot()) == saved
    e.reset(tape=e.tape)
    assert e.stage_periods() == ()
    e.restore(pickle.loads(saved))
    from airline_market.sim.quotes import QuoteHistory

    def values(snapshot):
        def owned(value):
            if isinstance(value, QuoteHistory):
                return owned(
                    module.jsonable(
                        {"records": value.records, "initial": value.initial}
                    )
                )
            if isinstance(value, dict):
                return {k: owned(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return tuple(owned(v) for v in value)
            return module._stage_encode(value)

        return owned(module.jsonable(snapshot))

    assert values(e.snapshot()) == values(pickle.loads(saved))
    bad = replace(e.snapshot(), row_limit=e.stage_row_limit + 1)
    with pytest.raises(ValueError, match="contract"):
        e.restore(bad)
    with pytest.raises(TypeError):
        row.view["stage"] = "mutated"


@pytest.mark.parametrize("where", ["input", "result", "close"])
def test_new_failures_retain_pending_mutations_without_positive_receipt_or_retry(
    where, monkeypatch
):
    pair = source(route="reserve")
    e, _parent = advance(pair, steps=1)

    def fail(*args):
        raise RuntimeError("NEW injected stage capture failure")

    name = {
        "input": "_capture_stage_input",
        "result": "_capture_stage_result",
        "close": "_finish_stage_period",
    }[where]
    monkeypatch.setattr(e, name, fail)
    with pytest.raises(RuntimeError, match="injected"):
        e.step(pair[2])
    assert e.snapshot().pending_inputs is not None
    assert len(e.snapshot().periods) == 1
    with pytest.raises(ValueError, match="partial"):
        e.stage_periods()
    with pytest.raises(ValueError, match="retry"):
        e.step(pair[2])


def test_new_structural_limits_and_forced_hook_refusal_before_mutation():
    e, _parent, actions = source(row_limit=1)
    saved = pickle.dumps(e.snapshot())
    with pytest.raises(module.StageInformationLimit):
        e.step(actions)
    assert pickle.dumps(e.snapshot()) == saved
    with pytest.raises(ValueError, match="genuine"):
        e.step(actions, _choices=[0])
    assert pickle.dumps(e.snapshot()) == saved


def test_new_contract_export_keeps_aware_dates_decimal_fraction_and_binary64():
    from fractions import Fraction

    value = {
        "time": datetime(2026, 12, 20, tzinfo=timezone.utc),
        "share": Decimal("0.10"),
        "tiny": Fraction(1, 10**500),
        "minus_zero": -0.0,
        "unknown": None,
        "nonfinite": float("inf"),
    }
    out = module._stage_encode(module.jsonable(value))
    assert out["share"]["decimal"] == "0.10"
    assert out["tiny"]["denominator"] == str(10**500)
    assert out["minus_zero"]["binary64_bits"] == "8000000000000000"
    assert out["nonfinite"]["exact"] is None and out["unknown"] is None
    assert out["time"]["aware_utc"] == "2026-12-20T00:00:00+00:00"
    json.dumps(out, allow_nan=False)


def test_new_postclose_limit_keeps_only_prior_complete_receipts(monkeypatch):
    pair = source(route="reserve")
    e, _ = advance(pair, steps=1)
    original = e._check_stage_limit

    def fail(proposed=None):
        if proposed is not None:
            raise module.StageInformationLimit("NEW injected postclose proposal limit")
        return original()

    monkeypatch.setattr(e, "_check_stage_limit", fail)
    with pytest.raises(module.StageInformationLimit, match="postclose"):
        e.step(pair[2])
    snap = e.snapshot()
    assert e.world.t == 2 and len(snap.periods) == 1 and snap.pending_inputs
    with pytest.raises(ValueError, match="partial"):
        e.stage_periods()


def test_new_connection_final_stage_source_is_same_original_market():
    e, parent = advance(source(route="reserve", connection=True))
    row = e.stage_periods()[1].inputs[-1]
    selected = e.stage_periods()[1].results[-1].selected_product
    product = e.compiled.spec.products[selected]
    assert len(product.legs) == 2 and product.market == row.view["market_id"]
    assert row.view["stage"] == "interline"
    assert pickle.dumps(e.snapshot().native) == pickle.dumps(parent.snapshot())


def test_new_no_proof_does_not_manufacture_rescue_fare_or_held_entitlement():
    e, parent = advance(source(route="reserve"), proof=False)
    row = e.stage_periods()[1].inputs[-1]
    offer = next(o for o in row.view["offers"] if o["product_id"] == "A_OH0")
    assert offer["fare"] == 200
    assert parent.rebooking_history[0]["rescue_eligible"] is False
    assert parent.rebooking_history[0]["held_seats_used"] == {}
    assert pickle.dumps(e.snapshot().native) == pickle.dumps(parent.snapshot())


def test_new_export_capture_flag_counts_actual_rows_including_zero_prefix():
    pair = source()
    e = pair[0]
    assert (
        module.reentry_stage_record(e)["native_reentry_stage_inputs_captured"] is False
    )
    advance(pair, steps=2)
    assert (
        module.reentry_stage_record(e)["native_reentry_stage_inputs_captured"] is True
    )
    empty, _ = advance(source(count=0))
    assert (
        module.reentry_stage_record(empty)["native_reentry_stage_inputs_captured"]
        is False
    )
