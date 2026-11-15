"""A.10/A.5/A.14/B0/V9: NEW actual native taste receipts, no old fixture calls."""

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

from airline_market.core.hashing import jsonable
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
from airline_market.sim.causal_information import causal_decision_view
from airline_market.sim.native_causal_information import (
    DeclaredKnownType,
    NativeCausalInformationMarketEngine,
)
from airline_market.sim.rebooking import RebookingMarketEngine
from airline_market.sim.received_information import DisclosureField
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

if os.environ.get("PRIVATE_TASTE_SOURCE_DIR"):
    folder = Path(os.environ["PRIVATE_TASTE_SOURCE_DIR"])
    sp = importlib.util.spec_from_file_location(
        "airline_market.sim.native_private_taste", folder / "native_private_taste.py"
    )
    loaded = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = loaded
    sp.loader.exec_module(loaded)
from airline_market.sim import native_private_taste as module


def source(*, route="base", mixed=False, interline=True, outside=False, count=2):
    # NEW model/date/namespace/primitive arrays; adapted constructor, not old helper.
    spec = screen_spec(
        {
            "T": 4,
            "actions": 2,
            "n_spokes": 1,
            "capacity": 20,
            "capacity_support": [3, 4],
        }
    )
    segments = tuple(
        replace(
            segment,
            classes=(
                replace(segment.classes[0], weight=0.25),
                replace(
                    segment.classes[0], weight=0.75, alpha=segment.classes[0].alpha * 2
                ),
            ),
            mixed={"log_alpha_sd": 0.2, "carrier_pref_sd": 0.3} if mixed else None,
        )
        for segment in spec.segments
    )
    c = compile_market(
        replace(
            spec,
            segments=segments,
            choice=replace(
                spec.choice,
                model="mixed_logit" if mixed else "latent_class_mnl",
                n_mixed_draws=4 if mixed else 0,
            ),
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
    start = datetime(2027, 2, 8, tzinfo=timezone.utc)
    clock = UtcBookingClock(tuple(start + timedelta(hours=12 * t) for t in range(5)))
    departures = tuple(
        (leg.id, start + timedelta(hours=84 if leg.id == "B_OH0" else 108))
        for leg in c.spec.legs
    )
    scenario = replace(
        compile_dated_demand(c, clock, departures),
        physical_shocks=(
            B1DatedShock(
                "new_private_exit", "rival_exit", "B_OH0", 3, rival_carrier="B"
            ),
        ),
    )
    kwargs = {"dated_rebooking": DatedRebookingPolicy(72, 20, 100, interline=interline)}
    if route != "base":
        kwargs["dated_menu"] = DatedRescueMenu(
            UtcWindow(clock.at(0), clock.at(4)),
            "same_route",
            route_fares=tuple((m.id, 99) for m in c.spec.markets),
        )
    if route in {"reserve", "noheld", "zero"}:
        kwargs["dated_reserve"] = DatedSeatReserve(
            clock.at(1),
            Decimal("0" if route == "zero" else ".1"),
            ("A_HD0" if route == "noheld" else "A_OH0",),
            departures,
        )
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(89974, "unit/V9_native_private_taste")
    )
    target = c.product_index["B_OH0"]
    market = int(c.product_market[target])
    marks = tuple(
        ArrivalMark(
            time=(i + 1) / (count + 1),
            market=market,
            segment=0,
            thinning=0,
            class_uniform=0.99,
        )
        for i in range(count)
    )
    g = tape.gumbel.copy()
    g[:] = -1e9
    g[:, :, -1] = 0
    g[:, :, target] = 1e9
    if count:
        g[:, count - 1, -1] = 2e9
    if outside:
        g[:, :, -1] = 2e9
    g.setflags(write=False)
    tape = replace(tape, arrival_marks=(marks, (), (), ()), gumbel=g)
    causal = {
        "causal_fields": (
            DisclosureField("notice", "text", "unit declaration", "text"),
        ),
        "causal_known_types": tuple(
            DeclaredKnownType(
                (0, i, market),
                None if i == 0 else "unrelated-label",
                "explicit unknown or arbitrary label",
                "new-unit",
            )
            for i in range(count)
        ),
        "causal_deliveries": (),
        "causal_definition": "new genuine private-taste source",
        "causal_unobserved_definition": "no inferred passenger knowledge",
    }
    e = module.NativePrivateTasteMarketEngine(scenario, **causal, **kwargs)
    p = NativeCausalInformationMarketEngine(scenario, **causal, **kwargs)
    for engine in (p, e):
        engine.reset(tape=tape)
    actions = {
        carrier.code: PriceAction({j: 200 if i == 0 else 300 for j in c.products_of[i]})
        for i, carrier in enumerate(c.spec.carriers)
    }
    return e, p, actions


def advance(pair, steps=4):
    e, p, actions = pair
    for t in range(steps):
        for engine in (p, e):
            if t == 1:
                for passenger in engine._passengers.values():
                    engine.present_cancelled_itinerary(
                        passenger.original, airport_counter=True
                    )
            engine.step(actions)
    return e, p


@pytest.mark.parametrize("route", ["base", "rescue", "reserve", "noheld", "zero"])
def test_new_all_routes_parent_world_trace_metrics_information_identity(route):
    before = pickle.dumps(np.random.get_state())
    e, p = advance(source(route=route))
    assert pickle.dumps(e.snapshot().native) == pickle.dumps(p.snapshot())
    assert e._payload() == p._payload()
    assert pickle.dumps(e.trace()) == pickle.dumps(p.trace())
    assert pickle.dumps(e.metrics()) == pickle.dumps(p.metrics())
    assert pickle.dumps(np.random.get_state()) == before
    periods = e.private_taste_periods()
    assert len(periods) == 4 and periods[0].rows == ()
    assert len(periods[1].rows) == 1
    row = periods[1].rows[0]
    assert row.original_booking.taste_class == 1
    assert row.original_booking.segment == 0
    assert row.analyst_original_key == e._own_keys[row.analyst_root]
    assert all(p.native_world_t == p.period + 1 for p in periods)
    record = module.native_private_taste_record(e)
    assert record["actual_returned_taste_rows_captured"]
    assert not record["gumbel_values_inspected_or_exported"]
    assert record["lifecycle_utility_CS_service_payoff"] is None


@pytest.mark.parametrize("mixed", [False, True])
def test_new_actual_return_identity_once_and_same_stage_utility_arguments(
    monkeypatch, mixed
):
    import airline_market.sim.reentry_stage_executor as executor

    e, _, actions = source(mixed=mixed)
    returned, utility_tastes = [], []
    actual_marks = RebookingMarketEngine._marks
    actual_utility = executor.systematic_utilities

    def marks(engine, passenger, period):
        result = actual_marks(engine, passenger, period)
        returned.append(result)
        return result

    def utility(c, market, taste, prices, active):
        utility_tastes.append(taste)
        return actual_utility(c, market, taste, prices, active)

    monkeypatch.setattr(RebookingMarketEngine, "_marks", marks)
    monkeypatch.setattr(executor, "systematic_utilities", utility)
    e.step(actions)
    e.step(actions)
    rows = e.private_taste_periods()[1].rows
    assert len(returned) == len(rows) == 1
    assert len(utility_tastes) == len(e._stage_periods[1].inputs) == 2
    assert all(t is returned[0][0] for t in utility_tastes)
    assert jsonable(rows[0].resolved_taste) == jsonable(returned[0][0])


@pytest.mark.parametrize("mixed", [False, True])
def test_new_repeated_failed_searches_persist_class_and_actual_taste(mixed):
    e, _ = advance(source(mixed=mixed, interline=False))
    rows = [r for p in e.private_taste_periods() for r in p.rows]
    assert len(rows) == 3
    assert all(r.original_booking.taste_class == 1 for r in rows)
    assert all(r.resolved_taste == rows[0].resolved_taste for r in rows)
    assert [r.native_period for r in rows] == [1, 2, 3]
    assert all(len(p.inputs) == 1 for p in e._stage_periods[1:])
    # This is repeated re-entry parameter persistence, not original mixed-taste identity.
    assert not module.native_private_taste_record(e)[
        "ordinary_resolved_mixed_taste_captured"
    ]


def test_new_actual_private_type_never_becomes_declared_known_label_or_view():
    e, _ = advance(source(interline=False))
    for p in e.causal_periods():
        for row in p.inputs:
            view = causal_decision_view(row.receipt)
            if row.analyst_original_key[1] == 0:
                assert view["known_type"]["value"] is None
            assert "resolved_taste" not in json.dumps(jsonable(view))
            assert "taste_class" not in json.dumps(jsonable(view))
    assert e.private_taste_periods()[1].rows[0].original_booking.taste_class == 1


@pytest.mark.parametrize("count,outside", [(0, False), (2, True)])
def test_new_empty_and_outside_people_retained_without_invented_taste_rows(
    count, outside
):
    e, p = advance(source(count=count, outside=outside))
    assert e._payload() == p._payload()
    assert not any(p.rows for p in e.private_taste_periods())
    assert not module.native_private_taste_record(e)[
        "actual_returned_taste_rows_captured"
    ]
    assert len(e._information_periods[0].rows) == count


def test_new_snapshot_independent_branch_reset_and_owned_export():
    e, _, actions = source(interline=False, mixed=True)
    e.step(actions)
    branch = e.branch()
    for engine in (e, branch):
        engine.step(actions)
    assert module.native_private_taste_record(e) == module.native_private_taste_record(
        branch
    )
    e.step(actions)
    assert len(branch.private_taste_periods()) == 2
    record = module.native_private_taste_record(branch)
    record["periods"][1]["rows"][0]["resolved_taste"]["alpha"] = "changed"
    assert module.native_private_taste_record(branch) != record
    with pytest.raises(TypeError):
        branch.private_taste_periods()[1].rows[0].resolved_taste["alpha"] = 0
    tape = branch.tape
    branch.reset(tape=tape)
    assert branch.private_taste_periods() == ()
    assert len(e.private_taste_periods()) == 3


@pytest.mark.parametrize("where", ["capture", "post_close"])
def test_new_failure_retains_pending_native_state_without_positive_query(
    monkeypatch, where
):
    e, _, actions = source()
    e.step(actions)
    name = "_capture_private_taste" if where == "capture" else "_finish_private_period"

    def fail(*args):
        raise RuntimeError("new injected private recorder failure")

    monkeypatch.setattr(e, name, fail)
    with pytest.raises(RuntimeError, match="injected"):
        e.step(actions)
    snap = e.snapshot()
    assert snap.failed and snap.pending is not None
    assert len(snap.periods) == 1
    assert e.world.t == (1 if where == "capture" else 2)
    with pytest.raises(ValueError):
        module.native_private_taste_record(e)
    with pytest.raises(ValueError):
        e.step(actions)


def test_new_limit_before_capture_append_and_forced_hooks_refused(monkeypatch):
    e, _, actions = source()
    with pytest.raises(ValueError):
        e.step(actions, _requests=[])
    e.step(actions)
    check = e._check_private_limit

    def limit(proposed=None):
        if type(proposed) is module.NativePrivateTaste:
            raise module.NativePrivateTasteLimit("new proposed row limit")
        return check(proposed)

    monkeypatch.setattr(e, "_check_private_limit", limit)
    with pytest.raises(module.NativePrivateTasteLimit):
        e.step(actions)
    assert e.snapshot().pending == ()
    assert len(e.snapshot().periods) == 1


def test_new_full_source_limit_and_snapshot_contract():
    e, _, actions = source()
    e.private_row_limit = 1
    with pytest.raises(module.NativePrivateTasteLimit):
        e.step(actions)
    with pytest.raises(ValueError):
        e.restore(replace(e.snapshot(), row_limit=2))
    with pytest.raises(TypeError):
        module.native_private_taste_record(object())


def test_new_actual_finite_native_transform_matches_booked_class_and_not_label():
    e, _ = advance(source(interline=False))
    row = e.private_taste_periods()[1].rows[0]
    b = row.original_booking
    base = e.compiled.spec.segments[b.segment].classes[b.taste_class]
    assert b.taste_class == 1
    assert row.resolved_taste["alpha"] == base.alpha / e.rebooking.wtp_multiplier
    expected = {
        k: v + e.rebooking.utility_bonus
        for k, v in {"*": 0.0, **base.market_value}.items()
    }
    assert row.resolved_taste["market_value"] == expected
    assert row.resolved_taste["carrier_pref"] == base.carrier_pref


def test_new_declared_scalar_export_signed_zero_nonfinite_is_not_native_witness():
    from airline_market.sim.reentry_stage_information import _stage_encode

    # Synthetic envelopes only, NOT native-produced invalid utility/taste values.
    row = module.NativePrivateTaste(
        0,
        (0, 0, 0),
        1,
        None,
        {"alpha": -0.0, "declared_unknown": None, "declared_nonfinite": float("inf")},
    )
    encoded = _stage_encode(jsonable(row))
    assert encoded["resolved_taste"]["alpha"]["float_hex"] == "-0x0.0p+0"
    assert encoded["resolved_taste"]["declared_unknown"] is None
    assert encoded["resolved_taste"]["declared_nonfinite"]["exact"] is None
