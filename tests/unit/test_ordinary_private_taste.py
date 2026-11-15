"""A.10/A.5/A.14/B0/V9: NEW ordinary-return source fixtures only."""

import ast
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
from airline_market.sim.native_causal_information import DeclaredKnownType
from airline_market.sim.native_private_taste import NativePrivateTasteMarketEngine
from airline_market.sim.received_information import DisclosureField
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

if os.environ.get("ORDINARY_TASTE_SOURCE_DIR"):
    folder = Path(os.environ["ORDINARY_TASTE_SOURCE_DIR"])
    for name in ["ordinary_taste_executor", "ordinary_private_taste"]:
        sp = importlib.util.spec_from_file_location(
            "airline_market.sim." + name, folder / (name + ".py")
        )
        loaded = importlib.util.module_from_spec(sp)
        sys.modules[sp.name] = loaded
        sp.loader.exec_module(loaded)
from airline_market.sim import ordinary_private_taste as module
from airline_market.sim import ordinary_taste_executor as executor


def source(
    *,
    route="base",
    mixed=False,
    interline=True,
    outside=False,
    count=2,
    rebooking=True,
    early_exit=False,
):
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
    start = datetime(2027, 2, 15, tzinfo=timezone.utc)
    clock = UtcBookingClock(tuple(start + timedelta(hours=12 * t) for t in range(5)))
    departures = tuple(
        (
            leg.id,
            start
            + timedelta(hours=(72 if early_exit else 84) if leg.id == "B_OH0" else 108),
        )
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
    if not rebooking:
        kwargs = {}
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
        c, 31, registry=StreamRegistry(90974, "unit/V9_ordinary_private_taste")
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
    g[:] = -1000000000.0
    g[:, :, -1] = 0
    g[:, :, target] = 1000000000.0
    if count:
        g[:, count - 1, -1] = 2000000000.0
    if outside:
        g[:, :, -1] = 2000000000.0
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
    e = module.OrdinaryPrivateTasteMarketEngine(scenario, **causal, **kwargs)
    p = NativePrivateTasteMarketEngine(scenario, **causal, **kwargs)
    for engine in (p, e):
        engine.reset(tape=tape)
    actions = {
        carrier.code: PriceAction({j: 200 if i == 0 else 300 for j in c.products_of[i]})
        for (i, carrier) in enumerate(c.spec.carriers)
    }
    return (e, p, actions)


def advance(pair, steps=4):
    e, p, actions = pair
    for _ in range(steps):
        for engine in (p, e):
            engine.step(actions)
    return e, p


@pytest.mark.parametrize("rebooking", [True, False])
@pytest.mark.parametrize("mixed", [True, False])
def test_new_dated_routes_finite_mixed_parent_identity(rebooking, mixed):
    rng = pickle.dumps(np.random.get_state())
    e, p = advance(source(rebooking=rebooking, mixed=mixed))
    assert pickle.dumps(e.snapshot().native) == pickle.dumps(p.snapshot())
    assert e._private_payload() == p._private_payload()
    assert pickle.dumps(e.trace()) == pickle.dumps(p.trace())
    assert pickle.dumps(e.metrics()) == pickle.dumps(p.metrics())
    assert pickle.dumps(np.random.get_state()) == rng
    periods = e.ordinary_taste_periods()
    assert len(periods) == 4 and len(periods[0].rows) == 2
    assert all(p.rows == () for p in periods[1:])
    for row in periods[0].rows:
        assert row.private_input.taste_class == row.analyst_person.taste_class == 1
        assert row.private_input.route == ("rebooking" if rebooking else "disrupted")
    assert periods[0].rows[1].analyst_person.original_record is None
    assert module.ordinary_private_taste_record(e)[
        "actual_ordinary_taste_rows_captured"
    ]


@pytest.mark.parametrize("route", ["rescue", "reserve", "noheld", "zero"])
def test_new_rescue_reserve_parent_evidence(route):
    e, p = advance(source(route=route, mixed=True))
    assert e._private_payload() == p._private_payload()
    assert e.private_taste_periods()[1].rows


@pytest.mark.parametrize("rebooking", [True, False])
def test_new_once_same_taste_at_utility_earlier_receipt_unchanged(
    monkeypatch, rebooking
):
    e, _, actions = source(rebooking=rebooking, mixed=True)
    native_taste, native_utility = executor.sampled_taste, executor.systematic_utilities
    native_capture = e._capture_ordinary_taste
    tastes, utilities = [], []

    def taste(*args):
        value = native_taste(*args)
        tastes.append(value)
        return value

    def utility(c, market, value, prices, active):
        utilities.append(value)
        return native_utility(c, market, value, prices, active)

    def capture(request, period, value, route):
        assert len(tastes) == len(utilities) + 1 and value is tastes[-1]
        before = pickle.dumps(e._causal_pending[-1].receipt)
        result = native_capture(request, period, value, route)
        assert pickle.dumps(e._causal_pending[-1].receipt) == before
        return result

    monkeypatch.setattr(executor, "sampled_taste", taste)
    monkeypatch.setattr(executor, "systematic_utilities", utility)
    monkeypatch.setattr(e, "_capture_ordinary_taste", capture)
    e.step(actions)
    assert len(tastes) == len(utilities) == 2
    assert all(t is u for t, u in zip(tastes, utilities))
    for row, value in zip(e.ordinary_taste_periods()[0].rows, tastes):
        assert jsonable(row.private_input.resolved_taste) == jsonable(value)
    for row in e.causal_periods()[0].inputs:
        text = json.dumps(jsonable(causal_decision_view(row.receipt)))
        assert "resolved_taste" not in text and "taste_class" not in text


def test_new_original_and_reentry_support_without_mixed_vector_equality():
    e, _, actions = source(mixed=True, interline=False)
    e.step(actions)
    original = e.ordinary_taste_periods()[0].rows[0]
    saved = pickle.dumps(original)
    for _ in range(3):
        e.step(actions)
    assert pickle.dumps(e.ordinary_taste_periods()[0].rows[0]) == saved
    rows = [r for p in e.private_taste_periods() for r in p.rows]
    assert len(rows) == 3
    assert all(r.analyst_original_key == original.analyst_person.key for r in rows)
    assert all(
        r.original_booking.taste_class == original.private_input.taste_class
        for r in rows
    )
    assert all(r.resolved_taste == rows[0].resolved_taste for r in rows)
    assert (
        causal_decision_view(e.causal_periods()[1].inputs[0].receipt)["known_type"][
            "value"
        ]
        is None
    )


@pytest.mark.parametrize("count,early_exit", [(0, False), (2, True)])
def test_new_empty_unavailable_outside_support(count, early_exit):
    e, p = advance(source(count=count, early_exit=early_exit))
    assert e._private_payload() == p._private_payload()
    rows = e.ordinary_taste_periods()[0].rows
    assert len(rows) == count and all(
        r.analyst_person.original_record is None for r in rows
    )
    assert module.ordinary_private_taste_record(e)[
        "actual_ordinary_taste_rows_captured"
    ] == bool(count)


def test_new_forward_counts_match_parent(monkeypatch):
    e, p, actions = source(mixed=True)
    counts = {"e": {}, "p": {}}
    for label, engine in [("e", e), ("p", p)]:
        for name in ["_sellable_global", "post_actions", "_marks", "_record_stranded"]:
            original = getattr(engine, name)

            def call(*args, original=original, label=label, name=name, **kwargs):
                counts[label][name] = counts[label].get(name, 0) + 1
                return original(*args, **kwargs)

            monkeypatch.setattr(engine, name, call)
    for _ in range(4):
        for engine in [p, e]:
            engine.step(actions)
    assert counts["e"] == counts["p"] and counts["e"]["_marks"] > 0


def test_new_owned_export_independent_branch_reset_contract():
    e, _, actions = source(mixed=True)
    e.step(actions)
    branch = e.branch()
    for engine in [e, branch]:
        engine.step(actions)
    assert module.ordinary_private_taste_record(
        e
    ) == module.ordinary_private_taste_record(branch)
    e.step(actions)
    record = module.ordinary_private_taste_record(branch)
    record["periods"][0]["rows"][0]["private_input"]["resolved_taste"]["alpha"] = (
        "changed"
    )
    assert module.ordinary_private_taste_record(branch) != record
    with pytest.raises(TypeError):
        branch.ordinary_taste_periods()[0].rows[0].private_input.resolved_taste[
            "alpha"
        ] = 0
    with pytest.raises(ValueError):
        branch.restore(replace(branch.snapshot(), row_limit=1))
    tape = branch.tape
    branch.reset(tape=tape)
    assert (
        branch.ordinary_taste_periods() == () and len(e.ordinary_taste_periods()) == 3
    )


@pytest.mark.parametrize("where", ["pre_hook", "post_hook", "utility", "post_close"])
def test_new_pending_failure_no_positive_query(monkeypatch, where):
    e, _, actions = source()
    name = (
        "_capture_ordinary_taste"
        if where in {"pre_hook", "post_hook"}
        else "_finish_ordinary_taste_period"
    )
    original = getattr(e, name)

    def fail(*args):
        if where == "post_hook":
            original(*args)
        raise RuntimeError("new injected failure")

    monkeypatch.setattr(
        executor if where == "utility" else e,
        "systematic_utilities" if where == "utility" else name,
        fail,
    )
    with pytest.raises(RuntimeError, match="injected"):
        e.step(actions)
    snap = e.snapshot()
    assert snap.failed and snap.pending is not None and snap.periods == ()
    assert len(snap.pending) == (
        1 if where in {"post_hook", "utility"} else 2 if where == "post_close" else 0
    )
    assert e.world.t == (1 if where == "post_close" else 0)
    with pytest.raises(ValueError):
        e.ordinary_taste_periods()
    with pytest.raises(ValueError):
        e.step(actions)


def test_new_limit_before_append_forced_hooks_refused(monkeypatch):
    e, _, actions = source()
    with pytest.raises(ValueError):
        e.step(actions, _requests=[])
    check = e._check_ordinary_taste_limit

    def limit(proposed=None):
        if type(proposed) is module.OrdinaryTasteInput:
            raise module.OrdinaryTasteLimit("new proposal limit")
        return check(proposed)

    monkeypatch.setattr(e, "_check_ordinary_taste_limit", limit)
    with pytest.raises(module.OrdinaryTasteLimit):
        e.step(actions)
    assert e.snapshot().pending == ()
    with pytest.raises(TypeError):
        module.ordinary_private_taste_record(object())


def test_new_executor_stripped_AST_matches_frozen_bodies():
    cls = next(
        n
        for n in ast.parse(Path(executor.__file__).read_text()).body
        if isinstance(n, ast.ClassDef)
    )
    for path, owner, name in [
        ("airline_market/sim/rebooking.py", "RebookingMarketEngine", "step"),
        (
            "airline_market/sim/disruption.py",
            "DisruptedMarketEngine",
            "_ordinary_disrupted_step",
        ),
    ]:
        native = next(
            n
            for n in ast.parse(Path(path).read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == owner
        )
        original = next(
            n
            for n in native.body
            if isinstance(n, ast.FunctionDef) and n.name == "step"
        )
        adapted = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        adapted.name = "step"
        for node in ast.walk(adapted):
            if isinstance(node, ast.For):
                node.body = [
                    s
                    for s in node.body
                    if not (
                        isinstance(s, ast.Expr)
                        and isinstance(s.value, ast.Call)
                        and ast.unparse(s.value.func) == "self._capture_ordinary_taste"
                    )
                ]
            if (
                name == "step"
                and isinstance(node, ast.Call)
                and ast.unparse(node.func) == "self._ordinary_disrupted_step"
            ):
                node.func = ast.parse("super().step").body[0].value
        assert ast.dump(adapted, include_attributes=False) == ast.dump(
            original, include_attributes=False
        )


def test_new_later_post_close_failure_retains_owned_prior_complete_period(monkeypatch):
    e, _, actions = source(mixed=True)
    e.step(actions)
    prior = pickle.dumps(e.ordinary_taste_periods()[0])

    def fail(period):
        raise RuntimeError("new later close failure")

    monkeypatch.setattr(e, "_finish_ordinary_taste_period", fail)
    with pytest.raises(RuntimeError, match="later close"):
        e.step(actions)
    snapshot = e.snapshot()
    assert snapshot.failed and snapshot.pending == () and e.world.t == 2
    assert len(snapshot.periods) == 1 and pickle.dumps(snapshot.periods[0]) == prior
    assert len(snapshot.native.periods) == 2
    with pytest.raises(ValueError):
        module.ordinary_private_taste_record(e)
