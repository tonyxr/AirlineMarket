"""A.5/B0/B1/V9: NEW six-member local family, not an adopted B1 grid."""

import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace

import numpy as np
import pytest

from airline_market.policy.dated_demand import DatedDemandMarketEngine
from airline_market.policy.dated_rescue import DatedRebookingPolicy
from airline_market.policy.dated_shocks import B1DatedShock, DatedShockMarketEngine
from airline_market.sim.tape import ExogenousTape
from tests.unit import test_dated_demand as prior
from tests.unit import test_rationing_native as native

if os.environ.get("DATED_B1_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_dated_b1", os.environ["DATED_B1_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.policy import dated_b1_family as module


def declarations(c):
    weather = B1DatedShock("wx", "weather", "B_OH0", 3, ".6", 1)
    outage = B1DatedShock("out", "rival_outage", "B_OH0", 3, ".6", 1, "B")
    exit_ = B1DatedShock("exit", "rival_exit", "B_OH0", 3, rival_carrier="B")
    d = prior.design(c)
    return tuple(
        module.DatedB1Declaration(*args)
        for args in [
            ("control",),
            ("weather", weather),
            ("outage", outage),
            ("exit", exit_),
            ("surge", None, d),
            ("hurricane", weather, d),
        ]
    )


def inputs():
    c, clock, dep = prior.inputs()
    return c, clock, dep, declarations(c)


def family():
    c, clock, dep, decl = inputs()
    return module.dated_b1_family(
        c, clock, dep, decl, seed=31, namespace="unit/V9_dated_b1_family"
    )


def policy():
    return DatedRebookingPolicy(72, 20)


def finish(e):
    while not e.terminal():
        e.step(native.actions(e))
    return e


def test_all_five_types_plus_control_keep_declared_date_demand_and_physical_contracts():
    c, clock, dep, decl = inputs()
    f = module.dated_b1_family(
        c, clock, dep, decl, seed=31, namespace="unit/V9_dated_b1_family"
    )
    s = dict(f.scenarios)
    assert {x.kind for x in s.values()} == {
        "control",
        "weather",
        "rival_outage",
        "rival_exit",
        "demand_surge",
        "hurricane",
    }
    for name in ["weather", "outage", "exit"]:
        assert (
            s[name].compiled is c
            and s[name].active_periods == s[name].mixture_witnesses == ()
        )
        assert s[name].design is None and s[name].reveal == prior.hour(24)
        assert s[name].record()["demand_state"] == "control"
        assert len(s[name].physical_shocks) == 1
    assert s["surge"].active_periods == s["hurricane"].active_periods == (2, 3)
    assert s["hurricane"].design.stop_utc == prior.hour(60)
    assert s["hurricane"].physical_shocks[0].duration_days == 1
    assert s["surge"].compiled.spec_hash == s["hurricane"].compiled.spec_hash
    assert s["surge"].contract_hash != s["hurricane"].contract_hash
    record = json.loads(json.dumps(f.record()))
    assert (
        len(record["declared_members"]) == 6
        and not record["full_B1_grid_or_split_adopted"]
    )
    assert (
        not record["economic_B2_channels_authenticated"]
        and record["scientific_budget"] is None
    )


def test_pure_member_compile_has_no_tape_draw_or_global_rng_or_input_mutation(
    monkeypatch,
):
    c, clock, dep, decl = inputs()
    before = pickle.dumps(c.spec)
    rng = pickle.dumps(np.random.get_state())

    def forbidden(*_a, **_k):
        raise AssertionError("pure compiler cannot generate tape")

    monkeypatch.setattr(ExogenousTape, "generate", forbidden)
    for d in decl:
        module.compile_dated_b1_member(c, clock, iter(dep), d)
    assert pickle.dumps(c.spec) == before and pickle.dumps(np.random.get_state()) == rng


def test_one_envelope_generation_then_all_engine_and_projection_calls_have_no_redraw(
    monkeypatch,
):
    c, clock, dep, decl = inputs()
    generate = ExogenousTape.generate
    calls = []

    def spy(*a, **k):
        calls.append(1)
        return generate(*a, **k)

    monkeypatch.setattr(ExogenousTape, "generate", spy)
    f = module.dated_b1_family(
        c, clock, dep, decl, seed=31, namespace="unit/V9_dated_b1_family"
    )
    assert len(calls) == 1
    for name, d in f.declarations:
        e = f.engine(name, dated_rebooking=policy())
        assert e.tape.seed == 31
        projected = f.bank.project(name)
        assert projected.gumbel is f.bank.tape.gumbel
        assert (
            projected.initial_capacity_uniform is f.bank.tape.initial_capacity_uniform
        )
        assert [len(x) for x in projected.arrival_marks] == [
            len(x) for x in f.bank.tape.arrival_marks
        ]
    assert len(calls) == 1
    assert f.record()["native_family"]["shared_bank"]["projection_rng_calls"] == 0
    assert (
        f.record()["native_family"]["shared_bank"]["generation_raw_rng_calls"] is None
    )


def test_one_shot_public_departures_and_registration_order_are_supported():
    c, clock, dep, decl = inputs()
    a = module.dated_b1_family(
        c, clock, iter(dep), iter(decl), seed=31, namespace="unit/V9_dated_b1_family"
    )
    b = module.dated_b1_family(
        c,
        clock,
        reversed(dep),
        reversed(decl),
        seed=31,
        namespace="unit/V9_dated_b1_family",
    )
    assert a.record() == b.record()
    assert pickle.dumps(a.bank.tape) == pickle.dumps(b.bank.tape)


@pytest.mark.parametrize("name", ["weather", "outage", "exit", "hurricane"])
def test_declared_physical_members_require_explicit_dated_reentry(name):
    f = family()
    with pytest.raises(ValueError, match="dated stranded re-entry"):
        f.engine(name)
    assert f.engine(name, dated_rebooking=policy()).world.t == 0


def test_control_delegates_native_parent_world_trace_and_metrics_on_same_bank():
    f = family()
    a = f.engine("control", dated_rebooking=policy())
    s = dict(f.scenarios)["control"]
    b = DatedDemandMarketEngine(s, dated_rebooking=policy())
    b.reset(tape=f.bank.project("control"))
    while not a.terminal():
        a.step(native.actions(a))
        b.step(native.actions(b))
        assert pickle.dumps(a.world) == pickle.dumps(b.world)
        assert pickle.dumps(a.trace()) == pickle.dumps(b.trace())
    assert a.metrics() == b.metrics()


def test_all_family_members_share_native_pre_reveal_world_and_potential_marks():
    f = family()
    engines = [f.engine(name, dated_rebooking=policy()) for name, d in f.declarations]
    for e in engines:
        e.step(native.actions(e))
    for e in engines[1:]:
        assert pickle.dumps(e.world) == pickle.dumps(engines[0].world)
        assert pickle.dumps(e.trace()) == pickle.dumps(engines[0].trace())
    assert all(e.world.t == 1 for e in engines)


def force_original_B_bookings(e):
    tape = e.tape
    g = tape.gumbel.copy()
    for t in (0, 1):
        for slot, m in enumerate(tape.arrival_marks[t]):
            choices = [
                j
                for j in e.compiled.market_products[m.market]
                if e.compiled.spec.products[j].carrier == "B"
            ]
            if choices:
                g[t, slot, :] = -1e9
                g[t, slot, choices[0]] = 1e9
    g.setflags(write=False)
    e.reset(tape=replace(tape, gumbel=g))
    e.step(native.actions(e))
    e.step(native.actions(e))
    return e


@pytest.mark.parametrize("name", ["weather", "outage", "exit", "hurricane"])
def test_new_all_type_family_executes_actual_native_refunds_and_root_conservation(name):
    f = family()
    e = force_original_B_bookings(f.engine(name, dated_rebooking=policy()))
    assert e._passengers and e.world.t == 2
    assert e.world.carriers[e.compiled.carrier_index["B"]].refunds > 0
    assert all(e.request_coordinate(root).potential_slot >= 0 for root in e._passengers)
    finish(e)
    assert (
        e.conservation()["original_booked_passengers"]
        == e.conservation()["active_plus_pending_plus_unserved"]
    )
    if name == "exit":
        assert not e.world.carriers[e.compiled.carrier_index["B"]].capacity.any()
    else:
        assert all(r["removed"] == r["restored"] for r in e.capacity_window_balance())


def test_physical_lift_matches_frozen_physical_parent_without_demand_change():
    f = family()
    s = dict(f.scenarios)["outage"]
    a = f.engine("outage", dated_rebooking=policy())
    b = DatedShockMarketEngine(
        s.compiled,
        shock_clock=s.clock,
        shock_leg_departures=s.departures,
        dated_shocks=s.physical_shocks,
        dated_rebooking=policy(),
    )
    b.reset(tape=f.bank.project("outage"))
    finish(a)
    finish(b)
    assert (
        pickle.dumps(a.world) == pickle.dumps(b.world)
        and pickle.dumps(a.trace()) == pickle.dumps(b.trace())
        and a.metrics() == b.metrics()
    )


def test_nested_snapshots_branch_reset_and_same_spec_different_physical_contract():
    f = family()
    e = f.engine("weather", dated_rebooking=policy())
    e.step(native.actions(e))
    snap = e.snapshot()
    b = e.branch()
    b.step(native.actions(b))
    assert e.world.t == 1 and b.world.t == 2
    b.restore(snap)
    assert pickle.dumps(b.snapshot()) == pickle.dumps(snap)
    assert (
        e.compiled.spec_hash
        == f.engine("outage", dated_rebooking=policy()).compiled.spec_hash
    )
    with pytest.raises(ValueError, match="demand/date/composite"):
        f.engine("outage", dated_rebooking=policy()).restore(snap)
    b.reset(tape=f.bank.project("weather"))
    assert b.world.t == 0 and not b.capacity_window_balance()


def test_terminal_physical_reveal_retained_unstarted_without_inserted_opportunity():
    c, clock, dep, _ = inputs()
    future = B1DatedShock("future", "weather", "B_OH0", 1, ".3", 1)
    f = module.dated_b1_family(
        c,
        clock,
        dep,
        [
            module.DatedB1Declaration("future", future),
            module.DatedB1Declaration("control"),
        ],
        seed=31,
        namespace="unit/V9_dated_b1_future",
    )
    s = dict(f.scenarios)["future"]
    assert s.reveal == clock.at(5) and s.compiled is c
    e = finish(f.engine("future", dated_rebooking=policy()))
    assert e.dated_shock_mappings()[0].status == "unstarted_at_or_after_terminal"
    assert not e.capacity_window_balance()
    b = finish(f.engine("control", dated_rebooking=policy()))
    assert pickle.dumps(e.world) == pickle.dumps(b.world)


def test_invalid_member_rejected_before_any_envelope_generation(monkeypatch):
    c, clock, dep, decl = inputs()

    def forbidden(*_a, **_k):
        raise AssertionError("invalid family must not draw")

    monkeypatch.setattr(ExogenousTape, "generate", forbidden)
    with pytest.raises(ValueError, match="distinct"):
        module.dated_b1_family(
            c, clock, dep, [decl[0], decl[0]], seed=31, namespace="unit/invalid"
        )
    bad = replace(
        decl[0],
        scenario_id="bad",
        physical=B1DatedShock("offclock", "weather", "B_OH0", 3, ".3", 1),
    )
    shifted = tuple((k, v + prior.timedelta(minutes=1)) for k, v in dep)
    with pytest.raises(ValueError):
        module.dated_b1_family(
            c, clock, shifted, [bad], seed=31, namespace="unit/invalid"
        )


def test_only_explicit_typed_B1_members_and_supported_composite_are_accepted():
    c, _, _, decl = inputs()
    with pytest.raises(TypeError):
        module.DatedB1Declaration("bad", physical={})
    with pytest.raises(TypeError):
        module.DatedB1Declaration("bad", demand={})
    with pytest.raises(ValueError, match="weather plus surge"):
        module.DatedB1Declaration("bad", decl[2].physical, prior.design(c))
    with pytest.raises(ValueError):
        module.DatedB1Declaration("")
