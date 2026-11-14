"""A.5/B0/B1/V9: exact elapsed-day B1 maps and new native-date integration."""

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
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.capacity_windows_v2 import CapacityWindowMarketEngineV2
from airline_market.policy.dated_rescue import (
    DatedRebookingPolicy,
    DatedRescueMenu,
    UtcWindow,
)
from airline_market.policy.dated_reserve import DatedSeatReserve, UtcBookingClock
from airline_market.policy.rationed_native import WholePeriodPriority
from airline_market.sim.tape import ExogenousTape
from airline_market.spec import compile_market
from tests.unit import test_capacity_windows_v2 as physical
from tests.unit import test_rationing_native as ordinary

if os.environ.get("DATED_SHOCKS_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_dated_shocks", os.environ["DATED_SHOCKS_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.policy import dated_shocks as module

BASE = datetime(2026, 3, 7, tzinfo=timezone.utc)


def hour(n):
    return BASE + timedelta(hours=n)


def fixture(shocks=(), *, policy=False, priority=False, clock=None, departures=None):
    c = compile_market(
        screen_spec(
            {
                "T": 5,
                "actions": 2,
                "n_spokes": 1,
                "capacity": 20,
                "capacity_support": [3, 4],
            }
        )
    )
    clock = clock or UtcBookingClock(tuple(hour(x) for x in (0, 12, 24, 48, 60, 72)))
    dep = departures or tuple(
        (l.id, hour(120 if l.id == "B_HD0" else 96 if l.carrier == "B" else 108))
        for l in c.spec.legs
    )
    kwargs = {}
    if policy:
        kwargs["dated_rebooking"] = DatedRebookingPolicy(72, 20)
        kwargs["dated_menu"] = DatedRescueMenu(
            UtcWindow(hour(24), hour(90)),
            "same_route",
            route_fares=tuple((m.id, 99) for m in c.spec.markets),
        )
        kwargs["dated_reserve"] = DatedSeatReserve(
            hour(24), ".10", tuple(l.id for l in c.spec.legs if l.carrier == "A"), dep
        )
    if priority:
        kwargs["ordinary_priority"] = WholePeriodPriority("unit/V9_dated_shocks")
    e = module.DatedShockMarketEngine(
        c, shock_clock=clock, shock_leg_departures=dep, dated_shocks=shocks, **kwargs
    )
    e.reset(
        tape=ExogenousTape.generate(
            c, 31, registry=StreamRegistry(33974, "unit/V9_dated_shocks")
        )
    )
    return e


def shock(kind="rival_outage", share=".9", days=1, reveal=3, id="unit-shock"):
    return module.B1DatedShock(
        id,
        kind,
        "B_OH0",
        reveal,
        None if kind == "rival_exit" else share,
        None if kind == "rival_exit" else days,
        None if kind == "weather" else "B",
    )


def force_two_originals(e):
    q = e.tape.request_uniforms.copy()
    q[:, 0] = 1
    g = e.tape.gumbel.copy()
    j = e.compiled.product_index["B_OD0_0"]
    for t in (0, 1):
        q[t, 0] = 0
        q[t, 1] = 0.999
        q[t, 2] = 0
        g[t, 0, :] = -1e9
        g[t, 0, j] = 1e9
    q.setflags(write=False)
    g.setflags(write=False)
    e.tape = replace(e.tape, request_uniforms=q, gumbel=g)
    e.step(ordinary.actions(e))
    e.step(ordinary.actions(e))
    return tuple(e._passengers.values())


def installed(e, leg):
    return physical.installed(e, leg)


def test_irregular_clock_maps_24_elapsed_hours_not_one_native_period():
    e = fixture((shock(),))
    m = e.dated_shock_mappings()[0]
    assert (m.reveal, m.recovery) == (hour(24), hour(48))
    assert (m.native_window.start, m.native_window.stop) == (2, 3)
    assert m.reference_departure == hour(96)
    assert dict(m.lead_microseconds)["B_OH0"] == 72 * 3600 * 1000000
    assert dict(m.lead_microseconds)["B_HD0"] == 96 * 3600 * 1000000


@pytest.mark.parametrize(
    "days,stop,status",
    [
        (1, 3, "exact_boundary"),
        (2, 5, "exact_boundary"),
        (3, 6, "beyond_horizon_outstanding"),
    ],
)
def test_all_registered_elapsed_durations_retain_exact_terminal_or_future_recovery(
    days, stop, status
):
    e = fixture((shock(days=days),))
    m = e.dated_shock_mappings()[0]
    assert m.recovery - m.reveal == timedelta(days=days)
    assert m.native_window.stop == stop and m.recovery_status == status
    while not e.terminal():
        e.step(ordinary.actions(e))
    rows = e.capacity_window_balance()
    assert all(
        r["removed"] == r["restored"] + r["blocked_by_exit"] + r["outstanding"]
        for r in rows
    )
    assert all(
        (r["outstanding"] == 0) if days < 3 else (r["outstanding"] == r["removed"])
        for r in rows
    )
    assert e.dated_shock_mappings()[0].record()[
        "outside_horizon_stop_is_not_a_dated_boundary"
    ] is (days == 3)


@pytest.mark.parametrize("reveal_days", [14, 7, 3, 1])
def test_each_b1_reveal_uses_named_departure_with_no_calendar_or_period_guess(
    reveal_days,
):
    e = fixture()
    dep = tuple((k, BASE + timedelta(days=20)) for k, v in e.shock_leg_departures)
    reveal = BASE + timedelta(days=20 - reveal_days)
    end = BASE + timedelta(days=20, hours=-1)
    times = [
        BASE,
        reveal,
        reveal + timedelta(hours=2),
        reveal + timedelta(hours=6),
        reveal + timedelta(hours=12),
        end,
    ]
    if reveal_days > 1:
        times[3] = reveal + timedelta(days=1)
        times[4] = reveal + timedelta(days=1, hours=1)
    clock = UtcBookingClock(tuple(times))
    m = module.compile_dated_shocks(
        e.compiled, clock, dep, (shock(reveal=reveal_days),)
    )[0]
    assert m.reveal == reveal and m.native_window.start == 1
    assert m.reference_departure - m.reveal == timedelta(days=reveal_days)


def test_weather_targets_both_carriers_outage_targets_explicit_rival_only():
    e = fixture((shock("weather", ".3"),))
    assert set(e.dated_shock_mappings()[0].affected_legs) == set(e.compiled.leg_index)
    e = fixture((shock(),))
    assert all(
        e.compiled.spec.legs[e.compiled.leg_index[k]].carrier == "B"
        for k in e.dated_shock_mappings()[0].affected_legs
    )
    before = {k: installed(e, k) for k in e.compiled.leg_index}
    e.step(ordinary.actions(e))
    e.step(ordinary.actions(e))
    assert all(installed(e, k) == v for k, v in before.items() if k.startswith("A_"))
    assert all(installed(e, k) < v for k, v in before.items() if k.startswith("B_"))


def test_dated_exit_strands_real_bookings_refunds_and_never_restores_rival():
    e = fixture((shock("rival_exit"),), policy=True)
    passengers = force_two_originals(e)
    assert len(passengers) == 2 and all(p.first_stranded_at == 2 for p in passengers)
    assert e.booking_clock.at(2) == hour(24)
    assert e.world.carriers[1].refunds == sum(p.original.fare for p in passengers)
    assert all(
        e.request_coordinate(p.original.record_id).period in (0, 1) for p in passengers
    )
    while not e.terminal():
        e.step(ordinary.actions(e))
    assert all(installed(e, k) == 0 for k in e.compiled.leg_index if k.startswith("B_"))
    assert (
        e.conservation()["original_booked_passengers"]
        == e.conservation()["active_plus_pending_plus_unserved"]
    )


def test_dated_temporary_loss_uses_native_refund_then_exact_recovery_without_resurrection():
    e = fixture((shock(),), policy=True)
    passengers = force_two_originals(e)
    assert len(passengers) == 1 and passengers[0].first_stranded_at == 2
    root = passengers[0].original.record_id
    deadline = e.reentry_deadline_utc(root)
    assert (
        deadline == hour(96)
        and e.world.carriers[1].refunds == passengers[0].original.fare
    )
    e.step(ordinary.actions(e))
    assert e.world.t == 3 and e.shock_clock.at(3) == hour(48)
    assert e.reentry_deadline_utc(root) == deadline
    assert not any(b.record_id == root for cs in e.world.carriers for b in cs.bookings)
    assert all(r["removed"] == r["restored"] for r in e.capacity_window_balance())


def test_zero_shock_adapter_is_byte_identical_to_parent_with_same_dates_and_marks():
    e = fixture(policy=True, priority=True)
    b = CapacityWindowMarketEngineV2(
        e.compiled,
        booking_clock=e.shock_clock,
        leg_departures=e.shock_leg_departures,
        dated_rebooking=e.dated_rebooking,
        dated_menu=e.dated_menu,
        dated_reserve=e.dated_reserve,
        ordinary_priority=e.ordinary_priority,
    )
    b.reset(tape=e.tape)
    while not e.terminal():
        e.step(ordinary.actions(e))
        b.step(ordinary.actions(b))
    assert pickle.dumps(e.world) == pickle.dumps(b.world)
    assert (
        pickle.dumps(e.trace()) == pickle.dumps(b.trace())
        and e.metrics() == b.metrics()
    )


def test_compiler_has_no_rng_or_actor_world_access_and_lossless_json_roundtrip():
    e = fixture((shock(),))
    before = pickle.dumps(e.snapshot())
    global_state = pickle.dumps(np.random.get_state())
    mappings = module.compile_dated_shocks(
        e.compiled, e.shock_clock, e.shock_leg_departures, e.dated_shocks
    )
    assert (
        pickle.dumps(e.snapshot()) == before
        and pickle.dumps(np.random.get_state()) == global_state
    )
    record = json.loads(json.dumps(mappings[0].record()))
    assert module.B1DatedShock(**record["design"]) == e.dated_shocks[0]
    assert (
        record["native_window"]["share"] == "0.9"
        and record["reveal_utc"] == hour(24).isoformat()
    )


def test_aware_offset_and_dst_spanning_dates_normalize_to_exact_elapsed_hours():
    e = fixture()
    dep = tuple(
        (k, v.astimezone(timezone(timedelta(hours=-5))))
        for k, v in e.shock_leg_departures
    )
    m = module.compile_dated_shocks(e.compiled, e.shock_clock, dep, (shock(),))[0]
    assert m.reveal == hour(24) and m.recovery - m.reveal == timedelta(hours=24)
    changed = tuple((k, v.isoformat()) for k, v in dep)
    assert module.compile_dated_shocks(
        e.compiled, e.shock_clock, changed, (shock(),)
    ) == (m,)


@pytest.mark.parametrize("boundary", [24, 48])
def test_missing_exact_in_horizon_reveal_or_recovery_is_not_rounded(boundary):
    e = fixture()
    times = tuple(
        t + timedelta(minutes=1) if t == hour(boundary) else t
        for t in e.shock_clock.boundaries
    )
    with pytest.raises(ValueError, match="exact declared UTC boundary"):
        module.compile_dated_shocks(
            e.compiled, UtcBookingClock(times), e.shock_leg_departures, (shock(),)
        )


def test_future_or_terminal_reveal_is_retained_unstarted_not_moved_into_cohort():
    e = fixture()
    dep = tuple((k, v + timedelta(hours=48)) for k, v in e.shock_leg_departures)
    e = fixture((shock(),), departures=dep)
    m = e.dated_shock_mappings()[0]
    assert m.reveal == hour(72) and m.native_window is None
    assert m.status == "unstarted_at_or_after_terminal"
    while not e.terminal():
        e.step(ordinary.actions(e))
    assert not e.capacity_window_events() and not e.capacity_window_balance()


@pytest.mark.parametrize(
    "change",
    [
        {"share": ".2"},
        {"duration_days": 4},
        {"reveal_days": 2},
        {"reveal_days": True},
        {"kind": "hurricane"},
        {"rival_carrier": None},
    ],
)
def test_unsupported_b1_design_is_explicit_not_silently_changed(change):
    with pytest.raises((ValueError, TypeError)):
        replace(shock(), **change)


def test_missing_duplicated_naive_departures_or_departure_inside_horizon_not_imputed():
    e = fixture()
    dep = e.shock_leg_departures
    for bad in [
        dep[:-1],
        dep + (dep[0],),
        tuple((k, v.replace(tzinfo=None)) for k, v in dep),
        tuple((k, hour(72)) for k, v in dep),
    ]:
        with pytest.raises((ValueError, TypeError)):
            module.compile_dated_shocks(e.compiled, e.shock_clock, bad, (shock(),))
    with pytest.raises(ValueError, match="pre-cohort reveal"):
        module.compile_dated_shocks(e.compiled, e.shock_clock, dep, (shock(reveal=7),))
    with pytest.raises(ValueError, match="affected carrier"):
        module.compile_dated_shocks(
            e.compiled, e.shock_clock, dep, (replace(shock(), reference_leg="A_OH0"),)
        )


def test_nested_snapshot_branch_reset_preserve_dates_and_refuse_mismatched_anchor_contract():
    e = fixture((shock(),), policy=True, priority=True)
    force_two_originals(e)
    snap = e.snapshot()
    b = e.branch()
    assert type(b) is type(e)
    b.step(ordinary.actions(b))
    assert e.world.t == 2 and b.world.t == 3
    b.restore(snap)
    assert pickle.dumps(b.snapshot()) == pickle.dumps(snap)
    other = fixture(
        (replace(shock(), shock_id="different"),), policy=True, priority=True
    )
    with pytest.raises(ValueError, match="dated-shock"):
        other.restore(snap)
    dep = tuple(
        (k, v + timedelta(hours=1)) if k == "A_OH0" else (k, v)
        for k, v in e.shock_leg_departures
    )
    other = fixture((shock(),), departures=dep)
    with pytest.raises(ValueError, match="dated-shock"):
        other.restore(snap)
    b.reset(tape=e.tape)
    assert b.world.t == 0 and not b.capacity_window_balance()


def test_native_override_and_dated_policy_clock_mismatch_refused():
    e = fixture()
    for kwargs in [
        {"capacity_windows": ()},
        {"exits": ()},
        {
            "booking_clock": UtcBookingClock(
                tuple(hour(x) for x in (0, 13, 24, 48, 60, 72))
            )
        },
        {
            "leg_departures": tuple(
                (k, v + timedelta(hours=1)) for k, v in e.shock_leg_departures
            )
        },
    ]:
        with pytest.raises(ValueError):
            module.DatedShockMarketEngine(
                e.compiled,
                shock_clock=e.shock_clock,
                shock_leg_departures=e.shock_leg_departures,
                **kwargs,
            )


@pytest.mark.parametrize(
    "kind,share",
    [
        ("weather", ".1"),
        ("weather", ".3"),
        ("weather", ".6"),
        ("rival_outage", ".3"),
        ("rival_outage", ".6"),
        ("rival_outage", ".9"),
    ],
)
def test_every_declared_capacity_magnitude_uses_same_exact_installed_floor(kind, share):
    from fractions import Fraction

    e = fixture((shock(kind, share),))
    e.step(ordinary.actions(e))
    e.step(ordinary.actions(e))
    frac = Fraction(share)
    rows = e.capacity_window_balance()
    assert len(rows) == (len(e.compiled.leg_index) if kind == "weather" else 2)
    assert all(
        r["removed"] == r["installed_at_reveal"] * frac.numerator // frac.denominator
        for r in rows
    )


def test_initial_reveal_is_applied_once_and_restored_at_exact_day_boundary():
    e = fixture()
    clock = UtcBookingClock(tuple(hour(x) for x in (0, 6, 12, 18, 24, 36)))
    dep = tuple(
        (k, hour(72 if k == "B_OH0" else 84)) for k, v in e.shock_leg_departures
    )
    e = fixture((shock(),), clock=clock, departures=dep)
    m = e.dated_shock_mappings()[0]
    assert (m.native_window.start, m.native_window.stop) == (0, 4)
    assert len(e.capacity_window_events()) == 1
    e.begin_period(ordinary.actions(e))
    e.begin_period(ordinary.actions(e))
    assert len(e.capacity_window_events()) == 1
    for _ in range(4):
        e.step(ordinary.actions(e))
    assert e.world.t == 4 and e.shock_clock.at(4) == hour(24)
    assert all(r["removed"] == r["restored"] for r in e.capacity_window_balance())


def test_native_window_indices_explicit_when_earlier_design_is_future_unstarted():
    future = replace(shock(id="future"), reveal_days=1)
    e = fixture((future, shock(id="present")))
    a, b = e.dated_shock_mappings()
    assert a.native_window is None and a.native_window_index is None
    assert b.native_window_index == 0 and b.record()["native_window_index"] == 0
    e.step(ordinary.actions(e))
    e.step(ordinary.actions(e))
    assert {r["window"] for r in e.capacity_window_balance()} == {0}


def test_permanent_dated_exit_blocks_future_dated_recovery_without_reopening():
    e = fixture((shock(id="outage"), shock("rival_exit", id="exit")), policy=True)
    force_two_originals(e)
    while not e.terminal():
        e.step(ordinary.actions(e))
    assert all(installed(e, k) == 0 for k in e.compiled.leg_index if k.startswith("B_"))
    assert all(
        r["removed"] == r["blocked_by_exit"] and r["restored"] == 0
        for r in e.capacity_window_balance()
    )
    a, b = e.dated_shock_mappings()
    assert a.native_window_index == 0 and b.native_exit_index == 0
    assert b.recovery_status == "permanent_exit"


def test_duplicate_ids_unknown_rival_or_missing_share_not_coerced_to_valid_shock():
    e = fixture()
    with pytest.raises(ValueError, match="distinct declared shock IDs"):
        module.compile_dated_shocks(
            e.compiled, e.shock_clock, e.shock_leg_departures, (shock(), shock())
        )
    with pytest.raises(ValueError, match="affected carrier"):
        module.compile_dated_shocks(
            e.compiled,
            e.shock_clock,
            e.shock_leg_departures,
            (replace(shock(), rival_carrier="missing"),),
        )
    with pytest.raises(ValueError, match="B1 share"):
        replace(shock(), share=None)


def test_elapsed_day_across_DST_is_24_utc_hours_and_25_local_wall_hours():
    from zoneinfo import ZoneInfo

    e = fixture((shock(),))
    m = e.dated_shock_mappings()[0]
    ny = ZoneInfo("America/New_York")
    a = m.reveal.astimezone(ny)
    b = m.recovery.astimezone(ny)
    assert a.utcoffset() == timedelta(hours=-5) and b.utcoffset() == timedelta(hours=-4)
    assert m.recovery - m.reveal == timedelta(hours=24)
    assert b.replace(tzinfo=None) - a.replace(tzinfo=None) == timedelta(hours=25)
    dep = tuple((k, v.astimezone(ny)) for k, v in e.shock_leg_departures)
    assert module.compile_dated_shocks(
        e.compiled, e.shock_clock, dep, e.dated_shocks
    ) == (m,)
