"""A.5/B0/B1/V9: exact-date demand/mix projection and native hurricane composition."""

import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from airline_market.core.hashing import spec_hash
from airline_market.policy.dated_rescue import DatedRebookingPolicy
from airline_market.policy.dated_reserve import UtcBookingClock
from airline_market.policy.dated_shocks import B1DatedShock, DatedShockMarketEngine
from airline_market.sim.arrivals import period_requests
from airline_market.sim.exogenous import demand_multiplier
from airline_market.sim.tape import ExogenousTape, segment_weights
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule
from tests.unit import test_rationing_native as native

if os.environ.get("DATED_DEMAND_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_dated_demand", os.environ["DATED_DEMAND_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.policy import dated_demand as module

BASE = datetime(2026, 3, 7, tzinfo=timezone.utc)


def hour(n):
    return BASE + timedelta(hours=n)


def inputs():
    c = native.fixture().compiled
    clock = UtcBookingClock(tuple(hour(x) for x in (0, 12, 24, 48, 60, 72)))
    dep = tuple(
        (l.id, hour(120 if l.id == "B_HD0" else 96 if l.carrier == "B" else 108))
        for l in c.spec.legs
    )
    return c, clock, dep


def design(
    c,
    *,
    surge=".3",
    stop=60,
    markets=None,
    weights=(("leisure", ".05"), ("business", ".95")),
    id="surge",
):
    return module.DatedDemandSurge(
        id,
        "B_OH0",
        3,
        surge,
        hour(stop),
        tuple(m.id for m in c.spec.markets) if markets is None else markets,
        weights,
    )


def scenarios(*, composite=False):
    c, clock, dep = inputs()
    d = design(c)
    out = {
        "control": module.compile_dated_demand(c, clock, dep),
        "surge": module.compile_dated_demand(c, clock, dep, d),
    }
    if composite:
        out["hurricane"] = module.compile_dated_hurricane(
            c,
            clock,
            dep,
            scenario_id="hurricane",
            capacity=B1DatedShock("weather", "weather", "B_OH0", 3, ".6", 1),
            demand=d,
        )
    return out


def family(items=None):
    return module.dated_demand_family(
        scenarios() if items is None else items,
        seed=31,
        namespace="unit/V9_dated_demand",
    )


@pytest.mark.parametrize("surge,mult", [(".3", 1.3), ("1", 2.0)])
def test_registered_surges_apply_only_exact_UTC_window_and_explicit_markets(
    surge, mult
):
    c, clock, dep = inputs()
    target = c.spec.markets[0].id
    out = module.compile_dated_demand(
        c, clock, dep, design(c, surge=surge, markets=(target,))
    )
    assert out.active_periods == (2, 3) and out.reveal == hour(24)
    for j, m in enumerate(c.spec.markets):
        for t in range(5):
            before = c.spec.demand.intensity[m.id].value_at(t)
            after = out.compiled.spec.demand.intensity[m.id].value_at(t)
            assert after == pytest.approx(
                before * (mult if m.id == target and t in (2, 3) else 1), rel=1e-15
            )
            old = tuple(segment_weights(c, t, j))
            new = tuple(segment_weights(out.compiled, t, j))
            assert new == ((0.05, 0.95) if m.id == target and t in (2, 3) else old)
        if m.id != target:
            assert (
                out.compiled.spec.demand.intensity[m.id]
                is c.spec.demand.intensity[m.id]
            )
    assert spec_hash(replace(c.spec, demand=None)) == spec_hash(
        replace(out.compiled.spec, demand=None)
    )


def test_mixture_shift_compared_against_each_native_period_not_only_initial_mix():
    c, clock, dep = inputs()
    assert tuple(segment_weights(c, 0, 0)) == (0.8, 0.2) and tuple(
        segment_weights(c, 2, 0)
    ) == (0.3, 0.7)
    with pytest.raises(ValueError, match="strictly shift"):
        module.compile_dated_demand(
            c, clock, dep, design(c, weights=(("leisure", ".4"), ("business", ".6")))
        )
    out = module.compile_dated_demand(c, clock, dep, design(c))
    assert len(out.mixture_witnesses) == 2 * len(c.spec.markets)
    assert all(
        all(b >= a for _, a, b in rows) and any(b > a for _, a, b in rows)
        for t, m, rows in out.mixture_witnesses
    )


def test_mean_alpha_decrease_is_insufficient_when_finite_distributions_cross():
    c, clock, dep = inputs()
    a, b = c.spec.segments
    cls = a.classes[0]
    a = replace(
        a,
        classes=(
            replace(cls, weight=0.5, alpha=0.01),
            replace(cls, weight=0.5, alpha=0.05),
        ),
    )
    b = replace(
        b,
        classes=(
            replace(cls, weight=0.8, alpha=0.005),
            replace(cls, weight=0.2, alpha=0.09),
        ),
    )
    d = replace(
        c.spec.demand, segment_weights=PeriodSchedule.parse({"constant": (1, 0)})
    )
    c = compile_market(replace(c.spec, segments=(a, b), demand=d))
    assert 0.8 * 0.005 + 0.2 * 0.09 < 0.5 * 0.01 + 0.5 * 0.05
    with pytest.raises(ValueError, match="strictly shift"):
        module.compile_dated_demand(
            c, clock, dep, design(c, weights=(("leisure", 0), ("business", 1)))
        )


def test_exact_nominal_CDF_records_preserve_zero_mass_and_json_rationals():
    c, clock, dep = inputs()
    out = module.compile_dated_demand(
        c, clock, dep, design(c, weights=(("leisure", 0), ("business", 1)))
    )
    record = json.loads(json.dumps(out.record()))
    assert record["design"]["post_segment_weights"]["leisure"] == {
        "numerator": "0",
        "denominator": "1",
    }
    for cert in record["mixture_witnesses"]:
        rows = cert["finite_alpha_cdf"]
        assert (
            rows[-1]["baseline_cdf"]
            == rows[-1]["post_cdf"]
            == {"numerator": "1", "denominator": "1"}
        )
    assert (
        record["active_periods"] == [2, 3]
        and record["full_B1_or_empirical_adoption"] is False
    )


def test_pure_compile_no_rng_world_or_source_spec_mutation():
    c, clock, dep = inputs()
    before = pickle.dumps(c.spec)
    state = pickle.dumps(np.random.get_state())
    out = module.compile_dated_demand(c, clock, dep, design(c))
    assert (
        pickle.dumps(c.spec) == before and pickle.dumps(np.random.get_state()) == state
    )
    assert out.base_spec_sha256 == c.spec_hash


@pytest.mark.parametrize("which", [24, 60])
def test_in_horizon_reveal_and_demand_stop_require_exact_public_boundaries(which):
    c, clock, dep = inputs()
    clock = UtcBookingClock(
        tuple(
            t + timedelta(minutes=1) if t == hour(which) else t
            for t in clock.boundaries
        )
    )
    with pytest.raises(ValueError, match="exact declared UTC boundary"):
        module.compile_dated_demand(c, clock, dep, design(c))


def test_future_demand_stop_preserved_without_inferred_duration_or_final_boundary():
    c, clock, dep = inputs()
    out = module.compile_dated_demand(c, clock, dep, design(c, stop=100))
    assert out.active_periods == (2, 3, 4) and out.design.stop_utc == hour(100)
    assert out.record()["design"]["stop_utc"] == hour(100).isoformat()
    assert clock.at(5) == hour(72)


def test_control_and_future_reveal_retain_original_compiled_object():
    c, clock, dep = inputs()
    control = module.compile_dated_demand(c, clock, dep)
    assert (
        control.compiled is c
        and control.kind == "control"
        and not control.mixture_witnesses
    )
    d = replace(design(c, stop=100), reveal_days=1)
    future = module.compile_dated_demand(c, clock, dep, d)
    assert (
        future.compiled is c
        and future.reveal == hour(72)
        and not future.active_periods
        and not future.mixture_witnesses
    )
    assert future.record()["demand_state"] == "unstarted_in_cohort"


def test_family_one_native_envelope_generation_and_no_projection_redraw(monkeypatch):
    items = scenarios()
    generate = ExogenousTape.generate
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return generate(*args, **kwargs)

    monkeypatch.setattr(ExogenousTape, "generate", spy)
    f = family(items)
    assert len(calls) == 1
    for name in ("control", "surge"):
        f.engine(name)
    assert len(calls) == 1
    assert f.record()["shared_bank"]["projection_rng_calls"] == 0
    assert f.record()["shared_bank"]["generation_raw_rng_calls"] is None


def test_every_family_native_rate_and_original_slot_predicate_matches_declared_window():
    f = family()
    for name, s in f.scenarios:
        c = s.compiled
        tape = f.bank.project(name)
        caps = dict(f.bank.member_caps)[name]
        for t in range(5):
            for ex in range(len(c.spec.exogenous.states)):
                r = demand_multiplier(c, ex) / max(
                    demand_multiplier(c, k) for k in range(len(c.spec.exogenous.states))
                )
                expected = [
                    i
                    for i, m in enumerate(f.bank.tape.arrival_marks[t])
                    if m.thinning
                    < caps[t, m.market, m.segment]
                    / f.bank.envelope_rates[t, m.market, m.segment]
                    * r
                ]
                assert [
                    q.gumbel_row for q in period_requests(c, tape, t, ex)
                ] == expected
                for j, m in enumerate(c.spec.markets):
                    for g, w in enumerate(segment_weights(c, t, j)):
                        target = (
                            c.spec.demand.intensity[m.id].value_at(t)
                            * w
                            * demand_multiplier(c, ex)
                        )
                        reconstructed = (
                            f.bank.envelope_rates[t, j, g]
                            * (caps[t, j, g] / f.bank.envelope_rates[t, j, g])
                            * r
                        )
                        assert reconstructed == pytest.approx(
                            target, rel=1e-14, abs=1e-14
                        )
    a = f.bank.project("control")
    b = f.bank.project("surge")
    assert a.arrival_marks[:2] == b.arrival_marks[:2]
    assert (
        a.gumbel is b.gumbel
        and a.initial_capacity_uniform is b.initial_capacity_uniform
    )
    for x, y, z in zip(f.bank.tape.arrival_marks, a.arrival_marks, b.arrival_marks):
        assert len(x) == len(y) == len(z)
        assert [(m.time, m.market, m.segment, m.class_uniform) for m in x] == [
            (m.time, m.market, m.segment, m.class_uniform) for m in z
        ]


def test_no_shock_native_identity_and_shared_pre_reveal_prefix():
    f = family()
    a = f.engine("control")
    b = f.engine("surge")
    parent = DatedShockMarketEngine(
        a.compiled,
        shock_clock=a.shock_clock,
        shock_leg_departures=a.shock_leg_departures,
    )
    parent.reset(tape=f.bank.project("control"))
    for _ in range(2):
        a.step(native.actions(a))
        b.step(native.actions(b))
        parent.step(native.actions(parent))
        assert (
            pickle.dumps(a.world) == pickle.dumps(b.world) == pickle.dumps(parent.world)
        )
        assert (
            pickle.dumps(a.trace())
            == pickle.dumps(b.trace())
            == pickle.dumps(parent.trace())
        )
    while not a.terminal():
        a.step(native.actions(a))
        parent.step(native.actions(parent))
    assert (
        pickle.dumps(a.world) == pickle.dumps(parent.world)
        and a.metrics() == parent.metrics()
    )


def test_hurricane_keeps_separate_declared_recovery_dates_and_requires_reentry():
    items = scenarios(composite=True)
    s = items["hurricane"]
    assert s.physical_shocks[0].duration_days == 1 and s.design.stop_utc == hour(60)
    assert s.active_periods == (2, 3) and s.kind == "hurricane"
    f = family(items)
    with pytest.raises(ValueError, match="stranded re-entry"):
        f.engine("hurricane")
    e = f.engine("hurricane", dated_rebooking=DatedRebookingPolicy(72, 20))
    tape = e.tape
    g = tape.gumbel.copy()
    for t in (0, 1):
        for slot, m in enumerate(tape.arrival_marks[t]):
            products = [
                j
                for j in e.compiled.market_products[m.market]
                if e.compiled.spec.products[j].carrier == "B"
            ]
            if products:
                g[t, slot, :] = -1e9
                g[t, slot, products[0]] = 1e9
    g.setflags(write=False)
    e.reset(tape=replace(tape, gumbel=g))
    e.step(native.actions(e))
    e.step(native.actions(e))
    assert e.world.t == 2 and e._passengers
    assert all(p.first_stranded_at == 2 for p in e._passengers.values())
    assert any(cs.refunds > 0 for cs in e.world.carriers)
    assert all(e.request_coordinate(root).potential_slot >= 0 for root in e._passengers)
    e.step(native.actions(e))
    assert e.world.t == 3
    assert all(r["removed"] == r["restored"] for r in e.capacity_window_balance())
    while not e.terminal():
        e.step(native.actions(e))
    assert (
        e.conservation()["original_booked_passengers"]
        == e.conservation()["active_plus_pending_plus_unserved"]
    )


def test_snapshot_branch_reset_and_cross_demand_contract_refusal():
    f = family()
    e = f.engine("surge")
    e.step(native.actions(e))
    snap = e.snapshot()
    b = e.branch()
    b.step(native.actions(b))
    assert e.world.t == 1 and b.world.t == 2
    b.restore(snap)
    assert pickle.dumps(b.snapshot()) == pickle.dumps(snap)
    other = f.engine("control")
    with pytest.raises(ValueError, match="demand/date/composite"):
        other.restore(snap)
    b.reset(tape=f.bank.project("surge"))
    assert b.world.t == 0


@pytest.mark.parametrize(
    "change",
    [
        {"surge": ".6"},
        {"reveal_days": 2},
        {"reveal_days": True},
        {"markets": ()},
        {"post_segment_weights": (("business", ".6"), ("leisure", ".2"))},
        {"post_segment_weights": (("business", 1), ("business", 0))},
    ],
)
def test_no_silent_third_magnitude_mix_normalization_or_missing_market_support(change):
    c, _, _ = inputs()
    with pytest.raises((ValueError, TypeError)):
        replace(design(c), **change)


def test_missing_segments_precohort_reveal_mixed_classes_and_conflicting_family_refused():
    c, clock, dep = inputs()
    with pytest.raises(ValueError, match="EVERY compiled segment"):
        module.compile_dated_demand(
            c, clock, dep, replace(design(c), post_segment_weights=(("business", 1),))
        )
    with pytest.raises(ValueError, match="pre-cohort"):
        module.compile_dated_demand(c, clock, dep, replace(design(c), reveal_days=7))
    mixed = compile_market(
        replace(
            c.spec,
            segments=(
                replace(
                    c.spec.segments[0], mixed={"alpha": {"mean": 0.02, "sd": 0.001}}
                ),
                c.spec.segments[1],
            ),
        )
    )
    with pytest.raises(ValueError, match="finite latent classes"):
        module.compile_dated_demand(mixed, clock, dep, design(mixed))
    items = scenarios()
    items["other"] = replace(
        items["control"],
        clock=UtcBookingClock(
            tuple(t + timedelta(minutes=1) for t in clock.boundaries)
        ),
    )
    with pytest.raises(ValueError, match="share base spec"):
        family(items)


def test_composite_wrong_carrier_kind_or_mismatched_reference_reveal_refused():
    c, clock, dep = inputs()
    d = design(c)
    for cap in [
        B1DatedShock("outage", "rival_outage", "B_OH0", 3, ".3", 1, "B"),
        B1DatedShock("weather", "weather", "B_OH0", 1, ".3", 1),
    ]:
        with pytest.raises(ValueError):
            module.compile_dated_hurricane(
                c, clock, dep, scenario_id="h", capacity=cap, demand=d
            )


@pytest.mark.parametrize("lead", [14, 7, 3, 1])
def test_every_registered_departure_lead_and_caller_duration_maps_exact_dates(lead):
    c, _, dep = inputs()
    depart = BASE + timedelta(days=20)
    dep = tuple((k, depart) for k, v in dep)
    reveal = depart - timedelta(days=lead)
    clock = UtcBookingClock(
        (
            BASE,
            reveal,
            reveal + timedelta(hours=6),
            reveal + timedelta(hours=12),
            depart - timedelta(hours=6),
            depart - timedelta(hours=1),
        )
    )
    d = replace(design(c), reveal_days=lead, stop_utc=reveal + timedelta(hours=12))
    out = module.compile_dated_demand(c, clock, dep, d)
    assert out.reveal == reveal and out.active_periods == (1, 2)
    assert depart - out.reveal == timedelta(days=lead)
    assert out.design.stop_utc - out.reveal == timedelta(hours=12)


def test_exact_date_offset_aliases_and_real_DST_reveal_preserved():
    from zoneinfo import ZoneInfo

    c, clock, dep = inputs()
    ny = ZoneInfo("America/New_York")
    d = design(c, stop=48)
    a = module.compile_dated_demand(c, clock, dep, d)
    aliases = tuple((k, v.astimezone(ny).isoformat()) for k, v in dep)
    b = module.compile_dated_demand(
        c, clock, aliases, replace(d, stop_utc=d.stop_utc.astimezone(ny))
    )
    assert a.contract_hash == b.contract_hash
    assert d.stop_utc - a.reveal == timedelta(hours=24)
    assert d.stop_utc.astimezone(ny).utcoffset() != a.reveal.astimezone(ny).utcoffset()


def test_nominal_class_ordering_has_no_implicit_baseline_weight_normalization():
    c, clock, dep = inputs()
    d = replace(
        c.spec.demand,
        segment_weights=PeriodSchedule.parse({"constant": (0.8000000000000001, 0.2)}),
    )
    c = compile_market(replace(c.spec, demand=d))
    with pytest.raises(ValueError, match="canonical segment weights"):
        module.compile_dated_demand(c, clock, dep, design(c))


def test_hurricane_snapshot_contract_differs_even_with_same_compiled_demand_hash():
    f = family(scenarios(composite=True))
    p = DatedRebookingPolicy(72, 20)
    a = f.engine("surge", dated_rebooking=p)
    b = f.engine("hurricane", dated_rebooking=p)
    assert a.compiled.spec_hash == b.compiled.spec_hash
    with pytest.raises(ValueError, match="demand/date/composite"):
        a.restore(b.snapshot())
    branch = b.branch()
    branch.step(native.actions(branch))
    assert b.world.t == 0 and branch.world.t == 1
    branch.restore(b.snapshot())
    assert pickle.dumps(branch.snapshot()) == pickle.dumps(b.snapshot())


def test_bernoulli_spec_is_not_silently_reinterpreted_as_hourly_poisson_demand():
    from airline_market.pipeline.screen_generators import screen_spec

    _, clock, dep = inputs()
    c = compile_market(screen_spec({"T": 5, "actions": 2, "n_spokes": 1}))
    with pytest.raises(ValueError, match="Bernoulli conversion"):
        module.compile_dated_demand(c, clock, dep, design(c))
