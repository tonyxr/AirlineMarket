"""A.10/A.5/B0/V9: NEW prospective individual-service observation references."""

import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.policy.dated_demand import compile_dated_demand
from airline_market.policy.dated_rescue import DatedRebookingPolicy
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.sim.ordinary_cohort import (
    OrdinaryCohortMarketEngine,
    ordinary_person_panel,
)
from airline_market.sim.passenger_accounts import original_passenger_accounts
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from tests.unit import test_dated_demand as dated
from tests.unit import test_rationing_native as native

if os.environ.get("SERVICE_OBSERVATIONS_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_service_observations", os.environ["SERVICE_OBSERVATIONS_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.sim import service_observations as module


def fixture(*, replacement=False, reentry=True):
    c, clock, dep = dated.inputs()
    scenario = compile_dated_demand(c, clock, dep)
    kwargs = {}
    if replacement:
        scenario = replace(
            scenario,
            physical_shocks=(
                B1DatedShock(
                    "new-service-source-exit",
                    "rival_exit",
                    "B_OH0",
                    3,
                    rival_carrier="B",
                ),
            ),
        )
        kwargs["dated_rebooking"] = DatedRebookingPolicy(72, 20, 100, interline=True)
    if not reentry:
        kwargs.pop("dated_rebooking", None)
    e = OrdinaryCohortMarketEngine(scenario, **kwargs)
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(70974, "unit/V9_service_observations")
    )
    j = next(
        i
        for i, p in enumerate(c.spec.products)
        if p.carrier == ("B" if replacement else "A") and len(p.legs) == 2
    )
    m = int(c.product_market[j])
    g = tape.gumbel.copy()
    g[0, :2, :] = -1e6
    g[0, 0, j] = 1e6
    g[0, 1, -1] = 1e6
    g.setflags(write=False)
    marks = (ArrivalMark(0.1, m, 0, 0, 0), ArrivalMark(0.2, 0, 0, 0, 0))
    e.reset(
        tape=replace(
            tape, arrival_marks=(marks, *(() for _ in tape.arrival_marks[1:])), gumbel=g
        )
    )
    while not e.terminal():
        e.step(native.actions(e))
    accounts = original_passenger_accounts(e)
    panel = ordinary_person_panel(e.ordinary_periods(), accounts)
    return e, panel, clock, dep


def journal(e, panel, clock, dep, **changes):
    kwargs = {
        "definition": "NEW confirmed individual leg outcomes supplied after a genuine complete booking panel; declared observation sources, no native service engine or welfare adoption",
        "observation_scope": "declared-individual-leg-observations",
        "context_row_limit": 10000,
        "observation_limit": 100,
    }
    kwargs.update(changes)
    return module.service_observation_journal(e.compiled, panel, clock, dep, **kwargs)


def target(j):
    return next(p for p in j.context["persons"] if p["current_ticket"] is not None)


def event(j, leg=0, outcome="delivered", *, name="event", offset=1, recorded_offset=2):
    p = target(j)
    k = p["required_legs"][leg]
    departure = module.utc(dict(j.context["leg_departures"])[k])
    return module.LegServiceObservation(
        name,
        p["person_key"],
        p["current_ticket"],
        k,
        outcome,
        departure + timedelta(hours=offset),
        departure + timedelta(hours=recorded_offset),
        "NEW explicit unit source, not authenticated native service",
    )


def summary(j):
    return module.service_observation_summary(
        j,
        as_of=max(module.utc(t) for k, t in j.context["leg_departures"])
        + timedelta(days=10),
    )


def test_complete_native_booking_and_passed_departures_do_not_imply_any_service():
    e, panel, clock, dep = fixture()
    j = journal(e, panel, clock, dep)
    s = summary(j)
    assert s["person_rows"] == 2 and s["received_observations"] == 0
    assert all(p["observed_current_itinerary_complete"] is None for p in s["persons"])
    assert all(
        l["observed_outcome"] == "unknown"
        for p in s["persons"]
        for l in p["leg_outcomes"]
    )
    assert any(
        p["booking_state"] == "ordinary_outside" and p["current_ticket"] is None
        for p in s["persons"]
    )
    assert s["population_service_rate"] is s["total_lifecycle_surplus"] is None


def test_connection_requires_each_individual_leg_not_flight_or_ticket_existence():
    e, panel, clock, dep = fixture()
    j = journal(e, panel, clock, dep)
    one = module.record_leg_service(j, event(j))
    first = summary(one)
    assert (
        first["observed_complete_current_itineraries"] == 0
        and first["unknown_current_itinerary_outcomes"] == 2
    )
    complete = module.record_leg_service(
        one, event(one, 1, name="second", offset=3, recorded_offset=4)
    )
    assert summary(complete)["observed_complete_current_itineraries"] == 1
    assert summary(complete)["unknown_current_itinerary_outcomes"] == 1
    assert j.observations == () and len(one.observations) == 1


def test_explicit_partial_non_delivery_is_not_a_causal_denial_or_welfare_label():
    e, panel, clock, dep = fixture()
    j = journal(e, panel, clock, dep)
    j = module.record_leg_service(j, event(j))
    j = module.record_leg_service(
        j, event(j, 1, "not_delivered", name="missed", offset=3, recorded_offset=4)
    )
    s = summary(j)
    assert s["observed_incomplete_current_itineraries"] == 1
    assert all(
        p["causal_denial_or_misallocation"] is p["lifecycle_consumer_surplus"] is None
        for p in s["persons"]
    )


def test_late_information_keeps_event_and_record_times_without_reversing_service_order():
    e, panel, clock, dep = fixture()
    j = journal(e, panel, clock, dep)
    second = event(j, 1, name="second-first-reported", offset=3, recorded_offset=4)
    j = module.record_leg_service(j, second)
    first = event(j, 0, name="late-first", offset=1, recorded_offset=5)
    j = module.record_leg_service(j, first)
    assert summary(j)["observed_complete_current_itineraries"] == 1
    assert j.observations[0].occurred_at > j.observations[1].occurred_at


def test_later_unknown_source_report_retained_without_erasing_known_outcome():
    e, panel, clock, dep = fixture()
    j = journal(e, panel, clock, dep)
    j = module.record_leg_service(j, event(j))
    j = module.record_leg_service(
        j,
        event(j, outcome="unknown", name="later-unknown", offset=3, recorded_offset=4),
    )
    p = next(p for p in summary(j)["persons"] if p["current_ticket"] is not None)
    assert p["leg_outcomes"][0]["observed_outcome"] == "delivered"
    assert len(p["leg_outcomes"][0]["observations"]) == 2


def test_new_native_replacement_receives_events_on_same_original_person_only():
    e, panel, clock, dep = fixture(replacement=True)
    j = journal(e, panel, clock, dep)
    p = target(j)
    assert (
        p["current_ticket"] != p["original_record"] and len(j.context["persons"]) == 2
    )
    j = module.record_leg_service(j, event(j))
    assert summary(j)["person_rows"] == 2
    bad = replace(event(j, name="old-cancelled"), ticket_record=p["original_record"])
    with pytest.raises(ValueError, match="CURRENT"):
        module.record_leg_service(j, bad)


@pytest.mark.parametrize(
    "damage",
    [
        "person",
        "outside",
        "ticket",
        "leg",
        "duplicate_id",
        "known_again",
        "reverse_record",
        "before_occurrence",
        "before_departure",
        "naive",
        "reverse_legs",
    ],
)
def test_incompatible_observations_refused_with_previous_journal_unchanged(damage):
    e, panel, clock, dep = fixture()
    j = journal(e, panel, clock, dep)
    j = module.record_leg_service(j, event(j))
    good = event(j, 1, name="new", offset=3, recorded_offset=4)
    before = pickle.dumps(j)
    if damage == "person":
        good = replace(good, person_key=(99, 0, 0))
    if damage == "outside":
        good = replace(
            good,
            person_key=next(
                p["person_key"]
                for p in j.context["persons"]
                if p["current_ticket"] is None
            ),
        )
    if damage == "ticket":
        good = replace(good, ticket_record=999)
    if damage == "leg":
        good = replace(good, leg="not-a-leg")
    if damage == "duplicate_id":
        good = replace(good, event_id="event")
    if damage == "known_again":
        good = replace(event(j, name="again"), outcome="not_delivered")
    if damage == "reverse_record":
        good = replace(
            good,
            occurred_at=j.observations[0].occurred_at,
            recorded_at=j.observations[0].occurred_at,
        )
    if damage == "before_occurrence":
        good = replace(good, recorded_at=good.occurred_at - timedelta(seconds=1))
    if damage == "before_departure":
        good = replace(
            good,
            occurred_at=module.utc(dict(j.context["leg_departures"])[good.leg])
            - timedelta(seconds=1),
        )
    if damage == "reverse_legs":
        good = replace(
            good, occurred_at=j.observations[0].occurred_at - timedelta(minutes=1)
        )
    if damage == "naive":
        with pytest.raises(ValueError, match="aware"):
            replace(good, recorded_at=good.recorded_at.replace(tzinfo=None))
        return
    with pytest.raises(ValueError):
        module.record_leg_service(j, good)
    assert pickle.dumps(j) == before


@pytest.mark.parametrize(
    "damage",
    [
        "incomplete",
        "missing_person",
        "outside_count",
        "departure",
        "clock",
        "context_limit",
    ],
)
def test_full_panel_and_all_context_support_not_filled_or_reconstructed(damage):
    e, panel, clock, dep = fixture()
    p = dict(panel)
    kwargs = {}
    if damage == "incomplete":
        p["completed_native_periods"] -= 1
    if damage == "missing_person":
        p["persons"] = p["persons"][:-1]
    if damage == "outside_count":
        p["ordinary_outside"] += 1
    if damage == "departure":
        dep = dep[:-1]
    if damage == "clock":
        clock = module.UtcBookingClock(
            (
                *clock.boundaries[:-1],
                max(module.utc(t) for k, t in dep) + timedelta(days=1),
            )
        )
    if damage == "context_limit":
        kwargs["context_row_limit"] = 1
    with pytest.raises((ValueError, RuntimeError)):
        journal(e, p, clock, dep, **kwargs)


def test_asof_cannot_silently_filter_future_information_or_fabricate_prefix():
    e, panel, clock, dep = fixture()
    j = journal(e, panel, clock, dep)
    o = event(j)
    j = module.record_leg_service(j, o)
    with pytest.raises(ValueError, match="full supplied"):
        module.service_observation_summary(j, as_of=o.occurred_at)


def test_observation_limit_includes_unknown_reports_without_partial_or_cap_raise():
    e, panel, clock, dep = fixture()
    j = journal(e, panel, clock, dep, observation_limit=1)
    j = module.record_leg_service(j, event(j, outcome="unknown"))
    before = pickle.dumps(j)
    with pytest.raises(RuntimeError, match="ALL observation"):
        module.record_leg_service(
            j, event(j, 1, name="over", offset=3, recorded_offset=4)
        )
    assert pickle.dumps(j) == before


def test_owned_functional_branches_lossless_context_and_no_world_actor_or_rng_effect(
    monkeypatch,
):
    e, panel, clock, dep = fixture()
    world = pickle.dumps((e.world, e.trace(), e.metrics()))
    rng = pickle.dumps(np.random.get_state())
    j = journal(e, panel, clock, dep)

    def forbidden(*a, **k):
        raise AssertionError("no native producer during journal/summary")

    monkeypatch.setattr(e, "step", forbidden)
    monkeypatch.setattr(ExogenousTape, "generate", forbidden)
    a = module.record_leg_service(j, event(j))
    b = module.record_leg_service(
        j, event(j, outcome="not_delivered", name="other-branch")
    )
    assert a.observations != b.observations
    for item in (a, b):
        exported = module.service_journal_record(item)
        json.dumps(exported, allow_nan=False)
        exported["context"]["persons"] = ()
        assert len(item.context["persons"]) == 2 and summary(item)["person_rows"] == 2
        with pytest.raises(TypeError):
            item.context["persons"][0]["current_ticket"] = 999
    assert (
        pickle.dumps((e.world, e.trace(), e.metrics())) == world
        and pickle.dumps(np.random.get_state()) == rng
    )


def test_new_native_no_reentry_cancellation_keeps_person_and_unknown_future_service():
    e, panel, clock, dep = fixture(replacement=True, reentry=False)
    j = journal(e, panel, clock, dep)
    p = next(p for p in j.context["persons"] if p["original_record"] is not None)
    assert p["booking_state"] == "no_active_reservation" and p["current_ticket"] is None
    assert len(j.context["persons"]) == 2
    assert all(
        p["observed_current_itinerary_complete"] is None for p in summary(j)["persons"]
    )
    leg, departure = dep[0]
    o = module.LegServiceObservation(
        "past-canceled",
        p["person_key"],
        p["original_record"],
        leg,
        "not_delivered",
        departure + timedelta(hours=1),
        departure + timedelta(hours=2),
        "NEW declared source",
    )
    with pytest.raises(ValueError, match="CURRENT"):
        module.record_leg_service(j, o)


def test_new_confirmed_non_delivery_can_be_observed_before_departure_without_native_cash_change():
    e, panel, clock, dep = fixture()
    j = journal(e, panel, clock, dep)
    p = target(j)
    before = pickle.dumps(e.world)
    o = module.LegServiceObservation(
        "confirmed-before-departure",
        p["person_key"],
        p["current_ticket"],
        p["required_legs"][0],
        "not_delivered",
        clock.at(e.compiled.spec.horizon.n_periods) + timedelta(hours=1),
        clock.at(e.compiled.spec.horizon.n_periods) + timedelta(hours=2),
        "NEW explicit confirmed non-delivery observation, not a forecast",
    )
    j = module.record_leg_service(j, o)
    s = module.service_observation_summary(j, as_of=o.recorded_at)
    assert s["observed_incomplete_current_itineraries"] == 1
    assert not next(p for p in s["persons"] if p["current_ticket"] is not None)[
        "leg_outcomes"
    ][0]["departure_due"]
    assert pickle.dumps(e.world) == before
