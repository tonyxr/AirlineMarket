"""A.10/A.5/B0/V9: NEW conditional post-booking service references."""

import importlib.util
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
from airline_market.sim.service_observations import (
    service_observation_journal,
    service_observation_summary,
)
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from tests.unit import test_dated_demand as dated
from tests.unit import test_rationing_native as native

if os.environ.get("CURRENT_SERVICE_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_current_service", os.environ["CURRENT_SERVICE_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.sim import current_itinerary_service as module


def fixture(*, replacement=False, reentry=True, buyers=1):
    c, clock, dep = dated.inputs()
    scenario = compile_dated_demand(c, clock, dep)
    kwargs = {}
    if replacement:
        scenario = replace(
            scenario,
            physical_shocks=(
                B1DatedShock(
                    "new-current-service-exit",
                    "rival_exit",
                    "B_OH0",
                    3,
                    rival_carrier="B",
                ),
            ),
        )
        if reentry:
            kwargs["dated_rebooking"] = DatedRebookingPolicy(
                72, 20, 100, interline=True
            )
    e = OrdinaryCohortMarketEngine(scenario, **kwargs)
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(71974, "unit/V9_current_itinerary_service")
    )
    j = next(
        i
        for i, p in enumerate(c.spec.products)
        if p.carrier == ("B" if replacement else "A") and len(p.legs) == 2
    )
    m = int(c.product_market[j])
    g = tape.gumbel.copy()
    g[0, : buyers + 1, :] = -1e6
    g[0, :buyers, j] = 1e6
    g[0, buyers, -1] = 1e6
    g.setflags(write=False)
    marks = tuple(ArrivalMark(0.1 + i * 0.1, m, 0, 0, 0) for i in range(buyers + 1))
    e.reset(
        tape=replace(
            tape, arrival_marks=(marks, *(() for _ in tape.arrival_marks[1:])), gumbel=g
        )
    )
    while not e.terminal():
        e.step(native.actions(e))
    panel = ordinary_person_panel(e.ordinary_periods(), original_passenger_accounts(e))
    journal = service_observation_journal(
        c,
        panel,
        clock,
        dep,
        definition="NEW native booking source for separate declared deterministic service model",
        observation_scope="declared-individual-leg-observations",
        context_row_limit=10000,
        observation_limit=100,
    )
    current = [p for p in journal.context["persons"] if p["current_ticket"] is not None]
    operations = []
    for leg, scheduled in dep:
        extra = 4 if current and leg == current[0]["required_legs"][-1] else 0
        departure = scheduled + timedelta(hours=extra)
        operations.append(
            module.LegOperation(
                leg, "operates", 20, departure, departure + timedelta(hours=1), None
            )
        )
    intents = tuple(
        (p["current_ticket"], leg, True) for p in current for leg in p["required_legs"]
    )
    orders = tuple(
        (leg, tuple(p["current_ticket"] for p in current if leg in p["required_legs"]))
        for leg, _ in dep
    )
    connections = tuple(
        (p["current_ticket"], a, b, 3600_000000)
        for p in current
        for a, b in zip(p["required_legs"], p["required_legs"][1:])
    )
    return e, journal, operations, intents, orders, connections


def run(f, **changes):
    e, journal, operations, intents, orders, connections = f
    args = dict(
        compiled=e.compiled,
        journal=journal,
        operations=operations,
        boarding_intents=intents,
        boarding_orders=orders,
        connections=connections,
        definition="NEW explicit deterministic service conditional on supplied plans; no native booking or welfare adoption",
        support_limit=10000,
    )
    args.update(changes)
    return module.execute_current_itinerary_service(**args)


def person(f):
    return next(p for p in f[1].context["persons"] if p["current_ticket"] is not None)


def change_leg(f, k, **changes):
    return [replace(o, **changes) if o.leg == k else o for o in f[2]]


def summary(r):
    return service_observation_summary(
        r.journal, as_of=max(module.utc(s["at"]) for s in r.steps)
    )


def test_actual_boarding_then_arrival_generates_delivery_and_preserves_source():
    f = fixture()
    e, j, *_ = f
    before = pickle.dumps((e.world, e.trace, e.metrics, np.random.get_state()))
    r = run(f)
    assert pickle.dumps((e.world, e.trace, e.metrics, np.random.get_state())) == before
    assert j.observations == ()
    p = person(f)
    first = p["required_legs"][0]
    d = next(s for s in r.steps if s["leg"] == first and s["kind"] == "departure")
    a = next(s for s in r.steps if s["leg"] == first and s["kind"] == "arrival")
    assert d["boarded_count"] == 1 and not d["new_observation_ids"]
    assert d["in_flight"] and not any(
        k == first for _, k, _ in d["completed_person_legs"]
    )
    assert len(a["new_observation_ids"]) == 1
    assert all(o.outcome == "delivered" for o in r.journal.observations)
    assert len(summary(r)["persons"]) == 2
    assert r.context["new_deterministic_service_state_machine_executed"]
    assert r.context["post_booking_payments_refunds_or_lifecycle_utility"] is None


def test_no_show_is_not_boarded_despite_native_active_ticket():
    f = fixture()
    intents = tuple((rid, leg, False) for rid, leg, _ in f[3])
    r = run(f, boarding_intents=intents)
    assert all(o.outcome == "not_delivered" for o in r.journal.observations)
    assert all(s["boarded_count"] == 0 for s in r.steps if s["kind"] == "departure")


@pytest.mark.parametrize(
    "delta,reason", [(0, "delivered"), (-1, "minimum_connection_not_met")]
)
def test_exact_connection_minimum_microsecond_boundary(delta, reason):
    f = fixture()
    p = person(f)
    first, second = p["required_legs"]
    arrival = next(o.arrival for o in f[2] if o.leg == first)
    departure = arrival + timedelta(hours=1, microseconds=delta)
    r = run(
        f,
        operations=change_leg(
            f, second, departure=departure, arrival=departure + timedelta(hours=1)
        ),
    )
    o = next(o for o in r.journal.observations if o.leg == second)
    assert (o.outcome == "delivered") == (reason == "delivered")
    assert (
        "boarded_flight_arrival" if reason == "delivered" else reason
    ) in o.source_ref


def test_same_instant_arrival_precedes_departure_with_explicit_zero_minimum():
    f = fixture()
    p = person(f)
    first, second = p["required_legs"]
    arrival = next(o.arrival for o in f[2] if o.leg == first)
    r = run(
        f,
        operations=change_leg(
            f, second, departure=arrival, arrival=arrival + timedelta(hours=1)
        ),
        connections=((p["current_ticket"], first, second, 0),),
    )
    assert all(o.outcome == "delivered" for o in r.journal.observations)
    simultaneous = [s["kind"] for s in r.steps if s["at"] == arrival.isoformat()]
    assert simultaneous.index("arrival") < simultaneous.index("departure")


def test_missed_connection_while_first_leg_still_in_flight():
    f = fixture()
    first, second = person(f)["required_legs"]
    late = next(o.arrival for o in f[2] if o.leg == second) + timedelta(hours=2)
    r = run(f, operations=change_leg(f, first, arrival=late))
    assert (
        next(o for o in r.journal.observations if o.leg == first).outcome == "delivered"
    )
    assert (
        "prior_service_not_completed"
        in next(o for o in r.journal.observations if o.leg == second).source_ref
    )


def test_declared_priority_seat_debits_and_no_show_retained():
    f = fixture(buyers=2)
    first, second = person(f)["required_legs"]
    orders = tuple((leg, tuple(reversed(ids))) for leg, ids in f[4])
    r = run(f, operations=change_leg(f, first, seats=1), boarding_orders=orders)
    d = next(s for s in r.steps if s["leg"] == first and s["kind"] == "departure")
    assert d["decisions"] == (
        (orders[[k for k, _ in orders].index(first)][1][0], "boarded"),
        (orders[[k for k, _ in orders].index(first)][1][1], "no_seat"),
    )
    assert d["boarded_count"] + d["unused_departure_seats"] == d["physical_seats"] == 1
    assert len(r.journal.observations) == 4
    assert (
        sum(
            o.outcome == "delivered" and o.leg == second for o in r.journal.observations
        )
        == 1
    )
    intents = tuple(
        (rid, leg, rid != orders[[k for k, _ in orders].index(first)][1][0])
        for rid, leg, _ in f[3]
    )
    r2 = run(
        f,
        operations=change_leg(f, first, seats=1),
        boarding_orders=orders,
        boarding_intents=intents,
    )
    d2 = next(s for s in r2.steps if s["leg"] == first and s["kind"] == "departure")
    assert tuple(reason for _, reason in d2["decisions"]) == (
        "no_boarding_intent",
        "boarded",
    )


def test_zero_seats_does_not_imply_flight_cancelled():
    f = fixture()
    first = person(f)["required_legs"][0]
    r = run(f, operations=change_leg(f, first, seats=0))
    assert (
        "no_seat"
        in next(o for o in r.journal.observations if o.leg == first).source_ref
    )
    assert any(s["kind"] == "arrival" and s["leg"] == first for s in r.steps)


@pytest.mark.parametrize("which", [0, 1])
def test_cancellation_and_partial_itinerary_are_actual_service_transitions(which):
    f = fixture()
    leg = person(f)["required_legs"][which]
    op = next(o for o in f[2] if o.leg == leg)
    r = run(
        f,
        operations=change_leg(
            f,
            leg,
            status="cancelled",
            departure=None,
            arrival=None,
            cancellation=op.departure - timedelta(minutes=1),
        ),
    )
    assert (
        "operation_cancelled"
        in next(o for o in r.journal.observations if o.leg == leg).source_ref
    )
    if which == 1:
        assert sum(o.outcome == "delivered" for o in r.journal.observations) == 1
    assert any(
        s["kind"] == "cancellation" and s["physical_seats"] == 20 for s in r.steps
    )


def test_actual_interline_replacement_uses_same_root_and_current_ticket():
    f = fixture(replacement=True)
    p = person(f)
    assert p["current_ticket"] != p["original_record"]
    r = run(f)
    assert all(
        o.ticket_record == p["current_ticket"] and o.person_key == p["person_key"]
        for o in r.journal.observations
    )
    assert len(summary(r)["persons"]) == 2


def test_no_reentry_unreserved_and_outside_rows_remain_without_invented_service():
    f = fixture(replacement=True, reentry=False)
    r = run(f)
    assert not r.journal.observations
    s = summary(r)
    assert len(s["persons"]) == 2
    assert all(p["observed_current_itinerary_complete"] is None for p in s["persons"])


@pytest.mark.parametrize(
    "field", ["operations", "boarding_intents", "boarding_orders", "connections"]
)
def test_complete_support_required_before_any_transition(field, monkeypatch):
    f = fixture()
    values = {
        "operations": f[2],
        "boarding_intents": f[3],
        "boarding_orders": f[4],
        "connections": f[5],
    }
    monkeypatch.setattr(
        module,
        "record_leg_service",
        lambda *a: pytest.fail("must fail before observation"),
    )
    with pytest.raises(ValueError):
        run(f, **{field: values[field][1:]})


@pytest.mark.parametrize("what", ["support", "observation"])
def test_all_support_limits_fail_before_observations(what, monkeypatch):
    f = fixture()
    monkeypatch.setattr(
        module,
        "record_leg_service",
        lambda *a: pytest.fail("must fail before observation"),
    )
    changes = (
        {"support_limit": 1}
        if what == "support"
        else {
            "journal": replace(f[1], context={**f[1].context, "observation_limit": 1})
        }
    )
    with pytest.raises(RuntimeError):
        run(f, **changes)


@pytest.mark.parametrize(
    "change",
    [
        dict(seats=True),
        dict(seats=-1),
        dict(status="unknown"),
        dict(arrival=None),
        dict(cancellation="2026-03-08T00:00:00+00:00"),
    ],
)
def test_invalid_operations_refused(change):
    f = fixture()
    with pytest.raises(ValueError):
        run(f, operations=change_leg(f, person(f)["required_legs"][0], **change))


def test_owned_immutable_full_inputs_steps_and_exports():
    f = fixture()
    r = run(f)
    out = module.current_service_record(r)
    out["context"]["definition"] = "changed export"
    f[2].clear()
    assert r.context["operations"] and r.context["definition"] != "changed export"
    with pytest.raises(TypeError):
        r.steps[0]["kind"] = "changed"
    with pytest.raises(ValueError):
        run(f, journal=r.journal)


def test_complete_model_binding_cannot_be_changed_by_typed_context():
    f = fixture()
    bad = replace(
        f[1], context={**f[1].context, "spec_hash": "not the full native model"}
    )
    with pytest.raises(ValueError):
        run(f, journal=bad)


def test_late_second_leg_cancellation_preserves_first_actual_arrival():
    f = fixture()
    first, second = person(f)["required_legs"]
    op = next(o for o in f[2] if o.leg == second)
    late = op.arrival + timedelta(hours=20)
    r = run(
        f,
        operations=change_leg(
            f,
            second,
            status="cancelled",
            departure=None,
            arrival=None,
            cancellation=late,
        ),
    )
    assert tuple(o.outcome for o in r.journal.observations) == (
        "delivered",
        "not_delivered",
    )
    assert r.journal.observations[0].leg == first
    assert r.journal.observations[-1].occurred_at == late
    assert summary(r)["persons"][0]["observed_current_itinerary_complete"] is False


def test_transition_failure_does_not_publish_partial_result_or_mutate_sources(
    monkeypatch,
):
    f = fixture()
    e, j, *_ = f
    before = pickle.dumps((e.world, e.trace, e.metrics, j))
    count = 0
    original = module.record_leg_service

    def fail_second(*args):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("NEW injected transition failure")
        return original(*args)

    monkeypatch.setattr(module, "record_leg_service", fail_second)
    with pytest.raises(RuntimeError, match="NEW injected"):
        run(f)
    assert count == 2
    assert pickle.dumps((e.world, e.trace, e.metrics, j)) == before
