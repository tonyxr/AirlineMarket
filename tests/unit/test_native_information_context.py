"""A.10/A.5/A.14/B0/V9: NEW genuine source/complete context join units."""

import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from airline_market.core.hashing import jsonable
from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.dated_demand import compile_dated_demand
from airline_market.policy.dated_rescue import DatedRebookingPolicy
from airline_market.policy.dated_reserve import UtcBookingClock
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.sim.proof_information import ProofInformationMarketEngine
from airline_market.sim.received_information import (
    DisclosureField,
    ReceivedDisclosure,
    append_received_disclosures,
    received_information_journal,
)
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

if os.environ.get("NATIVE_CONTEXT_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_native_context", os.environ["NATIVE_CONTEXT_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.sim import native_information_context as module


def source(
    *,
    kind="exit",
    interline=False,
    count=2,
    outside=False,
    terminal_proof=False,
    steps=5,
):
    # NEW local construction, no imported/called prior fixture helper.
    spec = screen_spec(
        {
            "T": 5,
            "actions": 2,
            "n_spokes": 1,
            "capacity": 3 if kind == "recurrent" else 20,
            "capacity_support": [3, 4],
        }
    )
    demand = replace(
        spec.demand,
        mode="poisson",
        arrival_probability=None,
        market_weights=None,
        intensity={m.id: PeriodSchedule.parse({"constant": 2.0}) for m in spec.markets},
    )
    c = compile_market(replace(spec, demand=demand))
    start = datetime(2026, 12, 8, tzinfo=timezone.utc)
    clock = UtcBookingClock(
        tuple(start + timedelta(hours=h) for h in (0, 12, 24, 36, 48, 60))
    )
    departures = tuple(
        (
            leg.id,
            start
            + timedelta(
                hours=84 if leg.id == "B_OH0" else 120 if leg.id == "B_HD0" else 108
            ),
        )
        for leg in c.spec.legs
    )
    scenario = compile_dated_demand(c, clock, departures)
    if kind == "exit":
        shocks = (B1DatedShock("exit", "rival_exit", "B_OH0", 3, rival_carrier="B"),)
    elif kind == "recurrent":
        shocks = (
            B1DatedShock("weather", "weather", "B_OH0", 3, ".6", 1),
            B1DatedShock("later-exit", "rival_exit", "B_HD0", 3, rival_carrier="B"),
        )
    else:
        shocks = ()
    e = ProofInformationMarketEngine(
        replace(scenario, physical_shocks=shocks),
        dated_rebooking=DatedRebookingPolicy(72, 20, 100, interline=interline),
    )
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(81974, "unit/V9_native_information_context")
    )
    marks = tuple(ArrivalMark((i + 1) / (count + 1), 0, 0, 0, 0) for i in range(count))
    g = tape.gumbel.copy()
    g[:] = -1e9
    g[:, :, -1] = 0
    if count:
        g[:, :, c.product_index["B_OH0"]] = 1e9
        if outside:
            g[:, :, -1] = 2e9
        elif count > 1 and kind != "recurrent":
            g[:, count - 1, -1] = 2e9
    g.setflags(write=False)
    e.reset(
        tape=replace(
            tape, arrival_marks=tuple(marks for _ in tape.arrival_marks), gumbel=g
        )
    )
    actions = {
        carrier.code: PriceAction({j: 200 if i == 0 else 300 for j in c.products_of[i]})
        for i, carrier in enumerate(c.spec.carriers)
    }
    for t in range(steps):
        if t == 1:
            for passenger in e._passengers.values():
                e.present_cancelled_itinerary(passenger.original, airport_counter=True)
                e.present_cancelled_itinerary(passenger.original, airport_counter=True)
        e.step(actions)
    if terminal_proof:
        for passenger in e._passengers.values():
            e.present_cancelled_itinerary(passenger.original)
    j = received_information_journal(
        c,
        e.information_periods(),
        clock,
        [DisclosureField("notice", "text", "explicit later received notice", "text")],
        definition="separate declared received messages",
        unobserved_definition="native captured context is not authenticated knowledge",
        row_limit=1000000,
    )
    histories = e.snapshot().native.histories
    return e, (
        j,
        e.own_information_periods(),
        e.proof_information_periods(),
        histories,
        e.presentation_calls(),
    )


def join(inputs, *, limit=5000000):
    return module.native_information_contexts(
        *inputs, definition="NEW complete captured native streams", row_limit=limit
    )


@pytest.mark.parametrize("interline", [False, True])
def test_new_complete_native_original_people_and_context_order(interline, monkeypatch):
    e, inputs = source(interline=interline, terminal_proof=True)
    saved = pickle.dumps(e.snapshot())
    rng = pickle.dumps(np.random.get_state())

    def fail(*args, **kwargs):
        raise AssertionError("pure join called source producer")

    for name in (
        "step",
        "information_periods",
        "own_information_periods",
        "proof_information_periods",
        "presentation_calls",
    ):
        monkeypatch.setattr(e, name, fail)
    p = join(inputs)
    assert len(p.people) == 10 and len(p.periods) == 5
    original = p.people[0]
    assert original["context_coordinates"][0] == (0, "ordinary", 0)
    assert (
        p.people[1]["analyst_root"] is None
        and len(p.people[1]["context_coordinates"]) == 1
    )
    rows = tuple(r for period in p.periods for r in period.reentry_contexts)
    assert len(rows) == (1 if interline else 4)
    assert [r.person_decision for r in rows] == list(range(1, len(rows) + 1))
    assert rows[0].view["earlier_proof"]["native_counter_this_period"] is True
    assert len(rows[0].view["earlier_proof"]["earlier_presentation_calls"]) == 2
    if not interline:
        assert rows[1].view["earlier_proof"]["native_counter_this_period"] is False
        assert len(rows[-1].view["earlier_own"]["own_facts"]) > len(
            rows[0].view["earlier_own"]["own_facts"]
        )
    assert (
        pickle.dumps(e.snapshot()) == saved
        and pickle.dumps(np.random.get_state()) == rng
    )
    assert module.native_information_context_record(p)["scientific_budget"] is None
    json.dumps(module.native_information_context_record(p), allow_nan=False)


def test_new_recurrent_native_cancellation_same_person_first_expiry():
    e, inputs = source(kind="recurrent", count=3)
    p = join(inputs)
    assert len(p.people) == 15
    recurrent = [
        row
        for row in inputs[3]
        if sum(f.kind == "native_ticket_cancelled" for f in row[1]) > 1
    ]
    assert recurrent
    for root, _facts, _tickets, key in recurrent:
        contexts = [
            r
            for period in p.periods
            for r in period.reentry_contexts
            if r.analyst_root == root
        ]
        assert all(r.analyst_original_key == key for r in contexts)
        assert (
            len({r.view["earlier_own"]["native_exact_expiry_utc"] for r in contexts})
            == 1
        )
    assert len(e.rebooking_history) == sum(len(x.reentry_contexts) for x in p.periods)


@pytest.mark.parametrize("kwargs", [{"count": 0}, {"outside": True}, {"kind": "none"}])
def test_new_empty_outside_unstranded_full_support(kwargs):
    _e, inputs = source(**kwargs)
    p = join(inputs)
    assert len(p.periods) == 5 and not any(x.reentry_contexts for x in p.periods)
    assert len(p.people) == len(inputs[0].person_keys)
    assert all(len(person["context_coordinates"]) == 1 for person in p.people)
    if kwargs.get("count") == 0:
        assert p.people == () and all(x.ordinary_contexts == () for x in p.periods)
    if kwargs.get("outside"):
        assert all(person["analyst_root"] is None for person in p.people)


def test_narrow_context_excludes_analyst_class_outcome_future_law_and_later_calls():
    _e, inputs = source(terminal_proof=True)
    p = join(inputs)
    row = p.periods[1].reentry_contexts[0]
    view = module.native_decision_context_view(row)
    assert (
        view["current_effective_reentry_menu"] is None
        and view["known_private_type"] is None
    )
    assert view["received_disclosures_or_complete_knowledge"] is None
    text = json.dumps(view)
    for forbidden in (
        "analyst_original_key",
        "analyst_root",
        "taste_class",
        "consumer_surplus",
        "gumbel",
        "replacement_record",
        "future_program",
    ):
        assert forbidden not in text
    assert (
        len(inputs[-1]) == 3
        and len(view["earlier_proof"]["earlier_presentation_calls"]) == 2
    )
    assert all(
        f["native_period"] < 5
        for f in view["earlier_proof"]["earlier_presentation_calls"]
    )
    # Current search's outside outcome is retained in ANALYST final facts only.
    assert view["earlier_own"]["own_facts"][-1]["kind"] == "native_ticket_cancelled"


def test_later_received_messages_retained_separately_no_earlier_native_knowledge_fill():
    _e, inputs = source()
    j = inputs[0]
    t = j.as_of + timedelta(minutes=1)
    d = ReceivedDisclosure(
        "later-only",
        j.person_keys[0],
        "notice",
        "later declared notice",
        t,
        t,
        t,
        "NEW explicit later message",
    )
    later = append_received_disclosures(j, [d], as_of=t)
    p = join((later, *inputs[1:]))
    assert len(p.analyst_context["supplied_source"]["disclosures"]) == 1
    assert "later declared notice" not in json.dumps(
        [
            module.native_decision_context_view(r)
            for period in p.periods
            for r in (*period.reentry_contexts, *period.ordinary_contexts)
        ]
    )


@pytest.mark.parametrize(
    "failure",
    [
        "period",
        "attempt",
        "history",
        "root",
        "ticket",
        "first_fare",
        "fact_order",
        "current_outcome",
        "proof_status",
        "proof_prefix",
        "proof_calls",
        "clock",
        "expiry",
    ],
)
def test_complete_source_fact_context_and_proof_refusals(failure):
    _e, inputs = source()
    j, own, proof, histories, calls = inputs
    own, proof, histories, calls = list(own), list(proof), list(histories), list(calls)
    if failure == "period":
        own.pop()
    elif failure == "attempt":
        own[1] = replace(own[1], reentry_rows=())
        proof[1] = replace(proof[1], reentry_rows=())
    elif failure == "history":
        histories.clear()
    elif failure == "root":
        root, facts, tickets, key = histories[0]
        histories[0] = root, facts, tickets, (0, 1, 0)
    elif failure == "ticket":
        root, facts, tickets, key = histories[0]
        histories[0] = root, facts, (), key
    elif failure == "first_fare":
        root, facts, tickets, key = histories[0]
        histories[0] = root, (replace(facts[0], fare=123.0), *facts[1:]), tickets, key
    elif failure == "fact_order":
        root, facts, tickets, key = histories[0]
        histories[0] = root, (replace(facts[0], ordinal=99), *facts[1:]), tickets, key
    elif failure == "proof_calls":
        calls.pop(0)
    else:
        if failure in {"current_outcome", "clock", "expiry"}:
            row = own[1].reentry_rows[0]
            v = row.view
            if failure == "current_outcome":
                v = replace(v, own_facts=histories[0][1][:3])
            elif failure == "clock":
                v = replace(v, compiled_opportunity_utc=j.source_closed_at.isoformat())
            else:
                v = replace(v, native_exact_expiry_utc="2026-12-08T00:00:00+00:00")
            own[1] = replace(own[1], reentry_rows=(replace(row, view=v),))
        else:
            row = proof[2].reentry_rows[0]
            v = row.view
            if failure == "proof_status":
                v = replace(v, native_proof_on_file=False)
            else:
                v = replace(
                    v,
                    earlier_presentation_calls=(),
                    native_proof_on_file=False,
                    native_counter_this_period=False,
                )
            proof[2] = replace(proof[2], reentry_rows=(replace(row, view=v),))
    values = j, own, proof, histories, calls
    saved = pickle.dumps(values)
    with pytest.raises((ValueError, TypeError)):
        join(values)
    assert pickle.dumps(values) == saved


def test_complete_owned_outputs_limits_and_atomic_pure_failure(monkeypatch):
    _e, inputs = source()
    p = join(inputs)
    units = module._units(jsonable(p))
    join(inputs, limit=units)
    with pytest.raises(module.NativeInformationContextLimit):
        join(inputs, limit=units - 1)
    out = module.native_information_context_record(p)
    out["result"]["analyst_context"]["supplied_source"]["definition"] = "changed export"
    assert (
        module.native_information_context_record(p)["result"]["analyst_context"][
            "supplied_source"
        ]["definition"]
        != "changed export"
    )
    with pytest.raises(TypeError):
        p.periods[1].reentry_contexts[0].view["known_private_type"] = "invented"
    saved = pickle.dumps(inputs)

    def fail(*args):
        raise RuntimeError("NEW injected context serialization failure")

    monkeypatch.setattr(module, "own_reentry_information_view", fail)
    with pytest.raises(RuntimeError, match="serialization"):
        join(inputs)
    assert pickle.dumps(inputs) == saved


def test_new_v3_exact_first_expiry_cannot_extend_with_same_native_cutoff():
    _e, inputs = source()
    j, own, proof, histories, calls = inputs
    own = list(own)
    row = own[2].reentry_rows[0]
    later = (
        module.utc(row.view.native_exact_expiry_utc) + timedelta(minutes=1)
    ).isoformat()
    changed = replace(row, view=replace(row.view, native_exact_expiry_utc=later))
    own[2] = replace(own[2], reentry_rows=(changed,))
    with pytest.raises(ValueError, match="cannot change"):
        join((j, own, proof, histories, calls))


def test_new_v3_complete_partial_and_zero_closed_prefix_no_future_people():
    for steps in (0, 2):
        _e, inputs = source(steps=steps)
        panel = join(inputs)
        assert len(panel.periods) == steps
        assert len(panel.people) == 2 * steps
        assert all(row["analyst_original_key"][0] < steps for row in panel.people)
        assert all(
            f.native_period < steps
            for _root, facts, _tickets, _key in inputs[3]
            for f in facts
        )
