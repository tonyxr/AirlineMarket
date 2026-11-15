"""A.10/A.5/B0/V9: NEW native-prefix/declared-receipt journal references."""

import importlib.util
import json
import os
import pickle
import struct
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.dated_demand import compile_dated_demand
from airline_market.policy.dated_rescue import DatedRebookingPolicy
from airline_market.policy.dated_reserve import UtcBookingClock
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.sim.ordinary_information import OrdinaryInformationMarketEngine
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

if os.environ.get("RECEIVED_INFORMATION_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_received_information", os.environ["RECEIVED_INFORMATION_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.sim import received_information as module


def source(*, count=2, outside=False, exit=False, interline=False, steps=3):
    # NEW construction adapted from the frozen source pattern, no old fixture helper.
    base = screen_spec(
        {
            "T": 3,
            "actions": 2,
            "n_spokes": 1,
            "capacity": 20,
            "capacity_support": [3, 4],
        }
    )
    demand = replace(
        base.demand,
        mode="poisson",
        arrival_probability=None,
        market_weights=None,
        intensity={m.id: PeriodSchedule.parse({"constant": 2.0}) for m in base.markets},
    )
    c = compile_market(replace(base, demand=demand))
    start = datetime(2026, 12, 1, tzinfo=timezone.utc)
    clock = UtcBookingClock(tuple(start + timedelta(hours=h) for h in (0, 12, 24, 36)))
    dep = tuple(
        (leg.id, start + timedelta(hours=84 if leg.id == "B_OH0" else 108))
        for leg in c.spec.legs
    )
    scenario = compile_dated_demand(c, clock, dep)
    if exit:
        scenario = replace(
            scenario,
            physical_shocks=(
                B1DatedShock("exit", "rival_exit", "B_OH0", 3, rival_carrier="B"),
            ),
        )
    e = OrdinaryInformationMarketEngine(
        scenario, dated_rebooking=DatedRebookingPolicy(72, 20, 100, interline=interline)
    )
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(79974, "unit/V9_received_information")
    )
    arrivals = tuple(
        ArrivalMark((i + 1) / (count + 1), 0, 0, 0, 0) for i in range(count)
    )
    g = tape.gumbel.copy()
    g[:] = -1e9
    g[:, :, -1] = 0
    if count:
        g[:, :, c.product_index["B_OH0"]] = 1e9
        if outside or count > 1:
            g[:, count - 1, -1] = 2e9
    g.setflags(write=False)
    tape = replace(
        tape, arrival_marks=tuple(arrivals for _ in tape.arrival_marks), gumbel=g
    )
    e.reset(tape=tape)
    for _ in range(steps):
        e.step(
            {
                carrier.code: PriceAction(
                    {j: 200 if i == 0 else 300 for j in c.products_of[i]}
                )
                for i, carrier in enumerate(c.spec.carriers)
            }
        )
    return e, c, clock, e.information_periods()


def fields():
    return tuple(
        module.DisclosureField(
            name, kind, "explicit received " + name, "declared unit " + name
        )
        for name, kind in [
            ("notice", "text"),
            ("permission", "boolean"),
            ("action_count", "integer"),
            ("belief_label", "rational"),
            ("quoted_amount", "binary64"),
        ]
    )


def journal(*, limit=200000, **kwargs):
    e, c, clock, periods = source(**kwargs)
    j = module.received_information_journal(
        c,
        periods,
        clock,
        fields(),
        definition="declared own received messages only",
        unobserved_definition="all private native marks/other people/future laws excluded",
        row_limit=limit,
    )
    return e, j


def delivery(
    j,
    *,
    delivery_id="m0",
    key=None,
    field="notice",
    value="received notice",
    minute=1,
    recorded=None,
    occurred=None,
):
    time = j.source_closed_at + timedelta(minutes=minute)
    return module.ReceivedDisclosure(
        delivery_id,
        j.person_keys[0] if key is None else key,
        field,
        value,
        time if occurred is None else occurred,
        time if recorded is None else recorded,
        time,
        "explicit NEW unit delivery source",
    )


def test_new_native_full_source_identity_outside_and_reader_no_producer(monkeypatch):
    e, j = journal()
    assert len(j.person_keys) == 6
    assert (
        sum(
            p.analyst_person.original_record is None
            for period in e.information_periods()
            for p in period.rows
        )
        == 3
    )
    saved = pickle.dumps(e.snapshot())
    rng = pickle.dumps(np.random.get_state())

    def fail(*args, **kwargs):
        raise AssertionError("pure reader called native source")

    monkeypatch.setattr(e, "step", fail)
    monkeypatch.setattr(e, "information_periods", fail)
    d = delivery(j)
    new = module.append_received_disclosures(j, [d], as_of=d.received_at)
    receipt = module.received_decision_receipt(
        new,
        j.person_keys[0],
        decision_id="d1",
        decision_at=d.received_at + timedelta(microseconds=1),
    )
    view = module.received_decision_view(receipt)
    assert view["received"][0]["value"] == "received notice"
    assert (
        pickle.dumps(e.snapshot()) == saved
        and pickle.dumps(np.random.get_state()) == rng
    )
    full = module.received_information_record(new)
    json.dumps(full, allow_nan=False)
    assert (
        "native_spec" in full["analyst_context"]
        and full["posterior_utility_CS_service_or_payoff"] is None
    )
    assert j.disclosures == () and len(new.disclosures) == 1


def test_functional_versions_strict_receipt_no_filtering_past():
    _, j = journal()
    d = delivery(j)
    v1 = module.append_received_disclosures(j, [d], as_of=d.received_at)
    t = d.received_at + timedelta(minutes=1)
    early = module.received_decision_receipt(
        v1, j.person_keys[0], decision_id="early", decision_at=t
    )
    late = delivery(
        j,
        delivery_id="later",
        minute=3,
        occurred=j.source_closed_at - timedelta(hours=2),
        recorded=j.source_closed_at + timedelta(minutes=2),
    )
    v2 = module.append_received_disclosures(v1, [late], as_of=late.received_at)
    with pytest.raises(ValueError, match="prior version"):
        module.received_decision_receipt(
            v2, j.person_keys[0], decision_id="past", decision_at=t
        )
    assert (
        len(module.received_decision_view(early)["received"]) == 1
        and len(v2.disclosures) == 2
    )
    with pytest.raises(ValueError, match="prior version"):
        module.received_decision_receipt(
            v1, j.person_keys[0], decision_id="equal", decision_at=d.received_at
        )
    with pytest.raises(ValueError, match="backfill"):
        module.append_received_disclosures(
            v1, [replace(d, delivery_id="backfill")], as_of=late.received_at
        )


def test_simultaneous_batch_all_own_actions_duplicates_content_retained():
    _, j = journal()
    a = delivery(j, delivery_id="a", field="action_count", value=1)
    b = delivery(j, delivery_id="b", field="action_count", value=1)
    v = module.append_received_disclosures(j, [a, b], as_of=a.received_at)
    rec = module.received_decision_receipt(
        v,
        j.person_keys[0],
        decision_id="d",
        decision_at=a.received_at + timedelta(seconds=1),
    )
    assert [
        r["person_message"] for r in module.received_decision_view(rec)["received"]
    ] == [0, 1]
    assert len(v.person_keys) == len(j.person_keys)
    with pytest.raises(ValueError, match="duplicate"):
        module.append_received_disclosures(j, [a, a], as_of=a.received_at)


def test_outside_recipient_and_other_person_privacy_and_local_key():
    _, j = journal()
    d1 = delivery(j, delivery_id="global0")
    d2 = delivery(j, delivery_id="global1", key=j.person_keys[1])
    d3 = delivery(
        j,
        delivery_id="secret-other",
        key=j.person_keys[2],
        value="other private notice",
    )
    v = module.append_received_disclosures(j, [d1, d2, d3], as_of=d1.received_at)
    time = d1.received_at + timedelta(seconds=1)
    a = module.received_decision_receipt(
        v, j.person_keys[0], decision_id="analyst0", decision_at=time
    )
    b = module.received_decision_receipt(
        v, j.person_keys[1], decision_id="analyst1", decision_at=time
    )
    assert module.received_decision_key(a) == module.received_decision_key(b)
    view = module.received_decision_view(a)
    text = json.dumps(view)
    for forbidden in (
        "person_key",
        "native_spec",
        "taste_class",
        "source_ref",
        "delivery_id",
        "global0",
        "other private notice",
        "analyst0",
    ):
        assert forbidden not in text
    view["received"][0]["value"] = "changed export"
    assert module.received_decision_view(a)["received"][0]["value"] == "received notice"


@pytest.mark.parametrize(
    "field,value",
    [
        ("action_count", True),
        ("permission", 1),
        ("belief_label", 0.5),
        ("quoted_amount", Fraction(1, 2)),
        ("notice", {}),
        ("missing", "not inferred"),
    ],
)
def test_undeclared_field_and_wrong_kind_no_coercion(field, value):
    _, j = journal()
    d = delivery(j, field=field, value=value)
    saved = pickle.dumps(j)
    with pytest.raises((TypeError, ValueError)):
        module.append_received_disclosures(j, [d], as_of=d.received_at)
    assert pickle.dumps(j) == saved


def test_exact_signed_tiny_unknown_and_binary64_payloads_not_native_invalid_fares():
    _, j = journal()
    nan = struct.unpack(">d", bytes.fromhex("7ff8000000000123"))[0]
    pairs = [
        ("belief_label", Fraction(1, 10**600)),
        ("belief_label", Fraction(-7, 3)),
        ("action_count", -(10**200)),
        ("permission", None),
        ("quoted_amount", 0.0),
        ("quoted_amount", -0.0),
        ("quoted_amount", float("inf")),
        ("quoted_amount", nan),
    ]
    ds = [
        delivery(j, delivery_id=str(i), field=f, value=v, minute=i + 1)
        for i, (f, v) in enumerate(pairs)
    ]
    v = module.append_received_disclosures(j, ds, as_of=ds[-1].received_at)
    receipt = module.received_decision_receipt(
        v,
        j.person_keys[0],
        decision_id="declared numerics",
        decision_at=v.as_of + timedelta(seconds=1),
    )
    rows = module.received_decision_view(receipt)["received"]
    json.dumps(rows, allow_nan=False)
    assert rows[0]["value"]["rational"]["numerator"] == "1" and rows[0]["value"][
        "rational"
    ]["denominator"] == str(10**600)
    assert rows[2]["value"]["integer"] == str(-(10**200)) and rows[3]["value"] is None
    assert rows[4]["value"]["binary64_bits"] != rows[5]["value"]["binary64_bits"]
    assert (
        rows[-1]["value"]["binary64_bits"] == "7ff8000000000123"
        and rows[-2]["value"]["exact"] is None
    )


def test_receipt_time_and_source_causality_not_occurrence_only():
    _, j = journal()
    d = delivery(
        j,
        minute=3,
        occurred=j.source_closed_at - timedelta(days=1),
        recorded=j.source_closed_at + timedelta(minutes=2),
    )
    with pytest.raises(ValueError, match="ordered receipt"):
        module.append_received_disclosures(
            j, [d], as_of=j.source_closed_at + timedelta(minutes=1)
        )
    with pytest.raises(ValueError, match="source recording"):
        replace(d, source_recorded_at=d.received_at + timedelta(seconds=1))
    with pytest.raises(ValueError, match="source recording"):
        replace(d, occurred_at=d.source_recorded_at + timedelta(seconds=1))
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(d, received_at=datetime(2026, 12, 2))
    with pytest.raises(ValueError, match="backfill"):
        module.append_received_disclosures(
            j,
            [replace(d, received_at=j.as_of, source_recorded_at=j.as_of)],
            as_of=j.as_of,
        )


def test_empty_and_partial_closed_prefix_no_future_people():
    e, c, clock, periods = source(count=0)
    j = module.received_information_journal(
        c,
        periods,
        clock,
        fields(),
        definition="empty",
        unobserved_definition="no people observed",
        row_limit=200000,
    )
    assert j.person_keys == () and len(j.analyst_context["periods"]) == 3
    with pytest.raises(ValueError, match="original person"):
        module.received_decision_receipt(
            j,
            (0, 0, 0),
            decision_id="no invented person",
            decision_at=j.as_of + timedelta(seconds=1),
        )
    _e, c, clock, periods = source(steps=1)
    j = module.received_information_journal(
        c,
        periods,
        clock,
        fields(),
        definition="prefix",
        unobserved_definition="later people unobserved",
        row_limit=200000,
    )
    assert len(j.person_keys) == 2 and j.source_closed_at == clock.at(1)
    d = delivery(j, key=(2, 0, 0))
    with pytest.raises(ValueError, match="unknown person"):
        module.append_received_disclosures(j, [d], as_of=d.received_at)


@pytest.mark.parametrize("interline", [False, True])
def test_cancelled_replaced_and_unreserved_source_remains_one_original_person(
    interline,
):
    e, j = journal(exit=True, interline=interline)
    assert e.rebooking_history
    assert len(j.person_keys) == 6
    d = delivery(j)
    v = module.append_received_disclosures(j, [d], as_of=d.received_at)
    assert len(v.person_keys) == 6 and len(v.disclosures) == 1
    assert {r["original_record"] for r in e.rebooking_history} == {0}
    assert (
        any(r["replacement_record"] is not None for r in e.rebooking_history)
        == interline
    )


def test_source_missing_period_log_menu_clock_and_ticket_partition_refused():
    _e, c, clock, periods = source()
    kwargs = {
        "definition": "new",
        "unobserved_definition": "explicit unknown",
        "row_limit": 200000,
    }
    with pytest.raises(ValueError):
        module.received_information_journal(c, periods[1:], clock, fields(), **kwargs)
    row = periods[0].rows[0]
    person = row.analyst_person
    altered = replace(row, analyst_person=replace(person, native_log_index=999))
    bad = (replace(periods[0], rows=(altered, *periods[0].rows[1:])), *periods[1:])
    with pytest.raises(ValueError, match="log"):
        module.received_information_journal(c, bad, clock, fields(), **kwargs)
    altered = replace(row, analyst_person=replace(person, original_record=None))
    bad = (replace(periods[0], rows=(altered, *periods[0].rows[1:])), *periods[1:])
    with pytest.raises(ValueError, match="partition"):
        module.received_information_journal(c, bad, clock, fields(), **kwargs)
    wrong = UtcBookingClock(tuple(t + timedelta(minutes=1) for t in clock.boundaries))
    with pytest.raises(ValueError, match="UTC clock"):
        module.received_information_journal(c, periods, wrong, fields(), **kwargs)
    altered = replace(
        row, menu_view=replace(row.menu_view, offers=row.menu_view.offers[:-1])
    )
    bad = (replace(periods[0], rows=(altered, *periods[0].rows[1:])), *periods[1:])
    with pytest.raises(ValueError, match="complete market"):
        module.received_information_journal(c, bad, clock, fields(), **kwargs)


def test_atomic_failure_order_limit_and_no_mutated_input(monkeypatch):
    _, j = journal()
    a = delivery(j)
    b = delivery(j, delivery_id="bad", field="undeclared", minute=2)
    saved = pickle.dumps(j)
    with pytest.raises(ValueError):
        module.append_received_disclosures(j, [a, b], as_of=b.received_at)
    assert pickle.dumps(j) == saved
    with pytest.raises(ValueError, match="ordered receipt"):
        module.append_received_disclosures(
            j,
            [replace(a, delivery_id="later", received_at=b.received_at), a],
            as_of=b.received_at,
        )
    with pytest.raises(module.ReceivedInformationLimit):
        module.append_received_disclosures(
            replace(j, row_limit=1), [a], as_of=a.received_at
        )
    base = module._units(
        {
            "source": j.analyst_context,
            "keys": j.person_keys,
            "schema": module.jsonable(j.fields),
            "definition": j.definition,
            "unobserved": j.unobserved_definition,
            "source_closed_at": j.source_closed_at.isoformat(),
            "as_of": j.as_of.isoformat(),
            "disclosures": (),
            "new_receipt": None,
            "rule": j.rule,
            "row_limit": j.row_limit,
        }
    )
    limited = replace(j, row_limit=base)
    module._require(limited)  # Full source fits; the complete NEW receipt does not.
    with pytest.raises(module.ReceivedInformationLimit):
        module.received_decision_receipt(
            limited,
            j.person_keys[0],
            decision_id="all source plus receipt",
            decision_at=j.as_of + timedelta(seconds=1),
        )

    def fail(*args):
        raise RuntimeError("NEW injected pure serialization failure")

    monkeypatch.setattr(module, "_message_record", fail)
    with pytest.raises(RuntimeError, match="serialization"):
        module.append_received_disclosures(j, [a], as_of=a.received_at)
    assert pickle.dumps(j) == saved


def test_owned_schema_sources_exports_rule_and_manual_consistency_refusals():
    _e, c, clock, periods = source()
    schema = list(fields())
    j = module.received_information_journal(
        c,
        list(periods),
        clock,
        schema,
        definition="owned",
        unobserved_definition="declared unknowns",
        row_limit=200000,
    )
    schema.clear()
    out = module.received_information_record(j)
    out["analyst_context"]["periods"][0]["rows"][0]["menu_view"]["market_id"] = (
        "corrupted export"
    )
    assert (
        module.received_information_record(j)["analyst_context"]["periods"][0]["rows"][
            0
        ]["menu_view"]["market_id"]
        != "corrupted export"
    )
    assert len(j.fields) == 5
    with pytest.raises(TypeError):
        module.received_information_record(replace(j, rule="old"))
    with pytest.raises(ValueError, match="support"):
        module.received_information_record(replace(j, person_keys=j.person_keys[:-1]))
    with pytest.raises(ValueError, match="unique disclosure"):
        module.received_information_journal(
            c,
            periods,
            clock,
            [*fields(), fields()[0]],
            definition="bad",
            unobserved_definition="unknown",
            row_limit=200000,
        )
    with pytest.raises(module.ReceivedInformationLimit):
        module.received_information_journal(
            c,
            periods,
            clock,
            fields(),
            definition="bad",
            unobserved_definition="unknown",
            row_limit=1,
        )


def test_new_receipt_budget_includes_analyst_link_but_narrow_view_excludes_it(
    monkeypatch,
):
    _, j = journal()
    seen = []
    native_units = module._units

    def capture(value):
        if (
            isinstance(value, dict)
            and "new_receipt" in value
            and value["new_receipt"] is not None
        ):
            seen.append(value["new_receipt"])
        return native_units(value)

    monkeypatch.setattr(module, "_units", capture)
    time = j.as_of + timedelta(seconds=1)
    receipt = module.received_decision_receipt(
        j, j.person_keys[0], decision_id="counted analyst linkage", decision_at=time
    )
    assert len(seen) == 1
    assert seen[0]["analyst_person_key"] == j.person_keys[0]
    assert seen[0]["analyst_decision_id"] == "counted analyst linkage"
    assert seen[0]["analyst_journal_asof"] == j.as_of.isoformat()
    assert "counted analyst linkage" not in json.dumps(
        module.received_decision_view(receipt)
    )
