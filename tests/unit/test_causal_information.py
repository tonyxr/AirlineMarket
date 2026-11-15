"""A.10/A.5/A.14/B0/V9: NEW pure causal journal units, no old example replay."""

import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

import pytest

from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.dated_reserve import UtcBookingClock
from airline_market.sim.received_information import DisclosureField
from airline_market.spec import compile_market

if os.environ.get("CAUSAL_SOURCE_DIR"):
    sp = importlib.util.spec_from_file_location(
        "airline_market.sim.causal_information",
        Path(os.environ["CAUSAL_SOURCE_DIR"]) / "causal_information.py",
    )
    loaded = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = loaded
    sp.loader.exec_module(loaded)

from airline_market.sim import causal_information as m

P, Q, R = (0, 0, 0), (0, 1, 0), (2, 0, 0)


def fresh(*, limit=500000, kind="text"):
    # NEW deterministic compiled unit declaration; no seed/world/tape/old fixture.
    c = compile_market(
        screen_spec(
            {
                "T": 4,
                "actions": 2,
                "n_spokes": 1,
                "capacity": 7,
                "capacity_support": [3, 4],
            }
        )
    )
    start = datetime(2027, 1, 11, tzinfo=timezone.utc)
    clock = UtcBookingClock(tuple(start + timedelta(hours=8 * t) for t in range(5)))
    j = m.causal_information_journal(
        c,
        clock,
        (
            DisclosureField(
                "signal", kind, "NEW explicit scalar signal", "declared unit"
            ),
        ),
        definition="NEW declared causal event stream",
        unobserved_definition="Native receipt/knowledge/type/law/utility unobserved",
        row_limit=limit,
    )
    return j


def emit(j, payload, *, key=P, period=0, name=None):
    anchor = j.clock[period]
    ordinal = (
        j.events[-1].position.ordinal + 1
        if j.events and j.events[-1].position.anchor_utc == anchor
        else 0
    )
    event = m.CausalEvent(
        name or "evt-" + str(len(j.events)),
        m.CausalPosition(anchor, ordinal),
        key,
        payload,
    )
    return m.append_causal_events(j, (event,))


def register(j, key=P, *, value="declared-L"):
    return emit(
        j,
        m.RegisterPerson(
            value, "Explicit persistent known label or unknown", "new-unit-type"
        ),
        key=key,
        period=key[0],
    )


def message(j, name, value, *, key=P, period=0):
    j = emit(
        j,
        m.MessageSource(name, "signal", value, "new-declared-source"),
        key=key,
        period=period,
    )
    j = emit(j, m.MessageRecorded(name), key=key, period=period)
    return emit(j, m.MessageReceived(name), key=key, period=period)


def ordinary(j, key=P, *, outside=False):
    name = "ordinary-" + str(key[1])
    j = emit(
        j, m.DecisionInput(name, "ordinary", None, None, "new-choice-input"), key=key
    )
    at_input = j
    j = emit(
        j,
        m.DecisionResult(
            name, outside, None if outside else "B_OH0", "new-choice-result"
        ),
        key=key,
    )
    return j, at_input


def own_prefix(*, interline=True, extra_other=False):
    j = register(fresh())
    j = message(j, "first", "initial declared signal")
    j, original = ordinary(j)
    j = register(j, Q, value=None)
    j, outside = ordinary(j, Q, outside=True)
    j = emit(
        j,
        m.OwnAction("cancel-1", "cancellation", "ticket metadata only", "new-cancel"),
        period=1,
    )
    j = emit(j, m.OwnAction("present-1", "presentation", None, "new-present"), period=1)
    if extra_other:
        j = message(j, "other", "other person's message", key=Q, period=1)
    j = emit(
        j,
        m.DecisionInput("own-1", "own", "attempt-local-1", interline, "new-own-input"),
        period=1,
    )
    return j, original, outside


def test_new_growing_population_saved_input_and_person_local_privacy():
    j, original, outside = own_prefix()
    assert len(original.events) < len(outside.events) < len(j.events)
    first = m.causal_decision_receipt(original, "ordinary-0")
    view = m.causal_decision_view(first)
    assert view["known_type"]["value"] == "declared-L"
    assert [r["kind"] for r in view["earlier_person_history"]] == ["declared_received"]
    assert "Q" not in json.dumps(view)
    assert not {
        "person_key",
        "ordinal",
        "event_id",
        "attempt_id",
        "source_ref",
        "native_spec",
        "results",
    } & set(view)
    qview = m.causal_decision_view(m.causal_decision_receipt(outside, "ordinary-1"))
    assert qview["known_type"]["value"] is None and not qview["earlier_person_history"]
    snapshot = pickle.dumps(j)
    later = register(journal_after_own_result(j), R)
    assert len(later.events) > len(j.events) and pickle.dumps(j) == snapshot
    with pytest.raises(ValueError, match="saved global version"):
        m.causal_decision_receipt(later, "ordinary-0")
    assert m.causal_decision_view(first) == view


def journal_after_own_result(j, *, outside=True):
    return emit(
        j,
        m.DecisionResult(
            "own-1", outside, None if outside else "B_OH0", "new-own-result"
        ),
        period=1,
    )


def test_new_interline_reach_same_person_earlier_result_no_current_outcome():
    j, _, _ = own_prefix()
    own_receipt = m.causal_decision_receipt(j, "own-1")
    j = journal_after_own_result(j)
    j = message(j, "own-notice", "explicit earlier own-outside message", period=1)
    j = emit(
        j,
        m.DecisionInput(
            "inter-1", "interline", "attempt-local-1", True, "new-inter-input"
        ),
        period=1,
    )
    receipt = m.causal_decision_receipt(j, "inter-1")
    view = m.causal_decision_view(receipt)
    assert view["current_stage"] == "interline"
    assert view["known_type"] == m.causal_decision_view(own_receipt)["known_type"]
    assert view["earlier_person_history"][-2]["kind"] == "earlier_choice"
    assert view["earlier_person_history"][-2]["outside"] is True
    assert (
        view["earlier_person_history"][-1]["value"]
        == "explicit earlier own-outside message"
    )
    j = emit(
        j, m.DecisionResult("inter-1", False, "A_OH0", "new-inter-result"), period=1
    )
    with pytest.raises(ValueError, match="saved global version"):
        m.causal_decision_receipt(j, "inter-1")
    assert m.causal_decision_view(receipt) == view
    assert m.causal_information_record(j)["pending_declared_decision"] is None


def test_new_other_person_interleaving_does_not_leak_global_slots():
    plain, _, _ = own_prefix()
    other, _, _ = own_prefix(extra_other=True)
    p = m.causal_decision_receipt(plain, "own-1")
    q = m.causal_decision_receipt(other, "own-1")
    assert p.analyst_context != q.analyst_context
    assert m.causal_decision_key(p) == m.causal_decision_key(q)
    future = message(
        plain, "late-other", "new other-person-only delivery", key=Q, period=1
    )
    with pytest.raises(ValueError, match="saved global version"):
        m.causal_decision_receipt(future, "own-1")
    assert m.causal_decision_key(p) == m.causal_decision_key(
        m.causal_decision_receipt(plain, "own-1")
    )


@pytest.mark.parametrize(
    "kind,value",
    [
        ("text", "new"),
        ("boolean", False),
        ("integer", 10**100),
        ("rational", Fraction(1, 10**450)),
        ("binary64", -0.0),
        ("binary64", float("nan")),
        ("text", None),
    ],
)
def test_new_explicit_payload_types_unknown_signed_nonfinite_owned_exports(kind, value):
    j = register(fresh(kind=kind))
    j = message(j, "typed", value)
    j = emit(j, m.DecisionInput("current", "ordinary", None, None, "new-input"))
    receipt = m.causal_decision_receipt(j, "current")
    view = m.causal_decision_view(receipt)
    saved = view["earlier_person_history"][0]["value"]
    assert saved == m._stage_encode(value)
    json.dumps(m.causal_information_record(j), allow_nan=False)
    view["earlier_person_history"][0]["value"] = "external mutation"
    assert (
        m.causal_decision_view(receipt)["earlier_person_history"][0]["value"] == saved
    )


@pytest.mark.parametrize(
    "change",
    [
        "gap",
        "backdate",
        "duplicate_event",
        "unregistered",
        "duplicate_person",
        "future_registration",
        "unknown_field",
        "wrong_scalar",
        "record_without_source",
        "receive_without_record",
        "other_person_receipt",
        "duplicate_receipt",
        "duplicate_decision",
        "result_wrong_person",
        "ordinary_outside_reentry",
        "interline_without_own",
        "own_purchase_then_interline",
        "disabled_interline",
        "wrong_market",
        "wrong_carrier",
        "terminal_decision",
    ],
)
def test_new_full_event_order_support_and_stage_refusals_are_atomic(change):
    j, _, _ = own_prefix()
    before = pickle.dumps(j)
    with pytest.raises((ValueError, TypeError)):
        if change == "gap":
            m.append_causal_events(
                j,
                (
                    m.CausalEvent(
                        "gap",
                        m.CausalPosition(j.clock[1], j.events[-1].position.ordinal + 2),
                        P,
                        m.MessageRecorded("first"),
                    ),
                ),
            )
        elif change == "backdate":
            emit(j, m.MessageSource("back", "signal", "x", "ref"), period=0)
        elif change == "duplicate_event":
            emit(
                j,
                m.MessageSource("dup", "signal", "x", "ref"),
                period=1,
                name=j.events[0].event_id,
            )
        elif change == "unregistered":
            emit(j, m.MessageSource("unknown", "signal", "x", "ref"), key=R, period=1)
        elif change == "duplicate_person":
            register(register(fresh()))
        elif change == "future_registration":
            emit(
                journal_after_own_result(j),
                m.RegisterPerson(None, "declared unknown", "ref"),
                key=R,
                period=1,
            )
        elif change == "unknown_field":
            emit(j, m.MessageSource("field", "private_stock", "x", "ref"), period=1)
        elif change == "wrong_scalar":
            emit(j, m.MessageSource("scalar", "signal", 1, "ref"), period=1)
        elif change == "record_without_source":
            emit(j, m.MessageRecorded("missing"), period=1)
        elif change == "receive_without_record":
            temp = emit(
                j, m.MessageSource("not-recorded", "signal", "x", "ref"), period=1
            )
            emit(temp, m.MessageReceived("not-recorded"), period=1)
        elif change == "other_person_receipt":
            emit(j, m.MessageReceived("first"), key=Q, period=1)
        elif change == "duplicate_receipt":
            emit(j, m.MessageReceived("first"), period=1)
        elif change == "duplicate_decision":
            emit(
                journal_after_own_result(j),
                m.DecisionInput("own-1", "own", "new-attempt", True, "ref"),
                period=2,
            )
        elif change == "result_wrong_person":
            emit(j, m.DecisionResult("own-1", True, None, "ref"), key=Q, period=1)
        elif change == "ordinary_outside_reentry":
            emit(
                journal_after_own_result(j),
                m.OwnAction("Q-cancel", "cancellation", None, "ref"),
                key=Q,
                period=1,
            )
        elif change == "interline_without_own":
            emit(
                journal_after_own_result(j),
                m.DecisionInput(
                    "new-inter", "interline", "different-attempt", True, "ref"
                ),
                period=1,
            )
        elif change == "own_purchase_then_interline":
            emit(
                journal_after_own_result(j, outside=False),
                m.DecisionInput(
                    "new-inter", "interline", "attempt-local-1", True, "ref"
                ),
                period=1,
            )
        elif change == "disabled_interline":
            disabled, _, _ = own_prefix(interline=False)
            emit(
                journal_after_own_result(disabled),
                m.DecisionInput(
                    "new-inter", "interline", "attempt-local-1", True, "ref"
                ),
                period=1,
            )
        elif change == "wrong_market":
            emit(j, m.DecisionResult("own-1", False, "B_HD0", "ref"), period=1)
        elif change == "wrong_carrier":
            emit(j, m.DecisionResult("own-1", False, "A_OH0", "ref"), period=1)
        else:
            emit(
                journal_after_own_result(j),
                m.DecisionInput("terminal", "own", "later-attempt", True, "ref"),
                period=4,
            )
    assert pickle.dumps(j) == before


def test_new_late_recording_unknown_source_is_not_early_received_knowledge():
    j = register(fresh())
    j = emit(j, m.MessageSource("not-yet-recorded", "signal", None, "late-ref"))
    j, earlier = ordinary(j)
    assert not m.causal_decision_view(m.causal_decision_receipt(earlier, "ordinary-0"))[
        "earlier_person_history"
    ]
    j = emit(j, m.MessageRecorded("not-yet-recorded"), period=1)
    j = emit(j, m.MessageReceived("not-yet-recorded"), period=1)
    with pytest.raises(ValueError, match="saved global version"):
        m.causal_decision_receipt(j, "ordinary-0")
    assert not m.causal_decision_view(m.causal_decision_receipt(earlier, "ordinary-0"))[
        "earlier_person_history"
    ]


def test_new_owned_failure_limits_complete_receipt_metadata_and_pure_operations():
    j, _, _ = own_prefix()
    receipt = m.causal_decision_receipt(j, "own-1")
    with pytest.raises(TypeError):
        receipt.view["injected"] = "private"
    with pytest.raises(ValueError, match="narrow"):
        m.causal_decision_view(
            replace(
                receipt, view={**m.causal_decision_view(receipt), "private_class": 1}
            )
        )
    raw = m.causal_information_record(j)
    limit = m._units({"full_journal": raw, "proposed_receipt": None})
    smaller = replace(j, row_limit=limit + 30)
    m.causal_information_record(smaller)
    with pytest.raises(m.CausalInformationLimit):
        m.causal_decision_receipt(smaller, "own-1")
    with pytest.raises(m.CausalInformationLimit):
        fresh(limit=1)
    assert (
        m.causal_information_record(j)["lifecycle_utility_CS_service_or_payoff"] is None
    )
    assert (
        m.causal_information_record(j)[
            "native_hook_producer_or_actual_reception_type_law_authenticated"
        ]
        is False
    )


@pytest.mark.parametrize(
    "change", ["omitted_history", "nested_private", "analyst_link"]
)
def test_new_receipt_full_binding_to_immutable_typed_prefix(change):
    j, _, _ = own_prefix()
    receipt = m.causal_decision_receipt(j, "own-1")
    before = pickle.dumps(j)
    view = m.causal_decision_view(receipt)
    if change == "omitted_history":
        view["earlier_person_history"] = view["earlier_person_history"][1:]
        forged = replace(receipt, view=view)
    elif change == "nested_private":
        view["earlier_person_history"][0]["private_class"] = 1
        forged = replace(receipt, view=view)
    else:
        analyst = m._stage_encode(receipt.analyst_context)
        analyst["person_key"] = Q
        forged = replace(receipt, analyst_context=analyst)
    with pytest.raises(ValueError, match="full saved typed prefix"):
        m.causal_decision_view(forged)
    assert pickle.dumps(j) == before
