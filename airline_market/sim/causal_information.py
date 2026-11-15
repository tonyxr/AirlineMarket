"""A.10/A.5/A.14/B0/V9: pure typed declared source-order information journal.

NEW causal order model; not a native hook producer or authenticated knowledge.
"""

import hashlib
from dataclasses import dataclass

from airline_market.core.hashing import canonical_json, jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.policy.dated_reserve import UtcBookingClock, utc
from airline_market.sim.received_information import (
    KINDS,
    DisclosureField,
    _key,
    _text,
    _units,
)
from airline_market.sim.reentry_stage_information import _stage_encode
from airline_market.spec.compiled import CompiledMarket

RULE = "declared-growing-causal-information-prefix-v1"


@dataclass(frozen=True, order=True)
class CausalPosition:
    anchor_utc: object
    ordinal: int

    def __post_init__(self):
        object.__setattr__(self, "anchor_utc", utc(self.anchor_utc))
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("nonnegative explicit source ordinal required")


@dataclass(frozen=True)
class RegisterPerson:
    known_type: str | None
    definition: str
    source_ref: str


@dataclass(frozen=True)
class MessageSource:
    delivery_id: str
    field: str
    value: object
    source_ref: str


@dataclass(frozen=True)
class MessageRecorded:
    delivery_id: str


@dataclass(frozen=True)
class MessageReceived:
    delivery_id: str


@dataclass(frozen=True)
class OwnAction:
    action_id: str
    kind: str
    detail: str | None
    source_ref: str


@dataclass(frozen=True)
class DecisionInput:
    decision_id: str
    stage: str
    attempt_id: str | None
    interline_enabled: bool | None
    source_ref: str


@dataclass(frozen=True)
class DecisionResult:
    decision_id: str
    outside: bool
    product_id: str | None
    source_ref: str


@dataclass(frozen=True)
class CausalEvent:
    event_id: str
    position: CausalPosition
    person_key: tuple
    payload: object

    def __post_init__(self):
        object.__setattr__(self, "person_key", _key(self.person_key))


@dataclass(frozen=True)
class CausalInformationJournal:
    analyst_context: object
    clock: tuple
    fields: tuple
    definition: str
    unobserved_definition: str
    row_limit: int
    events: tuple = ()
    rule: str = RULE

    def __post_init__(self):
        object.__setattr__(self, "analyst_context", deep_freeze(self.analyst_context))
        for name in ("clock", "fields", "events"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class CausalDecisionReceipt:
    source_journal: CausalInformationJournal
    analyst_context: object
    view: object
    rule: str = RULE

    def __post_init__(self):
        for name in ("analyst_context", "view"):
            object.__setattr__(self, name, deep_freeze(getattr(self, name)))


class CausalInformationLimit(RuntimeError):
    pass


def _state(journal):
    if type(journal) is not CausalInformationJournal or journal.rule != RULE:
        raise TypeError("NEW declared causal journal required")
    _text(journal.definition)
    _text(journal.unobserved_definition)
    if type(journal.row_limit) is not int or journal.row_limit < 1:
        raise ValueError("positive all-support structural limit required")
    clock = UtcBookingClock(journal.clock)
    spec = journal.analyst_context["native_spec"]
    if len(clock.boundaries) != spec["horizon"]["n_periods"] + 1:
        raise ValueError("complete T+1 native anchor clock required")
    if not journal.fields or any(
        type(f) is not DisclosureField for f in journal.fields
    ):
        raise TypeError("complete explicit typed field schema required")
    fields = {f.name: f for f in journal.fields}
    if len(fields) != len(journal.fields):
        raise ValueError("distinct schema field names required")
    ids, actions, people, sources, decisions, results, ordinary, attempts = (
        set(),
        set(),
        {},
        {},
        {},
        {},
        {},
        {},
    )
    previous, pending = None, None
    cancelled = set()
    for event in journal.events:
        if type(event) is not CausalEvent or type(event.position) is not CausalPosition:
            raise TypeError("typed complete causal event required")
        _text(event.event_id)
        if event.event_id in ids:
            raise ValueError("distinct global source event IDs required")
        ids.add(event.event_id)
        pos, key, p = event.position, event.person_key, event.payload
        period = clock.index(pos.anchor_utc)
        if key[0] >= len(clock.boundaries) - 1 or key[2] >= len(spec["markets"]):
            raise ValueError("original native key support differs from supplied model")
        expected = (
            0
            if previous is None or pos.anchor_utc > previous.anchor_utc
            else previous.ordinal + 1
        )
        if (
            previous is not None and pos.anchor_utc < previous.anchor_utc
        ) or pos.ordinal != expected:
            raise ValueError(
                "consecutive source slots at each nondecreasing anchor required"
            )
        previous = pos
        if pending is not None and type(p) not in (
            MessageSource,
            MessageRecorded,
            MessageReceived,
            DecisionResult,
        ):
            raise ValueError(
                "one declared input/result at a time; no native-event interleaving"
            )
        if type(p) is RegisterPerson:
            if key in people or period != key[0]:
                raise ValueError(
                    "one registration at original native opportunity required"
                )
            if p.known_type is not None:
                _text(p.known_type)
            _text(p.definition)
            _text(p.source_ref)
            people[key] = event
            continue
        if key not in people:
            raise ValueError(
                "actual source registration must precede supplied person events"
            )
        if type(p) is MessageSource:
            for value in (p.delivery_id, p.field, p.source_ref):
                _text(value)
            if p.delivery_id in sources or p.field not in fields:
                raise ValueError("distinct delivery and declared field required")
            if p.value is not None and type(p.value) is not KINDS[fields[p.field].kind]:
                raise TypeError(
                    "explicit exact scalar kind required; no coercion or zero fill"
                )
            sources[p.delivery_id] = [event, None, None]
        elif type(p) in (MessageRecorded, MessageReceived):
            _text(p.delivery_id)
            if (
                p.delivery_id not in sources
                or sources[p.delivery_id][0].person_key != key
            ):
                raise ValueError(
                    "same person's earlier explicit message source required"
                )
            row = sources[p.delivery_id]
            index = 1 if type(p) is MessageRecorded else 2
            if row[index] is not None or (index == 2 and row[1] is None):
                raise ValueError(
                    "one source then record then receipt; no missing-event fill"
                )
            row[index] = event
        elif type(p) is OwnAction:
            for value in (p.action_id, p.kind, p.source_ref):
                _text(value)
            if p.detail is not None:
                _text(p.detail)
            if p.action_id in actions or p.kind not in {
                "cancellation",
                "presentation",
                "other_own_action",
            }:
                raise ValueError("distinct explicit supported own action required")
            if p.kind in {"cancellation", "presentation"}:
                if (
                    key not in ordinary
                    or ordinary[key] not in results
                    or results[ordinary[key]].payload.outside
                ):
                    raise ValueError(
                        "declared original purchase result required for ticket action"
                    )
                if p.kind == "presentation" and key not in cancelled:
                    raise ValueError(
                        "earlier declared cancellation required for presentation"
                    )
            actions.add(p.action_id)
            if p.kind == "cancellation":
                cancelled.add(key)
        elif type(p) is DecisionInput:
            _text(p.decision_id)
            _text(p.source_ref)
            if p.decision_id in decisions or period == len(clock.boundaries) - 1:
                raise ValueError("distinct in-horizon decision required")
            if p.stage == "ordinary":
                if (
                    key in ordinary
                    or period != key[0]
                    or p.attempt_id is not None
                    or p.interline_enabled is not None
                ):
                    raise ValueError(
                        "one original ordinary decision, explicit null re-entry applicability"
                    )
                ordinary[key] = p.decision_id
            elif p.stage in {"own", "interline"}:
                _text(p.attempt_id)
                if type(p.interline_enabled) is not bool or key not in cancelled:
                    raise ValueError(
                        "explicit enabled contract and earlier declared cancellation required"
                    )
                attempt = (key, p.attempt_id)
                if p.stage == "own":
                    if attempt in attempts or any(
                        k[0] == key and v[0].position.anchor_utc == pos.anchor_utc
                        for k, v in attempts.items()
                    ):
                        raise ValueError(
                            "one own stage per person/opportunity; no repeated attempt"
                        )
                    attempts[attempt] = [event, None]
                else:
                    if attempt not in attempts:
                        raise ValueError(
                            "interline requires earlier SAME attempt own stage"
                        )
                    first, second = attempts[attempt]
                    if (
                        second is not None
                        or first.position.anchor_utc != pos.anchor_utc
                        or first.payload.interline_enabled is not True
                        or p.interline_enabled is not True
                    ):
                        raise ValueError(
                            "SAME enabled attempt and anchor required for interline"
                        )
                    if (
                        first.payload.decision_id not in results
                        or results[first.payload.decision_id].payload.outside
                        is not True
                    ):
                        raise ValueError(
                            "interline only after earlier actual DECLARED own outside"
                        )
                    attempts[attempt][1] = event
            else:
                raise ValueError("explicit ordinary/own/interline stage required")
            decisions[p.decision_id] = event
            pending = event
        elif type(p) is DecisionResult:
            _text(p.source_ref)
            if (
                pending is None
                or pending.person_key != key
                or pending.payload.decision_id != p.decision_id
            ):
                raise ValueError("result must follow SAME pending person/input")
            if type(p.outside) is not bool or (p.product_id is None) != p.outside:
                raise ValueError(
                    "explicit outside versus named selected product required"
                )
            if p.product_id is not None:
                _text(p.product_id)
                products = {product["id"]: product for product in spec["products"]}
                if (
                    p.product_id not in products
                    or products[p.product_id]["market"] != spec["markets"][key[2]]["id"]
                ):
                    raise ValueError(
                        "selected declared product must serve SAME original market"
                    )
                if pending.payload.stage != "ordinary":
                    original = products[results[ordinary[key]].payload.product_id][
                        "carrier"
                    ]
                    same = products[p.product_id]["carrier"] == original
                    if same != (pending.payload.stage == "own"):
                        raise ValueError(
                            "selected carrier must match declared own/interline stage"
                        )
            if pos.anchor_utc != pending.position.anchor_utc:
                raise ValueError("input/result must share compiled opportunity anchor")
            results[p.decision_id] = event
            pending = None
        else:
            raise TypeError("typed permitted event payload required")
    return {
        "people": people,
        "sources": sources,
        "decisions": decisions,
        "results": results,
        "pending": pending,
    }


def _raw(journal, state):
    return {
        "plan_id": "A.10/A.5/A.14/B0/V9",
        "rule": RULE,
        "analyst_context": _stage_encode(jsonable(journal)),
        "pending_declared_decision": None
        if state["pending"] is None
        else state["pending"].payload.decision_id,
        "native_hook_producer_or_actual_reception_type_law_authenticated": False,
        "candidate_same_anchor_order_adopted_into_native_physical_behavior_or_actor_observations": False,
        "lifecycle_utility_CS_service_or_payoff": None,
        "scientific_budget": None,
    }


def _require(journal, extra=None):
    state = _state(journal)
    if (
        _units({"full_journal": _raw(journal, state), "proposed_receipt": extra})
        > journal.row_limit
    ):
        raise CausalInformationLimit(
            "ALL source/schema/events/metadata/proposed receipt exceeds structural limit; no partial result"
        )
    return state


def causal_information_journal(
    compiled, clock, fields, *, definition, unobserved_definition, row_limit
):
    if type(compiled) is not CompiledMarket or type(clock) is not UtcBookingClock:
        raise TypeError("full native compiled model and explicit T+1 clock required")
    journal = CausalInformationJournal(
        _stage_encode(
            jsonable({"native_spec": compiled.spec, "spec_hash": compiled.spec_hash})
        ),
        clock.boundaries,
        tuple(fields),
        definition,
        unobserved_definition,
        row_limit,
    )
    _require(journal)
    return journal


def append_causal_events(journal, events):
    """Atomic NEW owned prefix; earlier versions remain intact on any refusal."""
    _require(journal)
    candidate = CausalInformationJournal(
        journal.analyst_context,
        journal.clock,
        journal.fields,
        journal.definition,
        journal.unobserved_definition,
        journal.row_limit,
        journal.events + tuple(events),
    )
    _require(candidate)
    return candidate


def causal_decision_receipt(journal, decision_id):
    """Only the version ending at THIS input; never filter a later global version."""
    state = _require(journal)
    if (
        not journal.events
        or type(journal.events[-1].payload) is not DecisionInput
        or journal.events[-1].payload.decision_id != decision_id
    ):
        raise ValueError(
            "use saved global version ending at this decision input; no past filtering"
        )
    event = journal.events[-1]
    key = event.person_key
    registration = state["people"][key].payload
    history = []
    for earlier in journal.events[:-1]:
        if earlier.person_key != key:
            continue
        p = earlier.payload
        row = None
        if type(p) is OwnAction:
            row = {"kind": p.kind, "detail": p.detail}
        elif type(p) is MessageReceived:
            original = state["sources"][p.delivery_id][0].payload
            row = {
                "kind": "declared_received",
                "field": original.field,
                "value": original.value,
            }
        elif type(p) is DecisionInput:
            row = {"kind": "earlier_input", "stage": p.stage}
        elif type(p) is DecisionResult:
            row = {
                "kind": "earlier_choice",
                "outside": p.outside,
                "product_id": p.product_id,
            }
        if row is not None:
            history.append({"person_event": len(history), **row})
    view = _stage_encode(
        {
            "scope": "declared-person-local-causal-prefix-not-authenticated-native-knowledge",
            "compiled_opportunity_utc": event.position.anchor_utc.isoformat(),
            "current_stage": event.payload.stage,
            "known_type": {
                "value": registration.known_type,
                "definition": registration.definition,
            },
            "definition": journal.definition,
            "unobserved_definition": journal.unobserved_definition,
            "schema": jsonable(journal.fields),
            "earlier_person_history": history,
        }
    )
    analyst = {
        "full_supplied_journal": _raw(journal, state),
        "decision_id": decision_id,
        "person_key": key,
        "source_position": _stage_encode(jsonable(event.position)),
        "last_prior_global_position": None
        if len(journal.events) == 1
        else _stage_encode(jsonable(journal.events[-2].position)),
    }
    receipt = CausalDecisionReceipt(journal, analyst, view)
    _require(journal, extra=_stage_encode(jsonable(receipt)))
    return receipt


def causal_information_record(journal):
    return _raw(journal, _require(journal))


def causal_decision_view(receipt):
    if type(receipt) is not CausalDecisionReceipt or receipt.rule != RULE:
        raise TypeError("NEW causal decision receipt required")
    expected = {
        "scope",
        "compiled_opportunity_utc",
        "current_stage",
        "known_type",
        "definition",
        "unobserved_definition",
        "schema",
        "earlier_person_history",
    }
    if set(receipt.view) != expected:
        raise ValueError(
            "exact narrow causal-view schema required; no analyst/private fields"
        )
    if type(receipt.source_journal) is not CausalInformationJournal:
        raise TypeError("full owned typed source prefix required")
    expected_receipt = causal_decision_receipt(
        receipt.source_journal, receipt.analyst_context["decision_id"]
    )
    if canonical_json(_stage_encode(jsonable(receipt))) != canonical_json(
        _stage_encode(jsonable(expected_receipt))
    ):
        raise ValueError(
            "complete view and analyst linkage must match SAME full saved typed prefix"
        )
    return _stage_encode(jsonable(receipt.view))


def causal_decision_key(receipt):
    return hashlib.sha256(
        canonical_json(causal_decision_view(receipt)).encode()
    ).hexdigest()
