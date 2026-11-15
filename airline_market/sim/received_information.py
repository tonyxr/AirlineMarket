"""A.10/A.5/B0/V9: declared received disclosures on a NEW closed native prefix.

Pure immutable journal/decision receipts. No source producer, actor input,
posterior, native receipt-time authentication or lifecycle utility adoption.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from airline_market.core.hashing import jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.policy.dated_reserve import UtcBookingClock, utc
from airline_market.sim.ordinary_cohort import OrdinaryPerson
from airline_market.sim.ordinary_information import (
    OrdinaryInformationOffer,
    OrdinaryInformationPeriod,
    OrdinaryInformationRow,
    OrdinaryMenuInformation,
)
from airline_market.sim.passenger_accounts import _binary, _encode, _number
from airline_market.spec.compiled import CompiledMarket

RULE = "declared-received-information-after-closed-prefix-v1"
KINDS = {
    "text": str,
    "boolean": bool,
    "integer": int,
    "rational": Fraction,
    "binary64": float,
}


def _text(value):
    if type(value) is not str or not value.strip():
        raise ValueError("explicit nonempty declaration required")
    return value


def _key(value):
    value = tuple(value)
    if len(value) != 3 or any(type(n) is not int or n < 0 for n in value):
        raise ValueError("original native period/potential-slot/market key required")
    return value


@dataclass(frozen=True)
class DisclosureField:
    name: str
    kind: str
    definition: str
    unit: str

    def __post_init__(self):
        for value in (self.name, self.definition, self.unit):
            _text(value)
        if self.kind not in KINDS:
            raise ValueError("explicit supported scalar disclosure kind required")


@dataclass(frozen=True)
class ReceivedDisclosure:
    delivery_id: str
    person_key: tuple
    field: str
    value: object
    occurred_at: object
    source_recorded_at: object
    received_at: object
    source_ref: str

    def __post_init__(self):
        for value in (self.delivery_id, self.field, self.source_ref):
            _text(value)
        object.__setattr__(self, "person_key", _key(self.person_key))
        for name in ("occurred_at", "source_recorded_at", "received_at"):
            object.__setattr__(self, name, utc(getattr(self, name)))
        if not self.occurred_at <= self.source_recorded_at <= self.received_at:
            raise ValueError("occurrence <= source recording <= receipt required")


@dataclass(frozen=True)
class ReceivedInformationJournal:
    analyst_context: object
    person_keys: tuple
    fields: tuple
    definition: str
    unobserved_definition: str
    source_closed_at: object
    as_of: object
    disclosures: tuple
    row_limit: int
    rule: str = RULE

    def __post_init__(self):
        object.__setattr__(self, "analyst_context", deep_freeze(self.analyst_context))
        object.__setattr__(
            self, "person_keys", tuple(_key(k) for k in self.person_keys)
        )
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "disclosures", tuple(self.disclosures))
        object.__setattr__(self, "source_closed_at", utc(self.source_closed_at))
        object.__setattr__(self, "as_of", utc(self.as_of))


@dataclass(frozen=True)
class ReceivedDecisionReceipt:
    analyst_person_key: tuple
    analyst_decision_id: str
    analyst_journal_asof: str
    view: object

    def __post_init__(self):
        object.__setattr__(self, "view", deep_freeze(self.view))


class ReceivedInformationLimit(RuntimeError):
    """Complete supplied structural support exceeds the infrastructure limit."""


def _units(value):
    if isinstance(value, Mapping):
        return 1 + sum(1 + _units(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return 1 + sum(_units(v) for v in value)
    return 1


def _value(value):
    if type(value) is Fraction:
        return {"rational": _number(value)}
    if type(value) is int:
        return {"integer": str(value)}
    if type(value) is float:
        return _binary(value)
    return value


def _message_record(d):
    return {
        "delivery_id": d.delivery_id,
        "person_key": d.person_key,
        "field": d.field,
        "value": _value(d.value),
        "occurred_at": d.occurred_at.isoformat(),
        "source_recorded_at": d.source_recorded_at.isoformat(),
        "received_at": d.received_at.isoformat(),
        "source_ref": d.source_ref,
    }


def _require(journal, extra=None):
    if type(journal) is not ReceivedInformationJournal or journal.rule != RULE:
        raise TypeError("declared received-information journal required")
    if type(journal.row_limit) is not int or journal.row_limit < 1:
        raise ValueError("positive complete-support infrastructure limit required")
    _text(journal.definition)
    _text(journal.unobserved_definition)
    if not journal.fields or any(
        type(f) is not DisclosureField for f in journal.fields
    ):
        raise TypeError("explicit typed scalar field schema required")
    names = [f.name for f in journal.fields]
    if len(set(names)) != len(names):
        raise ValueError("unique disclosure field names required")
    keys = tuple(
        tuple(r["analyst_person"]["key"])
        for p in journal.analyst_context["periods"]
        for r in p["rows"]
    )
    if keys != journal.person_keys or len(set(keys)) != len(keys):
        raise ValueError("full original-person source support differs")
    n = journal.analyst_context["completed_source_periods"]
    if (
        n != len(journal.analyst_context["periods"])
        or journal.source_closed_at.isoformat() != journal.analyst_context["clock"][n]
    ):
        raise ValueError("source closure/clock/support differs")
    if journal.as_of < journal.source_closed_at:
        raise ValueError("journal cannot precede its supplied source closure")
    ids = set()
    previous = journal.source_closed_at
    kinds = {f.name: KINDS[f.kind] for f in journal.fields}
    for d in journal.disclosures:
        if type(d) is not ReceivedDisclosure:
            raise TypeError("typed received disclosure required")
        if d.delivery_id in ids or d.person_key not in keys or d.field not in kinds:
            raise ValueError("duplicate delivery, unknown person or undeclared field")
        if d.value is not None and type(d.value) is not kinds[d.field]:
            raise TypeError("explicit scalar kind required; no coercion or zero fill")
        if (
            not journal.source_closed_at < d.received_at <= journal.as_of
            or d.received_at < previous
        ):
            raise ValueError("prospective ordered receipt support required")
        ids.add(d.delivery_id)
        previous = d.received_at
    support = {
        "source": journal.analyst_context,
        "keys": keys,
        "schema": jsonable(journal.fields),
        "definition": journal.definition,
        "unobserved": journal.unobserved_definition,
        "source_closed_at": journal.source_closed_at.isoformat(),
        "as_of": journal.as_of.isoformat(),
        "disclosures": tuple(_message_record(d) for d in journal.disclosures),
        "new_receipt": extra,
        "rule": journal.rule,
        "row_limit": journal.row_limit,
    }
    if _units(support) > journal.row_limit:
        raise ReceivedInformationLimit(
            "ALL source/schema/disclosure/receipt support exceeds limit; no partial result"
        )


def received_information_journal(
    compiled, periods, clock, fields, *, definition, unobserved_definition, row_limit
):
    """Own a complete 0..n-1 supplied NEW closed prefix; no producer call.

    Source closure is the next declared native opportunity, not actual message
    reception or within-period decision time. Future people are not invented.
    """
    if type(compiled) is not CompiledMarket or type(clock) is not UtcBookingClock:
        raise TypeError("compiled native model and explicit UTC clock required")
    periods = tuple(periods)
    T = compiled.spec.horizon.n_periods
    if len(clock.boundaries) != T + 1 or len(periods) > T:
        raise ValueError("full T+1 clock and closed prefix required")
    keys = []
    index = 0
    tickets = set()
    for t, p in enumerate(periods):
        if type(p) is not OrdinaryInformationPeriod or p.period != t:
            raise ValueError(
                "every closed source period including empty periods required"
            )
        for r in p.rows:
            if (
                type(r) is not OrdinaryInformationRow
                or type(r.analyst_person) is not OrdinaryPerson
            ):
                raise TypeError("complete typed ordinary information source required")
            person, menu = r.analyst_person, r.menu_view
            if type(menu) is not OrdinaryMenuInformation or any(
                type(o) is not OrdinaryInformationOffer
                or type(o.fare) is not float
                or type(o.offered) is not bool
                for o in menu.offers
            ):
                raise TypeError("complete typed market offer values required")
            key = _key(person.key)
            market = key[2]
            if market >= len(compiled.spec.markets) or key[0] != t:
                raise ValueError("source person coordinate differs from period/market")
            if (
                person.native_log_index != index
                or person.request_market != compiled.spec.markets[market].id
                or person.segment not in compiled.segment_index
            ):
                raise ValueError(
                    "complete ordinary log/market/segment support required"
                )
            if (
                person.native_row["period"],
                person.native_row["market"],
                person.native_row["segment"],
            ) != (t, market, compiled.segment_index[person.segment]):
                raise ValueError("source native row and person support differ")
            if (menu.native_period, menu.market_id, menu.compiled_opportunity_utc) != (
                t,
                person.request_market,
                clock.at(t).isoformat(),
            ):
                raise ValueError("source menu/person/UTC clock differ")
            chosen = person.native_row["chosen"]
            if (chosen is None) != (person.original_record is None):
                raise ValueError("source original ticket/outside partition differs")
            if chosen is not None:
                if (
                    chosen not in compiled.market_products[market]
                    or type(person.original_record) is not int
                    or person.original_record < 0
                    or person.original_record in tickets
                ):
                    raise ValueError("unique original physical ticket/market required")
                tickets.add(person.original_record)
            expected = tuple(
                compiled.spec.products[j].id for j in compiled.market_products[market]
            )
            if (
                tuple(o.product_id for o in menu.offers) != expected
                or menu.outside_available is not True
            ):
                raise ValueError("complete market offers and native outside required")
            keys.append(key)
            index += 1
    context = _encode(
        jsonable(
            {
                "plan_id": "A.10/A.5/B0/V9",
                "spec_hash": compiled.spec_hash,
                "native_spec": compiled.spec,
                "clock": tuple(t.isoformat() for t in clock.boundaries),
                "completed_source_periods": len(periods),
                "periods": periods,
            }
        )
    )
    closed = clock.at(len(periods))
    journal = ReceivedInformationJournal(
        context,
        tuple(keys),
        tuple(fields),
        definition,
        unobserved_definition,
        closed,
        closed,
        (),
        row_limit,
    )
    _require(journal)
    return journal


def append_received_disclosures(journal, disclosures, *, as_of):
    """Atomic NEW immutable version; never backfill an already closed as-of."""
    _require(journal)
    disclosures = tuple(disclosures)
    end = utc(as_of)
    if end < journal.as_of:
        raise ValueError("as-of cannot move backward")
    if any(type(d) is not ReceivedDisclosure for d in disclosures):
        raise TypeError("typed received disclosures required")
    if any(d.received_at <= journal.as_of for d in disclosures):
        raise ValueError(
            "new receipts must be strictly after previous as-of; no backfill"
        )
    new = ReceivedInformationJournal(
        journal.analyst_context,
        journal.person_keys,
        journal.fields,
        journal.definition,
        journal.unobserved_definition,
        journal.source_closed_at,
        end,
        journal.disclosures + disclosures,
        journal.row_limit,
    )
    _require(new)
    return new


def received_decision_receipt(journal, person_key, *, decision_id, decision_at):
    """ALL earlier supplied own receipts; no subset selector or future filtering."""
    _require(journal)
    key = _key(person_key)
    _text(decision_id)
    instant = utc(decision_at)
    if key not in journal.person_keys:
        raise ValueError("decision must name an original person in source support")
    if instant <= journal.as_of:
        raise ValueError(
            "decision strictly after journal as-of required; use prior version"
        )
    own = tuple(d for d in journal.disclosures if d.person_key == key)
    view = {
        "scope": "declared-received-history-only",
        "decision_at": instant.isoformat(),
        "definition": journal.definition,
        "unobserved_definition": journal.unobserved_definition,
        "schema": jsonable(journal.fields),
        "received": tuple(
            {
                "person_message": i,
                "field": d.field,
                "value": _value(d.value),
                "declared_received_at": d.received_at.isoformat(),
            }
            for i, d in enumerate(own)
        ),
    }
    _require(
        journal,
        extra={
            "analyst_person_key": key,
            "analyst_decision_id": decision_id,
            "analyst_journal_asof": journal.as_of.isoformat(),
            "view": view,
        },
    )
    return ReceivedDecisionReceipt(key, decision_id, journal.as_of.isoformat(), view)


def received_decision_view(receipt):
    if type(receipt) is not ReceivedDecisionReceipt:
        raise TypeError("received-decision receipt required")
    return _encode(jsonable(receipt.view))


def received_decision_key(receipt):
    payload = json.dumps(
        received_decision_view(receipt),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def received_information_record(journal):
    """Owned full ANALYST source/disclosure record; not passenger observation."""
    _require(journal)
    return {
        "plan_id": "A.10/A.5/B0/V9",
        "rule": RULE,
        "analyst_context": _encode(jsonable(journal.analyst_context)),
        "original_person_keys": journal.person_keys,
        "fields": jsonable(journal.fields),
        "definition": journal.definition,
        "unobserved_definition": journal.unobserved_definition,
        "source_closed_at": journal.source_closed_at.isoformat(),
        "as_of": journal.as_of.isoformat(),
        "disclosures": tuple(_message_record(d) for d in journal.disclosures),
        "infrastructure_row_limit": journal.row_limit,
        "actual_received_knowledge_public_legal_or_empirical_truth_authenticated": False,
        "native_actor_observations_choice_or_default_changed": False,
        "posterior_utility_CS_service_or_payoff": None,
        "scientific_budget": None,
    }
