"""A.10/A.5/B0/V9/B4: complete declared ticket settlements on NEW branch service.

Pure supplied-source reader; no native/service execution or financial truth,
pre-source registration, passenger utility, CS or preference authentication.
"""

import json
import math
from dataclasses import dataclass, replace
from datetime import datetime
from fractions import Fraction

from airline_market.policy.dated_reserve import utc
from airline_market.sim import bernoulli_service as service_reader
from airline_market.sim.received_information import _text, _units
from airline_market.sim.shared_mark_attempt import _encoded, _owned

RULE = "complete-actual-branch-ticket-financial-laws-v1"


def _check(value, message):
    if not value:
        raise ValueError(message)


def _same(a, b):
    return json.dumps(_encoded(a), sort_keys=True) == json.dumps(
        _encoded(b), sort_keys=True
    )


class BranchFinancialLimit(RuntimeError):
    """Complete structural support exceeds limit; no partial output or retry."""


@dataclass(frozen=True)
class BranchTicketTransfer:
    event_id: str
    person_key: tuple
    ticket: int
    carrier: str
    kind: str
    amount: float | None
    occurred: object
    recorded: object
    source_ref: str
    definition: str

    def __post_init__(self):
        key = _owned(self.person_key)
        _check(
            type(key) is tuple
            and len(key) == 3
            and all(type(x) is int and x >= 0 for x in key),
            "full nonnegative integer original-person key required",
        )
        object.__setattr__(self, "person_key", key)
        object.__setattr__(self, "occurred", utc(self.occurred))
        object.__setattr__(self, "recorded", utc(self.recorded))


@dataclass(frozen=True)
class BranchFinancialFollowup:
    person_key: tuple
    through: object
    closed: bool
    source_ref: str

    def __post_init__(self):
        key = _owned(self.person_key)
        _check(
            type(key) is tuple
            and len(key) == 3
            and all(type(x) is int and x >= 0 for x in key),
            "full nonnegative integer original-person key required",
        )
        object.__setattr__(self, "person_key", key)
        object.__setattr__(self, "through", utc(self.through))


@dataclass(frozen=True)
class BranchSettlement:
    name: str
    transfers: tuple
    followup: tuple
    definition: str

    def __post_init__(self):
        _check(
            type(self.transfers) in (tuple, list)
            and type(self.followup) in (tuple, list),
            "explicit tuple/list transfer and follow-up support required; no iterator consumption",
        )
        _check(
            all(type(e) is BranchTicketTransfer for e in self.transfers),
            "typed additional branch-ticket transfers required",
        )
        _check(
            all(type(f) is BranchFinancialFollowup for f in self.followup),
            "typed ALL-person financial follow-up required",
        )
        object.__setattr__(self, "transfers", tuple(replace(e) for e in self.transfers))
        object.__setattr__(self, "followup", tuple(replace(f) for f in self.followup))


@dataclass(frozen=True)
class BranchFinancialCell:
    service_atom: int
    programmes: tuple

    def __post_init__(self):
        _check(
            type(self.programmes) in (tuple, list),
            "explicit tuple/list complete settlement support required",
        )
        _check(
            all(type(p) is BranchSettlement for p in self.programmes),
            "typed complete settlement programmes required",
        )
        object.__setattr__(
            self, "programmes", tuple(replace(p) for p in self.programmes)
        )


@dataclass(frozen=True)
class BranchFinancialLaw:
    name: str
    probabilities: tuple
    definition: str

    def __post_init__(self):
        object.__setattr__(self, "probabilities", _owned(self.probabilities))


@dataclass(frozen=True)
class BranchFinancialEvent:
    name: str
    include: tuple
    definition: str

    def __post_init__(self):
        object.__setattr__(self, "include", _owned(self.include))


@dataclass(frozen=True)
class BranchFinancialResult:
    analyst: object

    def __post_init__(self):
        object.__setattr__(self, "analyst", _owned(self.analyst))


def _witness(report, plan):
    """Check complete supplied service witnesses, never execute service transitions."""
    _check(
        set(report) == set(service_reader._report_shape(plan)),
        "complete supplied service report required",
    )
    _check(_same(report["input_plan"], plan), "same full service input plan required")
    _check(
        report["new_branch_service_state_machine_executed"] is True
        and report["genuine_panel_journal_or_native_departure_engine_executed"]
        is False,
        "same branch-service route declarations required",
    )
    people = plan["people"]
    keys = tuple(p["person_key"] for p in people)
    targets = {
        p["current_ticket"]: p for p in people if p["current_ticket"] is not None
    }
    operations = {o["leg"]: o for o in plan["operations"]}
    orders = dict(plan["orders"])
    intents = {(rid, leg): b for rid, leg, b in plan["intents"]}
    steps = report["steps"]
    _check(
        len(steps) == len(plan["events"]),
        "ALL ordered service events including unused legs required",
    )
    airborne, delivered, outcomes = {}, set(), {}
    for step, (at, _, leg, kind) in zip(steps, plan["events"]):
        o = operations[leg]
        _check(
            set(step) == set(service_reader._report_shape(plan)["steps"][0]),
            "full service witness fields required",
        )
        _check(
            (step["at"], step["leg"], step["kind"]) == (at, leg, kind)
            and _same(step["all_person_keys"], keys)
            and _same(step["all_target_tickets"], orders[leg]),
            "complete service event/person/target support differs",
        )
        _check(
            type(step["physical_seats"]) is int
            and step["physical_seats"] == o["seats"],
            "same integer physical seat identity required",
        )
        decisions = []
        if kind == "departure":
            eligible = tuple(rid for rid in orders[leg] if intents[rid, leg])
            manifest = eligible[: o["seats"]]
            for rid in orders[leg]:
                reason = (
                    "no_boarding_intent"
                    if not intents[rid, leg]
                    else "boarded_not_delivered"
                    if rid in manifest
                    else "no_seat"
                )
                decisions.append((rid, reason))
                if rid not in manifest:
                    outcomes[rid] = False
            airborne[leg] = manifest
            _check(
                type(step["boarded_count"]) is int
                and type(step["unused_departure_seats"]) is int
                and step["boarded_count"] == len(manifest)
                and step["unused_departure_seats"] + len(manifest) == o["seats"],
                "complete declared-priority service seat debit witness differs",
            )
        else:
            _check(
                step["boarded_count"] is None
                and step["unused_departure_seats"] is None,
                "non-departure seat debit must remain inapplicable",
            )
            if kind == "arrival":
                _check(
                    leg in airborne, "arrival needs supplied earlier boarding witness"
                )
                for rid in airborne.pop(leg):
                    delivered.add(rid)
                    outcomes[rid] = True
                    decisions.append((rid, "delivered_at_arrival"))
            else:
                for rid in orders[leg]:
                    outcomes[rid] = False
                    decisions.append((rid, "operation_cancelled"))
        _check(
            _same(step["decisions"], decisions)
            and _same(step["airborne"], tuple(sorted(airborne.items())))
            and _same(step["delivered"], tuple(sorted(delivered)))
            and _same(step["outcomes"], tuple(sorted(outcomes.items()))),
            "complete service witness conservation/outcome chronology differs",
        )
    _check(
        not airborne and set(outcomes) == set(targets),
        "complete current-ticket service partition required",
    )
    expected = tuple(
        (
            p["person_key"],
            None if p["current_ticket"] is None else outcomes[p["current_ticket"]],
        )
        for p in people
    )
    _check(
        _same(report["current_service"], expected)
        and type(report["delivered_current_target_count"]) is int
        and report["delivered_current_target_count"] == len(delivered),
        "outside/unreserved NULL or current service result differs",
    )


def _service(c, source, clock, cases, result):
    record = service_reader.bernoulli_service_record(result)
    _check(
        record["rule"] == service_reader.RULE,
        "same complete frozen branch-service rule required",
    )
    template, actual = service_reader._preflight(
        c,
        source,
        clock,
        record["programme_names"],
        cases,
        record["support_limit"],
        record["definition"],
    )
    dynamic = {
        "all_atoms",
        "public_joint",
        "raw_weighted_delivered_current_count_diagnostic",
        "delivered_current_count_integral",
    }
    _check(
        set(template) == set(record), "complete branch-service export support required"
    )
    for field in set(template) - dynamic:
        _check(
            _same(template[field], record[field]),
            "full source/service input or rule metadata differs: " + field,
        )
    _check(
        len(record["all_atoms"]) == len(actual),
        "ALL source/service atoms including zero support required",
    )
    joint = []
    integral = Fraction()
    for atom, (common, plan) in zip(record["all_atoms"], actual):
        _check(
            set(atom) == set(common) | {"service"}
            and all(_same(atom[k], value) for k, value in common.items()),
            "same complete source/service atom factors required",
        )
        _witness(atom["service"], plan)
        joint.append({**common, "current_service": atom["service"]["current_service"]})
        integral += (
            common["raw_joint_mass"] * atom["service"]["delivered_current_target_count"]
        )
    _check(
        _same(
            record["public_joint"],
            joint if template["source_public_joint_known"] else None,
        )
        and _same(record["raw_weighted_delivered_current_count_diagnostic"], integral)
        and _same(
            record["delivered_current_count_integral"],
            integral if template["source_public_joint_known"] else None,
        ),
        "public/raw supplied service joint differs",
    )
    return template, actual


def _prepare(programme, people, close, asof):
    _text(programme.name)
    _text(programme.definition)
    keys = tuple(p["person_key"] for p in people)
    index = {p["person_key"]: p for p in people}
    _check(
        tuple(f.person_key for f in programme.followup) == keys,
        "ALL original people including outside require ordered follow-up",
    )
    for f in programme.followup:
        _check(
            f.through == asof and type(f.closed) is bool,
            "common as-of and explicit Boolean financial closure required",
        )
        _text(f.source_ref)
    previous, ids = close, set()
    for e in programme.transfers:
        for text in (e.event_id, e.source_ref, e.definition, e.carrier):
            _text(text)
        _check(
            e.event_id not in ids and e.person_key in index,
            "distinct event ID and same original person required",
        )
        ids.add(e.event_id)
        p = index[e.person_key]
        _check(
            type(e.ticket) is int
            and p["original_ticket"] is not None
            and e.ticket == p["original_ticket"]
            and e.carrier == p["carrier"],
            "SAME-person existing original ticket/carrier required; no outside ticket fill",
        )
        _check(
            e.kind in ("additional_payment", "additional_refund")
            and (e.amount is None or type(e.amount) is float),
            "explicit ADDITIONAL kind/binary64-or-unknown amount required",
        )
        _check(
            close <= e.occurred <= e.recorded <= asof and e.recorded >= previous,
            "causal post-close global transfer recording through common as-of required",
        )
        previous = e.recorded


def _payments(programme, people):
    rows = []
    for p, followup in zip(people, programme.followup):
        terms = []
        for e in programme.transfers:
            if e.person_key == p["person_key"]:
                value = (
                    Fraction.from_float(e.amount)
                    if e.amount is not None and math.isfinite(e.amount)
                    else None
                )
                terms.append(
                    None
                    if value is None
                    else value
                    if e.kind == "additional_payment"
                    else -value
                )
        recorded = (
            None
            if any(t is None for t in terms)
            else p["native_net_ticket_payment"] + sum(terms, Fraction())
        )
        rows.append(
            {
                "person_key": p["person_key"],
                "original_ticket": p["original_ticket"],
                "current_ticket": p["current_ticket"],
                "native_fare": p["original_fare"],
                "native_refund_once": p["native_source_refund"],
                "native_net": p["native_net_ticket_payment"],
                "additional_signed_net_terms": tuple(terms),
                "recorded_net": recorded,
                "financial_closed": followup.closed,
                "final_net": recorded if followup.closed else None,
            }
        )

    def total(field):
        values = tuple(r[field] for r in rows)
        return None if any(x is None for x in values) else sum(values, Fraction())

    return {
        "all_original_people": tuple(rows),
        "population_recorded_net": total("recorded_net"),
        "population_final_net": total("final_net"),
    }


def branch_financial_laws(
    compiled,
    source,
    clock,
    service_cases,
    service_result,
    programme_names,
    cells,
    laws,
    events,
    *,
    financial_asof,
    unit,
    definition,
    support_limit,
):
    """Read complete NEW supplied branch service; no producer or settlement generation.

    This actual-atom reader does not certify pre-source contingent registration.
    """
    _check(
        type(support_limit) is int and support_limit > 0,
        "positive structural support limit required",
    )
    for text in (unit, definition):
        _text(text)
    asof = utc(financial_asof)
    names = _owned(programme_names)
    _check(
        names and len(set(names)) == len(names),
        "complete distinct named settlement support required",
    )
    for name in names:
        _text(name)
    _check(
        all(type(x) in (tuple, list) for x in (service_cases, cells, laws, events)),
        "explicit tuple/list service, cell, law and event support required",
    )
    _check(
        all(type(c) is BranchFinancialCell for c in cells)
        and all(type(l) is BranchFinancialLaw for l in laws)
        and all(type(e) is BranchFinancialEvent for e in events),
        "typed complete financial cells/laws/events required",
    )
    cells, laws, events = (
        tuple(replace(c) for c in cells),
        tuple(replace(l) for l in laws),
        tuple(replace(e) for e in events),
    )
    # Own the full supplied context before pure delegated source traversal.
    base = {
        "plan_id": "A.10/A.5/B0/V9/B4",
        "rule": RULE,
        "complete_native_model": _encoded(compiled.spec),
        "complete_native_source": _encoded(source),
        "complete_service_clock": _encoded(clock),
        "complete_service_cases": _encoded(service_cases),
        "complete_service_result": _encoded(service_result),
        "programme_names": names,
        "cells": _encoded(cells),
        "laws": _encoded(laws),
        "events": _encoded(events),
        "financial_asof": asof.isoformat(),
        "unit": unit,
        "definition": definition,
        "support_limit": support_limit,
    }
    input_units = _units(base)
    if input_units > support_limit:
        raise BranchFinancialLimit(
            "full input support exceeds limit before pure source traversal/payment arithmetic"
        )
    template, actual = _service(compiled, source, clock, service_cases, service_result)
    n = len(actual)
    _check(
        len(cells) == n
        and all(
            type(c.service_atom) is int and c.service_atom == i
            for i, c in enumerate(cells)
        ),
        "ALL ordered actual service cells including zero mass required",
    )
    _check(
        laws
        and events
        and len({l.name for l in laws}) == len(laws)
        and len({e.name for e in events}) == len(events),
        "complete distinct named laws and analyst events required",
    )
    for law in laws:
        _text(law.name)
        _text(law.definition)
        _check(
            len(law.probabilities) == n,
            "ALL law/service-cell conditional probabilities required",
        )
        for row in law.probabilities:
            _check(
                len(row) == len(names)
                and all(type(p) is Fraction and p >= 0 for p in row)
                and sum(row, Fraction()) == 1,
                "full exact nonnegative Fraction conditional law sums one, no normalization",
            )
    for event in events:
        _text(event.name)
        _text(event.definition)
        _check(
            len(event.include) == n and all(type(x) is bool for x in event.include),
            "full Boolean analyst event mask required",
        )
    _check(
        asof >= clock.source_close
        and all(
            asof >= datetime.fromisoformat(at)
            for _, plan in actual
            for at, _, _, _ in plan["events"]
        ),
        "financial as-of must cover complete supplied service programme",
    )
    profiles = template["all_original_person_profiles"]
    shapes = []
    for cell, (common, plan) in zip(cells, actual):
        people = profiles[common["history_index"]]
        _check(
            tuple(p.name for p in cell.programmes) == names,
            "ALL same ordered settlement names required in every service cell",
        )
        for programme in cell.programmes:
            _prepare(programme, people, clock.source_close, asof)
            shaped_people = tuple(
                {
                    "person_key": p["person_key"],
                    "original_ticket": p["original_ticket"],
                    "current_ticket": p["current_ticket"],
                    "native_fare": p["original_fare"],
                    "native_refund_once": p["native_source_refund"],
                    "native_net": p["native_net_ticket_payment"],
                    "additional_signed_net_terms": tuple(
                        Fraction()
                        for e in programme.transfers
                        if e.person_key == p["person_key"]
                    ),
                    "recorded_net": Fraction(),
                    "financial_closed": True,
                    "final_net": Fraction(),
                }
                for p in people
            )
            shapes.append(
                {
                    "service_atom": cell.service_atom,
                    "programme": programme.name,
                    "all_original_people": shaped_people,
                    "population_recorded_net": Fraction(),
                    "population_final_net": Fraction(),
                }
            )
    law_shapes = tuple(
        {
            "law": law.name,
            "service_atom": i,
            "programme": name,
            "report_index": i * len(names) + j,
            "source_service_mass": Fraction(),
            "financial_probability": Fraction(),
            "raw_joint_mass": Fraction(),
        }
        for law in laws
        for i in range(n)
        for j, name in enumerate(names)
    )
    summary_shapes = tuple(
        {
            "law": law.name,
            "event": event.name,
            "raw_event_evidence": Fraction(),
            "positive_unknown_mass": Fraction(),
            "known_payment_numerator_diagnostic": Fraction(),
            "net_payment_integral": Fraction(),
            "raw_measure_conditional_payment_ratio": Fraction(),
        }
        for law in laws
        for event in events
    )
    output = {
        **base,
        "payment_reports": tuple(shapes),
        "raw_law_rows": law_shapes,
        "public_law_rows": law_shapes
        if template["source_public_joint_known"]
        else None,
        "summaries": summary_shapes,
        "source_public_joint_known": template["source_public_joint_known"],
        "raw_source_measure": template["raw_source_measure"],
        "raw_source_residual": template["raw_source_residual"],
        "all_support_bound": 0,
        "native_or_service_producer_RNG_actor_book_resolve_called": False,
        "supplied_service_witness_checked_not_executed": True,
        "pre_source_registration_or_financial_service_truth_authenticated": False,
        "material_utility_CS_denial_misallocation_penalty_incidence": None,
        "full_primary_default_empirical_production_planner_adopted": False,
        "scientific_budget": None,
    }
    required = (
        input_units
        + _units(_encoded(template))
        + sum(_units(_encoded(plan)) for _, plan in actual)
        + _units(_encoded(output))
    )
    if required > support_limit:
        raise BranchFinancialLimit(
            "ALL proposed output/context support exceeds limit before payment arithmetic"
        )
    reports = tuple(
        {
            "service_atom": cell.service_atom,
            "programme": programme.name,
            **_payments(
                programme, profiles[actual[cell.service_atom][0]["history_index"]]
            ),
        }
        for cell in cells
        for programme in cell.programmes
    )
    law_rows = []
    summaries = []
    for law in laws:
        for i, (common, _) in enumerate(actual):
            for j, name in enumerate(names):
                p = law.probabilities[i][j]
                law_rows.append(
                    {
                        "law": law.name,
                        "service_atom": i,
                        "programme": name,
                        "report_index": i * len(names) + j,
                        "source_service_mass": common["raw_joint_mass"],
                        "financial_probability": p,
                        "raw_joint_mass": common["raw_joint_mass"] * p,
                    }
                )
        for event in events:
            evidence = unknown = numerator = Fraction()
            for i, (common, _) in enumerate(actual):
                for j in range(len(names)):
                    mass = common["raw_joint_mass"] * law.probabilities[i][j]
                    value = reports[i * len(names) + j]["population_final_net"]
                    if event.include[i]:
                        evidence += mass
                        if mass > 0:
                            if value is None:
                                unknown += mass
                            else:
                                numerator += mass * value
            known = template["source_public_joint_known"] and unknown == 0
            summaries.append(
                {
                    "law": law.name,
                    "event": event.name,
                    "raw_event_evidence": evidence,
                    "positive_unknown_mass": unknown,
                    "known_payment_numerator_diagnostic": numerator,
                    "net_payment_integral": numerator if known else None,
                    "raw_measure_conditional_payment_ratio": numerator / evidence
                    if known and evidence > 0
                    else None,
                }
            )
    output.update(
        payment_reports=reports,
        raw_law_rows=tuple(law_rows),
        public_law_rows=tuple(law_rows)
        if template["source_public_joint_known"]
        else None,
        summaries=tuple(summaries),
        all_support_bound=required,
    )
    return BranchFinancialResult(_encoded(output))


def branch_financial_record(result):
    _check(
        type(result) is BranchFinancialResult,
        "typed complete branch financial result required",
    )
    return _encoded(result.analyst)
