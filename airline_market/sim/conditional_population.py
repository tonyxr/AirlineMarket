"""A.10/A.5/B0/V9/B4: finite conditional JOINT service/payment programmes.

All original people share one fixed supplied booking prefix. Future laws,
service/transfer incidence and material preferences remain declarations, not
native probability, actual knowledge, lifecycle CS or causal policy effects.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, pairwise

from airline_market.policy.dated_reserve import UtcBookingClock, utc
from airline_market.sim import current_itinerary_service as service
from airline_market.sim import person_material_payoff as material
from airline_market.sim import service_ticket_payments as payments
from airline_market.sim.payoff_logit import ContingentPayoff
from airline_market.sim.received_information import _text, _units
from airline_market.sim.service_observations import (
    ServiceObservationJournal,
    service_observation_journal,
    service_observation_summary,
)
from airline_market.sim.shared_mark_attempt import _encoded, _owned
from airline_market.spec.compiled import CompiledMarket

RULE = "conditional-joint-original-population-service-payment-v1"
SCOPE = "declared-future-law-conditional-on-fixed-original-native-prefix"


class ConditionalPopulationLimit(RuntimeError):
    pass


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _key(key):
    return (
        type(key) is tuple
        and len(key) == 3
        and all(type(v) is int and v >= 0 for v in key)
    )


@dataclass(frozen=True)
class PhysicalServiceProgramme:
    name: str
    operations: tuple
    boarding_intents: tuple
    boarding_orders: tuple
    connections: tuple
    definition: str
    source_ref: str

    def __post_init__(self):
        object.__setattr__(self, "operations", tuple(self.operations))
        for name in ("boarding_intents", "boarding_orders", "connections"):
            object.__setattr__(self, name, _owned(getattr(self, name)))


@dataclass(frozen=True)
class ServiceOutcomeTerms:
    current_itinerary_complete: bool | None
    transfers: tuple
    no_further_ticket_transfers: bool
    financial_followup_source: str
    payoff: ContingentPayoff
    component_sources: tuple
    no_further_material_components: bool
    material_followup_source: str

    def __post_init__(self):
        object.__setattr__(self, "transfers", tuple(self.transfers))
        object.__setattr__(self, "component_sources", _owned(self.component_sources))


@dataclass(frozen=True)
class PersonServiceTerms:
    person_key: tuple
    outcomes: tuple

    def __post_init__(self):
        object.__setattr__(self, "person_key", _owned(self.person_key))
        object.__setattr__(self, "outcomes", tuple(self.outcomes))


@dataclass(frozen=True)
class PopulationPayoffCell:
    physical: str
    financial: str
    persons: tuple
    transfer_order: tuple
    definition: str
    source_ref: str

    def __post_init__(self):
        object.__setattr__(self, "persons", tuple(self.persons))
        object.__setattr__(self, "transfer_order", _owned(self.transfer_order))


@dataclass(frozen=True)
class OriginalPersonInformation:
    person_key: tuple
    decisions: tuple

    def __post_init__(self):
        object.__setattr__(self, "person_key", _owned(self.person_key))
        object.__setattr__(self, "decisions", tuple(self.decisions))


@dataclass(frozen=True)
class ConditionalPopulationProgramme:
    physical: tuple
    financial_states: tuple
    cells: tuple
    information: tuple
    as_of: object
    payoff_unit: str
    conditioning_scope: str
    conditioning_definition: str
    definition: str
    source_ref: str

    def __post_init__(self):
        for name in ("physical", "financial_states", "cells", "information"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "as_of", utc(self.as_of))


@dataclass(frozen=True)
class ConditionalPopulationLaw:
    name: str
    weights: tuple
    conditioning_definition: str
    source_ref: str

    def __post_init__(self):
        object.__setattr__(self, "weights", _owned(self.weights))


@dataclass(frozen=True)
class ConditionalPopulationResult:
    analyst: object
    row_limit: int
    rule: str = RULE

    def __post_init__(self):
        object.__setattr__(self, "analyst", _owned(self.analyst))


def _limit(required, row_limit):
    if required > row_limit:
        raise ConditionalPopulationLimit(
            "complete source/programme/proposed output exceeds infrastructure limit"
        )


def _names(values):
    _check(
        bool(values) and len(set(values)) == len(values),
        "complete distinct named support required",
    )
    for value in values:
        _text(value)


def _baselines(ctx):
    """Pure complete native ticket arithmetic; outside has no native tickets."""
    baseline, owners = {}, {}
    for person in ctx["complete_ordinary_panel"]["persons"]:
        key = tuple(person["potential_coordinate"])
        account = person["booked_passenger_account"]
        if account is None:
            baseline[key] = Fraction()
            continue
        gross, refunds = [], []
        for ticket in account["tickets"]:
            rid = ticket["record_id"]
            _check(rid not in owners, "distinct complete ticket IDs required")
            owners[rid] = (key, ticket["carrier"])
            gross.append(payments._native_amount(ticket["fare"]))
            if ticket["cancellation"] is not None:
                refunds.append(
                    payments._native_amount(ticket["cancellation"]["fare_refunded"])
                )
        g, r = payments._sum(gross), payments._sum(refunds)
        value = payments._minus(g, r)
        _check(
            (g, r, value)
            == tuple(
                payments._rational(account[n])
                for n in ("gross_ticket_fares", "fare_refunds", "net_ticket_fare_paid")
            ),
            "ALL native monetary envelopes and saved sums must agree",
        )
        baseline[key] = value
    return baseline, owners


def _preflight(compiled, journal, programme, laws, row_limit):
    if type(row_limit) is not int or row_limit <= 0:
        raise ValueError("positive structural infrastructure limit required")
    if type(programme) is not ConditionalPopulationProgramme:
        raise TypeError("complete typed conditional programme required")
    if (
        type(compiled) is not CompiledMarket
        or type(journal) is not ServiceObservationJournal
    ):
        raise TypeError("compiled native model and NEW empty supplied journal required")
    if journal.observations:
        raise ValueError("NEW empty journal only; no continuation/replay")
    for name in ("payoff_unit", "conditioning_definition", "definition", "source_ref"):
        _text(getattr(programme, name))
    _check(
        programme.conditioning_scope == SCOPE,
        "explicit fixed-prefix conditional declaration required",
    )
    ctx = journal.context
    # Bound full input and conservatively repeated parent-source/output structures.
    inputs = _encoded(
        {
            "source": ctx,
            "programme": programme,
            "laws": laws,
            "rule": RULE,
            "row_limit": row_limit,
        }
    )
    count = len(programme.cells)
    # Complete event-dependent reports repeat source and shared declarations.
    # Every law also returns people x cells contributions and ALL person pairs.
    max_events = 2 * max(
        (
            len(p.operations)
            for p in programme.physical
            if type(p) is PhysicalServiceProgramme
        ),
        default=0,
    )
    people = len(ctx["persons"])
    required = (
        32 * (_units(inputs) + 1) * (count + 1)
        + 64
        * (_units(_encoded(ctx)) + 1)
        * (len(programme.physical) + 4 * count + 1)
        * (max_events + 1)
        + 256 * (len(laws) + 1) * (people + 1) * (count + people + 1)
        + 50000
    )
    _limit(required, row_limit)
    _check(
        all(type(p) is PhysicalServiceProgramme for p in programme.physical),
        "typed full physical programmes required",
    )
    _names(tuple(p.name for p in programme.physical))
    _names(programme.financial_states)
    expected_cells = tuple(
        (p.name, f) for p in programme.physical for f in programme.financial_states
    )
    _check(
        all(type(c) is PopulationPayoffCell for c in programme.cells)
        and tuple((c.physical, c.financial) for c in programme.cells) == expected_cells,
        "ALL ordered physical x financial cells including zero weight required",
    )
    _check(
        bool(laws) and all(type(law) is ConditionalPopulationLaw for law in laws),
        "complete typed exact conditional laws required",
    )
    _names(tuple(law.name for law in laws))
    for law in laws:
        _text(law.conditioning_definition)
        _text(law.source_ref)
        _check(
            len(law.weights) == count
            and all(type(w) is Fraction and w >= 0 for w in law.weights)
            and sum(law.weights, Fraction()) == 1,
            "ALL joint weights exact nonnegative Fractions summing one required",
        )
    # Pure validation prefix, returning BEFORE any service transition, for ALL P.
    for p in programme.physical:
        _text(p.definition)
        _text(p.source_ref)
        _, events, _targets = _service_preflight(
            compiled,
            journal,
            p.operations,
            p.boarding_intents,
            p.boarding_orders,
            p.connections,
            definition=p.definition,
            support_limit=row_limit,
        )
        _check(
            all(at <= programme.as_of for at, *_ in events),
            "common as-of must cover EVERY declared physical event",
        )
    keys = tuple(p["person_key"] for p in ctx["persons"])
    close = utc(ctx["booking_closed_at"])
    _check(programme.as_of >= close, "common post-close follow-up required")
    _check(
        all(type(i) is OriginalPersonInformation for i in programme.information)
        and tuple(i.person_key for i in programme.information) == keys,
        "ONE shared information declaration for ALL original people in source order required",
    )
    for i in programme.information:
        _check(
            _key(i.person_key) and bool(i.decisions),
            "explicit original key/earlier information required",
        )
        ids, previous = set(), None
        for d in i.decisions:
            _check(
                type(d) is material.DeclaredDecisionInformation
                and d.person_key == i.person_key,
                "same original person typed shared information required",
            )
            for value in (d.decision_id, d.definition, d.source_ref):
                _text(value)
            _check(
                d.decision_id not in ids
                and utc(ctx["booking_clock"][i.person_key[0]]) <= d.decision_at <= close
                and (previous is None or d.decision_at >= previous)
                and not d.service_event_ids
                and not d.transfer_event_ids,
                "shared earlier fixed-prefix information only; no future event disclosures",
            )
            ids.add(d.decision_id)
            previous = d.decision_at
    baseline, owners = _baselines(ctx)
    for cell in programme.cells:
        _text(cell.definition)
        _text(cell.source_ref)
        _check(
            all(type(p) is PersonServiceTerms for p in cell.persons)
            and tuple(p.person_key for p in cell.persons) == keys,
            "ALL original people in EACH joint cell required",
        )
        transfers = {}
        for person, original in zip(cell.persons, ctx["persons"]):
            expected = (
                (False, True) if original["current_ticket"] is not None else (None,)
            )
            _check(
                all(type(t) is ServiceOutcomeTerms for t in person.outcomes),
                "typed complete outcome terms required",
            )
            _check(
                all(
                    t.current_itinerary_complete is None
                    or type(t.current_itinerary_complete) is bool
                    for t in person.outcomes
                )
                and tuple(t.current_itinerary_complete for t in person.outcomes)
                == expected,
                "ALL applicable current-service outcomes including unreachable rows required",
            )
            for term in person.outcomes:
                _check(
                    type(term.no_further_ticket_transfers) is bool
                    and type(term.no_further_material_components) is bool
                    and type(term.payoff) is ContingentPayoff,
                    "explicit FOUR components and independent closures required",
                )
                _text(term.financial_followup_source)
                _text(term.material_followup_source)
                sources = dict(term.component_sources)
                _check(
                    len(sources) == len(term.component_sources)
                    and set(sources) == set(material.COMPONENTS),
                    "source for EACH component required",
                )
                for value in sources.values():
                    _text(value)
                added, refunded = [], []
                for t in term.transfers:
                    _check(
                        type(t) is payments.AdditionalTicketTransfer
                        and t.person_key == person.person_key
                        and type(t.ticket_record) is int
                        and owners.get(t.ticket_record)
                        == (person.person_key, t.carrier)
                        and t.kind in ("additional_payment", "additional_refund"),
                        "same original-root ticket/carrier transfer required",
                    )
                    _text(t.event_id)
                    _text(t.source_ref)
                    _check(
                        t.event_id not in transfers
                        and close <= t.occurred_at <= t.recorded_at <= programme.as_of,
                        "distinct contingent transfer IDs/causal post-close times required",
                    )
                    _check(
                        t.amount is None or type(t.amount) is float,
                        "explicit binary64-or-unknown additional transfer required",
                    )
                    amount = (
                        None
                        if t.amount is None
                        else payments._native_amount(payments._binary(t.amount))
                    )
                    (added if t.kind == "additional_payment" else refunded).append(
                        amount
                    )
                    transfers[t.event_id] = t
                observed = payments._sum(
                    (
                        baseline[person.person_key],
                        payments._minus(payments._sum(added), payments._sum(refunded)),
                    )
                )
                final = observed if term.no_further_ticket_transfers else None
                _check(
                    term.payoff.net_ticket_payment == final,
                    "EVERY outcome NET payment must match its complete explicit financial plan, never prefix/rounding/fill",
                )
        _check(
            len(set(cell.transfer_order)) == len(cell.transfer_order)
            and set(cell.transfer_order) == set(transfers),
            "explicit complete contingent transfer order required",
        )
        times = tuple(transfers[e].recorded_at for e in cell.transfer_order)
        _check(
            all(a <= b for a, b in pairwise(times)),
            "declared full transfer order must preserve recording times; no sorting/repair",
        )
    return required


def _mean(weights, values):
    contributions = tuple(
        Fraction() if w == 0 else None if v is None else w * v
        for w, v in zip(weights, values)
    )
    return {
        "mean": None if None in contributions else sum(contributions, Fraction()),
        "contributions": contributions,
        "unknown_mass": sum(
            (w for w, v in zip(weights, values) if v is None), Fraction()
        ),
        "known_subtotal_diagnostic_only": sum(
            (v for v in contributions if v is not None), Fraction()
        ),
    }


def evaluate_conditional_population(compiled, journal, programme, laws, *, row_limit):
    """Execute ALL declared future support; never generate a native source."""
    laws = tuple(laws)
    required = _preflight(compiled, journal, programme, laws, row_limit)
    ctx = journal.context
    source = _owned(_encoded(ctx))
    owned_programme = _owned(_encoded(programme))
    owned_laws = _owned(_encoded(laws))
    physical_results, cell_results = [], []
    for p in programme.physical:
        execution = service.execute_current_itinerary_service(
            compiled,
            journal,
            p.operations,
            p.boarding_intents,
            p.boarding_orders,
            p.connections,
            definition=p.definition,
            support_limit=row_limit,
        )
        summary = service_observation_summary(execution.journal, as_of=programme.as_of)
        delivery = tuple(
            r["observed_current_itinerary_complete"] for r in summary["persons"]
        )
        physical_results.append(
            {
                "physical": p.name,
                "execution": service.current_service_record(execution),
                "summary": _encoded(summary),
            }
        )
        for cell in (c for c in programme.cells if c.physical == p.name):
            selected = tuple(
                next(
                    t
                    for t in person.outcomes
                    if t.current_itinerary_complete is outcome
                )
                for person, outcome in zip(cell.persons, delivery)
            )
            available = {t.event_id: t for term in selected for t in term.transfers}
            transfers = tuple(
                available[e] for e in cell.transfer_order if e in available
            )
            followup = tuple(
                payments.TicketPaymentFollowup(
                    person.person_key,
                    programme.as_of,
                    term.no_further_ticket_transfers,
                    term.financial_followup_source,
                )
                for person, term in zip(cell.persons, selected)
            )
            payment = payments.reconcile_service_ticket_payments(
                execution,
                transfers,
                followup,
                as_of=programme.as_of,
                native_ticket_unit=programme.payoff_unit,
                definition=cell.definition,
                row_limit=row_limit,
            )
            declarations = tuple(
                material.OriginalPersonMaterialPayoff(
                    person.person_key,
                    term.payoff,
                    term.component_sources,
                    programme.as_of,
                    term.no_further_material_components,
                    term.material_followup_source,
                    info.decisions,
                )
                for person, term, info in zip(
                    cell.persons, selected, programme.information
                )
            )
            payoff = material.reconcile_person_material_payoff(
                payment,
                declarations,
                payoff_unit=programme.payoff_unit,
                definition=cell.definition,
                scope=material.SCOPE,
                shock_rule=material.SHOCK_RULE,
                row_limit=row_limit,
            )
            cell_results.append(
                {
                    "physical": p.name,
                    "financial": cell.financial,
                    "delivery": delivery,
                    "selected_terms": _encoded(selected),
                    "payments": payments.service_ticket_payment_record(payment),
                    "material": material.person_material_payoff_record(payoff),
                    "person_payments": tuple(
                        payments._rational(r["declared_closed_total_ticket_payment"])
                        for r in payment.data["all_original_persons"]
                    ),
                    "person_material": tuple(
                        payments._rational(r["declared_closed_material_payoff"])
                        for r in payoff.data["all_original_persons"]
                    ),
                    "population_payment": payments._rational(
                        payment.data["declared_closed_population_total_ticket_payment"]
                    ),
                    "population_material": payments._rational(
                        payoff.data["declared_closed_population_material_payoff"]
                    ),
                }
            )
    results = []
    keys = tuple(r["person_key"] for r in ctx["persons"])
    applicable = tuple(r["current_ticket"] is not None for r in ctx["persons"])
    for law in laws:
        weights = law.weights
        joint = dict.fromkeys((c["delivery"] for c in cell_results), Fraction())
        for w, c in zip(weights, cell_results):
            joint[c["delivery"]] += w
        marginals = tuple(
            sum(
                (w for w, c in zip(weights, cell_results) if c["delivery"][i] is True),
                Fraction(),
            )
            if applies
            else None
            for i, applies in enumerate(applicable)
        )
        pairs = []
        for i, j in combinations(range(len(keys)), 2):
            both = (
                sum(
                    (
                        w
                        for w, c in zip(weights, cell_results)
                        if c["delivery"][i] is True and c["delivery"][j] is True
                    ),
                    Fraction(),
                )
                if applicable[i] and applicable[j]
                else None
            )
            product = None if both is None else marginals[i] * marginals[j]
            pairs.append(
                {
                    "person_keys": (keys[i], keys[j]),
                    "both_current_delivered": both,
                    "product_marginals_DIFFERENT_joint": product,
                    "covariance_declared_law": None if both is None else both - product,
                }
            )
        counts = tuple(sum(x is True for x in c["delivery"]) for c in cell_results)
        mean = sum((w * n for w, n in zip(weights, counts)), Fraction())
        results.append(
            {
                "name": law.name,
                "weights": weights,
                "joint_delivery": tuple(
                    {"delivery": d, "mass": w} for d, w in joint.items()
                ),
                "current_service_marginals": marginals,
                "pairwise_current_service": tuple(pairs),
                "delivered_current_count_mean": mean,
                "delivered_current_count_variance_declared_law": sum(
                    (w * n * n for w, n in zip(weights, counts)), Fraction()
                )
                - mean * mean,
                "population_payment": _mean(
                    weights, tuple(c["population_payment"] for c in cell_results)
                ),
                "population_material": _mean(
                    weights, tuple(c["population_material"] for c in cell_results)
                ),
                "persons": tuple(
                    {
                        "person_key": key,
                        "payment": _mean(
                            weights,
                            tuple(c["person_payments"][i] for c in cell_results),
                        ),
                        "material": _mean(
                            weights,
                            tuple(c["person_material"][i] for c in cell_results),
                        ),
                    }
                    for i, key in enumerate(keys)
                ),
            }
        )
    return ConditionalPopulationResult(
        {
            "plan_id": "A.10/A.5/B0/V9/B4",
            "source_context": source,
            "programme": owned_programme,
            "laws": owned_laws,
            "physical_results": tuple(physical_results),
            "cells": tuple(cell_results),
            "law_results": tuple(results),
            "structural_bound": required,
            "service_state_machine_calls": len(programme.physical),
            "payment_reconciliation_calls": len(programme.cells),
            "material_reconciliation_calls": len(programme.cells),
            "all_original_people": keys,
            "empty_population_sum_only": not keys,
            "source_native_engine_account_panel_actor_RNG_or_choice_executed": False,
            "native_probability_knowledge_preferences_authenticated": False,
            "lifecycle_utility_CS_total_denial_misallocation": None,
            "population_service_rate": None,
            "primary_default_empirical_production_planner_adopted": False,
            "scientific_budget": None,
        },
        row_limit,
    )


def conditional_population_record(result):
    if type(result) is not ConditionalPopulationResult or result.rule != RULE:
        raise TypeError("complete typed conditional population result required")
    return _encoded(result)


def _service_preflight(
    compiled,
    journal,
    operations,
    boarding_intents,
    boarding_orders,
    connections,
    *,
    definition,
    support_limit,
):
    if (
        type(compiled) is not CompiledMarket
        or type(journal) is not ServiceObservationJournal
    ):
        raise TypeError("native compiled model and NEW typed journal required")
    if journal.observations:
        raise ValueError("NEW empty journal required; no continuation or replay")
    if not isinstance(definition, str) or not definition.strip():
        raise ValueError("explicit service-model definition required")
    if type(support_limit) is not int or support_limit < 1:
        raise ValueError("positive ALL-support infrastructure limit required")
    ctx = journal.context
    # Revalidate complete supplied values, not a producer/loader or source seal.
    checked = service_observation_journal(
        compiled,
        ctx["complete_ordinary_panel"],
        UtcBookingClock(tuple(utc(t) for t in ctx["booking_clock"])),
        ctx["leg_departures"],
        definition=ctx["definition"],
        observation_scope=ctx["observation_scope"],
        context_row_limit=ctx["context_row_limit"],
        observation_limit=ctx["observation_limit"],
    )
    if checked.context != ctx:
        raise ValueError("complete native model/panel/clock/journal context differs")
    close = utc(ctx["booking_closed_at"])
    scheduled = {k: utc(v) for k, v in ctx["leg_departures"]}
    ops = tuple(operations)
    if any(type(o) is not service.LegOperation for o in ops):
        raise TypeError("typed complete leg-operation rows required")
    if len({o.leg for o in ops}) != len(ops) or {o.leg for o in ops} != set(scheduled):
        raise ValueError("one operation for EVERY native leg required")
    event_program = []
    for o in ops:
        if type(o.seats) is not int or o.seats < 0:
            raise ValueError("explicit nonnegative integer physical seats required")
        if o.status == "operates":
            if (
                o.departure is None
                or o.arrival is None
                or o.cancellation is not None
                or o.departure < scheduled[o.leg]
                or o.arrival <= o.departure
            ):
                raise ValueError(
                    "explicit non-early departure and later arrival required"
                )
            event_program.extend(
                ((o.departure, 2, o.leg, "departure"), (o.arrival, 1, o.leg, "arrival"))
            )
        elif o.status == "cancelled":
            if (
                o.cancellation is None
                or o.cancellation < close
                or o.departure is not None
                or o.arrival is not None
            ):
                raise ValueError("explicit post-close cancellation only required")
            event_program.append((o.cancellation, 0, o.leg, "cancellation"))
        else:
            raise ValueError("explicit operates/cancelled operation required")
    persons = tuple(ctx["persons"])
    tickets = {
        p["current_ticket"]: p for p in persons if p["current_ticket"] is not None
    }
    targets = {(rid, leg) for rid, p in tickets.items() for leg in p["required_legs"]}
    required_connections = set()
    legs = {leg.id: leg for leg in compiled.spec.legs}
    for rid, p in tickets.items():
        itinerary = tuple(p["required_legs"])
        if len(set(itinerary)) != len(itinerary):
            raise ValueError("repeated-leg itineraries unsupported on this path")
        for a, b in pairwise(itinerary):
            if legs[a].dest != legs[b].origin:
                raise ValueError("declared itinerary leg endpoints do not connect")
            required_connections.add((rid, a, b))
    intent_rows = tuple(tuple(row) for row in boarding_intents)
    intents = {}
    for rid, leg, intent in intent_rows:
        if type(rid) is not int or type(intent) is not bool or (rid, leg) in intents:
            raise ValueError("distinct ticket/leg and explicit boolean intent required")
        intents[rid, leg] = intent
    if set(intents) != targets:
        raise ValueError("ALL current ticket/leg intents required, without extras")
    order_rows = tuple((leg, tuple(order)) for leg, order in boarding_orders)
    orders = {}
    for leg, order in order_rows:
        expected = {rid for rid, k in targets if k == leg}
        if (
            leg in orders
            or leg not in scheduled
            or any(type(rid) is not int for rid in order)
            or len(set(order)) != len(order)
            or set(order) != expected
        ):
            raise ValueError("full per-leg boarding order including no-shows required")
        orders[leg] = order
    if set(orders) != set(scheduled):
        raise ValueError("boarding orders for EVERY leg including empty legs required")
    connection_rows = tuple(tuple(row) for row in connections)
    minimums = {}
    for rid, a, b, micros in connection_rows:
        if (
            type(rid) is not int
            or type(micros) is not int
            or micros < 0
            or (rid, a, b) in minimums
        ):
            raise ValueError(
                "distinct connections and explicit nonnegative integer microseconds required"
            )
        minimums[rid, a, b] = micros
    if set(minimums) != required_connections:
        raise ValueError("ALL adjacent current-itinerary connection minima required")
    rows = (
        len(persons)
        + len(ops)
        + len(event_program)
        + len(intent_rows)
        + len(order_rows)
        + sum(len(v) for v in orders.values())
        + len(connection_rows)
    )
    if rows > support_limit or len(targets) > ctx["observation_limit"]:
        raise RuntimeError(
            "ALL service support exceeds limit; no partial/fallback/retry"
        )
    return rows, event_program, targets
