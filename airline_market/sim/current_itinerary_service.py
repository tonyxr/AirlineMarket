"""A.10/A.5/B0/V9: deterministic post-booking current-itinerary service.

Separate opt-in service model, conditional on complete declared operation,
boarding-intent, priority and connection plans. No native booking-engine change.
"""

from dataclasses import dataclass
from datetime import timedelta
from itertools import pairwise

from airline_market.core.snapshot import deep_freeze
from airline_market.policy.dated_reserve import UtcBookingClock, utc
from airline_market.sim.passenger_accounts import _encode
from airline_market.sim.service_observations import (
    LegServiceObservation,
    ServiceObservationJournal,
    record_leg_service,
    service_observation_journal,
)
from airline_market.spec.compiled import CompiledMarket


@dataclass(frozen=True)
class LegOperation:
    leg: str
    status: str
    seats: int
    departure: object
    arrival: object
    cancellation: object

    def __post_init__(self):
        for name in ("departure", "arrival", "cancellation"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, utc(value))


@dataclass(frozen=True)
class CurrentItineraryService:
    context: object
    steps: tuple
    journal: ServiceObservationJournal

    def __post_init__(self):
        object.__setattr__(self, "context", deep_freeze(self.context))
        object.__setattr__(self, "steps", tuple(deep_freeze(s) for s in self.steps))


def execute_current_itinerary_service(
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
    """Execute one complete NEW deterministic service program, without RNG.

    Intents are (ticket, leg, bool), orders are (leg, ALL target ticket IDs),
    connections are (ticket, previous leg, next leg, minimum microseconds).
    Cancellation/arrival precede departure at a shared instant. Cancellation
    only changes this service state, never the native ticket/cash ledger.
    """
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
    if any(type(o) is not LegOperation for o in ops):
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
    operations_by_leg = {o.leg: o for o in ops}
    completed, in_flight, outcomes, steps = {}, {}, {}, []
    result_journal = journal

    def observe(rid, leg, when, outcome, reason):
        nonlocal result_journal
        if (rid, leg) in outcomes:
            raise RuntimeError("internal duplicate service outcome")
        event_id = f"current-service/{rid}/{leg}/{outcome}"
        result_journal = record_leg_service(
            result_journal,
            LegServiceObservation(
                event_id,
                tickets[rid]["person_key"],
                rid,
                leg,
                outcome,
                when,
                when,
                f"deterministic-current-itinerary-service:{reason}",
            ),
        )
        outcomes[rid, leg] = outcome
        return event_id

    for when, _, leg, kind in sorted(event_program):
        o = operations_by_leg[leg]
        decisions, new_ids = [], []
        if kind == "cancellation":
            for rid in orders[leg]:
                new_ids.append(
                    observe(rid, leg, when, "not_delivered", "operation_cancelled")
                )
                decisions.append((rid, "operation_cancelled"))
        elif kind == "departure":
            boarded = []
            for rid in orders[leg]:
                itinerary = tickets[rid]["required_legs"]
                i = itinerary.index(leg)
                prior = itinerary[:i]
                if not intents[rid, leg]:
                    reason = "no_boarding_intent"
                elif any((rid, k) not in completed for k in prior):
                    reason = "prior_service_not_completed"
                elif i and when - completed[rid, prior[-1]] < timedelta(
                    microseconds=minimums[rid, prior[-1], leg]
                ):
                    reason = "minimum_connection_not_met"
                elif len(boarded) >= o.seats:
                    reason = "no_seat"
                else:
                    reason = "boarded"
                    boarded.append(rid)
                decisions.append((rid, reason))
                if reason != "boarded":
                    new_ids.append(observe(rid, leg, when, "not_delivered", reason))
            in_flight[leg] = tuple(boarded)
        else:
            for rid in in_flight.pop(leg):
                completed[rid, leg] = when
                new_ids.append(
                    observe(rid, leg, when, "delivered", "boarded_flight_arrival")
                )
                decisions.append((rid, "delivered"))
        steps.append(
            {
                "at": when.isoformat(),
                "kind": kind,
                "leg": leg,
                "all_target_tickets": orders[leg],
                "decisions": tuple(decisions),
                "physical_seats": o.seats,
                "boarded_count": sum(reason == "boarded" for _, reason in decisions)
                if kind == "departure"
                else None,
                "unused_departure_seats": o.seats - len(in_flight[leg])
                if kind == "departure"
                else None,
                "new_observation_ids": tuple(new_ids),
                "in_flight": tuple(sorted(in_flight.items())),
                "completed_person_legs": tuple(
                    (rid, k, t.isoformat()) for (rid, k), t in sorted(completed.items())
                ),
                "all_person_keys": tuple(p["person_key"] for p in persons),
            }
        )
    if set(outcomes) != targets or in_flight:
        raise RuntimeError("internal complete service partition failed")
    return CurrentItineraryService(
        {
            "plan_id": "A.10/A.5/B0/V9",
            "definition": definition,
            "source_context": _encode(ctx),
            "operations": tuple(
                {
                    "leg": o.leg,
                    "status": o.status,
                    "seats": o.seats,
                    "departure": None
                    if o.departure is None
                    else o.departure.isoformat(),
                    "arrival": None if o.arrival is None else o.arrival.isoformat(),
                    "cancellation": None
                    if o.cancellation is None
                    else o.cancellation.isoformat(),
                }
                for o in ops
            ),
            "boarding_intents": intent_rows,
            "boarding_orders": order_rows,
            "connections_minimum_microseconds": connection_rows,
            "all_support_rows": rows,
            "support_limit": support_limit,
            "event_order": "cancellation, arrival, departure at equal UTC time; leg ID ties",
            "new_deterministic_service_state_machine_executed": True,
            "native_booking_departure_engine_actor_or_RNG_executed": False,
            "physical_probability_public_empirical_or_source_truth_authenticated": False,
            "post_booking_payments_refunds_or_lifecycle_utility": None,
            "primary_default_empirical_production_planner_adopted": False,
            "scientific_budget": None,
        },
        tuple(steps),
        result_journal,
    )


def current_service_record(result):
    """Owned lossless export; no producer or saved-result loader."""
    if type(result) is not CurrentItineraryService:
        raise TypeError("typed current-itinerary service result required")
    return _encode(
        {
            "context": result.context,
            "steps": result.steps,
            "journal_context": result.journal.context,
            "observations": tuple(
                {
                    "event_id": o.event_id,
                    "person_key": o.person_key,
                    "ticket_record": o.ticket_record,
                    "leg": o.leg,
                    "outcome": o.outcome,
                    "occurred_at": o.occurred_at.isoformat(),
                    "recorded_at": o.recorded_at.isoformat(),
                    "source_ref": o.source_ref,
                }
                for o in result.journal.observations
            ),
        }
    )
