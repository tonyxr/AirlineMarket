"""A.10/A.5/B0/V9: explicit post-booking ticket-transfer reconciliation.

Read-only NEW service inputs, not a refund policy, cash mutation, utility or CS.
"""

import struct
from dataclasses import dataclass
from fractions import Fraction

from airline_market.core.snapshot import deep_freeze
from airline_market.policy.dated_reserve import utc
from airline_market.sim.current_itinerary_service import (
    CurrentItineraryService,
    current_service_record,
)
from airline_market.sim.passenger_accounts import _binary, _encode, _number
from airline_market.sim.service_observations import service_observation_summary


@dataclass(frozen=True)
class AdditionalTicketTransfer:
    event_id: str
    person_key: tuple
    ticket_record: int
    carrier: str
    kind: str
    amount: object
    occurred_at: object
    recorded_at: object
    source_ref: str

    def __post_init__(self):
        object.__setattr__(self, "person_key", tuple(self.person_key))
        object.__setattr__(self, "occurred_at", utc(self.occurred_at))
        object.__setattr__(self, "recorded_at", utc(self.recorded_at))


@dataclass(frozen=True)
class TicketPaymentFollowup:
    person_key: tuple
    observed_through: object
    no_further_ticket_transfers: bool
    source_ref: str

    def __post_init__(self):
        object.__setattr__(self, "person_key", tuple(self.person_key))
        object.__setattr__(self, "observed_through", utc(self.observed_through))


@dataclass(frozen=True)
class ServiceTicketPayments:
    data: object

    def __post_init__(self):
        object.__setattr__(self, "data", deep_freeze(self.data))


def _rational(value):
    if value is None:
        return None
    q = Fraction(int(value["numerator"]), int(value["denominator"]))
    if _number(q) != value:
        raise ValueError("canonical exact rational envelope required")
    return q


def _native_amount(envelope):
    value = struct.unpack(">d", bytes.fromhex(envelope["binary64_bits"]))[0]
    if _binary(value) != envelope:
        raise ValueError("complete native binary64 envelope differs")
    return _rational(envelope["exact"])


def _sum(values):
    values = tuple(values)
    return None if any(v is None for v in values) else sum(values, Fraction(0))


def _minus(a, b):
    return None if a is None or b is None else a - b


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _key(value):
    return len(value) == 3 and all(type(v) is int and v >= 0 for v in value)


def reconcile_service_ticket_payments(
    service,
    transfers,
    followup,
    *,
    as_of,
    native_ticket_unit,
    definition,
    row_limit,
):
    """Join explicit ADDITIONAL ticket transfers to ALL original people.

    Payment adds to net ticket payment; refund subtracts. Signed amounts stay
    signed. Missing/nonfinite inputs and open future follow-up never become 0.
    No service outcome generates a refund or closes financial follow-up.
    """
    if type(service) is not CurrentItineraryService:
        raise TypeError("NEW typed current-itinerary service result required")
    if not _text(native_ticket_unit) or not _text(definition):
        raise ValueError("explicit common native ticket unit and definition required")
    if type(row_limit) is not int or row_limit < 1:
        raise ValueError("positive ALL-support infrastructure limit required")
    as_of = utc(as_of)
    ctx = service.journal.context
    if service.context["source_context"] != deep_freeze(_encode(ctx)):
        raise ValueError("full source journal context differs from service binding")
    close = utc(ctx["booking_closed_at"])
    operations = tuple(service.context["operations"])
    if len({o["leg"] for o in operations}) != len(operations) or {
        o["leg"] for o in operations
    } != {leg for leg, _ in ctx["leg_departures"]}:
        raise ValueError("ALL native leg-operation support required")
    event_program = []
    for o in operations:
        if o["status"] == "operates":
            event_program.extend(
                (
                    (utc(o["departure"]), 2, o["leg"], "departure"),
                    (utc(o["arrival"]), 1, o["leg"], "arrival"),
                )
            )
        elif o["status"] == "cancelled":
            event_program.append((utc(o["cancellation"]), 0, o["leg"], "cancellation"))
        else:
            raise ValueError("explicit complete leg-operation event program required")
    order = {"cancellation": 0, "arrival": 1, "departure": 2}
    actual_program = tuple(
        (utc(s["at"]), order[s["kind"]], s["leg"], s["kind"]) for s in service.steps
    )
    if actual_program != tuple(sorted(event_program)):
        raise ValueError("ALL ordered service events including empty legs required")
    if not service.steps or as_of < max(utc(s["at"]) for s in service.steps):
        raise ValueError("complete service event program and later as-of required")
    people = tuple(ctx["persons"])
    keys = tuple(p["person_key"] for p in people)
    observations = service.journal.observations
    ids = tuple(o.event_id for o in observations)
    published = tuple(k for s in service.steps for k in s["new_observation_ids"])
    if (
        len(set(keys)) != len(keys)
        or len(set(ids)) != len(ids)
        or published != ids
        or any(tuple(s["all_person_keys"]) != keys for s in service.steps)
    ):
        raise ValueError(
            "complete distinct service receipts and all original keys required"
        )
    summary = service_observation_summary(service.journal, as_of=as_of)
    targets = {
        (p["person_key"], p["current_ticket"], leg)
        for p in people
        for leg in p["required_legs"]
    }
    if {(o.person_key, o.ticket_record, o.leg) for o in observations} != targets or any(
        o.outcome not in {"delivered", "not_delivered"} for o in observations
    ):
        raise ValueError(
            "complete current person-leg service outcome partition required"
        )
    panel = ctx["complete_ordinary_panel"]
    by_key = {tuple(p["potential_coordinate"]): p for p in panel["persons"]}
    if len(by_key) != len(panel["persons"]) or set(by_key) != set(keys):
        raise ValueError("full source panel original-person support differs")
    followup = tuple(followup)
    transfers = tuple(transfers)
    if any(type(f) is not TicketPaymentFollowup for f in followup):
        raise TypeError("explicit typed follow-up for EVERY original person required")
    coverage = {}
    for f in followup:
        if (
            not _key(f.person_key)
            or f.person_key in coverage
            or f.person_key not in by_key
            or f.observed_through != as_of
            or as_of < close
            or type(f.no_further_ticket_transfers) is not bool
            or not _text(f.source_ref)
        ):
            raise ValueError(
                "distinct complete common-as-of follow-up declarations required"
            )
        coverage[f.person_key] = f
    if set(coverage) != set(keys):
        raise ValueError(
            "ALL people including outside/unreserved need financial follow-up"
        )
    ticket_owner, baseline, all_tickets = {}, {}, 0
    for key, p in by_key.items():
        account = p["booked_passenger_account"]
        person = next(row for row in people if row["person_key"] == key)
        if person["original_record"] != p["original_record"]:
            raise ValueError("source original root differs from current service person")
        if account is None:
            if p["original_record"] is not None:
                raise ValueError("outside person cannot acquire a ticket account")
            baseline[key] = Fraction(
                0
            )  # Recorded native ticket ledger only, not outside utility.
            continue
        if (
            tuple(account["potential_coordinate"]) != key
            or account["original_record"] != p["original_record"]
            or account["current_active_record"] != person["current_ticket"]
            or p["original_record"] not in {t["record_id"] for t in account["tickets"]}
        ):
            raise ValueError("original person/root/account differs")
        gross, refunds = [], []
        for ticket in account["tickets"]:
            rid = ticket["record_id"]
            if type(rid) is not int or rid < 0 or rid in ticket_owner:
                raise ValueError("complete distinct native ticket IDs required")
            ticket_owner[rid] = (key, ticket["carrier"])
            fare = _native_amount(ticket["fare"])
            gross.append(fare)
            cancel = ticket["cancellation"]
            if cancel is not None:
                refunds.append(_native_amount(cancel["fare_refunded"]))
            all_tickets += 1
        g, r = _sum(gross), _sum(refunds)
        n = _minus(g, r)
        if (
            g != _rational(account["gross_ticket_fares"])
            or r != _rational(account["fare_refunds"])
            or n != _rational(account["net_ticket_fare_paid"])
        ):
            raise ValueError("full native ticket arithmetic and saved account differ")
        baseline[key] = n
    rows = (
        ctx["all_context_rows"]
        + service.context["all_support_rows"]
        + len(service.steps)
        + len(observations)
        + all_tickets
        + len(followup)
        + len(transfers)
    )
    if rows > row_limit:
        raise RuntimeError(
            "ALL source/transfer/follow-up rows exceed limit; no partial/fallback/retry"
        )
    events, event_ids = [], set()
    amounts = {key: {"additional_payment": [], "additional_refund": []} for key in keys}
    previous_recording = None
    for t in transfers:
        if type(t) is not AdditionalTicketTransfer:
            raise TypeError("typed explicit additional ticket transfer required")
        if (
            not _text(t.event_id)
            or t.event_id in event_ids
            or not _text(t.source_ref)
            or not _key(t.person_key)
            or t.kind not in {"additional_payment", "additional_refund"}
            or type(t.ticket_record) is not int
            or ticket_owner.get(t.ticket_record) != (t.person_key, t.carrier)
            or t.person_key not in coverage
        ):
            raise ValueError(
                "distinct additional transfer/current original-root ticket/carrier/source required"
            )
        if (
            t.occurred_at < close
            or t.recorded_at < t.occurred_at
            or t.recorded_at > as_of
            or (previous_recording is not None and t.recorded_at < previous_recording)
        ):
            raise ValueError(
                "post-close occurrence/causal ordered recording/common follow-up required"
            )
        if t.amount is not None and type(t.amount) is not float:
            raise TypeError("explicit binary64 amount or unknown None required")
        raw = None if t.amount is None else _binary(t.amount)
        amount = None if raw is None else _rational(raw["exact"])
        amounts[t.person_key][t.kind].append(amount)
        event_ids.add(t.event_id)
        previous_recording = t.recorded_at
        events.append(
            {
                "event_id": t.event_id,
                "person_key": t.person_key,
                "ticket_record": t.ticket_record,
                "carrier": t.carrier,
                "kind": t.kind,
                "amount": raw,
                "occurred_at": t.occurred_at.isoformat(),
                "recorded_at": t.recorded_at.isoformat(),
                "source_ref": t.source_ref,
            }
        )
    output, observed_values, closed_values = [], [], []
    for p in people:
        key = p["person_key"]
        paid = _sum(amounts[key]["additional_payment"])
        refunded = _sum(amounts[key]["additional_refund"])
        additional = _minus(paid, refunded)
        observed = _sum((baseline[key], additional))
        closed = observed if coverage[key].no_further_ticket_transfers else None
        observed_values.append(observed)
        closed_values.append(closed)
        output.append(
            {
                "person_key": key,
                "original_record": p["original_record"],
                "current_ticket": p["current_ticket"],
                "booking_state": p["booking_state"],
                "native_recorded_net_ticket_payment": _number(baseline[key]),
                "additional_payments": _number(paid),
                "additional_refunds": _number(refunded),
                "observed_net_ticket_payment": _number(observed),
                "declared_closed_total_ticket_payment": _number(closed),
                "observed_through": as_of.isoformat(),
                "no_further_ticket_transfers_declared": coverage[
                    key
                ].no_further_ticket_transfers,
                "followup_source_ref": coverage[key].source_ref,
                "additional_transfer_event_ids": tuple(
                    e["event_id"] for e in events if e["person_key"] == key
                ),
                "service_benefit": None,
                "waiting_loss": None,
                "cancellation_loss": None,
                "lifecycle_utility_or_consumer_surplus": None,
            }
        )
    return ServiceTicketPayments(
        {
            "plan_id": "A.10/A.5/B0/V9",
            "definition": definition,
            "native_ticket_unit": native_ticket_unit,
            "as_of": as_of.isoformat(),
            "complete_service_input": current_service_record(service),
            "service_observation_summary": _encode(summary),
            "all_original_persons": tuple(output),
            "additional_transfers": tuple(events),
            "observed_population_net_ticket_payment": _number(_sum(observed_values)),
            "declared_closed_population_total_ticket_payment": _number(
                _sum(closed_values)
            ),
            "all_support_rows": rows,
            "row_limit": row_limit,
            "financial_closure_or_transfer_source_authenticated": False,
            "native_cash_refund_ticket_or_service_mutated": False,
            "native_actor_RNG_utility_producer_or_solver_executed": False,
            "carrier_profit_or_passenger_penalty_incidence": None,
            "lifecycle_consumer_surplus_or_total_surplus": None,
            "primary_default_empirical_production_planner_adopted": False,
            "scientific_budget": None,
        }
    )


def service_ticket_payment_record(result):
    if type(result) is not ServiceTicketPayments:
        raise TypeError("typed service ticket-payment reconciliation required")
    return _encode(result.data)
