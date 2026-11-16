"""A.10/A.5/B0/V9: prospective individual leg-service observations.

Declared observations on a NEW complete booking panel. No native departure
engine, outcome authentication, actor input or lifecycle welfare adoption.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise

from airline_market.core.hashing import jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.policy.dated_reserve import UtcBookingClock, utc
from airline_market.sim.ordinary_cohort import RULE
from airline_market.sim.passenger_accounts import _encode
from airline_market.spec.compiled import CompiledMarket


@dataclass(frozen=True)
class LegServiceObservation:
    event_id: str
    person_key: tuple
    ticket_record: int
    leg: str
    outcome: str
    occurred_at: object
    recorded_at: object
    source_ref: str

    def __post_init__(self):
        object.__setattr__(self, "person_key", tuple(self.person_key))
        object.__setattr__(self, "occurred_at", utc(self.occurred_at))
        object.__setattr__(self, "recorded_at", utc(self.recorded_at))


@dataclass(frozen=True)
class ServiceObservationJournal:
    context: object
    observations: tuple

    def __post_init__(self):
        object.__setattr__(self, "context", deep_freeze(self.context))
        object.__setattr__(self, "observations", tuple(self.observations))


def service_observation_journal(
    compiled,
    panel,
    clock,
    leg_departures,
    *,
    definition,
    observation_scope,
    context_row_limit,
    observation_limit,
):
    """Bind ALL supplied people/tickets/current legs without replaying producers.

    Requires the complete booking horizon ending before every departure.
    Current reservations define observation support, never delivered service.
    """
    if type(compiled) is not CompiledMarket or type(clock) is not UtcBookingClock:
        raise TypeError("native compiled model and explicit UTC booking clock required")
    if not isinstance(panel, Mapping) or panel.get("capture_rule") != RULE:
        raise TypeError("NEW complete ordinary-person panel required")
    if (
        not isinstance(definition, str)
        or not definition.strip()
        or observation_scope != "declared-individual-leg-observations"
    ):
        raise ValueError("explicit individual observation definition/scope required")
    if any(type(n) is not int or n < 1 for n in (context_row_limit, observation_limit)):
        raise ValueError("positive full-support infrastructure limits required")
    c = compiled
    T = c.spec.horizon.n_periods
    if (
        panel["completed_native_periods"] != T
        or len(clock.boundaries) != T + 1
        or len(panel["period_request_counts"]) != T
        or any(type(n) is not int or n < 0 for n in panel["period_request_counts"])
        or sum(panel["period_request_counts"]) != len(panel["persons"])
        or panel["ordinary_original_requests"] != len(panel["persons"])
    ):
        raise ValueError("complete booking-horizon panel/empty periods/clock required")
    dep = tuple((k, utc(v)) for k, v in leg_departures)
    if len({k for k, v in dep}) != len(dep) or {k for k, v in dep} != set(c.leg_index):
        raise ValueError("one explicit departure for EVERY native leg required")
    departures = dict(dep)
    if clock.at(T) >= min(departures.values()):
        raise ValueError(
            "booking horizon must end before EVERY departure; no engine extension"
        )
    people, keys, roots, all_ids, indices = [], set(), set(), set(), set()
    ticket_rows = current_legs = outside = 0
    by_period = [0] * T
    for row in panel["persons"]:
        key = tuple(row["potential_coordinate"])
        if (
            len(key) != 3
            or any(type(v) is not int or v < 0 for v in key)
            or key[0] >= T
            or key[2] >= len(c.spec.markets)
            or key in keys
            or row["request_market"] != c.spec.markets[key[2]].id
            or row["segment"] not in c.segment_index
            or type(row["taste_class"]) is not int
            or not 0
            <= row["taste_class"]
            < len(c.spec.segments[c.segment_index[row["segment"]]].classes)
            or type(row["native_log_index"]) is not int
            or row["native_log_index"] in indices
        ):
            raise ValueError(
                "complete distinct original person/native support required"
            )
        keys.add(key)
        indices.add(row["native_log_index"])
        by_period[key[0]] += 1
        account = row["booked_passenger_account"]
        root = row["original_record"]
        current = None
        legs = ()
        product = None
        if account is None:
            if root is not None:
                raise ValueError("outside person must not acquire an invented account")
            outside += 1
        else:
            if (
                type(root) is not int
                or root < 0
                or root in roots
                or root != account["original_record"]
                or tuple(account["potential_coordinate"]) != key
                or account["request_market"] != row["request_market"]
                or account["segment"] != row["segment"]
            ):
                raise ValueError(
                    "one matching original account/root per booked person required"
                )
            roots.add(root)
            tickets = tuple(account["tickets"])
            ids = []
            for ticket in tickets:
                rid = ticket["record_id"]
                if (
                    type(rid) is not int
                    or rid < 0
                    or rid in all_ids
                    or ticket["product"] not in c.product_index
                    or type(ticket["native_period"]) is not int
                    or not 0 <= ticket["native_period"] < T
                    or type(ticket["active_reservation"]) is not bool
                    or ticket["active_reservation"] != (ticket["cancellation"] is None)
                ):
                    raise ValueError(
                        "complete distinct active/cancelled ticket support required"
                    )
                j = c.product_index[ticket["product"]]
                if (
                    int(c.product_market[j]) != key[2]
                    or ticket["carrier"]
                    != c.spec.carriers[int(c.product_carrier[j])].code
                ):
                    raise ValueError(
                        "native ticket product/carrier/request market differs"
                    )
                ids.append(rid)
                all_ids.add(rid)
            active = [ticket for ticket in tickets if ticket["active_reservation"]]
            current = account["current_active_record"]
            if (
                root not in ids
                or len(active) > 1
                or (current is not None and type(current) is not int)
                or current != (active[0]["record_id"] if active else None)
            ):
                raise ValueError(
                    "complete current original/replacement ticket partition required"
                )
            expected_state = (
                "active_original"
                if current == root
                else "active_replacement"
                if current is not None
                else "no_active_reservation"
            )
            if account["current_reservation_state"] != expected_state:
                raise ValueError(
                    "current reservation label and ticket partition differ"
                )
            ticket_rows += len(tickets)
            if active:
                product = active[0]["product"]
                legs = tuple(c.spec.products[c.product_index[product]].legs)
                current_legs += len(legs)
        people.append(
            {
                "person_key": key,
                "original_record": root,
                "current_ticket": current,
                "current_product": product,
                "required_legs": legs,
                "booking_state": "ordinary_outside"
                if account is None
                else account["current_reservation_state"],
            }
        )
    if (
        indices != set(range(len(people)))
        or tuple(by_period) != tuple(panel["period_request_counts"])
        or outside != panel["ordinary_outside"]
        or len(roots) != panel["originally_booked"]
    ):
        raise ValueError("ALL original rows/periods/outside/booked support required")
    rows = len(people) + ticket_rows + current_legs + len(dep) + len(clock.boundaries)
    if rows > context_row_limit:
        raise RuntimeError(
            "ALL person/ticket/leg/clock rows exceed limit; no partial/fallback/retry"
        )
    return ServiceObservationJournal(
        {
            "plan_id": "A.10/A.5/B0/V9",
            "definition": definition,
            "observation_scope": observation_scope,
            "spec_hash": c.spec_hash,
            "complete_native_model": jsonable(c.spec),
            "complete_ordinary_panel": jsonable(panel),
            "booking_clock": tuple(v.isoformat() for v in clock.boundaries),
            "booking_closed_at": clock.at(T).isoformat(),
            "leg_departures": tuple(
                (leg.id, departures[leg.id].isoformat()) for leg in c.spec.legs
            ),
            "persons": tuple(people),
            "all_context_rows": rows,
            "context_row_limit": context_row_limit,
            "observation_limit": observation_limit,
            "native_departure_engine_or_actor_executed": False,
            "observation_source_public_empirical_or_service_truth_authenticated": False,
            "primary_lifecycle_welfare_or_planner_adopted": False,
            "scientific_budget": None,
        },
        (),
    )


def record_leg_service(journal, observation):
    """Append ONE NEW observation functionally; previous journal stays owned.

    Known outcomes cannot be retracted/repaired on this path. Later unknown
    source reports remain visible and never erase an earlier known outcome.
    """
    if (
        type(journal) is not ServiceObservationJournal
        or type(observation) is not LegServiceObservation
    ):
        raise TypeError("typed journal and prospective leg observation required")
    o = observation
    if len(journal.observations) >= journal.context["observation_limit"]:
        raise RuntimeError(
            "ALL observation support exceeds limit; no partial/fallback/retry"
        )
    if (
        not isinstance(o.event_id, str)
        or not o.event_id.strip()
        or not isinstance(o.source_ref, str)
        or not o.source_ref.strip()
        or o.outcome not in {"delivered", "not_delivered", "unknown"}
        or type(o.ticket_record) is not int
        or o.ticket_record < 0
        or len(o.person_key) != 3
        or any(type(v) is not int or v < 0 for v in o.person_key)
        or any(o.event_id == old.event_id for old in journal.observations)
    ):
        raise ValueError(
            "distinct event ID/explicit outcome/source/current ticket required"
        )
    person = next(
        (p for p in journal.context["persons"] if p["person_key"] == o.person_key), None
    )
    if (
        person is None
        or person["current_ticket"] != o.ticket_record
        or o.leg not in person["required_legs"]
    ):
        raise ValueError(
            "matching original person/CURRENT native ticket/itinerary leg required"
        )
    departure = utc(dict(journal.context["leg_departures"])[o.leg])
    if (
        o.occurred_at < utc(journal.context["booking_closed_at"])
        or o.recorded_at < o.occurred_at
        or (
            journal.observations
            and o.recorded_at < journal.observations[-1].recorded_at
        )
        or (o.outcome == "delivered" and o.occurred_at < departure)
    ):
        raise ValueError(
            "post-booking occurrence/causal recording/ordered information times required"
        )
    same = [
        old
        for old in journal.observations
        if old.person_key == o.person_key and old.leg == o.leg
    ]
    known = [old for old in same if old.outcome != "unknown"]
    if o.outcome != "unknown" and known:
        raise ValueError(
            "known leg outcome already recorded; no duplicate/correction/repair"
        )
    delivered = {
        old.leg: old.occurred_at
        for old in journal.observations
        if old.person_key == o.person_key and old.outcome == "delivered"
    }
    if o.outcome == "delivered":
        delivered[o.leg] = o.occurred_at
    times = [delivered[leg] for leg in person["required_legs"] if leg in delivered]
    if any(a > b for a, b in pairwise(times)):
        raise ValueError(
            "observed delivered itinerary legs must respect declared product order"
        )
    return ServiceObservationJournal(journal.context, (*journal.observations, o))


def service_observation_summary(journal, *, as_of):
    """Keep every person and missing outcome; no saved-loader or producer call."""
    if type(journal) is not ServiceObservationJournal:
        raise TypeError("ServiceObservationJournal required")
    at = utc(as_of)
    if at < utc(journal.context["booking_closed_at"]) or (
        journal.observations and at < journal.observations[-1].recorded_at
    ):
        raise ValueError(
            "as-of must include the full supplied information history; no future filtering"
        )
    departures = dict(journal.context["leg_departures"])
    people = []
    for p in journal.context["persons"]:
        legs = []
        for leg in p["required_legs"]:
            history = tuple(
                o
                for o in journal.observations
                if o.person_key == p["person_key"] and o.leg == leg
            )
            known = [o for o in history if o.outcome != "unknown"]
            outcome = known[0].outcome if known else "unknown"
            legs.append(
                {
                    "leg": leg,
                    "departure_due": utc(departures[leg]) <= at,
                    "observed_outcome": outcome,
                    "observations": tuple(_observation_record(o) for o in history),
                }
            )
        outcomes = tuple(leg["observed_outcome"] for leg in legs)
        complete = (
            (
                True
                if all(o == "delivered" for o in outcomes)
                else False
                if "not_delivered" in outcomes
                else None
            )
            if legs
            else None
        )
        people.append(
            {
                **dict(p),
                "leg_outcomes": tuple(legs),
                "observed_current_itinerary_complete": complete,
                "unobserved_current_legs": sum(o == "unknown" for o in outcomes),
                "lifecycle_consumer_surplus": None,
                "causal_denial_or_misallocation": None,
            }
        )
    return deep_freeze(
        {
            "plan_id": "A.10/A.5/B0/V9",
            "as_of": at.isoformat(),
            "persons": tuple(people),
            "person_rows": len(people),
            "received_observations": len(journal.observations),
            "observed_complete_current_itineraries": sum(
                p["observed_current_itinerary_complete"] is True for p in people
            ),
            "observed_incomplete_current_itineraries": sum(
                p["observed_current_itinerary_complete"] is False for p in people
            ),
            "unknown_current_itinerary_outcomes": sum(
                p["observed_current_itinerary_complete"] is None for p in people
            ),
            "population_service_rate": None,
            "total_lifecycle_surplus": None,
            "primary_B4_metric_or_native_service_truth_adopted": False,
            "scientific_budget": None,
        }
    )


def _observation_record(o):
    return {
        "event_id": o.event_id,
        "person_key": o.person_key,
        "ticket_record": o.ticket_record,
        "leg": o.leg,
        "outcome": o.outcome,
        "occurred_at": o.occurred_at.isoformat(),
        "recorded_at": o.recorded_at.isoformat(),
        "source_ref": o.source_ref,
    }


def service_journal_record(journal):
    if type(journal) is not ServiceObservationJournal:
        raise TypeError("ServiceObservationJournal required")
    return _encode(
        jsonable(
            {
                "context": journal.context,
                "observations": tuple(
                    _observation_record(o) for o in journal.observations
                ),
            }
        )
    )
