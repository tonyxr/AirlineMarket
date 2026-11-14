"""A.10/A.5/B0/V9: prospective original-person closed-boundary reservations.

Point observations only; native period counts are not elapsed waiting or travel.
"""

from collections import Counter
from dataclasses import dataclass

from airline_market.core.hashing import jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.sim.ordinary_cohort import (
    OrdinaryCohortMarketEngine,
    OrdinaryCohortOrderEngine,
)

RULE = "original-person-closed-boundary-reservation-v1"
STATES = (
    "ordinary_outside",
    "active_original",
    "active_replacement",
    "no_active_reservation",
)


@dataclass(frozen=True)
class ReservationObservation:
    key: tuple
    segment: str
    taste_class: int
    request_market: str
    native_original_rank: int
    original_record: int | None
    active_record: int | None
    active_product: str | None
    active_carrier: str | None
    reservation_state: str

    def __post_init__(self):
        object.__setattr__(self, "key", tuple(self.key))
        if any(
            type(v) is not int or v < 0
            for v in (self.taste_class, self.native_original_rank)
        ) or any(
            not isinstance(v, str) or not v for v in (self.segment, self.request_market)
        ):
            raise ValueError(
                "immutable native class/rank/segment/market identifiers required"
            )


@dataclass(frozen=True)
class ReservationBoundary:
    closed_period: int
    native_world_t: int
    rows: tuple

    def __post_init__(self):
        object.__setattr__(self, "rows", tuple(self.rows))


@dataclass(frozen=True)
class ReservationCensusSnapshot:
    native: object
    boundaries: tuple
    pending: int | None
    capture_rule: str = RULE


class _ReservationCensus:
    """Observe the parent ledger only AFTER one complete successful native step."""

    def __init__(self, *args, **kwargs):
        self._reservation_boundaries = []
        self._pending_reservation = None
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, *, tape=None):
        self._reservation_boundaries = []
        self._pending_reservation = None
        return super().reset(seed, tape=tape)

    def _capture_reservation_boundary(self, period):
        people = tuple(row for p in self.ordinary_periods() for row in p.rows)
        roots = {
            row.original_record: row
            for row in people
            if row.original_record is not None
        }
        if len(roots) != sum(row.original_record is not None for row in people):
            raise RuntimeError("one original request per booked root required")
        active = {}
        for carrier in self.world.carriers:
            for ticket in carrier.bookings:
                root = self._replacement_roots.get(ticket.record_id, ticket.record_id)
                if root not in roots or root in active:
                    raise RuntimeError(
                        "one current native reservation per known original root required"
                    )
                active[root] = ticket
        rows = []
        for person in sorted(people, key=lambda r: r.key):
            root = person.original_record
            ticket = active.get(root)
            if (
                root is not None
                and self._physical_coordinates[root].stream_key != person.key
            ):
                raise RuntimeError(
                    "native root and original potential coordinate differ"
                )
            rows.append(
                ReservationObservation(
                    person.key,
                    person.segment,
                    person.taste_class,
                    person.request_market,
                    person.native_rank,
                    root,
                    None if ticket is None else ticket.record_id,
                    None
                    if ticket is None
                    else self.compiled.spec.products[ticket.product].id,
                    None
                    if ticket is None
                    else self.compiled.spec.carriers[ticket.carrier].code,
                    "ordinary_outside"
                    if root is None
                    else "no_active_reservation"
                    if ticket is None
                    else "active_original"
                    if ticket.record_id == root
                    else "active_replacement",
                )
            )
        if self.world.t != period + 1 or len({r.key for r in rows}) != len(rows):
            raise RuntimeError("complete original-person closing boundary required")
        return ReservationBoundary(period, self.world.t, tuple(rows))

    def step(self, actions=None, *, _requests=None, _choices=None):
        if _requests is not None or _choices is not None:
            raise ValueError(
                "reservation census requires genuine parent requests/choices"
            )
        if self._pending_reservation is not None:
            raise ValueError(
                "unfinished reservation capture; no continuation as a successful census"
            )
        if self.world is None or self.terminal():
            raise ValueError("initialized nonterminal native opportunity required")
        period = self.world.t
        self._pending_reservation = period
        result = super().step(actions)
        boundary = self._capture_reservation_boundary(period)
        self._reservation_boundaries.append(boundary)
        self._pending_reservation = None
        return result

    def reservation_boundaries(self):
        if self._pending_reservation is not None:
            raise ValueError(
                "incomplete reservation capture; no partial positive census"
            )
        if self.world is None or len(self._reservation_boundaries) != self.world.t:
            raise ValueError("complete current native boundary history required")
        return tuple(self._reservation_boundaries)

    def snapshot(self):
        return ReservationCensusSnapshot(
            super().snapshot(),
            tuple(self._reservation_boundaries),
            self._pending_reservation,
        )

    def restore(self, snapshot):
        if type(snapshot) is not ReservationCensusSnapshot:
            raise TypeError("combined reservation census snapshot required")
        if snapshot.capture_rule != RULE:
            raise ValueError("snapshot reservation capture rule differs")
        super().restore(snapshot.native)
        self._reservation_boundaries = list(snapshot.boundaries)
        self._pending_reservation = snapshot.pending


class ReservationCensusMarketEngine(_ReservationCensus, OrdinaryCohortMarketEngine):
    """Opt-in chronological/random native engine with closing point observations."""


class ReservationCensusOrderEngine(_ReservationCensus, OrdinaryCohortOrderEngine):
    """Opt-in private native-order benchmark with identical closing observations."""


def reservation_census_summary(boundaries):
    """Consume NEW complete point receipts; no engine/account/timing replay.

    All originally arrived people persist at each later observed boundary. Counts
    are observations, not independent people or elapsed-time integrals. No old
    closing state is reconstructed from final ticket/cancellation period numbers.
    """
    boundaries = tuple(boundaries)
    people = {}
    ticket_roots = {}
    previous = set()
    counts = []
    for index, boundary in enumerate(boundaries):
        if type(boundary) is not ReservationBoundary or (
            boundary.closed_period,
            boundary.native_world_t,
        ) != (index, index + 1):
            raise ValueError("complete ordered native closing boundaries required")
        current = set()
        for row in boundary.rows:
            if (
                type(row) is not ReservationObservation
                or len(row.key) != 3
                or any(type(k) is not int or k < 0 for k in row.key)
                or not 0 <= row.key[0] <= index
                or row.key in current
            ):
                raise ValueError("distinct arrived original physical keys required")
            current.add(row.key)
            if row.reservation_state not in STATES:
                raise ValueError("known reservation state required")
            if (row.original_record is None) != (
                row.reservation_state == "ordinary_outside"
            ):
                raise ValueError("outside observations cannot invent a booked root")
            if any(
                type(k) is not int or k < 0
                for k in (row.original_record, row.active_record)
                if k is not None
            ):
                raise ValueError("nonnegative native ticket IDs required")
            active_fields = (row.active_product, row.active_carrier)
            if (row.active_record is not None) != row.reservation_state.startswith(
                "active_"
            ) or (
                not all(v is None for v in active_fields)
                if row.active_record is None
                else not all(isinstance(v, str) and v for v in active_fields)
            ):
                raise ValueError(
                    "native reservation state/ticket/product/carrier disagree"
                )
            if row.active_record is not None:
                if (
                    row.active_record in ticket_roots
                    and ticket_roots[row.active_record] != row.original_record
                ):
                    raise ValueError(
                        "native ticket cannot transfer to a different person"
                    )
                ticket_roots[row.active_record] = row.original_record
            if (
                row.reservation_state == "active_original"
                and row.active_record != row.original_record
                or (
                    row.reservation_state == "active_replacement"
                    and row.active_record <= row.original_record
                )
            ):
                raise ValueError("original/replacement reservation kind differs")
            identity = (
                row.segment,
                row.taste_class,
                row.request_market,
                row.native_original_rank,
                row.original_record,
            )
            if row.key not in people:
                if row.key[0] != index:
                    raise ValueError("original arrival missing from earlier boundary")
                people[row.key] = (identity, [])
            elif people[row.key][0] != identity:
                raise ValueError("original person identity changes across boundaries")
            people[row.key][1].append((index + 1, row))
        if not previous <= current:
            raise ValueError(
                "all previously arrived people must remain, including outside/unreserved"
            )
        roots = [
            r.original_record for r in boundary.rows if r.original_record is not None
        ]
        active_ids = [
            r.active_record for r in boundary.rows if r.active_record is not None
        ]
        if len(set(roots)) != len(roots) or len(set(active_ids)) != len(active_ids):
            raise ValueError("distinct original roots/current native tickets required")
        previous = current
        c = Counter(row.reservation_state for row in boundary.rows)
        counts.append(
            {
                "native_world_t": index + 1,
                "arrived_original_people": len(current),
                "states": tuple((state, c[state]) for state in STATES),
            }
        )
    rows = []
    for key, (identity, history) in sorted(people.items()):
        if len(history) != len(boundaries) - key[0]:
            raise ValueError("full observed follow-up support required")
        replacements = [
            t for t, r in history if r.reservation_state == "active_replacement"
        ]
        rows.append(
            {
                "potential_coordinate": key,
                "identity": identity,
                "observations": tuple(history),
                "observed_closing_boundaries": len(history),
                "active_reservation_observations": sum(
                    r.active_record is not None for t, r in history
                ),
                "unreserved_booked_person_observations": sum(
                    r.reservation_state == "no_active_reservation" for t, r in history
                ),
                "outside_observations": sum(
                    r.reservation_state == "ordinary_outside" for t, r in history
                ),
                "first_observed_active_replacement_boundary": min(replacements)
                if replacements
                else None,
                "first_actual_replacement_booking_time": None,
                "elapsed_waiting_hours": None,
                "delivered_travel": None,
                "lifecycle_consumer_surplus": None,
                "followup_after_prefix_unknown": True,
            }
        )
    return deep_freeze(
        {
            "plan_id": "A.10/A.5/B0/V9",
            "capture_rule": RULE,
            "completed_native_periods": len(boundaries),
            "unique_original_people": len(people),
            "boundaries": boundaries,
            "by_boundary": tuple(counts),
            "persons": tuple(rows),
            "boundary_observations_are_additional_people": False,
            "within_period_reservation_duration_or_continuous_time_inferred": False,
            "same_producer_bank_or_original_capture_authenticated": False,
            "terminal_booking_horizon_is_delivered_lifecycle": False,
            "elapsed_waiting_hours": None,
            "total_lifecycle_surplus": None,
            "denied_requests": None,
            "misallocation": None,
            "primary_default_empirical_production_adopted": False,
            "scientific_budget": None,
        }
    )


def reservation_census_record(result):
    """Owned immutable receipt/summary export, no old-prefix loader or replay."""
    return jsonable(result)
