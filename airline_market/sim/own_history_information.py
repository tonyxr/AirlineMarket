"""A.10/A.5/B0/V9: prospective native own history BEFORE re-entry marks.

Native facts only, not authenticated disclosure, full passenger knowledge,
current stage menus, perceived probabilities, utility or changed observations.
"""

import hashlib
import json
from dataclasses import dataclass

from airline_market.core.hashing import jsonable
from airline_market.sim.ordinary_information import (
    OrdinaryInformationMarketEngine,
    ordinary_information_view,
)
from airline_market.sim.passenger_accounts import _encode

RULE = "prospective-native-own-history-before-reentry-marks-v1"


@dataclass(frozen=True)
class OwnNativeFact:
    ordinal: int
    kind: str
    native_period: int
    person_ticket: int | None
    product_id: str | None
    fare: float | None
    stage: str | None


@dataclass(frozen=True)
class OwnReentryView:
    native_period: int
    compiled_opportunity_utc: str
    original_market: str
    original_carrier: str
    first_stranded_period: int
    exclusive_deadline_period: int
    native_exact_expiry_utc: str
    interline_enabled: bool
    own_facts: tuple
    scope: str = "native-own-facts-before-current-reentry-marks-only"


@dataclass(frozen=True)
class OwnReentryRow:
    analyst_root: int
    analyst_original_key: tuple
    view: OwnReentryView


@dataclass(frozen=True)
class OwnInformationPeriod:
    period: int
    reentry_rows: tuple


@dataclass(frozen=True)
class OwnInformationSnapshot:
    native: object
    histories: tuple
    periods: tuple
    pending: object
    spec_hash: str
    rule: str = RULE


class OwnHistoryInformationMarketEngine(OrdinaryInformationMarketEngine):
    """ONE inherited native lifecycle with causal own-fact capture.

    Actual ordinary booking, cancellation callback and finished search records
    supply facts. Each genuine re-entry _marks hook reads earlier own facts,
    then delegates native marks ONCE without inspecting its returned values.
    """

    def __init__(self, *args, **kwargs):
        self._own_histories = {}
        self._own_tickets = {}
        self._own_keys = {}
        self._own_periods = []
        self._pending_own = None
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, *, tape=None):
        self._own_histories = {}
        self._own_tickets = {}
        self._own_keys = {}
        self._own_periods = []
        self._pending_own = None
        return super().reset(seed, tape=tape)

    def _fact(self, root, kind, period, ticket, product, fare, stage=None):
        history = self._own_histories[root]
        history.append(
            OwnNativeFact(
                len(history),
                kind,
                period,
                ticket,
                None if product is None else self.compiled.spec.products[product].id,
                None if fare is None else float(fare),
                stage,
            )
        )

    def _track_requests(self, requests):
        for request in super()._track_requests(requests):
            period, record = self.world.t, self.world.next_record_id
            key = (period, request.gumbel_row, request.market)
            yield request
            if self.world.next_record_id == record + 1:
                found = [
                    b
                    for cs in self.world.carriers
                    for b in cs.bookings
                    if b.record_id == record
                ]
                if len(found) != 1 or record in self._own_histories:
                    raise RuntimeError("one genuine new original booking required")
                b = found[0]
                self._own_histories[record] = []
                self._own_tickets[record] = [record]
                self._own_keys[record] = key
                self._fact(
                    record, "native_ordinary_purchase", period, 0, b.product, b.fare
                )

    def _record_stranded(self, bookings, period, cause):
        bookings = tuple(bookings)
        super()._record_stranded(bookings, period, cause)
        for b in bookings:
            root = self._replacement_roots.get(b.record_id, b.record_id)
            if (
                root not in self._own_tickets
                or b.record_id not in self._own_tickets[root]
            ):
                raise ValueError(
                    "prospective own ticket history absent; no reconstruction"
                )
            self._fact(
                root,
                "native_ticket_cancelled",
                period,
                self._own_tickets[root].index(b.record_id),
                b.product,
                b.fare,
            )

    def _marks(self, passenger, period):
        root = passenger.original.record_id
        if self._pending_own is None or root not in self._own_histories:
            raise ValueError("genuine prospective own history required before marks")
        b = passenger.original
        view = OwnReentryView(
            period,
            self.demand_scenario.clock.at(period).isoformat(),
            self.compiled.spec.markets[int(self.compiled.product_market[b.product])].id,
            self.compiled.spec.carriers[b.carrier].code,
            passenger.first_stranded_at,
            passenger.deadline,
            self.reentry_deadline_utc(root).isoformat(),
            self.rebooking.interline,
            tuple(self._own_histories[root]),
        )
        self._pending_own.append(OwnReentryRow(root, self._own_keys[root], view))
        return super()._marks(passenger, period)

    def _reenter(self, prices):
        start = len(self.rebooking_history)
        result = super()._reenter(prices)
        for row in self.rebooking_history[start:]:
            root = row["original_record"]
            if root not in self._own_tickets:
                raise ValueError("prospective original root required; no history fill")
            ticket = row["replacement_record"]
            local = None
            if ticket is not None:
                if ticket in self._own_tickets[root]:
                    raise RuntimeError("new native replacement ticket required")
                local = len(self._own_tickets[root])
                self._own_tickets[root].append(ticket)
            self._fact(
                root,
                "native_reentry_purchase"
                if ticket is not None
                else "native_reentry_outside",
                row["period"],
                local,
                row["chosen"],
                row["fare"],
                row["stage"],
            )
        return result

    def step(self, actions=None, *, _requests=None, _choices=None):
        if _requests is not None or _choices is not None:
            raise ValueError("genuine native requests and choices required")
        if self._pending_own is not None:
            raise ValueError("incomplete own capture; no continuation or retry")
        if self.ordinary_priority is not None:
            raise ValueError("chronological own-information path required")
        if self.world is None or self.terminal():
            raise ValueError("initialized nonterminal opportunity required")
        for name in ("booking_clock", "shock_clock"):
            clock = getattr(self, name, None)
            if clock is not None and clock != self.demand_scenario.clock:
                raise ValueError("native and compiled own-information clocks differ")
        period, start = self.world.t, len(self.rebooking_history)
        self._pending_own = []
        result = super().step(actions)
        attempts = self.rebooking_history[start:]
        if len(attempts) != len(self._pending_own):
            raise RuntimeError("complete native attempts and own contexts differ")
        for attempt, context in zip(attempts, self._pending_own):
            if (attempt["original_record"], attempt["period"]) != (
                context.analyst_root,
                context.view.native_period,
            ):
                raise RuntimeError("native attempt and original own context differ")
        for root, key in self._own_keys.items():
            if self.request_coordinate(root).stream_key != key:
                raise RuntimeError("prospective original own coordinate differs")
        self._own_periods.append(OwnInformationPeriod(period, tuple(self._pending_own)))
        self._pending_own = None
        return result

    def own_information_periods(self):
        if self._pending_own is not None:
            raise ValueError("incomplete own capture; no partial query or salvage")
        return tuple(self._own_periods)

    def snapshot(self):
        return OwnInformationSnapshot(
            super().snapshot(),
            tuple(
                (
                    root,
                    tuple(facts),
                    tuple(self._own_tickets[root]),
                    self._own_keys[root],
                )
                for root, facts in sorted(self._own_histories.items())
            ),
            tuple(self._own_periods),
            None if self._pending_own is None else tuple(self._pending_own),
            self.compiled.spec_hash,
        )

    def restore(self, snapshot):
        if type(snapshot) is not OwnInformationSnapshot:
            raise TypeError("combined own-information snapshot required")
        if snapshot.rule != RULE or snapshot.spec_hash != self.compiled.spec_hash:
            raise ValueError("own-information capture contract differs")
        super().restore(snapshot.native)
        self._own_histories = {
            root: list(facts) for root, facts, _, _ in snapshot.histories
        }
        self._own_tickets = {
            root: list(tickets) for root, _, tickets, _ in snapshot.histories
        }
        self._own_keys = {root: key for root, _, _, key in snapshot.histories}
        self._own_periods = list(snapshot.periods)
        self._pending_own = None if snapshot.pending is None else list(snapshot.pending)


def own_reentry_information_view(row):
    """Owned own-fact projection; analyst root/key/global ticket IDs excluded."""
    if type(row) is not OwnReentryRow:
        raise TypeError("own-reentry information row required")
    return _encode(jsonable(row.view))


def own_reentry_information_key(row):
    """Key for this narrow native-history view, NOT full knowledge/source law."""
    payload = json.dumps(
        own_reentry_information_view(row),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def own_information_record(engine):
    """Read NEW complete own receipts only, no producer/account/timing replay."""
    if type(engine) is not OwnHistoryInformationMarketEngine:
        raise TypeError("prospective native own-information producer required")
    periods = engine.own_information_periods()
    return {
        "plan_id": "A.10/A.5/B0/V9",
        "capture_rule": RULE,
        "analyst_context": _encode(
            jsonable(
                {
                    "native_spec": engine.compiled.spec,
                    "spec_hash": engine.compiled.spec_hash,
                    "original_roots": tuple(
                        (
                            root,
                            engine._own_keys[root],
                            tuple(engine._own_tickets[root]),
                            tuple(facts),
                        )
                        for root, facts in sorted(engine._own_histories.items())
                    ),
                    "reentry_links": tuple(
                        tuple(
                            (row.analyst_root, row.analyst_original_key)
                            for row in p.reentry_rows
                        )
                        for p in periods
                    ),
                }
            )
        ),
        "ordinary_views_by_period": tuple(
            tuple(ordinary_information_view(row) for row in p.rows)
            for p in engine.information_periods()
        ),
        "reentry_views_by_period": tuple(
            {
                "period": p.period,
                "views": tuple(
                    own_reentry_information_view(row) for row in p.reentry_rows
                ),
                "view_keys": tuple(
                    own_reentry_information_key(row) for row in p.reentry_rows
                ),
            }
            for p in periods
        ),
        "native_own_history_before_reentry_marks_captured": True,
        "complete_passenger_knowledge_or_disclosure_authenticated": False,
        "current_reentry_stage_menu_prices_or_private_marks_exported": False,
        "subjective_true_probability_or_lifecycle_CS_identified": False,
        "native_actor_choice_or_default_changed": False,
        "scientific_budget": None,
    }
