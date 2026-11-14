"""A.5/A.8/B0/V9: opt-in original-potential request lineage for native re-entry.

Native Booking.arrival_rank remains the original thinned admission rank. A
sidecar captures potential slot BEFORE native end-of-period cancellations. Old
re-entry marks are delegated once with a private immutable coordinate view;
no old ledger, arrival, physical engine or published result is changed.
"""

import copy
from dataclasses import dataclass, replace

from airline_market.policy.rationed_native import RationedNativeMarketEngine
from airline_market.sim.arrivals import period_requests

RULE = "original-potential-slot-v1"


@dataclass(frozen=True)
class PhysicalRequestCoordinate:
    period: int
    potential_slot: int
    market: int
    native_rank: int
    segment: int
    taste_class: int

    @property
    def stream_key(self):
        return (self.period, self.potential_slot, self.market)


@dataclass(frozen=True)
class PotentialLineageSnapshot:
    native: object
    coordinates: tuple
    coordinate_rule: str = RULE


class PotentialLineageMarketEngine(RationedNativeMarketEngine):
    """Record every genuine ordinary booking; re-entry follows its original slot.

    New cohorts only: no reconstruction/fallback for unknown pre-existing tickets.
    Forced request/choice hooks are refused. Sidecars are analyst-only; native
    request ranks, Booking fields, controller observations/callbacks stay native.
    """

    def __init__(self, *args, **kwargs):
        self._physical_coordinates = {}
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, *, tape=None):
        self._physical_coordinates = {}
        return super().reset(seed, tape=tape)

    def _track_requests(self, requests):
        for request in requests:
            period = self.world.t
            record = self.world.next_record_id
            logged = len(self.world.request_log)
            yield request
            # Parent resumes this iterator before its background/events/callback
            # closure. Any first cancellation therefore sees the sidecar already.
            if len(self.world.request_log) != logged + 1:
                raise RuntimeError("native ordinary request log correspondence changed")
            row = self.world.request_log[-1]
            if row["period"] != period or row["market"] != request.market:
                raise RuntimeError("native ordinary request identity changed")
            delta = self.world.next_record_id - record
            if row["chosen"] is None:
                if delta:
                    raise RuntimeError("outside request created a native ticket")
                continue
            if delta != 1:
                raise RuntimeError(
                    "one physical ordinary request must create one ticket"
                )
            carrier = int(self.compiled.product_carrier[row["chosen"]])
            booking = next(
                (
                    b
                    for b in self.world.carriers[carrier].bookings
                    if b.record_id == record
                ),
                None,
            )
            if booking is None or (
                booking.period,
                booking.arrival_rank,
                booking.segment,
                booking.taste_class,
                booking.product,
            ) != (
                period,
                request.rank,
                request.segment,
                request.taste_class,
                row["chosen"],
            ):
                raise RuntimeError("native ticket does not match ordinary request")
            self._physical_coordinates[record] = PhysicalRequestCoordinate(
                period,
                request.gumbel_row,
                request.market,
                request.rank,
                request.segment,
                request.taste_class,
            )

    def _native_current_requests(self):
        yield from period_requests(
            self.compiled, self.tape, self.world.t, self.world.exo
        )

    def _ordered_ordinary_requests(self, pending):
        yield from self._track_requests(super()._ordered_ordinary_requests(pending))

    def step(self, actions=None, *, _requests=None, _choices=None):
        if _requests is not None or _choices is not None:
            raise ValueError(
                "potential lineage requires genuine native requests and choices"
            )
        if self.ordinary_priority is None:
            return super().step(
                actions, _requests=self._track_requests(self._native_current_requests())
            )
        return super().step(actions)

    def request_coordinate(self, record_id):
        root = self._replacement_roots.get(record_id, record_id)
        if root not in self._physical_coordinates:
            raise ValueError("original potential coordinate absent; no rank fallback")
        return self._physical_coordinates[root]

    def _marks(self, passenger, period):
        b = passenger.original
        coordinate = self.request_coordinate(b.record_id)
        if (
            b.period,
            b.arrival_rank,
            b.segment,
            b.taste_class,
            int(self.compiled.product_market[b.product]),
        ) != (
            coordinate.period,
            coordinate.native_rank,
            coordinate.segment,
            coordinate.taste_class,
            coordinate.market,
        ):
            raise ValueError("original native booking differs from physical coordinate")
        # The frozen native method reads only this temporary immutable view.
        # Same namespace/distribution as native, potential slot replaces rank in
        # the key. Actual Booking/passenger/ledger/history objects are untouched.
        view = replace(
            passenger, original=replace(b, arrival_rank=coordinate.potential_slot)
        )
        return super()._marks(view, period)

    def _reenter(self, prices):
        start = len(self.rebooking_history)
        purchases = super()._reenter(prices)
        for row in self.rebooking_history[start:]:
            coordinate = self.request_coordinate(row["original_record"])
            row.update(
                original_potential_request=coordinate.stream_key,
                native_original_rank=coordinate.native_rank,
                coordinate_rule=RULE,
            )
        return purchases

    def coordinate_records(self):
        return tuple(sorted(self._physical_coordinates.items()))

    def snapshot(self):
        return PotentialLineageSnapshot(super().snapshot(), self.coordinate_records())

    def restore(self, snapshot):
        if type(snapshot) is not PotentialLineageSnapshot:
            raise TypeError("potential-lineage combined snapshot required")
        if snapshot.coordinate_rule != RULE:
            raise ValueError("snapshot potential-coordinate rule differs")
        super().restore(snapshot.native)
        self._physical_coordinates = dict(snapshot.coordinates)

    def branch(self):
        other = PotentialLineageMarketEngine(
            self.compiled,
            {i: copy.deepcopy(ctrl) for i, ctrl in self.controllers.items()},
            ordinary_priority=self.ordinary_priority,
            trigger_rule=self.trigger.rule if self.trigger else None,
            mechanical_benchmark=self.trigger.benchmark if self.trigger else None,
            booking_clock=self.booking_clock,
            leg_departures=self.leg_departures,
            dated_rebooking=self.dated_rebooking,
            dated_menu=self.dated_menu,
            dated_reserve=self.dated_reserve,
            exits=self.exits,
            cap_policy=self.cap_policy,
            check_invariants=self.check_invariants,
        )
        other.restore(self.snapshot())
        return other


def coordinate_record(engine):
    """Lossless analyst-only coordinates; neither controller data nor PNR proof."""
    return {
        "plan_id": "A.5/A.8/B0/V9",
        "coordinate_rule": RULE,
        "ordinary_tickets": tuple(
            {"native_record_id": k, **vars(v), "stream_key": v.stream_key}
            for k, v in engine.coordinate_records()
        ),
        "replacement_roots": tuple(sorted(engine._replacement_roots.items())),
        "old_receipts_rewritten": False,
        "empirical_identity_authenticated": False,
    }
