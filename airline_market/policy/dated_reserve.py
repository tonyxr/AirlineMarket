"""A.14/B0/V9: per-leg reserve release exactly 24 elapsed hours before departure.

Opt-in public UTC clock for a pre-departure booking cohort. No inferred mapping
from native periods to days, no departed-flight lifecycle or planner adoption.
"""

import copy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import ROUND_FLOOR, Decimal

from airline_market.policy.reserve import ReserveMarketEngine, SeatReserve


def utc(value):
    """Normalize an explicitly zoned instant; never infer a local timezone."""
    if isinstance(value, str):
        value = value.strip()
        value = datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value
        )
    if not isinstance(value, datetime):
        raise TypeError("explicit aware datetime or ISO instant required")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timezone-aware instant required")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class UtcBookingClock:
    boundaries: tuple

    def __post_init__(self):
        times = tuple(utc(x) for x in self.boundaries)
        if len(times) < 2 or any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError("strictly increasing public UTC boundaries required")
        object.__setattr__(self, "boundaries", times)

    def at(self, period):
        if type(period) is not int or not 0 <= period < len(self.boundaries):
            raise ValueError("period outside declared public clock")
        return self.boundaries[period]

    def index(self, instant):
        try:
            return self.boundaries.index(utc(instant))
        except ValueError as error:
            raise ValueError("event must be an exact declared UTC boundary") from error


@dataclass(frozen=True)
class DatedSeatReserve:
    reveal_utc: object
    share: Decimal
    affected_legs: tuple[str, ...]
    leg_departures: tuple[tuple[str, object], ...]

    def __post_init__(self):
        # Reuse the existing declared 0/2/5/10% share and distinct-leg contract.
        p = SeatReserve(0, 1, self.share, self.affected_legs)
        dep = tuple(sorted((k, utc(v)) for k, v in self.leg_departures))
        if len({k for k, v in dep}) != len(dep) or any(
            not isinstance(k, str) or not k for k, v in dep
        ):
            raise ValueError("distinct public leg departure IDs required")
        object.__setattr__(self, "reveal_utc", utc(self.reveal_utc))
        object.__setattr__(self, "share", p.share)
        object.__setattr__(self, "affected_legs", p.affected_legs)
        object.__setattr__(self, "leg_departures", dep)

    def release_at(self, leg):
        return dict(self.leg_departures)[leg] - timedelta(hours=24)


@dataclass(frozen=True)
class DatedReserveSnapshot:
    native: object
    clock: object
    policy: object
    released_legs: frozenset


class DatedReserveMarketEngine(ReserveMarketEngine):
    """Reuse native held-first sales; only the per-leg public release clock is new.

    Every leg departs after this booking horizon. In-horizon reveal/release events
    must be explicit clock boundaries. Late reveals hold zero; future cutoffs stay
    held at terminal. No continuously arriving demand or hour-based WTP is added.
    """

    def __init__(
        self,
        compiled,
        controllers=None,
        *,
        booking_clock=None,
        dated_reserve=None,
        **kwargs,
    ):
        if "seat_reserve" in kwargs:
            raise ValueError("dated engine derives its native admission interval")
        if (booking_clock is None) != (dated_reserve is None):
            raise ValueError("explicit clock and dated policy required together")
        if booking_clock is not None and not isinstance(booking_clock, UtcBookingClock):
            raise TypeError("UtcBookingClock required")
        if dated_reserve is not None and not isinstance(
            dated_reserve, DatedSeatReserve
        ):
            raise TypeError("DatedSeatReserve required")
        self.booking_clock = booking_clock
        self.dated_reserve = dated_reserve
        native = None
        if dated_reserve is not None:
            c, p = booking_clock, dated_reserve
            T = compiled.spec.horizon.n_periods
            if len(c.boundaries) != T + 1:
                raise ValueError(
                    "one UTC boundary per native period plus terminal required"
                )
            dep = dict(p.leg_departures)
            if set(dep) != set(compiled.leg_index) or set(p.affected_legs) - set(dep):
                raise ValueError(
                    "explicit departure instant for every compiled leg required"
                )
            if c.at(T) >= min(dep.values()):
                raise ValueError(
                    "pre-departure cohort must end before every flight departure"
                )
            if p.reveal_utc < c.at(0):
                raise ValueError(
                    "reveal before cohort cannot identify reveal-time remaining seats"
                )
            if p.reveal_utc < c.at(T):
                reveal = c.index(p.reveal_utc)
                active = []
                release_indices = []
                for leg in p.affected_legs:
                    due = p.release_at(leg)
                    if due <= p.reveal_utc:
                        continue
                    active.append(leg)
                    release_indices.append(c.index(due) if due <= c.at(T) else T)
                if active:
                    # This parent interval validates menus and guards forced-choice
                    # hooks only. Actual release is ALWAYS the per-leg UTC cutoff.
                    native = SeatReserve(
                        reveal, max(release_indices), p.share, tuple(active)
                    )
        super().__init__(compiled, controllers, seat_reserve=native, **kwargs)

    def _clear_reserve(self):
        super()._clear_reserve()
        self._dated_released = set()

    def _sync_reserve(self):
        p = self.dated_reserve
        if p is None:
            return super()._sync_reserve()
        if self.world is None:
            return
        now = self.booking_clock.at(self.world.t)
        stock = self._remaining()
        if (
            not self._reserve_activated
            and self.world.t < self.compiled.spec.horizon.n_periods
            and now >= p.reveal_utc
        ):
            self._reserve_activated = True
            for leg in p.affected_legs:
                due = p.release_at(leg)
                late = now >= due
                n = (
                    0
                    if late
                    else int(
                        (p.share * stock[leg]).to_integral_value(rounding=ROUND_FLOOR)
                    )
                )
                self._reserve_ledger[leg] = dict(
                    initial=n, held=n, used=0, released=0, capacity_lost=0
                )
                self._event(
                    leg,
                    "hold_skipped_after_release" if late else "hold",
                    n,
                    reveal_remaining=stock[leg],
                    rounding="floor",
                    observed_utc=now.isoformat(),
                    release_due_utc=due.isoformat(),
                    departure_utc=dict(p.leg_departures)[leg].isoformat(),
                )
                if late:
                    self._dated_released.add(leg)
        for leg, row in self._reserve_ledger.items():
            lost = max(0, row["held"] - stock[leg])
            if lost:
                row["held"] -= lost
                row["capacity_lost"] += lost
                self._event(leg, "capacity_loss", lost, observed_utc=now.isoformat())
            if leg not in self._dated_released and now >= p.release_at(leg):
                n = row["held"]
                row["held"] = 0
                row["released"] += n
                self._dated_released.add(leg)
                self._event(
                    leg,
                    "release",
                    n,
                    observed_utc=now.isoformat(),
                    release_due_utc=p.release_at(leg).isoformat(),
                )
        self._reserve_released = self._reserve_activated and len(
            self._dated_released
        ) == len(p.affected_legs)

    def snapshot(self):
        return DatedReserveSnapshot(
            super().snapshot(),
            self.booking_clock,
            self.dated_reserve,
            frozenset(self._dated_released),
        )

    def restore(self, snapshot):
        if not isinstance(snapshot, DatedReserveSnapshot):
            raise TypeError("dated reserve snapshot required")
        if (
            snapshot.clock != self.booking_clock
            or snapshot.policy != self.dated_reserve
        ):
            raise ValueError("snapshot public clock/policy differs")
        super().restore(snapshot.native)
        self._dated_released = set(snapshot.released_legs)

    def branch(self):
        other = DatedReserveMarketEngine(
            self.compiled,
            {i: copy.deepcopy(ctrl) for i, ctrl in self.controllers.items()},
            booking_clock=self.booking_clock,
            dated_reserve=self.dated_reserve,
            rescue_menu=self.rescue_menu,
            rebooking=self.rebooking,
            exits=self.exits,
            cap_policy=self.cap_policy,
            check_invariants=self.check_invariants,
        )
        other.restore(self.snapshot())
        return other
