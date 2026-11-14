"""A.5/A.14/B0/V9: explicit UTC rescue windows and re-entry deadlines.

Physical pre-departure execution only. Native booking/choice/refunds and named
request-coordinate draws remain unchanged. No legal PNR or planner adoption.
"""

import copy
from dataclasses import dataclass, replace
from datetime import timedelta

from airline_market.policy.dated_reserve import UtcBookingClock, utc
from airline_market.policy.rescue import RescueFareMenu, RescueMarketEngine
from airline_market.sim.rebooking import RebookingPolicy


@dataclass(frozen=True)
class UtcWindow:
    start: object
    stop: object

    def __post_init__(self):
        start, stop = utc(self.start), utc(self.stop)
        if start >= stop:
            raise ValueError("ordered UTC window required")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)

    def contains(self, instant):
        return self.start <= utc(instant) < self.stop


@dataclass(frozen=True)
class DatedRebookingPolicy:
    valid_hours: int
    wtp_multiplier: float
    utility_bonus: float = 0.0
    interline: bool = False

    def __post_init__(self):
        if type(self.valid_hours) is not int or self.valid_hours <= 0:
            raise ValueError("explicit positive integer re-entry hours required")
        RebookingPolicy(1, self.wtp_multiplier, self.utility_bonus, self.interline)


@dataclass(frozen=True)
class DatedRescueMenu:
    window: UtcWindow
    design: str
    distance_tiers: tuple = ()
    market_distances_km: tuple = ()
    route_fares: tuple = ()
    cancelled_departures: UtcWindow | None = None

    def __post_init__(self):
        if not isinstance(self.window, UtcWindow) or (
            self.cancelled_departures is not None
            and not isinstance(self.cancelled_departures, UtcWindow)
        ):
            raise TypeError("explicit UTC windows required")
        native = RescueFareMenu(
            0,
            1,
            self.design,
            self.distance_tiers,
            self.market_distances_km,
            self.route_fares,
        )
        for key in ["distance_tiers", "market_distances_km", "route_fares"]:
            object.__setattr__(self, key, getattr(native, key))

    def native(self, T):
        return RescueFareMenu(
            0,
            T,
            self.design,
            self.distance_tiers,
            self.market_distances_km,
            self.route_fares,
        )


@dataclass(frozen=True)
class DatedRescueSnapshot:
    native: object
    clock: object
    departures: tuple
    policy: object
    menu: object


class DatedRescueMarketEngine(RescueMarketEngine):
    """Only observed cancellation lineages re-enter, on declared UTC opportunities.

    Eligibility is queried at each native decision instant, so an expiry between
    instants does not create an interpolated attempt. Windows are [start, stop).
    The caller supplies all leg dates and a horizon before every departure.
    """

    def __init__(
        self,
        compiled,
        controllers=None,
        *,
        booking_clock=None,
        leg_departures=(),
        dated_rebooking=None,
        dated_menu=None,
        **kwargs,
    ):
        if "rebooking" in kwargs or "rescue_menu" in kwargs:
            raise ValueError("dated engine derives native windows from UTC inputs")
        self.booking_clock = booking_clock
        self.dated_rebooking = dated_rebooking
        self.dated_menu = dated_menu
        dep = tuple(sorted((k, utc(v)) for k, v in leg_departures))
        self.leg_departures = dep
        p = None
        menu = None
        T = compiled.spec.horizon.n_periods
        if dated_rebooking is None:
            if booking_clock is not None or dep or dated_menu is not None:
                raise ValueError("explicit dated re-entry policy required")
        else:
            if not isinstance(dated_rebooking, DatedRebookingPolicy) or not isinstance(
                booking_clock, UtcBookingClock
            ):
                raise TypeError("dated re-entry policy and public UTC clock required")
            if len(booking_clock.boundaries) != T + 1:
                raise ValueError(
                    "one clock boundary per native period plus terminal required"
                )
            if len(dict(dep)) != len(dep) or set(dict(dep)) != set(compiled.leg_index):
                raise ValueError(
                    "distinct public departure instant for every compiled leg required"
                )
            if booking_clock.at(T) >= min(v for k, v in dep):
                raise ValueError(
                    "pre-departure horizon must end before every departure"
                )
            p = RebookingPolicy(
                T,
                dated_rebooking.wtp_multiplier,
                dated_rebooking.utility_bonus,
                dated_rebooking.interline,
            )
            if dated_menu is not None:
                if not isinstance(dated_menu, DatedRescueMenu):
                    raise TypeError("DatedRescueMenu required")
                menu = dated_menu.native(T)
        super().__init__(compiled, controllers, rebooking=p, rescue_menu=menu, **kwargs)

    def reentry_deadline_utc(self, root):
        passenger = self._passengers[root]
        return self.booking_clock.at(passenger.first_stranded_at) + timedelta(
            hours=self.dated_rebooking.valid_hours
        )

    def _record_stranded(self, bookings, period, cause):
        bookings = tuple(bookings)
        super()._record_stranded(bookings, period, cause)
        if self.dated_rebooking is None:
            return
        T = self.compiled.spec.horizon.n_periods
        for root in {
            self._replacement_roots.get(b.record_id, b.record_id) for b in bookings
        }:
            passenger = self._passengers[root]
            due = self.reentry_deadline_utc(root)
            # Exclusive index of first ineligible opportunity. The exact UTC
            # expiry is retained separately; no opportunity is inserted/rounded.
            end = next(
                (
                    i
                    for i, instant in enumerate(self.booking_clock.boundaries[:-1])
                    if instant >= due
                ),
                T,
            )
            self._passengers[root] = replace(
                passenger, deadline=min(passenger.deadline, end)
            )

    def _rescue_prices(self, root, passenger, prices):
        effective, record = super()._rescue_prices(root, passenger, prices)
        menu = self.dated_menu
        now = self.booking_clock.at(self.world.t)
        departure = min(
            dict(self.leg_departures)[leg]
            for leg in self.compiled.spec.products[passenger.original.product].legs
        )
        open_now = menu.window.contains(now)
        original_allowed = (
            menu.cancelled_departures is None
            or menu.cancelled_departures.contains(departure)
        )
        if not (open_now and original_allowed):
            effective = prices
            record.update(rescue_eligible=False, rescue_option_fare=None)
        record.update(
            offer_utc=now.isoformat(),
            dated_menu_active=open_now,
            dated_cancelled_departure_eligible=original_allowed,
            original_first_departure_utc=departure.isoformat(),
        )
        return effective, record

    def _reenter(self, prices):
        start = len(self.rebooking_history)
        purchases = super()._reenter(prices)
        if self.dated_rebooking is not None:
            for r in self.rebooking_history[start:]:
                root = r["original_record"]
                passenger = self._passengers[root]
                r.update(
                    attempt_utc=self.booking_clock.at(r["period"]).isoformat(),
                    first_stranded_utc=self.booking_clock.at(
                        passenger.first_stranded_at
                    ).isoformat(),
                    reentry_deadline_utc=self.reentry_deadline_utc(root).isoformat(),
                )
        return purchases

    def timed_rebooking_metrics(self, hours=(24, 48, 72)):
        """Analyst-only first-booking counts, active replacements and censoring.

        Point first-booking rates are null if an unserved lineage lacks the stated
        follow-up. Bounds retain every stranded passenger in the denominator.
        First success is not a guarantee its replacement remains active.
        """
        if self.dated_rebooking is None:
            raise ValueError("dated re-entry policy required for timed metrics")
        if any(type(h) is not int or h <= 0 for h in hours) or len(set(hours)) != len(
            hours
        ):
            raise ValueError("distinct positive integer metric hours required")
        now = self.booking_clock.at(self.world.t)
        n = len(self._passengers)
        rows = []
        successes = {}
        for r in self.rebooking_history:
            if r["replacement_record"] is not None:
                successes.setdefault(r["original_record"], []).append(r)
        for h in hours:
            served = complete = censored = active = 0
            for root, p in self._passengers.items():
                begin = self.booking_clock.at(p.first_stranded_at)
                limit = begin + timedelta(hours=h)
                bookings = [
                    r for r in successes.get(root, []) if utc(r["attempt_utc"]) <= limit
                ]
                success = bool(bookings)
                served += success
                complete += now >= limit
                censored += not success and now < limit
                active += p.active_record is not None and any(
                    r["replacement_record"] == p.active_record for r in bookings
                )
            rows.append(
                dict(
                    hours=h,
                    stranded_lineages=n,
                    first_rebooked_within_hours=served,
                    active_replacement_booked_within_hours=active,
                    followup_at_least_hours=complete,
                    right_censored_without_first_success=censored,
                    first_rebooked_rate=None if not n or censored else served / n,
                    first_rebooked_rate_lower=None if not n else served / n,
                    first_rebooked_rate_upper=None
                    if not n
                    else (served + censored) / n,
                    active_replacement_rate_at_current_time=None
                    if not n
                    else active / n,
                )
            )
        return dict(
            clock_utc=now.isoformat(),
            unit="original stranded native passenger lineage",
            window_rule="first booking elapsed <=h; re-entry eligibility expires at its explicit exclusive UTC deadline",
            rows=rows,
            primary_B4_metric_adopted=False,
        )

    def snapshot(self):
        return DatedRescueSnapshot(
            super().snapshot(),
            self.booking_clock,
            self.leg_departures,
            self.dated_rebooking,
            self.dated_menu,
        )

    def restore(self, snapshot):
        if not isinstance(snapshot, DatedRescueSnapshot):
            raise TypeError("dated rescue snapshot required")
        if (snapshot.clock, snapshot.departures, snapshot.policy, snapshot.menu) != (
            self.booking_clock,
            self.leg_departures,
            self.dated_rebooking,
            self.dated_menu,
        ):
            raise ValueError("snapshot public dates/policy differ")
        super().restore(snapshot.native)

    def branch(self):
        other = DatedRescueMarketEngine(
            self.compiled,
            {i: copy.deepcopy(ctrl) for i, ctrl in self.controllers.items()},
            booking_clock=self.booking_clock,
            leg_departures=self.leg_departures,
            dated_rebooking=self.dated_rebooking,
            dated_menu=self.dated_menu,
            exits=self.exits,
            cap_policy=self.cap_policy,
            check_invariants=self.check_invariants,
        )
        other.restore(self.snapshot())
        return other
