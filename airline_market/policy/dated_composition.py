"""A.5/A.14/B0/V9: compose dated rescue eligibility and per-leg reserve release.

Physical pre-departure executor only. Frozen parent hooks retain native choice,
book/resolve, refund, quota and named-stream behavior. No old planner adoption.
"""

import copy

from airline_market.policy.dated_rescue import (
    DatedRebookingPolicy,
    DatedRescueMarketEngine,
    DatedRescueMenu,
)
from airline_market.policy.dated_reserve import (
    DatedReserveMarketEngine,
    DatedSeatReserve,
    UtcBookingClock,
    utc,
)
from airline_market.sim.rebooking import RebookingPolicy


class DatedRescueReserveMarketEngine(DatedRescueMarketEngine, DatedReserveMarketEngine):
    """One public UTC clock for eligibility, expiry and exact per-leg release.

    The MRO wraps reserve re-entry with dated rescue timestamps and eligibility.
    Nested parent snapshots own both ledgers; no independent engine is stepped.
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
        dated_reserve=None,
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
        if dated_reserve is not None:
            if not isinstance(dated_reserve, DatedSeatReserve):
                raise TypeError("DatedSeatReserve required")
            if dep != dated_reserve.leg_departures:
                raise ValueError("rescue and reserve public departure banks differ")
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
        # Initialize the reserve side once. The inherited rescue hooks then wrap
        # its held-first sales, ledger and snapshots through this explicit MRO.
        DatedReserveMarketEngine.__init__(
            self,
            compiled,
            controllers,
            booking_clock=booking_clock if dated_reserve is not None else None,
            dated_reserve=dated_reserve,
            rebooking=p,
            rescue_menu=menu,
            **kwargs,
        )
        self.booking_clock = booking_clock

    def branch(self):
        other = DatedRescueReserveMarketEngine(
            self.compiled,
            {i: copy.deepcopy(ctrl) for i, ctrl in self.controllers.items()},
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
