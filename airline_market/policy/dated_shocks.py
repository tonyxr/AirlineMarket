"""A.5/B0/B1/V9: exact public-date adapters for B1 physical capacity shocks.

Prospective pre-departure execution only. Native capacity-window v2, refunds,
lineage, choice and period closure stay inherited; no empirical/grid adoption.
"""

import copy
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal, InvalidOperation

from airline_market.policy.capacity_windows_v2 import (
    CapacityLossWindow,
    CapacityWindowMarketEngineV2,
)
from airline_market.policy.dated_reserve import UtcBookingClock, utc
from airline_market.sim.disruption import RivalExit


@dataclass(frozen=True)
class B1DatedShock:
    """One declared B1 capacity design, anchored to a named public leg departure.

    A day means 24 elapsed UTC hours. Other legs can depart at different instants;
    their actual lead times remain in the mapping, not labelled anchor-day leads.
    Demand surge/hurricane composites and empirical anchoring are separate tasks.
    """

    shock_id: str
    kind: str
    reference_leg: str
    reveal_days: int
    share: object = None
    duration_days: int | None = None
    rival_carrier: str | None = None

    def __post_init__(self):
        if not isinstance(self.shock_id, str) or not self.shock_id.strip():
            raise ValueError("explicit shock ID required")
        if self.kind not in {"weather", "rival_outage", "rival_exit"}:
            raise ValueError("supported B1 physical capacity type required")
        if not isinstance(self.reference_leg, str) or not self.reference_leg:
            raise ValueError("explicit reference departure leg required")
        if type(self.reveal_days) is not int or self.reveal_days not in {14, 7, 3, 1}:
            raise ValueError("B1 reveal days must be 14, 7, 3 or 1")
        if self.kind == "weather":
            if self.rival_carrier is not None:
                raise ValueError("weather targets both compiled carriers")
        elif not isinstance(self.rival_carrier, str) or not self.rival_carrier:
            raise ValueError("explicit rival carrier required")
        if self.kind == "rival_exit":
            if self.duration_days is not None or self.share is not None:
                raise ValueError("permanent exit has no temporary duration/share")
        else:
            if type(self.duration_days) is not int or self.duration_days not in {
                1,
                2,
                3,
            }:
                raise ValueError("B1 temporary duration must be 1, 2 or 3 elapsed days")
            try:
                share = Decimal(str(self.share))
            except (InvalidOperation, ValueError) as error:
                raise ValueError("explicit finite B1 share required") from error
            allowed = (
                {Decimal(".1"), Decimal(".3"), Decimal(".6")}
                if self.kind == "weather"
                else {Decimal(".3"), Decimal(".6"), Decimal(".9")}
            )
            if not share.is_finite() or share not in allowed:
                raise ValueError("share outside this B1 type's declared magnitudes")
            object.__setattr__(self, "share", share)

    def record(self):
        return {
            "shock_id": self.shock_id,
            "kind": self.kind,
            "reference_leg": self.reference_leg,
            "reveal_days": self.reveal_days,
            "share": None if self.share is None else str(self.share),
            "duration_days": self.duration_days,
            "rival_carrier": self.rival_carrier,
        }


@dataclass(frozen=True)
class DatedShockMapping:
    design: B1DatedShock
    reference_departure: object
    reveal: object
    recovery: object
    affected_legs: tuple
    lead_microseconds: tuple
    native_window: object
    native_exit: object
    native_window_index: object
    native_exit_index: object
    status: str
    recovery_status: str

    def record(self):
        w = self.native_window
        e = self.native_exit
        return {
            "plan_id": "A.5/B0/B1/V9",
            "design": self.design.record(),
            "reference_departure_utc": self.reference_departure.isoformat(),
            "reveal_utc": self.reveal.isoformat(),
            "recovery_utc": None
            if self.recovery is None
            else self.recovery.isoformat(),
            "affected_legs": list(self.affected_legs),
            "departure_lead_microseconds": dict(self.lead_microseconds),
            "native_window": None
            if w is None
            else {
                "start": w.start,
                "stop": w.stop,
                "share": str(w.share),
                "legs": list(w.legs),
            },
            "native_exit": None
            if e is None
            else {"carrier": e.carrier, "reveal_period": e.reveal_period},
            "native_window_index": self.native_window_index,
            "native_exit_index": self.native_exit_index,
            "status": self.status,
            "recovery_status": self.recovery_status,
            "day_definition": "24 elapsed UTC hours",
            "outside_horizon_stop_is_not_a_dated_boundary": bool(
                w is not None and self.recovery_status == "beyond_horizon_outstanding"
            ),
        }


def _public_dates(compiled, clock, departures):
    if not isinstance(clock, UtcBookingClock):
        raise TypeError("explicit UtcBookingClock required")
    if len(clock.boundaries) != compiled.spec.horizon.n_periods + 1:
        raise ValueError("one UTC boundary per native period plus terminal required")
    dep = tuple(sorted((leg, utc(value)) for leg, value in departures))
    if len(dict(dep)) != len(dep) or set(dict(dep)) != set(compiled.leg_index):
        raise ValueError(
            "distinct public departure for every compiled physical leg required"
        )
    if clock.boundaries[-1] >= min(value for _, value in dep):
        raise ValueError("booking horizon must end before every flight departure")
    return dep


def compile_dated_shocks(compiled, clock, leg_departures, shocks):
    """Pure date/physical-ID projection. No world, actor, marks or RNG access.

    In-horizon events must equal a declared boundary. An exact future recovery is
    retained as a date with a non-executable T+1 sentinel; no boundary is invented.
    Reveal at/after terminal is unstarted, never moved back into the cohort.
    """
    dep = dict(_public_dates(compiled, clock, leg_departures))
    shocks = tuple(shocks)
    if any(type(s) is not B1DatedShock for s in shocks):
        raise TypeError("B1DatedShock records required")
    if len({s.shock_id for s in shocks}) != len(shocks):
        raise ValueError("distinct declared shock IDs required")
    if shocks and len(compiled.spec.carriers) != 2:
        raise ValueError("B1 physical adapter requires two explicit compiled carriers")
    T = compiled.spec.horizon.n_periods
    result = []
    window_count = exit_count = 0
    for s in shocks:
        if s.reference_leg not in dep:
            raise ValueError("reference leg outside public departure bank")
        legs = tuple(
            sorted(
                l.id
                for l in compiled.spec.legs
                if (s.kind == "weather" or l.carrier == s.rival_carrier)
            )
        )
        if not legs or s.reference_leg not in legs:
            raise ValueError(
                "reference leg must belong to declared affected carrier/legs"
            )
        reveal = dep[s.reference_leg] - timedelta(days=s.reveal_days)
        recovery = (
            None if s.kind == "rival_exit" else reveal + timedelta(days=s.duration_days)
        )
        if reveal < clock.at(0):
            raise ValueError(
                "pre-cohort reveal cannot identify installed stock/booked lineages"
            )
        window = exit_plan = None
        window_index = exit_index = None
        status = "unstarted_at_or_after_terminal"
        recovery_status = "permanent_exit" if recovery is None else "unstarted"
        if reveal < clock.at(T):
            start = clock.index(reveal)
            status = "mapped_exact_reveal"
            if recovery is None:
                exit_plan = RivalExit(s.rival_carrier, start)
                exit_index = exit_count
                exit_count += 1
            else:
                if recovery <= clock.at(T):
                    stop = clock.index(recovery)
                    recovery_status = "exact_boundary"
                else:
                    stop = T + 1
                    recovery_status = "beyond_horizon_outstanding"
                window = CapacityLossWindow(start, stop, s.share, legs)
                window_index = window_count
                window_count += 1
        result.append(
            DatedShockMapping(
                s,
                dep[s.reference_leg],
                reveal,
                recovery,
                legs,
                tuple(
                    (leg, (dep[leg] - reveal) // timedelta(microseconds=1))
                    for leg in legs
                ),
                window,
                exit_plan,
                window_index,
                exit_index,
                status,
                recovery_status,
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class DatedShockSnapshot:
    native: object
    clock: object
    departures: tuple
    designs: tuple
    mappings: tuple


class DatedShockMarketEngine(CapacityWindowMarketEngineV2):
    """One world; UTC projection only, all native physical execution inherited."""

    def __init__(
        self,
        compiled,
        controllers=None,
        *,
        shock_clock,
        shock_leg_departures,
        dated_shocks=(),
        **kwargs,
    ):
        if "capacity_windows" in kwargs or "exits" in kwargs:
            raise ValueError("dated adapter derives native capacity windows and exits")
        self.shock_clock = shock_clock
        self.shock_leg_departures = _public_dates(
            compiled, shock_clock, shock_leg_departures
        )
        self.dated_shocks = tuple(dated_shocks)
        self._dated_shock_mappings = compile_dated_shocks(
            compiled, shock_clock, self.shock_leg_departures, self.dated_shocks
        )
        if (
            kwargs.get("booking_clock") is not None
            and kwargs["booking_clock"] != shock_clock
        ):
            raise ValueError("shock and dated policy public clocks differ")
        if (
            kwargs.get("leg_departures")
            and tuple(
                sorted((leg, utc(value)) for leg, value in kwargs["leg_departures"])
            )
            != self.shock_leg_departures
        ):
            raise ValueError("shock and dated policy departure banks differ")
        if kwargs.get("dated_rebooking") is not None:
            kwargs["booking_clock"] = shock_clock
            kwargs["leg_departures"] = self.shock_leg_departures
        super().__init__(
            compiled,
            controllers,
            capacity_windows=tuple(
                m.native_window
                for m in self._dated_shock_mappings
                if m.native_window is not None
            ),
            exits=tuple(
                m.native_exit
                for m in self._dated_shock_mappings
                if m.native_exit is not None
            ),
            **kwargs,
        )

    def dated_shock_mappings(self):
        return self._dated_shock_mappings

    def snapshot(self):
        return DatedShockSnapshot(
            super().snapshot(),
            self.shock_clock,
            self.shock_leg_departures,
            self.dated_shocks,
            self._dated_shock_mappings,
        )

    def restore(self, snapshot):
        if type(snapshot) is not DatedShockSnapshot:
            raise TypeError("dated-shock combined snapshot required")
        if (
            snapshot.clock != self.shock_clock
            or snapshot.departures != self.shock_leg_departures
            or snapshot.designs != self.dated_shocks
            or snapshot.mappings != self._dated_shock_mappings
        ):
            raise ValueError(
                "snapshot dated-shock clock/departure/design contract differs"
            )
        super().restore(snapshot.native)

    def branch(self):
        other = DatedShockMarketEngine(
            self.compiled,
            {i: copy.deepcopy(c) for i, c in self.controllers.items()},
            shock_clock=self.shock_clock,
            shock_leg_departures=self.shock_leg_departures,
            dated_shocks=self.dated_shocks,
            ordinary_priority=self.ordinary_priority,
            trigger_rule=self.trigger.rule if self.trigger else None,
            mechanical_benchmark=self.trigger.benchmark if self.trigger else None,
            dated_rebooking=self.dated_rebooking,
            dated_menu=self.dated_menu,
            dated_reserve=self.dated_reserve,
            cap_policy=self.cap_policy,
            check_invariants=self.check_invariants,
        )
        other.restore(self.snapshot())
        return other
