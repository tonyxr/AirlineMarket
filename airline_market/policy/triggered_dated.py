"""A.5/A.14/B0/V9: compose public trigger posting with dated rescue/reserve.

One native world; frozen parent price, choice, refund, lineage and quota hooks.
Trigger windows remain explicit native opportunities on the public UTC clock,
not inferred elapsed-hour activation durations or a B2 economic benchmark.
"""

import copy
from dataclasses import dataclass

from airline_market.policy.dated_composition import DatedRescueReserveMarketEngine
from airline_market.policy.dated_reserve import utc
from airline_market.policy.trigger import PublicTriggerReport, TriggerMarketEngine


@dataclass(frozen=True)
class TriggeredDatedSnapshot:
    native: object
    rule: object
    benchmark: object


class TriggeredDatedMarketEngine(TriggerMarketEngine, DatedRescueReserveMarketEngine):
    """Trigger posting precedes all ordinary and private dated native purchases.

    The cooperative parent chain initializes one world and nests trigger state
    around both dated snapshots. Existing public reports are never filled from
    private inventory. UTC report binding adds no forecast or observation source.
    """

    def publish_regulator_report_at(self, instant, report):
        if type(report) is not PublicTriggerReport:
            raise TypeError("explicit public trigger report required")
        if self.booking_clock is None:
            raise ValueError("explicit public UTC clock required for dated reports")
        if utc(instant) != self.booking_clock.at(report.period):
            raise ValueError("report UTC instant differs from its public opportunity")
        return self.publish_regulator_report(report)

    def snapshot(self):
        return TriggeredDatedSnapshot(
            super().snapshot(),
            self.trigger.rule if self.trigger else None,
            self.trigger.benchmark if self.trigger else None,
        )

    def restore(self, snapshot):
        if not isinstance(snapshot, TriggeredDatedSnapshot):
            raise TypeError("combined dated trigger snapshot required")
        contract = (
            self.trigger.rule if self.trigger else None,
            self.trigger.benchmark if self.trigger else None,
        )
        if (snapshot.rule, snapshot.benchmark) != contract:
            raise ValueError("snapshot public trigger rule/benchmark differ")
        super().restore(snapshot.native)

    def branch(self):
        other = TriggeredDatedMarketEngine(
            self.compiled,
            {i: copy.deepcopy(ctrl) for i, ctrl in self.controllers.items()},
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
