"""A.5/B0/B1/V9: simultaneous native resolution for temporary window losses.

Supersedes v1 ONLY for new cohorts. v1 bytes/tests/results remain retained. A
combined boundary changes all declared legs before one resolve per carrier.
"""

import copy
from dataclasses import dataclass
from fractions import Fraction

from airline_market.policy.capacity_windows import (
    CapacityLossWindow,
)
from airline_market.policy.capacity_windows import CapacityWindowEvent
from airline_market.policy.capacity_windows import (
    CapacityWindowMarketEngine as V1MarketEngine,
)
from airline_market.sim.ledger import resolve

__all__ = (
    "CapacityLossWindow",
    "CapacityWindowMarketEngineV2",
    "CapacityWindowMarketEngine",
)


@dataclass(frozen=True)
class CapacityWindowV2Snapshot:
    native: object
    semantics: str = "simultaneous-boundary-loss-v2"


class CapacityWindowMarketEngineV2(V1MarketEngine):
    """Same contracts/physical hooks, with atomic per-carrier boundary losses."""

    def _apply_capacity_windows(self, period):
        if self._loss_boundary == period:
            return ()
        if self._loss_boundary is not None and period < self._loss_boundary:
            raise ValueError(
                "capacity-window clock cannot go backwards without restore"
            )
        self._loss_boundary = period
        events = []
        # Expiry precedes adjacent reveal so the latter measures fresh installed stock.
        for k, w in enumerate(self.capacity_windows):
            if (
                k not in self._loss_started
                or k in self._loss_finished
                or period < w.stop
            ):
                continue
            changes = []
            restored = []
            blocked = []
            for leg, n in self._loss_removed[k]:
                state, local = self._installed(leg)
                carrier = self.compiled.spec.legs[self.compiled.leg_index[leg]].carrier
                denied = any(
                    e.carrier == carrier and period >= e.reveal_period
                    for e in self.exits
                )
                amount = 0 if denied else n
                if amount:
                    state.capacity[local] += amount
                restored.append((leg, amount))
                blocked.append((leg, n - amount))
                changes.append((leg, amount))
            self._loss_restored[k] = tuple(restored)
            self._loss_blocked[k] = tuple(blocked)
            self._loss_finished.add(k)
            event = CapacityWindowEvent(k, period, "restore", tuple(changes))
            self._loss_history.append(event)
            if any(n for _, n in self._loss_removed[k]):
                events.append(event)
        starts = []
        all_changes = []
        affected = set()
        for k, w in enumerate(self.capacity_windows):
            if k in self._loss_started or period < w.start:
                continue
            baseline = []
            removed = []
            for leg in w.legs:
                state, local = self._installed(leg)
                n = int(state.capacity[local])
                fraction = Fraction(w.share)
                loss = (fraction.numerator * n) // fraction.denominator
                baseline.append((leg, n))
                removed.append((leg, loss))
                all_changes.append((leg, -loss))
                if loss:
                    state.capacity[local] -= loss
                    affected.add(
                        int(self.compiled.leg_carrier[self.compiled.leg_index[leg]])
                    )
            self._loss_baselines[k] = tuple(baseline)
            self._loss_removed[k] = tuple(removed)
            self._loss_started.add(k)
            starts.append(k)
        if starts:
            canceled = []
            refund = 0.0
            penalty = 0.0
            # All same-boundary losses are installed before ONE native resolution
            # per affected carrier. Sequential resolution can over-cancel because
            # a later leg loss may release stock held by an earlier kept booking.
            for i in sorted(affected):
                outcome = resolve(self.world, i, self.compiled)
                canceled.extend(outcome.canceled)
                refund += outcome.refund
                penalty += outcome.penalty
            self._record_stranded(tuple(canceled), period, "temporary_capacity_window")
            event = CapacityWindowEvent(
                starts[0] if len(starts) == 1 else -1,
                period,
                "remove" if len(starts) == 1 else "remove_batch",
                tuple(all_changes),
                tuple(b.record_id for b in canceled),
                refund,
                penalty,
            )
            self._loss_history.append(event)
            if affected:
                events.append(event)
        return tuple(events)

    def branch(self):
        other = CapacityWindowMarketEngineV2(
            self.compiled,
            {i: copy.deepcopy(ctrl) for i, ctrl in self.controllers.items()},
            capacity_windows=self.capacity_windows,
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

    def snapshot(self):
        return CapacityWindowV2Snapshot(super().snapshot())

    def restore(self, snapshot):
        if type(snapshot) is not CapacityWindowV2Snapshot:
            raise TypeError("v2 simultaneous capacity-window snapshot required")
        if snapshot.semantics != "simultaneous-boundary-loss-v2":
            raise ValueError("snapshot simultaneous capacity semantics differ")
        super().restore(snapshot.native)


CapacityWindowMarketEngine = CapacityWindowMarketEngineV2
