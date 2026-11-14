"""A.5/B0/B1/V9: opt-in temporary installed-seat loss on native boundaries.

Floor-rounded share of actual installed capacity at reveal; restore that debit
at expiry unless permanent exit blocks recovery. Native resolve/refund/lineage
and inherited choice/booking/closure remain. No empirical duration/seat adoption.
"""

import copy
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from airline_market.policy.potential_lineage import PotentialLineageMarketEngine
from airline_market.sim.ledger import resolve


@dataclass(frozen=True)
class CapacityLossWindow:
    start: int
    stop: int
    share: object
    legs: tuple

    def __post_init__(self):
        if (
            type(self.start) is not int
            or type(self.stop) is not int
            or not 0 <= self.start < self.stop
        ):
            raise ValueError("explicit ordered native-period window required")
        share = Decimal(str(self.share))
        if not share.is_finite() or not 0 <= share <= 1:
            raise ValueError("finite installed-seat loss share in [0,1] required")
        legs = tuple(self.legs)
        if (
            not legs
            or len(set(legs)) != len(legs)
            or any(not isinstance(k, str) or not k for k in legs)
        ):
            raise ValueError("distinct explicit physical target legs required")
        object.__setattr__(self, "share", share)
        object.__setattr__(self, "legs", legs)


@dataclass(frozen=True)
class CapacityWindowEvent:
    window: int
    period: int
    kind: str
    changes: tuple
    canceled_ids: tuple = ()
    refunds: float = 0.0
    penalties: float = 0.0

    def record(self):
        return {"disruption": "temporary_capacity_window", **vars(self)}


@dataclass(frozen=True)
class CapacityWindowSnapshot:
    native: object
    windows: tuple
    exits: tuple
    started: frozenset
    finished: frozenset
    baselines: tuple
    removed: tuple
    restored: tuple
    blocked: tuple
    boundary: object
    history: tuple


class CapacityWindowMarketEngine(PotentialLineageMarketEngine):
    """Prospective physical executor for nonoverlapping per-leg temporary losses.

    Native capacity-process effects on target legs require a later explicit
    reconciliation revision; other legs' native events remain inherited. Native
    permanent exits dominate restoration. Stops beyond horizon stay outstanding.
    No policy/zero shares leave native world/trace/metrics unchanged in fixtures.
    """

    def __init__(self, *args, capacity_windows=(), **kwargs):
        self.capacity_windows = tuple(capacity_windows)
        self._clear_windows()
        super().__init__(*args, **kwargs)
        T = self.compiled.spec.horizon.n_periods
        targets = set()
        for w in self.capacity_windows:
            if type(w) is not CapacityLossWindow:
                raise TypeError("CapacityLossWindow records required")
            if w.start >= T or set(w.legs) - set(self.compiled.leg_index):
                raise ValueError("window reveal or physical leg outside native model")
            targets.update(w.legs if w.share else ())
        for k, a in enumerate(self.capacity_windows):
            for b in self.capacity_windows[k + 1 :]:
                if (
                    a.share
                    and b.share
                    and set(a.legs) & set(b.legs)
                    and max(a.start, b.start) < min(a.stop, b.stop)
                ):
                    raise ValueError(
                        "overlapping losses on one leg require explicit reconciliation"
                    )
        if any(
            leg in targets and delta
            for e in self.compiled.spec.capacity_process.events
            for leg, delta in e.effects
        ):
            raise ValueError(
                "native capacity events on loss legs require explicit reconciliation"
            )

    def _clear_windows(self):
        self._loss_started = set()
        self._loss_finished = set()
        self._loss_baselines = {}
        self._loss_removed = {}
        self._loss_restored = {}
        self._loss_blocked = {}
        self._loss_boundary = None
        self._loss_history = []

    def _installed(self, leg):
        c = self.compiled
        l = c.leg_index[leg]
        i = int(c.leg_carrier[l])
        local = c.legs_of[i].index(l)
        return self.world.carriers[i], local

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
        for k, w in enumerate(self.capacity_windows):
            if k in self._loss_started or period < w.start:
                continue
            baseline = []
            removed = []
            affected = set()
            for leg in w.legs:
                state, local = self._installed(leg)
                n = int(state.capacity[local])
                fraction = Fraction(w.share)
                loss = (fraction.numerator * n) // fraction.denominator
                baseline.append((leg, n))
                removed.append((leg, loss))
                if loss:
                    state.capacity[local] -= loss
                    affected.add(
                        int(self.compiled.leg_carrier[self.compiled.leg_index[leg]])
                    )
            canceled = []
            refund = 0.0
            penalty = 0.0
            for i in sorted(affected):
                outcome = resolve(self.world, i, self.compiled)
                canceled.extend(outcome.canceled)
                refund += outcome.refund
                penalty += outcome.penalty
            self._record_stranded(tuple(canceled), period, "temporary_capacity_window")
            self._loss_baselines[k] = tuple(baseline)
            self._loss_removed[k] = tuple(removed)
            self._loss_started.add(k)
            event = CapacityWindowEvent(
                k,
                period,
                "remove",
                tuple((leg, -n) for leg, n in removed),
                tuple(b.record_id for b in canceled),
                refund,
                penalty,
            )
            self._loss_history.append(event)
            if affected:
                events.append(event)
        return tuple(events)

    def _enforce_exits(self, period):
        return (*self._apply_capacity_windows(period), *super()._enforce_exits(period))

    def reset(self, seed=None, *, tape=None):
        self._clear_windows()
        result = super().reset(seed, tape=tape)
        if self._loss_boundary != 0:
            if any(w.start == 0 and w.share for w in self.capacity_windows):
                self._open_period()
            self._pending_events += self._apply_capacity_windows(0)
        return result

    def capacity_window_balance(self):
        rows = []
        for k in sorted(self._loss_started):
            for leg, n in self._loss_removed[k]:
                returned = dict(self._loss_restored.get(k, ())).get(leg, 0)
                blocked = dict(self._loss_blocked.get(k, ())).get(leg, 0)
                rows.append(
                    {
                        "window": k,
                        "leg": leg,
                        "installed_at_reveal": dict(self._loss_baselines[k])[leg],
                        "removed": n,
                        "restored": returned,
                        "blocked_by_exit": blocked,
                        "outstanding": n - returned - blocked,
                        "finished": k in self._loss_finished,
                    }
                )
        return tuple(rows)

    def capacity_window_events(self):
        return tuple(self._loss_history)

    def snapshot(self):
        return CapacityWindowSnapshot(
            super().snapshot(),
            self.capacity_windows,
            self.exits,
            frozenset(self._loss_started),
            frozenset(self._loss_finished),
            tuple(self._loss_baselines.items()),
            tuple(self._loss_removed.items()),
            tuple(self._loss_restored.items()),
            tuple(self._loss_blocked.items()),
            self._loss_boundary,
            tuple(self._loss_history),
        )

    def restore(self, snapshot):
        if type(snapshot) is not CapacityWindowSnapshot:
            raise TypeError("capacity-window combined snapshot required")
        if snapshot.windows != self.capacity_windows or snapshot.exits != self.exits:
            raise ValueError("snapshot capacity-window/exit contract differs")
        super().restore(snapshot.native)
        self._loss_started = set(snapshot.started)
        self._loss_finished = set(snapshot.finished)
        self._loss_baselines = dict(snapshot.baselines)
        self._loss_removed = dict(snapshot.removed)
        self._loss_restored = dict(snapshot.restored)
        self._loss_blocked = dict(snapshot.blocked)
        self._loss_boundary = snapshot.boundary
        self._loss_history = list(snapshot.history)

    def branch(self):
        other = CapacityWindowMarketEngine(
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
