"""A.10/A.5/A.14/B0/V9: analyst-only actual native re-entry taste capture.

The returned taste is observed after ONE inherited marks call. Gumbels are not
inspected, copied or redrawn. A declared known label remains a separate object.
"""

import copy
from dataclasses import dataclass

from airline_market.core.hashing import jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.sim.native_causal_information import (
    NativeCausalInformationMarketEngine,
)
from airline_market.sim.received_information import _units
from airline_market.sim.reentry_stage_information import _stage_encode

RULE = "prospective-native-returned-private-taste-v1"


@dataclass(frozen=True)
class NativePrivateTaste:
    analyst_root: int
    analyst_original_key: tuple
    native_period: int
    original_booking: object
    resolved_taste: object

    def __post_init__(self):
        object.__setattr__(self, "resolved_taste", deep_freeze(self.resolved_taste))


@dataclass(frozen=True)
class PrivateTastePeriod:
    period: int
    native_world_t: int
    rows: tuple


@dataclass(frozen=True)
class PrivateTasteSnapshot:
    native: object
    periods: tuple
    pending: object
    failed: bool
    row_limit: int
    spec_hash: str
    rule: str = RULE


class NativePrivateTasteLimit(RuntimeError):
    pass


class NativePrivateTasteMarketEngine(NativeCausalInformationMarketEngine):
    """One inherited complete step; private taste never enters an input view."""

    def __init__(self, *args, private_row_limit=4000000, **kwargs):
        if type(private_row_limit) is not int or private_row_limit < 1:
            raise ValueError("positive all-source structural limit required")
        self.private_row_limit = private_row_limit
        self._private_periods = []
        self._private_pending = None
        self._private_failed = False
        super().__init__(*args, **kwargs)

    def _private_payload(self):
        return {
            "plan_id": "A.10/A.5/A.14/B0/V9",
            "rule": RULE,
            "row_limit": self.private_row_limit,
            "parent_source": self._payload(),
            "periods": _stage_encode(jsonable(self._private_periods)),
            "pending": _stage_encode(jsonable(self._private_pending)),
            "failed": self._private_failed,
            "actual_returned_taste_rows_captured": any(
                p.rows for p in self._private_periods
            ),
            "scope": "analyst-only-actual-returned-native-reentry-taste",
            "ordinary_resolved_mixed_taste_captured": False,
            "gumbel_values_inspected_or_exported": False,
            "native_taste_is_declared_passenger_known_label": False,
            "native_joint_probability_or_received_knowledge_authenticated": False,
            "lifecycle_utility_CS_service_payoff": None,
            "scientific_budget": None,
        }

    def _check_private_limit(self, proposed=None):
        if (
            _units(
                {
                    "source": self._private_payload(),
                    "proposed": _stage_encode(jsonable(proposed)),
                }
            )
            > self.private_row_limit
        ):
            raise NativePrivateTasteLimit("complete source/proposed support limit")

    def reset(self, seed=None, *, tape=None):
        self._private_periods = []
        self._private_pending = None
        self._private_failed = True
        result = super().reset(seed, tape=tape)
        self._check_private_limit()
        self._private_failed = False
        return result

    def _capture_private_taste(self, passenger, period, taste):
        if self._private_pending is None:
            raise RuntimeError("actual inherited step required")
        b = passenger.original
        row = NativePrivateTaste(
            b.record_id,
            tuple(self._own_keys[b.record_id]),
            period,
            b,
            jsonable(taste),
        )
        if any(r.analyst_root == row.analyst_root for r in self._private_pending):
            raise RuntimeError("one actual marks return per root/period required")
        self._check_private_limit(row)
        self._private_pending.append(row)

    def _marks(self, passenger, period):
        marks = super()._marks(passenger, period)
        self._capture_private_taste(passenger, period, marks[0])
        return marks

    def _finish_private_period(self, period):
        expected = []
        for row in self._stage_periods[-1].inputs:
            key = (row.analyst_root, row.analyst_original_key, period)
            if key not in expected:
                expected.append(key)
        actual = [
            (r.analyst_root, r.analyst_original_key, r.native_period)
            for r in self._private_pending
        ]
        if actual != expected or self.world.t != period + 1:
            raise RuntimeError("ALL actual attempted roots and native closure required")
        candidate = PrivateTastePeriod(
            period, self.world.t, tuple(self._private_pending)
        )
        self._check_private_limit(candidate)
        self._private_periods.append(candidate)
        self._private_pending = None

    def _require_private_complete(self):
        self._require_causal_complete()
        if self._private_pending is not None or self._private_failed:
            raise ValueError("incomplete private capture; no query/continuation/retry")

    def step(self, actions=None, *, _requests=None, _choices=None):
        self._require_private_complete()
        if _requests is not None or _choices is not None:
            raise ValueError("genuine native requests and choices required")
        if self.world is None or self.terminal():
            raise ValueError("initialized nonterminal opportunity required")
        self._check_private_limit()
        period = self.world.t
        self._private_pending = []
        try:
            result = super().step(actions)
            self._finish_private_period(period)
        except Exception:
            self._private_failed = True
            raise
        return result

    def present_cancelled_itinerary(self, booking, *, airport_counter=False):
        self._require_private_complete()
        result = super().present_cancelled_itinerary(
            booking, airport_counter=airport_counter
        )
        self._private_failed = True
        self._check_private_limit()
        self._private_failed = False
        return result

    def private_taste_periods(self):
        self._require_private_complete()
        self._check_private_limit()
        return tuple(self._private_periods)

    def snapshot(self):
        return PrivateTasteSnapshot(
            super().snapshot(),
            tuple(self._private_periods),
            None if self._private_pending is None else tuple(self._private_pending),
            self._private_failed,
            self.private_row_limit,
            self.compiled.spec_hash,
        )

    def restore(self, snapshot):
        if type(snapshot) is not PrivateTasteSnapshot:
            raise TypeError("combined private-taste/native snapshot required")
        if (snapshot.rule, snapshot.row_limit, snapshot.spec_hash) != (
            RULE,
            self.private_row_limit,
            self.compiled.spec_hash,
        ):
            raise ValueError("private rule/limit/full native model differs")
        super().restore(snapshot.native)
        self._private_periods = list(snapshot.periods)
        self._private_pending = (
            None if snapshot.pending is None else list(snapshot.pending)
        )
        self._private_failed = snapshot.failed

    def branch(self):
        other = type(self)(
            self.demand_scenario,
            {i: copy.deepcopy(c) for i, c in self.controllers.items()},
            private_row_limit=self.private_row_limit,
            causal_fields=self.causal_fields,
            causal_known_types=self.causal_known_types,
            causal_deliveries=self.causal_deliveries,
            causal_definition=self.causal_definition,
            causal_unobserved_definition=self.causal_unobserved_definition,
            causal_row_limit=self.causal_row_limit,
            stage_row_limit=self.stage_row_limit,
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


def native_private_taste_record(engine):
    if type(engine) is not NativePrivateTasteMarketEngine:
        raise TypeError("prospective native private-taste source required")
    engine.private_taste_periods()
    return engine._private_payload()
