"""A.10/A.5/A.14/B0/V9: actual ordinary taste-return capture, analyst only."""

import copy
from dataclasses import dataclass

from airline_market.core.hashing import jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.sim.native_private_taste import NativePrivateTasteMarketEngine
from airline_market.sim.ordinary_taste_executor import _OrdinaryTasteExecutor
from airline_market.sim.received_information import _units
from airline_market.sim.reentry_stage_information import _stage_encode

RULE = "same-call-native-ordinary-returned-taste-v1"


@dataclass(frozen=True)
class OrdinaryTasteInput:
    key: tuple
    native_rank: int
    segment: int
    taste_class: int
    native_log_index: int
    parent_input_index: int
    route: str
    resolved_taste: object

    def __post_init__(self):
        object.__setattr__(self, "resolved_taste", deep_freeze(self.resolved_taste))


@dataclass(frozen=True)
class OrdinaryTasteRow:
    analyst_person: object
    private_input: OrdinaryTasteInput


@dataclass(frozen=True)
class OrdinaryTastePeriod:
    period: int
    native_world_t: int
    rows: tuple


@dataclass(frozen=True)
class OrdinaryTasteSnapshot:
    native: object
    periods: tuple
    pending: object
    failed: bool
    row_limit: int
    spec_hash: str
    rule: str = RULE


class OrdinaryTasteLimit(RuntimeError):
    pass


class OrdinaryPrivateTasteMarketEngine(
    NativePrivateTasteMarketEngine, _OrdinaryTasteExecutor
):
    """Real inherited two-route step; earlier information views stay unchanged."""

    def __init__(self, *args, ordinary_taste_row_limit=8000000, **kwargs):
        if type(ordinary_taste_row_limit) is not int or ordinary_taste_row_limit < 1:
            raise ValueError("positive complete-source structural limit required")
        self.ordinary_taste_row_limit = ordinary_taste_row_limit
        self._ordinary_taste_periods = []
        self._ordinary_taste_pending = None
        self._ordinary_taste_failed = False
        super().__init__(*args, **kwargs)

    def _ordinary_taste_payload(self):
        return {
            "plan_id": "A.10/A.5/A.14/B0/V9",
            "rule": RULE,
            "row_limit": self.ordinary_taste_row_limit,
            "parent_source": self._private_payload(),
            "periods": _stage_encode(jsonable(self._ordinary_taste_periods)),
            "pending": _stage_encode(jsonable(self._ordinary_taste_pending)),
            "failed": self._ordinary_taste_failed,
            "actual_ordinary_taste_rows_captured": any(
                p.rows for p in self._ordinary_taste_periods
            ),
            "scope": "analyst-only-actual-ordinary-taste-return-before-utility",
            "earlier_input_views_augmented_with_private_taste": False,
            "gumbel_values_inspected_or_exported": False,
            "native_joint_probability_knowledge_preferences_authenticated": False,
            "lifecycle_utility_CS_service_payoff": None,
            "scientific_budget": None,
        }

    def _check_ordinary_taste_limit(self, proposed=None):
        if (
            _units(
                {
                    "source": self._ordinary_taste_payload(),
                    "proposed": _stage_encode(jsonable(proposed)),
                }
            )
            > self.ordinary_taste_row_limit
        ):
            raise OrdinaryTasteLimit("complete source/proposed support limit")

    def reset(self, seed=None, *, tape=None):
        self._ordinary_taste_periods = []
        self._ordinary_taste_pending = None
        self._ordinary_taste_failed = True
        result = super().reset(seed, tape=tape)
        self._check_ordinary_taste_limit()
        self._ordinary_taste_failed = False
        return result

    def _capture_ordinary_taste(self, request, period, taste, route):
        if self._ordinary_taste_pending is None:
            raise RuntimeError("actual complete native step required")
        key = (period, request.gumbel_row, request.market)
        index = len(self.world.request_log)
        parent_index = len(self._causal_pending) - 1
        parent = self._causal_pending[parent_index]
        if (
            self._causal_inflight != (key, index)
            or parent.kind != "ordinary"
            or parent.analyst_original_key != key
            or self._causal_decision is None
        ):
            raise RuntimeError("SAME genuine request/earlier native input required")
        row = OrdinaryTasteInput(
            key,
            request.rank,
            request.segment,
            request.taste_class,
            index,
            parent_index,
            route,
            jsonable(taste),
        )
        if any(r.key == key for r in self._ordinary_taste_pending):
            raise RuntimeError("one actual taste return per ordinary request required")
        self._check_ordinary_taste_limit(row)
        self._ordinary_taste_pending.append(row)

    def _finish_ordinary_taste_period(self, period):
        people = self._ordinary_periods[-1].rows
        inputs = self._causal_periods[-1].inputs
        if (
            len(people) != len(self._ordinary_taste_pending)
            or self.world.t != period + 1
        ):
            raise RuntimeError("ALL ordinary people and full native closure required")
        rows = []
        for person, row in zip(people, self._ordinary_taste_pending):
            parent = inputs[row.parent_input_index]
            if (
                (row.key, row.native_rank, row.taste_class, row.native_log_index)
                != (
                    person.key,
                    person.native_rank,
                    person.taste_class,
                    person.native_log_index,
                )
                or self.compiled.spec.segments[row.segment].id != person.segment
                or parent.kind != "ordinary"
                or parent.analyst_original_key != row.key
            ):
                raise RuntimeError("actual taste/source/person correspondence differs")
            rows.append(OrdinaryTasteRow(person, row))
        candidate = OrdinaryTastePeriod(period, self.world.t, tuple(rows))
        self._check_ordinary_taste_limit(candidate)
        self._ordinary_taste_periods.append(candidate)
        self._ordinary_taste_pending = None

    def _require_ordinary_taste_complete(self):
        self._require_private_complete()
        if self._ordinary_taste_pending is not None or self._ordinary_taste_failed:
            raise ValueError("incomplete ordinary taste; no query/continuation/retry")

    def step(self, actions=None, *, _requests=None, _choices=None):
        self._require_ordinary_taste_complete()
        if _requests is not None or _choices is not None:
            raise ValueError("genuine native requests/choices required")
        if self.world is None or self.terminal():
            raise ValueError("initialized nonterminal opportunity required")
        self._check_ordinary_taste_limit()
        period = self.world.t
        self._ordinary_taste_pending = []
        try:
            result = super().step(actions)
            self._finish_ordinary_taste_period(period)
        except Exception:
            self._ordinary_taste_failed = True
            raise
        return result

    def present_cancelled_itinerary(self, booking, *, airport_counter=False):
        self._require_ordinary_taste_complete()
        result = super().present_cancelled_itinerary(
            booking, airport_counter=airport_counter
        )
        self._ordinary_taste_failed = True
        self._check_ordinary_taste_limit()
        self._ordinary_taste_failed = False
        return result

    def ordinary_taste_periods(self):
        self._require_ordinary_taste_complete()
        self._check_ordinary_taste_limit()
        return tuple(self._ordinary_taste_periods)

    def snapshot(self):
        return OrdinaryTasteSnapshot(
            super().snapshot(),
            tuple(self._ordinary_taste_periods),
            None
            if self._ordinary_taste_pending is None
            else tuple(self._ordinary_taste_pending),
            self._ordinary_taste_failed,
            self.ordinary_taste_row_limit,
            self.compiled.spec_hash,
        )

    def restore(self, snapshot):
        if type(snapshot) is not OrdinaryTasteSnapshot:
            raise TypeError("combined ordinary-taste/native snapshot required")
        if (snapshot.rule, snapshot.row_limit, snapshot.spec_hash) != (
            RULE,
            self.ordinary_taste_row_limit,
            self.compiled.spec_hash,
        ):
            raise ValueError("ordinary taste rule/limit/full model differs")
        super().restore(snapshot.native)
        self._ordinary_taste_periods = list(snapshot.periods)
        self._ordinary_taste_pending = (
            None if snapshot.pending is None else list(snapshot.pending)
        )
        self._ordinary_taste_failed = snapshot.failed

    def branch(self):
        other = type(self)(
            self.demand_scenario,
            {i: copy.deepcopy(c) for i, c in self.controllers.items()},
            ordinary_taste_row_limit=self.ordinary_taste_row_limit,
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


def ordinary_private_taste_record(engine):
    if type(engine) is not OrdinaryPrivateTasteMarketEngine:
        raise TypeError("prospective ordinary private-taste source required")
    engine.ordinary_taste_periods()
    return engine._ordinary_taste_payload()
