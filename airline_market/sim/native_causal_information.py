"""A.10/A.5/A.14/B0/V9: prospective native hooks for declared causal receipts.

Source order is genuine hook order. Receipt/type content and same-anchor event
semantics remain explicit declarations, not authenticated passenger knowledge.
"""

import copy
import hashlib
from dataclasses import dataclass

from airline_market.core.hashing import canonical_json, jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.sim.causal_information import (
    CausalEvent,
    CausalPosition,
    DecisionInput,
    DecisionResult,
    MessageReceived,
    MessageRecorded,
    MessageSource,
    OwnAction,
    RegisterPerson,
    append_causal_events,
    causal_decision_receipt,
    causal_information_journal,
    causal_information_record,
)
from airline_market.sim.received_information import (
    KINDS,
    DisclosureField,
    _key,
    _text,
    _units,
)
from airline_market.sim.reentry_stage_information import (
    ReentryStageInformationMarketEngine,
    _stage_encode,
)

RULE = "prospective-native-causal-input-source-v1"
TIMING = "three-declared-events-immediately-before-reached-native-input"


@dataclass(frozen=True)
class DeclaredKnownType:
    person_key: tuple
    value: str | None
    definition: str
    source_ref: str

    def __post_init__(self):
        object.__setattr__(self, "person_key", _key(self.person_key))
        if self.value is not None:
            _text(self.value)
        _text(self.definition)
        _text(self.source_ref)


@dataclass(frozen=True)
class DeclaredInputDelivery:
    person_key: tuple
    period: int
    stage: str
    delivery_id: str
    field: str
    value: object
    source_ref: str
    timing: str

    def __post_init__(self):
        object.__setattr__(self, "person_key", _key(self.person_key))
        if type(self.period) is not int or self.period < 0:
            raise ValueError("explicit nonnegative period required")
        if self.stage not in {"ordinary", "own", "interline"} or self.timing != TIMING:
            raise ValueError("explicit reached-input three-event schedule required")
        for x in (self.delivery_id, self.field, self.source_ref):
            _text(x)


@dataclass(frozen=True)
class NativeCausalInput:
    kind: str
    parent_input_index: int
    analyst_original_key: tuple
    analyst_root: int | None
    receipt: object
    computation_input: object

    def __post_init__(self):
        object.__setattr__(
            self, "computation_input", deep_freeze(self.computation_input)
        )


@dataclass(frozen=True)
class NativeCausalPeriod:
    period: int
    native_world_t: int
    inputs: tuple
    journal: object


@dataclass(frozen=True)
class NativeCausalSnapshot:
    native: object
    journal: object
    periods: tuple
    pending: object
    ordinary_inflight: object
    pending_decision: object
    failed: bool
    contract_hash: str
    rule: str = RULE


class NativeCausalInformationLimit(RuntimeError):
    pass


class NativeCausalInformationMarketEngine(ReentryStageInformationMarketEngine):
    """One inherited complete step; no copied choice or lifecycle algorithm."""

    def __init__(
        self,
        *args,
        causal_fields,
        causal_known_types,
        causal_deliveries,
        causal_definition,
        causal_unobserved_definition,
        causal_row_limit=2000000,
        **kwargs,
    ):
        self.causal_fields = tuple(causal_fields)
        self.causal_known_types = tuple(causal_known_types)
        self.causal_deliveries = tuple(causal_deliveries)
        if type(causal_row_limit) is not int or causal_row_limit < 1:
            raise ValueError("positive all-source structural limit required")
        _text(causal_definition)
        _text(causal_unobserved_definition)
        self.causal_definition = causal_definition
        self.causal_unobserved_definition = causal_unobserved_definition
        self.causal_row_limit = causal_row_limit
        if any(type(x) is not DisclosureField for x in self.causal_fields):
            raise TypeError("typed disclosure schema required")
        fields = {x.name: x for x in self.causal_fields}
        if len(fields) != len(self.causal_fields) or not fields:
            raise ValueError("nonempty distinct disclosure schema required")
        if any(type(x) is not DeclaredKnownType for x in self.causal_known_types):
            raise TypeError("explicit typed known or unknown declarations required")
        self._causal_types = {x.person_key: x for x in self.causal_known_types}
        if len(self._causal_types) != len(self.causal_known_types):
            raise ValueError("one persistent type declaration per key required")
        ids = set()
        for row in self.causal_deliveries:
            if type(row) is not DeclaredInputDelivery:
                raise TypeError("explicit typed input delivery required")
            if row.delivery_id in ids or row.field not in fields:
                raise ValueError("distinct deliveries and declared fields required")
            if (
                row.value is not None
                and type(row.value) is not KINDS[fields[row.field].kind]
            ):
                raise TypeError("exact declared scalar kind required")
            ids.add(row.delivery_id)
        self._causal_journal = None
        self._causal_periods = []
        self._causal_pending = None
        self._causal_inflight = None
        self._causal_decision = None
        self._causal_failed = False
        super().__init__(*args, **kwargs)
        for row in (*self.causal_known_types, *self.causal_deliveries):
            key = row.person_key
            if key[0] >= self.compiled.spec.horizon.n_periods or key[2] >= len(
                self.compiled.spec.markets
            ):
                raise ValueError("programme key outside supplied model")
            if (
                type(row) is DeclaredInputDelivery
                and not key[0] <= row.period < self.compiled.spec.horizon.n_periods
            ):
                raise ValueError("delivery schedule outside registered booking support")
        self._causal_contract_hash = hashlib.sha256(
            canonical_json(self._programme()).encode()
        ).hexdigest()

    def _programme(self):
        return _stage_encode(
            jsonable(
                {
                    "rule": RULE,
                    "spec_hash": self.compiled.spec_hash,
                    "fields": self.causal_fields,
                    "known_types": self.causal_known_types,
                    "deliveries": self.causal_deliveries,
                    "definition": self.causal_definition,
                    "unobserved_definition": self.causal_unobserved_definition,
                    "row_limit": self.causal_row_limit,
                }
            )
        )

    def reset(self, seed=None, *, tape=None):
        journal = causal_information_journal(
            self.compiled,
            self.demand_scenario.clock,
            self.causal_fields,
            definition=self.causal_definition,
            unobserved_definition=self.causal_unobserved_definition,
            row_limit=self.causal_row_limit,
        )
        result = super().reset(seed, tape=tape)
        self._causal_journal = journal
        self._causal_periods = []
        self._causal_pending = None
        self._causal_inflight = self._causal_decision = None
        self._causal_failed = True
        self._check_causal_limit()
        self._causal_failed = False
        return result

    def _payload(self):
        return {
            "plan_id": "A.10/A.5/A.14/B0/V9",
            "rule": RULE,
            "programme": self._programme(),
            "parent_source": self._stage_payload(),
            "journal": None
            if self._causal_journal is None
            else causal_information_record(self._causal_journal),
            "periods": _stage_encode(jsonable(self._causal_periods)),
            "pending": _stage_encode(jsonable(self._causal_pending)),
            "failed": self._causal_failed,
            "native_parent_steps_completed": len(self._causal_periods),
            "native_decision_input_rows_captured": any(
                x.inputs for x in self._causal_periods
            ),
            "declared_message_schedule_is_actual_external_reception": False,
            "known_type_physical_law_preferences_or_lifecycle_CS_authenticated": False,
            "lifecycle_utility_CS_service_payoff": None,
            "scientific_budget": None,
        }

    def _check_causal_limit(self, proposed=None):
        if (
            _units(
                {
                    "source": self._payload(),
                    "proposed": _stage_encode(jsonable(proposed)),
                }
            )
            > self.causal_row_limit
        ):
            raise NativeCausalInformationLimit(
                "ALL parent/programme/journal/output exceeds structural limit; no partial receipt"
            )

    def _emit(self, key, period, payload):
        journal = self._causal_journal
        anchor = self.demand_scenario.clock.at(period)
        ordinal = (
            journal.events[-1].position.ordinal + 1
            if journal.events and journal.events[-1].position.anchor_utc == anchor
            else 0
        )
        event = CausalEvent(
            f"native-source-{len(journal.events)}",
            CausalPosition(anchor, ordinal),
            key,
            payload,
        )
        proposed = append_causal_events(journal, (event,))
        self._check_causal_limit(proposed)
        self._causal_journal = proposed

    def _input(self, key, period, stage, index, root, computation):
        for row in self.causal_deliveries:
            if (row.person_key, row.period, row.stage) == (key, period, stage):
                self._emit(
                    key,
                    period,
                    MessageSource(
                        row.delivery_id, row.field, row.value, row.source_ref
                    ),
                )
                self._emit(key, period, MessageRecorded(row.delivery_id))
                self._emit(key, period, MessageReceived(row.delivery_id))
        decision = f"native-input-{len(self._causal_journal.events)}"
        attempt = None if stage == "ordinary" else f"native-attempt-{root}-{period}"
        self._emit(
            key,
            period,
            DecisionInput(
                decision,
                stage,
                attempt,
                None if stage == "ordinary" else self.rebooking.interline,
                "genuine-native-input-hook",
            ),
        )
        receipt = causal_decision_receipt(self._causal_journal, decision)
        row = NativeCausalInput(stage, index, key, root, receipt, computation)
        self._check_causal_limit(row)
        self._causal_pending.append(row)
        self._causal_decision = (key, period, decision)

    def _result(self, product):
        key, period, decision = self._causal_decision
        name = None if product is None else self.compiled.spec.products[product].id
        self._emit(
            key,
            period,
            DecisionResult(
                decision, product is None, name, "genuine-native-choice-result"
            ),
        )
        self._causal_decision = None

    def _track_requests(self, requests):
        for request in super()._track_requests(requests):
            key = (self.world.t, request.gumbel_row, request.market)
            declaration = self._causal_types.get(key)
            if declaration is None:
                raise ValueError(
                    "genuinely arrived key needs explicit known or unknown type declaration"
                )
            self._emit(
                key,
                self.world.t,
                RegisterPerson(
                    declaration.value, declaration.definition, declaration.source_ref
                ),
            )
            index = len(self.world.request_log)
            self._causal_inflight = (key, index)
            yield request
            if (
                len(self.world.request_log) != index + 1
                or self._causal_decision is None
            ):
                raise RuntimeError(
                    "one genuine ordinary input and native result required"
                )
            self._result(self.world.request_log[index]["chosen"])
            self._causal_inflight = None

    def _make_information(self, native_menu):
        value = super()._make_information(native_menu)
        key, index, view = value
        if self._causal_inflight != (key, index):
            raise RuntimeError("same genuine ordinary request hook required")
        self._input(
            key,
            self.world.t,
            "ordinary",
            len(self._pending_information),
            None,
            _stage_encode(jsonable(view)),
        )
        return value

    def _capture_stage_input(self, root, period, market, stage, route, prices, active):
        super()._capture_stage_input(root, period, market, stage, route, prices, active)
        row = self._pending_stage_inputs[-1]
        self._input(
            row.analyst_original_key,
            period,
            stage,
            len(self._pending_stage_inputs) - 1,
            root,
            row.view,
        )

    def _capture_stage_result(self, root, period, columns, selected):
        super()._capture_stage_result(root, period, columns, selected)
        self._result(self._pending_stage_results[-1].selected_product)

    def _record_stranded(self, bookings, period, cause):
        bookings = tuple(bookings)
        super()._record_stranded(bookings, period, cause)
        for booking in bookings:
            root = self._replacement_roots.get(booking.record_id, booking.record_id)
            key = self._own_keys[root]
            event_id = f"native-cancel-{len(self._causal_journal.events)}"
            self._emit(
                key,
                period,
                OwnAction(
                    event_id,
                    "cancellation",
                    str(cause),
                    "genuine-native-stranding-callback",
                ),
            )

    def present_cancelled_itinerary(self, booking, *, airport_counter=False):
        self._require_causal_complete()
        result = super().present_cancelled_itinerary(
            booking, airport_counter=airport_counter
        )
        self._causal_failed = True
        row = self._proof_calls[-1]
        self._emit(
            row.analyst_original_key,
            row.fact.native_period,
            OwnAction(
                f"native-present-{len(self._causal_journal.events)}",
                "presentation",
                "airport_counter" if airport_counter else "itinerary",
                "genuine-native-presentation-call",
            ),
        )
        self._causal_failed = False
        return result

    def _finish_causal_period(self, period):
        stages = self._stage_periods[-1].inputs
        ordinary = self._information_periods[-1].rows
        expected = [
            (x.view["stage"], i, x.analyst_original_key, x.analyst_root)
            for i, x in enumerate(stages)
        ]
        expected += [
            ("ordinary", i, x.analyst_person.key, None) for i, x in enumerate(ordinary)
        ]
        actual = [
            (x.kind, x.parent_input_index, x.analyst_original_key, x.analyst_root)
            for x in self._causal_pending
        ]
        if (
            actual != expected
            or self._causal_inflight is not None
            or self._causal_decision is not None
        ):
            raise RuntimeError(
                "ALL actual parent inputs/results required in native source order"
            )
        candidate = NativeCausalPeriod(
            period, self.world.t, tuple(self._causal_pending), self._causal_journal
        )
        if self.world.t != period + 1:
            raise RuntimeError("complete inherited native closure required")
        self._check_causal_limit(candidate)
        self._causal_periods.append(candidate)
        self._causal_pending = None

    def step(self, actions=None, *, _requests=None, _choices=None):
        self._require_causal_complete()
        if _requests is not None or _choices is not None:
            raise ValueError("genuine native requests and choices required")
        if self.world is None or self.terminal():
            raise ValueError("initialized nonterminal opportunity required")
        self._check_causal_limit()
        period = self.world.t
        self._causal_pending = []
        try:
            result = super().step(actions)
            self._finish_causal_period(period)
        except Exception:
            self._causal_failed = True
            raise
        return result

    def _require_causal_complete(self):
        if (
            self._causal_journal is None
            or self._causal_pending is not None
            or self._causal_failed
        ):
            raise ValueError(
                "incomplete causal capture; no positive query/continuation/retry"
            )

    def causal_periods(self):
        self._require_causal_complete()
        return tuple(self._causal_periods)

    def snapshot(self):
        return NativeCausalSnapshot(
            super().snapshot(),
            self._causal_journal,
            tuple(self._causal_periods),
            None if self._causal_pending is None else tuple(self._causal_pending),
            self._causal_inflight,
            self._causal_decision,
            self._causal_failed,
            self._causal_contract_hash,
        )

    def restore(self, snapshot):
        if type(snapshot) is not NativeCausalSnapshot:
            raise TypeError("combined causal/native snapshot required")
        if (snapshot.rule, snapshot.contract_hash) != (
            RULE,
            self._causal_contract_hash,
        ):
            raise ValueError("full causal programme/spec/schema/limit differs")
        super().restore(snapshot.native)
        self._causal_journal = snapshot.journal
        self._causal_periods = list(snapshot.periods)
        self._causal_pending = (
            None if snapshot.pending is None else list(snapshot.pending)
        )
        self._causal_inflight = snapshot.ordinary_inflight
        self._causal_decision = snapshot.pending_decision
        self._causal_failed = snapshot.failed

    def branch(self):
        other = type(self)(
            self.demand_scenario,
            {i: copy.deepcopy(c) for i, c in self.controllers.items()},
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


def native_causal_information_record(engine):
    if type(engine) is not NativeCausalInformationMarketEngine:
        raise TypeError("prospective native causal source required")
    engine.causal_periods()
    engine._check_causal_limit()
    return engine._payload()
