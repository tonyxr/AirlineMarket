"""A.10/A.5/A.14/B0/V9: prospective final native re-entry stage inputs.

NEW versioned executor supplies same-call hooks; frozen parents stay unchanged.
Choice inputs are ANALYST records, not actual displayed/received knowledge.
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction

from airline_market.core.hashing import jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.sim.passenger_accounts import _encode
from airline_market.sim.proof_information import ProofInformationMarketEngine
from airline_market.sim.received_information import _units
from airline_market.sim.reentry_stage_executor import _StageReentryExecutor

RULE = "same-call-final-native-reentry-stage-inputs-v1"


def _stage_encode(value):
    """Lossless owned contract scalars; native binary64 encoding unchanged."""
    if isinstance(value, Mapping):
        return {k: _stage_encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_stage_encode(v) for v in value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("aware contract UTC instant required")
        return {"aware_utc": value.astimezone(timezone.utc).isoformat()}
    if isinstance(value, Decimal):
        return {"decimal": str(value)}
    if isinstance(value, Fraction):
        return {
            "numerator": str(value.numerator),
            "denominator": str(value.denominator),
        }
    return _encode(value)


class StageInformationLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class ReentryStageInput:
    analyst_root: int
    analyst_original_key: tuple
    period: int
    stage_ordinal: int
    route: str
    prices: tuple
    offered: tuple
    view: object

    def __post_init__(self):
        object.__setattr__(self, "view", deep_freeze(self.view))
        object.__setattr__(self, "prices", deep_freeze(self.prices))
        object.__setattr__(self, "offered", deep_freeze(self.offered))
        object.__setattr__(
            self, "analyst_original_key", deep_freeze(self.analyst_original_key)
        )


@dataclass(frozen=True)
class ReentryStageResult:
    input_index: int
    selected_product: int | None
    outside: bool


@dataclass(frozen=True)
class ReentryStagePeriod:
    period: int
    inputs: tuple
    results: tuple


@dataclass(frozen=True)
class ReentryStageSnapshot:
    native: object
    periods: tuple
    pending_inputs: object
    pending_results: object
    row_limit: int
    spec_hash: str
    rule: str = RULE


class ReentryStageInformationMarketEngine(
    ProofInformationMarketEngine, _StageReentryExecutor
):
    """ONE inherited dated step, versioned native re-entry bodies, owned hooks.

    Hook sees final arrays AFTER marks generation; no taste/Gumbels inspected.
    Full parent scenarios/identities stay analyst. No observation/default change.
    """

    def __init__(self, *args, stage_row_limit=1000000, **kwargs):
        if type(stage_row_limit) is not int or stage_row_limit < 1:
            raise ValueError("positive full-source structural limit required")
        self.stage_row_limit = stage_row_limit
        self._stage_periods = []
        self._pending_stage_inputs = None
        self._pending_stage_results = None
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, *, tape=None):
        self._stage_periods = []
        self._pending_stage_inputs = None
        self._pending_stage_results = None
        return super().reset(seed, tape=tape)

    def _stage_payload(self):
        return {
            "plan_id": "A.10/A.5/A.14/B0/V9",
            "rule": RULE,
            "row_limit": self.stage_row_limit,
            "source_context": _stage_encode(
                jsonable(
                    {
                        "native_spec": self.compiled.spec,
                        "spec_hash": self.compiled.spec_hash,
                        "scenario": self.demand_scenario.record(),
                        "dated_rebooking": self.dated_rebooking,
                        "dated_menu": self.dated_menu,
                        "dated_reserve": self.dated_reserve,
                        "cap_policy": self.cap_policy,
                        "trigger_rule": self.trigger.rule if self.trigger else None,
                        "mechanical_benchmark": self.trigger.benchmark
                        if self.trigger
                        else None,
                        "ordinary_information": self._information_periods,
                        "own_information": self._own_periods,
                        "proof_information": self._proof_periods,
                        "presentation_calls": self._proof_calls,
                        "final_own_histories": self._own_histories,
                        "own_tickets": self._own_tickets,
                        "own_keys": self._own_keys,
                    }
                )
            ),
            "periods": _encode(jsonable(self._stage_periods)),
            "pending_inputs": _encode(jsonable(self._pending_stage_inputs)),
            "pending_results": _encode(jsonable(self._pending_stage_results)),
            "actual_received_information_type_probability_or_utility_authenticated": False,
            "native_choice_law_actor_observations_metrics_defaults_changed": False,
            "native_reentry_stage_inputs_captured": any(
                p.inputs for p in self._stage_periods
            ),
            "executor_revision_scope": "opt-in source path; stage reach retained in complete inputs/results",
            "lifecycle_utility_CS_or_payoff": None,
            "scientific_budget": None,
        }

    def _check_stage_limit(self, proposed=None):
        payload = self._stage_payload()
        if proposed is not None:
            payload["proposed_complete_period"] = _encode(jsonable(proposed))
        if _units(payload) > self.stage_row_limit:
            raise StageInformationLimit(
                "ALL source/contracts/stage output exceeds limit; no partial receipt"
            )

    def _capture_stage_input(self, root, period, market, stage, route, prices, active):
        if self._pending_stage_inputs is None or root not in self._own_keys:
            raise ValueError("genuine original-person stage capture required")
        if len(self._pending_stage_inputs) != len(self._pending_stage_results):
            raise RuntimeError("previous reached stage result is incomplete")
        previous = [
            (i, x)
            for i, x in enumerate(self._pending_stage_inputs)
            if x.analyst_root == root
        ]
        ordinal = len(previous)
        if (ordinal, stage) not in {(0, "own"), (1, "interline")}:
            raise RuntimeError(
                "only actual own then reached interline stages supported"
            )
        earlier = ()
        if previous:
            index, first = previous[0]
            result = self._pending_stage_results[index]
            if not result.outside or not self.rebooking.interline:
                raise RuntimeError(
                    "interline requires actual earlier own outside and enabled contract"
                )
            earlier = ({"stage": first.view["stage"], "outside": True},)
        c = self.compiled
        view = {
            "native_period": period,
            "compiled_opportunity_utc": self.demand_scenario.clock.at(
                period
            ).isoformat(),
            "market_id": c.spec.markets[market].id,
            "stage": stage,
            "stage_ordinal": ordinal,
            "earlier_within_attempt_results": earlier,
            "offers": tuple(
                {
                    "product_id": c.spec.products[j].id,
                    "fare": float(prices[j]),
                    "offered": bool(active[j]),
                }
                for j in c.market_products[market]
            ),
            "outside_available": True,
            "scope": "native-choice-inputs-before-stage-utility-not-received-knowledge",
        }
        self._pending_stage_inputs.append(
            ReentryStageInput(
                root,
                self._own_keys[root],
                period,
                ordinal,
                route,
                tuple(float(x) for x in prices),
                tuple(bool(x) for x in active),
                view,
            )
        )
        self._check_stage_limit()

    def _capture_stage_result(self, root, period, columns, selected):
        index = len(self._pending_stage_results)
        if index >= len(self._pending_stage_inputs):
            raise RuntimeError("result requires actual prior stage input")
        row = self._pending_stage_inputs[index]
        if (row.analyst_root, row.period) != (root, period) or not 0 <= selected < len(
            columns
        ):
            raise RuntimeError("actual same-stage result linkage differs")
        outside = selected == len(columns) - 1
        product = None if outside else int(columns[selected])
        if product is not None and not row.offered[product]:
            raise RuntimeError(
                "native selected product differs from captured final offer"
            )
        self._pending_stage_results.append(ReentryStageResult(index, product, outside))
        self._check_stage_limit()

    def _finish_stage_period(self, period, attempts):
        inputs, results = self._pending_stage_inputs, self._pending_stage_results
        if len(inputs) != len(results):
            raise RuntimeError("ALL reached stage results required")
        roots = []
        for attempt in attempts:
            root = attempt["original_record"]
            indices = [i for i, row in enumerate(inputs) if row.analyst_root == root]
            if not indices or root in roots or len(indices) not in (1, 2):
                raise RuntimeError(
                    "EVERY actual attempt requires complete reached stages"
                )
            roots.append(root)
            last = results[indices[-1]]
            if last.selected_product != attempt["chosen"]:
                raise RuntimeError("final stage and native attempt outcome differ")
            if (
                not last.outside
                and inputs[indices[-1]].view["stage"] != attempt["stage"]
            ):
                raise RuntimeError("native final purchase stage differs")
        if {r.analyst_root for r in inputs} != set(roots):
            raise RuntimeError("no extra stage-only attempts permitted")
        proposed = ReentryStagePeriod(period, tuple(inputs), tuple(results))
        self._check_stage_limit(proposed)
        self._stage_periods.append(proposed)
        self._pending_stage_inputs = None
        self._pending_stage_results = None

    def step(self, actions=None, *, _requests=None, _choices=None):
        if self._pending_stage_inputs is not None:
            raise ValueError("incomplete stage capture; no continuation or retry")
        if _requests is not None or _choices is not None:
            raise ValueError("genuine native requests and choices required")
        if self.world is None or self.terminal():
            raise ValueError("initialized nonterminal opportunity required")
        self._check_stage_limit()
        period, start = self.world.t, len(self.rebooking_history)
        self._pending_stage_inputs, self._pending_stage_results = [], []
        result = super().step(actions)
        # Keep pending state even if validation fails after native closure.
        self._finish_stage_period(period, self.rebooking_history[start:])
        return result

    def stage_periods(self):
        if self._pending_stage_inputs is not None:
            raise ValueError("incomplete stage capture; no positive partial query")
        return tuple(self._stage_periods)

    def snapshot(self):
        return ReentryStageSnapshot(
            super().snapshot(),
            tuple(self._stage_periods),
            None
            if self._pending_stage_inputs is None
            else tuple(self._pending_stage_inputs),
            None
            if self._pending_stage_results is None
            else tuple(self._pending_stage_results),
            self.stage_row_limit,
            self.compiled.spec_hash,
        )

    def restore(self, snapshot):
        if type(snapshot) is not ReentryStageSnapshot:
            raise TypeError("combined re-entry stage snapshot required")
        if (snapshot.rule, snapshot.spec_hash, snapshot.row_limit) != (
            RULE,
            self.compiled.spec_hash,
            self.stage_row_limit,
        ):
            raise ValueError("stage capture contract differs")
        super().restore(snapshot.native)
        self._stage_periods = list(snapshot.periods)
        self._pending_stage_inputs = (
            None if snapshot.pending_inputs is None else list(snapshot.pending_inputs)
        )
        self._pending_stage_results = (
            None if snapshot.pending_results is None else list(snapshot.pending_results)
        )

    def branch(self):
        other = type(self)(
            self.demand_scenario,
            {i: copy.deepcopy(c) for i, c in self.controllers.items()},
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


def reentry_stage_view(row):
    """Owned market-local computation input; no actual disclosure permission."""
    if type(row) is not ReentryStageInput:
        raise TypeError("native stage-input row required")
    return _encode(jsonable(row.view))


def reentry_stage_record(engine):
    """NEW complete live-source record; no upstream producer or saved loader."""
    if type(engine) is not ReentryStageInformationMarketEngine:
        raise TypeError("prospective re-entry stage producer required")
    engine.stage_periods()
    return engine._stage_payload()
