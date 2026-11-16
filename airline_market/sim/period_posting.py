"""A.10/A.5/A.14/B0/V9/B4: complete native period posting receipts.

Regular computation fares and compiled source order are analyst evidence, not
received knowledge, bookability, native probability, preferences or lifecycle CS.
"""

import copy
import hashlib
import json
from dataclasses import dataclass

from airline_market.core.hashing import jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.policy.cap_evidence import posting_record
from airline_market.sim.native_causal_information import (
    RULE as PARENT_RULE,
)
from airline_market.sim.native_causal_information import (
    NativeCausalInformationMarketEngine,
)
from airline_market.sim.native_stage_contexts import _fare
from airline_market.sim.received_information import _units
from airline_market.sim.reentry_stage_information import _stage_encode

RULE = "complete-native-period-posting-source-v1"


def _encoded(value):
    return _stage_encode(jsonable(value))


def _same(a, b):
    return json.dumps(_encoded(a), sort_keys=True) == json.dumps(
        _encoded(b), sort_keys=True
    )


def _digest(value):
    return hashlib.sha256(
        json.dumps(_encoded(value), sort_keys=True).encode()
    ).hexdigest()


class PeriodPostingLimit(RuntimeError):
    """All-support infrastructure limit; no partial result or automatic retry."""


@dataclass(frozen=True)
class PeriodPostingSource:
    body: object
    digest: str
    rule: str = RULE

    def __post_init__(self):
        object.__setattr__(self, "body", deep_freeze(self.body))

    def export(self):
        return period_posting_record(self)


@dataclass(frozen=True)
class PeriodPostingSnapshot:
    native: object
    periods: tuple
    pending: object
    calls: int
    failed: bool
    contract_hash: str
    rule: str = RULE


def _bound(parent_limit, calls, horizon, products, contract):
    # Parent's full encoded source is bounded by parent_limit at each successful
    # closure. Each saved posting may repeat ONE full earlier journal. Links are
    # a fixed-size projection of parent inputs; all revisions count, even unused.
    # Input + proposed full detached export are both included. This counts scalar
    # structure, never bytes, rational bits, RSS, snapshots or scientific cost.
    return 2 * (
        (calls + 16) * parent_limit
        + calls * (256 + 128 * products)
        + 256 * horizon
        + _units(contract)
        + 512
    )


def period_posting_record(source):
    """Pure owned reader; current hashes are not producer authentication."""
    if type(source) is not PeriodPostingSource or source.rule != RULE:
        raise TypeError("current typed complete-period source required")
    body = source.body
    if source.digest != _digest(body) or body["rule"] != RULE:
        raise ValueError("complete stored source integrity differs")
    contract = body["contract"]
    required = _bound(
        contract["parent_limit"],
        contract["posting_call_limit"],
        contract["horizon"],
        len(contract["products"]),
        contract,
    )
    if body["required_support"] != required or required > contract["support_limit"]:
        raise PeriodPostingLimit(
            "complete recomputed input/output support exceeds limit"
        )
    if _units(body) * 2 > required:
        raise PeriodPostingLimit(
            "actual complete export exceeds conservative support bound"
        )
    parent = body["parent_source"]
    periods = body["periods"]
    if (
        parent["rule"] != PARENT_RULE
        or parent["failed"]
        or parent["pending"] is not None
        or parent["native_parent_steps_completed"] != len(periods)
        or len(parent["periods"]) != len(periods)
        or parent["programme"]["row_limit"] != contract["parent_limit"]
        or body["native_probability_received_knowledge_preferences_or_CS_authenticated"]
        is not False
        or body["lifecycle_utility_CS"] is not None
        or body["scientific_budget"] is not None
    ):
        raise ValueError("complete successful parent and unadopted scope required")
    context = parent["parent_source"]["source_context"]
    if (
        tuple(p["id"] for p in context["native_spec"]["products"])
        != contract["products"]
        or context["native_spec"]["horizon"]["n_periods"] != contract["horizon"]
        or _digest(parent["programme"]) != contract["causal_programme_hash"]
    ):
        raise ValueError("same complete parent model/programme required")
    final = parent["journal"]["analyst_context"]["events"]
    calls = 0
    for t, (row, native) in enumerate(zip(periods, parent["periods"])):
        if (
            row["period"] != t
            or row["native_closed_world_t"] != t + 1
            or native["period"] != t
            or native["native_world_t"] != t + 1
            or row["compiled_opportunity_utc"] != contract["clock"][t]
            or not row["postings"]
            or len(row["input_links"]) != len(native["inputs"])
        ):
            raise ValueError(
                "ALL consecutive periods, empty support and full input links required"
            )
        for revision, post in enumerate(row["postings"], 1):
            calls += 1
            cap = post["native_cap_receipt"]
            prefix = post["earlier_global_journal"]["events"]
            for product in cap["products"]:
                for name in (
                    "native_proposal",
                    "after_static",
                    "executed",
                    "static_ceiling",
                    "trigger_ceiling",
                ):
                    if product[name] is not None:
                        _fare(product[name])
            for price in post["prices"]:
                _fare(price)
            if (
                cap["period"] != t
                or cap["revision"] != revision
                or post["native_world_t_at_post"] != t
                or post["earlier_global_event_count"] != len(prefix)
                or not _same(prefix, final[: len(prefix)])
                or tuple(p["product"] for p in cap["products"]) != contract["products"]
                or not _same(
                    tuple(p["executed"] for p in cap["products"]), post["prices"]
                )
            ):
                raise ValueError(
                    "same-call full ordered products/revisions/earlier global source required"
                )
        for link, inp in zip(row["input_links"], native["inputs"]):
            prefix = inp["receipt"]["source_journal"]["events"]
            earlier = row["postings"][-1]["earlier_global_journal"]["events"]
            if (
                link["posting_revision"] != len(row["postings"])
                or link["kind"] != inp["kind"]
                or link["parent_input_index"] != inp["parent_input_index"]
                or not _same(link["original_key"], inp["analyst_original_key"])
                or link["root"] != inp["analyst_root"]
                or link["global_input_event_count"] != len(prefix)
                or not _same(prefix[: len(earlier)], earlier)
                or not _same(prefix, final[: len(prefix)])
            ):
                raise ValueError(
                    "exact input-ending GLOBAL prefix and final regular posting required"
                )
    if calls != body["posting_calls"] or calls > contract["posting_call_limit"]:
        raise ValueError("every actual revision required, no omitted or extra posting")
    return _encoded(body)


class PeriodPostingMarketEngine(NativeCausalInformationMarketEngine):
    """ONE inherited native step; only post-return/input/close observations added.

    Explicit standalone native reposts are retained as revisions, pending until
    full period closure. The last regular posting precedes ALL genuine inputs.
    """

    def __init__(
        self, *args, posting_call_limit=1000, posting_support_limit=10**10, **kwargs
    ):
        if (
            type(posting_call_limit) is not int
            or posting_call_limit < 1
            or type(posting_support_limit) is not int
            or posting_support_limit < 1
        ):
            raise ValueError(
                "positive explicit all-call and structural limits required"
            )
        self.posting_call_limit = posting_call_limit
        self.posting_support_limit = posting_support_limit
        self._posting_periods = []
        self._posting_pending = None
        self._posting_calls = 0
        self._posting_failed = self._posting_inside = False
        super().__init__(*args, **kwargs)
        self._posting_contract = deep_freeze(
            {
                "rule": RULE,
                "parent_limit": self.causal_row_limit,
                "posting_call_limit": posting_call_limit,
                "support_limit": posting_support_limit,
                "products": tuple(p.id for p in self.compiled.spec.products),
                "horizon": self.compiled.spec.horizon.n_periods,
                "clock": tuple(
                    t.isoformat() for t in self.demand_scenario.clock.boundaries
                ),
                "causal_programme_hash": _digest(self._programme()),
            }
        )
        self._posting_contract_hash = _digest(self._posting_contract)
        self._check_posting_bound()

    def _check_posting_bound(self):
        c = self._posting_contract
        required = _bound(
            c["parent_limit"],
            c["posting_call_limit"],
            c["horizon"],
            len(c["products"]),
            c,
        )
        if required > self.posting_support_limit:
            raise PeriodPostingLimit(
                "complete proposed all-period/revision export exceeds limit before native forward"
            )
        return required

    def _posting_ready(self):
        if self._posting_failed or self._posting_inside:
            raise ValueError(
                "failed/inflight posting source; no query, continuation or retry"
            )
        self._require_causal_complete()

    def reset(self, seed=None, *, tape=None):
        self._check_posting_bound()
        self._posting_periods = []
        self._posting_pending = None
        self._posting_calls = 0
        self._posting_inside = False
        self._posting_failed = True
        result = super().reset(seed, tape=tape)
        self._posting_failed = False
        return result

    def _open_posting_period(self):
        if self._posting_pending is None:
            self._posting_pending = {
                "period": self.world.t,
                "compiled_opportunity_utc": self.demand_scenario.clock.at(
                    self.world.t
                ).isoformat(),
                "postings": [],
                "input_links": [],
            }

    def begin_period(self, actions=None):
        # A standalone native leader opening is pending, never a closed source.
        if not self._posting_inside:
            self._posting_ready()
            self._check_posting_bound()
            if self.world is None or self.terminal():
                raise ValueError("initialized nonterminal native opening required")
            self._open_posting_period()
        try:
            return super().begin_period(actions)
        except Exception:
            self._posting_failed = True
            raise

    def post_actions(self, actions=None):
        if not self._posting_inside:
            self._posting_ready()
        if self.world is None or self.terminal():
            raise ValueError("initialized nonterminal native posting required")
        self._check_posting_bound()
        if self._posting_calls >= self.posting_call_limit:
            self._posting_failed = True
            raise PeriodPostingLimit(
                "ALL actual posting calls exhausted before next forward"
            )
        self._open_posting_period()
        if self._posting_pending["input_links"]:
            raise ValueError("no repost after a genuine input")
        self._posting_calls += 1
        before = len(self.cap_postings())
        try:
            prices = super().post_actions(actions)
            actual = self.cap_postings()
            if len(actual) != before + 1 or actual[-1].period != self.world.t:
                raise RuntimeError("one actual same-call cap receipt required")
            if not _same(
                tuple(r.executed for r in actual[-1].products),
                tuple(float(p) for p in prices),
            ):
                raise RuntimeError("same actual returned regular price vector required")
            self._capture_posting(prices, actual[-1])
        except Exception:
            self._posting_failed = True
            raise
        return prices

    def _capture_posting(self, prices, cap):
        self._posting_pending["postings"].append(
            deep_freeze(
                _encoded(
                    {
                        "native_world_t_at_post": self.world.t,
                        "prices": tuple(float(x) for x in prices),
                        "native_cap_receipt": posting_record(cap),
                        "earlier_global_journal": self._causal_journal,
                        "earlier_global_event_count": len(self._causal_journal.events),
                    }
                )
            )
        )

    def _input(self, key, period, stage, index, root, computation):
        pending = self._posting_pending
        if pending is None or pending["period"] != period or not pending["postings"]:
            raise ValueError("actual final regular posting must precede genuine input")
        self._check_posting_bound()
        super()._input(key, period, stage, index, root, computation)
        row = self._causal_pending[-1]
        pending["input_links"].append(
            deep_freeze(
                {
                    "posting_revision": len(pending["postings"]),
                    "kind": stage,
                    "parent_input_index": index,
                    "original_key": key,
                    "root": root,
                    "global_input_event_count": len(row.receipt.source_journal.events),
                }
            )
        )

    def _finish_posting_period(self, period):
        row = copy.deepcopy(self._posting_pending)
        if row is None or self.world.t != period + 1:
            raise RuntimeError("full inherited native period closure required")
        row["native_closed_world_t"] = self.world.t
        if not _same(row["postings"][-1]["prices"], self._trace[-1]["prices"]):
            raise RuntimeError(
                "native closing trace must use same final regular posting"
            )
        self._posting_periods.append(deep_freeze(row))
        self._posting_pending = None

    def step(self, actions=None, *, _requests=None, _choices=None):
        self._posting_ready()
        if _requests is not None or _choices is not None:
            raise ValueError("genuine inherited requests and choices only")
        if self.world is None or self.terminal():
            raise ValueError("initialized nonterminal period required")
        self._check_posting_bound()
        period = self.world.t
        self._posting_inside = True
        try:
            result = super().step(actions)
            self._finish_posting_period(period)
        except Exception:
            self._posting_failed = True
            raise
        finally:
            self._posting_inside = False
        return result

    def posting_source(self):
        self._posting_ready()
        if self._posting_pending is not None:
            raise ValueError(
                "pending standalone postings have no complete period receipt"
            )
        body = {
            "rule": RULE,
            "contract": self._posting_contract,
            "required_support": self._check_posting_bound(),
            "parent_source": self._payload(),
            "periods": tuple(self._posting_periods),
            "posting_calls": self._posting_calls,
            "native_probability_received_knowledge_preferences_or_CS_authenticated": False,
            "lifecycle_utility_CS": None,
            "scientific_budget": None,
        }
        value = PeriodPostingSource(_encoded(body), _digest(body))
        period_posting_record(value)
        return value

    def snapshot(self):
        if self._posting_inside:
            raise ValueError(
                "combined snapshots require a boundary between native calls"
            )
        return PeriodPostingSnapshot(
            super().snapshot(),
            tuple(self._posting_periods),
            copy.deepcopy(self._posting_pending),
            self._posting_calls,
            self._posting_failed,
            self._posting_contract_hash,
        )

    def restore(self, snapshot):
        if type(snapshot) is not PeriodPostingSnapshot or snapshot.rule != RULE:
            raise TypeError("combined posting/native snapshot required")
        if snapshot.contract_hash != self._posting_contract_hash:
            raise ValueError("full posting programme/rule/limits differ")
        if self._posting_inside:
            raise ValueError("cannot restore inside a native forward")
        super().restore(snapshot.native)
        self._posting_periods = list(snapshot.periods)
        self._posting_pending = copy.deepcopy(snapshot.pending)
        self._posting_calls = snapshot.calls
        self._posting_failed = snapshot.failed

    def branch(self):
        other = type(self)(
            self.demand_scenario,
            {i: copy.deepcopy(c) for i, c in self.controllers.items()},
            posting_call_limit=self.posting_call_limit,
            posting_support_limit=self.posting_support_limit,
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
