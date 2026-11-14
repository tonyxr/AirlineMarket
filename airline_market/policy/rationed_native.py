"""A.8/B0/V9: opt-in whole-native-period ordinary applicant priority.

The native parent opens shocks, posts once, attempts genuine re-entry first,
and consumes a lazy ordinary-request iterable. Only that iterable is reordered.
Choice/book/resolve/refunds/callbacks/closure remain native parent operations.
This explicit batching rule is not a selected empirical or primary B4 policy.
"""

import copy
from dataclasses import dataclass

from airline_market.core.rng import StreamRegistry
from airline_market.policy.cap_evidence import CapEvidenceMarketEngine, posting_record
from airline_market.policy.rationing import priority_allocation
from airline_market.policy.rationing_native import native_ordinary_pool
from airline_market.sim.arrivals import period_requests


@dataclass(frozen=True)
class WholePeriodPriority:
    """One complete native ordinary pool; re-entry first, including held sales.

    Any strict native proposal clipping activates one uniform permutation of ALL
    actual ordinary requests, including outside-preferring/unavailable cases and
    markets with no clipped offer. This is an explicit period-wide scope, not a
    claim that those other markets have binding caps or excess demand.
    """

    namespace: str

    def __post_init__(self):
        if not isinstance(self.namespace, str) or not self.namespace.strip():
            raise ValueError("explicit named ordinary-priority namespace required")


@dataclass(frozen=True)
class AdmissionRecord:
    period: int
    keys: tuple
    priority_order: tuple
    predicted_choices: tuple
    actual_choices: tuple
    request_coordinates: tuple
    free_capacity_before: tuple
    cap_posting: object
    priority_namespace: str | None
    lottery_applied: bool
    scope: str = "whole native period ordinary pool, genuine re-entry first"


@dataclass(frozen=True)
class RationedNativeSnapshot:
    native: object
    policy: object
    admission_records: tuple


class RationedNativeMarketEngine(CapEvidenceMarketEngine):
    """Explicit native period batching, no copied simulator/booking lifecycle.

    A missing policy delegates the exact native path. With a policy, forced
    request/choice hooks are refused. The current marked pool is captured lazily
    after native posting and required re-entry. Lottery order is keyed by the
    physical tape seed and period in a declared namespace, never booking IDs.
    """

    def __init__(self, *args, ordinary_priority=None, **kwargs):
        if (
            ordinary_priority is not None
            and type(ordinary_priority) is not WholePeriodPriority
        ):
            raise TypeError("WholePeriodPriority or None required")
        self.ordinary_priority = ordinary_priority
        self._admission_records = []
        super().__init__(*args, **kwargs)

    def reset(self, seed=None, *, tape=None):
        self._admission_records = []
        return super().reset(seed, tape=tape)

    def _ordered_ordinary_requests(self, pending):
        # Both enumerations reuse current fixed tape marks; neither draws RNG.
        # The first declares complete zero objective support to the frozen pool
        # adapter. Scores are unused for ordering, not claimed welfare/efficiency.
        c, t = self.compiled, self.world.t
        requests = tuple(period_requests(c, self.tape, t, self.world.exo))
        scores = {
            ((t, r.gumbel_row, r.market), c.spec.products[j].id): 0
            for r in requests
            for j in c.market_products[r.market]
        }
        pool = native_ordinary_pool(
            self,
            posted_prices=self._prices_global(),
            allocation_scores=scores,
            score_definition="zero scores unused by priority ordering; no welfare objective",
        )
        if pool.requests != requests:
            raise RuntimeError("current native marked pool changed during capture")
        evidence = self.pool_cap_evidence(pool)
        binding = evidence.any_price_constraint_binding
        allocation = priority_allocation(
            pool.problem,
            binding_cap=binding,
            registry=StreamRegistry(self.tape.seed, self.ordinary_priority.namespace)
            if binding
            else None,
        )
        by_key = {(t, r.gumbel_row, r.market): r for r in requests}
        pending.update(pool=pool, cap=evidence, allocation=allocation)
        for key in allocation["priority_order"]:
            yield by_key[key]

    def step(self, actions=None, *, _requests=None, _choices=None):
        if self.ordinary_priority is None:
            return super().step(actions, _requests=_requests, _choices=_choices)
        if _requests is not None or _choices is not None:
            raise ValueError(
                "native priority requires the full physical arrival pool and native choice"
            )
        if self.world is None or self.terminal():
            raise ValueError("initialized nonterminal native opportunity required")
        t = self.world.t
        before = len(self.world.request_log)
        pending = {}
        result = super().step(
            actions, _requests=self._ordered_ordinary_requests(pending)
        )
        pool, allocation = pending["pool"], pending["allocation"]
        logged = self.world.request_log[before:]
        actual = tuple(
            (
                key,
                None
                if row["chosen"] is None
                else self.compiled.spec.products[row["chosen"]].id,
            )
            for key, row in zip(allocation["priority_order"], logged)
        )
        if len(logged) != len(pool.requests) or actual != allocation["assignments"]:
            raise RuntimeError(
                "native ordinary choices differ from fixed-pool priority witness"
            )
        self._admission_records.append(
            AdmissionRecord(
                t,
                tuple(x.key for x in pool.problem.applicants),
                allocation["priority_order"],
                allocation["assignments"],
                actual,
                tuple(
                    (
                        r.rank,
                        r.market,
                        r.segment,
                        r.taste_class,
                        r.gumbel_row,
                        r.mixed_row,
                    )
                    for r in pool.requests
                ),
                pool.problem.capacities,
                pending["cap"],
                allocation["priority_namespace"],
                pending["cap"].any_price_constraint_binding,
            )
        )
        return result

    def admission_records(self):
        return tuple(self._admission_records)

    def snapshot(self):
        return RationedNativeSnapshot(
            super().snapshot(), self.ordinary_priority, tuple(self._admission_records)
        )

    def restore(self, snapshot):
        if type(snapshot) is not RationedNativeSnapshot:
            raise TypeError("native priority combined snapshot required")
        if snapshot.policy != self.ordinary_priority:
            raise ValueError("snapshot ordinary-priority contract differs")
        super().restore(snapshot.native)
        self._admission_records = list(snapshot.admission_records)

    def branch(self):
        other = RationedNativeMarketEngine(
            self.compiled,
            {i: copy.deepcopy(ctrl) for i, ctrl in self.controllers.items()},
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


def admission_record(row):
    """Analyst-only native admission receipt; not primary B4 welfare/denial."""
    if type(row) is not AdmissionRecord:
        raise TypeError("native AdmissionRecord required")
    return {
        "plan_id": "A.8/B0/V9",
        **vars(row),
        "cap_posting": posting_record(row.cap_posting),
        "lottery_group": "all actual ordinary requests in this native period",
        "reentry_priority": "native genuine re-entry before ordinary pool; held-first semantics unchanged",
        "primary_B4_policy_adopted": False,
        "excess_demand_measured": False,
        "efficient_assignment_adopted": False,
    }
