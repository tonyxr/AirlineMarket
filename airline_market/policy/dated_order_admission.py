"""A.8/A.5/B0/V9: opt-in dated native-order benchmark, not a B4 policy.

Only full ordinary request order changes under actual strict proposal clipping.
The inherited dated world owns posting/re-entry/choice/booking/lineage/closure.
"""

import copy
from dataclasses import dataclass

from airline_market.core.hashing import jsonable
from airline_market.policy.allocation_comparison import AllocationObjective
from airline_market.policy.cap_evidence import posting_record
from airline_market.policy.dated_demand import DatedDemandMarketEngine
from airline_market.policy.native_order_bound import (
    exact_native_order_bound,
    native_order_allocation,
)
from airline_market.policy.rationing import result_record
from airline_market.policy.rationing_native import (
    assert_pool_current,
    native_ordinary_pool,
    native_pool_record,
)
from airline_market.sim.arrivals import period_requests


@dataclass(frozen=True)
class EfficientNativeOrder:
    """Explicit private-current-mark benchmark objective and infrastructure cap.

    This is best *native-choice order*, not the broader IR assignment relaxation,
    a publicly observable/truthful mechanism, lifecycle surplus or equilibrium.
    """

    objective: AllocationObjective
    state_limit: int

    def __post_init__(self):
        if type(self.objective) is not AllocationObjective:
            raise TypeError("explicit AllocationObjective required")
        if type(self.state_limit) is not int or self.state_limit < 1:
            raise ValueError(
                "explicit positive order infrastructure state limit required"
            )


@dataclass(frozen=True)
class DatedOrderAdmission:
    pool: object
    cap_posting: object
    selection: object
    actual_choices: tuple
    optimal_order_applied: bool
    policy: EfficientNativeOrder


@dataclass(frozen=True)
class DatedOrderSnapshot:
    native: object
    policy: EfficientNativeOrder | None
    records: tuple
    pending: object


class DatedOrderBenchmarkEngine(DatedDemandMarketEngine):
    """Genuine inherited dated execution with a lazy ordinary-order source.

    No forced request/choice hooks, random-priority draw or copied step lifecycle.
    A solve limit propagates after posting/re-entry but before any ordinary sale;
    no chronology fallback, partial receipt, rollback or automatic retry.
    """

    def __init__(self, scenario, controllers=None, *, order_policy=None, **kwargs):
        if order_policy is not None and type(order_policy) is not EfficientNativeOrder:
            raise TypeError("EfficientNativeOrder or None required")
        if kwargs.get("ordinary_priority") is not None:
            raise ValueError(
                "order benchmark and random ordinary priority are separate contracts"
            )
        kwargs["ordinary_priority"] = None
        if scenario.physical_shocks and kwargs.get("dated_rebooking") is None:
            raise ValueError(
                "dated physical order benchmark requires explicit stranded re-entry"
            )
        if (
            order_policy is not None
            and order_policy.objective.mode == "product_scores"
            and {p for p, value in order_policy.objective.product_values}
            != {p.id for p in scenario.compiled.spec.products}
        ):
            raise ValueError("fixed score table must cover EVERY physical product")
        self.order_policy = order_policy
        self._order_records = []
        self._pending_order = None
        super().__init__(scenario, controllers, **kwargs)

    def reset(self, seed=None, *, tape=None):
        self._order_records = []
        self._pending_order = None
        return super().reset(seed, tape=tape)

    def _native_current_requests(self):
        # Consumed lazily by PotentialLineage's existing _track_requests wrapper:
        # after posting/re-entry, before ordinary booking; track resumes before
        # native end-period cancellation/closure. No lineage hook is bypassed.
        if self.order_policy is None:
            yield from super()._native_current_requests()
            return
        c, t = self.compiled, self.world.t
        requests = tuple(period_requests(c, self.tape, t, self.world.exo))
        support = {
            ((t, r.gumbel_row, r.market), c.spec.products[j].id): 0
            for r in requests
            for j in c.market_products[r.market]
        }
        pool = native_ordinary_pool(
            self,
            posted_prices=self._prices_global(),
            allocation_scores=support,
            score_definition="complete temporary zero support, replaced by explicit objective",
        )
        if pool.requests != requests:
            raise RuntimeError(
                "genuine current request pool changed before order selection"
            )
        pool = self.order_policy.objective.scored_pool(pool)
        assert_pool_current(self, pool)
        cap = self.pool_cap_evidence(pool)
        applied = cap.any_price_constraint_binding
        selection = (
            exact_native_order_bound(
                pool.problem, state_limit=self.order_policy.state_limit
            )
            if applied
            else native_order_allocation(
                pool.problem, tuple(a.key for a in pool.problem.applicants)
            )
        )
        self._pending_order = (pool, cap, selection, applied)
        lookup = {(t, r.gumbel_row, r.market): r for r in requests}
        for key in selection["priority_order"]:
            yield lookup[key]

    def step(self, actions=None, *, _requests=None, _choices=None):
        if self.order_policy is None:
            return super().step(actions, _requests=_requests, _choices=_choices)
        if _requests is not None or _choices is not None:
            raise ValueError(
                "order benchmark requires genuine full native requests and choices"
            )
        if self.world is None or self.terminal():
            raise ValueError("initialized nonterminal native opportunity required")
        before = len(self.world.request_log)
        self._pending_order = None
        result = super().step(actions)
        pool, cap, selection, applied = self._pending_order
        logged = self.world.request_log[before:]
        actual = tuple(
            (
                key,
                None
                if row["chosen"] is None
                else self.compiled.spec.products[row["chosen"]].id,
            )
            for key, row in zip(selection["priority_order"], logged)
        )
        if len(logged) != len(pool.requests) or actual != selection["assignments"]:
            raise RuntimeError(
                "native dated admission differs from selected full-order witness"
            )
        self._order_records.append(
            DatedOrderAdmission(
                pool, cap, selection, actual, applied, self.order_policy
            )
        )
        self._pending_order = None
        return result

    def order_admissions(self):
        return tuple(self._order_records)

    def snapshot(self):
        return DatedOrderSnapshot(
            super().snapshot(),
            self.order_policy,
            self.order_admissions(),
            self._pending_order,
        )

    def restore(self, snapshot):
        if type(snapshot) is not DatedOrderSnapshot:
            raise TypeError("dated order benchmark combined snapshot required")
        if snapshot.policy != self.order_policy:
            raise ValueError("snapshot order objective/state-limit contract differs")
        super().restore(snapshot.native)
        self._order_records = list(snapshot.records)
        self._pending_order = snapshot.pending

    def branch(self):
        other = DatedOrderBenchmarkEngine(
            self.demand_scenario,
            {i: copy.deepcopy(c) for i, c in self.controllers.items()},
            order_policy=self.order_policy,
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


def dated_order_record(row: DatedOrderAdmission):
    """Analyst-only full input/native witness; no welfare/primary B4 assertion."""
    if type(row) is not DatedOrderAdmission:
        raise TypeError("DatedOrderAdmission required")
    obj = row.policy.objective
    return jsonable(
        {
            "plan_id": "A.8/A.5/B0/V9",
            "period": row.pool.problem.period,
            "scope": "all actual ordinary requests in one native period; genuine re-entry first",
            "objective": {
                "mode": obj.mode,
                "definition": obj.definition,
                "product_values": {
                    p: {
                        "numerator": str(v.numerator),
                        "denominator": str(v.denominator),
                    }
                    for p, v in obj.product_values
                },
            },
            "input_pool": native_pool_record(row.pool),
            "cap_posting": posting_record(row.cap_posting),
            "selection": result_record(row.selection),
            "actual_choices": row.actual_choices,
            "optimal_native_order_applied": row.optimal_order_applied,
            "mode": "optimal_native_order"
            if row.optimal_order_applied
            else "chronological_no_strict_clip",
            "caller_state_limit": row.policy.state_limit,
            "random_lottery_applied": False,
            "native_order_benchmark_executed": True,
            "static_IR_assignment_booked": False,
            "public_observability_or_truthfulness_established": False,
            "producer_or_objective_semantics_authenticated": False,
            "lifecycle_welfare_denial_or_misallocation_identified": False,
            "primary_B4_adopted": False,
            "scientific_budget": None,
        }
    )
