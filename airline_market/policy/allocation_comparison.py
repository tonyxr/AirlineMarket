"""A.8/A.5/B0/V9: native dated priority execution plus analyst-only static bound.

No efficient assignment is booked. Actual native choices are scored and compared
with a fixed-opportunity individual-rational relaxation, never lifecycle welfare.
"""

import copy
from dataclasses import dataclass, replace
from fractions import Fraction

from airline_market.core.hashing import jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.policy.dated_demand import DatedDemandMarketEngine
from airline_market.policy.rationing import (
    AllocationLimitError,
    exact_assignment_bound,
    rational,
    result_record,
)
from airline_market.policy.rationing_native import (
    assert_pool_current,
    native_pool_record,
)


@dataclass(frozen=True)
class AllocationObjective:
    """Explicit current native-utility gains OR full fixed product score table.

    Utility gains already include the current native taste/Gumbel sums. They are
    NOT logsum CS, money, lifecycle utility or a fitted welfare objective.
    """

    mode: str
    definition: str
    product_values: tuple = ()

    def __post_init__(self):
        if self.mode not in {"native_choice_gain", "product_scores"}:
            raise ValueError("explicit supported allocation objective mode required")
        if not isinstance(self.definition, str) or not self.definition.strip():
            raise ValueError("explicit allocation objective definition required")
        rows = tuple(sorted((p, rational(v)) for p, v in self.product_values))
        if len(dict(rows)) != len(rows) or any(
            not isinstance(p, str) or not p for p, v in rows
        ):
            raise ValueError("distinct explicit physical-product scores required")
        if self.mode == "native_choice_gain" and rows:
            raise ValueError("native choice gains do not take extra product scores")
        object.__setattr__(self, "product_values", rows)

    def scored_pool(self, pool):
        values = dict(self.product_values)
        if self.mode == "product_scores" and set(values) != {
            p for p, path in pool.problem.products
        }:
            raise ValueError("fixed score table must cover EVERY physical product")
        people = tuple(
            replace(
                a,
                offers=tuple(
                    replace(
                        o,
                        allocation_value=o.choice_value
                        if self.mode == "native_choice_gain"
                        else values[o.product],
                    )
                    for o in a.offers
                ),
            )
            for a in pool.problem.applicants
        )
        return replace(
            pool,
            problem=replace(pool.problem, applicants=people),
            score_definition=self.definition,
        )


def score_native_assignments(pool, assignments):
    """Verify one complete native ordinary assignment and exact stock/score identity."""
    people = {a.key: a for a in pool.problem.applicants}
    rows = tuple(assignments)
    if (
        len(rows) != len(people)
        or len({k for k, p in rows}) != len(rows)
        or {k for k, p in rows} != set(people)
    ):
        raise ValueError(
            "actual assignments must retain EVERY original applicant exactly once"
        )
    stock = dict(pool.problem.capacities)
    paths = dict(pool.problem.products)
    value = Fraction()
    for key, product in rows:
        if product is None:
            continue
        offer = next((o for o in people[key].offers if o.product == product), None)
        if offer is None or offer.choice_value < 0:
            raise ValueError("actual native choice is outside offered/willing support")
        if any(stock[k] <= 0 for k in paths[product]):
            raise ValueError("actual native assignments exceed net free capacity")
        for leg in paths[product]:
            stock[leg] -= 1
        value += offer.allocation_value
    capacity = tuple(
        {"leg": k, "initial": n, "used": n - stock[k], "remaining": stock[k]}
        for k, n in pool.problem.capacities
    )
    return value, capacity


@dataclass(frozen=True)
class AllocationComparison:
    pool: object
    actual_choices: tuple
    actual_score: Fraction
    actual_capacity: tuple
    bound: object
    gap: Fraction | None
    status: str
    caller_state_limit: int
    reason: str | None
    objective: object = None


def compare_completed_choices(pool, actual_choices, *, state_limit, objective=None):
    """Pure full-pool comparison; no actor/world/tape access and no partial bound.

    Caller supplies genuinely completed native choices. The engine below makes
    that join prospectively; this standalone function does not authenticate a
    saved producer, arbitrary caller choices or an economic welfare definition.
    """
    choices = tuple(actual_choices)
    value, capacity = score_native_assignments(pool, choices)
    try:
        bound = exact_assignment_bound(pool.problem, state_limit=state_limit)
    except AllocationLimitError as error:
        return AllocationComparison(
            pool,
            choices,
            value,
            deep_freeze(capacity),
            None,
            None,
            "state_limit_no_bound",
            state_limit,
            str(error),
            objective,
        )
    gap = bound["allocation_value"] - value
    if gap < 0:
        raise RuntimeError(
            "exact static assignment score below feasible native witness"
        )
    return AllocationComparison(
        pool,
        choices,
        value,
        deep_freeze(capacity),
        deep_freeze(bound),
        gap,
        "complete_static_comparison",
        state_limit,
        None,
        objective,
    )


@dataclass(frozen=True)
class AllocationComparisonSnapshot:
    native: object
    objective: object
    state_limit: object
    records: tuple
    pending: object


class DatedAllocationComparisonEngine(DatedDemandMarketEngine):
    """Opt-in analyst comparison; native whole-period priority/booking unchanged."""

    def __init__(
        self,
        scenario,
        controllers=None,
        *,
        allocation_objective=None,
        allocation_state_limit=None,
        **kwargs,
    ):
        if allocation_objective is not None:
            if type(allocation_objective) is not AllocationObjective:
                raise TypeError("AllocationObjective required")
            if kwargs.get("ordinary_priority") is None:
                raise ValueError(
                    "comparison requires existing whole-period native priority execution"
                )
            if type(allocation_state_limit) is not int or allocation_state_limit < 1:
                raise ValueError(
                    "explicit positive assignment infrastructure state limit required"
                )
            if allocation_objective.mode == "product_scores" and {
                p for p, v in allocation_objective.product_values
            } != {p.id for p in scenario.compiled.spec.products}:
                raise ValueError("fixed score table must cover EVERY physical product")
        if scenario.physical_shocks and kwargs.get("dated_rebooking") is None:
            raise ValueError(
                "dated physical comparison requires explicit stranded re-entry"
            )
        self.allocation_objective = allocation_objective
        self.allocation_state_limit = allocation_state_limit
        self._allocation_comparisons = []
        self._pending_allocation_pool = None
        super().__init__(scenario, controllers, **kwargs)

    def reset(self, seed=None, *, tape=None):
        self._allocation_comparisons = []
        self._pending_allocation_pool = None
        return super().reset(seed, tape=tape)

    def _ordered_ordinary_requests(self, pending):
        iterator = super()._ordered_ordinary_requests(pending)
        sentinel = object()
        first = next(iterator, sentinel)
        if self.allocation_objective is not None:
            pool = pending["pool"]
            assert_pool_current(self, pool)
            self._pending_allocation_pool = self.allocation_objective.scored_pool(pool)
        if first is not sentinel:
            yield first
        yield from iterator

    def step(self, actions=None, *, _requests=None, _choices=None):
        result = super().step(actions, _requests=_requests, _choices=_choices)
        if self.allocation_objective is not None:
            pool = self._pending_allocation_pool
            receipt = self.admission_records()[-1]
            if (
                pool is None
                or receipt.period != pool.problem.period
                or receipt.keys != tuple(a.key for a in pool.problem.applicants)
            ):
                raise RuntimeError(
                    "completed native receipt and captured full pool differ"
                )
            row = compare_completed_choices(
                pool,
                receipt.actual_choices,
                state_limit=self.allocation_state_limit,
                objective=self.allocation_objective,
            )
            self._allocation_comparisons.append(row)
            self._pending_allocation_pool = None
        return result

    def allocation_comparisons(self):
        return tuple(self._allocation_comparisons)

    def snapshot(self):
        return AllocationComparisonSnapshot(
            super().snapshot(),
            self.allocation_objective,
            self.allocation_state_limit,
            tuple(self._allocation_comparisons),
            self._pending_allocation_pool,
        )

    def restore(self, snapshot):
        if type(snapshot) is not AllocationComparisonSnapshot:
            raise TypeError("combined allocation-comparison snapshot required")
        if (snapshot.objective, snapshot.state_limit) != (
            self.allocation_objective,
            self.allocation_state_limit,
        ):
            raise ValueError(
                "snapshot declared allocation objective/state-limit contract differs"
            )
        super().restore(snapshot.native)
        self._allocation_comparisons = list(snapshot.records)
        self._pending_allocation_pool = snapshot.pending

    def branch(self):
        other = DatedAllocationComparisonEngine(
            self.demand_scenario,
            {i: copy.deepcopy(c) for i, c in self.controllers.items()},
            allocation_objective=self.allocation_objective,
            allocation_state_limit=self.allocation_state_limit,
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


def comparison_record(row):
    """Lossless analyst record; never a controller observation or primary B4 outcome."""
    if type(row) is not AllocationComparison:
        raise TypeError("AllocationComparison required")

    def number(v):
        return (
            None
            if v is None
            else {"numerator": str(v.numerator), "denominator": str(v.denominator)}
        )

    return jsonable(
        {
            "plan_id": "A.8/A.5/B0/V9",
            "period": row.pool.problem.period,
            "status": row.status,
            "objective": None
            if row.objective is None
            else {
                "mode": row.objective.mode,
                "definition": row.objective.definition,
                "product_values": {
                    p: number(v) for p, v in row.objective.product_values
                },
            },
            "input_pool": native_pool_record(row.pool),
            "actual_choices": row.actual_choices,
            "actual_declared_score": number(row.actual_score),
            "actual_capacity": row.actual_capacity,
            "exact_assignment_relaxation": None
            if row.bound is None
            else result_record(row.bound),
            "static_score_gap": number(row.gap),
            "caller_state_limit": row.caller_state_limit,
            "reason": row.reason,
            "native_efficient_assignment_booked": False,
            "lifecycle_welfare_denial_or_misallocation_identified": False,
            "dynamic_or_equilibrium_bound": False,
            "primary_B4_adopted": False,
            "producer_or_objective_semantics_authenticated": False,
            "scientific_budget": None,
        }
    )
