"""A.8/B0/V9: exact full-pool native-choice ordering bound at fixed postings.

The analyst can vary order only, never offers, choices, prices or stock. This
static realized-score bound is not an admission policy or lifecycle welfare.
"""

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

from airline_market.core.hashing import jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.policy.allocation_comparison import (
    AllocationComparison,
    comparison_record,
    score_native_assignments,
)
from airline_market.policy.rationing import (
    AllocationLimitError,
    AllocationProblem,
    priority_allocation,
    result_record,
)


class NativeOrderLimitError(AllocationLimitError):
    """Infrastructure limit; no partial order optimum or witness is returned."""


def _choice(problem, person, stock):
    """Delegate native feasible argmax/offer/outside ties to the frozen primitive."""
    one = AllocationProblem(problem.period, stock, problem.products, (person,))
    result = priority_allocation(one, binding_cap=False)
    remaining = tuple((x["leg"], x["remaining"]) for x in result["capacity"])
    return result["assignments"][0], result["allocation_value"], remaining


def native_order_allocation(problem: AllocationProblem, order: tuple):
    """Replay every supplied applicant once using only native greedy choice.

    Order is explicit and may use current private marks: this is an analyst
    witness, not a publicly observable or truthful regulator mechanism.
    """
    if type(problem) is not AllocationProblem:
        raise TypeError("AllocationProblem required")
    order = tuple(order)
    people = {a.key: a for a in problem.applicants}
    if (
        len(order) != len(people)
        or len(set(order)) != len(order)
        or set(order) != set(people)
    ):
        raise ValueError("order must retain EVERY original applicant exactly once")
    stock = problem.capacities
    value = Fraction(0)
    assignments = []
    for key in order:
        choice, score, stock = _choice(problem, people[key], stock)
        assignments.append(choice)
        value += score
    return deep_freeze(
        {
            "plan_id": "A.8/B0/V9",
            "mode": "explicit_full_pool_native_order",
            "priority_order": order,
            "assignments": tuple(assignments),
            "allocation_value": value,
            "capacity": tuple(
                {
                    "leg": leg,
                    "initial": initial,
                    "used": initial - remaining,
                    "remaining": remaining,
                }
                for (leg, initial), (_, remaining) in zip(problem.capacities, stock)
            ),
            "supplied_requests": len(people),
            "unassigned_requests": sum(p is None for key, p in assignments),
            "native_engine_adopted": False,
            "dynamic_or_equilibrium_bound": False,
        }
    )


def exact_native_order_bound(problem: AllocationProblem, *, state_limit: int):
    r"""Solve V(R,x)=max_i[g_i(x)+V(R\{i},x-a_i(x))] exactly (A.8).

    g_i and a_i come from ONE frozen native choice for applicant i at stock x.
    Every full permutation remains eligible, including outside/zero-stock rows.
    Canonical-key first ties are fixed before outcomes. Negative optima remain.
    Caller state_limit bounds visited memoized states, not scientific/F9 cost.
    """
    if type(problem) is not AllocationProblem:
        raise TypeError("AllocationProblem required")
    if type(state_limit) is not int or state_limit < 1:
        raise ValueError("explicit positive state limit required")
    visits = 0
    people = problem.applicants

    @lru_cache(None)
    def solve(mask, stock):
        nonlocal visits
        visits += 1
        if visits > state_limit:
            raise NativeOrderLimitError("native order state limit; no partial bound")
        if mask == 0:
            return Fraction(0), ()
        best = None
        for i, person in enumerate(people):
            if not mask & (1 << i):
                continue
            _, score, remaining = _choice(problem, person, stock)
            future, order = solve(mask ^ (1 << i), remaining)
            candidate = score + future
            if best is None or candidate > best[0]:
                best = candidate, (person.key,) + order
        return best

    value, order = solve((1 << len(people)) - 1, problem.capacities)
    witness = native_order_allocation(problem, order)
    if witness["allocation_value"] != value:
        raise RuntimeError("native order optimum and full witness differ")
    return deep_freeze(
        {
            **witness,
            "mode": "exact_full_pool_native_order_bound",
            "visited_states": visits,
            "caller_state_limit": state_limit,
            "tie_rule": "lexicographically first full original-key order",
            "offers_or_native_choices_rationed": False,
            "lifecycle_welfare_denial_or_misallocation_identified": False,
            "primary_B4_adopted": False,
            "scientific_budget": None,
        }
    )


@dataclass(frozen=True)
class NativeOrderComparison:
    completed: AllocationComparison
    native_order_bound: object
    ordering_score_gap: Fraction | None
    alternative_rationing_relaxation_gap: Fraction | None
    status: str
    caller_state_limit: int
    reason: str | None


def compare_native_order(completed: AllocationComparison, *, state_limit: int):
    """Prospective bracket: actual <= best native order <= prior IR relaxation.

    Verify native greedy choices in their supplied execution order. Reuse the
    prior IR result, preserving NULL if limited; NEVER run that solver again.
    Mathematical witness checks do not authenticate producer/source semantics.
    """
    if type(completed) is not AllocationComparison:
        raise TypeError("AllocationComparison required")
    actual = native_order_allocation(
        completed.pool.problem, tuple(k for k, p in completed.actual_choices)
    )
    value, capacity = score_native_assignments(completed.pool, completed.actual_choices)
    if (
        actual["assignments"] != completed.actual_choices
        or value != completed.actual_score
        or deep_freeze(capacity) != completed.actual_capacity
    ):
        raise ValueError("completed witness differs from full-pool native choice order")
    try:
        bound = exact_native_order_bound(
            completed.pool.problem, state_limit=state_limit
        )
    except NativeOrderLimitError as error:
        return NativeOrderComparison(
            completed,
            None,
            None,
            None,
            "state_limit_no_order_bound",
            state_limit,
            str(error),
        )
    gap = bound["allocation_value"] - value
    ir_gap = (
        None
        if completed.bound is None
        else completed.bound["allocation_value"] - bound["allocation_value"]
    )
    if gap < 0 or (ir_gap is not None and ir_gap < 0):
        raise RuntimeError("native order / static assignment score bracket violated")
    return NativeOrderComparison(
        completed,
        bound,
        gap,
        ir_gap,
        "complete_native_order_comparison",
        state_limit,
        None,
    )


def native_order_comparison_record(row: NativeOrderComparison):
    """Lossless immutable-row export; exact scores, preserved NULLs and raw input."""
    if type(row) is not NativeOrderComparison:
        raise TypeError("NativeOrderComparison required")

    def number(value):
        return (
            None
            if value is None
            else {
                "numerator": str(value.numerator),
                "denominator": str(value.denominator),
            }
        )

    return jsonable(
        {
            "plan_id": "A.8/B0/V9",
            "status": row.status,
            "completed_comparison": comparison_record(row.completed),
            "native_order_bound": None
            if row.native_order_bound is None
            else result_record(row.native_order_bound),
            "ordering_score_gap": number(row.ordering_score_gap),
            "alternative_rationing_relaxation_gap": number(
                row.alternative_rationing_relaxation_gap
            ),
            "caller_state_limit": row.caller_state_limit,
            "reason": row.reason,
            "native_efficient_assignment_booked": False,
            "public_observability_or_truthfulness_established": False,
            "producer_or_objective_semantics_authenticated": False,
            "lifecycle_welfare_denial_or_misallocation_identified": False,
            "primary_B4_adopted": False,
            "scientific_budget": None,
        }
    )
