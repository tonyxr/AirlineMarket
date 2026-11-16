"""A.10/A.5/B0/V9: full finite declared BASE Bernoulli ledger histories.

Future programs and conditional independence are declarations. Native myopic
choice/book/resolve are delegated; no re-entry, lifecycle welfare or planner.
"""

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from airline_market.core.hashing import jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.sim.bernoulli_boundary import (
    DeclaredBaseLedger,
    resolve_bernoulli_boundary,
)
from airline_market.sim.bernoulli_person import MARK_LAW, native_bernoulli_person_law
from airline_market.sim.passenger_accounts import _encode
from airline_market.sim.payoff_logit import _ratio
from airline_market.sim.state import Booking
from airline_market.spec.compiled import CompiledMarket


@dataclass(frozen=True)
class BernoulliOpportunity:
    exogenous_state: int
    prices: tuple
    installed_after: tuple

    def __post_init__(self):
        object.__setattr__(self, "prices", tuple(self.prices))
        object.__setattr__(self, "installed_after", tuple(self.installed_after))


@dataclass(frozen=True)
class BernoulliFuturePath:
    name: str
    weight: Fraction
    opportunities: tuple

    def __post_init__(self):
        object.__setattr__(self, "opportunities", tuple(self.opportunities))


@dataclass(frozen=True)
class BernoulliFutureResult:
    joint: tuple | None
    expected_counts: object | None
    details: object


def _free(compiled, ledger):
    # Readonly full-incidence projection of declared installed seats/tickets.
    free = list(ledger.installed)
    for ticket in ledger.tickets:
        if type(ticket) is not Booking or not 0 <= ticket.product < len(
            compiled.spec.products
        ):
            raise ValueError("complete native carried tickets required")
        for l, debit in enumerate(compiled.A_global[:, ticket.product]):
            free[l] -= int(debit)
    if any(v < 0 for v in free):
        raise ValueError("declared carried ledger exceeds installed capacity")
    return tuple(free)


def bernoulli_future_expectation(
    compiled,
    ledger,
    paths,
    *,
    start_period,
    conditioning,
    mark_law,
    independent_future_private_marks,
    unreserved_base_ledger,
    tree_row_limit,
    conditional_call_limit,
    ticket_limit,
):
    """Propagate ALL NEW ordinary request histories, not mean stock or marginals.

    Complete fixed path programs are weighted once before future private marks.
    Each period has at most one ordinary request. Current retained tickets are
    carried, cancellations persist in history; no cancelled person is re-entered.
    """
    if type(compiled) is not CompiledMarket or type(ledger) is not DeclaredBaseLedger:
        raise TypeError(
            "native compiled model and complete declared BASE ledger required"
        )
    if compiled.spec.demand.mode != "single_request_bernoulli":
        raise ValueError("native Bernoulli only; no Poisson fallback")
    if (
        mark_law != MARK_LAW
        or independent_future_private_marks is not True
        or unreserved_base_ledger is not True
        or not isinstance(conditioning, str)
        or not conditioning.strip()
    ):
        raise ValueError(
            "explicit pre-future information and independent BASE mark law required"
        )
    if any(
        type(n) is not int or n < 1
        for n in (tree_row_limit, conditional_call_limit, ticket_limit)
    ):
        raise ValueError("positive ALL-support infrastructure limits required")
    paths = tuple(paths)
    if not paths or any(type(p) is not BernoulliFuturePath for p in paths):
        raise TypeError("complete nonempty typed future paths required")
    if (
        any(not isinstance(p.name, str) or not p.name.strip() for p in paths)
        or len({p.name for p in paths}) != len(paths)
        or any(type(p.weight) is not Fraction or p.weight < 0 for p in paths)
        or sum((p.weight for p in paths), Fraction()) != 1
    ):
        raise ValueError(
            "unique names and exact nonnegative Fraction path weights summing one required"
        )
    periods = len(paths[0].opportunities)
    if (
        type(start_period) is not int
        or periods < 1
        or start_period < 0
        or start_period + periods > compiled.spec.horizon.n_periods
        or any(len(p.opportunities) != periods for p in paths)
    ):
        raise ValueError(
            "common complete in-horizon future opportunity support required"
        )
    if len(ledger.installed) != len(compiled.spec.legs) or any(
        type(v) is not int or not 0 <= v <= np.iinfo(np.int64).max
        for v in ledger.installed
    ):
        raise ValueError("complete installed integer capacity required")
    for path in paths:
        installed = ledger.installed
        for opportunity in path.opportunities:
            if type(opportunity) is not BernoulliOpportunity:
                raise TypeError("typed complete future opportunities required")
            if (
                type(opportunity.exogenous_state) is not int
                or not 0
                <= opportunity.exogenous_state
                < len(compiled.spec.exogenous.states)
                or len(opportunity.prices) != len(compiled.spec.products)
                or any(
                    isinstance(v, (bool, np.bool_)) or not math.isfinite(float(v))
                    for v in opportunity.prices
                )
                or len(opportunity.installed_after) != len(installed)
                or any(
                    type(v) is not int or not 0 <= v <= old
                    for v, old in zip(opportunity.installed_after, installed)
                )
            ):
                raise ValueError(
                    "full finite fares/exogenous state/nonincreasing installed program required"
                )
            installed = opportunity.installed_after
    rows = 1 + sum(
        (len(compiled.market_products[m]) + 1) * len(g.classes)
        for m in range(len(compiled.spec.markets))
        for g in compiled.spec.segments
    )
    calls = frontier = 0
    for depth in range(periods):
        frontier = 1 if depth == 0 else frontier * rows
        calls += frontier * len(paths)
        if calls > conditional_call_limit or calls * rows > tree_row_limit:
            raise RuntimeError(
                "ALL-support full-tree upper bound exceeds limit; no partial/fallback/retry"
            )
    if len(ledger.tickets) + periods > ticket_limit:
        raise RuntimeError(
            "ALL future ticket support exceeds limit; no partial/fallback/retry"
        )
    initial_free = _free(compiled, ledger)
    kernels = len(compiled.spec.markets) * sum(
        len(g.classes) for g in compiled.spec.segments
    )
    nodes, terminals, needed_unknown = [], [], []
    names = (
        "new_original_arrivals",
        "new_outside_choices",
        "new_purchases",
        "all_cancellations",
        "terminal_active_tickets",
    )
    sums = dict.fromkeys(names, Fraction())
    initial_ids = {b.record_id for b in ledger.tickets}
    for path in paths:
        # Keeping zero-weight paths/zero-mass histories is deliberate.
        current = [(ledger, path.weight, path.weight > 0, (), (0, 0, 0, 0))]
        for depth, opportunity in enumerate(path.opportunities):
            following = []
            period = start_period + depth
            for state, mass, needed, history, counts in current:
                law = native_bernoulli_person_law(
                    compiled,
                    period=period,
                    exogenous_state=opportunity.exogenous_state,
                    prices=opportunity.prices,
                    net_free_stock=_free(compiled, state),
                    mark_law=mark_law,
                    conditioning=conditioning,
                    fixed_before_current_private_marks=True,
                    row_limit=rows,
                    kernel_call_limit=kernels,
                )
                result = resolve_bernoulli_boundary(
                    compiled,
                    law,
                    state,
                    installed_after=opportunity.installed_after,
                    definition=conditioning,
                    unreserved_base_ledger=True,
                    boundary_fixed_before_current_private_marks=True,
                    branch_limit=rows,
                    ticket_limit=ticket_limit,
                )
                node = len(nodes)
                underflows = {tuple(row) for row in law.details["needed_underflows"]}
                if needed and law.joint is None:
                    needed_unknown.append(node)
                nodes.append(
                    {
                        "node": node,
                        "path": path.name,
                        "period": period,
                        "history": history,
                        "raw_reaching_mass": _ratio(mass),
                        "structurally_needed": needed,
                        "boundary": result.details,
                    }
                )
                for branch in result.details["branches"]:
                    if not branch["executed"]:
                        continue  # Inactive zero row is retained in the full node, has no physical continuation.
                    m, g, k, j, _stock, ratio = branch["input"]
                    weight = Fraction(
                        int(ratio["numerator"]), int(ratio["denominator"])
                    )
                    reachable = needed and (weight > 0 or (m, g, k, j) in underflows)
                    carrier_rows = branch["native_carrier_ledgers"]
                    next_state = DeclaredBaseLedger(
                        opportunity.installed_after,
                        tuple(Booking(**dict(b)) for b in branch["kept"]),
                        tuple(
                            (
                                r["cash"],
                                r["refunds"],
                                r["penalties"],
                                r["canceled"],
                                r["revenue"],
                            )
                            for r in carrier_rows
                        ),
                        branch["next_record_id"],
                    )
                    arrived = m is not None
                    totals = tuple(
                        a + b
                        for a, b in zip(
                            counts,
                            (
                                int(arrived),
                                int(arrived and j == len(compiled.spec.products)),
                                int(branch["new_ticket"] is not None),
                                len(branch["canceled"]),
                            ),
                        )
                    )
                    next_history = (*history, (node, branch["row"]))
                    next_mass = mass * weight
                    if depth + 1 == periods:
                        values = (*totals, len(next_state.tickets))
                        for name, value in zip(names, values):
                            sums[name] += next_mass * value
                        terminals.append(
                            {
                                "path": path.name,
                                "history": next_history,
                                "raw_mass": _ratio(next_mass),
                                "structurally_needed": reachable,
                                "ledger": jsonable(next_state),
                                "counts": tuple(zip(names, values)),
                                "new_active_ticket_ids": tuple(
                                    b.record_id
                                    for b in next_state.tickets
                                    if b.record_id not in initial_ids
                                ),
                            }
                        )
                    else:
                        following.append(
                            (next_state, next_mass, reachable, next_history, totals)
                        )
            current = following
    total_mass = sum(
        (
            Fraction(int(t["raw_mass"]["numerator"]), int(t["raw_mass"]["denominator"]))
            for t in terminals
        ),
        Fraction(),
    )
    known = not needed_unknown
    return BernoulliFutureResult(
        tuple(
            (
                t["path"],
                t["history"],
                Fraction(
                    int(t["raw_mass"]["numerator"]), int(t["raw_mass"]["denominator"])
                ),
            )
            for t in terminals
        )
        if known
        else None,
        deep_freeze(sums) if known else None,
        deep_freeze(
            {
                "plan_id": "A.10/A.5/B0/V9",
                "complete_native_model": jsonable(compiled.spec),
                "conditioning": conditioning,
                "mark_law": mark_law,
                "independent_future_private_marks_declared_not_authenticated": True,
                "initial_declared_ledger": jsonable(ledger),
                "initial_net_free_stock": initial_free,
                "start_period": start_period,
                "paths": tuple(
                    {
                        "name": p.name,
                        "weight": _ratio(p.weight),
                        "opportunities": jsonable(p.opportunities),
                    }
                    for p in paths
                ),
                "nodes": tuple(nodes),
                "terminals": tuple(terminals),
                "needed_unknown_nodes": tuple(needed_unknown),
                "exact_raw_mass": _ratio(total_mass),
                "raw_float_mass": float(total_mass),
                "exact_raw_mass_residual": _ratio(total_mass - 1),
                "raw_weighted_counts": tuple(
                    (name, _ratio(v)) for name, v in sums.items()
                ),
                "tree_rows_upper_bound": calls * rows,
                "conditional_calls_upper_bound": calls,
                "actual_conditional_calls": len(nodes),
                "all_rows_per_conditional": rows,
                "tree_row_limit": tree_row_limit,
                "conditional_call_limit": conditional_call_limit,
                "ticket_limit": ticket_limit,
                "native_book_resolve_delegated_on_owned_BASE_ledger_hosts": True,
                "full_engine_native_period_advance_reentry_actor_tape_RNG_or_planner": False,
                "physical_program_probability_information_seed_empirical_law_authenticated": False,
                "lifecycle_CS_delivered_waiting_penalty_incidence_causal_denial_misallocation": None,
                "full_phase_primary_default_empirical_production_adoption": False,
                "scientific_budget": None,
            }
        ),
    )


def bernoulli_future_record(result):
    if type(result) is not BernoulliFutureResult:
        raise TypeError("BernoulliFutureResult required")
    return _encode(jsonable(result.details))
