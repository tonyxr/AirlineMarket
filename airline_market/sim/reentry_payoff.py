"""A.10/A.5/B0/V9: ONE person's declared finite re-entry policy payoff.

Persistent type law; frozen ordered choice. Payoffs exclude choice-shock utility.
No native lifecycle CS, delivered outcome inference, optimizer or physical run.
"""

import math
from dataclasses import dataclass
from fractions import Fraction

from airline_market.core.hashing import jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.sim.ordered_reentry_logit import (
    OrderedReentryLogit,
    ordered_reentry_expectation,
)
from airline_market.sim.payoff_logit import ContingentPayoff, _ratio


def _name(s):
    return isinstance(s, str) and bool(s.strip())


@dataclass(frozen=True)
class PayoffTransition:
    name: str
    probability: Fraction
    next_state: str
    payoff: ContingentPayoff

    def __post_init__(self):
        if (
            not _name(self.name)
            or not _name(self.next_state)
            or type(self.payoff) is not ContingentPayoff
        ):
            raise ValueError(
                "explicit branch/next state and FOUR payoff components required"
            )
        if type(self.probability) is not Fraction or self.probability < 0:
            raise ValueError(
                "explicit nonnegative rational conditional branch probability required"
            )


@dataclass(frozen=True)
class ReentryPayoffState:
    name: str
    period: int
    choice: OrderedReentryLogit | None
    transitions: tuple
    terminal_payoff: ContingentPayoff | None

    def __post_init__(self):
        rows = tuple(
            sorted(
                (k, tuple(sorted(edges, key=lambda e: e.name)))
                for k, edges in self.transitions
            )
        )
        if not _name(self.name) or type(self.period) is not int or self.period < 0:
            raise ValueError("explicit state name/nonnegative native period required")
        if self.choice is None:
            if rows or type(self.terminal_payoff) is not ContingentPayoff:
                raise ValueError(
                    "terminal requires explicit FOUR payoff components and no choice branches"
                )
        else:
            if (
                type(self.choice) is not OrderedReentryLogit
                or self.terminal_payoff is not None
            ):
                raise ValueError("fixed ordered choice OR explicit terminal required")
            names = {k for k, v in (*self.choice.own, *self.choice.other)} | {
                self.choice.outside_name
            }
            if len(dict(rows)) != len(rows) or set(dict(rows)) != names:
                raise ValueError(
                    "ALL own/other/outside action transitions required, including inactive/zero rows"
                )
            for key, edges in rows:
                if (
                    not edges
                    or any(type(e) is not PayoffTransition for e in edges)
                    or len({e.name for e in edges}) != len(edges)
                ):
                    raise ValueError("complete distinct conditional branches required")
                if sum((e.probability for e in edges), Fraction()) != 1:
                    raise ValueError(
                        "each conditional branch law sums EXACTLY to one, no normalization"
                    )
        object.__setattr__(self, "transitions", rows)


@dataclass(frozen=True)
class ReentryPayoffType:
    name: str
    probability: Fraction
    initial: str
    states: tuple

    def __post_init__(self):
        states = tuple(sorted(self.states, key=lambda s: (s.period, s.name)))
        if (
            not _name(self.name)
            or type(self.probability) is not Fraction
            or self.probability < 0
            or not states
            or any(type(s) is not ReentryPayoffState for s in states)
        ):
            raise ValueError(
                "named persistent type, rational probability and complete states required"
            )
        table = {s.name: s for s in states}
        if len(table) != len(states) or self.initial not in table:
            raise ValueError("distinct states and explicit initial state required")
        alphas = {s.choice.alpha for s in states if s.choice is not None}
        if len(alphas) > 1:
            raise ValueError(
                "one fixed current re-entry alpha per persistent type; no taste redraw"
            )
        seen = {self.initial}
        for s in states:
            for action, edges in s.transitions:
                for e in edges:
                    if (
                        e.next_state not in table
                        or table[e.next_state].period <= s.period
                    ):
                        raise ValueError(
                            "ALL transitions strictly advance native periods to existing states; no cycle/fill"
                        )
        for s in states:
            if s.name in seen:
                seen.update(
                    e.next_state for action, edges in s.transitions for e in edges
                )
        if seen != set(table):
            raise ValueError(
                "retain connected full branch support including zero probabilities, no unattached states"
            )
        object.__setattr__(self, "states", states)


@dataclass(frozen=True)
class SinglePersonReentryPayoff:
    person_key: tuple
    types: tuple
    horizon: int
    payoff_unit: str
    conditioning: str

    def __post_init__(self):
        key = tuple(self.person_key)
        types = tuple(sorted(self.types, key=lambda t: t.name))
        if len(key) != 3 or any(type(x) is not int or x < 0 for x in key):
            raise ValueError(
                "explicit original potential coordinate required, not ticket/search reconstruction"
            )
        if (
            type(self.horizon) is not int
            or self.horizon < 1
            or not _name(self.payoff_unit)
            or not _name(self.conditioning)
        ):
            raise ValueError(
                "explicit native horizon/common payoff unit/conditioning required"
            )
        if (
            not types
            or any(type(t) is not ReentryPayoffType for t in types)
            or len({t.name for t in types}) != len(types)
            or sum((t.probability for t in types), Fraction()) != 1
        ):
            raise ValueError(
                "complete persistent type law sums EXACTLY to one; no averaging or normalization"
            )
        if any(
            s.period > self.horizon
            or (s.choice is not None and s.period == self.horizon)
            for t in types
            for s in t.states
        ):
            raise ValueError(
                "every choice before explicit horizon, every terminal within it"
            )
        object.__setattr__(self, "person_key", key)
        object.__setattr__(self, "types", types)


class ReentryPayoffLimitError(RuntimeError):
    """Infrastructure scope only; no partial payoff or automatic retry."""


def reentry_policy_payoff(
    problem: SinglePersonReentryPayoff, *, node_limit: int, edge_limit: int
):
    """Evaluate fixed myopic ordered-choice policy, never optimize future payoff.

    V(s)=sum_a p_numeric(a|s) sum_e q_exact(e|s,a)[z(e)+V(next)].
    Payoffs exclude Gumbels. No expected per-attempt CS/gains are added.
    """
    if type(problem) is not SinglePersonReentryPayoff:
        raise TypeError("SinglePersonReentryPayoff required")
    if any(type(v) is not int or v < 1 for v in (node_limit, edge_limit)):
        raise ValueError("explicit positive infrastructure node/edge limits required")
    nodes = sum(len(t.states) for t in problem.types)
    edges = sum(
        len(e) for t in problem.types for s in t.states for a, e in s.transitions
    )
    if nodes > node_limit or edges > edge_limit:
        raise ReentryPayoffLimitError(
            "full declared node/edge support exceeds infrastructure limit; no partial result/retry"
        )
    type_records = []
    aggregate = Fraction()
    global_unknown = False
    for kind in problem.types:
        values = {}
        records = {}
        for state in reversed(kind.states):
            reasons = []
            rows = []
            choice_record = None
            if state.choice is None:
                value = state.terminal_payoff.net_value()
                if value is None:
                    reasons.append("unknown_terminal_payoff")
            else:
                q = state.choice
                choice_record = ordered_reentry_expectation(q)
                probabilities = dict(choice_record["choice_probabilities"])
                active = (
                    {k for k, v in q.own if math.isfinite(v)}
                    | {k for k, v in q.other if q.interline and math.isfinite(v)}
                    | {q.outside_name}
                )
                value = Fraction()
                for action, branches in state.transitions:
                    p = Fraction.from_float(probabilities[action])
                    positive = action in active
                    if positive and p == 0:
                        reasons.append("positive_choice_probability_underflow")
                    for e in branches:
                        net = e.payoff.net_value()
                        child = values[e.next_state]
                        needed = positive and e.probability > 0
                        if needed and net is None:
                            reasons.append("positive_branch_unknown_payoff")
                        if needed and child is None:
                            reasons.append("positive_branch_unknown_continuation")
                        contribution = (
                            Fraction()
                            if not needed
                            else None
                            if net is None or child is None or p == 0
                            else p * e.probability * (net + child)
                        )
                        if contribution is not None:
                            value += contribution
                        rows.append(
                            {
                                "action": action,
                                "choice_probability": probabilities[action],
                                "structurally_positive_action": positive,
                                "branch": e.name,
                                "conditional_probability": _ratio(e.probability),
                                "next_state": e.next_state,
                                "payoff": e.payoff.record(),
                                "continuation_value": _ratio(child),
                                "weighted_payoff_plus_continuation": _ratio(
                                    contribution
                                ),
                            }
                        )
                if reasons:
                    value = None
            values[state.name] = value
            records[state.name] = {
                "state": state.name,
                "period": state.period,
                "terminal_payoff": None
                if state.terminal_payoff is None
                else state.terminal_payoff.record(),
                "ordered_choice": choice_record,
                "exact_binary64_choice_probability_mass": None
                if choice_record is None
                else _ratio(
                    sum(
                        (
                            Fraction.from_float(p)
                            for k, p in choice_record["choice_probabilities"]
                        ),
                        Fraction(),
                    )
                ),
                "exact_binary64_choice_mass_residual": None
                if choice_record is None
                else _ratio(
                    sum(
                        (
                            Fraction.from_float(p)
                            for k, p in choice_record["choice_probabilities"]
                        ),
                        Fraction(),
                    )
                    - 1
                ),
                "branches": tuple(rows),
                "value_exact_weighting_of_binary64_probabilities": _ratio(value),
                "null_reasons": tuple(sorted(set(reasons))),
            }
        value = values[kind.initial]
        if kind.probability > 0:
            if value is None:
                global_unknown = True
            else:
                aggregate += kind.probability * value
        type_records.append(
            {
                "type": kind.name,
                "probability": _ratio(kind.probability),
                "initial_state": kind.initial,
                "states": tuple(records[s.name] for s in kind.states),
                "initial_value": _ratio(value),
            }
        )
    return deep_freeze(
        {
            "plan_id": "A.10/A.5/B0/V9",
            "original_person_key": problem.person_key,
            "declared_original_people": 1,
            "payoff_unit": problem.payoff_unit,
            "conditioning": problem.conditioning,
            "native_horizon": problem.horizon,
            "types": tuple(type_records),
            "expected_declared_policy_payoff": None
            if global_unknown
            else _ratio(aggregate),
            "positive_support_unknown_or_choice_underflow": global_unknown,
            "nodes_evaluated": nodes,
            "edges_retained": edges,
            "limits": {
                "node_limit": node_limit,
                "edge_limit": edge_limit,
                "scientific_budget": None,
            },
            "precision": "Exact rational branch/type/payoff arithmetic weighting actual numerical binary64 ordered-choice probabilities; raw choice residuals retained, no normalization or rigorous interval/exact market expectation",
            "payoff_model": "Explicit incremental/terminal service benefit - NET ticket payment - waiting loss - cancellation loss; conditional branch law/payoff independent of remaining Gumbel magnitudes given selected action and declared state",
            "choice_shock_utility_in_payoff": False,
            "future_payoff_optimized_or_fed_back_into_choice": False,
            "per_search_expected_gains_or_CS_added": False,
            "types_independently_redrawn_between_attempts": False,
            "native_periods_interpolated_to_elapsed_waiting": False,
            "source_empirical_identity_or_native_transition_law_authenticated": False,
            "world_actor_tape_RNG_physical_book_inference_or_old_solver_called": False,
            "native_lifecycle_consumer_surplus": None,
            "delivered_travel": None,
            "denied_requests": None,
            "misallocation": None,
            "primary_default_empirical_production_planner_adopted": False,
            "scientific_budget": None,
        }
    )


def reentry_payoff_record(result):
    """Owned full finite law and payoff output, no saved-run/account loader."""
    return jsonable(result)
