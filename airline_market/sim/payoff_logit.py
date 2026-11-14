"""A.10/B0/V9: declared single-commitment payoff lotteries and logit timing.

Risk-neutral/quasilinear finite model only, not adopted native lifecycle welfare.
"""

import math
from dataclasses import dataclass
from fractions import Fraction

from airline_market.core.hashing import jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.sim.segmented_welfare import conditional_logit_surplus


def _ratio(x):
    return (
        None
        if x is None
        else {"numerator": str(x.numerator), "denominator": str(x.denominator)}
    )


@dataclass(frozen=True)
class ContingentPayoff:
    service_benefit: Fraction | None
    net_ticket_payment: Fraction | None
    waiting_loss: Fraction | None
    cancellation_loss: Fraction | None

    def __post_init__(self):
        if any(x is not None and type(x) is not Fraction for x in self.components()):
            raise TypeError(
                "explicit rational payoff components or unknown None required"
            )

    def components(self):
        return (
            self.service_benefit,
            self.net_ticket_payment,
            self.waiting_loss,
            self.cancellation_loss,
        )

    def net_value(self):
        b, p, w, c = self.components()
        return None if any(x is None for x in (b, p, w, c)) else b - p - w - c

    def record(self):
        return {
            **{
                k: _ratio(v)
                for k, v in zip(
                    (
                        "service_benefit",
                        "net_ticket_payment",
                        "waiting_loss",
                        "cancellation_loss",
                    ),
                    self.components(),
                )
            },
            "net_value": _ratio(self.net_value()),
        }


@dataclass(frozen=True)
class PayoffState:
    name: str
    probability: Fraction
    payoffs: tuple

    def __post_init__(self):
        rows = tuple((k, v) for k, v in self.payoffs)
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("explicit payoff state name required")
        if type(self.probability) is not Fraction or self.probability < 0:
            raise ValueError("explicit nonnegative rational state probability required")
        if any(
            not isinstance(k, str) or not k or type(v) is not ContingentPayoff
            for k, v in rows
        ) or len(dict(rows)) != len(rows):
            raise ValueError(
                "distinct named alternatives with explicit payoff components required"
            )
        object.__setattr__(self, "payoffs", tuple(sorted(rows)))


@dataclass(frozen=True)
class SingleCommitmentLogit:
    alternatives: tuple
    outside: str
    alpha: Fraction
    states: tuple
    payoff_unit: str
    definition: str

    def __post_init__(self):
        alternatives = tuple(sorted((k, v) for k, v in self.alternatives))
        states = tuple(self.states)
        if (
            not alternatives
            or any(
                not isinstance(k, str) or not k or type(v) is not Fraction
                for k, v in alternatives
            )
            or len(dict(alternatives)) != len(alternatives)
            or self.outside not in dict(alternatives)
        ):
            raise ValueError(
                "complete distinct fixed alternatives/rational utility offsets and explicit outside required"
            )
        if type(self.alpha) is not Fraction or self.alpha <= 0:
            raise ValueError(
                "explicit positive rational money-to-utility alpha required"
            )
        if (
            not states
            or any(type(s) is not PayoffState for s in states)
            or len({s.name for s in states}) != len(states)
        ):
            raise ValueError("complete distinct finite joint payoff states required")
        if sum((s.probability for s in states), Fraction()) != 1:
            raise ValueError(
                "rational state probabilities must sum to one; no normalization"
            )
        if any(set(dict(s.payoffs)) != set(dict(alternatives)) for s in states):
            raise ValueError(
                "every state x alternative including outside required; no fill"
            )
        if any(
            not isinstance(v, str) or not v.strip()
            for v in (self.payoff_unit, self.definition)
        ):
            raise ValueError("explicit common payoff unit and definition required")
        object.__setattr__(self, "alternatives", alternatives)
        object.__setattr__(self, "states", tuple(sorted(states, key=lambda s: s.name)))


def payoff_information_surplus(problem: SingleCommitmentLogit):
    """CS_before=LSE(a+alpha E[z])-E[u_out]; CS_revealed=E[LSE(u)-u_out].

    Both are divided by the SAME alpha. One choice with iid unit Gumbels,
    independent of the declared state; risk-neutral quasilinear commitments.
    No dynamic re-entry, repeated-search CS, event-time or delivered-service inference.
    """
    if type(problem) is not SingleCommitmentLogit:
        raise TypeError("SingleCommitmentLogit required")
    choices = tuple(k for k, v in problem.alternatives)
    outside_index = choices.index(problem.outside)
    offsets = dict(problem.alternatives)

    def utility_values(values):
        exact = tuple(offsets[k] + problem.alpha * v for k, v in zip(choices, values))
        baseline = exact[outside_index]
        return tuple(float(u - baseline) for u in exact)

    positive = []
    state_records = []
    unknown = False
    for state in problem.states:
        payoffs = dict(state.payoffs)
        values = tuple(payoffs[k].net_value() for k in choices)
        cs = None
        utilities = None
        if state.probability == 0:
            status = "declared_zero_probability_not_evaluated"
        elif any(v is None for v in values):
            unknown = True
            status = "positive_probability_unknown_payoff"
            positive.append((state.probability, values, None))
        else:
            utilities = utility_values(values)
            if not all(math.isfinite(v) for v in utilities):
                raise FloatingPointError(
                    "payoff utility exceeds finite numerical backend; no partial result"
                )
            cs = conditional_logit_surplus(
                utilities, float(problem.alpha), outside_index=outside_index
            )
            if not math.isfinite(cs):
                raise FloatingPointError(
                    "nonfinite conditional surplus; no partial result"
                )
            positive.append((state.probability, values, cs))
            status = "evaluated_numerically"
        state_records.append(
            {
                "name": state.name,
                "probability": _ratio(state.probability),
                "payoffs": tuple((k, payoffs[k].record()) for k in choices),
                "utilities": utilities,
                "conditional_surplus_if_state_revealed": cs,
                "status": status,
            }
        )
    expected = None
    before = revealed = gap = None
    weighted_numeric = None
    if not unknown:
        expected = tuple(
            sum((p * values[i] for p, values, cs in positive), Fraction())
            for i in range(len(choices))
        )
        utilities = utility_values(expected)
        if not all(math.isfinite(v) for v in utilities):
            raise FloatingPointError(
                "expected utility exceeds finite numerical backend; no partial result"
            )
        before = conditional_logit_surplus(
            utilities, float(problem.alpha), outside_index=outside_index
        )
        # Exact rational weighting of the already numerical binary64 state CS;
        # do not underflow a positive rational probability before multiplying.
        weighted_numeric = sum(
            (p * Fraction.from_float(cs) for p, values, cs in positive), Fraction()
        )
        revealed = float(weighted_numeric)
        gap = revealed - before
        if not all(math.isfinite(v) for v in (before, revealed, gap)):
            raise FloatingPointError("nonfinite timing comparison; no partial result")
    return deep_freeze(
        {
            "plan_id": "A.10/B0/V9",
            "definition": problem.definition,
            "payoff_unit": problem.payoff_unit,
            "alternatives_and_utility_offsets": tuple(
                (k, _ratio(v)) for k, v in problem.alternatives
            ),
            "outside": problem.outside,
            "alpha": _ratio(problem.alpha),
            "states": tuple(state_records),
            "expected_net_values": None
            if expected is None
            else tuple((k, _ratio(v)) for k, v in zip(choices, expected)),
            "single_commitment_CS_choose_before_payoff_state": before,
            "single_commitment_CS_observe_payoff_state_before_choice": revealed,
            "information_timing_difference": gap,
            "exact_weighted_binary64_state_surplus": _ratio(weighted_numeric),
            "positive_probability_unknown_input": unknown,
            "assumptions": (
                "one original person / one commitment",
                "risk-neutral quasilinear supplied payoff values",
                "common fixed menu and alpha",
                "iid unit-scale Gumbels at choice independent of payoff state",
            ),
            "utility_reference": "Outside baseline subtracted in exact rational units before Float64 conversion",
            "numerical_precision": "Exact declared rational payoff/weight arithmetic; Float64 logsum/alpha and final timing difference, no certified error interval",
            "gap_clipped_or_probabilities_normalized": False,
            "exact_market_expectation_or_rigorous_interval_certified": False,
            "state_law_preferences_or_delivered_outcomes_authenticated": False,
            "dynamic_reentry_native_choice_or_actor_modified": False,
            "reentry_searches_counted_as_independent_people": False,
            "native_lifecycle_consumer_surplus": None,
            "delivered_travel": None,
            "denied_requests": None,
            "misallocation": None,
            "primary_default_empirical_production_adopted": False,
            "scientific_budget": None,
        }
    )


def payoff_logit_record(result):
    """Owned records of declared inputs and analytic component values only."""
    return jsonable(result)
