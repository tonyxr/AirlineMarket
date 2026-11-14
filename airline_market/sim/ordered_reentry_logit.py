"""A.10/A.5/B0/V9: analytic one-attempt own-first/interline native choice.

Shared outside Gumbel across both stages; no repeated-person lifecycle CS sum.
"""

import math
from dataclasses import dataclass

import numpy as np

from airline_market.core.hashing import jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.sim.choice import systematic_utilities
from airline_market.sim.passenger_accounts import _binary


def _label(value):
    return isinstance(value, str) and bool(value.strip())


def _value(x):
    if isinstance(x, (bool, np.bool_)):
        raise TypeError("explicit numerical utility/alpha required")
    return float(x)


@dataclass(frozen=True)
class OrderedReentryLogit:
    own: tuple
    other: tuple
    outside_name: str
    outside_utility: float
    alpha: float
    interline: bool
    definition: str

    def __post_init__(self):
        own = tuple(sorted((k, _value(v)) for k, v in self.own))
        other = tuple(sorted((k, _value(v)) for k, v in self.other))
        outside, alpha = _value(self.outside_utility), _value(self.alpha)
        keys = [k for k, v in (*own, *other)]
        if (
            not _label(self.outside_name)
            or not _label(self.definition)
            or any(not _label(k) for k in keys)
            or len(set(keys)) != len(keys)
            or self.outside_name in keys
        ):
            raise ValueError(
                "distinct full product groups/outside and explicit definition required"
            )
        if (
            any(math.isnan(v) or v == math.inf for k, v in (*own, *other))
            or not math.isfinite(outside)
            or not math.isfinite(alpha)
            or alpha <= 0
        ):
            raise ValueError(
                "finite outside/positive alpha and finite or explicit unavailable -inf product utilities required"
            )
        if type(self.interline) is not bool:
            raise TypeError("explicit interline enable/disable required")
        object.__setattr__(self, "own", own)
        object.__setattr__(self, "other", other)
        object.__setattr__(self, "outside_utility", outside)
        object.__setattr__(self, "alpha", alpha)


def _finite(x):
    if not math.isfinite(x):
        raise FloatingPointError(
            "ordered logit exceeds finite numerical backend; no partial result"
        )
    return x


def ordered_reentry_expectation(problem: OrderedReentryLogit):
    """One fixed taste/menu: own first, interline only if outside beats all own.

    b=log(1+Wown/Wout); d=log(1+Wother/(Wout+Wown)).
    Expected selected-minus-SAME-outside utility = b+exp(-b)*d.
    This is not optimal pooled-menu logsum, dynamic lifecycle utility or CS.
    """
    if type(problem) is not OrderedReentryLogit:
        raise TypeError("OrderedReentryLogit required")
    groups = (problem.own, problem.other)
    centered = tuple(
        tuple(
            (k, -math.inf if u == -math.inf else _finite(u - problem.outside_utility))
            for k, u in rows
        )
        for rows in groups
    )
    own, other = centered
    b = _finite(float(np.logaddexp.reduce((0.0, *(v for k, v in own)))))
    log_other = (
        float(np.logaddexp.reduce(tuple(v for k, v in other))) if other else -math.inf
    )
    d = (
        0.0
        if log_other == -math.inf
        else _finite(float(np.logaddexp(0.0, _finite(log_other - b))))
    )
    if not problem.interline:
        d = 0.0
    c = _finite(b + d)
    first_outside = math.exp(-b)
    extra_gain = 0.0 if d == 0 else math.exp(_finite(math.log(d) - b))
    gain = _finite(b + extra_gain)
    probabilities = []
    for is_own, rows in ((True, own), (False, other)):
        for key, u in rows:
            p = (
                0.0
                if u == -math.inf or (not is_own and not problem.interline)
                else math.exp(_finite(u - b if is_own else (u - c) - b))
            )
            probabilities.append((key, p))
    probabilities.append((problem.outside_name, math.exp(-c)))
    raw_mass = math.fsum(p for k, p in probabilities)
    return deep_freeze(
        {
            "plan_id": "A.10/A.5/B0/V9",
            "definition": problem.definition,
            "own_utilities": tuple((k, _binary(u)) for k, u in problem.own),
            "other_utilities": tuple((k, _binary(u)) for k, u in problem.other),
            "outside": problem.outside_name,
            "outside_utility": _binary(problem.outside_utility),
            "alpha": _binary(problem.alpha),
            "interline_enabled": problem.interline,
            "choice_probabilities": tuple(probabilities),
            "raw_probability_mass": raw_mass,
            "raw_mass_residual": raw_mass - 1,
            "minimum_returned_probability": min(p for k, p in probabilities),
            "first_stage_outside_probability": first_outside,
            "own_stage_expected_gain_utility": b,
            "interline_increment_expected_gain_utility": extra_gain,
            "expected_selected_minus_same_outside_utility": gain,
            "expected_gain_divided_by_declared_alpha": _finite(gain / problem.alpha),
            "same_full_menu_expected_max_gain_utility": c if problem.interline else b,
            "raw_pooled_menu_minus_ordered_gain_utility": _finite(c - gain),
            "conditional_model": "One current fixed taste/prices/availability; iid unit-scale product and outside Gumbels, SAME draws reused across both own/interline stages",
            "precision": "Float64 logaddexp/exp and native utilities, not symbolic exactness or certified rigorous interval; underflow zeroes retained, no pruning/normalization/clipping",
            "structural_unavailable_products": tuple(
                k for k, v in (*own, *other) if v == -math.inf
            ),
            "underflow_or_inactive_zero_is_true_probability_or_variance_claim": False,
            "state_source_public_observability_or_empirical_identity_authenticated": False,
            "native_actor_world_book_RNG_or_solver_called": False,
            "native_reentry_policy_modified": False,
            "reentry_attempts_are_additional_people": False,
            "native_lifecycle_consumer_surplus": None,
            "delivered_travel": None,
            "denied_requests": None,
            "misallocation": None,
            "primary_default_empirical_production_planner_adopted": False,
            "scientific_budget": None,
        }
    )


def native_ordered_reentry_expectation(
    compiled,
    market_id,
    taste,
    *,
    prices,
    sellable,
    original_carrier,
    interline,
    definition,
):
    """Read ONLY caller-supplied current native inputs; one native utility forward.

    Caller supplies the ALREADY adjusted re-entry taste and complete NET sellable
    product mask after holds/stock. No taste draw, stock/hold update or actor call.
    """
    prices = tuple(_value(x) for x in prices)
    sellable = tuple(sellable)
    if (
        len(prices) != len(compiled.spec.products)
        or len(sellable) != len(prices)
        or any(not math.isfinite(p) for p in prices)
        or any(type(v) not in (bool, np.bool_) for v in sellable)
    ):
        raise ValueError(
            "complete finite native physical-product prices and explicit boolean net-sellable mask required"
        )
    if (
        market_id not in compiled.market_index
        or original_carrier not in compiled.carrier_index
    ):
        raise ValueError(
            "explicit existing native market and original carrier required"
        )
    market, carrier = (
        compiled.market_index[market_id],
        compiled.carrier_index[original_carrier],
    )
    utilities = systematic_utilities(compiled, market, taste, prices, sellable)
    own, other = [], []
    for u, j in zip(utilities[:-1], compiled.market_products[market]):
        if sellable[j] and not math.isfinite(u):
            raise FloatingPointError(
                "nonfinite available native utility; no unavailable-state repair"
            )
        rows = own if compiled.product_carrier[j] == carrier else other
        rows.append((compiled.spec.products[j].id, u))
    problem = OrderedReentryLogit(
        tuple(own),
        tuple(other),
        "outside",
        utilities[-1],
        taste.alpha,
        interline,
        definition,
    )
    result = ordered_reentry_expectation(problem)
    return deep_freeze(
        {
            **dict(result),
            "native_projection": {
                "spec_hash": compiled.spec_hash,
                "market": market_id,
                "original_carrier": original_carrier,
                "supplied_current_adjusted_taste": jsonable(taste),
                "complete_global_prices": tuple(
                    (p.id, _binary(v)) for p, v in zip(compiled.spec.products, prices)
                ),
                "complete_global_net_sellable": tuple(
                    (p.id, bool(v)) for p, v in zip(compiled.spec.products, sellable)
                ),
                "stock_holds_taste_or_public_source_authenticated": False,
            },
        }
    )


def ordered_reentry_record(result):
    """Owned complete inputs/numerical outputs, no saved-run or account loader."""
    return jsonable(result)
