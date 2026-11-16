"""A.10/A.5/B0/V9: declared native Bernoulli arrival/person/stock law.

Uniform53 arrival marks and independent ideal MNL shocks are explicit assumptions.
No-arrival is distinct from one original person's outside choice.
"""

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from airline_market.core.hashing import jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.sim.arrivals import bernoulli_probability
from airline_market.sim.choice import choice_probabilities, systematic_utilities
from airline_market.sim.class_mark_law import _grid_pushforward
from airline_market.sim.ledger import sellable
from airline_market.sim.passenger_accounts import _binary, _encode
from airline_market.sim.payoff_logit import _ratio
from airline_market.sim.tape import segment_weights
from airline_market.spec.compiled import CompiledMarket

MARK_LAW = "iid-uniform53-arrival-marks-and-independent-ideal-MNL"
SIZE = 1 << 53


@dataclass(frozen=True)
class NativeBernoulliPersonLaw:
    joint: tuple | None
    details: object


def native_bernoulli_person_law(
    compiled,
    *,
    period,
    exogenous_state,
    prices,
    net_free_stock,
    mark_law,
    conditioning,
    fixed_before_current_private_marks,
    row_limit,
    kernel_call_limit,
):
    """Integrate native arrival/market/segment/class and ONE ordinary choice.

    Complete zero/inactive support remains. Exact mark-grid fractions weight
    numerical native MNL probabilities; no native seed or Gumbel-law proof.
    """
    if type(compiled) is not CompiledMarket:
        raise TypeError("native CompiledMarket required")
    s = compiled.spec
    if s.demand.mode != "single_request_bernoulli":
        raise ValueError("single-request Bernoulli only; no count-law fallback")
    if s.choice.model != "latent_class_mnl" or any(
        g.mixed is not None for g in s.segments
    ):
        raise ValueError("finite native classes only; no mixed fallback")
    if type(period) is not int or not 0 <= period < s.horizon.n_periods:
        raise ValueError("native in-horizon request period required")
    if type(exogenous_state) is not int or not 0 <= exogenous_state < len(
        s.exogenous.states
    ):
        raise ValueError("explicit native exogenous state required")
    if mark_law != MARK_LAW or fixed_before_current_private_marks is not True:
        raise ValueError(
            "explicit independent mark law/fixed pre-private context required"
        )
    if not isinstance(conditioning, str) or not conditioning.strip():
        raise ValueError("complete pre-current-arrival/choice conditioning required")
    if any(type(n) is not int or n < 1 for n in (row_limit, kernel_call_limit)):
        raise ValueError("positive all-support infrastructure limits required")
    prices = tuple(prices)
    if len(prices) != len(s.products) or any(
        isinstance(p, (bool, np.bool_)) for p in prices
    ):
        raise ValueError("complete finite global fares required")
    prices = tuple(float(p) for p in prices)
    if any(not math.isfinite(p) for p in prices):
        raise ValueError("complete finite global fares required")
    stock = tuple(net_free_stock)
    if len(stock) != len(s.legs) or any(
        type(v) is not int or not 0 <= v <= np.iinfo(np.int64).max for v in stock
    ):
        raise ValueError("complete integer NET free leg stock required")
    kernels_required = len(s.markets) * sum(len(g.classes) for g in s.segments)
    rows_required = 1 + sum(
        (len(compiled.market_products[m]) + 1) * len(g.classes)
        for m in range(len(s.markets))
        for g in s.segments
    )
    if kernels_required > kernel_call_limit or rows_required > row_limit:
        raise RuntimeError(
            "ALL support exceeds infrastructure limit; no partial law/retry"
        )

    grid_records = []

    def grid(label, raw):
        weights = tuple(float(w) for w in raw)
        rows, calls = _grid_pushforward(weights, 53)
        probs = tuple(Fraction(n, SIZE) for k, lo, hi, n, first, last in rows)
        exact = sum((Fraction.from_float(w) for w in weights), Fraction())
        grid_records.append(
            {
                "label": label,
                "raw_weights": tuple(_binary(w) for w in weights),
                "intervals": rows,
                "probabilities": tuple(_ratio(w) for w in probs),
                "probability_minus_raw_weight": tuple(
                    _ratio(p - Fraction.from_float(w)) for p, w in zip(probs, weights)
                ),
                "exact_raw_weight_mass": _ratio(exact),
                "exact_raw_weight_mass_residual": _ratio(exact - 1),
                "all_native_selector_calls": calls,
            }
        )
        return probs

    probability = float(bernoulli_probability(compiled, period, exogenous_state))
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("finite native arrival threshold in [0,1] required")
    count = math.ceil(Fraction.from_float(probability) * SIZE)
    arrival = Fraction(count, SIZE)
    markets = grid(("market", period), s.demand.market_weights.value_at(period))
    segments = tuple(
        grid(("segment", m, period), segment_weights(compiled, period, m))
        for m in range(len(s.markets))
    )
    classes = tuple(
        grid(("class", g.id), (c.weight for c in g.classes)) for g in s.segments
    )
    mask = tuple(
        bool(v) for v in sellable(np.asarray(stock, dtype=np.int64), compiled.A_global)
    )
    outside = len(s.products)
    joint = [(None, None, None, None, stock, 1 - arrival)]
    kernel_records, outcome_records, underflows = [], [], []
    for m, mw in enumerate(markets):
        for g, gw in enumerate(segments[m]):
            for k, cw in enumerate(classes[g]):
                weight = arrival * mw * gw * cw
                utilities = tuple(
                    float(u)
                    for u in systematic_utilities(
                        compiled, m, s.segments[g].classes[k], prices, mask
                    )
                )
                columns = (*compiled.market_products[m], outside)
                active = tuple(j == outside or mask[j] for j in columns)
                if len(utilities) != len(columns) or any(
                    not math.isfinite(u)
                    for u, enabled in zip(utilities, active)
                    if enabled
                ):
                    raise FloatingPointError(
                        "nonfinite/missing active native utility; no partial law"
                    )
                p = tuple(float(v) for v in choice_probabilities(utilities))
                if len(p) != len(columns) or any(
                    not math.isfinite(v) or not 0 <= v <= 1 for v in p
                ):
                    raise FloatingPointError(
                        "invalid native MNL probabilities; no repair"
                    )
                if any(v != 0 for v, enabled in zip(p, active) if not enabled):
                    raise FloatingPointError(
                        "inactive native product has mass; no silent drop"
                    )
                mass = sum((Fraction.from_float(v) for v in p), Fraction())
                kernel_records.append(
                    {
                        "market": m,
                        "segment": g,
                        "class": k,
                        "columns": columns,
                        "active": active,
                        "utilities": utilities,
                        "probabilities": p,
                        "raw_float_mass": math.fsum(p),
                        "exact_binary64_mass": _ratio(mass),
                        "exact_binary64_mass_residual": _ratio(mass - 1),
                    }
                )
                for j, enabled, v in zip(columns, active, p):
                    needed = weight > 0 and enabled
                    if needed and v == 0:
                        underflows.append((m, g, k, j))
                    after = None
                    if enabled:
                        after = (
                            stock
                            if j == outside
                            else tuple(
                                x - int(a)
                                for x, a in zip(stock, compiled.A_global[:, j])
                            )
                        )
                    value = weight * Fraction.from_float(v)
                    joint.append((m, g, k, j, after, value))
                    outcome_records.append(
                        {
                            "original_potential_key": (period, 0, m),
                            "segment": g,
                            "class": k,
                            "choice": j,
                            "active": enabled,
                            "exact_arrival_type_weight": _ratio(weight),
                            "needed_positive_choice_underflow": needed and v == 0,
                            "after_net_stock": after,
                            "weighted_binary64_choice_mass": _ratio(value),
                        }
                    )
    mass = sum((row[-1] for row in joint), Fraction())
    assert len(joint) == rows_required and len(kernel_records) == kernels_required
    return NativeBernoulliPersonLaw(
        None if underflows else tuple(joint),
        deep_freeze(
            {
                "plan_id": "A.10/A.5/B0/V9",
                "spec_hash": compiled.spec_hash,
                "complete_native_model": jsonable(s),
                "period": period,
                "exogenous_state": exogenous_state,
                "mark_law": mark_law,
                "conditioning": conditioning,
                "global_fixed_prices": prices,
                "initial_net_free_stock": stock,
                "global_net_sellable": mask,
                "native_arrival_threshold": _binary(probability),
                "arrival_grid_count": str(count),
                "grid_size": str(SIZE),
                "exact_arrival_probability": _ratio(arrival),
                "arrival_probability_minus_native_threshold": _ratio(
                    arrival - Fraction.from_float(probability)
                ),
                "no_arrival_mass": _ratio(1 - arrival),
                "outside_column": outside,
                "native_selector_grids": tuple(grid_records),
                "kernels": tuple(kernel_records),
                "outcomes": tuple(outcome_records),
                "raw_joint": tuple((*row[:-1], _ratio(row[-1])) for row in joint),
                "raw_float_mass": float(mass),
                "exact_weighted_binary64_mass": _ratio(mass),
                "exact_mass_residual": _ratio(mass - 1),
                "needed_underflows": tuple(underflows),
                "infrastructure_rows": rows_required,
                "infrastructure_kernels": kernels_required,
                "row_limit": row_limit,
                "kernel_call_limit": kernel_call_limit,
                "uniform53_joint_mark_independence_and_ideal_MNL_declared_not_authenticated": True,
                "discrete_Gumbel_seed_population_public_or_empirical_law_authenticated": False,
                "physical_future_lifecycle_CS_or_delivered_service_adopted": False,
                "scientific_budget": None,
            }
        ),
    )


def bernoulli_person_record(result):
    if type(result) is not NativeBernoulliPersonLaw:
        raise TypeError("NativeBernoulliPersonLaw required")
    return _encode(jsonable(result.details))
