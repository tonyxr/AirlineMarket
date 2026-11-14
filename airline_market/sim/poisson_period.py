"""A.10/B0/V9: analytic fixed-price ordinary Poisson depletion block.

Finite-class/native MNL, explicit conditional forecast and net free stock. No
world/tape, re-entry, physical events, rationing, controller or planner adoption.
"""

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.linalg import expm

from airline_market.core.hashing import jsonable, spec_hash
from airline_market.core.snapshot import deep_freeze
from airline_market.sim.choice import choice_probabilities, systematic_utilities
from airline_market.sim.segmented_welfare import conditional_logit_surplus

MAX_EXPONENTIAL_DIMENSION = 256


class PoissonPeriodLimitError(RuntimeError):
    """Explicit infrastructure/backend limit; no partial transition/reward result."""


@dataclass(frozen=True)
class NativePoissonPeriod:
    compiled: object
    prices: tuple
    free_capacity: tuple
    market_segment_intensities: tuple
    forecast_definition: str

    def __post_init__(self):
        c = self.compiled
        if c.spec.demand.mode != "poisson":
            raise ValueError("native Poisson ordinary block required")
        if any(s.mixed for s in c.spec.segments):
            raise ValueError("explicit finite classes only; no mixed-logit fallback")
        for seg in c.spec.segments:
            if sum((Fraction(str(t.weight)) for t in seg.classes), Fraction(0)) != 1:
                raise ValueError(
                    "canonical finite class weights must sum to one; no normalization"
                )
            if any(
                not math.isfinite(t.alpha) or t.alpha <= 0 or t.weight < 0
                for t in seg.classes
            ):
                raise ValueError(
                    "finite positive class alpha/nonnegative weights required"
                )
        prices = tuple(sorted((p, float(v)) for p, v in self.prices))
        stock = tuple(sorted(self.free_capacity))
        rates = tuple(
            sorted((tuple(key), float(v)) for key, v in self.market_segment_intensities)
        )
        if (
            len(dict(prices)) != len(prices)
            or set(dict(prices)) != {p.id for p in c.spec.products}
            or any(not math.isfinite(v) for p, v in prices)
        ):
            raise ValueError(
                "complete finite fixed physical-product price vector required"
            )
        if (
            len(dict(stock)) != len(stock)
            or set(dict(stock)) != set(c.leg_index)
            or any(type(v) is not int or v < 0 for p, v in stock)
        ):
            raise ValueError(
                "complete explicit nonnegative integer net free leg stock required"
            )
        support = {(m.id, s.id) for m in c.spec.markets for s in c.spec.segments}
        if (
            len(dict(rates)) != len(rates)
            or set(dict(rates)) != support
            or any(not math.isfinite(v) or v < 0 for key, v in rates)
        ):
            raise ValueError(
                "complete explicit nonnegative finite market x segment conditional intensities required"
            )
        if not math.isfinite(math.fsum(v for key, v in rates)):
            raise ValueError("finite total intensity required")
        if (
            not isinstance(self.forecast_definition, str)
            or not self.forecast_definition.strip()
        ):
            raise ValueError("explicit conditional forecast definition required")
        object.__setattr__(self, "prices", prices)
        object.__setattr__(self, "free_capacity", stock)
        object.__setattr__(self, "market_segment_intensities", rates)


def poisson_period_expectation(problem: NativePoissonPeriod, *, state_limit: int):
    """Q[x,x-a_j]=sum_ms lambda_ms P_j(x); E rewards via exp([[Q,R],[0,0]]).

    Duration is ONE native period, never elapsed hours. This integrates the full
    Poisson count law without cohort sampling/count truncation. Native logits and
    the finite matrix exponential are Float64 numerical evaluations, not exact
    rational/transcendental certificates. All raw numerical residuals retained.
    """
    if type(problem) is not NativePoissonPeriod:
        raise TypeError("NativePoissonPeriod required")
    if type(state_limit) is not int or state_limit < 1:
        raise ValueError("explicit positive state limit required")
    c = problem.compiled
    leg_ids = tuple(k for k, v in problem.free_capacity)
    initial = tuple(v for k, v in problem.free_capacity)
    paths = tuple(tuple(leg_ids.index(k) for k in p.legs) for p in c.spec.products)
    prices = np.array([dict(problem.prices)[p.id] for p in c.spec.products])
    groups = tuple((m.id, s.id) for m in c.spec.markets for s in c.spec.segments)
    rates = dict(problem.market_segment_intensities)
    total_intensity = math.fsum(rates.values())
    market_active = {
        m.id: any(rates[(m.id, s.id)] > 0 for s in c.spec.segments)
        for m in c.spec.markets
    }
    states = [initial]
    indices = {initial: 0}
    transitions = []
    rewards = []
    cursor = 0
    while cursor < len(states):
        stock = states[cursor]
        feasible = np.array([all(stock[k] > 0 for k in path) for path in paths])
        product_rates = np.zeros(len(paths))
        cs_rates = np.zeros(len(groups))
        for m, market in enumerate(c.spec.markets):
            for s, seg in enumerate(c.spec.segments):
                intensity = rates[(market.id, seg.id)]
                if intensity == 0:
                    continue
                g = groups.index((market.id, seg.id))
                for taste in seg.classes:
                    utilities = systematic_utilities(c, m, taste, prices, feasible)
                    probs = choice_probabilities(utilities)
                    weight = intensity * taste.weight
                    for local, j in enumerate(c.market_products[m]):
                        product_rates[j] += weight * probs[local]
                    cs_rates[g] += weight * conditional_logit_surplus(
                        utilities, taste.alpha
                    )
        if not np.isfinite(product_rates).all() or not np.isfinite(cs_rates).all():
            raise FloatingPointError("nonfinite native rate/reward; no partial result")
        outgoing = []
        for j, rate in enumerate(product_rates):
            if not feasible[j] or not market_active[c.spec.products[j].market]:
                continue
            nxt = list(stock)
            for k in paths[j]:
                nxt[k] -= 1
            nxt = tuple(nxt)
            if nxt not in indices:
                if len(states) >= state_limit:
                    raise PoissonPeriodLimitError(
                        "ordinary Poisson state limit; no partial result"
                    )
                if (
                    len(states) + 1 + len(paths) + len(groups)
                    > MAX_EXPONENTIAL_DIMENSION
                ):
                    raise PoissonPeriodLimitError(
                        "ordinary Poisson dense exponential dimension limit; no partial result"
                    )
                indices[nxt] = len(states)
                states.append(nxt)
            outgoing.append((indices[nxt], float(rate)))
        transitions.append(tuple(outgoing))
        rewards.append(np.r_[product_rates, cs_rates])
        cursor += 1
    n = len(states)
    k = len(paths) + len(groups)
    dimension = n + k
    # This small dense backend stays below SciPy's n>=400 norm-estimation branch.
    # A larger operator needs a NEW tested backend, never hidden RNG or truncation.
    if dimension > MAX_EXPONENTIAL_DIMENSION:
        raise PoissonPeriodLimitError(
            "ordinary Poisson dense exponential dimension limit; no partial result"
        )
    block = np.zeros((dimension, dimension))
    for i, outgoing in enumerate(transitions):
        for target, rate in outgoing:
            block[i, target] += rate
        block[i, i] = -math.fsum(rate for target, rate in outgoing)
        block[i, n:] = rewards[i]
    # Row-vector convention: top-right block integrates reward rates over [0,1].
    row = expm(block)[0]
    if not np.isfinite(row).all():
        raise FloatingPointError("nonfinite matrix exponential; no partial result")
    distribution = row[:n]
    bookings = row[n : n + len(paths)]
    cs = row[n + len(paths) :]
    expected_remaining = np.array(states, dtype=float).T @ distribution
    booked_leg_use = np.zeros(len(leg_ids))
    for j, path in enumerate(paths):
        for leg in path:
            booked_leg_use[leg] += bookings[j]
    output = {
        "plan_id": "A.10/B0/V9",
        "method": "finite CTMC/Markov reward matrix exponential, chronological ordinary native block",
        "input_spec_sha256": spec_hash(c.spec),
        "fixed_prices": problem.prices,
        "initial_net_free_capacity": problem.free_capacity,
        "conditional_market_segment_intensities": problem.market_segment_intensities,
        "forecast_definition": problem.forecast_definition,
        "native_period_duration": 1,
        "states": tuple(states),
        "transition_probabilities": tuple(float(v) for v in distribution),
        "expected_product_bookings": tuple(
            (p.id, float(bookings[j])) for j, p in enumerate(c.spec.products)
        ),
        "expected_ordinary_conditional_CS": tuple(
            (group, float(cs[g])) for g, group in enumerate(groups)
        ),
        "expected_gross_ticket_revenue": tuple(
            (
                carrier.code,
                float(math.fsum(bookings[j] * prices[j] for j in c.products_of[i])),
            )
            for i, carrier in enumerate(c.spec.carriers)
        ),
        "expected_ordinary_arrivals": total_intensity,
        "expected_ordinary_outside_choices": total_intensity
        - float(math.fsum(bookings)),
        "numerical_probability_mass_residual": float(math.fsum(distribution) - 1),
        "numerical_minimum_probability": float(min(distribution)),
        "numerical_leg_stock_identity_residual": tuple(
            (leg, float(initial[i] - expected_remaining[i] - booked_leg_use[i]))
            for i, leg in enumerate(leg_ids)
        ),
        "state_count": n,
        "matrix_dimension": dimension,
        "caller_state_limit": state_limit,
        "backend_max_dimension": MAX_EXPONENTIAL_DIMENSION,
        "input_observability_or_forecast_authentication": False,
        "monte_carlo_or_poisson_count_truncation": False,
        "symbolic_exactness_or_rigorous_error_interval_certified": False,
        "numerical_precision": "Native Float64 logits and matrix exponential; underflow zeros are not certified true zeros",
        "numerical_clipping_or_normalization": False,
        "ordinary_block_only": True,
        "reentry_shock_rationing_or_planner_adopted": False,
        "lifecycle_consumer_surplus": None,
        "denied_requests": None,
        "misallocation": None,
        "primary_B4_welfare_adopted": False,
        "scientific_budget": None,
    }
    return deep_freeze(output)


def poisson_period_record(result):
    """Owned analytic ordinary-block record; no physical execution or repair."""
    return jsonable(result)
