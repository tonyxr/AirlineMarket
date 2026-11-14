"""A.10/B0/V9: full-stock propagation over declared finite Poisson forecast paths.

Analyst expectation only; no world, draws, physical events, policies or planners.
"""

import math
from dataclasses import dataclass, replace
from fractions import Fraction

from airline_market.core.hashing import jsonable, spec_hash
from airline_market.core.snapshot import deep_freeze
from airline_market.sim.poisson_period import (
    NativePoissonPeriod,
    poisson_period_expectation,
)


class PoissonForecastLimitError(RuntimeError):
    """Infrastructure limit; no partial expectation or alternative forecast."""


@dataclass(frozen=True)
class PoissonForecastPath:
    name: str
    weight: Fraction
    periods: tuple

    def __post_init__(self):
        periods = tuple(self.periods)
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("explicit path name required")
        if type(self.weight) is not Fraction or self.weight < 0:
            raise ValueError("explicit nonnegative rational path weight required")
        if not periods or any(type(p) is not NativePoissonPeriod for p in periods):
            raise ValueError("nonempty complete NativePoissonPeriod path required")
        # Every template carries the SAME initial net stock. Later stock is the
        # propagated distribution, never an ignored caller intervention/reset.
        if any(p.free_capacity != periods[0].free_capacity for p in periods):
            raise ValueError("path templates must share initial stock; no stock reset")
        object.__setattr__(self, "periods", periods)


@dataclass(frozen=True)
class FinitePoissonForecast:
    paths: tuple
    start_period: int
    conditioning: str

    def __post_init__(self):
        paths = tuple(self.paths)
        if not paths or any(type(p) is not PoissonForecastPath for p in paths):
            raise ValueError("explicit nonempty finite forecast paths required")
        paths = tuple(sorted(paths, key=lambda p: p.name))
        if len({p.name for p in paths}) != len(paths):
            raise ValueError("unique forecast path names required")
        if sum((p.weight for p in paths), Fraction()) != 1:
            raise ValueError("rational path weights must sum to one; no normalization")
        c = paths[0].periods[0].compiled
        identity = spec_hash(c.spec)
        length = len(paths[0].periods)
        if any(
            len(path.periods) != length
            or any(spec_hash(p.compiled.spec) != identity for p in path.periods)
            for path in paths
        ):
            raise ValueError("common model and full period support required")
        if (
            type(self.start_period) is not int
            or self.start_period < 0
            or self.start_period + length > c.spec.horizon.n_periods
        ):
            raise ValueError("explicit in-horizon native period range required")
        if not isinstance(self.conditioning, str) or not self.conditioning.strip():
            raise ValueError("explicit joint forecast conditioning required")
        object.__setattr__(self, "paths", paths)


def _weighted_rows(weighted, field):
    """Finite-law linear expectation; no normalization/repair of raw residuals."""
    weighted = tuple(weighted)
    keys = tuple(k for k, v in weighted[0][1][field])
    rows = tuple(
        (key, math.fsum(w * dict(r[field])[key] for w, r in weighted)) for key in keys
    )
    if any(not math.isfinite(v) for key, v in rows):
        raise FloatingPointError("nonfinite forecast expectation; no partial result")
    return rows


def poisson_forecast_expectation(
    forecast: FinitePoissonForecast, *, state_limit: int, conditional_limit: int
):
    """Law of total expectation, mu[t+1]=mu[t]P[t], summed Markov rewards.

    Integrate whole declared paths, preserving their temporal dependence and
    initial stock law. Do not substitute mean intensity, stock, fare or alpha.
    Frozen single-period backend retains Float64 precision/software restrictions.
    """
    if type(forecast) is not FinitePoissonForecast:
        raise TypeError("FinitePoissonForecast required")
    if any(type(v) is not int or v < 1 for v in (state_limit, conditional_limit)):
        raise ValueError("explicit positive state and conditional limits required")
    c = forecast.paths[0].periods[0].compiled
    fields = (
        "expected_product_bookings",
        "expected_ordinary_conditional_CS",
        "expected_gross_ticket_revenue",
    )
    calls = 0
    paths = []
    for path in forecast.paths:
        leg_ids = tuple(k for k, v in path.periods[0].free_capacity)
        initial = tuple(v for k, v in path.periods[0].free_capacity)
        distribution = {initial: 1.0}
        periods = []
        for offset, template in enumerate(path.periods):
            conditionals = []
            outgoing = {}
            for stock, probability in sorted(distribution.items()):
                if calls >= conditional_limit:
                    raise PoissonForecastLimitError(
                        "forecast conditional-call limit; no partial result"
                    )
                problem = replace(template, free_capacity=tuple(zip(leg_ids, stock)))
                result = poisson_period_expectation(problem, state_limit=state_limit)
                calls += 1
                conditionals.append((probability, result))
                for state, mass in zip(
                    result["states"], result["transition_probabilities"]
                ):
                    outgoing.setdefault(state, []).append(probability * mass)
                    if len(outgoing) > state_limit:
                        raise PoissonForecastLimitError(
                            "forecast propagated-state limit; no partial result"
                        )
            distribution = {k: math.fsum(v) for k, v in sorted(outgoing.items())}
            if any(not math.isfinite(v) for v in distribution.values()):
                raise FloatingPointError("nonfinite forecast mass; no partial result")
            record = {
                "native_period": forecast.start_period + offset,
                **{field: _weighted_rows(conditionals, field) for field in fields},
                "expected_ordinary_arrivals": math.fsum(
                    w * r["expected_ordinary_arrivals"] for w, r in conditionals
                ),
                "expected_ordinary_outside_choices": math.fsum(
                    w * r["expected_ordinary_outside_choices"] for w, r in conditionals
                ),
                "end_stock_distribution": tuple(distribution.items()),
                "numerical_mass_residual": math.fsum(distribution.values()) - 1,
                "conditionals": tuple(
                    {"incoming_probability": w, "ordinary_period": r}
                    for w, r in conditionals
                ),
            }
            periods.append(record)
        totals = {
            field: _weighted_rows(((1, p) for p in periods), field) for field in fields
        }
        paths.append(
            {
                "name": path.name,
                "weight": (path.weight.numerator, path.weight.denominator),
                "float_weight": float(path.weight),
                "initial_net_free_capacity": path.periods[0].free_capacity,
                "periods": tuple(periods),
                "terminal_stock_distribution": tuple(distribution.items()),
                **totals,
                "expected_ordinary_arrivals": math.fsum(
                    p["expected_ordinary_arrivals"] for p in periods
                ),
                "expected_ordinary_outside_choices": math.fsum(
                    p["expected_ordinary_outside_choices"] for p in periods
                ),
            }
        )
    weighted = tuple((p["float_weight"], p) for p in paths)
    totals = {field: _weighted_rows(weighted, field) for field in fields}
    terminal = {}
    for w, path in weighted:
        for state, mass in path["terminal_stock_distribution"]:
            terminal.setdefault(state, []).append(w * mass)
    terminal = tuple((s, math.fsum(v)) for s, v in sorted(terminal.items()))
    initial_mean = _weighted_rows(weighted, "initial_net_free_capacity")
    bookings = dict(totals["expected_product_bookings"])
    residuals = tuple(
        (
            leg,
            initial
            - math.fsum(state[i] * mass for state, mass in terminal)
            - math.fsum(bookings[p.id] for p in c.spec.products if leg in p.legs),
        )
        for i, (leg, initial) in enumerate(initial_mean)
    )
    output = {
        "plan_id": "A.10/B0/V9",
        "input_spec_sha256": spec_hash(c.spec),
        "conditioning": forecast.conditioning,
        "native_period_range": (
            forecast.start_period,
            forecast.start_period + len(paths[0]["periods"]),
        ),
        "paths": tuple(paths),
        **totals,
        "expected_ordinary_arrivals": math.fsum(
            w * p["expected_ordinary_arrivals"] for w, p in weighted
        ),
        "expected_ordinary_outside_choices": math.fsum(
            w * p["expected_ordinary_outside_choices"] for w, p in weighted
        ),
        "terminal_stock_distribution": terminal,
        "expected_initial_net_free_capacity": initial_mean,
        "numerical_terminal_mass_residual": math.fsum(m for s, m in terminal) - 1,
        "numerical_leg_stock_identity_residual": residuals,
        "numerical_float_path_weight_residual": math.fsum(w for w, p in weighted) - 1,
        "conditional_operator_calls": calls,
        "state_limit": state_limit,
        "conditional_limit": conditional_limit,
        "precision": "Float64 native logits/expm/propagation/converted rational path weights; all raw zeroes/residuals retained",
        "monte_carlo_or_count_truncation": False,
        "probability_pruning_clipping_normalization_or_repair": False,
        "symbolic_exactness_or_rigorous_error_interval": False,
        "forecast_observability_or_joint_law_authenticated": False,
        "state_contingent_policy_or_inference_adopted": False,
        "physical_reentry_rationing_or_lifecycle_adopted": False,
        "lifecycle_consumer_surplus": None,
        "denied_requests": None,
        "misallocation": None,
        "primary_default_empirical_production_adopted": False,
        "scientific_budget": None,
    }
    return deep_freeze(output)


def poisson_forecast_record(result):
    """Owned finite forecast record; no saved-run loader or producer replay."""
    return jsonable(result)
