"""Opt-in continuous-price exponential-demand benchmark using native Eq12 ledgers."""

import math
from dataclasses import dataclass

import numpy as np

from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PeriodPhase
from airline_market.sim.arrivals import Request
from airline_market.sim.fixed_fare_admission import FixedFareAdmissionEngine
from airline_market.sim.ledger import book, inventory


def polynomial(n, mu):
    """Sum the finite exponential polynomial; declared benchmark mu<=100,n<=64."""
    term = total = 1.0
    for j in range(1, n + 1):
        term *= mu / j
        total += term
    return total, term


def optimal_value(n, remaining, a, beta):
    """Solve the exponential-demand HJB via Z_n'=a/e Z_(n-1), Z_n(0)=1."""
    return math.log(polynomial(n, a * remaining / math.e)[0]) / beta


@dataclass(frozen=True)
class PricingObservation:
    """Current public continuous clock and own native inventory only."""

    elapsed: float
    remaining: float
    stock: int


def dynamic_price(obs, a, beta):
    """Post (1+log(Z_n/Z_(n-1)))/beta, before seeing the purchase mark."""
    if obs.stock == 0:
        return None
    mu = a * obs.remaining / math.e
    prev, _ = polynomial(obs.stock - 1, mu)
    term = 1.0
    for j in range(1, obs.stock + 1):
        term *= mu / j
    return (1 + math.log1p(term / prev)) / beta


def fixed_price(capacity, horizon, a, beta):
    """Fluid fixed-price heuristic, not an outcome-tuned best fixed price."""
    return max(1.0, math.log(a * horizon / capacity)) / beta


@dataclass(frozen=True)
class OpportunityTape:
    """One physical Poisson opportunity bank shared across all scales/policies."""

    seed: int
    rows: np.ndarray


def draw_tape(seed, a_max, horizon):
    """Homogeneous Poisson times plus independent scale and purchase marks."""
    rng = StreamRegistry(seed, "exponential_opportunity_times").generator()
    n = int(rng.poisson(a_max * horizon))
    times = np.sort(rng.uniform(0, horizon, size=n))
    marks = (
        StreamRegistry(seed, "exponential_opportunity_marks")
        .generator()
        .uniform(size=(n, 2))
    )
    rows = np.column_stack((times, marks))
    rows.flags.writeable = False
    return OpportunityTape(seed, rows)


def run_profile(compiled, tape, scale_fraction, a, beta, horizon, policy):
    """Price before the hidden mark; native booking at its full continuous fare.

    All native Booking.period values are0 within this single continuous selling
    horizon; arrival_rank is the shared potential-event index. Continuous times
    are retained separately. The original MNL/cent-price protocol is unchanged.
    """
    if policy not in {"fixed", "dynamic"}:
        raise ValueError("unknown exponential benchmark policy")
    base = FixedFareAdmissionEngine(compiled, [[0.0], [0.0]])
    world = base.world
    K = int(world.carriers[0].capacity[0])
    constant = fixed_price(K, horizon, a, beta)
    events = []
    for idx, (time, scale_mark, purchase_mark) in enumerate(tape.rows):
        if scale_mark >= scale_fraction:
            continue
        stock = int(inventory(world.carriers[0], compiled, 0)[0])
        obs = PricingObservation(float(time), horizon - float(time), stock)
        price = (
            dynamic_price(obs, a, beta)
            if policy == "dynamic"
            else constant
            if stock
            else None
        )
        assert price is None or price >= 1 / beta
        accepted = bool(price is not None and purchase_mark < math.exp(-beta * price))
        record = -1
        if accepted:
            b = book(world, 0, 0, price, Request(idx, 0, 0, 0, 0, 0), compiled)
            record = b.record_id
            assert b.period == b.product == b.carrier == b.segment == b.taste_class == 0
            assert b.arrival_rank == idx and b.fare == price
        after = int(inventory(world.carriers[0], compiled, 0)[0])
        assert after == stock - int(accepted)
        events.append(
            [idx, math.nan if price is None else price, stock, int(accepted), record]
        )
    world.t = 2
    world.phase = PeriodPhase.TERMINAL
    return world, np.asarray(events, dtype=np.float64).reshape(-1, 5)
