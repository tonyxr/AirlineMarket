"""Posted history and explicit shopping observations/likelihoods (Eq. 16)."""

from __future__ import annotations
from dataclasses import dataclass
import copy
import math
import numpy as np
from scipy.special import ndtr, ndtri
from ..core.money import round_fare
from .ledger import inventory, sellable
from .background import quote as background_quote


@dataclass(frozen=True)
class TrueQuote:
    """
    Physical posted fare and sellability before shopping-channel transformations (Eqs.
    3–16).
    """

    product: int
    period_posted: int
    fare: float
    available: bool
    seats: int


class QuoteHistory:
    """
    Initial and period-posted true offers for explicitly lagged shopping (Eqs. 3–16).
    """

    def __init__(self):
        """Bind declared inputs and initialize QuoteHistory bookkeeping (Eqs. 3–16)."""
        self.records = {}
        self.initial = ()

    def clone(self):
        """
        Clone mutable state while sharing immutable model data and random tapes (Eqs.
        3–16).
        """
        obj = QuoteHistory()
        obj.records = dict(self.records)
        obj.initial = self.initial
        return obj

    def get(self, t: int):
        """
        Read posted history, using the declared initial quote for negative periods (Eqs.
        3–16).
        """
        return self.records.get(t, self.initial) if t < 0 else self.records[t]


def posted_quotes(world, compiled):
    """
    Construct true current fares and physical availability for all products (Eqs. 3–16).
    """
    values = []
    inventories = [inventory(cs, compiled, i) for i, cs in enumerate(world.carriers)]
    locals_by_product = {
        j: k for products in compiled.products_of for k, j in enumerate(products)
    }
    for j, p in enumerate(compiled.spec.products):
        i = int(compiled.product_carrier[j])
        local = locals_by_product[j]
        inv = inventories[i]
        use = compiled.A[i][:, local]
        seats = min(int(x) for x, u in zip(inv, use) if u)
        values.append(
            TrueQuote(
                j,
                world.t,
                float(world.carriers[i].posted_prices[local]),
                seats > 0,
                max(0, seats),
            )
        )
    for index, (state, spec) in enumerate(
        zip(world.background, compiled.spec.background)
    ):
        fare, active, seats = background_quote(spec, state.stock)
        values.append(
            TrueQuote(len(compiled.spec.products) + index, world.t, fare, active, seats)
        )
    return tuple(values)


def shop(history, request_spec, t: int, tape, compiled, request_index: int):
    """
    Apply the declared lag, cadence, rounding and display/noise channel (Eqs. 3–16).
    """
    from ..pomdp.observation import QuoteObs

    if t % request_spec.every_k_periods:
        return ()
    period = t - request_spec.lag_periods
    quotes = history.get(period)
    result = []
    for k, target in enumerate(request_spec.target_products):
        j = compiled.product_index.get(target)
        if j is None:
            j = len(compiled.spec.products) + next(
                i for i, b in enumerate(compiled.spec.background) if b.id == target
            )
        q = quotes[j]
        u = tape.quote_uniforms[min(t, len(tape.quote_uniforms) - 1), request_index, k]
        if u[0] < request_spec.missing_probability:
            result.append(QuoteObs(j, period, None, None, None, True))
            continue
        fare = q.fare
        if request_spec.fare_noise_sd:
            fare += request_spec.fare_noise_sd * float(
                ndtri(np.clip(u[1], 1e-15, 1 - 1e-15))
            )
        fare = (
            round(fare / request_spec.fare_bin) * request_spec.fare_bin
            if request_spec.fare_bin
            else round_fare(fare)
        )
        active = q.available if request_spec.availability == "flag" else None
        seats = (
            min(q.seats, request_spec.seats_cap)
            if request_spec.availability == "seats_capped"
            else None
        )
        result.append(QuoteObs(j, period, round_fare(fare), active, seats, False))
    return tuple(result)


def quote_likelihood(obs, true_quote, request_spec) -> float:
    """
    Evaluate a displayed quote's probability under its declared measurement channel
    (Eqs. 3–16).
    """
    if obs.missing:
        return request_spec.missing_probability
    p = 1 - request_spec.missing_probability
    if obs.available is not None and obs.available != true_quote.available:
        return 0.0
    if obs.seats is not None and obs.seats != min(
        true_quote.seats, request_spec.seats_cap
    ):
        return 0.0
    width = request_spec.fare_bin or 0.01
    if request_spec.fare_noise_sd:
        sd = request_spec.fare_noise_sd
        return p * float(
            ndtr((obs.fare + width / 2 - true_quote.fare) / sd)
            - ndtr((obs.fare - width / 2 - true_quote.fare) / sd)
        )
    predicted = (
        round(true_quote.fare / request_spec.fare_bin) * request_spec.fare_bin
        if request_spec.fare_bin
        else round_fare(true_quote.fare)
    )
    return p * float(round_fare(predicted) == obs.fare)
