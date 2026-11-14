"""Chronological marked Bernoulli/Poisson arrivals (Eq. 5)."""

from dataclasses import dataclass
from .exogenous import categorical, demand_multiplier
from .tape import segment_weights


@dataclass(frozen=True)
class Request:
    """
    Chronological market/segment/class mark for one potential passenger (Eqs. 3–16).
    """

    rank: int
    market: int
    segment: int
    taste_class: int
    gumbel_row: int
    mixed_row: int


def bernoulli_probability(compiled, t: int, exo: int) -> float:
    """
    Return the exogenous-state-adjusted request probability, respecting its cap (Eqs.
    3–16).
    """
    d = compiled.spec.demand
    return min(
        d.arrival_probability_cap,
        d.arrival_probability.value_at(t)
        * d.holiday_multiplier
        * d.exogenous_multipliers.get(compiled.spec.exogenous.states[exo].name, 1.0),
    )


def period_requests(compiled, tape, t: int, exo: int, rng=None) -> list[Request]:
    """
    Select chronological potential requests from the immutable exogenous tape (Eqs.
    3–16).
    """
    s = compiled.spec
    if s.demand.mode == "single_request_bernoulli":
        u = tape.request_uniforms[t]
        if u[0] >= bernoulli_probability(compiled, t, exo):
            return []
        m = categorical(u[1], s.demand.market_weights.value_at(t))
        g = categorical(u[2], segment_weights(compiled, t, m))
        c = categorical(
            tape.class_uniform[t, 0], [c.weight for c in s.segments[g].classes]
        )
        return [Request(0, m, g, c, 0, 0)]
    max_multiplier = max(
        demand_multiplier(compiled, e) for e in range(len(s.exogenous.states))
    )
    multiplier = demand_multiplier(compiled, exo)
    requests = []
    for row, mark in enumerate(tape.arrival_marks[t]):
        if mark.thinning >= multiplier / max_multiplier:
            continue
        classes = s.segments[mark.segment].classes
        c = categorical(mark.class_uniform, [c.weight for c in classes])
        requests.append(Request(len(requests), mark.market, mark.segment, c, row, row))
    return requests
