"""Posterior entropy, dependence and signal value in nats (Appendix A.4)."""

from collections import defaultdict
from math import isfinite, log
from collections.abc import Mapping, Sequence

from .events import rival_inventory


def normalized_mass(mass: Mapping) -> dict:
    """Normalize a nonnegative finite law without rounding or pruning."""
    if any(not isfinite(p) or p < 0 for p in mass.values()):
        raise ValueError("probability mass must be finite and nonnegative")
    total = sum(mass.values())
    if total <= 0:
        raise ValueError("empty probability law")
    return {
        state: probability / total
        for state, probability in mass.items()
        if probability > 0
    }


def entropy(mass: Mapping) -> float:
    """Compute Shannon entropy using natural logarithms (Eq. 30 family)."""
    return -sum(
        probability * log(probability) for probability in normalized_mass(mass).values()
    )


def mutual_information(
    mass: Mapping, left: Sequence[int] = (0,), right: Sequence[int] = (1,)
) -> float:
    """Compute dependence between two disjoint coordinate groups in a joint law."""
    if set(left) & set(right):
        raise ValueError("mutual-information coordinate groups must be disjoint")
    joint, px, py = defaultdict(float), defaultdict(float), defaultdict(float)
    for state, probability in normalized_mass(mass).items():
        x, y = tuple(state[k] for k in left), tuple(state[k] for k in right)
        joint[x, y] += probability
        px[x] += probability
        py[y] += probability
    return sum(p * log(p / (px[x] * py[y])) for (x, y), p in joint.items())


def total_correlation(mass: Mapping) -> float:
    """Measure joint inventory dependence as sum of marginal entropies minus joint."""
    law = normalized_mass(mass)
    dimensions = len(next(iter(law)))
    marginals = [defaultdict(float) for _ in range(dimensions)]
    for state, probability in law.items():
        for k, value in enumerate(state):
            marginals[k][value] += probability
    return sum(entropy(marginal) for marginal in marginals) - entropy(law)


def expected_information_gain(prior: Mapping, likelihood: Mapping) -> float:
    """Integrate KL(posterior || prior) over a declared finite signal channel."""
    joint = {}
    for state, probability in normalized_mass(prior).items():
        row = likelihood[state]
        if (
            any(not isfinite(p) or p < 0 for p in row.values())
            or abs(sum(row.values()) - 1) > 1e-12
        ):
            raise ValueError("each signal likelihood row must sum to one")
        for signal, conditional in row.items():
            if conditional:
                joint[state, signal] = probability * conditional
    return mutual_information(joint)


def inventory_mass(belief, legs: Sequence[str] | None = None) -> dict:
    """Marginalize nuisance hypotheses onto declared rival leg stocks only."""
    c, focal = belief.compiled, belief.focal
    selected = (
        list(legs)
        if legs is not None
        else [
            c.spec.legs[k].id
            for i in range(len(c.spec.carriers))
            if i != focal
            for k in c.legs_of[i]
        ]
    )
    coordinates = []
    for leg_id in selected:
        index = c.leg_index[leg_id]
        carrier = c.carrier_index[c.spec.legs[index].carrier]
        if carrier == focal:
            raise ValueError("inventory information measures require rival legs")
        coordinates.append((carrier, c.legs_of[carrier].index(index)))
    result = defaultdict(float)
    for hidden, probability in belief.mass.items():
        stocks = {
            carrier: rival_inventory(hidden, c, focal, carrier)[1]
            for carrier, _ in coordinates
        }
        result[tuple(stocks[carrier][local] for carrier, local in coordinates)] += (
            probability
        )
    return normalized_mass(result)
