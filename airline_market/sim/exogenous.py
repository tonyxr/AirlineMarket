"""Finite public/latent condition transitions (Eq. 4)."""

import numpy as np


def categorical(u: float, weights) -> int:
    """
    Resolve one pre-indexed uniform draw against ordered category weights (Eqs. 3–16).
    """
    total = 0.0
    for i, w in enumerate(weights):
        total += w
        if u < total:
            return i
    return len(weights) - 1


def transition_row(compiled, exo: int, t: int):
    """
    Resolve the time-specific finite-state condition transition probabilities (Eqs.
    3–16).
    """
    s = compiled.spec.exogenous
    matrix = s.transitions.get(t, s.default_transition)
    if matrix is None:
        return tuple(float(j == exo) for j in range(len(s.states)))
    return matrix[exo]


def advance(compiled, exo: int, t: int, u: float) -> int:
    """Apply one finite-state exogenous transition (Eqs. 3–16)."""
    return categorical(u, transition_row(compiled, exo, t))


def public_signal(compiled, exo: int):
    """Expose only the declared public component of the exogenous state (Eqs. 3–16)."""
    state = compiled.spec.exogenous.states[exo]
    if not state.public:
        return {}
    return {"state:" + state.name: 1.0, **state.attributes}


def demand_multiplier(compiled, exo: int) -> float:
    """Return the declared holiday and exogenous demand multiplier (Eqs. 3–16)."""
    s = compiled.spec
    return s.demand.holiday_multiplier * s.demand.exogenous_multipliers.get(
        s.exogenous.states[exo].name, 1.0
    )
