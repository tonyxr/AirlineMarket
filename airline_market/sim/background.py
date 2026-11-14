"""Independent routes used for matched placebo observations."""

import math
from .state import BackgroundState


def quote(spec, stock: int):
    """
    Return the independent reference route's fare and current sellability (Eqs. 3–16).
    """
    rule = spec.quote_rule
    return (
        float(
            rule["base"]
            + rule.get("scarcity_premium", 0) * (stock <= rule.get("threshold", 1))
        ),
        stock > 0,
        max(0, stock),
    )


def purchase_probability(spec, stock: int) -> float:
    """Evaluate the independent reference-route purchase probability (Eqs. 3–16)."""
    fare, active, _ = quote(spec, stock)
    p = spec.purchase
    return (
        p["max_prob"] / (1 + math.exp(-(p["intercept"] - p["slope"] * fare)))
        if active
        else 0.0
    )


def step(state: BackgroundState, spec, u: float) -> None:
    """
    Advance one period using the supplied action and declared event timing (Eqs. 3–16).
    """
    fare, _, _ = quote(spec, state.stock)
    if u < purchase_probability(spec, state.stock):
        state.stock -= 1
        state.revenue += fare
