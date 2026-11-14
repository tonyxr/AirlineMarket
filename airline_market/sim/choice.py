"""Shared random-utility primitives (Eqs. 7–9)."""

from dataclasses import replace
import math
import numpy as np


def systematic_utilities(
    compiled, market: int, taste, prices_global, sellable_global
) -> np.ndarray:
    """
    Build market-local latent-class utilities with an explicit outside option (Eqs.
    3–16).
    """
    s = compiled.spec
    market_id = s.markets[market].id
    result = []
    value = taste.market_value.get(market_id, taste.market_value.get("*", 0.0))
    for j in compiled.market_products[market]:
        p = s.products[j]
        if not sellable_global[j]:
            result.append(-np.inf)
            continue
        if s.choice.connection_penalty_rule == "legacy_market_od":
            connection = s.metadata.get("connection_penalties", {}).get(market_id, 0.0)
        else:
            connection = taste.connection_penalty * max(0, len(p.legs) - 1)
        u = (
            value
            + taste.carrier_pref.get(p.carrier, 0.0)
            - taste.alpha * prices_global[j]
            - connection
        )
        u -= taste.time_penalty_per_hour * p.attributes.get("duration_h", 0.0)
        if taste.ideal_dep_hour is not None:
            u -= taste.schedule_penalty_per_hour * abs(
                p.attributes.get("dep_hour", taste.ideal_dep_hour)
                - taste.ideal_dep_hour
            )
        u += taste.product_shift.get(p.id, 0.0)
        result.append(u)
    return np.array([*result, 0.0], dtype=np.float64)


def choice_probabilities(utilities) -> np.ndarray:
    """Compute numerically stable multinomial-logit probabilities (Eqs. 3–16)."""
    u = np.asarray(utilities, dtype=np.float64)
    exp = np.exp(u - np.max(u))
    return exp / exp.sum()


def sampled_choice(utilities, gumbel_row_market) -> int:
    """
    Select the random-utility argmax using pre-indexed alternative draws (Eqs. 3–16).
    """
    return int(np.argmax(np.asarray(utilities) + gumbel_row_market))


def class_mixture_probabilities(compiled, market: int, segment: int, prices, sellable):
    """Integrate the declared latent-class utility probabilities (Eqs. 3–16)."""
    classes = compiled.spec.segments[segment].classes
    return sum(
        (
            c.weight
            * choice_probabilities(
                systematic_utilities(compiled, market, c, prices, sellable)
            )
            for c in classes
        ),
        np.zeros(len(compiled.market_products[market]) + 1),
    )


def sampled_taste(compiled, segment: int, class_index: int, draws=None):
    """Resolve a latent class or its declared mixed-logit taste draw (Eqs. 3–16)."""
    seg = compiled.spec.segments[segment]
    base = seg.classes[class_index]
    if compiled.spec.choice.model != "mixed_logit" or not seg.mixed:
        return base
    cfg = seg.mixed
    z = np.asarray(draws)
    alpha = math.exp(
        cfg.get("log_alpha_mean", math.log(base.alpha))
        + cfg.get("log_alpha_sd", 0.0) * z[0]
    )
    preferences = dict(base.carrier_pref)
    for k, c in enumerate(compiled.spec.carriers):
        preferences[c.code] = (
            preferences.get(c.code, 0.0) + cfg.get("carrier_pref_sd", 0.0) * z[k + 1]
        )
    return replace(base, alpha=alpha, carrier_pref=preferences)
