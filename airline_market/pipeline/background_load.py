"""Declared scenario mapping of observed excluded coupon footprints to core legs.

Quarterly ticket counts do not identify booking times or dated flights. This
adapter exposes both assumptions and preserves missing-route/ambiguous load
mass rather than reallocating it or fitting the T100 residual.
"""

from collections import defaultdict

import numpy as np

from ..core.rng import StreamRegistry
from ..sim.shared_background import LoadEvent


def generate_load_events(compiled, patterns, seed, *, quarter_days=91,
                         period_weights=None):
    """Draw Poisson attempts with uniform dated-flight assignment within each route.

    The result is a resource-load sensitivity scenario, not reconstructed
    passenger itineraries. Repeated routes use distinct flights. No legal
    off-network connection or actual booking-time claim is made.
    """
    T = compiled.spec.horizon.n_periods
    weights = np.asarray(period_weights if period_weights is not None else np.ones(T), dtype=float)
    if (len(weights) != T or not np.isfinite(weights).all() or np.any(weights < 0)
            or weights.sum() <= 0 or quarter_days <= 0):
        raise ValueError("invalid booking-period weights or quarter duration")
    weights = weights / weights.sum()
    routes = defaultdict(list)
    for leg in compiled.spec.legs:
        if not leg.id.startswith("OUTSIDE_"):
            routes[leg.carrier, leg.origin, leg.dest].append(leg.id)
    for values in routes.values():
        values.sort()
    reg = StreamRegistry(seed, "empirical/shared_core_load/v1")
    events, unmatched = [], []
    excluded_mass, covered_mass = 0.0, 0.0
    for index, pattern in enumerate(patterns):
        if pattern["category"] == "represented_core":
            continue
        edges = tuple(tuple(e) for e in pattern["ordered_core_coupon_edges"])
        mass = float(pattern["expanded_passengers"])
        if not np.isfinite(mass) or mass < 0:
            raise ValueError("invalid observed pattern mass")
        excluded_mass += mass
        multiplicities = {e: edges.count(e) for e in edges}
        missing = [e for e, n in multiplicities.items() if len(routes[e]) < n]
        if missing:
            unmatched.append({"pattern_index": index, "expanded_ticket_passengers": mass,
                              "missing_or_insufficient_routes": missing})
            continue
        covered_mass += mass
        rng = reg.generator(index)
        counts = rng.poisson(mass / quarter_days * weights)
        for t, n in enumerate(counts):
            for _ in range(int(n)):
                candidates = {e: list(rng.choice(routes[e], size=k, replace=False))
                              for e, k in multiplicities.items()}
                legs = tuple(str(candidates[e].pop()) for e in edges)
                events.append(LoadEvent(len(events), t, legs, float(rng.random())))
    events.sort(key=lambda e: (e.period, e.id))
    return tuple(events), {
        "excluded_Q2_ticket_pattern_mass": excluded_mass,
        "covered_Q2_ticket_pattern_mass": covered_mass,
        "unmatched_patterns": unmatched,
        "potential_load_groups": len(events),
        "potential_load_components": sum(len(e.legs) for e in events),
        "quarter_days": quarter_days, "period_weights": weights.tolist(),
        "booking_profile": "declared weights, not identified from DB1B",
        "dated_assignment": "uniform without replacement within carrier/route; resource footprint scenario, not reconstructed itinerary or connection-feasibility certificate",
        "remaining_T100_residual_used": False, "counts_rescaled": False,
        "background_revenue_identified": False}
