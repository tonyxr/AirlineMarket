"""Q2 conditional resource footprints for already modeled OUTSIDE purchases."""

from collections import Counter, defaultdict

import numpy as np

from ..core.rng import StreamRegistry
from ..sim.coupled_outside import PurchaseFootprint


def build_catalogue(frame, spec, core_routes, core_carriers):
    """Keep per-market outside denominators and avoid a second arrival population."""
    if not ((frame.year == 2022) & (frame.quarter == 2)
            & (frame.split == "calibration")).all():
        raise ValueError("outside catalogue requires Q2 calibration records only")
    outside = {r["market"]: r["outside_network_Q2_passengers"]
               for r in spec.metadata["outside_traffic"]}
    available = defaultdict(set)
    for product in spec.products:
        if product.carrier in core_carriers:
            available[product.market].add(product.carrier)
    masses, denominator = Counter(), Counter()
    unknown_ticket_rows = 0
    omitted = Counter()
    for row in frame.itertuples(index=False):
        mass = float(row.passengers_expanded)
        if not np.isfinite(mass) or mass < 0:
            raise ValueError("invalid Q2 ticket mass")
        if row.market not in outside:
            omitted["outside_dated_market_scope_passengers"] += mass
            continue
        represented = row.model_path_eligible and row.path_network in available[row.market]
        if represented:
            continue
        denominator[row.market] += mass
        if not row.sequence_valid:
            omitted["invalid_sequence_outside_passengers"] += mass
            continue
        airports = row.airport_sequence.split(":")
        tickets = row.ticket_carrier_sequence.split(":")
        if any(ticket in ("--", "99") for ticket in tickets):
            unknown_ticket_rows += 1
        edges = tuple((c, a, b) for c, a, b in zip(tickets, airports, airports[1:])
                      if c in core_carriers and (c, a, b) in core_routes)
        if edges:
            masses[row.market, edges] += mass
    result = {}
    totals = Counter()
    for market, target in sorted(outside.items()):
        if abs(denominator[market] - target) > 1e-7:
            raise ValueError("outside ticket denominator changed for " + market)
        paths = []
        for (mid, edges), mass in sorted(masses.items()):
            if mid != market:
                continue
            paths.append({"core_coupon_edges": edges, "Q2_ticket_passengers": mass,
                          "probability_given_outside": mass / target})
            for c, _, _ in edges:
                totals[c] += mass
        probability = sum(p["probability_given_outside"] for p in paths)
        if probability > 1 + 1e-12:
            raise ValueError("outside footprint probability exceeds its ticket population")
        result[market] = {"outside_Q2_ticket_passengers": target,
            "paths": paths, "identified_core_path_probability": probability,
            "remaining_subtype": "no represented core-resource use identified; not proof of no physical use"}
    return {"markets": result, "core_coupon_edges_by_carrier": dict(totals),
        "unknown_ticket_rows_unattributed": unknown_ticket_rows, "omitted": dict(omitted),
        "outside_denominators_match": True,
        "arrival_population_added": False, "remaining_T100_residual_used": False,
        "scope": "Conditional distribution of identified Q2 coupon footprints given an existing outside ticket, not new passenger arrivals or a retained-revenue model."}


def generate_purchase_tape(compiled, tape, catalogue, seed):
    """Pre-draw latent outside subtypes independently of policies and private inventory."""
    routes = defaultdict(list)
    for leg in compiled.spec.legs:
        routes[leg.carrier, leg.origin, leg.dest].append(leg.id)
    for values in routes.values():
        values.sort()
    registry = StreamRegistry(seed, "empirical/outside_core_footprints/v1")
    result, counts = {}, Counter()
    for t, period in enumerate(tape.arrival_marks):
        uniforms = registry.generator(1, t).random(len(period))
        for row, (mark, uniform) in enumerate(zip(period, uniforms)):
            market = compiled.spec.markets[mark.market].id
            paths = catalogue["markets"].get(market, {}).get("paths", ())
            cumulative = 0.0
            selected = None
            for path in paths:
                cumulative += path["probability_given_outside"]
                if uniform < cumulative:
                    selected = path
                    break
            if selected is None:
                continue
            counts["identified_potential_footprints"] += 1
            edges = tuple(tuple(e) for e in selected["core_coupon_edges"])
            multiplicities = Counter(edges)
            if any(len(routes[e]) < n for e, n in multiplicities.items()):
                result[t, row] = PurchaseFootprint((), False)
                counts["unmapped_potential_footprints"] += 1
                continue
            rng = registry.generator(2, t, row)
            choices = {e: list(rng.choice(routes[e], size=n, replace=False))
                       for e, n in multiplicities.items()}
            result[t, row] = PurchaseFootprint(tuple(str(choices[e].pop()) for e in edges))
    return result, {**dict(counts), "latent_type_timing": "before native utility choice",
        "policy_independent": True, "private_inventory_used_for_tape_generation": False,
        "routing": "uniform dated flights within each known route; resource footprint sensitivity, not legal itinerary reconstruction",
        "unmapped_rule": "outside subtype is infeasible; no zero-filling or reassignment"}
