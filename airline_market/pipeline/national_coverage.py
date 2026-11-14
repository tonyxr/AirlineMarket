"""Trace national Q2 coupon-edge demand on the already scoped core routes.

No full CSV is extracted. Ticket and operating sequences remain distinct;
the output is a coupon-edge coverage audit, not identified background demand
or exact flown-segment reconciliation. No residual is imputed or rescaled.
"""

from collections import Counter, defaultdict
from pathlib import Path
import math

from .scoped_io import batches
from ..campaign.artifacts import atomic_json, file_hash

COLUMNS = ["ItinID", "MktID", "Year", "Quarter", "Origin", "Dest", "AirportGroup",
           "MktCoupons", "TkCarrierGroup", "OpCarrierGroup", "RPCarrier", "Passengers",
           "ItinGeoType", "MktGeoType"]


def classify_market(row, scope, selected, routes):
    """Keep per-coupon network identity and categorize only the retained resource use."""
    if any(not isinstance(row.get(k), str) or not row[k]
           for k in ("AirportGroup", "TkCarrierGroup", "OpCarrierGroup")):
        return {"reason": "missing_ordered_sequence"}
    try:
        n = int(row["MktCoupons"])
        passengers = float(row["Passengers"])
    except (ValueError, TypeError):
        return {"reason": "invalid_coupon_count_or_passengers"}
    if not math.isfinite(passengers) or passengers < 0:
        return {"reason": "invalid_passengers"}
    airports = str(row["AirportGroup"]).split(":")
    ticket = str(row["TkCarrierGroup"]).split(":")
    operator = str(row["OpCarrierGroup"]).split(":")
    if not (len(airports) == n + 1 and airports[0] == row["Origin"]
            and airports[-1] == row["Dest"] and len(ticket) == len(operator) == n
            and all(airports) and all(ticket) and all(operator)):
        return {"reason": "malformed_ordered_sequence", "sample_passengers": passengers}
    core = tuple((c, a, b) for c, a, b in zip(ticket, airports, airports[1:])
                 if (c, a, b) in routes)
    unknown = tuple((op, a, b) for tk, op, a, b in zip(ticket, operator, airports, airports[1:])
                    if tk in ("--", "99") and any((c, a, b) in routes for c in scope["carriers"]))
    pool, hub = set(scope["airport_pool"]), scope["hub_candidate"]
    inside = row["Origin"] in pool and row["Dest"] in pool
    mid = row["Origin"] + "-" + row["Dest"]
    same_core = len(set(ticket)) == 1 and ticket[0] in scope["carriers"]
    represented = (inside and mid in selected and same_core
                   and ((n == 1 and hub in (airports[0], airports[-1]))
                        or (n == 2 and airports[1] == hub)))
    category = ("represented_core" if represented else "off_pool_od" if not inside
                else "in_pool_unselected_market" if mid not in selected
                else "in_pool_unrepresented_path")
    unmatched = tuple((c, a, b) for c, a, b in zip(ticket, airports, airports[1:])
                      if c in scope["carriers"] and hub in (a, b) and a in pool and b in pool
                      and (c, a, b) not in routes)
    pairs = tuple((tk, op, a, b) for tk, op, a, b in zip(ticket, operator, airports, airports[1:])
                  if (tk, a, b) in routes)
    return {"reason": None, "core": core, "unknown_ticket_edges": unknown,
            "unmatched_core_edges": unmatched, "ticket_operator_pairs": pairs,
            "category": category, "ticketing_class": "same_core_network" if same_core else "mixed_or_other",
            "coupons": n, "sample_passengers": passengers,
            "repeated_core_route": len(set(core)) != len(core)}


def audit_national(config, context):
    """Scan one existing Q2 archive and retain small route/pattern aggregates only."""
    import pandas as pd

    scope = __import__("json").loads(Path(config["scope"]).read_text())
    selected = set(pd.read_csv(config["selected_markets"]).market)
    capacity = pd.read_csv(config["capacity_targets"])
    if not (capacity.year.eq(2022).all() and capacity.month.isin([4, 5, 6]).all()):
        raise ValueError("coverage audit requires Q2 capacity targets")
    core_cap = capacity[capacity.marketing_carrier.isin(scope["carriers"])]
    targets = core_cap.groupby(["marketing_carrier", "origin", "dest"]).allocated_passengers.sum()
    routes = set(targets.index)
    pattern_counts, flow_counts = defaultdict(lambda: [0, 0.0]), defaultdict(lambda: [0, 0.0])
    reasons, geography, unknown_ticket = Counter(), Counter(), Counter()
    unmatched_edges, operator_pairs = Counter(), Counter()
    counters = {"rows_scanned": 0, "hub_path_rows": 0, "rows_touching_core": 0}
    context.counters = lambda: dict(counters)
    for batch in batches(config["db1b_q2_archive"], COLUMNS):
        frame = batch.to_pandas()
        counters["rows_scanned"] += len(frame)
        if not (pd.to_numeric(frame.Year).eq(2022).all()
                and pd.to_numeric(frame.Quarter).eq(2).all()):
            raise ValueError("national input contains non-Q2 records")
        contains = frame.AirportGroup.str.contains(r"(?:^|:)" + scope["hub_candidate"] + r"(?::|$)", na=False)
        for row in frame.loc[contains].to_dict(orient="records"):
            counters["hub_path_rows"] += 1
            classified = classify_market(row, scope, selected, routes)
            if classified["reason"]:
                reasons[classified["reason"]] += 1
                continue
            for edge in classified["unknown_ticket_edges"]:
                unknown_ticket[edge] += classified["sample_passengers"]
            for edge in classified["unmatched_core_edges"]:
                unmatched_edges[edge] += classified["sample_passengers"]
            for edge in classified["ticket_operator_pairs"]:
                operator_pairs[edge] += classified["sample_passengers"]
            core = classified["core"]
            if not core:
                continue
            counters["rows_touching_core"] += 1
            category = classified["category"]
            mass = classified["sample_passengers"]
            geography[(str(row["ItinGeoType"]), str(row["MktGeoType"]))] += 1
            pattern = (category, classified["ticketing_class"], classified["coupons"],
                       classified["repeated_core_route"], core)
            pattern_counts[pattern][0] += 1
            pattern_counts[pattern][1] += mass
            for c, origin, dest in core:
                flow_counts[c, origin, dest, category][0] += 1
                flow_counts[c, origin, dest, category][1] += mass
    flows = pd.DataFrame([{"carrier": c, "origin": o, "dest": d, "category": category,
                          "coupon_edge_rows": n, "sample_passengers": mass,
                          "expanded_coupon_edge_passengers": 10 * mass}
                         for (c, o, d, category), (n, mass) in sorted(flow_counts.items())],
                        columns=["carrier", "origin", "dest", "category", "coupon_edge_rows",
                                 "sample_passengers", "expanded_coupon_edge_passengers"])
    flows.to_csv(context.directory / "route_category_flows.csv", index=False)
    patterns = [{"category": cat, "ticketing_class": cls, "coupons": coupons,
                 "repeated_core_route": repeat, "ordered_core_coupon_edges": edges,
                 "market_rows": n, "sample_passengers": mass, "expanded_passengers": 10 * mass}
                for (cat, cls, coupons, repeat, edges), (n, mass) in sorted(pattern_counts.items())]
    atomic_json(context.directory / "core_route_patterns.json", patterns)
    by_route = flows.groupby(["carrier", "origin", "dest"]).expanded_coupon_edge_passengers.sum()
    reconciliation = []
    for key, target in targets.items():
        coupon = float(by_route.get(key, 0))
        reconciliation.append({"carrier": key[0], "origin": key[1], "dest": key[2],
            "national_expanded_coupon_edge_passengers": coupon,
            "allocated_T100_passengers": float(target), "coupon_minus_T100": coupon - target})
    pd.DataFrame(reconciliation).to_csv(context.directory / "route_reconciliation.csv", index=False)
    by_carrier = []
    for c in scope["carriers"]:
        group = flows[flows.carrier == c]
        categories = group.groupby("category").expanded_coupon_edge_passengers.sum().to_dict()
        total, target = float(group.expanded_coupon_edge_passengers.sum()), float(targets.loc[c].sum())
        by_carrier.append({"carrier": c, "expanded_coupon_edges_by_category": categories,
                           "national_expanded_coupon_edges": total, "allocated_T100_passengers": target,
                           "national_coupon_to_T100": total / target,
                           "national_coupon_minus_T100": total - target})
    unknown_rows = [{"operating_carrier": key[0], "origin": key[1], "dest": key[2],
                     "sample_passengers": value, "expanded_passengers": 10 * value,
                     "attributed_to_network": False} for key, value in sorted(unknown_ticket.items())]
    atomic_json(context.directory / "unknown_ticket_edges.json", unknown_rows)
    atomic_json(context.directory / "unmatched_core_route_edges.json", [
        {"carrier": c, "origin": o, "dest": d, "expanded_passengers": 10 * mass}
        for (c, o, d), mass in sorted(unmatched_edges.items())])
    atomic_json(context.directory / "ticket_operator_pairs.json", [
        {"ticket_carrier": tk, "operating_carrier": op, "origin": o, "dest": d,
         "expanded_coupon_edge_passengers": 10 * mass}
        for (tk, op, o, d), mass in sorted(operator_pairs.items())])
    summary = {"status": "completed", **counters, "carrier_coverage": by_carrier,
        "invalid_hub_rows": dict(reasons), "core_route_count": len(routes),
        "patterns": len(patterns), "multi_core_edge_patterns": sum(len(p["ordered_core_coupon_edges"]) > 1 for p in patterns),
        "geography_row_counts": [{"itinerary_geo_type": key[0], "market_geo_type": key[1],
                                  "rows": value} for key, value in sorted(geography.items())],
        "input_hashes": {config[key]: file_hash(config[key]) for key in
                         ["scope", "selected_markets", "capacity_targets", "db1b_q2_archive"]},
        "background_process_fitted": False, "counts_rescaled": False,
        "scope": "Existing national Q2 DB1B coupon-edge flows on scoped core network routes. Ordered tickets/operators remain distinct. Coupon edges are not guaranteed flown non-stop segments; no flight-number/via-point reconstruction. Geographic/source coverage diagnostic, not identified background arrivals or exact T100 reconciliation. No Q3 targets or new raw acquisition."}
    atomic_json(context.directory / "coverage_summary.json", summary)
    return summary
