"""Summarize sealed national-to-core coverage without fitting missing demand."""

import argparse
import json
from pathlib import Path

import pandas as pd

from airline_market.campaign.artifacts import atomic_json, file_hash


def summarize(root, previous):
    """Check the independent represented-path totals and keep residuals explicit."""
    directories = list(root.glob("cells/*/attempt-*/COMPLETE.json"))
    if len(directories) != 1:
        raise ValueError("expected exactly one sealed national audit")
    directory = directories[0].parent
    receipt = json.loads((directory / "COMPLETE.json").read_text())
    if file_hash(directory / "results.json") != receipt["results_sha256"]:
        raise ValueError("sealed result hash mismatch")
    result = json.loads((directory / "results.json").read_text())
    if result["status"] != "completed":
        raise ValueError("audit did not complete")
    old = {x["carrier"]: x for x in json.loads(previous.read_text())["rows"]}
    patterns = json.loads((directory / "core_route_patterns.json").read_text())
    routes = pd.read_csv(directory / "route_reconciliation.csv")
    flows = pd.read_csv(directory / "route_category_flows.csv")
    pattern_mass = sum(len(x["ordered_core_coupon_edges"]) * x["expanded_passengers"]
                       for x in patterns)
    if pattern_mass != flows.expanded_coupon_edge_passengers.sum():
        raise ValueError("pattern incidence does not conserve coupon-edge totals")
    rows = []
    for x in result["carrier_coverage"]:
        carrier = x["carrier"]
        represented = x["expanded_coupon_edges_by_category"]["represented_core"]
        if represented != old[carrier]["represented_ticket_leg_boardings"]:
            raise ValueError("national represented paths differ from the scoped audit")
        target = x["allocated_T100_passengers"]
        if abs(target - old[carrier]["allocated_T100_Q2_passenger_boardings"]) > 1e-8:
            raise ValueError("capacity comparison population changed")
        initial_gap = target - represented
        additional = x["national_expanded_coupon_edges"] - represented
        r = routes[routes.carrier == carrier]
        rows.append({**x, "represented_to_T100": represented / target,
                     "additional_observed_coupon_edges": additional,
                     "additional_fraction_of_initial_gap": additional / initial_gap,
                     "routes_above_T100": int(r.coupon_minus_T100.gt(0).sum()),
                     "represented_paths_match_previous_audit": True})
    unknown = json.loads((directory / "unknown_ticket_edges.json").read_text())
    mixed_core = [x for x in patterns if len({z[0] for z in x["ordered_core_coupon_edges"]}) > 1]
    report = {
        "status": "completed_coverage_diagnostic",
        "carrier_coverage": rows,
        "ordered_patterns": len(patterns),
        "multi_core_edge_patterns": result["multi_core_edge_patterns"],
        "cross_core_carrier_patterns": len(mixed_core),
        "cross_core_carrier_expanded_ticket_passengers": sum(x["expanded_passengers"] for x in mixed_core),
        "unknown_ticket_expanded_coupon_edges_unattributed": sum(x["expanded_passengers"] for x in unknown),
        "pattern_route_mass_conserved": True,
        "audit_wall_seconds": result["wall_seconds"],
        "audit_peak_rss_bytes": result["peak_rss_bytes"],
        "provenance": {str(p): file_hash(p) for p in [directory / "results.json", previous,
            directory / "core_route_patterns.json", directory / "route_reconciliation.csv",
            directory / "route_category_flows.csv", directory / "unknown_ticket_edges.json"]},
        "background_process_fitted": False,
        "remaining_residual_imputed": False,
        "next_operation": "Design a separately verified shared-resource background-flow process from observed excluded coupon patterns, with explicit booking-time/date allocation assumptions. Preserve the unidentified residual separately; do not inflate represented O&D demand to absorb it.",
        "limitations": [
            "Coupon-edge counts and allocated physical segment passengers have different observational units and coverage.",
            "The source does not identify dated-flight assignment, booking-time profiles, capacity entitlement, or retained revenue for excluded paths.",
            "Unknown ticket labels remain unattributed; operating labels do not substitute for ticketing networks.",
            "Domestic-to-international O&D coverage, sampling and marketing allocation may differ. Their contributions to the remaining residual are not estimated.",
            "This is neither a calibrated background process nor a V1/V5 pass. No counts were rescaled."
        ]
    }
    atomic_json(root / "coverage_analysis.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("previous", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.previous), indent=2))
