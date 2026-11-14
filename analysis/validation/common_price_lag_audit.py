"""Check the declared common-action-class lag contrasts on sealed exact controls."""

import argparse
import json
from pathlib import Path

from airline_market.campaign.artifacts import atomic_json, file_hash


def audit(root):
    """Report the finite control's criteria without declaring the complete V3 suite."""
    path = root / "frontier_coverage.json"
    source = json.loads(path.read_text())
    tolerance = 1e-12
    contrasts = source["descriptive_lag_contrasts"]
    lag_violations = [x for x in contrasts if x["lag0_minus_lag1"] < -tolerance]
    privileged = [x for x in contrasts if x["arm"] in ["oracle", "full_state"]]
    privileged_error = max((abs(x["lag0_minus_lag1"]) for x in privileged), default=None)
    unrelated_error = max((abs(x["unrelated_minus_direct"])
        for x in source["unrelated_direct_contrasts"]), default=None)
    complete = (source["all_eight_corners_five_arms_within_hour"] and len(contrasts) == 20
                and len(privileged) == 8 and not source["quote_dependent_pricing_features"])
    passed = (complete and not source["within_corner_information_order_violations"]
        and not lag_violations and privileged_error <= tolerance and unrelated_error <= tolerance)
    report = {"status": "passed" if passed else "incomplete_or_failed",
        "absolute_tolerance": tolerance, "source_sha256": file_hash(path),
        "all40_exact_controls_complete": complete,
        "maximum_solve_seconds": source["maximum_sealed_solve_seconds"],
        "lag_monotonicity_violations": lag_violations,
        "within_corner_information_order_violations": source["within_corner_information_order_violations"],
        "maximum_privileged_lag_difference": privileged_error,
        "maximum_unrelated_direct_difference": unrelated_error,
        "lag_contrasts": contrasts,
        "scope": "T4/A4 common-price-rule control only. Four physical/prior families, lag0/1, five arms. This neither replaces the original quote-dependent pricing results nor selects a largest exact scale or completes V3. Lag2, nested fare bins and remaining structural properties are separate."}
    atomic_json(root / "common_price_lag_audit.json", report)
    return {k: v for k, v in report.items() if k != "lag_contrasts"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    print(json.dumps(audit(parser.parse_args().root), indent=2))
