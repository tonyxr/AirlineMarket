"""Audit one paired outside-purchase resource probe without re-running its baseline."""

import argparse
import json
from pathlib import Path

import pandas as pd

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.spec import load_spec


def summarize(root, dated_legs):
    """Compare the same physical instance with and without the purchase attachment."""
    receipts = list(root.glob("cells/*/attempt-*/COMPLETE.json"))
    if len(receipts) != 1:
        raise ValueError("expected one sealed purchase-resource probe")
    directory = receipts[0].parent
    seal = json.loads(receipts[0].read_text())
    if file_hash(directory / "results.json") != seal["results_sha256"]:
        raise ValueError("purchase-resource result hash mismatch")
    result = json.loads((directory / "results.json").read_text())
    if result["status"] != "completed":
        report = {"status": result["status"], "source": str(directory),
                  "value": None, "reason": result.get("reason")}
        atomic_json(root / "coupling_analysis.json", report)
        return report
    reused = result["baseline_reused"]
    baseline_path = Path(reused["path"])
    if file_hash(baseline_path) != reused["sha256"]:
        raise ValueError("reused purchase-resource baseline changed")
    baseline = json.loads(baseline_path.read_text())
    if (baseline["spec_hash"] != result["spec_hash"] or baseline["seed"] != result["seed"]
            or baseline["potential_strategic_requests"] != result["potential_requests"]
            or baseline["mapping"]["event_tape_sha256"] != result["external_event_tape_sha256"]):
        raise ValueError("paired physical/event inputs differ")
    config = json.loads((directory / "config.json").read_text())["parameters"]
    spec = load_spec(config["source_spec"])
    physical = pd.read_parquet(dated_legs, columns=["id", "partner_operated"],
        filters=[("cohort_date", "==", spec.horizon.departure_date)])
    attached = pd.DataFrame(json.loads((directory / "resource_usage.json").read_text()))
    base = pd.DataFrame(json.loads((baseline_path.parent / "resource_usage.json").read_text()))
    if attached.sort_values("leg").capacity.tolist() != base.sort_values("leg").capacity.tolist():
        raise ValueError("paired physical capacities differ")
    attached = attached.merge(physical, left_on="leg", right_on="id", how="left", validate="one_to_one")
    if attached.partner_operated.isna().any():
        raise ValueError("lost regional-flight identity")
    identity = attached.capacity - attached.strategic_usage - attached.external_usage - attached.outside_purchase_usage
    if not (identity.eq(attached.inventory) & attached.inventory.ge(0)).all():
        raise ValueError("core resource decomposition is inconsistent")
    rows = []
    for code, economics in result["core_carriers"].items():
        a, b = attached[attached.carrier == code], base[base.carrier == code]
        partner = a[a.partner_operated.astype(bool)]
        rows.append({"carrier": code, "baseline_strategic_boardings": int(b.strategic_usage.sum()),
            "coupled_strategic_boardings": int(a.strategic_usage.sum()),
            "baseline_external_usage": int(b.external_usage.sum()),
            "coupled_external_usage": int(a.external_usage.sum()),
            "outside_purchase_core_usage": int(a.outside_purchase_usage.sum()),
            "partner_outside_purchase_usage": int(partner.outside_purchase_usage.sum()),
            "partner_external_usage": int(partner.external_usage.sum()),
            "strategic_contribution_difference": economics["profit"] - baseline["core_carriers"][code]["profit"]})
    pd.DataFrame(rows).to_csv(root / "purchase_resource_contrast.csv", index=False)
    report = {"status": "complete", "source": str(directory), "results_sha256": seal["results_sha256"],
        "baseline_reused": reused, "physical_inputs_matched": True,
        "one_choice_one_ticket": result["one_choice_one_ticket"],
        "core_resource_accounting_exact": True, "requests": result["realized_requests"],
        "active_logical_tickets": result["active_logical_tickets"],
        "outside_resource_coupling": result["outside_resource_coupling"],
        "catalogue_core_coupon_edges": result["conditional_core_coupon_edges"],
        "cohort_seconds": result["cohort_seconds"], "worker_wall_seconds": result["wall_seconds"],
        "worker_cpu_seconds": result["cpu_seconds"], "worker_peak_rss_bytes": result["peak_rss_bytes"],
        "sample_size": 1, "calibrated": False,
        "scope": "One paired uncalibrated resource-footprint observation, no inferential interval, full legal itinerary reconstruction, retained net revenue or total marketing-carrier profit claim. No independent in-pool arrival population was added. Remaining T100 residual is not imputed; exact/PF/MPC/learning and full V1/V5 remain unverified for this extension."}
    atomic_json(root / "coupling_analysis.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("dated_legs", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.dated_legs), indent=2))
