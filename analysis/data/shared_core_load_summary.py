"""Audit the matched one-cohort load sensitivities, including regional operators."""

import argparse
import json
from pathlib import Path

import pandas as pd

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.spec import load_spec


def summarize(root, dated_legs):
    """Retain every sealed result; do not select a favorable load assumption."""
    rows, receipts, identities, event_identities = [], [], [], []
    campaign = json.loads((root / "config.json").read_text())
    inputs = [(p, False) for p in sorted(root.glob("cells/*/attempt-*/COMPLETE.json"))]
    reuse = campaign.get("baseline_reuse")
    if reuse:
        inputs.append((Path(reuse["directory"]) / "COMPLETE.json", True))
    for receipt, reused in inputs:
        directory = receipt.parent
        seal = json.loads(receipt.read_text())
        if file_hash(directory / "results.json") != seal["results_sha256"]:
            raise ValueError("load-study result hash mismatch")
        result = json.loads((directory / "results.json").read_text())
        receipts.append({"directory": str(directory), "status": result["status"],
                         "results_sha256": seal["results_sha256"], "reused": reused})
        if result["status"] != "completed":
            continue
        if reused and (seal["results_sha256"] != reuse["results_sha256"]
                or result["fraction"] != 0
                or result["shared_core_load"].get("accepted_groups", 0) != 0):
            raise ValueError("reused baseline must be the declared disabled-load case")
        config = json.loads((directory / "config.json").read_text())["parameters"]
        spec = load_spec(config["source_spec"])
        resources = pd.DataFrame(json.loads((directory / "resource_usage.json").read_text()))
        physical = pd.read_parquet(dated_legs, columns=["id", "partner_operated", "operating_carrier"],
            filters=[("cohort_date", "==", spec.horizon.departure_date)])
        resources = resources.merge(physical, left_on="leg", right_on="id", how="left",
                                    validate="one_to_one")
        if resources.partner_operated.isna().any():
            raise ValueError("resource ID lost its regional/physical provenance")
        if not ((resources.capacity - resources.strategic_usage - resources.external_usage
                 == resources.inventory) & resources.inventory.ge(0)).all():
            raise ValueError("load-study resource identity failed")
        capacities = tuple(resources.sort_values("leg").capacity)
        identities.append((result["spec_hash"], result["potential_strategic_requests"],
                           result["seed"], capacities))
        if not reused:
            event_identities.append(result["mapping"]["event_tape_sha256"])
        for code, economic in result["core_carriers"].items():
            frame = resources[resources.carrier == code]
            regional = frame[frame.partner_operated.astype(bool)]
            rows.append({"fraction": result["fraction"], "carrier": code, "reused": reused,
                "physical_capacity": int(frame.capacity.sum()),
                "strategic_coupon_boardings": int(frame.strategic_usage.sum()),
                "external_component_boardings": int(frame.external_usage.sum()),
                "inventory": int(frame.inventory.sum()),
                "partner_operated_legs": len(regional),
                "partner_strategic_boardings": int(regional.strategic_usage.sum()),
                "partner_external_boardings": int(regional.external_usage.sum()),
                "competitive_market_journeys": economic["journeys"],
                "strategic_gross_contribution": economic["profit"],
                "strategic_revenue": economic["revenue"],
                "modeled_strategic_service_cost": economic["service_cost"],
                "cohort_seconds": result["cohort_seconds"],
                "worker_wall_seconds": result["wall_seconds"],
                "worker_cpu_seconds": result["cpu_seconds"],
                "worker_peak_rss_bytes": result["peak_rss_bytes"],
                "nonpurchase_requests": result["nonpurchase_requests"],
                "outside_ticket_bookings": result["outside_network_bookings"]})
    matched = bool(identities) and all(x == identities[0] for x in identities)
    if identities and not matched:
        raise ValueError("sensitivity cases do not share physical tapes")
    event_match = bool(event_identities) and all(x == event_identities[0] for x in event_identities)
    if event_identities and not event_match:
        raise ValueError("new sensitivity cases do not share event tapes")
    configured = len(json.loads((root / "cells.json").read_text()))
    newly_sealed = sum(not x["reused"] for x in receipts)
    successful_new = sum(not x["reused"] and x["status"] == "completed" for x in receipts)
    report = {"status": ("complete" if successful_new == configured else "complete_with_failures")
                  if newly_sealed == configured else "partial",
              "configured_cells": configured, "sealed_cells": newly_sealed,
              "completed_cells_including_reused": len(identities), "completed_new_cells": successful_new,
              "matched_physical_spec_requests_seed_capacity": matched,
              "new_cases_share_event_tape": event_match,
              "reused_zero_baseline": reuse,
              "population_scope": "Disjoint external O&Ds only; resource-bearing in-pool OUTSIDE choices still deferred"
                  if campaign.get("partition") else "All-excluded resource stress; in-pool overlap prevents interpreting it as a population-closed calibration model",
              "source_dated_legs_sha256": file_hash(dated_legs), "receipts": receipts,
              "sample_size_per_fraction": 1, "fit_run": False,
              "timing_scope": "cohort_seconds includes reset/simulation; worker wall/CPU also includes setup, tape generation, metrics and study artifact work. These are single-case observations under contention, not a calibration budget.",
              "interpretation": "Paired resource-footprint sensitivity on one uncalibrated cohort/tape. No standard error, calibration, parameter selection, monotonic economic criterion, total airline/net-profit claim or CEM throughput budget. Date/booking-time assignment remains assumed; unmatched mass and the unidentified residual are not imputed."}
    pd.DataFrame(rows).to_csv(root / "load_sensitivity.csv", index=False)
    atomic_json(root / "load_analysis.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("dated_legs", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root, args.dated_legs), indent=2))
