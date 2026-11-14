"""Write prospective feasibility designs without opening any final test seed."""

from itertools import product
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json


def write(path: str, data: dict) -> None:
    """Save the declared design as ordinary, reviewable JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2) + "\n")


def main() -> None:
    """Prepare the complete F1 requested grid and the benchmark specification."""
    arms = ["direct", "linked", "unrelated", "oracle", "full_state"]
    cases = []
    # First obtain short frontier anchors; start the full toy next, not last.
    priorities = [(3, 2), (5, 4), (10, 18)]
    pairs = priorities + [
        pair
        for pair in product([3, 4, 5, 6, 8, 10], [2, 4, 6, 9, 18])
        if pair not in priorities
    ]
    for T, actions in pairs:
        for arm in arms:
            cases.append(
                {
                    "instance": "I0",
                    "T": T,
                    "actions": actions,
                    "arm": arm,
                    "backend": "compact",
                    "reservation_gb": 12 if T >= 8 and actions >= 6 else 2,
                }
            )
    for T, arm in product([4, 6, 8], arms):
        cases.append(
            {
                "instance": "I3",
                "instance_generator": "hub_feeder_small",
                "T": T,
                "actions": 4,
                "arm": arm,
                "backend": "compact",
                "reservation_gb": 12 if T >= 6 else 3,
            }
        )
    for share, prior, lag, T, actions, arm in product(
        [0.1, 0.7], ["narrow", "trimodal"], [0, 1], [4, 5, 6, 8], [4, 6, 9], arms
    ):
        cases.append(
            {
                "instance": "I4",
                "instance_generator": "hub_feeder_family",
                "n_spokes": 2,
                "connecting_share": share,
                "rival_capacity_prior": prior,
                "lag_periods": lag,
                "T": T,
                "actions": actions,
                "arm": arm,
                "backend": "compact",
                "reservation_gb": 12 if T >= 6 else 3,
            }
        )
    anchors = [
        row
        for row in cases
        if row["instance"] == "I0" and (row["T"], row["actions"]) in [(3, 2), (5, 4)]
    ]
    first_long = [
        row
        for row in cases
        if row["instance"] == "I0"
        and (row["T"], row["actions"], row["arm"]) == (10, 18, "direct")
    ]
    family_anchors = [
        row
        for row in cases
        if row["instance"] in ("I3", "I4") and row["T"] == 4 and row["actions"] == 4
    ]
    prefix = anchors + first_long + family_anchors
    cases = prefix + [row for row in cases if row not in prefix]
    frontier = {
        "schema": "airline-campaign/1",
        "id": "F1",
        "phase": 1,
        "implementation_revision": "validation-campaign-v1",
        "task": "exact_dp",
        "generator": "toy_hub",
        "design": "list",
        "cells": cases,
        "workers": 9,
        "memory_budget_gb": 12,
        "limits": {"seconds": 14400, "rss_gb": 12, "max_nodes": 20000000},
        "notes": [
            "Nested 18-action subsets are defined in campaign.instances.grid_indices.",
            "Compact-incompatible cells are explicitly unsupported; their general-kernel runs must be scheduled separately and cannot be counted as solved.",
            "Full-state and oracle expose the same current physical truth in v1; neither reveals future random draws.",
            "12 GiB large-cell reservation serializes memory-heavy work on the 16 GiB Mac; CPU-light cells share the remaining declared reservations.",
        ],
    }
    digest = sha256(json.dumps(frontier, sort_keys=True).encode()).hexdigest()[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    frontier["out"] = f"results/F1/{stamp}_{digest}"
    write("config/campaigns/F1.json", frontier)
    write(
        "config/benchmarks/R0.json",
        {
            "generator": "toy_hub",
            "warmup": 1,
            "repeats": 5,
            "batch_size": 16,
            "n_scenarios": 256,
        },
    )
    write(
        "config/campaigns/hypotheses.json",
        {
            "schema": "airline-hypotheses/1",
            "source_plan_sha256": "ba19d3af61c6544df71d5f8404a42b417c360ed75520556beaff9d4ac27135be",
            "status": "prospectively_registered_before_test_release",
            "planning_master": 773,
            "hypotheses": [
                {
                    "id": "H-A1",
                    "primary": "exact linked-minus-direct value / direct revenue",
                    "region": "connecting_share>=0.3; bottleneck=rival_feeder; prior in {wide,trimodal}; private rival capacity",
                    "threshold": 0.0025,
                    "negative_control": "resource-independent exact difference <=1e-12",
                },
                {
                    "id": "H-A2",
                    "primary": "factorization loss vs posterior mutual information",
                    "threshold": "mean loss>=0.5% of value where MI>=0.1 nats; positive clustered slope; LCE>=Lfact in >=80% of cells",
                },
                {
                    "id": "H-A3",
                    "primary": "relative linked value at hub / matched exact scale",
                    "threshold": 0.5,
                },
                {
                    "id": "H-A4",
                    "primary": "linked value under common demand and rival misspecification",
                    "threshold": "report sign and magnitude; no invented numerical rejection threshold",
                },
                {
                    "id": "H-A5",
                    "primary": "strategic minus naive exact best response",
                    "threshold": "positive on at least one designed family",
                },
                {
                    "id": "H-B1",
                    "primary": "mechanical share of weather-induced fare spike",
                    "threshold": 0.5,
                },
                {
                    "id": "H-B2",
                    "primary": "triggered 1.5x cap: consumer-surplus gain / profit loss",
                    "threshold": ">=2; stranded served within 48h not lower; 1.0x cap misallocation higher than 1.5x",
                },
                {
                    "id": "H-D1",
                    "primary": "SRL-CA environment steps to 95% of MPC / PPO steps, >=8 spokes",
                    "threshold": "<=0.25",
                },
                {
                    "id": "H-D2",
                    "primary": "network-credit policy-gradient variance reduction",
                    "threshold": ">=0.30",
                },
            ],
            "analysis": {
                "primary_fwer": "Holm alpha=0.05 within paper",
                "secondary_fdr": "Benjamini-Hochberg q=0.10 exploratory",
                "equivalence_revenue_margin": 0.001,
                "bootstrap_replicates": 2000,
            },
            "note": "The source mentions H-A6 in a pivot sentence but does not define it in its hypothesis table; monitoring design A6 remains exploratory until explicitly specified. Null/negative findings are retained.",
        },
    )


if __name__ == "__main__":
    main()
