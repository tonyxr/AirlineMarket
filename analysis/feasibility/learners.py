"""Aggregate sealed F6 measurements without declaring learning or final budgets."""

from pathlib import Path
import argparse
import csv
import json

from airline_market.campaign.artifacts import atomic_json, file_hash, utc_now
from airline_market.campaign.runner import completed_attempt


def summarize(root):
    """Retain every failed/limited factor and separate measured cost from projections."""
    rows = []
    for cell in sorted((root / "cells").iterdir()):
        configs = sorted(cell.glob("attempt-*/config.json"))
        if not configs:
            continue
        cfg = json.loads(configs[-1].read_text())
        p = cfg["parameters"]
        finished = completed_attempt(cell, cfg["config_hash"])
        row = {"instance": p["instance"], "learner": p["learner"],
               "cohorts_per_update": p["cohorts_per_update"], "status": "not_completed",
               "seconds_per_cohort": None, "seconds_per_update": None,
               "training_seconds_per_cohort": None, "measured_updates": None,
               "peak_rss_bytes": None, "results_sha256": None}
        if finished:
            directory, result = finished
            row.update(status=result["status"], peak_rss_bytes=result.get("peak_rss_bytes"),
                       results_sha256=file_hash(directory / "results.json"))
            if result["status"] == "completed":
                rollout = result["median_seconds_per_cohort"]
                update = result["median_seconds_per_update"]
                row.update(seconds_per_cohort=rollout, seconds_per_update=update,
                           training_seconds_per_cohort=rollout + update / p["cohorts_per_update"],
                           measured_updates=result["measured_updates"])
        rows.append(row)
    with (root / "training_throughput.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows)
    result = {"updated_at": utc_now(), "sealed_cells": sum(x["status"] != "not_completed" for x in rows),
              "completed_measurements": sum(x["status"] == "completed" for x in rows),
              "configured_cells": 48, "grid_alias": {"8_spokes_2_carriers_T20": "I5"},
              "learning_signal_established": False, "final_training_budget": None,
              "remaining": ["20k-cohort I0/I5 pilots with fitted-fixed comparison",
                            "P2 belief-feature PPO path and tuning",
                            "fresh V7 structured/credit/auxiliary verification",
                            "TR curriculum/domain-randomization and final training"]}
    atomic_json(root / "training_summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
