"""Audit sealed exact episode-type game receipts, preserving limits and failures."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from airline_market.campaign.artifacts import atomic_json, file_hash, utc_now
from airline_market.campaign.runner import completed_attempt
from airline_market.spec import compile_market, load_spec


def summarize(root):
    """Report actual response costs and finite-class gains without rerunning games."""
    cells = json.loads((root / "cells.json").read_text())
    rows = []
    for cell in cells:
        row = {
            "cell_id": cell["cell_id"],
            "parameters": cell["parameters"],
            "status": "not_completed",
            "certified": False,
            "max_deviation_gain": None,
            "iterations": None,
            "payoff_entries": None,
            "wall_seconds": None,
            "peak_rss_bytes": None,
        }
        found = completed_attempt(root / "cells" / cell["cell_id"], cell["config_hash"])
        if found:
            directory, result = found
            row.update(
                status=result["status"],
                wall_seconds=result["wall_seconds"],
                peak_rss_bytes=result.get("peak_rss_bytes"),
                results_sha256=file_hash(directory / "results.json"),
            )
            if result.get("certified"):
                if file_hash(directory / "policies.json") != result["policies_sha256"]:
                    raise ValueError("sealed game policies changed")
                c = compile_market(load_spec(directory / "spec.json"))
                if c.spec_hash != result["spec_hash"]:
                    raise ValueError("sealed game physical specification changed")
                history = result["history"]
                tolerance = float(cell["parameters"].get("tolerance", 1e-6))
                last = history[-1]
                if max(last["deviation_gains"]) > tolerance:
                    raise ValueError("stopping deviation criterion failed")
                if any(
                    not p.controllable_features or len(p.theta_grid) < 2
                    for p in c.spec.pricing
                ):
                    raise ValueError("two-sided pricing class is degenerate")
                if (
                    cell["parameters"].get("symmetric")
                    and abs(last["payoffs"][0] - last["payoffs"][1]) > 1e-10
                ):
                    raise ValueError(
                        "registered symmetric game has unequal equilibrium payoffs"
                    )
                row.update(
                    certified=True,
                    max_deviation_gain=result["max_deviation_gain"],
                    iterations=len(history),
                    payoff_entries=result["payoff_entries"],
                    population_sizes=last["population_sizes"],
                    exact_payoffs=last["payoffs"],
                    response_seconds=sum(
                        sum(h["best_response_seconds"]) for h in history
                    ),
                    response_nodes=sum(sum(h["best_response_nodes"]) for h in history),
                    scope=result["scope"],
                    is_original_I1=result["is_original_I1"],
                )
        rows.append(row)
    summary = {
        "updated_at": utc_now(),
        "configured_cells": len(cells),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "certified_cells": sum(r["certified"] for r in rows),
        "rows": rows,
        "global_game_budget": None,
        "I3_MPC_completed": False,
        "scope": "Receipt-only audit of new two-sided finite-grid variants. Original I1/I3 are preserved. Limits and failures provide no deviation certificate; no continuous-price, empirical-game, or complete F7 claim.",
    }
    atomic_json(root / "game_summary.json", summary)
    with (root / "game_costs.csv").open("w") as handle:
        fields = [
            "cell_id",
            "status",
            "certified",
            "max_deviation_gain",
            "iterations",
            "payoff_entries",
            "wall_seconds",
            "response_seconds",
            "response_nodes",
            "peak_rss_bytes",
        ]
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return {k: v for k, v in summary.items() if k != "rows"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    print(json.dumps(summarize(parser.parse_args().root), indent=2))
