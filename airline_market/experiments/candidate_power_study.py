"""Read-only paired model-profit precision catalog from saved F5 cohorts.

The plan's section2.4 normal-approximation formula is conditional on a fixed
candidate, the256 pilot cohorts and the declared revenue denominator. This
catalog does not select a final planner, Phase4 sample size or compute budget.
"""

import csv
import json
import math
from pathlib import Path
from statistics import NormalDist

import numpy as np

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.spec import compile_market, load_spec


def power_projection(
    sd, delta, *, alpha=0.05, power=0.8, minimum=512, maximum=8192, multiple=256
):
    """Plan§2.4: ceil, round up, bound, and report target failure at the cap."""
    if (
        not all(math.isfinite(x) for x in [sd, delta, alpha, power])
        or sd < 0
        or delta <= 0
        or not 0 < alpha < 1
        or not 0.5 < power < 1
    ):
        raise ValueError("invalid finite power parameters")
    if (
        minimum < 1
        or maximum < minimum
        or multiple < 1
        or minimum % multiple
        or maximum % multiple
    ):
        raise ValueError("sample bounds must align with the declared block size")
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    raw = math.ceil((z * sd / delta) ** 2)
    rounded = max(minimum, math.ceil(raw / multiple) * multiple)
    capped = min(maximum, rounded)
    mde = z * sd / math.sqrt(capped)
    return {
        "raw_n": raw,
        "rounded_unbounded_n": rounded,
        "capped_candidate_n": capped,
        "cap_binds": rounded > maximum,
        "mde_at_candidate_n": mde,
        "target_met_at_candidate_n": mde <= delta * (1 + 1e-12),
        "pilot_variance_is_zero": sd == 0,
    }


def audit_entry(entry):
    """Check frozen bytes, original result receipt, exact pairing and dollar units."""
    files = entry["files"]
    for value in files.values():
        if file_hash(value["path"]) != value["sha256"]:
            raise ValueError("changed frozen candidate input")
    read = lambda name: json.loads(Path(files[name]["path"]).read_text())
    cfg, result, receipt, rows = (
        read(n)
        for n in ["config.json", "results.json", "COMPLETE.json", "episodes.json"]
    )
    p = cfg["parameters"]
    if (
        any(x["config_hash"] != entry["config_hash"] for x in [cfg, result, receipt])
        or receipt["results_sha256"] != files["results.json"]["sha256"]
        or receipt["status"] != result["status"]
    ):
        raise ValueError("candidate receipt/config mismatch")
    if (
        result["status"] != "completed"
        or result["n_cohorts"] != 256
        or p["n_cohorts"] != 256
        or len(rows) != 256
    ):
        raise ValueError("precision requires a complete256-cohort candidate")
    seeds = list(range(p["seed_start"], p["seed_start"] + 256))
    if (
        any(not 200000 <= x < 300000 for x in seeds)
        or [r["seed"] for r in rows] != seeds
    ):
        raise ValueError("missing/duplicate/unpaired/nonvalidation cohort")
    if (
        compile_market(load_spec(files["spec.json"]["path"])).spec_hash
        != result["spec_hash"]
        or files["tails.json"]["sha256"] != p["tails_sha256"]
    ):
        raise ValueError("saved physical or fitted-tail identity changed")
    data = np.asarray(
        [
            [
                r[k]
                for k in [
                    "profit",
                    "fixed_profit",
                    "revenue",
                    "fixed_revenue",
                    "direct_minus_fixed_profit",
                ]
            ]
            for r in rows
        ],
        dtype=float,
    )
    if not np.isfinite(data).all():
        raise ValueError("nonfinite model profit/revenue")
    difference = data[:, 0] - data[:, 1]
    if not np.allclose(difference, data[:, 4], rtol=0, atol=1e-9):
        raise ValueError("saved paired profit difference disagrees with accounting")
    sd = float(difference.std(ddof=1))
    checks = [
        (float(difference.mean()), result["mean_direct_minus_fixed_profit"]),
        (sd / 16, result["paired_standard_error"]),
        (float(data[:, 2].mean()), result["mean_cohort_revenue"]),
    ]
    if any(not math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-9) for x, y in checks):
        raise ValueError("saved cohort aggregates disagree with sealed results")
    revenue = float(data[:, 2].mean())
    if revenue <= 0:
        raise ValueError("positive direct revenue required for a relative threshold")
    baseline = [
        (seed, float(profit), float(rev))
        for seed, profit, rev in zip(seeds, data[:, 1], data[:, 3])
    ]
    return p, result, difference, revenue, baseline


def catalog(manifest, parameters, directory, progress=None):
    """Retain every snapshotted candidate and project only its same F5 operation."""
    directory = Path(directory)
    rows, baselines = [], {}
    for entry in manifest["entries"]:
        if progress:
            progress(
                {
                    "stage": "candidate_audit",
                    "completed": len(rows),
                    "total": len(manifest["entries"]),
                }
            )
        p, result, difference, revenue, baseline = audit_entry(entry)
        for seed, profit, rev in baseline:
            key = result["spec_hash"], p["tails_sha256"], seed
            if key in baselines and not np.allclose(
                baselines[key], [profit, rev], rtol=0, atol=1e-9
            ):
                raise ValueError(
                    "shared fitted baseline differs across matched candidates"
                )
            baselines[key] = profit, rev
        sd = float(difference.std(ddof=1))
        delta = parameters["revenue_fraction"] * revenue
        projection = power_projection(
            sd,
            delta,
            alpha=parameters["alpha"],
            power=parameters["power"],
            minimum=parameters["min_n"],
            maximum=parameters["max_n"],
            multiple=parameters["round_multiple"],
        )
        wall, cpu = result.get("wall_seconds"), result.get("cpu_seconds")
        for cost in [wall, cpu]:
            if cost is not None and (not math.isfinite(cost) or cost < 0):
                raise ValueError("invalid measured operation cost")
        n = projection["capped_candidate_n"]
        rows.append(
            {
                "cell_id": entry["cell_id"],
                **{k: p[k] for k in ["instance", "tail", "method", "scenarios"]},
                "pilot_cohorts": 256,
                "mean_profit_difference_usd": float(difference.mean()),
                "paired_profit_sd_usd": sd,
                "paired_profit_se_usd": sd / 16,
                "mean_direct_revenue_usd": revenue,
                "target_delta_usd": delta,
                "observed_profit_difference_percent_revenue": 100
                * float(difference.mean())
                / revenue,
                **projection,
                "mde_at_candidate_n_percent_revenue": 100
                * projection["mde_at_candidate_n"]
                / revenue,
                "F5_comparison_wall_seconds_per_cohort": None
                if wall is None
                else wall / 256,
                "F5_comparison_cpu_seconds_per_cohort": None
                if cpu is None
                else cpu / 256,
                "projected_same_F5_comparison_wall_hours": None
                if wall is None
                else wall * n / 256 / 3600,
                "projected_same_F5_comparison_cpu_hours": None
                if cpu is None
                else cpu * n / 256 / 3600,
                "phase4_primary_n_selected": None,
                "phase4_primary_core_hours": None,
                "source_results_sha256": entry["files"]["results.json"]["sha256"],
                "snapshotted_episode_sha256": entry["files"]["episodes.json"]["sha256"],
            }
        )
    if not rows:
        raise ValueError("empty candidate catalog")
    report = {
        "schema": "airline-F9-candidate-precision/1",
        "status": "completed_candidate_analysis_primary_budget_unselected",
        "parameters": parameters,
        "candidate_count": len(rows),
        "rows": rows,
        "pending_source_cells": len(manifest["not_sealed_at_snapshot"]),
        "shared_baseline_seed_contexts": len(baselines),
        "final_planner_selected": False,
        "phase4_primary_n_selected": None,
        "phase4_core_hours": None,
        "scope": "Saved256-cohort F5direct-MPC-minus-fitted-fixed model-profit precision for every registered completed candidate. Not all Phase4 primary contrasts, a chosen planner, budget, policy ranking, information-value power, D4primarysteps/gradientvariance or empirical calibration. Conditional normal-approximation plug-in SD; a zero sample SD is not proof of true zero variance/equivalence. Cap failures retained.",
        "provenance": "Original COMPLETE receipts seal results.json only. Episodes/spec/config/tails were byte-snapshotted before this analysis; accounting, exact seed pairing and sealed mean/SE/revenue aggregates corroborate them. No retroactive artifact-sealing claim.",
        "cost_scope": "Observed full F5comparison cost includes coldsetup/exactQ/sharedand onpolicy execution/durable artifacts. Same-factor constant-cost extrapolation only, not measured n-cohort cost, deployment-only evaluation or Phase4 core-hour budget. Contention/time drift not randomized.",
    }
    atomic_json(directory / "candidate_precision.json", report)
    with (directory / "candidate_precision.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return report


def study(cell, context):
    """Run the separately registered immutable analysis through the campaign runner."""
    p = cell["parameters"]
    if (
        file_hash(__file__) != p["implementation_sha256"]
        or file_hash(p["manifest"]) != p["manifest_sha256"]
    ):
        raise ValueError("changed registered implementation or source manifest")
    manifest = json.loads(Path(p["manifest"]).read_text())
    current = {"stage": "initializing"}
    context.counters = lambda: current.copy()
    report = catalog(manifest, p["power"], context.directory, current.update)
    return {
        "status": "completed",
        "candidates": report["candidate_count"],
        "source_cohorts_per_candidate": 256,
        "candidate_precision_sha256": file_hash(
            context.directory / "candidate_precision.json"
        ),
        "primary_n_selected": None,
        "phase4_core_hours": None,
        "scope": report["scope"],
        "simulation_cohorts_rerun": 0,
    }
