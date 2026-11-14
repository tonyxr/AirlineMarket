"""Read-only all-five analysis of the registered single-leg PPO benchmark."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from airline_market.campaign.artifacts import atomic_json, file_hash, utc_now
from airline_market.core.rng import StreamRegistry


def iqm(values):
    """Fractional 25%-75% trimmed mean, including nonmultiples of four."""
    x = np.sort(np.asarray(values, dtype=float))
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("finite nonempty one-dimensional sample required")
    n = len(x)
    weights = np.maximum(
        0,
        np.minimum(np.arange(1, n + 1), 0.75 * n) - np.maximum(np.arange(n), 0.25 * n),
    )
    return float(np.dot(x, weights) / (0.5 * n))


def interval(values):
    return np.percentile(values, [2.5, 97.5]).tolist()


def bootstrap(ppo, dp, exact_ratios, vstar_J, count=2000):
    """Keep shared physical cohort indices identical in all NN strata and DP."""
    ppo, dp, exact_ratios = (
        np.asarray(x, dtype=float) for x in (ppo, dp, exact_ratios)
    )
    k, n = ppo.shape
    if (
        dp.shape != (n,)
        or exact_ratios.shape != (k,)
        or not all(np.isfinite(x).all() for x in (ppo, dp, exact_ratios))
    ):
        raise ValueError("paired finite matrix/vector dimensions required")
    if count < 2 or vstar_J <= 0:
        raise ValueError("positive target and at least two bootstrap draws required")
    registry = StreamRegistry(773, "planning/V7/single_leg_saved_analysis")
    exact_rng, cohort_rng, joint_rng = (registry.generator(i) for i in range(3))
    deltas = np.empty((count, k))
    exact_iqms, conditional_iqms, joint_iqms = [], [], []
    for b in range(count):
        exact_iqms.append(iqm(exact_ratios[exact_rng.integers(k, size=k)]))
        columns = cohort_rng.integers(n, size=n)
        means = ppo[:, columns].mean(axis=1)
        deltas[b] = means - dp[columns].mean()
        conditional_iqms.append(iqm(means / vstar_J))
        rows = joint_rng.integers(k, size=k)
        columns = joint_rng.integers(n, size=n)
        joint_iqms.append(iqm(ppo[rows][:, columns].mean(axis=1) / vstar_J))
    return {
        "replicates": count,
        "exact_ratio_IQM_NN_resampling_descriptive_95": interval(exact_iqms),
        "per_NN_delta_J_fixed_strata_common_cohort_95": np.percentile(
            deltas, [2.5, 97.5], axis=0
        ).T.tolist(),
        "sample_mean_ratio_IQM_fixed_NN_common_cohort_95": interval(conditional_iqms),
        "sample_mean_ratio_IQM_exploratory_joint_NN_common_cohort_95": interval(
            joint_iqms
        ),
        "exact_expectation_Monte_Carlo_interval": None,
        "scope": "Five NN strata share each resampled physical index vector. Exact-value interval resamples only five NN labels and is descriptive initialization variability. Sample-mean ratios use the analytic target; they can exceed one by sampling noise.",
    }


def criterion(ratios):
    x = np.asarray(ratios, dtype=float)
    if x.shape != (5,) or not np.isfinite(x).all():
        raise ValueError("all five finite final ratios required")
    return bool(np.all(x >= 0.98))


def analyze(root, output, registration):
    root, output, registration = map(Path, (root, output, registration))
    if output.exists():
        raise FileExistsError(
            "preserve the existing all-five analysis; do not rerun it"
        )
    reg = json.loads(registration.read_text())
    if reg["bootstrap_replicates"] != 2000:
        raise ValueError("registered bootstrap count changed")
    saved_hashes = {str(registration): file_hash(registration)}
    entries = []
    for receipt in root.glob("cells/*/attempt-*/COMPLETE.json"):
        directory = receipt.parent
        result_path = directory / "results.json"
        result = json.loads(result_path.read_text())
        seal = json.loads(receipt.read_text())
        if (
            result["status"] != "completed"
            or seal["status"] != "completed"
            or file_hash(result_path) != seal["results_sha256"]
        ):
            raise ValueError("all five completed sealed inputs required")
        for name, key in [
            ("exact_curve.json", "exact_curve_sha256"),
            ("validation_pairs.jsonl", "validation_pairs_sha256"),
        ]:
            path = directory / name
            if file_hash(path) != result[key]:
                raise ValueError("saved analysis input changed")
            saved_hashes[str(path)] = file_hash(path)
        saved_hashes[str(receipt)] = file_hash(receipt)
        saved_hashes[str(result_path)] = file_hash(result_path)
        entries.append((result["network_seed"], directory, result))
    entries.sort()
    if [x[0] for x in entries] != list(range(5)):
        raise ValueError("NN0..4 must all finish before aggregation")
    reference = Path(entries[0][2]["parameters"]["reference_directory"])
    baseline_path = reference / "DP_validation.json"
    baseline = json.loads(baseline_path.read_text())
    saved_hashes[str(baseline_path)] = file_hash(baseline_path)
    if len(baseline) != 1280 or [row["seed"] for row in baseline] != list(
        range(204288, 205568)
    ):
        raise ValueError("shared baseline seed order changed")
    dp = np.array([row["DP_J"] for row in baseline])
    ppo, curves, table = [], [], []
    vstar = entries[0][2]["V_star_USD"]
    for seed, directory, result in entries:
        if result["V_star_USD"] != vstar or result["parameters"][
            "reference_directory"
        ] != str(reference):
            raise ValueError("reference/target differs across NN cells")
        if (
            result["training_cohorts"],
            result["updates"],
            result["training_environment_steps"],
            result["validation_cohorts"],
        ) != (100000, 1563, 500000, 1280):
            raise ValueError("registered full budgets not complete")
        pairs = [
            json.loads(line)
            for line in (directory / "validation_pairs.jsonl").read_text().splitlines()
        ]
        if len(pairs) != 1280:
            raise ValueError("incomplete shared-bank validation")
        for row, base in zip(pairs, baseline):
            if (
                any(row[key] != base[key] for key in ("seed", "tape_sha256", "DP_J"))
                or abs(row["difference_J"] - row["PPO_J"] + row["DP_J"]) > 1e-12
            ):
                raise ValueError("physical pairing or difference arithmetic changed")
        curve = json.loads((directory / "exact_curve.json").read_text())
        if [x["training_cohorts"] for x in curve] != [
            0,
            20032,
            40000,
            60032,
            80000,
            100000,
        ]:
            raise ValueError("cadence budgets changed")
        if curve[0] != result["initial"] or curve[-1] != result["final"]:
            raise ValueError("final checkpoint differs from final curve")
        first = next(
            (x["training_cohorts"] for x in curve if x["ratio_to_V_star"] >= 0.98), None
        )
        if (
            first != result["first_observed_0_98_cohorts"]
            or result["initial_already_meets_0_98"] != curve[0]["meets_0_98"]
        ):
            raise ValueError("crossing/initial pass metadata differs")
        for row in curve:
            if abs(row["exact_USD"] / vstar - row["ratio_to_V_star"]) > 1e-12:
                raise ValueError("exact target units differ")
            curves.append({"network_seed": seed, **row})
        values = np.array([row["PPO_J"] for row in pairs])
        delta = values - dp
        ppo.append(values)
        table.append(
            {
                "network_seed": seed,
                "initial_exact_ratio": curve[0]["ratio_to_V_star"],
                "initial_already_pass": result["initial_already_meets_0_98"],
                "final_exact_ratio": curve[-1]["ratio_to_V_star"],
                "final_point_pass": curve[-1]["ratio_to_V_star"] >= 0.98,
                "first_observed_crossing_cohorts": first,
                "training_cohorts": 100000,
                "updates": 1563,
                "validation_cohorts": 1280,
                "PPO_sample_mean_J": float(values.mean()),
                "DP_sample_mean_J": float(dp.mean()),
                "paired_mean_delta_J": float(delta.mean()),
                "paired_SE_J": float(delta.std(ddof=1) / np.sqrt(len(delta))),
                "paired_wins": int(np.sum(delta > 1e-12)),
                "paired_ties": int(np.sum(np.abs(delta) <= 1e-12)),
                "paired_losses": int(np.sum(delta < -1e-12)),
                "wall_seconds": result["wall_seconds"],
                "cpu_seconds": result["cpu_seconds"],
                "peak_rss_bytes": result["peak_rss_bytes"],
            }
        )
    matrix = np.array(ppo)
    ratios = np.array([row["final_exact_ratio"] for row in table])
    uncertainty = bootstrap(matrix, dp, ratios, vstar / 100, 2000)
    for row, bounds in zip(
        table, uncertainty["per_NN_delta_J_fixed_strata_common_cohort_95"]
    ):
        row["paired_delta_J_percentile_low"], row["paired_delta_J_percentile_high"] = (
            bounds
        )
    thresholds = np.arange(160, 201) / 200
    profiles = [
        {
            "threshold": float(t),
            "fraction_of_five_exact_final_ratios_at_least_threshold": float(
                np.mean(ratios >= t)
            ),
        }
        for t in thresholds
    ]
    output.mkdir(parents=True)
    for name, rows in [
        ("all_five_NN.csv", table),
        ("all_learning_curves.csv", curves),
        ("exact_performance_profile.csv", profiles),
    ]:
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    atomic_json(output / "uncertainty.json", uncertainty)
    summary = {
        "analyzed_at": utc_now(),
        "all_five_seed_criterion": criterion(ratios),
        "final_exact_ratios_NN0_to_4": ratios.tolist(),
        "failed_NN_seeds": [
            x["network_seed"] for x in table if not x["final_point_pass"]
        ],
        "initial_already_pass_seeds": [
            x["network_seed"] for x in table if x["initial_already_pass"]
        ],
        "final_exact_ratio_IQM": iqm(ratios),
        "NN_rows": table,
        "training_profile_cohorts": 500000,
        "unique_training_physical_seed_IDs": 100000,
        "training_environment_steps": 2500000,
        "native_optimizer_updates": 7815,
        "PPO_validation_profile_rows": 6400,
        "unique_validation_physical_seed_IDs": 1280,
        "DP_baseline_generated_once_rows": 1280,
        "all_five_learning_worker_wall_seconds_sum": sum(
            x["wall_seconds"] for x in table
        ),
        "all_five_learning_worker_CPU_seconds_sum": sum(
            x["cpu_seconds"] for x in table
        ),
        "V_star_USD": vstar,
        "input_hashes": saved_hashes,
        "new_training_cohorts": 0,
        "new_actor_forwards": 0,
        "new_tapes_or_DP_outcomes": 0,
        "full_V7_pass": None,
        "scientific_RL_budget": None,
        "scope": "Registered all-five final-100k criterion on the declared native cent-fare monopoly class. Exact expectations have no sampling CI. Keep final failures and earlier crossings separate. Shared-bank sampled results are conditional and do not certify a different model, unrounded price class, learner ranking, hub/full P2/TR or full V7.",
    }
    for name, expected in saved_hashes.items():
        if file_hash(name) != expected:
            raise ValueError("input mutated during read-only analysis")
    summary["artifact_hashes"] = {
        name: file_hash(output / name)
        for name in [
            "all_five_NN.csv",
            "all_learning_curves.csv",
            "exact_performance_profile.csv",
            "uncertainty.json",
        ]
    }
    atomic_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.out, args.registration), indent=2))
