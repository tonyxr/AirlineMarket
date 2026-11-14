"""Train-only coefficient selection and paired held-out information evaluation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

from .belief import Belief
from .model import AirlineEnv, COEFFICIENT_GRID, Config, Information, Model
from .policies import Decision, RolloutPlanner
from .ride_response import FixedPolicy, PairedEstimate


def statistics(values) -> dict:
    estimate = PairedEstimate.from_values(values)
    se = estimate.standard_error
    finite = math.isfinite(se)
    return {"n": len(estimate.values), "mean": estimate.mean,
            "standard_error": se if finite else None,
            "ci95_normal": [estimate.mean - 1.96 * se, estimate.mean + 1.96 * se] if finite else None}


def paired_statistics(left, right, metric: str) -> dict:
    if [r["seed"] for r in left] != [r["seed"] for r in right]:
        raise ValueError("paired evaluations require identical ordered cohort seeds")
    return statistics([a[metric] - b[metric] for a, b in zip(left, right)])


def planning_seed(master: int, episode_index: int, period: int) -> int:
    # Does not consume, derive from, or receive the true evaluation tape seed.
    return int(np.random.SeedSequence([master, episode_index, period, 0xA17]).generate_state(1)[0])


def rollout(model: Model, seed: int, information: Information, *, fixed=None,
            planner: RolloutPlanner | None = None, plan_master: int = 773,
            episode_index: int = 0, trace: bool = False) -> dict:
    env = AirlineEnv(model)
    env.reset(seed)
    observation = env.observe(information)
    belief = Belief(model)
    belief.condition(observation)
    fixed_policy = FixedPolicy(fixed) if fixed is not None else None
    decisions = []
    while observation.time < model.config.horizon:
        if fixed_policy is not None:
            decision = Decision(tuple(float(x) for x in fixed_policy(None, observation.time)), None)
        elif planner is not None:
            decision = planner.choose(observation, belief, planning_seed(plan_master, episode_index, observation.time))
        else:
            raise ValueError("a fixed policy or planner is required")
        if trace:
            decisions.append({"time": observation.time, "own_inventory": observation.own_inventory,
                              "quotes": [asdict(q) for q in observation.quotes],
                              "coefficients": decision.coefficients,
                              "belief_entropy": belief.entropy(),
                              "belief_mean_rival_inventory": belief.mean_inventory(),
                              "belief_states": len(belief.mass),
                              "forecast_objective": decision.expected_objective})
        own_sale = env.step(decision.coefficients)
        after = env.observe(information)
        belief.advance(observation, decision.coefficients, own_sale, after)
        observation = after
    return {"seed": seed, **model.metrics(env.state), "decisions": decisions}


def fit_fixed(model: Model, seeds) -> tuple[tuple[float, ...], list[dict]]:
    results = []
    for coefficients in COEFFICIENT_GRID:
        values = [rollout(model, seed, Information.DIRECT, fixed=coefficients)["objective"] for seed in seeds]
        results.append({"coefficients": coefficients, "training_objective": float(np.mean(values))})
    best = max(results, key=lambda row: row["training_objective"])
    return tuple(best["coefficients"]), results


def run_experiment(config: Config, train_seeds, evaluation_seeds, *, scenarios: int,
                   plan_master: int = 773, trace: bool = False, progress_every: int = 8) -> dict:
    train_seeds, evaluation_seeds = list(train_seeds), list(evaluation_seeds)
    if not train_seeds or len(evaluation_seeds) < 2:
        raise ValueError("at least one training and two evaluation cohorts required")
    if len(set(train_seeds)) != len(train_seeds) or len(set(evaluation_seeds)) != len(evaluation_seeds):
        raise ValueError("cohort seeds must be unique within each split")
    if set(train_seeds) & set(evaluation_seeds):
        raise ValueError("training and evaluation cohort seeds must be disjoint")
    started = time.monotonic()
    model = Model(config)
    fixed, training = fit_fixed(model, train_seeds)
    print(f"Selected coefficients on {len(train_seeds)} training cohorts: {fixed}", flush=True)
    planner = RolloutPlanner(model, fixed, scenarios=scenarios)
    episodes = {name: [] for name in ("fixed", "direct", "linked", "unrelated", "oracle")}
    for idx, seed in enumerate(evaluation_seeds):
        episodes["fixed"].append(rollout(model, seed, Information.DIRECT, fixed=fixed, trace=trace))
        for information in Information:
            episodes[information.value].append(rollout(model, seed, information, planner=planner,
                                                      plan_master=plan_master, episode_index=idx, trace=trace))
        if progress_every and ((idx + 1) % progress_every == 0 or idx + 1 == len(evaluation_seeds)):
            print(f"Evaluated {idx + 1}/{len(evaluation_seeds)} paired cohorts ({time.monotonic() - started:.1f}s)", flush=True)
    metrics = ("profit", "share", "objective", "journeys", "refunds", "cancellations")
    summary = {method: {metric: statistics([row[metric] for row in rows]) for metric in metrics}
               for method, rows in episodes.items()}
    contrasts = {}
    for name, left, right in (("linked_minus_direct", "linked", "direct"),
                              ("unrelated_minus_direct", "unrelated", "direct"),
                              ("oracle_minus_direct", "oracle", "direct"),
                              ("direct_minus_fixed", "direct", "fixed")):
        contrasts[name] = {metric: paired_statistics(episodes[left], episodes[right], metric)
                           for metric in metrics}
    identical_placebo = all(a["profit"] == b["profit"] and a["share"] == b["share"]
                            for a, b in zip(episodes["direct"], episodes["unrelated"]))
    if not identical_placebo:
        raise AssertionError("independent quote changed the exact-model matched policy")
    return {"schema": "airline-information-experiment/1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config), "train_seeds": train_seeds, "evaluation_seeds": evaluation_seeds,
            "planning_seed": plan_master, "scenarios_per_decision": scenarios,
            "action_grid": COEFFICIENT_GRID, "fixed_coefficients": fixed,
            "training_grid": training, "summary": summary, "contrasts": contrasts, "episodes": episodes,
            "placebo_identical": identical_placebo, "elapsed_seconds": time.monotonic() - started,
            "planner_cache_hits": planner.cache_hits, "planner_cache_misses": planner.cache_misses,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__},
            "scope": "synthetic finite request experiment; approximate rollout, not equilibrium or optimal information value"}


def save_results(result: dict, output: Path, command: list[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    result["command"] = command
    root = Path(__file__).resolve().parent
    result["source_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (output / "episodes.csv").open("w", newline="") as handle:
        fields = ["method", "seed", "profit", "rival_profit", "share", "objective", "journeys",
                  "rival_segment_journeys", "refunds", "cancellations"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, rows in result["episodes"].items():
            for row in rows:
                writer.writerow({"method": method, **{key: row[key] for key in fields[1:]}})
    lines = ["# Two airline prototype experiment", "", "## Material Passport", "",
             "- Type: synthetic simulation and paired held-out comparison.",
             "- Status: completed; no empirical calibration or equilibrium claim.",
             "- Framework: academic-research-suite experiment execution; implementation authorized by the user.",
             f"- Evaluation cohorts: {len(result['evaluation_seeds'])}; scenarios/decision: {result['scenarios_per_decision']}.",
             "- Uncertainty: descriptive normal-approximation 95% intervals over independent paired cohorts.",
             "- Training selects one fixed coefficient policy on disjoint seeds; forecast randomness is separate.",
             "- AI assistance: implementation, tests, and descriptive reporting.", "",
             "| Method | Mean profit | Mean segment share | Mean objective |",
             "| --- | ---: | ---: | ---: |"]
    for method, values in result["summary"].items():
        lines.append(f"| {method} | {values['profit']['mean']:.3f} | {values['share']['mean']:.3f} | {values['objective']['mean']:.4f} |")
    lines += ["", "| Paired contrast | Profit difference | 95% interval |", "| --- | ---: | --- |"]
    for contrast, values in result["contrasts"].items():
        stats = values["profit"]
        interval = stats["ci95_normal"]
        lines.append(f"| {contrast} | {stats['mean']:.3f} | [{interval[0]:.3f}, {interval[1]:.3f}] |")
    lines += ["", "The oracle is a privileged current-state rollout reference, not an upper bound. "
              "The direct/linked/placebo arms share actions, price features, tail policy, and scenario budget. "
              "An independent reference quote cancels from this known-model Bayesian posterior, so the placebo "
              "is expected to match direct exactly. A finite-scenario rollout can lose from extra information; "
              "these contrasts do not estimate exact optimal information value.", "",
              "All monetary values are synthetic units. Positive intervals are exploratory within this declared "
              "toy model and do not establish novelty or real-airline gains.", ""]
    (output / "report.md").write_text("\n".join(lines))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--train-episodes", type=int, default=64)
    parser.add_argument("--scenarios", type=int, default=12)
    parser.add_argument("--train-seed", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=10000)
    parser.add_argument("--planning-seed", type=int, default=773)
    parser.add_argument("--capacity-shock-probability", type=float, default=0.0)
    parser.add_argument("--holiday-multiplier", type=float, default=1.0)
    parser.add_argument("--share-weight", type=float, default=0.0)
    parser.add_argument("--feeder-premium", type=float, default=35.0)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/prototype"))
    args = parser.parse_args(argv)
    config = Config(capacity_shock_probability=args.capacity_shock_probability,
                    holiday_multiplier=args.holiday_multiplier, share_weight=args.share_weight,
                    feeder_premium=args.feeder_premium)
    result = run_experiment(config, range(args.train_seed, args.train_seed + args.train_episodes),
                            range(args.eval_seed, args.eval_seed + args.episodes),
                            scenarios=args.scenarios, plan_master=args.planning_seed, trace=args.trace)
    save_results(result, args.output, sys.argv if argv is None else ["airline-experiment", *argv])
    print((args.output / "report.md").resolve(), flush=True)


if __name__ == "__main__":
    main()
