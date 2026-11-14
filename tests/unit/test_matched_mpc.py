"""Planning-prefix identity and real study-worker/exact-Q integration checks."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from airline_market.belief import ExactBelief
from airline_market.campaign.artifacts import file_hash
from airline_market.campaign.instances import build_instance
from airline_market.campaign.runner import THREAD_ENV
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.solvers.matched_mpc import MatchedScenarioMPC
from airline_market.solvers.mpc import MPCConfig
from airline_market.spec import compile_market


def initial():
    compiled = compile_market(build_instance("toy_hub", {"T": 3, "actions": 2}))
    env = FocalCarrierEnv(compiled, "A", "direct")
    obs, _ = env.reset(200000)
    belief = ExactBelief(compiled, "A", "direct")
    belief.reset(obs)
    planner = MatchedScenarioMPC(
        compiled,
        "A",
        "direct",
        belief,
        config=MPCConfig(evaluation="sampling", n_scenarios=128, selection="racing"),
        tail_coefficients=compiled.spec.pricing[0].theta_grid[-1],
    )
    return obs, belief, planner


def assert_same(first, second):
    assert first.tape.seed == second.tape.seed
    assert np.array_equal(first.tape.gumbel, second.tape.gumbel)
    assert [x.capacity.tolist() for x in first.world.carriers] == [
        x.capacity.tolist() for x in second.world.carriers
    ]


def test_scenario_count_independent_prefix_and_reporting_separation():
    obs, belief, planner = initial()
    mass = belief.mass.copy()
    small = planner._scenarios(obs, 0, 4, 0)
    large = planner._scenarios(obs, 0, 8, 0)
    for first, second in zip(small, large):
        assert_same(first, second)
    reporting = planner._scenarios(obs, 0, 4, 1)
    assert [x.tape.seed for x in small] != [x.tape.seed for x in reporting]
    assert belief.mass == mass


def test_racing_rounds_use_the_same_full_selection_prefix():
    obs, _, planner = initial()
    all_samples = planner._scenarios(obs, 1, 128, 0)
    first = planner._scenarios(obs, 1, 64, 2)
    second = planner._scenarios(obs, 1, 64, 3)
    for a, b in zip(all_samples, first + second):
        assert_same(a, b)
    assert [x["offset"] for x in planner.draw_requests] == [0, 0, 64]


def test_isolated_f5_tail_and_exact_q_comparison_receipts(tmp_path):
    def launch(config):
        path = tmp_path / (config["id"] + ".json")
        path.write_text(json.dumps(config))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "airline_market.experiments.study_campaign",
                "--config",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=45,
            env={**os.environ, **THREAD_ENV},
        )
        result_path = next(Path(config["out"]).glob("cells/*/attempt-*/results.json"))
        result = json.loads(result_path.read_text())
        assert result["status"] == "completed", result
        return result_path, result

    base = {
        "schema": "airline-campaign/1",
        "phase": 1,
        "implementation_revision": "f5-indexed-test-v1",
        "generator": "toy_hub",
        "design": "list",
        "workers": 1,
        "memory_budget_gb": 1,
        "limits": {"seconds": 40, "rss_gb": 1},
    }
    fitted_path, _ = launch(
        {
            **base,
            "id": "tails",
            "task": "study:airline_market.experiments.f5_study:fit_tails",
            "out": str(tmp_path / "tails"),
            "cells": [
                {
                    "T": 3,
                    "actions": 2,
                    "train_seed_start": 100000,
                    "n_train": 2,
                    "cem": {"population": 2, "iterations": 1, "seed": 100773},
                }
            ],
        }
    )
    tails = fitted_path.with_name("tails.json")
    compare = {
        **base,
        "id": "compare",
        "task": "study:airline_market.experiments.f5_study:compare",
        "out": str(tmp_path / "compare"),
        "cells": [
            {
                "T": 3,
                "actions": 2,
                "seed_start": 200000,
                "n_cohorts": 2,
                "tails_file": str(tails),
                "tails_sha256": file_hash(tails),
                "tail": "fixed",
                "scenarios": 8,
                "selection": "all",
                "protect_incumbent": False,
                "k_se": 0,
            }
        ],
    }
    path, result = launch(compare)
    assert result["criterion_status"] == "pilot_only"
    assert result["paired_ci_method"].startswith("paired-cohort")
    assert result["paired_95_ci"] is not None
    decisions = json.loads(path.with_name("decisions.json").read_text())
    assert {r["mode"] for r in decisions} == {"on_policy", "shared_fixed_trajectory"}
    assert all(r["regret_currency"] >= 0 for r in decisions)
    assert len([r for r in decisions if r["mode"] == "shared_fixed_trajectory"]) == 6
    receipt = file_hash(path)
    launch(compare)
    assert file_hash(path) == receipt
    assert len(list(path.parent.parent.glob("attempt-*"))) == 1


def test_f5_analysis_preserves_failed_sealed_cells(tmp_path):
    from analysis.feasibility.planner import aggregate

    cell = {
        "cell_id": "failure",
        "config_hash": "declared",
        "parameters": {"instance": "I0"},
    }
    (tmp_path / "cells.json").write_text(json.dumps([cell]))
    directory = tmp_path / "cells/failure/attempt-0001"
    directory.mkdir(parents=True)
    result = directory / "results.json"
    result.write_text(
        json.dumps(
            {"config_hash": "declared", "status": "time_limit", "reason": "time_limit"}
        )
    )
    (directory / "COMPLETE.json").write_text(
        json.dumps(
            {
                "config_hash": "declared",
                "status": "time_limit",
                "results_sha256": file_hash(result),
            }
        )
    )
    summary = aggregate(tmp_path)
    assert summary["sealed_cells"] == 1
    assert summary["status"] == "complete_with_failures"
    assert summary["selection_status"] == "no_successful_comparison"
