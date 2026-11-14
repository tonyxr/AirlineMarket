"""Fresh-only cohort guards, native checkpoint parity and paired arithmetic."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.campaign.instances import build_instance
from airline_market.experiments.learning_pilot import initial_agent
from airline_market.learn.ppo import collect, save_checkpoint
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.spec import compile_market, dump_spec

if os.environ.get("PAIR_STAGE"):
    spec = importlib.util.spec_from_file_location(
        "airline_market.experiments.checkpoint_pair_study",
        Path(os.environ["PAIR_STAGE"]) / "checkpoint_pair_study.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
from airline_market.experiments import checkpoint_pair_study as study


@pytest.mark.parametrize(
    "start,count", [(199999, 2), (299999, 2), (300000, 2), (203008, 1)]
)
def test_no_physical_rollout_for_invalid_split(start, count):
    with pytest.raises(ValueError, match="validation"):
        study.run({"parameters": {"seed_start": start, "n_cohorts": count}}, None)


def fixture(tmp_path):
    c = compile_market(build_instance("toy_hub", {"T": 2, "actions": 4}))
    adapter, agent = initial_agent(
        c, {"learner": "ppo", "network_seed": 0, "ppo": {"hidden_dim": 8}}
    )
    dump_spec(c.spec, tmp_path / "spec.json")
    pricing = next(x for x in c.spec.pricing if x.carrier == "A")
    baseline = {
        "spec_hash": c.spec_hash,
        "arm": "direct",
        "focal": "A",
        "fixed": list(pricing.theta_grid[-1]),
        "validation_seeds": [200000, 200001],
        "fixed_training_table": [
            {"coefficients": list(x), "training_objective": k}
            for k, x in enumerate(pricing.theta_grid)
        ],
    }
    atomic_json(tmp_path / "baseline.json", baseline)
    bh = file_hash(tmp_path / "baseline.json")
    save_checkpoint(
        agent,
        tmp_path / "last.pt",
        adapter,
        {"learner": "ppo", "consumed": 0, "validation": [{}]},
    )
    payload = torch.load(tmp_path / "last.pt", weights_only=False)
    payload["pilot_contract"] = {"baseline_sha256": bh}
    torch.save(payload, tmp_path / "last.pt")
    atomic_json(
        tmp_path / "manifest.json",
        {
            "checkpoints": [
                {
                    "path": str(tmp_path / "last.pt"),
                    "sha256": file_hash(tmp_path / "last.pt"),
                }
            ]
        },
    )
    parent = {
        "spec_hash": c.spec_hash,
        "learner": "ppo",
        "policy_class": "coefficient_recurrent",
        "cohorts": 0,
        "updates": 0,
        "baseline_sha256": bh,
        "parameters": {
            "focal": "A",
            "arm": "direct",
            "validation_seed_start": 200000,
            "n_validation": 2,
        },
    }
    atomic_json(tmp_path / "parent.json", parent)
    atomic_json(tmp_path / "base_results.json", {"status": "completed"})
    for name, result in [("parent", "parent.json"), ("base", "base_results.json")]:
        atomic_json(
            tmp_path / (name + "_receipt.json"),
            {"status": "completed", "results_sha256": file_hash(tmp_path / result)},
        )
    roles = {
        "source_spec": "spec.json",
        "checkpoint": "last.pt",
        "checkpoint_manifest": "manifest.json",
        "parent_results": "parent.json",
        "parent_receipt": "parent_receipt.json",
        "baseline_file": "baseline.json",
        "baseline_results": "base_results.json",
        "baseline_receipt": "base_receipt.json",
    }
    p = {k: str(tmp_path / v) for k, v in roles.items()}
    p.update(
        seed_start=203008,
        n_cohorts=3,
        compiled_spec_hash=c.spec_hash,
        focal="A",
        arm="direct",
    )
    p["input_hashes"] = {
        x: file_hash(Path(x))
        for x in list(p.values())
        if isinstance(x, str) and Path(x).is_file()
    }
    return p, adapter, agent, baseline


def test_native_deterministic_parity_tape_identity_and_no_learning(tmp_path):
    _, adapter, agent, baseline = fixture(tmp_path)
    before = study.inference_fingerprint(agent)
    rng = torch.get_rng_state().clone()
    row = study.one_pair(
        adapter,
        agent,
        FocalCarrierEnv(adapter.env.compiled, "A", "direct"),
        baseline["fixed"],
        203011,
    )
    expected = collect(adapter, agent, 203011, deterministic=True)[2]
    assert row["ppo_J"] == expected
    assert row["difference_J"] == row["ppo_J"] - row["fixed_J"]
    assert before == study.inference_fingerprint(agent)
    assert torch.equal(rng, torch.get_rng_state())
    assert len(row["policy_episode"]["rewards"]) == 2


def test_driver_uses_all_ordered_new_pairs_and_seals_arithmetic(tmp_path):
    p, _, _, _ = fixture(tmp_path)
    directory = tmp_path / "output"
    directory.mkdir()
    result = study.run({"parameters": p}, SimpleNamespace(directory=directory))
    rows = [json.loads(x) for x in (directory / "pairs.jsonl").read_text().splitlines()]
    assert [x["seed"] for x in rows] == [203008, 203009, 203010]
    assert result["cohorts"] == result["n"] == 3
    assert result["pairs_sha256"] == file_hash(directory / "pairs.jsonl")
    assert result["mean_difference_J"] == np.mean([x["difference_J"] for x in rows])
    assert result["full_V7_pass"] is None and not result["test_seeds_opened"]
    assert result["inference_state_unchanged"]


def test_fresh_source_bytes_and_last_checkpoint_required(tmp_path):
    p, _, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="overlaps"):
        study.load_sources(p, [200000, 200001])
    p["input_hashes"][p["checkpoint"]] = "changed"
    with pytest.raises(ValueError, match="changed"):
        study.load_sources(p, [203008, 203009])
    del p["input_hashes"][p["source_spec"]]
    with pytest.raises(ValueError, match="missing prospective"):
        study.check_inputs(p)


def test_negative_contrast_is_not_selected_away():
    rows = [{"ppo_J": 1.0, "fixed_J": 2.0, "difference_J": -1.0}] * 3
    s = study.paired_summary(rows)
    assert s["mean_difference_J"] == -1.0 and s["bootstrap95_J"] == [-1.0, -1.0]
    assert not s["checkpoint_meets_fixed_point_criterion"]
    assert s["losses"] == 3 and s["wins"] == 0
