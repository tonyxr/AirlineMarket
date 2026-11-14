"""Native three-stage collection, heldout isolation and exact split continuation."""

import importlib.util
import json
import os
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from airline_market.campaign.artifacts import file_hash
from airline_market.core.errors import ZeroLikelihoodError
from airline_market.experiments.type_game_study import micro_spec
from airline_market.game.type_belief import HistoryPolicy
from airline_market.spec import compile_market

if os.environ.get("TR_STAGE"):
    root = Path(os.environ["TR_STAGE"])
    for name in [
        "airline_market.learn.curriculum_ppo",
        "airline_market.experiments.curriculum_study",
    ]:
        spec = importlib.util.spec_from_file_location(
            name, root / (name.split(".")[-1] + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
from airline_market.experiments import curriculum_study as study
from airline_market.learn import curriculum_ppo as learner


def domains():
    result = []
    for k, (T, symmetric) in enumerate([(2, False), (3, True), (3, False)]):
        c = compile_market(micro_spec({"T": T, "symmetric": symmetric}))
        result.append(
            {
                "id": str(k),
                "role": "heldout" if k == 2 else "train",
                "compiled": c,
                "populations": {
                    q.carrier: tuple(
                        HistoryPolicy(q.theta_grid, {}, default=i, arm="own")
                        for i in [0, 1]
                    )
                    for q in c.spec.pricing
                },
                "mixtures": {q.code: (0.4, 0.6) for q in c.spec.carriers},
            }
        )
    return result


def config():
    return {
        "n_train": 20,
        "cohorts_per_update": 2,
        "n_validation": 3,
        "validate_every_cohorts": 10,
        "network_seed": 0,
        "mpc": {"n_scenarios": 2, "n_reporting": 2, "tail_index": 0},
        "ppo": {"hidden_dim": 8, "update_epochs": 1, "minibatch_episodes": 2},
    }


def test_cohort_boundaries_independent_of_batch_and_domain_selection():
    assert [
        sum(learner.curriculum_stage(i, 20) == stage for i in range(20))
        for stage in range(3)
    ] == [8, 6, 6]
    for seed in range(100000, 100300):
        assert learner.choose_domain(domains(), seed, True)["id"] != "2"
    assert {
        learner.choose_domain(domains(), s, True)["id"] for s in range(100000, 100100)
    } == {"0", "1"}


def test_real_rival_paths_and_permitted_view_replay():
    d = domains()[0]
    seen = set()
    for seed in range(100000, 100060):
        _, rival, label = learner.make_adapter(d, "A", 0, seed, config()["mpc"])
        seen.add(label)
    assert seen == {"threshold", "matching", "DLP"}
    for stage in [0, 1, 2]:
        adapter, rival, _ = learner.make_adapter(d, "A", stage, 100000, config()["mpc"])
        x = adapter.reset(100000)
        obs = adapter.env.engine.observe(1, rival.arm_name)
        state = rival.snapshot()
        first = rival.act(obs)
        rival.restore(state)
        # Unused changes to the external focal hidden world cannot reach rival act.
        adapter.env.engine.world.carriers[0].cash += 999
        second = rival.act(obs)
        assert first == second
        assert len(x) == len(adapter.names())


def tensors_equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(tensors_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(tensors_equal(x, y) for x, y in zip(a, b))
    return a == b


def test_actual_updates_three_stages_and_exact_half_resume(tmp_path):
    d, cfg = domains(), config()
    full, split, resumed = [tmp_path / n for n in ["full", "split", "resumed"]]
    result = learner.train_curriculum(d, cfg, full, lambda _: None)
    partial = learner.train_curriculum(d, cfg, split, lambda _: None, stop_after=10)
    assert partial["cohorts"] == 10
    resumed_cfg = {
        **cfg,
        "resume_file": str(split / "last.pt"),
        "resume_sha256": file_hash(split / "last.pt"),
    }
    resumed_result = learner.train_curriculum(d, resumed_cfg, resumed, lambda _: None)
    assert (
        result["stage_counts"] == [8, 6, 6]
        and resumed_result["updates"] == result["updates"]
    )
    a, b = [torch.load(p / "last.pt", weights_only=False) for p in [full, resumed]]
    for field in [
        "actor",
        "value",
        "actor_optimizer",
        "value_optimizer",
        "numpy_rng",
        "torch_rng",
        "observation_normalizer",
    ]:
        assert tensors_equal(a[field], b[field]), field
    assert a["metadata"]["assignments"] == b["metadata"]["assignments"]
    metadata = a["metadata"]
    assert {r["member"] for r in metadata["assignments"]} == {"0", "1"}
    assert all(
        r["member"] != "2" for v in metadata["validation"] for r in v["assignments"]
    )
    assert all(r["MPC_decisions"] for r in metadata["assignments"] if r["stage"] == 2)
    assert all(
        r["opponent_type"] in (0, 1) for r in metadata["assignments"] if r["stage"] == 3
    )
    assert (
        not result["calibrated_TR2_completed"] and result["gradient_variance"] is None
    )


@pytest.mark.parametrize(
    "field,value", [("train_seed_start", 300000), ("validation_seed_start", 300000)]
)
def test_final_seed_refused_before_collection(tmp_path, field, value, monkeypatch):
    monkeypatch.setattr(
        learner, "collect", lambda *a, **k: pytest.fail("physical collection reached")
    )
    with pytest.raises(ValueError, match="split"):
        learner.train_curriculum(
            domains(), {**config(), field: value}, tmp_path, lambda _: None
        )


def test_domain_contract_change_rejected_on_resume(tmp_path):
    d, cfg = domains(), config()
    learner.train_curriculum(d, cfg, tmp_path / "first", lambda _: None, stop_after=10)
    altered = deepcopy(d)
    altered[0]["mixtures"]["A"] = (0.5, 0.5)
    resume = tmp_path / "first/last.pt"
    with pytest.raises(ValueError, match="training contract"):
        learner.train_curriculum(
            altered,
            {**cfg, "resume_file": str(resume), "resume_sha256": file_hash(resume)},
            tmp_path / "changed",
            lambda _: None,
        )


def test_native_continuous_focal_actions_do_not_silently_change_rival_model():
    d = domains()[1]
    adapter, rival, label = learner.make_adapter(d, "A", 1, 100004, config()["mpc"])
    adapter.reset(100004)
    for _ in range(d["compiled"].spec.horizon.n_periods):
        _, _, done, _ = adapter.step(np.array([3.1415]))
    assert done and len(rival.decisions) == 3
    assert "reference_population_model" in label


def test_adaptive_model_impossible_observation_is_not_recovered(tmp_path, monkeypatch):
    def impossible(self, observation):
        raise ZeroLikelihoodError("registered reference model impossible")

    monkeypatch.setattr(learner.EpisodeTypeMPCController, "act", impossible)
    monkeypatch.setattr(learner, "curriculum_stage", lambda *args: 1)
    with pytest.raises(ZeroLikelihoodError, match="reference model impossible"):
        learner.train_curriculum(domains(), config(), tmp_path, lambda _: None)


def test_saved_game_manifest_and_real_driver(tmp_path):
    root = Path(
        "results/F7/20261112_episode_type_micro_v1/cells/d8c19cce2fd15982/attempt-0001"
    )
    names = [
        "spec.json",
        "policies.json",
        "config.json",
        "results.json",
        "COMPLETE.json",
    ]
    member = {
        "id": "saved",
        "role": "train",
        "directory": str(root),
        "hashes": {n: file_hash(root / n) for n in names},
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"members": [member]}))
    cfg = config()
    cfg.update(n_train=10, validate_every_cohorts=10)
    p = {
        "manifest": str(manifest),
        "manifest_sha256": file_hash(manifest),
        "training": cfg,
    }
    result = study.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
    assert (
        result["stage_counts"] == [4, 3, 3] and not result["heldout_members_executed"]
    )
    bad = deepcopy(member)
    bad["hashes"]["spec.json"] = "0" * 64
    with pytest.raises(ValueError, match="changed frozen"):
        learner.load_domains([bad])
