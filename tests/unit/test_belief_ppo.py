"""Exact belief-feature semantics, information access, actual updates and continuation."""

import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.campaign.instances import build_instance
from airline_market.core.errors import ZeroLikelihoodError
from airline_market.learn.ppo import load_checkpoint
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.solvers.fixed_search import episode_value, search_fixed_grid
from airline_market.spec import compile_market, dump_spec, load_spec

if os.environ.get("BELIEF_PPO_STAGE"):
    stage = Path(os.environ["BELIEF_PPO_STAGE"])
    for name in [
        "airline_market.learn.belief_ppo",
        "airline_market.experiments.belief_ppo_study",
    ]:
        spec = importlib.util.spec_from_file_location(
            name, stage / (name.split(".")[-1] + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
from airline_market.experiments import belief_ppo_study as study
from airline_market.learn import belief_ppo as learner


def model():
    return compile_market(build_instance("hub_feeder_small", {"T": 3, "actions": 2}))


def config():
    return {
        "n_train": 4,
        "cohorts_per_update": 2,
        "n_validation": 2,
        "validation_seed_start": 200000,
        "train_seed_start": 100000,
        "network_seed": 0,
        "validate_every_cohorts": 2,
        "focal": "A",
        "arm": "direct",
        "ppo": {"hidden_dim": 8, "update_epochs": 1, "minibatch_episodes": 2},
        "joint_events": [
            {"id": "B_joint_le1", "rival": "B", "legs": ["B_HD0", "B_HD1"], "k": 1}
        ],
    }


@pytest.mark.parametrize("arm", ["direct", "linked"])
def test_every_named_posterior_input_equals_the_actual_exact_law(arm):
    c = model()
    cfg = config()
    cfg["arm"] = arm
    view = learner.adapter(c, cfg)
    x = view.reset(200000)
    for _ in range(3):
        expected = {}
        for (carrier, local), law in view.belief.leg_marginals().items():
            leg = c.spec.legs[c.legs_of[carrier][local]].id
            expected.update(
                {
                    f"belief/{leg}/mean": float(np.arange(len(law)) @ law),
                    f"belief/{leg}/p0": law[0],
                    f"belief/{leg}/p1": law[1] if len(law) > 1 else 0,
                }
            )
        expected["belief/event/B_joint_le1"] = view.belief.event_probability(
            view.events[0]
        )
        for name, value in expected.items():
            assert x[view.names().index(name)] == np.float32(value)
        x, _, _, _ = view.step(c.spec.pricing[0].theta_grid[0])


def test_event_meaning_changes_checkpoint_schema_even_if_names_and_dimensions_match(
    tmp_path,
):
    c = model()
    cfg = config()
    learner.train_belief(c, cfg, tmp_path, lambda _: None)
    altered = config()
    altered["joint_events"][0]["k"] = 0
    first, second = learner.adapter(c, cfg), learner.adapter(c, altered)
    assert (
        first.names() == second.names() and first.schema_hash() != second.schema_hash()
    )
    with pytest.raises(ValueError, match="schema"):
        load_checkpoint(tmp_path / "last.pt", second)


def test_hidden_world_and_critic_columns_do_not_change_the_actor():
    c = model()
    cfg = config()
    cfg["critic_privileged"] = True
    view, agent = learner.initial_agent(c, cfg)
    x = view.reset(200000)
    action = agent.act(x, deterministic=True)[0]
    view.env.engine.world.carriers[1].capacity[:] = 999
    changed = view.encode(view.obs, view.env._info())
    np.testing.assert_array_equal(x[: view.public_dim], changed[: view.public_dim])
    assert not np.array_equal(x[view.public_dim :], changed[view.public_dim :])
    np.testing.assert_array_equal(action, agent.act(changed, deterministic=True)[0])
    assert not bool(agent.actor.input_mask[view.public_dim : len(view.names())].any())


def test_actual_native_updates_and_exact_checkpoint_continuation(tmp_path):
    torch.set_num_threads(1)
    c = model()
    cfg = config()
    _, fresh = learner.initial_agent(c, cfg)
    initial = {k: v.clone() for k, v in fresh.actor.state_dict().items()}
    full = tmp_path / "full"
    short = tmp_path / "short"
    resumed = tmp_path / "resumed"
    learner.train_belief(c, cfg, full, lambda _: None)
    part = config()
    part["n_train"] = 2
    learner.train_belief(c, part, short, lambda _: None)
    continuation = config()
    continuation.update(
        resume_file=str(short / "last.pt"), resume_sha256=file_hash(short / "last.pt")
    )
    result = learner.train_belief(c, continuation, resumed, lambda _: None)
    a = torch.load(full / "last.pt", weights_only=False)
    b = torch.load(resumed / "last.pt", weights_only=False)
    assert (
        result["cohorts"] == 4
        and result["updates"] == 2
        and result["behavior_logp_checks"] > 0
    )
    assert any(not torch.equal(v, a["actor"][k]) for k, v in initial.items())
    for field in ["actor", "value"]:
        for key, value in a[field].items():
            torch.testing.assert_close(value, b[field][key], rtol=0, atol=0)
    assert a["numpy_rng"] == b["numpy_rng"]
    assert a["metadata"]["validation"] == b["metadata"]["validation"]
    assert (resumed / "best.pt").exists()
    changed = config()
    changed["ppo"]["gae_lambda"] = 0.9
    changed.update(
        resume_file=str(short / "last.pt"), resume_sha256=file_hash(short / "last.pt")
    )
    with pytest.raises(ValueError, match="training contract"):
        learner.train_belief(c, changed, tmp_path / "wrong", lambda _: None)


def test_impossible_observation_propagates_without_recovery(monkeypatch):
    c = model()
    view = learner.adapter(c, config())
    view.reset(200000)
    original = view.env.step

    def impossible(action):
        obs, reward, done, info = original(action)
        return replace(obs, own_cash=obs.own_cash + 1e9), reward, done, info

    monkeypatch.setattr(view.env, "step", impossible)
    with pytest.raises(ZeroLikelihoodError):
        view.step(c.spec.pricing[0].theta_grid[0])


def test_hub_poisson_does_not_silently_substitute_unverified_filter():
    c = compile_market(load_spec("config/instances/I5.json"))
    with pytest.raises(ValueError, match="precursor"):
        learner.adapter(c, config())


@pytest.mark.parametrize(
    "field,value", [("train_seed_start", 300000), ("validation_seed_start", 300000)]
)
def test_final_streams_refused_before_any_rollout(tmp_path, field, value):
    cfg = config()
    cfg[field] = value
    with pytest.raises(ValueError, match="split"):
        learner.train_belief(model(), cfg, tmp_path, lambda _: None)


def fixture_pilot(tmp_path):
    c = model()
    cfg = config()
    cfg["n_train"] = 2
    physical = tmp_path / "physical.json"
    dump_spec(c.spec, physical)
    fit = search_fixed_grid(
        lambda: FocalCarrierEnv(c, "A", "direct"),
        c.spec.pricing[0].theta_grid,
        [100000, 100001],
    )
    env = FocalCarrierEnv(c, "A", "direct")
    baseline = {
        "spec_hash": c.spec_hash,
        "focal": "A",
        "arm": "direct",
        "validation_seeds": [200000, 200001],
        "validation_values": [
            episode_value(env, s, lambda obs: np.array(fit.best_theta))
            for s in [200000, 200001]
        ],
    }
    path = tmp_path / "baseline.json"
    atomic_json(path, baseline)
    p = {
        **cfg,
        "instance": "fixture",
        "source_spec": str(physical),
        "source_spec_sha256": file_hash(physical),
        "baseline_file": str(path),
        "baseline_sha256": file_hash(path),
    }
    return p, SimpleNamespace(directory=tmp_path, update=lambda _: None)


def test_actual_pilot_and_checkpoint_feature_contract(tmp_path):
    p, ctx = fixture_pilot(tmp_path)
    result = study.pilot({"parameters": p}, ctx)
    assert result["cohorts"] == 2 and result["updates"] == 1
    assert (
        result["final_minus_fixed"]["n"] == 2 and not result["full_P2_tuning_completed"]
    )
    manifest = json.loads((tmp_path / "checkpoint_manifest.json").read_text())
    assert "belief/event/B_joint_le1" in manifest["names"]
    assert manifest["belief_contract"]["events"][0]["legs"] == ["B_HD0", "B_HD1"]
    assert all(
        item["sha256"] == file_hash(tmp_path / item["file"])
        for item in manifest["checkpoints"]
    )


@pytest.mark.parametrize("field", ["source_spec_sha256", "baseline_sha256"])
def test_changed_physical_or_fitted_input_refused(tmp_path, field):
    p, ctx = fixture_pilot(tmp_path)
    p[field] = "0" * 64
    with pytest.raises(ValueError, match="changed frozen"):
        study.pilot({"parameters": p}, ctx)
