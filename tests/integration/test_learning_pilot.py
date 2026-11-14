"""Baseline reuse, native initial policies and complete learning-pilot receipts."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from airline_market.campaign.artifacts import file_hash
from airline_market.experiments import learning_pilot as study


def context(path):
    """Provide an isolated task directory, as the owned campaign worker does."""
    path.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(directory=path)


def parameters():
    """Use only small offline training/validation cohorts and a complete tiny grid."""
    return {"instance": "fixture", "T": 3, "actions": 2, "n_fit": 2,
            "n_validation": 2, "n_train": 4, "cohorts_per_update": 2,
            "network_seed": 0, "event_legs": ["B_OH", "B_HD"],
            "credit": {"fraction": 0, "K": 2, "H": 1, "mix": 0.5},
            "ppo": {"hidden_dim": 8, "update_epochs": 1, "minibatch_episodes": 2}}


@pytest.mark.parametrize("learner", ["ppo", "srl_ca"])
def test_initializer_matches_native_training_parameters(tmp_path, monkeypatch, learner):
    from airline_market.experiments import f6_study
    from airline_market.learn import structured_train

    cfg = dict(parameters(), learner=learner)
    compiled = study.market(cfg)
    _, expected = study.initial_agent(compiled, cfg)
    module = f6_study if learner == "ppo" else structured_train
    name = "RecurrentPPOAgent" if learner == "ppo" else "StructuredPPOAgent"
    original = getattr(module, name)
    captured = {}

    def capture(*args, **kwargs):
        """Capture the native untrained state before any actual optimizer step."""
        agent = original(*args, **kwargs)
        captured.update({key: deepcopy(getattr(agent, key).state_dict())
                         for key in ("actor", "value")})
        return agent

    monkeypatch.setattr(module, name, capture)
    directory = context(tmp_path / learner).directory
    if learner == "ppo":
        f6_study.flat_train(compiled, cfg, directory, lambda row: None)
    else:
        structured_train.train_structured(compiled, cfg, directory)
    for key in ("actor", "value"):
        for name, value in getattr(expected, key).state_dict().items():
            assert torch.equal(value, captured[key][name])


def test_completed_training_fit_is_reused_without_repeating_search(tmp_path, monkeypatch):
    cfg = parameters()
    first = context(tmp_path / "fit")
    study.fit_fixed({"parameters": cfg}, first)
    source = first.directory / "baseline.json"
    calls = []
    original = study.episode_value

    def counted(env, seed, policy):
        """Count physical evaluations to distinguish reuse from a repeated fit."""
        calls.append(seed)
        return original(env, seed, policy)

    monkeypatch.setattr(study, "episode_value", counted)
    second = context(tmp_path / "reuse")
    result = study.fit_fixed({"parameters": dict(cfg, reuse_tails=str(source),
                                                 reuse_tails_sha256=file_hash(source))}, second)
    assert result["fit_reused"] is True
    assert calls == [200000, 200001]
    a, b = [json.loads((d.directory / "baseline.json").read_text()) for d in (first, second)]
    assert a["fixed_training_table"] == b["fixed_training_table"]
    assert a["validation_values"] == b["validation_values"]


def test_paired_contrasts_require_matching_cohorts_and_preserve_zero():
    result = study.paired_summary([1, 2, 3], [1, 2, 3])
    assert result["mean"] == 0
    assert result["bootstrap95"] == [0, 0]
    with pytest.raises(ValueError, match="matching cohort"):
        study.paired_summary([1, 2], [1])


@pytest.mark.parametrize("learner", ["ppo", "srl_ca"])
def test_actual_pilot_compares_initial_final_fixed_and_extends_checkpoint_metadata(tmp_path, learner):
    cfg = dict(parameters(), learner=learner)
    baseline = context(tmp_path / "baseline")
    study.fit_fixed({"parameters": cfg}, baseline)
    path = baseline.directory / "baseline.json"
    pilot_context = context(tmp_path / "pilot")
    result = study.pilot({"parameters": dict(cfg, baseline_file=str(path),
                                             baseline_sha256=file_hash(path))}, pilot_context)
    report = json.loads((pilot_context.directory / "pilot.json").read_text())
    assert result["cohorts"] == 4
    assert result["final_minus_fixed"]["n"] == 2
    assert report["curve"][-1]["cohorts"] == 4
    assert report["parameter_counts"]["actor"] > 0
    payload = torch.load(pilot_context.directory / "last.pt", weights_only=False)
    assert payload["pilot_contract"]["observation_names"]
    assert payload["pilot_contract"]["action_transform"]["names"]
    assert payload["pilot_contract"]["final_test_seeds_opened"] is False


def test_bad_baseline_hash_fails_before_training_without_recovery(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="baseline changed"):
        study.pilot({"parameters": dict(parameters(), learner="ppo", baseline_file=str(path),
                                         baseline_sha256="0" * 64)}, context(tmp_path / "run"))
    assert not (tmp_path / "run/learning.json").exists()
