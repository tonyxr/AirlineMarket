"""R3 forecast timing, proper scores, exact targets and native-update parity."""

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

from airline_market.belief import ExactBelief
from airline_market.belief.events import AllOf, LegInventoryAtMost
from airline_market.campaign.artifacts import file_hash
from airline_market.campaign.instances import build_instance
from airline_market.experiments.learning_pilot import initial_agent
from airline_market.learn.structured_train import collect_structured, load_structured
from airline_market.sim.kernel import ExactKernel
from airline_market.spec import compile_market, dump_spec

if os.environ.get("EVENT_STAGE"):
    name = "airline_market.experiments.event_calibration_study"
    spec = importlib.util.spec_from_file_location(
        name, Path(os.environ["EVENT_STAGE"]) / "event_calibration_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
from airline_market.experiments import event_calibration_study as study


def fixture():
    c = compile_market(build_instance("hub_feeder_small", {"T": 2, "actions": 4}))
    actor = {
        "learner": "srl_ca",
        "network_seed": 0,
        "focal": "A",
        "arm": "direct",
        "aux_weight": 0.3,
        "ppo": {"hidden_dim": 8, "update_epochs": 1, "minibatch_episodes": 2},
    }
    adapter, agent = initial_agent(c, actor)
    contract = study.event_contract(c, "A", ["B_HD0", "B_HD1"])
    return c, actor, adapter, agent, contract


def row(p, q, y=0, seed=202560):
    return {
        "forecast": p,
        "posterior_probability": q,
        "realized_label": y,
        "seed": seed,
    }


def test_tower_calibration_does_not_require_pointwise_posterior_equality():
    result = study.calibration([row(0.5, 0.0, 0), row(0.5, 1.0, 1)])
    assert result["posterior_ECE"] == 0
    assert result["expected_binary_brier"] == 0.25
    assert result["posterior_squared_error_diagnostic"] == 0.25
    result = study.calibration([row(0.5, 0.5)])
    assert result["expected_binary_brier"] == 0.25
    assert result["posterior_squared_error_diagnostic"] == 0


def test_fixed_bin_edges_empty_bins_and_raw_target_roundoff():
    result = study.calibration([row(0.0, 0.0), row(0.1, 0.2), row(1.0, 1.0 + 2e-16, 1)])
    assert [r["bin"] for r in result["bins"] if r["n"]] == [0, 1, 9]
    assert result["bins"][2]["forecast"] is None
    assert result["bins"][9]["posterior"] > 1
    with pytest.raises(ValueError):
        study.calibration([row(0.5, 1.1)])


def test_bootstrap_resamples_whole_correlated_cohorts():
    rows = [
        row(0.5, q, seed=seed)
        for seed, q in [(202560, 0.0), (202561, 1.0)]
        for _ in range(4)
    ]
    interval = study.cohort_bootstrap(rows, repetitions=512)
    assert interval["ECE_95_interval"] == [0.0, 0.5]
    assert study.cohort_bootstrap(rows[:4])["ECE_95_interval"] is None


def test_event_semantics_reject_own_and_ambiguous_legs():
    c, *_ = fixture()
    for legs in [["A_HD0", "B_HD1"], ["B_HD0", "B_HD0"], ["B_HD0"]]:
        with pytest.raises(ValueError):
            study.event_contract(c, "A", legs)


def test_forecast_hook_preserves_actual_action_recurrence_and_rng():
    _, _, adapter, agent, _ = fixture()
    x = adapter.reset(202560)
    original, probe = deepcopy(agent), deepcopy(agent)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        a, h, info = original.act(x)
        rng = torch.random.get_rng_state().clone()
        torch.manual_seed(42)
        b, k, other, p = study.act_with_forecast(probe, x)
        assert torch.equal(rng, torch.random.get_rng_state())
    np.testing.assert_array_equal(a, b)
    assert torch.equal(h.actor, k.actor) and torch.equal(h.value, k.value)
    np.testing.assert_array_equal(
        info["policy_coordinates"], other["policy_coordinates"]
    )
    assert 0 <= p <= 1 and not probe.actor.norm._forward_hooks


def test_pre_action_forecast_does_not_condition_on_current_sampled_action():
    _, _, adapter, agent, _ = fixture()
    x = adapter.reset(202561)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4)
        a, _, _, p = study.act_with_forecast(deepcopy(agent), x)
        torch.manual_seed(9)
        b, _, _, q = study.act_with_forecast(deepcopy(agent), x)
    assert p == q and not np.array_equal(a, b)


def test_default_exact_event_posteriors_equal_general_reference_on_native_actions():
    c, _, adapter, agent, contract = fixture()
    event = AllOf(tuple(LegInventoryAtMost("B", x, 1) for x in contract["legs"]))
    x = adapter.reset(202562)
    fast = ExactBelief(c, "A", "direct")
    general = ExactBelief(c, "A", "direct", ExactKernel(c))
    fast.reset(adapter.obs)
    general.reset(adapter.obs)
    assert type(fast.kernel).__name__ == "FastExactKernel"
    for _ in range(2):
        physical, _, _, _ = study.act_with_forecast(agent, x)
        before = adapter.obs
        action = adapter.price_action(physical, before)
        x, *_ = adapter.step(physical)
        fast.update(before, action, adapter.obs)
        general.update(before, action, adapter.obs)
        q = sum(w for h, w in fast.mass.items() if event.evaluate(h, c, 0))
        reference = sum(w for h, w in general.mass.items() if event.evaluate(h, c, 0))
        assert abs(q - reference) <= 1e-12


def test_training_loop_matches_native_batch_and_checkpoint_event_guard(tmp_path):
    c, actor, adapter, agent, contract = fixture()
    cfg = {
        "actor": actor,
        "train_seed_start": 100000,
        "n_train": 2,
        "batch": 2,
        "credit": {"fraction": 1.0, "K": 4, "H": 5, "mix": 0.5},
    }
    trained, other_adapter, _ = study.train_native(
        c, cfg, contract, tmp_path, lambda row: None
    )
    for seed in [100000, 100001]:
        episode, logp, _ = collect_structured(
            adapter,
            agent,
            seed,
            training=True,
            credit=cfg["credit"],
            event_legs=tuple(contract["legs"]),
        )
        agent.add_episode(
            episode, actions_are_physical=False, behavior_log_probabilities=logp
        )
    agent.update()
    assert trained.updates == agent.updates == 1
    for name, weight in agent.actor.state_dict().items():
        assert torch.equal(weight, trained.actor.state_dict()[name])
    restored, _ = study.load_event_agent(tmp_path / "last.pt", other_adapter, contract)
    assert restored.updates == 1
    changed = deepcopy(contract)
    changed["legs"] = ["B_OH0", "B_HD1"]
    with pytest.raises(ValueError, match="semantic"):
        study.load_event_agent(tmp_path / "last.pt", other_adapter, changed)


@pytest.mark.parametrize(
    "start,count,training", [(300000, 1, False), (199999, 1, False), (199999, 2, True)]
)
def test_seed_ranges_reject_before_rollout(start, count, training):
    with pytest.raises(ValueError):
        study.validate_interval(start, count, training)


def test_actual_small_driver_saves_ordered_forecasts_and_no_selection(tmp_path):
    c, actor, _, _, _ = fixture()
    source = tmp_path / "source.json"
    dump_spec(c.spec, source)
    directory = tmp_path / "study"
    directory.mkdir()
    p = {
        "source_spec": str(source),
        "source_spec_sha256": file_hash(source),
        "compiled_spec_hash": c.spec_hash,
        "actor": actor,
        "event_legs": ["B_HD0", "B_HD1"],
        "train_seed_start": 100000,
        "n_train": 4,
        "batch": 2,
        "validation_seed_start": 202560,
        "n_validation": 2,
        "max_support": 20000,
        "credit": {"fraction": 0.1, "K": 4, "H": 5, "mix": 0.5},
    }
    result = study.run({"parameters": p}, SimpleNamespace(directory=directory))
    assert result["training_cohorts"] == 4 and result["updates"] == 2
    for label in ["initial", "final"]:
        rows = [
            json.loads(x)
            for x in (directory / f"predictions_{label}.jsonl").read_text().splitlines()
        ]
        assert [(r["seed"], r["period"]) for r in rows] == [
            (202560, 0),
            (202560, 1),
            (202561, 0),
            (202561, 1),
        ]
        assert all(r["target_period"] == r["period"] + 1 for r in rows)
        assert rows[1]["terminal"] and rows[3]["terminal"]
    adapter = initial_agent(c, actor)[0]
    _, metadata = load_structured(directory / "last.pt", adapter)
    assert metadata["validation_used_for_selection"] is False
    assert result["full_V7_pass"] is None and result["scientific_RL_budget"] is None
