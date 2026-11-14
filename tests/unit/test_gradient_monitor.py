"""Native backward parity, conditional variance law and controlled monitor failures."""

import importlib.util
import json
import os
import struct
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from airline_market.campaign.instances import build_instance
from airline_market.core.rng import StreamRegistry
from airline_market.learn.adapters import RecurrentEnvAdapter
from airline_market.learn.ppo import collect
from airline_market.learn.structured_runtime import (
    StructuredAdapter,
    StructuredPPOAgent,
)
from airline_market.learn.structured_train import collect_structured
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.spec import compile_market
from airline_market.vendor.ride_response.recurrent.ppo import (
    RecurrentPPOAgent,
    RecurrentPPOConfig,
)

if os.environ.get("GRAD_STAGE"):
    root = Path(os.environ["GRAD_STAGE"])
    for name in [
        "airline_market.learn.gradient_monitor",
        "airline_market.experiments.monitored_learning",
    ]:
        spec = importlib.util.spec_from_file_location(
            name, root / (name.split(".")[-1] + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
from airline_market.experiments import monitored_learning as study
from airline_market.learn import gradient_monitor as measurement


def batch(kind, count=4):
    c = compile_market(build_instance("hub_feeder_small", {"T": 3, "actions": 2}))
    env = FocalCarrierEnv(c, "A", "direct")
    adapter = RecurrentEnvAdapter(env) if kind == "ppo" else StructuredAdapter(env)
    cfg = RecurrentPPOConfig(
        hidden_dim=8, update_epochs=1, minibatch_episodes=count, seed=7
    )
    with torch.random.fork_rng(devices=[]):
        agent = (
            RecurrentPPOAgent(len(adapter.names()), adapter.action_transform, cfg)
            if kind == "ppo"
            else StructuredPPOAgent(adapter, cfg, auxiliary=kind == "srl_ca")
        )
    if kind == "ppo":
        adapter.configure_actor(agent)
    for k in range(count):
        seed = 100000 + k
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(
                StreamRegistry(seed, "test/gradient/native").child_seed() % 2**31
            )
            episode, logp, _ = (
                collect(adapter, agent, seed)
                if kind == "ppo"
                else collect_structured(
                    adapter,
                    agent,
                    seed,
                    training=True,
                    credit={"fraction": 1.0, "K": 2, "H": 2, "mix": 0.5}
                    if kind in ("srl_c", "srl_ca")
                    else None,
                    event_legs=("B_HD0", "B_HD1") if kind == "srl_ca" else (),
                )
            )
        agent.add_episode(
            episode, actions_are_physical=False, behavior_log_probabilities=logp
        )
    return agent, adapter


@pytest.mark.parametrize("kind", ["ppo", "srl", "srl_c", "srl_ca"])
def test_unclipped_gradient_matches_actual_native_backward(kind):
    agent, _ = batch(kind)
    prepared = deepcopy(agent)._prepare_episodes()
    gradient, _ = measurement.loss_gradient(agent, prepared, range(4))
    native = deepcopy(agent)
    captured = {}
    handles = []
    for name, p in native.actor.named_parameters():

        def record(g, name=name):
            captured[name] = g.detach().clone()

        handles.append(p.register_hook(record))
    native.update()
    for h in handles:
        h.remove()
    expected = torch.cat(
        [
            captured.get(name, torch.zeros_like(p)).flatten()
            for name, p in native.actor.named_parameters()
        ]
    ).numpy()
    np.testing.assert_allclose(gradient, expected, atol=2e-6, rtol=2e-5)


def equal(a, b):
    if torch.is_tensor(a):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def state(agent):
    return {
        "actor": deepcopy(agent.actor.state_dict()),
        "value": deepcopy(agent.value.state_dict()),
        "actor_optimizer": deepcopy(agent.actor_optimizer.state_dict()),
        "value_optimizer": deepcopy(agent.value_optimizer.state_dict()),
        "normalizer": deepcopy(agent.observation_normalizer.state_dict()),
        "rng": deepcopy(agent.rng.bit_generator.state),
        "torch_rng": torch.get_rng_state().clone(),
        "updates": agent.updates,
    }


def test_diagnostic_does_not_change_next_native_update_or_rng():
    agent, _ = batch("srl_c")
    untouched = deepcopy(agent)
    before = state(agent)
    measured = measurement.gradient_repeats(agent, update=10)
    assert equal(before, state(agent))
    assert measured["repeats"] == 32 and measured["gradients"].shape[0] == 32
    untouched.update()
    agent.update()
    assert equal(state(untouched), state(agent))


def test_fresh_after_update_probe_preserves_training_agent(tmp_path):
    agent, adapter = batch("srl_c")
    agent.update()
    before = state(agent)
    cfg = {
        "diagnostic_seed_start": 150000,
        "diagnostic_pool": 4,
        "credit": {"fraction": 1.0, "K": 2, "H": 2, "mix": 0.5},
    }
    measured = study.diagnostic(agent, adapter, cfg, tmp_path, 1)
    assert equal(before, state(agent))
    assert measured["seed_start"] == 150004 and measured["pool_episodes"] == 4
    assert (
        len(
            (tmp_path / "diagnostic_1_attempt0/gradient_draws.jsonl")
            .read_text()
            .splitlines()
        )
        == 32
    )


def test_two_episode_gradient_variance_matches_bernoulli_frequency_formula():
    agent, _ = batch("srl", 2)
    prepared = deepcopy(agent)._prepare_episodes()
    g0, _ = measurement.loss_gradient(agent, prepared, [0])
    g1, _ = measurement.loss_gradient(agent, prepared, [1])
    result = measurement.gradient_repeats(agent, update=20, minibatch=1)
    n0 = int(np.sum(result["indices"][:, 0] == 0))
    n1 = 32 - n0
    difference = g1.astype(np.float64) - g0.astype(np.float64)
    expected = n0 * n1 / (32 * 31) * float(difference @ difference)
    assert result["trace_covariance"] == pytest.approx(expected, rel=1e-10, abs=1e-12)
    with pytest.raises(ValueError, match="32"):
        measurement.gradient_repeats(agent, update=20, repeats=16)


def test_minibatches_match_native_distinct_episode_sampling():
    agent, _ = batch("srl", 8)
    result = measurement.gradient_repeats(agent, update=10, minibatch=4)
    assert all(len(set(row)) == 4 for row in result["indices"])
    assert len({tuple(row) for row in result["indices"]}) > 1


def test_twenty_kl_threshold_half_lr_once_and_nan():
    agent, _ = batch("srl")
    monitor = measurement.MonitorState()
    for _ in range(19):
        assert monitor.inspect({"approx_kl": 0.10001}) is None
    assert monitor.inspect({"approx_kl": 0.10001}) == "persistent_KL"
    old = [
        o.param_groups[0]["lr"] for o in [agent.actor_optimizer, agent.value_optimizer]
    ]
    monitor.restart(agent)
    assert [
        o.param_groups[0]["lr"] for o in [agent.actor_optimizer, agent.value_optimizer]
    ] == [x * 0.5 for x in old]
    assert monitor.state() == {"high_kl": 0, "restarts": 1}
    assert monitor.inspect({"approx_kl": float("nan")}) == "nonfinite"
    with pytest.raises(FloatingPointError, match="exhausted"):
        monitor.restart(agent)


def fields(data):
    def varint(offset):
        result, shift = 0, 0
        while True:
            byte = data[offset]
            offset += 1
            result |= (byte & 127) << shift
            if byte < 128:
                return result, offset
            shift += 7

    out = []
    offset = 0
    while offset < len(data):
        key, offset = varint(offset)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, offset = varint(offset)
        elif wire in (1, 5):
            width = 8 if wire == 1 else 4
            value = data[offset : offset + width]
            offset += width
        elif wire == 2:
            size, offset = varint(offset)
            value = data[offset : offset + size]
            offset += size
        else:
            raise AssertionError("unexpected protobuf wire type")
        out.append((number, wire, value))
    return out


def test_scalar_event_schema_crc_vector_and_missing_values(tmp_path):
    assert measurement.crc32c(b"123456789") == 0xE3069283
    log = measurement.MetricLog(tmp_path)
    log.write(
        {
            "update": 10,
            "cohorts": 640,
            "return_mean": 1.25,
            "gradient_variance": 0.5,
            "explained_variance": None,
        }
    )
    data = log.events.read_bytes()
    records = []
    while data:
        size = struct.unpack("<Q", data[:8])[0]
        body = data[12 : 12 + size]
        for raw, crc in [(data[:8], data[8:12]), (body, data[12 + size : 16 + size])]:
            c = measurement.crc32c(raw)
            expected = (((c >> 15) | (c << 17)) + 0xA282EAD8) & 0xFFFFFFFF
            assert struct.unpack("<I", crc)[0] == expected
        records.append(body)
        data = data[16 + size :]
    assert records[0] == b"\x1a\x0dbrain.Event:2"
    event = fields(records[1])
    assert next(v for n, w, v in event if n == 2) == 10
    summary = next(v for n, w, v in event if n == 5)
    scalars = {}
    for n, w, value in fields(summary):
        item = {n: v for n, w, v in fields(value)}
        scalars[item[1].decode()] = struct.unpack("<f", item[2])[0]
    assert (
        scalars["return_mean"] == 1.25
        and scalars["gradient_variance"] == 0.5
        and "explained_variance" not in scalars
    )
    with pytest.raises(FloatingPointError):
        log.write({"update": 11, "return_mean": float("nan")})


def test_actual_continuation_rolls_back_once_without_repeating_parent(
    tmp_path, monkeypatch
):
    agent, adapter = batch("srl")
    agent.update()
    assert agent.updates == 1
    original = StructuredPPOAgent.update
    calls = 0

    def fail_once(self):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise FloatingPointError("controlled native failure")
        return original(self)

    monkeypatch.setattr(StructuredPPOAgent, "update", fail_once)
    cfg = {
        "batch": 2,
        "target_update": 2,
        "diagnostic_seed_start": 150000,
        "diagnostic_pool": 4,
        "validation_seed_start": 200000,
        "n_validation": 2,
        "credit": None,
    }
    result = study.continuation(
        agent,
        adapter,
        cfg,
        tmp_path,
        {"consumed": 4, "training_fingerprint": "fixture"},
        lambda _: None,
    )
    assert (
        result["restarts"] == 1
        and result["additional_optimizer_cohorts"] == 2
        and result["training_rollout_executions"] == 4
    )
    assert (tmp_path / "abort_1.json").exists() and (tmp_path / "aborted_1.pt").exists()
    assert result["aborts"][0]["completed_rollout_seeds"] == [100004, 100005]
    metadata = json.loads((tmp_path / "monitoring.json").read_text())
    assert metadata["history"][0]["learning_rate"] == pytest.approx(1.5e-4)


def test_final_seed_guard_precedes_any_diagnostic_collection(tmp_path, monkeypatch):
    agent, adapter = batch("srl")
    agent.reset_rollout()
    monkeypatch.setattr(
        study,
        "collect_structured",
        lambda *a, **k: pytest.fail("physical collection reached"),
    )
    cfg = {
        "batch": 2,
        "target_update": 10,
        "diagnostic_seed_start": 300000,
        "diagnostic_pool": 4,
        "validation_seed_start": 200000,
        "n_validation": 2,
        "credit": None,
    }
    with pytest.raises(ValueError, match="split"):
        study.continuation(
            agent,
            adapter,
            cfg,
            tmp_path,
            {"consumed": 4, "training_fingerprint": "fixture"},
            lambda _: None,
        )


def test_frozen_parent_guard_precedes_any_training(tmp_path):
    from types import SimpleNamespace

    parent = tmp_path / "parent"
    parent.mkdir()
    (parent / "spec.json").write_text("{}")
    with pytest.raises(ValueError, match="changed frozen"):
        study.run(
            {
                "parameters": {
                    "parent_directory": str(parent),
                    "parent_hashes": {"spec.json": "0" * 64},
                }
            },
            SimpleNamespace(directory=tmp_path),
        )
