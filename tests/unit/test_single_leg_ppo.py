"""Executed price image, independent DP law, native-update parity and tape reuse."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.special import expit

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.hashing import schema_hash
from airline_market.core.rng import StreamRegistry
from airline_market.core.types import CoefficientAction
from airline_market.experiments.learning_pilot import initial_agent
from airline_market.learn.ppo import collect, load_checkpoint
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.sim.kernel import ExactKernel
from airline_market.spec import compile_market, dump_spec, load_spec

if os.environ.get("SINGLE_LEG_STAGE"):
    spec = importlib.util.spec_from_file_location(
        "airline_market.experiments.single_leg_ppo_study",
        Path(os.environ["SINGLE_LEG_STAGE"]) / "single_leg_ppo_study.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
from airline_market.experiments import single_leg_ppo_study as study


def setup(tmp_path, *, T=2, lo=80.0, hi=80.03):
    data = json.loads(Path("config/benchmarks/V7_single_leg_T5.json").read_text())
    data["horizon"]["n_periods"] = T
    data["products"][0].update(fare_floor=lo, fare_cap=hi, reference_fare=(lo + hi) / 2)
    data["pricing"][0].update(
        base={"A_OD": (lo + hi) / 2}, theta_bounds=[[-(hi - lo) / 2, (hi - lo) / 2]]
    )
    c = compile_market(load_spec(data))
    path = tmp_path / "input.json"
    dump_spec(c.spec, path)
    return c, {
        "source_spec": str(path),
        "source_spec_sha256": file_hash(path),
        "compiled_spec_hash": c.spec_hash,
        "validation_seed_start": 205600,
        "n_validation": 2,
    }


def reference(tmp_path):
    c, p = setup(tmp_path)
    root = tmp_path / "reference"
    root.mkdir()
    result = study.prepare({"parameters": p}, SimpleNamespace(directory=root))
    atomic_json(root / "results.json", result)
    atomic_json(
        root / "COMPLETE.json",
        {"status": "completed", "results_sha256": file_hash(root / "results.json")},
    )
    p.update(
        reference_directory=str(root),
        reference_hashes={str(x): file_hash(x) for x in root.iterdir() if x.is_file()},
        train_seed_start=100000,
        n_train=4,
        network_seed=0,
        cohorts_per_update=4,
        audit_every_cohorts=4,
        ppo={
            "hidden_dim": 8,
            "update_epochs": 1,
            "minibatch_episodes": 2,
            "gamma": 1.0,
        },
    )
    return c, p


def test_all_native_attainable_cents_are_the_declared_DP_class():
    c = compile_market(load_spec("config/benchmarks/V7_single_leg_T5.json"))
    target = study.classical_dp(c)
    assert target["attainable_fare_count"] == 28001
    env = FocalCarrierEnv(c, "A", "direct")
    obs, _ = env.reset(100010)
    for cents in range(2000, 30001):
        fare = cents / 100
        assert env.price_rule.prices((fare - 160.0,), obs)[0] == fare
    assert c.spec.pricing[0].theta_grid is None


def test_classical_DP_equals_general_kernel_policy_expectation(tmp_path):
    c, _ = setup(tmp_path, T=3)
    target = study.classical_dp(c)
    price = np.asarray(target["prices_USD"])
    value, nodes = study.exact_tree(
        c,
        lambda o, h, a, z, r: (
            (price[o.t, o.own_inventory[0]] - target["base"],),
            None,
            None,
        ),
    )
    assert value * target["profit_scale"] == pytest.approx(
        target["V_star_USD"], abs=1e-10
    )
    assert nodes and max(abs(x["mass"] - 1) for x in nodes) < 1e-12


def test_native_transition_probability_equals_independent_logit(tmp_path):
    c, _ = setup(tmp_path)
    k = ExactKernel(c)
    _, state = k.initial_distribution()[0]
    theta = 0.003
    transitions = k.transition(state, 0, CoefficientAction((theta,)))
    purchase = sum(p for p, s, _ in transitions if s.carriers[0].counts[0] == 1)
    env = FocalCarrierEnv(c, "A", "direct")
    obs, _ = env.reset(100011)
    fare = env.price_rule.prices((theta,), obs)[0]
    assert purchase == pytest.approx(0.8 * expit(3.0 - 0.025 * fare), abs=1e-12)


def test_saved_tape_bank_reuses_exact_immutable_native_draws(tmp_path, monkeypatch):
    c, p = reference(tmp_path)
    rows = json.loads(
        (Path(p["reference_directory"]) / "DP_validation.json").read_text()
    )
    with np.load(
        Path(p["reference_directory"]) / "validation_tapes.npz", allow_pickle=False
    ) as source:
        bank = {n: source[n] for n in source.files}
    monkeypatch.setattr(
        study.ExogenousTape,
        "generate",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("new tape generated")),
    )
    env = study.StoredTapeEnv(c, "A", "direct")
    tape = study.tape_from_bank(bank, 0, 205600, 2)
    assert (
        schema_hash(tape) == rows[0]["tape_sha256"] and not tape.gumbel.flags.writeable
    )
    env.tape = tape
    env.reset(205600)
    assert env.engine.tape is tape
    with pytest.raises(ValueError, match="tape/seed"):
        env.reset(205601)


def test_native_first_update_parity_and_complete_worker_contract(tmp_path):
    c, p = reference(tmp_path)
    root = tmp_path / "train"
    root.mkdir()
    result = study.train({"parameters": p}, SimpleNamespace(directory=root))
    assert (
        result["status"] == "completed"
        and result["training_cohorts"] == 4
        and result["updates"] == 1
    )
    assert result["all_five_seed_criterion"] is None
    manual, agent = initial_agent(
        c, {"learner": "ppo", "network_seed": 0, "ppo": p["ppo"]}
    )
    for seed in range(100000, 100004):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(
                StreamRegistry(seed, "train/flat/policy").child_seed() % 2**31
            )
            e, logp, _ = collect(manual, agent, seed)
        agent.add_episode(
            e, actions_are_physical=False, behavior_log_probabilities=logp
        )
    agent.update()
    loaded = load_checkpoint(root / "last.pt", manual, restore_rng=True)
    assert study.inference_fingerprint(agent) == study.inference_fingerprint(loaded)
    assert result["validation_pairs_sha256"] == file_hash(
        root / "validation_pairs.jsonl"
    )
    assert [
        json.loads(x)["seed"]
        for x in (root / "validation_pairs.jsonl").read_text().splitlines()
    ] == [205600, 205601]


def test_exact_audit_preserves_parameters_normalizer_rng_and_public_inputs(tmp_path):
    c, _ = setup(tmp_path)
    adapter, agent = initial_agent(
        c, {"learner": "ppo", "network_seed": 1, "ppo": {"hidden_dim": 8}}
    )
    before = study.inference_fingerprint(agent)
    rng = torch.get_rng_state().clone()
    value, rows = study.exact_actor(adapter, agent)
    assert value > 0 and before == study.inference_fingerprint(agent)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(r["observation"].privileged is None for r in rows)


def test_exact_tree_preserves_recurrent_history_and_previous_rewards(tmp_path):
    c, _ = setup(tmp_path, T=3, lo=20.0, hi=300.0)

    def policy(obs, hidden, previous, coordinates, reward):
        assert (hidden or 0) == obs.t
        fare = 20.0 if obs.t == 0 else 300.0 if reward > 0 else 90.0
        return (fare - 160.0,), obs.t + 1, None

    measured, nodes = study.exact_tree(c, policy)

    def independent(t, stock, previous_sale):
        if t == 3 or stock == 0:
            return 0.0
        fare = 20.0 if t == 0 else 300.0 if previous_sale else 90.0
        p = 0.8 * expit(3.0 - 0.025 * fare)
        return p * (fare - 20.0 + independent(t + 1, stock - 1, True)) + (
            1 - p
        ) * independent(t + 1, stock, False)

    assert measured * 100 == pytest.approx(independent(0, 2, False), abs=1e-10)
    assert len({tuple(n["history_branch_path"]) for n in nodes}) == len(nodes)


@pytest.mark.parametrize(
    "p",
    [
        {"validation_seed_start": 300000, "n_validation": 2},
        {"validation_seed_start": 299999, "n_validation": 2},
        {
            "validation_seed_start": 205600,
            "n_validation": 2,
            "train_seed_start": 100000,
            "n_train": 100001,
        },
        {
            "validation_seed_start": 205600,
            "n_validation": 2,
            "train_seed_start": 100000,
            "n_train": 4,
            "network_seed": 10,
        },
    ],
)
def test_split_guards_precede_any_market_construction(p, monkeypatch):
    monkeypatch.setattr(
        study,
        "market",
        lambda *a: (_ for _ in ()).throw(AssertionError("market opened")),
    )
    with pytest.raises(ValueError):
        study.train({"parameters": p}, None)
