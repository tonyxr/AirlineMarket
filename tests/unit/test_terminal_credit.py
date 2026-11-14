"""Terminal Q identity, native R2 contrast law, zeros, hygiene and split guards."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from airline_market.campaign.artifacts import file_hash
from airline_market.campaign.instances import build_instance
from airline_market.core.hashing import schema_hash
from airline_market.experiments.learning_pilot import initial_agent
from airline_market.legacy_toy.model import Config, Model
from airline_market.sim.kernel import ExactKernel
from airline_market.spec import compile_market, dump_spec

if os.environ.get("CREDIT_STAGE"):
    spec = importlib.util.spec_from_file_location(
        "airline_market.experiments.terminal_credit_study",
        Path(os.environ["CREDIT_STAGE"]) / "terminal_credit_study.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
from airline_market.experiments import terminal_credit_study as study


def setup(T=3):
    c = compile_market(build_instance("toy_hub", {"T": T, "actions": 4}))
    adapter, agent = initial_agent(
        c, {"learner": "srl_c", "network_seed": 0, "ppo": {"hidden_dim": 8}}
    )
    return adapter, agent, ExactKernel(c)


def test_three_way_sign_and_distinct_credit_targets():
    assert study.contrasts([2.0, 1.0, 8.0, 5.0]) == {
        "centered": -2.0,
        "hold_relative": 1.0,
    }
    assert [study.sign(v) for v in [-2.0, -1e-13, 0.0, 1e-13, 2.0]] == [-1, 0, 0, 0, 1]
    rows = [
        {
            "exact": {"centered": a, "hold_relative": a},
            "estimated": {"centered": b, "hold_relative": b},
        }
        for a, b in [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (-1.0, -1.0)]
    ]
    s = study.summarize(rows)["centered"]
    assert s["agreement"] == 0.5 and s["nonzero_agreement"] == 0.5
    assert s["exact_zero_count"] == 2 and s["disagreement_count"] == 2
    assert not s["terminal_subset_meets_0_90"]


def test_all_zero_nonzero_denominator_stays_null():
    row = {
        "exact": study.contrasts([0.0, 0.0]),
        "estimated": study.contrasts([0.0, 0.0]),
    }
    assert study.summarize([row])["centered"]["nonzero_agreement"] is None


@pytest.mark.parametrize(
    "start,count", [(199999, 1), (299999, 2), (300000, 1), (200000, 0)]
)
def test_reserved_and_wrong_split_rejected_before_rollout(start, count):
    with pytest.raises(ValueError, match="validation"):
        study.validation_seeds(start, count)


def test_native_branch_and_terminal_q_equal_independent_legacy_expectations():
    adapter, agent, kernel = setup()
    before_actor = schema_hash(
        {k: v.detach().numpy() for k, v in agent.actor.state_dict().items()}
    )
    row = study.one_cohort(adapter, agent, kernel, 200037)
    assert row["parent_unchanged"] and adapter.obs.t == 2
    assert before_actor == schema_hash(
        {k: v.detach().numpy() for k, v in agent.actor.state_dict().items()}
    )
    env, state = adapter.env, kernel.encode(adapter.env.engine.world)
    legacy = Model(Config(horizon=3, shock_step=2))
    obs = adapter.obs
    own = obs.own_inventory
    rival = kernel.engine(state).world.carriers[1]
    from airline_market.sim.ledger import inventory

    rival_stock = tuple(inventory(rival, adapter.c, 1))
    costs = [15.0, 30.0]
    sunk_cost = sum(costs[j] * state.carriers[0].counts[j] for j in range(2))
    expected = []
    for prices in row["price_alternatives"]:
        own_prices = tuple(prices[j] for j in adapter.c.products_of[0])
        probs = legacy.outcomes(
            obs.t,
            tuple(own),
            rival_stock,
            own_prices,
            obs.own_previous_prices,
            bool(state.exo),
        )
        expected.append(
            (-sunk_cost + sum(probs[j] * (own_prices[j] - costs[j]) for j in range(2)))
            / env.objective.spec.profit_scale
        )
    np.testing.assert_allclose(row["exact_Q"], expected, atol=1e-12, rtol=0)
    assert np.ptp(row["exact_Q"]) > 1e-6
    for a, realized in zip(row["physical_alternatives"], row["sampled_returns"]):
        branch = adapter.env.branch()
        assert branch.step(adapter.price_action(a, adapter.obs))[1] == realized
    assert study.contrasts(row["sampled_returns"]) == row["estimated"]
    assert np.allclose(row["exact_probability_mass"], 1.0, atol=1e-12)


def test_nonterminal_exact_target_rejected_and_kernel_mass_failure_kept():
    adapter, agent, kernel = setup()
    adapter.reset(200038)
    choices = study.alternatives(agent, np.zeros(agent.action_dim), None, 773)
    with pytest.raises(ValueError, match="last decision"):
        study.terminal_q(kernel, adapter, choices)
    for _ in range(2):
        adapter.step(choices[0])
    kernel.transition = lambda *args, **kwargs: []
    with pytest.raises(ValueError, match="probability mass"):
        study.terminal_q(kernel, adapter, choices)


def test_real_driver_keeps_all_rows_and_source_hash_guards(tmp_path):
    adapter, _, _ = setup()
    source = tmp_path / "source.json"
    dump_spec(adapter.c.spec, source)
    directory = tmp_path / "output"
    directory.mkdir()
    p = {
        "source_spec": str(source),
        "source_spec_sha256": file_hash(source),
        "compiled_spec_hash": adapter.c.spec_hash,
        "seed_start": 200039,
        "n_cohorts": 3,
        "K": 4,
        "H": 5,
        "sign_tolerance": 1e-12,
        "instance": "synthetic_unit_T3A4",
        "actor": {"learner": "srl_c", "network_seed": 0, "ppo": {"hidden_dim": 8}},
    }
    result = study.run({"parameters": p}, SimpleNamespace(directory=directory))
    rows = [
        json.loads(line)
        for line in (directory / "comparisons.jsonl").read_text().splitlines()
    ]
    assert [r["seed"] for r in rows] == [200039, 200040, 200041]
    assert result["full_V7_R2_pass"] is None and result["cohorts"] == 3
    assert result["comparisons_sha256"] == file_hash(directory / "comparisons.jsonl")
    p["source_spec_sha256"] = "changed"
    with pytest.raises(ValueError, match="physical source"):
        study.run({"parameters": p}, SimpleNamespace(directory=directory))
