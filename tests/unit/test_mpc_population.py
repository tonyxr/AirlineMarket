"""Actual MPC policy export, episode mixing, paired deviations and expansion."""

import importlib
import os
from dataclasses import replace

import numpy as np
import pytest

from airline_market.core.errors import KernelLimitError
from airline_market.core.hashing import canonical_json
from airline_market.game.type_belief import HistoryController, HistoryPolicy
from airline_market.game.type_mpc import EpisodeTypeMPCController, TypeMPCConfig
from airline_market.sim.engine import MarketEngine
from airline_market.sim.kernel import ExactKernel
from airline_market.solvers.brute_history import binary_sale_micro
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

module = importlib.import_module(
    os.environ.get(
        "AIRLINE_POPULATION_TEST_MODULE", "airline_market.game.mpc_population"
    )
)


def fixture(single=False):
    s = binary_sale_micro(
        {"T": 2, "rival_support": [2], "own_capacity": 2, "arrival": 0.9}
    )
    a = s.pricing[0]
    return compile_market(
        replace(
            s,
            name="MPC_population_verification_new_family",
            timing=replace(s.timing, mode="simultaneous", leaders=()),
            pricing=(a, replace(a, carrier="B", base={"B_p": 100})),
            information=tuple(
                replace(s.information[0], name="own", observer=c) for c in ["A", "B"]
            ),
            segments=tuple(
                replace(
                    g,
                    classes=tuple(
                        replace(
                            k, carrier_pref={"A": 0, "B": 0}, market_value={"*": 4.0}
                        )
                        for k in g.classes
                    ),
                )
                for g in s.segments
            ),
            demand=replace(
                s.demand, arrival_probability=PeriodSchedule((0, 1), (0.9, 0))
            )
            if single
            else s.demand,
        )
    )


def bank(c, carrier, indices=(0, 1)):
    p = next(q for q in c.spec.pricing if q.carrier == carrier)
    return [HistoryPolicy(p.theta_grid, {}, default=i, arm="own") for i in indices]


def test_exported_policy_replays_online_joint_type_MPC(monkeypatch):
    from airline_market.game.type_psro_keyed import EpisodeTypeDP

    # The response materializer must use native MPC, never a Bellman oracle.
    monkeypatch.setattr(
        EpisodeTypeDP,
        "q_values",
        lambda *a: (_ for _ in ()).throw(AssertionError("Bellman called")),
    )
    c = fixture()
    opponents = bank(c, "B")
    config = TypeMPCConfig(8, 8)
    policy = module.materialize_response(
        c, "A", "own", opponents, [0.3, 0.7], config, episode_index=7
    )
    assert policy.default == config.tail_index and len(policy.table) >= 3
    loaded = HistoryPolicy.from_dict(policy.to_dict())
    assert loaded.hash() == policy.hash()
    from airline_market.game.type_mpc import JointTypeFilter

    target = JointTypeFilter(
        c, "A", "own", [{"B": HistoryController(p)} for p in opponents], [0.3, 0.7]
    )
    frontier = [n for p, n in target.model.roots() if p > 0]
    seen = set()
    while frontier:
        node = frontier.pop()
        if node.observation.t == c.spec.horizon.n_periods:
            continue
        assert node.path in loaded.table
        seen.add(node.path)
        frontier.extend(
            n
            for p, _, n in target.model.successors(node, loaded.action(node.path))
            if p > 0
        )
    assert seen == set(loaded.table)
    assert loaded.action(("unseen permitted-history key",)) == config.tail_index
    for seed in range(200896, 200904):
        online = EpisodeTypeMPCController(
            "A",
            "own",
            [{"B": HistoryController(p)} for p in opponents],
            [0.3, 0.7],
            config,
            episode_index=7,
        )
        left = MarketEngine(
            c, {"A": online, "B": module.EpisodeHistoryMixture(opponents, [0.3, 0.7])}
        )
        right = MarketEngine(
            c,
            {
                "A": HistoryController(loaded),
                "B": module.EpisodeHistoryMixture(opponents, [0.3, 0.7]),
            },
        )
        left.reset(seed)
        right.reset(seed)
        while not left.terminal():
            left.step()
            right.step()
            assert canonical_json(left.observe(0, "own")) == canonical_json(
                right.observe(0, "own")
            )
    assert (
        HistoryController.exact_compatible is False and online.exact_compatible is False
    )


def test_partial_policy_nodes_survive_explicit_limits():
    c = fixture()
    events = []
    with pytest.raises(KernelLimitError, match="max_policy_nodes"):
        module.materialize_response(
            c,
            "A",
            "own",
            bank(c, "B"),
            [0.5, 0.5],
            TypeMPCConfig(2, 2),
            max_policy_nodes=1,
            progress=events.append,
        )
    assert any(
        x["stage"] == "response_policy_node" and "policy_row" in x for x in events
    )
    with pytest.raises(KernelLimitError, match="max_support"):
        module.materialize_response(
            c, "A", "own", bank(c, "B"), [0.5, 0.5], TypeMPCConfig(2, 2), max_support=1
        )


def test_episode_mixture_never_redraws_and_snapshot_replays():
    c = fixture()
    ctrl = module.EpisodeHistoryMixture(bank(c, "A"), [0.5, 0.5])
    e = MarketEngine(c, {"A": ctrl, "B": HistoryController(bank(c, "B")[0])})
    e.reset(200896)
    choice = ctrl.choice
    e.step()
    before = e.snapshot()
    e.step()
    obs = canonical_json(e.observe(0, "own"))
    assert ctrl.choice == choice
    e.restore(before)
    e.step()
    assert canonical_json(e.observe(0, "own")) == obs
    assert ctrl.choice == choice
    with pytest.raises(ValueError, match="persistent"):
        state = ctrl.snapshot()
        ctrl.restore((1 - choice, state[1], state[2]))
    with pytest.raises(ValueError, match="joint type"):
        ExactKernel(c, {"A": ctrl})


def test_native_profile_cache_avoids_matching_profile_cohort_reruns(monkeypatch):
    c = fixture()
    policies = {q: bank(c, q)[0] for q in ["A", "B"]}
    cache = {}
    calls = []
    original = module._native

    def counted(*args):
        calls.append(args[-1])
        return original(*args)

    monkeypatch.setattr(module, "_native", counted)
    first = module.profile_values(c, policies, [200896, 200897], cache)
    assert module.profile_values(c, policies, [200896, 200897], cache) == first
    assert len(calls) == 2 and len(cache) == 2


def test_fresh_deviations_pair_opponent_type_and_retain_negative_gains():
    c = fixture(single=True)
    populations = {q: bank(c, q) for q in ["A", "B"]}
    mixtures = {"A": [0.5, 0.5], "B": [0.3, 0.7]}
    result = module.paired_deviation(
        c,
        "A",
        populations["A"][0],
        populations,
        mixtures,
        tuple(range(200896, 200960)),
        bootstrap=100,
    )
    assert result["n"] == 64 and len(result["rows"]) == 64
    assert [r["seed"] for r in result["rows"]] == list(range(200896, 200960))
    assert result["mean"] == pytest.approx(
        np.mean([r["objective_difference"] for r in result["rows"]])
    )
    assert all(r["opponent_type"] in [0, 1] for r in result["rows"])
    # No clipping or replacement of unfavorable realized differences.
    assert any(r["objective_difference"] < 0 for r in result["rows"])


def test_real_population_expansion_uses_responses_and_fresh_deviations():
    c = fixture(single=True)
    populations = {"A": bank(c, "A", (0,)), "B": bank(c, "B", (1,))}
    events = []
    result = module.sampled_mpc_psro(
        c,
        populations,
        TypeMPCConfig(64, 8, switch_standard_errors=0),
        max_iters=2,
        n_payoff=16,
        payoff_seed_start=200512,
        n_deviation=32,
        deviation_seed_start=200768,
        progress=events.append,
    )
    assert (
        result["certified"] is False
        and result["exact_exploitability_lower_bound"] is None
    )
    assert any(e["stage"] == "response_complete" for e in events)
    assert set(result["history"][0]["responses"]) == {"A", "B"}
    assert max(len(v) for v in result["populations"].values()) >= 2
    for row in result["history"]:
        for response in row["responses"].values():
            seeds = [r["seed"] for r in response["evaluation"]["rows"]]
            assert seeds == list(
                range(200768 + row["iteration"] * 32, 200800 + row["iteration"] * 32)
            )
            assert not set(seeds) & set(range(200512, 200528))
    assert populations["A"][0].table == {}  # input populations never mutated


def test_seed_and_prior_guards_precede_any_collection():
    c = fixture()
    pop = {q: bank(c, q, (0,)) for q in ["A", "B"]}
    with pytest.raises(ValueError, match="validation"):
        module.sampled_mpc_psro(c, pop, TypeMPCConfig(2, 2), payoff_seed_start=300000)
    with pytest.raises(ValueError, match="disjoint"):
        module.sampled_mpc_psro(
            c, pop, TypeMPCConfig(2, 2), deviation_seed_start=200512
        )
    with pytest.raises(ValueError, match="mixture"):
        module.EpisodeHistoryMixture(pop["A"], [0.2])


def test_actual_driver_writes_policies_and_deviations(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from types import SimpleNamespace

    from airline_market.campaign.artifacts import file_hash
    from airline_market.spec import dump_spec

    if os.environ.get("AIRLINE_POPULATION_TEST_MODULE") == "mpc_population":
        monkeypatch.setitem(sys.modules, "airline_market.game.mpc_population", module)
        spec = importlib.util.spec_from_file_location(
            "staged_pop_driver",
            __file__.replace("test_mpc_population.py", "mpc_population_study.py"),
        )
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
    else:
        driver = importlib.import_module(
            "airline_market.experiments.mpc_population_study"
        )
    source = tmp_path / "spec.json"
    dump_spec(fixture(single=True).spec, source)
    p = {
        "source_spec": str(source),
        "source_spec_sha256": file_hash(source),
        "arms": {"A": "own", "B": "own"},
        "initial_action_indices": [0],
        "mpc": {"n_scenarios": 2, "n_reporting": 2},
        "max_iters": 1,
        "tolerance": 1e-6,
        "payoff_seed_start": 200512,
        "n_payoff": 2,
        "deviation_seed_start": 200768,
        "n_deviation": 2,
        "max_policy_nodes": 100,
        "max_support": 100,
    }
    result = driver.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
    assert result["status"] == "completed" and result["certified"] is False
    assert (tmp_path / "events.jsonl").exists() and (
        tmp_path / "policies.json"
    ).exists()
