"""Persistent type law, common-action rollout and native observation-only control."""

import importlib
import os
from dataclasses import replace

import numpy as np
import pytest

from airline_market.agents.fixed import FixedCoefficientController
from airline_market.core.errors import KernelLimitError, ZeroLikelihoodError
from airline_market.core.types import CoefficientAction
from airline_market.game.type_belief import TypeNode
from airline_market.sim.accounting import carrier_metrics
from airline_market.sim.engine import MarketEngine
from airline_market.sim.kernel import ExactKernel
from airline_market.solvers.brute_history import binary_sale_micro
from airline_market.spec import compile_market, load_spec
from airline_market.spec.schedule import PeriodSchedule

module = importlib.import_module(
    os.environ.get("AIRLINE_TYPE_MPC_TEST_MODULE", "airline_market.game.type_mpc")
)
JointTypeFilter, EpisodeTypeMPC, TypeMPCConfig, EpisodeTypeMPCController = (
    module.JointTypeFilter,
    module.EpisodeTypeMPC,
    module.TypeMPCConfig,
    module.EpisodeTypeMPCController,
)


def fixture(single=False):
    s = binary_sale_micro(
        {"T": 2, "own_capacity": 2, "rival_support": [2], "arrival": 0.9}
    )
    a = s.pricing[0]
    return compile_market(
        replace(
            s,
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
                            k, carrier_pref={"A": 0, "B": 0}, market_value={"*": 2.5}
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


def make(c, focal="A", probabilities=(0.7, 0.3), max_support=20000):
    rival = "B" if focal == "A" else "A"
    types = [{rival: FixedCoefficientController((theta,))} for theta in [-40, 40]]
    belief = JointTypeFilter(c, focal, "own", types, probabilities, max_support)
    belief.reset(belief.model.roots()[0][1].observation)
    return belief


def exact_fixed_tail(model, node, first, tail=0):
    def value(n, index):
        if n.observation.t == model.compiled.spec.horizon.n_periods:
            return model.terminal(n)
        return sum(
            p * (reward + value(child, tail))
            for p, reward, child in model.successors(n, index)
        )

    return value(node, first) + node.observation.own_cash / model.objective.profit_scale


def test_filter_matches_exact_joint_posteriors_and_learns_type():
    belief = make(fixture())
    root = belief.snapshot()
    rows = belief.model.successors(root, 0)
    assert any(
        np.max(np.abs(n.type_probabilities(2) - [0.7, 0.3])) > 0.01 for _, _, n in rows
    )
    for _, _, expected in rows:
        belief.restore(root)
        belief.update(0, expected.observation)
        assert belief.model.key(belief.node) == belief.model.key(expected)


def test_impossible_observations_and_explicit_support_limits():
    belief = make(fixture())
    with pytest.raises(ZeroLikelihoodError):
        belief.update(0, replace(belief.node.observation, t=1, own_cash=1234567))
    with pytest.raises(KernelLimitError, match="max_support"):
        make(fixture(), max_support=1)


@pytest.mark.parametrize("focal", ["A", "B"])
def test_common_action_mixture_value_matches_independent_exact_tail(focal):
    c = fixture(single=True)
    belief = make(c, focal)
    planner = EpisodeTypeMPC(belief, TypeMPCConfig(1024, 32, switch_standard_errors=0))
    scores = planner.scores([0, 1], planner.scenarios(1024, 7))
    exact = np.array([exact_fixed_tail(belief.model, belief.node, a) for a in [0, 1]])
    assert np.all(
        np.abs(scores.mean(axis=1) - exact)
        <= 5 * scores.std(axis=1, ddof=1) / 32 + 1e-4
    )
    # This family has opposing type-revealed optimizers, so strategy fusion is strict.
    components = []
    for kind in range(2):
        mass = {key: p for key, p in belief.node.mass.items() if key[0] == kind}
        total = sum(mass.values())
        node = TypeNode(
            belief.node.observation, {k: p / total for k, p in mass.items()}
        )
        components.append([exact_fixed_tail(belief.model, node, a) for a in [0, 1]])
    assert np.argmax(components[0]) != np.argmax(components[1])
    assert np.dot([0.7, 0.3], np.max(components, axis=1)) > exact.max() + 1e-4
    decision = planner.choose(7)
    assert decision.estimate < np.dot([0.7, 0.3], np.max(components, axis=1)) + 0.008
    assert decision.action.theta in c.spec.pricing[c.carrier_index[focal]].theta_grid


def test_scenario_prefix_reporting_independence_and_type_persistence():
    belief = make(fixture())
    planner = EpisodeTypeMPC(belief)
    short, long = planner.scenarios(8, 9), planner.scenarios(16, 9)
    assert [(s.kind, s.forecast_seed) for s in short] == [
        (s.kind, s.forecast_seed) for s in long[:8]
    ]
    held = planner.scenarios(8, 9, 1)
    assert [s.forecast_seed for s in short] != [s.forecast_seed for s in held]
    for scenario in long:
        e = scenario.engine.branch()
        while not e.terminal():
            e.step({0: CoefficientAction((-40,))})
            assert e.controllers[1].theta == (
                (-40.0,) if scenario.kind == 0 else (40.0,)
            )


def test_rollouts_do_not_change_parent_hypotheses_or_filter():
    belief = make(fixture())
    planner = EpisodeTypeMPC(belief, TypeMPCConfig(8, 8))
    scenarios = planner.scenarios(8)
    before = [
        (
            s.kind,
            s.forecast_seed,
            belief.model.kernels[s.kind].encode(s.engine.world, 0),
        )
        for s in scenarios
    ]
    node = belief.model.key(belief.node)
    x = planner.scores([0, 1], scenarios)
    np.testing.assert_array_equal(planner.scores([1, 0], scenarios), x[::-1])
    assert node == belief.model.key(belief.node)
    assert before == [
        (
            s.kind,
            s.forecast_seed,
            belief.model.kernels[s.kind].encode(s.engine.world, 0),
        )
        for s in scenarios
    ]


def test_online_controller_native_replay_and_no_old_kernel_flag_flip():
    c = fixture()
    ctrl = EpisodeTypeMPCController(
        "A",
        "own",
        [
            {"B": FixedCoefficientController((-40,))},
            {"B": FixedCoefficientController((40,))},
        ],
        [0.7, 0.3],
        TypeMPCConfig(8, 8),
    )
    e = MarketEngine(c, {"A": ctrl, "B": FixedCoefficientController((40,))})
    e.reset(200512)
    e.step()
    snapshot = e.snapshot()
    e.step()
    result = carrier_metrics(e.world, c, 0)
    decisions = e.controllers[0].decisions
    e.restore(snapshot)
    e.step()
    assert carrier_metrics(e.world, c, 0) == result
    assert e.controllers[0].decisions == decisions
    assert len(decisions) == 2 and any(
        abs(x - 0.7) > 0.01 for x in decisions[1]["type_probabilities"]
    )
    assert ctrl.exact_compatible is False
    with pytest.raises(ValueError, match="joint type belief"):
        ExactKernel(c, {"A": ctrl})


def test_controller_uses_same_public_history_after_hidden_world_change():
    c = fixture()
    ctrl = EpisodeTypeMPCController(
        "A",
        "own",
        [
            {"B": FixedCoefficientController((-40,))},
            {"B": FixedCoefficientController((40,))},
        ],
        [0.7, 0.3],
        TypeMPCConfig(4, 4),
    )
    from airline_market.agents.base import ControllerContext

    context = ControllerContext(c, 0, c.spec.pricing[0], c.spec.objectives[0])
    obs = make(c).node.observation
    ctrl.reset(context, np.random.default_rng(12))
    first = ctrl.act(obs)
    ctrl.reset(context, np.random.default_rng(999))
    assert ctrl.act(obs) == first
    assert ctrl.belief.node.observation.privileged is None


def test_quoted_I3_native_score_uses_permitted_first_stage_prices():
    c = compile_market(load_spec("airline_market/spec/library/hub_feeder_small.json"))
    from airline_market.agents import make_controller

    types = [{"B": make_controller(c.spec.controllers[1])}]
    belief = JointTypeFilter(c, "A", "direct", types, [1])
    belief.reset(belief.model.roots()[0][1].observation)
    planner = EpisodeTypeMPC(belief, TypeMPCConfig(2, 2))
    scenarios = planner.scenarios(2)
    values = planner.scores([0], scenarios)[0]
    # Independent native coefficient execution has the same observation/pricing law.
    for k, scenario in enumerate(scenarios):
        e = scenario.engine.branch()
        while not e.terminal():
            e.step({0: CoefficientAction(c.spec.pricing[0].theta_grid[0])})
        assert values[k] == carrier_metrics(e.world, c, 0)["objective"]


def test_configuration_and_consecutive_history_guards():
    with pytest.raises(ValueError):
        TypeMPCConfig(n_scenarios=1)
    with pytest.raises(ValueError):
        TypeMPCConfig(switch_standard_errors=-1)
    belief = make(fixture())
    with pytest.raises(ValueError, match="consecutive"):
        belief.update(0, belief.node.observation)
    with pytest.raises(ValueError, match="tail"):
        EpisodeTypeMPC(belief, TypeMPCConfig(tail_index=2))


def test_history_type_memory_and_full_ledger_forecast_are_preserved():
    from airline_market.core.hashing import canonical_json
    from airline_market.game.type_belief import HistoryController, HistoryPolicy
    from airline_market.game.type_psro_keyed import EpisodeTypeDP

    c = fixture()
    initial = (
        EpisodeTypeDP(c, "B", "own", [{"A": FixedCoefficientController((-40,))}], [1])
        .roots()[0][1]
        .observation
    )
    policies = [
        HistoryPolicy(
            c.spec.pricing[1].theta_grid,
            {(canonical_json(initial),): 0},
            default=i,
            arm="own",
        )
        for i in [0, 1]
    ]
    belief = JointTypeFilter(
        c, "A", "own", [{"B": HistoryController(p)} for p in policies], [0.5, 0.5]
    )
    belief.reset(belief.model.roots()[0][1].observation)
    child = next(
        n
        for _, _, n in belief.model.successors(belief.node, 0)
        if n.observation.own_cash > 0
    )
    belief.update(0, child.observation)
    np.testing.assert_allclose(belief.node.type_probabilities(2), [0.5, 0.5])
    assert all(state.carriers[1].controller_memory for _, state in belief.node.mass)
    planner = EpisodeTypeMPC(belief, TypeMPCConfig(2, 2))
    scenarios = planner.scenarios(2)
    values = planner.scores([0], scenarios)[0]
    for k, scenario in enumerate(scenarios):
        e = scenario.engine.branch()
        assert e.controllers[1].memory()
        e.step({0: CoefficientAction(c.spec.pricing[0].theta_grid[0])})
        assert values[k] == carrier_metrics(e.world, c, 0)["objective"]


def test_poisson_family_is_not_silently_certified():
    c = compile_market(load_spec("config/instances/I5.json"))
    with pytest.raises(ValueError, match="Bernoulli"):
        JointTypeFilter(
            c, "A", "direct", [{"B": FixedCoefficientController((0, 0, 0))}], [1]
        )


def test_real_probe_and_frozen_inputs(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from types import SimpleNamespace

    from airline_market.campaign.artifacts import file_hash
    from airline_market.spec import dump_spec

    if os.environ.get("AIRLINE_TYPE_MPC_TEST_MODULE") == "type_mpc":
        monkeypatch.setitem(sys.modules, "airline_market.game.type_mpc", module)
        spec = importlib.util.spec_from_file_location(
            "staged_driver", __file__.replace("test_type_mpc.py", "type_mpc_study.py")
        )
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
    else:
        driver = importlib.import_module("airline_market.experiments.type_mpc_study")
    source = tmp_path / "frozen.json"
    dump_spec(fixture().spec, source)
    out = tmp_path / "out"
    out.mkdir()
    p = {
        "source_spec": str(source),
        "source_spec_sha256": file_hash(source),
        "focal": "A",
        "arm": "own",
        "rival_arm": "own",
        "n_cohorts": 1,
        "seed_start": 200512,
        "type_probabilities": [0.5, 0.5],
        "opponent_schedules": [[[-40], [-40]], [[-40], [40]]],
        "mpc": {"n_scenarios": 2, "n_reporting": 2},
    }
    result = driver.probe({"parameters": p}, SimpleNamespace(directory=out))
    assert result["decisions"] == 2 and result["certified"] is False
    assert result["PSRO_iterations"] is None
    assert (out / "cohorts.json").exists()
    with pytest.raises(ValueError, match="final seeds"):
        driver.probe(
            {"parameters": {**p, "seed_start": 300000}}, SimpleNamespace(directory=out)
        )
    with pytest.raises(ValueError, match="changed registered"):
        driver.probe(
            {"parameters": {**p, "source_spec_sha256": "wrong"}},
            SimpleNamespace(directory=out),
        )
