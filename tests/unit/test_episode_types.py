"""Joint hidden-type law, full-ledger replay and true response expansion checks."""

from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from airline_market.agents.fixed import FixedCoefficientController
from airline_market.core.hashing import canonical_json
from airline_market.core.types import CoefficientAction
from airline_market.game.type_belief import (
    EpisodeTypeDP,
    FullLedgerKernel,
    HistoryController,
    HistoryPolicy,
)
from airline_market.game.type_psro import exact_type_psro
from airline_market.sim.engine import MarketEngine
from airline_market.sim.kernel import ExactKernel
from airline_market.solvers.brute_history import binary_sale_micro
from airline_market.solvers.exact_dp import ExactBeliefDP
from airline_market.spec import compile_market


def two_sided(T=2):
    """Fixed-capacity symmetric action classes with a nontrivial binary sales history."""
    s = binary_sale_micro({"T": T, "rival_support": [2], "arrival": 0.9})
    a = s.pricing[0]
    return replace(
        s,
        name="two_sided_type_micro",
        timing=replace(s.timing, mode="simultaneous", leaders=()),
        pricing=(a, replace(a, carrier="B", base={"B_p": 100})),
        information=tuple(
            replace(s.information[0], name="own", observer=c) for c in ["A", "B"]
        ),
    )


def planner(c, probabilities=(0.3, 0.7), focal="A"):
    """Bind two deterministic rival types to an independent episode prior."""
    rival = "B" if focal == "A" else "A"
    return EpisodeTypeDP(
        c,
        focal,
        "own",
        [
            {rival: FixedCoefficientController((-40,))},
            {rival: FixedCoefficientController((40,))},
        ],
        probabilities,
    )


def forward(c, actions, weights):
    """Independently integrate complete binary-sale policies, without any DP calls."""
    total = 0.0
    for theta, weight in zip([(-40,), (40,)], weights):
        kernel = ExactKernel(c, {"B": FixedCoefficientController(theta)})
        arm = next(a for a in c.spec.information if a.observer == "A")
        frontier = []
        for p, state in kernel.initial_distribution(0):
            for q, obs in kernel.observation_distribution(state, 0, arm):
                frontier.append((p * q, state, obs, ()))
        while frontier[0][1].t < c.spec.horizon.n_periods:
            following = []
            for p, state, obs, prefix in frontier:
                index = (
                    2 ** len(prefix)
                    - 1
                    + sum(x * 2 ** (len(prefix) - j - 1) for j, x in enumerate(prefix))
                )
                action = CoefficientAction(c.spec.pricing[0].theta_grid[actions[index]])
                for q, after, signal in kernel.transition(
                    state, 0, action, arm=arm, observation=obs
                ):
                    following.append(
                        (
                            p * q,
                            after,
                            signal.observation,
                            (*prefix, int(bool(signal.purchases))),
                        )
                    )
            frontier = following
        assert sum(p for p, *_ in frontier) == pytest.approx(1)
        total += weight * sum(
            p * (state.carriers[0].cash - 10 * sum(state.carriers[0].counts)) / 1000
            for p, state, _, _ in frontier
        )
    return total


def test_joint_law_preserves_identical_type_labels_and_persistent_prior():
    c = compile_market(two_sided())
    dp = EpisodeTypeDP(
        c, "A", "own", [{"B": FixedCoefficientController((-40,))}] * 2, [0.3, 0.7]
    )
    _, root = dp.roots()[0]
    assert len(root.mass) == 2
    np.testing.assert_allclose(root.type_probabilities(2), [0.3, 0.7])
    for _, _, child in dp.successors(root, 0):
        np.testing.assert_allclose(child.type_probabilities(2), [0.3, 0.7])
        assert {k for k, _ in child.mass} == {0, 1}


def test_type_posterior_learns_from_own_sales_without_type_observation():
    dp = planner(compile_market(two_sided()))
    root = dp.roots()[0][1]
    posteriors = [child.type_probabilities(2) for _, _, child in dp.successors(root, 0)]
    assert any(np.max(np.abs(p - [0.3, 0.7])) > 0.01 for p in posteriors)
    assert root.observation.privileged is None
    assert all(
        state.t == 1
        for _, _, child in dp.successors(root, 0)
        for _, state in child.mass
    )


@pytest.mark.parametrize("focal", ["A", "B"])
def test_dirac_value_matches_unmodified_reference_and_export_replays(focal):
    c = compile_market(two_sided())
    rival = "B" if focal == "A" else "A"
    ctrl = {rival: FixedCoefficientController((40,))}
    dp = EpisodeTypeDP(c, focal, "own", [ctrl], [1])
    reference = ExactBeliefDP(c, focal, "own", ctrl, backend="reference")
    assert dp.solve() == pytest.approx(reference.solve().value, abs=1e-12)
    policy = dp.policy()
    assert dp.evaluate(policy) == pytest.approx(dp.solve(), abs=1e-12)


def test_mixture_DP_matches_exhaustive_eight_policies_and_common_policy_weighting():
    c = compile_market(two_sided())
    dp = planner(c)
    brute = max(
        forward(c, actions, [0.3, 0.7]) for actions in product([0, 1], repeat=3)
    )
    assert dp.solve() == pytest.approx(brute, abs=1e-12)
    policy = HistoryPolicy(c.spec.pricing[0].theta_grid, {}, default=1, arm="own")
    component = [
        EpisodeTypeDP(c, "A", "own", [t], [1]).evaluate(policy)
        for t in [
            {"B": FixedCoefficientController((-40,))},
            {"B": FixedCoefficientController((40,))},
        ]
    ]
    assert dp.evaluate(policy) == pytest.approx(
        0.3 * component[0] + 0.7 * component[1], abs=1e-12
    )


def test_full_ledger_roundtrip_preserves_rival_cash_fares_dates_and_policy_memory():
    c = compile_market(two_sided(T=3))
    p = HistoryPolicy(c.spec.pricing[1].theta_grid, {}, default=1, arm="own")
    engine = MarketEngine(
        c, {"A": FixedCoefficientController((-40,)), "B": HistoryController(p)}
    )
    engine.reset(200001)
    engine.step()
    engine.step()
    kernel = FullLedgerKernel(
        c, {"A": FixedCoefficientController((-40,)), "B": HistoryController(p)}
    )
    state = kernel.encode(engine.world, 0)
    restored = kernel.engine(state, 0)
    assert kernel.encode(restored.world, 0) == state
    for i in range(2):
        arm = next(
            a for a in c.spec.information if a.observer == c.spec.carriers[i].code
        )
        assert canonical_json(restored.observe(i, arm)) == canonical_json(
            engine.observe(i, arm)
        )
    state = kernel.encode(restored.world, 0)
    before = restored.snapshot()
    restored.step()
    restored.restore(before)
    assert kernel.encode(restored.world, 0) == state
    with pytest.raises(ValueError, match="joint type belief"):
        ExactKernel(c, {"B": HistoryController(p)})


def test_actual_two_sided_PSRO_has_DP_deviation_checks_and_finite_history_certificate():
    c = compile_market(two_sided())
    populations = {
        code: [HistoryPolicy(c.spec.pricing[i].theta_grid, {}, default=i, arm="own")]
        for i, code in enumerate(["A", "B"])
    }
    result = exact_type_psro(c, populations, max_iters=8, tol=1e-9)
    assert result["status"] == "converged"
    assert result["certified"]
    assert result["max_deviation_gain"] <= 1e-9
    assert all(
        row["best_response_nodes"][0] > 0 and row["best_response_nodes"][1] > 0
        for row in result["history"]
    )
    assert len(result["history"]) > 1


def test_invalid_prior_and_node_limits_remain_explicit():
    c = compile_market(two_sided())
    with pytest.raises(ValueError, match="probability vector"):
        planner(c, [0.2, 0.3])
    dp = planner(c)
    dp.max_nodes = 1
    with pytest.raises(Exception, match="max_nodes"):
        dp.solve()


def test_policy_serialization_and_observation_only_replay():
    c = compile_market(two_sided())
    policy = planner(c).policy()
    loaded = HistoryPolicy.from_dict(policy.to_dict())
    assert loaded.hash() == policy.hash()
    rng = np.random.default_rng(773)
    controllers = [HistoryController(loaded), HistoryController(loaded)]
    from airline_market.agents.base import ControllerContext

    ctx = ControllerContext(c, 0, c.spec.pricing[0], c.spec.objectives[0])
    obs = planner(c).roots()[0][1].observation
    for ctrl in controllers:
        ctrl.reset(ctx, rng)
    assert controllers[0].act(obs) == controllers[1].act(obs)
    # Controller state consists only of observations and own action indexes.
    assert controllers[0].snapshot() == controllers[1].snapshot()


def test_study_runs_actual_PSRO_and_writes_replayable_policies(tmp_path):
    from types import SimpleNamespace

    from airline_market.experiments.type_game_study import run

    result = run(
        {"parameters": {"T": 2, "symmetric": True}, "limits": {"max_nodes": 10000}},
        SimpleNamespace(directory=tmp_path),
    )
    assert result["certified"]
    assert result["is_original_I1"] is False
    assert (tmp_path / "events.json").exists()
    assert (tmp_path / "policies.json").exists()
    assert result["history"][-1]["payoffs"][0] == pytest.approx(
        result["history"][-1]["payoffs"][1], abs=1e-12
    )


def test_leader_response_replaces_default_rule_and_both_actions_affect_native_payoff():
    c = compile_market(
        replace(
            two_sided(),
            timing=replace(two_sided().timing, mode="leader_follower", leaders=("B",)),
        )
    )
    # The base spec binds a B scarcity rule. The new focal B control must replace it.
    rival = {"A": FixedCoefficientController((-40,))}
    dp = EpisodeTypeDP(c, "B", "own", [rival], [1])
    values = [
        dp.evaluate(HistoryPolicy(dp.grid, {}, default=i, arm="own")) for i in [0, 1]
    ]
    assert abs(values[0] - values[1]) > 0.001
    assert all(
        type(k.controllers_spec["B"]).__name__ == "ExternalController"
        for k in dp.kernels
    )


def test_full_ledger_transition_contains_native_successor_with_history_rival():
    c = compile_market(two_sided(T=3))
    p = HistoryPolicy(c.spec.pricing[1].theta_grid, {}, default=1, arm="own")
    controls = {"A": FixedCoefficientController((-40,)), "B": HistoryController(p)}
    engine = MarketEngine(c, controls)
    engine.reset(200003)
    engine.step()
    arm = next(a for a in c.spec.information if a.observer == "A")
    obs = engine.observe("A", arm)
    kernel = FullLedgerKernel(c, controls)
    before = kernel.encode(engine.world, 0)
    native = engine.branch()
    native.step({"A": CoefficientAction((-40,))})
    expected = kernel.encode(native.world, 0)
    rows = kernel.transition(
        before, 0, CoefficientAction((-40,)), arm=arm, observation=obs
    )
    assert sum(q for q, *_ in rows) == pytest.approx(1)
    assert any(state == expected and probability > 0 for probability, state, _ in rows)


def test_random_rival_shopping_channels_are_rejected_without_joint_noise_enumeration():
    from airline_market.campaign.instances import build_instance

    s = build_instance("toy_hub_reduced", {"T": 2, "actions": 2})
    request = replace(s.quotes.requests[0], fare_noise_sd=1.0, fare_bin=1.0)
    c = compile_market(
        replace(s, quotes=replace(s.quotes, requests=(request, *s.quotes.requests[1:])))
    )
    with pytest.raises(ValueError, match="deterministic shopping"):
        FullLedgerKernel(c, {"B": FixedCoefficientController(())})
