"""Exact conditional-tail Q vectors, root values, disabled cases and node limits."""

from dataclasses import replace
import numpy as np
import pytest

from airline_market.campaign.instances import build_instance
from airline_market.core.errors import KernelLimitError
from airline_market.core.rng import StreamRegistry
from airline_market.solvers.conditional_tail import ConditionalTailSolver
from airline_market.solvers.exact_dp import ExactBeliefDP
from airline_market.solvers.oracle_dp import OracleDP
from airline_market.spec import compile_market


def solver(compiled, arm, optimized=False, max_nodes=20000):
    result = (
        OracleDP(compiled, "A", backend="compact", max_nodes=max_nodes)
        if arm == "full_state"
        else ExactBeliefDP(compiled, "A", arm, backend="compact", max_nodes=max_nodes)
    )
    if optimized:
        old = result.finite
        result.finite = ConditionalTailSolver(old.model, old.grid, max_nodes)
    return result


@pytest.mark.parametrize("generator", ["toy_hub", "hub_feeder_small"])
@pytest.mark.parametrize(
    "arm", ["direct", "linked", "unrelated", "oracle", "full_state"]
)
def test_conditional_tail_root_values_match_original_compact(generator, arm):
    compiled = compile_market(build_instance(generator, {"T": 4, "actions": 4}))
    original = solver(compiled, arm)
    optimized = solver(compiled, arm, True)
    assert optimized.finite.tail_enabled
    assert optimized.solve().value == pytest.approx(
        original.solve().value, abs=1e-12, rel=0
    )
    assert optimized.finite.tail_vector_evaluations > 0


@pytest.mark.parametrize("generator", ["toy_hub", "hub_feeder_small"])
def test_every_q_at_random_reachable_and_reweighted_penultimate_beliefs(generator):
    compiled = compile_market(build_instance(generator, {"T": 4, "actions": 4}))
    for arm in ["direct", "linked", "unrelated"]:
        old = solver(compiled, arm)
        new = solver(compiled, arm, True)
        rng = StreamRegistry(200000, "validation/conditional-tail").generator()
        roots = list(old.roots())
        for k in range(16):
            node = old.finite.model.project(
                roots[k % len(roots)][1].observation, roots[k % len(roots)][1].belief
            )
            while node[0] < compiled.spec.horizon.n_periods - 2:
                children = list(old.finite.successors(node, int(rng.integers(4))))
                probabilities = np.array([child[0] for child in children])
                chosen = int(
                    rng.choice(len(children), p=probabilities / probabilities.sum())
                )
                node = children[chosen][2]
            np.testing.assert_allclose(
                new.finite.q_values(node), old.finite.q_values(node), atol=1e-12, rtol=0
            )
            if len(node[-1]) > 1:
                mass = rng.dirichlet(np.ones(len(node[-1])))
                weighted = (
                    *node[:-1],
                    tuple((h, float(w)) for (h, _), w in zip(node[-1], mass)),
                )
                np.testing.assert_allclose(
                    new.finite.q_values(weighted),
                    old.finite.q_values(weighted),
                    atol=1e-12,
                    rtol=0,
                )


def test_share_objective_disables_certificate_and_keeps_original_recursion():
    spec = build_instance("toy_hub", {"T": 3, "actions": 2})
    spec = replace(
        spec,
        objectives=(
            replace(spec.objectives[0], share_weight=0.3),
            *spec.objectives[1:],
        ),
    )
    compiled = compile_market(spec)
    old = solver(compiled, "direct")
    new = solver(compiled, "direct", True)
    assert not new.finite.tail_enabled
    assert new.solve().value == pytest.approx(old.solve().value, abs=1e-12, rel=0)
    assert new.finite.tail_vector_evaluations == 0


def test_tail_vectors_and_live_work_obey_the_declared_node_budget():
    compiled = compile_market(build_instance("toy_hub", {"T": 2, "actions": 2}))
    result = solver(compiled, "direct", True, max_nodes=1)
    with pytest.raises(KernelLimitError):
        result.solve()
    assert result.finite.memo_entries() <= 1
