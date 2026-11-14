import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st
import numpy as np
from airline_market.spec import compile_market
from airline_market.spec.library_builders import toy_hub
from airline_market.legacy_toy.model import Config
from airline_market.sim.engine import MarketEngine
from airline_market.sim.kernel import ExactKernel
from airline_market.sim.ledger import counts, inventory
from airline_market.belief import ExactBelief
from airline_market.core.types import CoefficientAction


@settings(max_examples=15, deadline=None, derandomize=True)
@given(
    T=st.integers(2, 4),
    own=st.integers(1, 3),
    rival=st.integers(1, 3),
    seed=st.integers(0, 100),
    action=st.integers(0, 17),
)
def test_random_small_conservation_kernel_and_filter_support(
    T, own, rival, seed, action
):
    c = compile_market(
        toy_hub(
            Config(
                horizon=T,
                shock_step=1,
                own_capacity=(own, own),
                rival_oh_capacity=rival,
                rival_hd_capacities=tuple(sorted({1, rival})),
                capacity_shock_probability=0.3,
            )
        )
    )
    e = MarketEngine(c)
    e.reset(seed)
    obs = e.observe("A", "linked")
    b = ExactBelief(c, "A", "linked")
    b.reset(obs)
    kernel = ExactKernel(c)
    theta = CoefficientAction(c.spec.pricing[0].theta_grid[action])
    for _ in range(T):
        ks = kernel.encode(e.world)
        prob = sum(
            p for p, _, _ in kernel.transition(ks, 0, theta, arm=b.arm, observation=obs)
        )
        assert abs(prob - 1) < 1e-12
        e.step({"A": theta})
        after = e.observe("A", "linked")
        b.update(obs, theta, after)
        obs = after
        for i, cs in enumerate(e.world.carriers):
            assert np.array_equal(
                inventory(cs, c, i) + c.A[i] @ counts(cs, c, i), cs.capacity
            )
        cs = e.world.carriers[1]
        assert any(
            h.rivals[0].capacity == tuple(cs.capacity)
            and h.rivals[0].counts == tuple(counts(cs, c, 1))
            for h in b.mass
        )
