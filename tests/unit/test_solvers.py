import numpy as np
import pytest
from airline_market.spec import compile_market
from airline_market.spec.library_builders import cx_example, toy_hub
from airline_market.solvers.exact_dp import (
    ExactBeliefDP,
    DPNode,
    minimax_regret_marginal_rule,
    argmax_first,
)
from airline_market.solvers.oracle_dp import OracleDP
from airline_market.solvers.fixed_search import search_fixed_grid
from airline_market.belief import factorize, ExactBelief
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.solvers.mpc import ScenarioMPC, MPCConfig
from airline_market.core.errors import KernelLimitError
from airline_market.core.rng import StreamRegistry


def test_cx_values_compression_loss_and_minimax():
    for label, expected, loss in [
        ("C", [150.0, 100.0], 0.0),
        ("X", [150.0, 200.0], 50.0),
    ]:
        c = compile_market(cx_example(label))
        dp = ExactBeliefDP(c, "F", "empty", node_key="belief")
        _, root = dp.roots()[0]
        q = dp.q_values(root)
        assert q == pytest.approx(expected, abs=1e-9)
        fq = dp.q_values(DPNode(root.observation, factorize(root.belief)))
        assert fq == pytest.approx([150.0, 150.0], abs=1e-9)
        assert q.max() - q[argmax_first(fq)] == pytest.approx(loss)
        assert max(
            q.max() - q[k] for k in np.flatnonzero(fq >= fq.max() - 1e-9)
        ) == pytest.approx(50.0)
        assert dp.evaluate_policy(dp.policy()) == pytest.approx(dp.solve().value)
        assert (
            OracleDP(c, "F", node_key="belief").solve().value >= dp.solve().value - 1e-9
        )
    assert minimax_regret_marginal_rule([150, 100], [150, 200]) == pytest.approx(
        {"regret": 25.0, "mixing_probability": 0.5}
    )
    with pytest.raises(KernelLimitError):
        ExactBeliefDP(c, "F", "empty", max_nodes=1).solve()


def test_fixed_identity_and_mpc_determinism():
    from airline_market.legacy_toy.model import Model, Config
    from airline_market.legacy_toy.experiment import fit_fixed
    from airline_market.sim.tape import LegacyToyTapeAdapter
    from airline_market.legacy_toy.model import NoiseTape

    c = compile_market(toy_hub())
    factory = lambda: FocalCarrierEnv(c, "A", "direct")

    class LegacyEnv(FocalCarrierEnv):
        def reset(self, seed):
            self.engine.reset(
                tape=LegacyToyTapeAdapter.from_legacy(NoiseTape.from_seed(seed, 10), c)
            )
            return self.engine.observe("A", "direct"), {}

    result = search_fixed_grid(
        lambda: LegacyEnv(c, "A", "direct"), c.spec.pricing[0].theta_grid, [11, 12]
    )
    expected, table = fit_fixed(Model(Config()), [11, 12])
    assert result.best_theta == expected
    assert [r["training_objective"] for r in result.table] == pytest.approx(
        [r["training_objective"] for r in table], abs=1e-12
    )
    c = compile_market(cx_example("X"))
    env = FocalCarrierEnv(c, "F", "empty")
    obs, _ = env.reset(3)
    b = ExactBelief(c, "F", "empty")
    b.reset(obs)
    config = MPCConfig(n_scenarios=20, selection="split")
    p = ScenarioMPC(
        c,
        "F",
        "empty",
        b,
        config=config,
        plan_streams=StreamRegistry(99, "plan"),
        tail_coefficients=(0.0,),
    )
    a = p.choose(obs, 0)
    assert a == p.choose(obs, 0)
