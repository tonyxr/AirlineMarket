from dataclasses import replace
import pytest
from airline_market.spec import compile_market
from airline_market.spec.library_builders import toy_hub
from airline_market.legacy_toy.model import Config
from airline_market.solvers.exact_dp import ExactBeliefDP


def test_nested_information_values_micro():
    s = toy_hub(
        Config(
            horizon=2,
            shock_step=1,
            own_capacity=(1, 1),
            rival_oh_capacity=1,
            rival_hd_capacities=(1, 2),
        )
    )
    s = replace(
        s,
        pricing=(
            replace(
                s.pricing[0],
                theta_grid=(s.pricing[0].theta_grid[0], s.pricing[0].theta_grid[-1]),
            ),
            s.pricing[1],
        ),
    )
    values = {
        a: ExactBeliefDP(compile_market(s), "A", a, node_key="belief", max_nodes=10000)
        .solve()
        .value
        for a in ["direct", "linked", "unrelated", "oracle"]
    }
    assert values["linked"] >= values["direct"] - 1e-12
    assert values["unrelated"] == pytest.approx(values["direct"], abs=1e-12)
    assert values["oracle"] >= values["linked"] - 1e-12
