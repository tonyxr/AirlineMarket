"""Quoted-history posterior-key regression and I1 two-sided native information."""

from dataclasses import replace

import pytest

from airline_market.game.type_belief import HistoryController, HistoryPolicy, TypeNode
from airline_market.game.type_psro_keyed import EpisodeTypeDP
from airline_market.spec import compile_market, load_spec
from airline_market.spec.schedule import PeriodSchedule


def quoted_I1():
    """Retain I1's actual leader/follower quote laws but only one active arrival period."""
    s = load_spec("config/instances/I1_two_sided_T4A4_v1.json")
    return compile_market(
        replace(
            s,
            demand=replace(
                s.demand, arrival_probability=PeriodSchedule((0, 1), (0.7, 0.0))
            ),
        )
    )


@pytest.mark.parametrize("focal", ["A", "B"])
def test_actual_I1_quoted_histories_evaluate_and_cache_without_object_ordering(focal):
    c = quoted_I1()
    rival = "B" if focal == "A" else "A"
    i, j = c.carrier_index[focal], c.carrier_index[rival]
    arms = {
        a.observer: a.name
        for a in c.spec.information
        if a.name in ["direct", "B_direct"]
    }
    own = HistoryPolicy(c.spec.pricing[i].theta_grid, {}, default=0, arm=arms[focal])
    other = HistoryPolicy(c.spec.pricing[j].theta_grid, {}, default=3, arm=arms[rival])
    dp = EpisodeTypeDP(c, focal, own.arm, [{rival: HistoryController(other)}], [1])
    value = dp.evaluate(own)
    assert value > 0
    frontier = [n for _, n in dp.roots()]
    visited_nontrivial = False
    for _ in range(c.spec.horizon.n_periods):
        following = []
        for n in frontier:
            if len(n.mass) > 1 and any(state.history for _, state in n.mass):
                visited_nontrivial = True
                permuted = TypeNode(
                    n.observation, dict(reversed(list(n.mass.items()))), n.prefix
                )
                assert dp.key(n) == dp.key(permuted)
            following.extend(child for p, _, child in dp.successors(n, 0) if p)
        frontier = following
    assert visited_nontrivial


def test_keyed_driver_executes_and_seals_the_same_symmetric_micro_class(tmp_path):
    from types import SimpleNamespace

    from airline_market.experiments.type_game_study_keyed import run

    result = run(
        {"parameters": {"T": 2, "symmetric": True}, "limits": {"max_nodes": 10000}},
        SimpleNamespace(directory=tmp_path),
    )
    assert result["status"] == "converged"
    assert result["certified"]
    assert result["max_deviation_gain"] <= 1e-9
    assert result["history"][-1]["payoffs"][0] == pytest.approx(
        result["history"][-1]["payoffs"][1], abs=1e-12
    )
