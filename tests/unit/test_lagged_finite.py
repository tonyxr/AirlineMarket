"""Lagged-state values and transitions against the unchanged general exact kernel."""

from collections import defaultdict
from dataclasses import replace
import numpy as np
import pytest

from airline_market.campaign.instances import build_instance
from airline_market.core.types import CoefficientAction
from airline_market.sim.finite import UnsupportedFiniteModel
from airline_market.solvers.exact_dp import ExactBeliefDP
from airline_market.solvers.oracle_dp import OracleDP
from airline_market.solvers.lagged_dp import history_dp
from airline_market.spec import compile_market
from airline_market.spec import dump_spec
from airline_market.campaign.artifacts import file_hash
from types import SimpleNamespace
from airline_market.spec.model import ShoppingRequestSpec, InformationArmSpec
from airline_market.core.hashing import jsonable


def compiled(lag, **parameters):
    """Keep the full original observation process and small positive-capacity priors."""
    return compile_market(build_instance("hub_feeder_family", {
        "n_spokes": 1, "T": 2, "actions": 2, "lag_periods": lag,
        "capacity_support": [1, 2], **parameters}))


@pytest.mark.parametrize("arm", ["direct", "linked", "unrelated", "oracle", "full_state"])
@pytest.mark.parametrize("lag", [0, 1])
def test_root_values_match_general_reference_for_every_arm(lag, arm):
    c = compiled(lag)
    reference = (OracleDP(c, "A", backend="reference", max_nodes=20000)
                 if arm == "full_state" else ExactBeliefDP(c, "A", arm, backend="reference", max_nodes=20000))
    new = history_dp(c, "A", arm, max_nodes=20000)
    assert new.solve().value == pytest.approx(reference.solve().value, abs=1e-11, rel=0)
    assert new.compiled.spec_hash == reference.compiled.spec_hash
    if lag:
        assert not new.finite.model._stock_only_last_step


def compact_transition(model, node, index, grid):
    """Aggregate only future-relevant state and observable own sale/reward fields."""
    t, capacity, own, journeys, previous, signal, mass = node
    prices = model.prices(grid[index], t, capacity, own, signal)
    result = defaultdict(lambda: np.zeros(2))
    for hidden, weight in mass:
        for p, stock, count, h, reward, sale in model.step(t, own, journeys, hidden, prices, previous):
            key = stock, count, h, model.signal(t + 1, h, prices), sale
            result[key] += weight * p * np.array([1, reward])
    return dict(result)


def reference_transition(dp, model, node, index):
    """Project general-kernel transitions without reading a sampled evaluation truth."""
    result = defaultdict(lambda: np.zeros(2))
    for hidden, weight in node.belief.mass.items():
        for p, state, outcome in dp.kernel.transition(
                hidden.kernel, dp.focal, CoefficientAction(dp.grid[index]),
                arm=dp.arm, observation=node.observation):
            obs = outcome.observation
            sale = model._own_local[outcome.purchases[0][0]] if outcome.purchases else -1
            cost = sum(model.compiled.product_cost[j] for j, _ in outcome.purchases)
            reward = (outcome.cash_delta - cost) / model.objective.profit_scale
            key = (obs.own_inventory, sum(model.spec.products[b.product].market in model.markets
                                         for b in obs.own_bookings) if model.track_journeys else 0,
                   model.hidden(state), model.observed_signal(obs), sale)
            result[key] += weight * p * np.array([1, reward])
    return dict(result)


@pytest.mark.parametrize("channel", ["flag", "seats_capped"])
def test_lagged_transition_law_matches_general_at_initial_and_reachable_nodes(channel):
    c = compiled(1, availability=channel, fare_bin=10)
    reference = ExactBeliefDP(c, "A", "linked", backend="reference", max_nodes=20000)
    new = history_dp(c, "A", "linked", max_nodes=20000)
    model = new.finite.model
    roots = reference.roots()
    nodes = [roots[0][1]]
    nodes += [child for _, _, child in reference.successors(nodes[0], 0)][:3]
    for node in nodes:
        for index in range(len(reference.grid)):
            expected = reference_transition(reference, model, node, index)
            actual = compact_transition(model, model.project(node.observation, node.belief),
                                        index, reference.grid)
            assert actual.keys() == expected.keys()
            for key in actual:
                np.testing.assert_allclose(actual[key], expected[key], atol=1e-11, rtol=0)


def test_equal_current_inventory_can_have_distinct_reachable_lagged_offers():
    c = compiled(1)
    dp = history_dp(c, "A", "linked", max_nodes=20000)
    m = dp.finite.model
    states = defaultdict(list)
    for _, root in dp.roots():
        for hidden in root.belief.mass:
            for p, state, signal in dp.kernel.transition(hidden.kernel, dp.focal,
                                                        CoefficientAction(dp.grid[0]),
                                                        arm=dp.arm, observation=root.observation):
                h = m.hidden(state)
                states[h[:2]].append((h[2], m.observed_signal(signal.observation)))
    assert any(len({memory for memory, signal in values}) > 1
               and len({signal for memory, signal in values}) > 1 for values in states.values())


def test_lagged_state_tracks_rival_reaction_to_previous_own_prices():
    c = compiled(1)
    assignments = list(c.spec.controllers)
    rival = next(k for k, a in enumerate(assignments) if a.carrier == "B")
    params = jsonable(assignments[rival].params)
    params["rules"]["B_OD0_0"]["responses"] = [
        {"product": "A_OD0_0", "coefficient": 0.25, "reference": 185}]
    params["arm"] = "rival_response"
    assignments[rival] = replace(assignments[rival], params=params)
    spec = replace(c.spec, controllers=tuple(assignments),
                   quotes=replace(c.spec.quotes, requests=c.spec.quotes.requests + (
                       ShoppingRequestSpec(id="q_rival_response", observer="B",
                                           target_products=("A_OD0_0",), lag_periods=1),)),
                   information=c.spec.information + (
                       InformationArmSpec(name="rival_response", observer="B",
                                          requests=("q_rival_response",)),))
    c = compile_market(spec)
    dp = history_dp(c, "A", "direct", max_nodes=20000)
    assert not dp.finite.model.previous_encoded_by_signal
    root = dp.roots()[0][1]
    node = dp.finite.model.project(root.observation, root.belief)
    altered = (*node[:4], tuple(x + 20 for x in node[4]), *node[5:])
    assert dp.finite.key(node) != dp.finite.key(altered)
    assert np.max(np.abs(dp.finite.q_values(node) - dp.finite.q_values(altered))) > 1e-8
    reference = ExactBeliefDP(c, "A", "direct", backend="reference", max_nodes=20000)
    assert dp.solve().value == pytest.approx(reference.solve().value, abs=1e-11, rel=0)


def test_old_compact_refusal_is_unchanged_and_unsupported_channels_stay_explicit():
    c = compiled(1)
    with pytest.raises(UnsupportedFiniteModel, match="history"):
        ExactBeliefDP(c, "A", "direct", backend="compact")
    noisy = replace(c.spec, quotes=replace(c.spec.quotes, requests=tuple(
        replace(q, missing_probability=0.1) if q.observer == "A" else q
        for q in c.spec.quotes.requests)))
    with pytest.raises(UnsupportedFiniteModel):
        history_dp(compile_market(noisy), "A", "direct")


def test_history_study_loads_frozen_spec_without_rebuilding_metadata(tmp_path):
    from airline_market.experiments.history_study import solve

    c = compiled(1)
    source = tmp_path / "source.json"
    dump_spec(c.spec, source)
    out = tmp_path / "output"
    out.mkdir()
    cell = {"parameters": {"source_spec": str(source), "source_spec_sha256": file_hash(source),
                            "arm": "direct", "arbitrary_provenance": "must not enter generator metadata"},
            "limits": {"max_nodes": 20000}}
    result = solve(cell, SimpleNamespace(directory=out))
    assert result["status"] == "solved"
    assert result["spec_hash"] == c.spec_hash
    assert result["tail_certificate_enabled"] is False
    assert file_hash(out / "spec.json") == file_hash(source)
