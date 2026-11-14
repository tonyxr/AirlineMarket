"""S1 saved-policy metrics: analytical laws and new native micro fixtures."""

import gzip
import math
import pickle

import pytest

from airline_market.experiments.screen_diagnostics import exact_diagnostics
from airline_market.experiments.screen_saved_metrics import (
    analyze_settled_setting,
    first_price_divergence,
    pairwise_information,
    policy_information,
    read_policy,
)
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.solvers.finite_dp import FiniteBeliefSolver
from airline_market.solvers.lagged_dp import history_dp
from airline_market.spec import compile_market


def test_pairwise_mi_distinguishes_joint_dependence_and_marginalizes_memory():
    # Three pairwise independent bits with a parity constraint: TC=ln2, every MI=0.
    parity = tuple(((stock, 0), 0.25) for stock in [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)])
    assert pairwise_information(parity) == pytest.approx({(0, 1): 0, (0, 2): 0, (1, 2): 0}, abs=1e-15)
    correlated = ((((0, 0), 0, 'old'), 0.2), (((0, 0), 1, 'new'), 0.3), (((1, 1), 0, 'old'), 0.5))
    assert pairwise_information(correlated)[0, 1] == pytest.approx(math.log(2))


def test_information_rejects_mass_damage_without_repair():
    with pytest.raises(ValueError, match='mass'):
        pairwise_information(((((0, 0), 0), 0.9),))
    with pytest.raises(ValueError, match='weights'):
        pairwise_information(((((0, 0), 0), -0.1), (((1, 1), 0), 1.1)))


def test_policy_stream_rejects_truncation_and_duplicate_nodes(tmp_path):
    path = tmp_path / 'policy.gz'
    with gzip.open(path, 'wb') as stream:
        pickle.dump((('node',), 1), stream)
    assert read_policy(path, 1) == {('node',): 1}
    with pytest.raises(ValueError, match='incomplete'):
        read_policy(path, 2)
    with gzip.open(path, 'ab') as stream:
        pickle.dump((('node',), 1), stream)
    with pytest.raises(ValueError, match='duplicate'):
        read_policy(path, 2)


def fresh_pair(lag):
    c = compile_market(screen_spec({
        'T': 2, 'actions': 2, 'n_spokes': 1, 'capacity': 2,
        'capacity_support': [1, 2], 'connecting_share': 0.42,
        'arrival_probability': 0.73, 'price_scale': 1.17,
        'outside_utility_shift': 0.13, 'lag_periods': lag,
    }))
    result = []
    for arm in ('direct', 'linked'):
        dp = history_dp(c, 'A', arm)
        dp.solve()
        records = {}
        diagnostics = exact_diagnostics(dp, policy_sink=lambda n, i, target=records: target.__setitem__(n, i))
        result.append((dp, records, diagnostics))
    return result


@pytest.mark.parametrize('lag', [0, 1])
def test_native_saved_analysis_uses_no_bellman_optimization(monkeypatch, lag):
    (direct, dr, diag), (linked, lr, _) = fresh_pair(lag)

    def forbidden(*args, **kwargs):
        raise AssertionError('analysis called an optimizer')

    monkeypatch.setattr(FiniteBeliefSolver, 'q_values', forbidden)
    monkeypatch.setattr(direct, 'solve', forbidden)
    monkeypatch.setattr(linked, 'solve', forbidden)
    information = policy_information(direct, dr)
    assert information['pairs'] == [['B_OH0', 'B_HD0']]
    # With two rival legs the old TC and pairwise MI coincide, on decision frames only.
    for row, tc in zip(information['periods'][:-1], diag['posterior_total_correlation_by_period']):
        assert row['pairwise_mi_nats'][0] == pytest.approx(tc, abs=1e-12)
    assert all(r['reach_mass'] == pytest.approx(1, abs=1e-12) for r in information['periods'])
    outcome = first_price_divergence(direct, linked, dr, lr)
    assert outcome['total_mass'] == pytest.approx(1, abs=1e-12)
    assert outcome['bellman_or_sampling_calls'] == 0


def test_missing_saved_action_is_not_reoptimized():
    (dp, records, _), _ = fresh_pair(0)
    records.pop(next(iter(records)))
    with pytest.raises(KeyError):
        policy_information(dp, records)


@pytest.mark.parametrize('alias', [False, True])
def test_price_divergence_counts_physical_probability_and_keeps_aliases(monkeypatch, alias):
    (direct, dr, _), (linked, lr, _) = fresh_pair(0)
    # Distinct saved coefficient actions; posted prices are deliberately aliases in one case.
    dr = {n: 0 for n in dr}
    lr = {n: 1 for n in lr}
    for dp in (direct, linked):
        width = len(dp.finite.model.own_products)
        monkeypatch.setattr(dp.finite.model, 'prices', lambda theta, *args, width=width: (100.0 if alias else 100 + 10 * theta[0],) * width)
    if alias:
        # Enumerate this declared fixed policy's reachable nodes, no Bellman calls.
        from airline_market.experiments.screen_saved_metrics import projected_roots

        for dp, records, index in [(direct, dr, 0), (linked, lr, 1)]:
            records.clear()
            current = [node for _, node in projected_roots(dp)]
            for _ in range(dp.finite.model.T):
                following = []
                for node in current:
                    records[node] = index
                    following.extend(child for _, _, child in dp.finite.successors(node, index))
                current = following
    outcome = first_price_divergence(direct, linked, dr, lr)
    assert outcome['first_period'] == (None if alias else 0)
    assert outcome['no_price_difference_mass'] == pytest.approx(1 if alias else 0)
    assert outcome['coefficient_difference_first_mass_before_price_split'][0] == pytest.approx(1)


def test_coupling_rejects_unmatched_source():
    (direct, dr, _), _ = fresh_pair(0)
    _, (linked, lr, _) = fresh_pair(1)
    with pytest.raises(ValueError, match='same physical'):
        first_price_divergence(direct, linked, dr, lr)


@pytest.mark.parametrize('delayed', [False, True])
def test_first_difference_has_correct_time_and_partial_root_mass(monkeypatch, delayed):
    from airline_market.experiments.screen_saved_metrics import projected_roots

    (direct, dr, _), (linked, lr, _) = fresh_pair(0)
    roots = projected_roots(linked)
    selected = roots[0][1][5]
    expected = math.fsum(p for p, node in roots if node[5] == selected)
    assert 0 < expected < 1
    width = len(direct.finite.model.own_products)
    monkeypatch.setattr(direct.finite.model, 'prices', lambda *args: (100.0,) * width)

    def linked_prices(theta, t, capacity, own, signal):
        different = (t == 1) if delayed else (t == 0 and signal == selected)
        return (120.0 if different else 100.0,) * width

    monkeypatch.setattr(linked.finite.model, 'prices', linked_prices)
    for dp, records in [(direct, dr), (linked, lr)]:
        records.clear()
        current = [node for _, node in projected_roots(dp)]
        for _ in range(dp.finite.model.T):
            following = []
            for node in current:
                records[node] = 0
                following.extend(child for _, _, child in dp.finite.successors(node, 0))
            current = following
    result = first_price_divergence(direct, linked, dr, lr)
    assert result['first_period'] == int(delayed)
    assert result['first_price_difference_mass_by_period'] == pytest.approx([0, 1] if delayed else [expected, 0])
    assert result['no_price_difference_mass'] == pytest.approx(0 if delayed else 1 - expected)
    # Same coefficients, different fare map: do not conflate the two definitions.
    assert result['coefficient_difference_first_mass_before_price_split'] == [0, 0]


def test_saved_setting_reader_checks_acknowledgement_and_source(tmp_path):
    import json

    from airline_market.campaign.artifacts import atomic_json, file_hash
    from airline_market.campaign.worker import CellContext
    from airline_market.experiments.screen_study import run
    from airline_market.spec import dump_spec

    source = tmp_path / 'source.json'
    dump_spec(screen_spec({'T': 2, 'actions': 2, 'n_spokes': 1, 'capacity': 2,
                          'capacity_support': [1, 2], 'price_scale': 1.23,
                          'outside_utility_shift': 0.16, 'arrival_probability': 0.67}), source)
    cells = []
    for arm in ['direct', 'linked', 'unrelated', 'oracle', 'full_state']:
        directory = tmp_path / 'cells' / arm / 'attempt-0001'
        directory.mkdir(parents=True)
        cell = {'cell_id': arm, 'config_hash': arm, 'parameters': {'setting_id': 'fixture',
                'source_spec': str(source), 'source_spec_sha256': file_hash(source), 'arm': arm},
                'limits': {'max_nodes': 100000}}
        result = {'config_hash': arm, **run(cell, CellContext(directory))}
        atomic_json(directory / 'results.json', result)
        atomic_json(directory / 'COMPLETE.json', {'config_hash': arm, 'status': result['status'],
                    'results_sha256': file_hash(directory / 'results.json')})
        cells.append(cell)
    atomic_json(tmp_path / 'cells.json', cells)
    atomic_json(tmp_path / 'settled_cells.json', {'cell_ids': [c['cell_id'] for c in cells][:-1]})
    with pytest.raises(ValueError, match='acknowledged'):
        analyze_settled_setting(tmp_path, 'fixture')
    atomic_json(tmp_path / 'settled_cells.json', {'cell_ids': [c['cell_id'] for c in cells]})
    result = analyze_settled_setting(tmp_path, 'fixture')
    assert len(result['pairwise_information']) == 5
    assert result['policy_divergence']['total_mass'] == pytest.approx(1)
    assert json.loads((tmp_path / 'cells' / 'direct' / 'attempt-0001' / 'results.json').read_bytes())['status'] == 'solved'
    source.write_bytes(source.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='source changed'):
        analyze_settled_setting(tmp_path, 'fixture')
