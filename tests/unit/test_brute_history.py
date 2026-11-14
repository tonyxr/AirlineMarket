"""Independent analytic and exhaustive-policy checks, without Bellman reuse."""

from dataclasses import replace
from types import SimpleNamespace
import math

import numpy as np
import pytest

from airline_market.solvers.brute_history import binary_sale_micro, BruteHistoryEnumerator
from airline_market.solvers.exact_dp import ExactBeliefDP
from airline_market.experiments.verification_study import enumerate_micro, profile_empirical
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule


def test_forward_enumerator_matches_two_closed_form_single_period_fares(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("brute enumeration called a Bellman optimization")

    monkeypatch.setattr(ExactBeliefDP, "solve", forbidden)
    monkeypatch.setattr(ExactBeliefDP, "q_values", forbidden)
    spec = binary_sale_micro({"T": 2, "rival_support": [0], "arrival": 0.7})
    spec = replace(spec, demand=replace(spec.demand,
        arrival_probability=PeriodSchedule((0, 1), (0.7, 0.0))))
    c = compile_market(spec)
    result = BruteHistoryEnumerator(c).enumerate()
    expected = [0.7 * (fare - 10) / 1000 / (1 + math.exp(-(3.5 - 0.02 * fare)))
                for fare in [60, 140]]
    np.testing.assert_allclose(result["values"], [expected[0]] * 4 + [expected[1]] * 4,
                               atol=1e-12, rtol=0)


def test_all128_T3_histories_match_both_DPs_with_hidden_inventory_and_share(tmp_path):
    result = enumerate_micro({"parameters": {"T": 3, "own_capacity": 2,
        "rival_support": [0, 2], "arrival": 0.9, "share_weight": 0.2}},
        SimpleNamespace(directory=tmp_path))
    assert result["policies_enumerated"] == 128
    assert result["maximum_absolute_error"] <= 1e-10
    assert result["status"] == "completed"


def test_enumerator_rejects_an_unrepresented_initial_observation_branch():
    spec = binary_sale_micro({"T": 3})
    own = replace(spec.legs[0], capacity=replace(spec.legs[0].capacity,
        support=(1, 2), probabilities=(0.5, 0.5)))
    with pytest.raises(ValueError, match="initial observable"):
        BruteHistoryEnumerator(compile_market(replace(spec, legs=(own, spec.legs[1]))))


def test_profile_failure_retains_profiler_output_and_ineligible_timing_marker(tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError("declared test failure before a completed measurement")

    monkeypatch.setattr("airline_market.experiments.verification_study.measure", fail)
    with pytest.raises(ValueError, match="declared test failure"):
        profile_empirical({}, SimpleNamespace(directory=tmp_path))
    assert (tmp_path / "PROFILING_ONLY.json").exists()
    assert (tmp_path / "cohort_profile.pstats").exists()
