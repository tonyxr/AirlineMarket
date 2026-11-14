"""Distinct two-period fixtures; no registered production cases or RNG."""

import importlib.util
import json
import os
import sys
from fractions import Fraction
from pathlib import Path

import pytest

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.campaign.worker import CellContext

if os.environ.get("BOOKING_TIMING_STAGE"):
    name = "airline_market.experiments.booking_timing_study"
    spec = importlib.util.spec_from_file_location(
        name, Path(os.environ["BOOKING_TIMING_STAGE"]) / "booking_timing_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
from airline_market.experiments import booking_timing_study as study


def fixture(order):
    s = json.loads(
        Path("config/benchmarks/V4_fixed_fare_single_two_class.json").read_text()
    )
    s["case"] = "unit_C1_" + order
    s["physical_spec"]["name"] = "unit_distinct_T2_booking_profile"
    s["physical_spec"]["horizon"]["n_periods"] = 2
    s["physical_spec"]["legs"][0]["capacity"]["support"] = [1]
    for p in s["physical_spec"]["products"]:
        p["reference_fare"] = 100 if p["id"] == "HIGH" else 60
        p["service_cost"] = 0
        s["physical_spec"]["pricing"][0]["base"][p["id"]] = p["reference_fare"]
    s["class_probabilities"] = [[0.5, 0] if x == "H" else [0, 1] for x in order]
    s["timing_factors"] = {
        "capacity": 1,
        "high_request_probability": 0.5,
        "low_request_probability": 1.0,
        "order": order,
    }
    return s


def test_count_distribution_commutes_but_keeps_zero_masses():
    x = study.total_count_law([[0, 1], [0.5, 0]])
    assert x == study.total_count_law([[0.5, 0], [0, 1]])
    assert (
        x[(0, 1)] == x[(1, 1)] == Fraction(1, 2)
        and x[(2, 2)] == 0
        and sum(x.values()) == 1
    )
    assert x == study.independent_binomial_law([[0, 1], [0.5, 0]])


def test_native_two_period_timing_counterexample_with_same_full_count_law():
    low = study.evaluate(fixture("LH"), 1e-10, 20000)
    high = study.evaluate(fixture("HL"), 1e-10, 20000)
    assert low["joint_total_count_law"] == high["joint_total_count_law"]
    assert low["native_DP_USD"] == 60 and high["native_DP_USD"] == 80
    assert (
        low["offline_total_counts_bound_fraction_USD"]
        == high["offline_total_counts_bound_fraction_USD"]
        == "80"
    )
    assert (
        low["max_native_classical_error_USD"]
        == high["max_native_classical_error_USD"]
        == 0
    )
    for r in [low, high]:
        for node in r["native_nodes"]:
            for b in node["branches"]:
                obs = b["observation"]
                assert set(obs) == {
                    "period",
                    "own_inventory",
                    "requested_product",
                    "fixed_fare_USD",
                    "can_accept",
                }
                assert (
                    obs["period"] == node["t"] and obs["own_inventory"] == node["stock"]
                )


def test_offline_count_bound_uses_whole_joint_law_and_rejects_negative_margin():
    law = {(1, 1): Fraction(1)}
    assert study.offline_bound(law, 1, [Fraction(100), Fraction(60)]) == 100
    assert study.offline_bound(law, 2, [Fraction(100), Fraction(-10)]) == 100


@pytest.mark.parametrize(
    "probabilities", [[[0.5, 0.5]], [[0, 0]], [[0.2, 0], [0.3, 0], [0, 0.5]]]
)
def test_independent_binomial_reference_requires_declared_class_clock(probabilities):
    with pytest.raises(ValueError):
        study.independent_binomial_law(probabilities)


def test_actual_context_driver_retains_both_new_cases_and_hashes(tmp_path):
    sources = []
    for order in ["LH", "HL"]:
        p = tmp_path / (order + ".json")
        atomic_json(p, fixture(order))
        sources.append({"path": str(p), "sha256": file_hash(p)})
    reg = tmp_path / "registration.json"
    atomic_json(reg, {"unit_fixture": True})
    parameters = {
        "problems": sources,
        "design_registration": str(reg),
        "design_registration_sha256": file_hash(reg),
        "tolerance": 1e-10,
        "max_nodes": 20000,
    }
    out = tmp_path / "out"
    out.mkdir()
    r = study.run({"parameters": parameters}, CellContext(out))
    assert (
        r["problems"] == 2
        and r["corners"] == 1
        and r["corner_value_ranges_fraction_USD"] == ["20"]
    )
    assert (
        r["new_sampled_physical_cohorts"]
        == r["RNG_calls"]
        == r["old_problem_solver_or_DLP_reruns"]
        == 0
    )
    assert r["full_P3_or_V4_acceptance"] is None
    assert all(file_hash(out / n) == h for n, h in r["artifact_hashes"].items())
    assert all(file_hash(x["path"]) == x["sha256"] for x in sources)
