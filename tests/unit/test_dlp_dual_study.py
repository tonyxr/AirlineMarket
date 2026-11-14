"""Independent rational solver, nonunique duals and native contract verification."""

import importlib.util
import os
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

from airline_market.campaign.artifacts import file_hash

if os.environ.get("DLP_STAGE"):
    spec = importlib.util.spec_from_file_location(
        "airline_market.experiments.dlp_dual_study",
        Path(os.environ["DLP_STAGE"]) / "dlp_dual_study.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
from airline_market.experiments import dlp_dual_study as study


def fixture():
    return [[1, 1]], [3], [9, 4], [1, 4]


def test_reference_is_independent_of_native_optimizer(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("native solver called")

    monkeypatch.setattr(study, "solve_dlp", forbidden)
    ref = study.exact_reference(*fixture())
    assert ref["objective_USD"] == "17"
    assert ref["resource_dual_ranges"] == [["4", "4"]]
    assert ref["resource_duals_unique"]


def test_fractional_data_are_solved_exactly_without_rounding():
    a, _, r, d = fixture()
    ref = study.exact_reference(a, [1.5], r, d)
    assert ref["objective_USD"] == "11"
    assert ref["optimal_primal_vertices"] == [["1", "1/2"]]
    assert ref["exact_strong_duality"]


def test_nonunique_dual_retains_entire_face_and_null_vector_agreement():
    a, _, r, d = fixture()
    ref = study.exact_reference(a, [5], r, d)
    assert ref["resource_dual_ranges"] == [["0", "4"]]
    assert not ref["resource_duals_unique"]
    native = study.native_comparison(a, [5], r, d, ref, 1e-10)
    assert native["certificate_pass"] and native["case_pass"]
    assert native["unique_vector_agreement"] is None
    assert native["unique_dual_max_abs_error"] is None


def test_singular_bases_are_skipped_and_infeasible_vertices_rejected():
    f = Fraction
    assert study.solve_square([[f(1), f(1)], [f(2), f(2)]], [f(1), f(2)]) is None
    assert study.vertices([[f(1)], [f(-1)], [f(2)]], [f(3), f(0), f(6)], 1) == [
        (f(0),),
        (f(3),),
    ]


def test_native_unique_dual_and_objective_are_unrounded_and_certified():
    ref = study.exact_reference(*fixture())
    native = study.native_comparison(*fixture(), ref, 1e-10)
    assert native["native_raw_resource_duals_USD_per_seat"] == [4.0]
    assert native["unique_vector_agreement"] and native["certificate_pass"]
    assert native["native_objective_USD"] == 17.0


def test_wrong_native_duals_fail_without_repair(monkeypatch):
    ref = study.exact_reference(*fixture())
    monkeypatch.setattr(study, "solve_dlp", lambda *args: ([1, 2], [0], 17))
    native = study.native_comparison(*fixture(), ref, 1e-10)
    assert not native["case_pass"] and not native["unique_vector_agreement"]
    assert native["native_raw_resource_duals_USD_per_seat"] == [0.0]


@pytest.mark.parametrize("capacity", [[0], [float("nan")], [True]])
def test_invalid_inputs_are_rejected(capacity):
    a, _, r, d = fixture()
    with pytest.raises(ValueError):
        study.exact_reference(a, capacity, r, d)


def test_driver_status_hashes_and_no_simulation(tmp_path, monkeypatch):
    from airline_market.sim.engine import MarketEngine
    from airline_market.sim.tape import ExogenousTape

    def forbidden(*args, **kwargs):
        raise AssertionError("engine or tape called")

    monkeypatch.setattr(MarketEngine, "reset", forbidden)
    monkeypatch.setattr(ExogenousTape, "generate", forbidden)
    reg = tmp_path / "registration.json"
    reg.write_text("{}")
    native = "airline_market/agents/bidprice.py"
    a, c, r, d = fixture()
    p = {
        "A": a,
        "capacity": c,
        "revenues": r,
        "demand": d,
        "name": "distinct_unit_fixture",
        "native_source": native,
        "native_source_sha256": file_hash(native),
        "design_registration": str(reg),
        "design_registration_sha256": file_hash(reg),
        "tolerance": 1e-10,
    }
    result = study.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
    assert result["status"] == "completed" and result["DLP_primal_dual_case_pass"]
    assert result["simulation_cohorts"] == result["optimizer_updates"] == 0
    assert result["native_fixed_fare_DP_bound_pass"] is None
    for name, h in result["artifact_hashes"].items():
        assert file_hash(tmp_path / name) == h
    p["native_source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen DLP input"):
        study.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
