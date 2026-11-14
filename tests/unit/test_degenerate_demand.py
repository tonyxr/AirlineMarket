"""Distinct T3/price101 fixtures for native degenerate-demand verification."""

import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.spec import compile_market, load_spec

if os.environ.get("MR6_STAGE"):
    spec = importlib.util.spec_from_file_location(
        "airline_market.experiments.degenerate_demand_study",
        Path(os.environ["MR6_STAGE"]) / "degenerate_demand_study.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
from airline_market.experiments import degenerate_demand_study as study


def fixture_model(delta=0):
    d = json.loads(Path("config/benchmarks/V3_MR6_conversion_delta0.json").read_text())
    d["horizon"]["n_periods"] = 3
    d["legs"][0]["capacity"]["support"] = [3]
    d["products"][0]["reference_fare"] = 101.0
    d["pricing"][0]["base"]["A_OD"] = 101.0
    d["segments"][0]["classes"][0]["market_value"]["OD"] = 3.025 + delta
    return d


def test_native_limit_keeps_nonzero_outside_tail_and_exact_expectation():
    a = study.conversion_reference(compile_market(load_spec(fixture_model())))
    b = study.conversion_reference(compile_market(load_spec(fixture_model(20))))
    assert a["outside_probability"] > b["outside_probability"] > 0
    assert b["outside_probability"] <= b["outside_probability_upper_bound"] < 2e-9
    assert b["native_DP_USD"] > a["native_DP_USD"]
    assert a["max_requests"] == a["capacity"] == 3


def test_relative_outside_normalization():
    for delta in [0, 3, 20]:
        r = study.conversion_reference(compile_market(load_spec(fixture_model(delta))))
        # Original pair (inside .5, outside -delta), normalized pair (.5+delta,0).
        independently = math.exp(-delta - 0.5) / (1 + math.exp(-delta - 0.5))
        assert math.isclose(r["outside_probability"], independently, rel_tol=1e-12)


def test_binding_capacity_is_rejected():
    d = fixture_model()
    d["legs"][0]["capacity"]["support"] = [2]
    with pytest.raises(ValueError, match="capacity must"):
        study.conversion_reference(compile_market(load_spec(d)))


def cell_fixture(tmp_path):
    original = fixture_model()
    original["background"] = [
        {
            "id": "new_bg",
            "carrier": "A",
            "capacity": {"support": [2], "probabilities": [1.0], "public": True},
            "quote_rule": {"base": 80.0, "scarcity_premium": 0.0, "threshold": 0},
            "purchase": {"max_prob": 0.5, "intercept": 2.0, "slope": 0.025},
        }
    ]
    source = tmp_path / "original.json"
    atomic_json(source, original)
    zero = []
    for all_zero in [False, True]:
        d = json.loads(json.dumps(original))
        d["demand"]["arrival_probability"] = {"breakpoints": [0], "values": [0.0]}
        if all_zero:
            d["background"][0]["purchase"]["max_prob"] = 0.0
        path = tmp_path / ("zero" + str(all_zero) + ".json")
        atomic_json(path, d)
        c = compile_market(load_spec(path))
        zero.append(
            {
                "id": "unit_zero" + str(all_zero),
                "path": str(path),
                "sha256": file_hash(path),
                "spec_hash": c.spec_hash,
                "all_demand_zero": all_zero,
            }
        )
    conversion = []
    for delta in [0, 3]:
        path = tmp_path / ("conversion" + str(delta) + ".json")
        atomic_json(path, fixture_model(delta))
        c = compile_market(load_spec(path))
        conversion.append(
            {
                "id": "unit_delta" + str(delta),
                "path": str(path),
                "sha256": file_hash(path),
                "spec_hash": c.spec_hash,
                "inside_shift": delta,
            }
        )
    reg = tmp_path / "registration.json"
    atomic_json(reg, {"unit_fixture": True, "T": 3, "price": 101})
    groups = [
        {
            "id": "unit_zero",
            "kind": "zero",
            "original_path": str(source),
            "original_sha256": file_hash(source),
            "models": zero,
        },
        {"id": "unit_conversion", "kind": "conversion", "models": conversion},
    ]
    return {
        "parameters": {
            "seed_start": 209500,
            "n_cohorts": 3,
            "groups": groups,
            "design_registration": str(reg),
            "design_registration_sha256": file_hash(reg),
        }
    }


def test_complete_driver_uses_one_tape_per_group_and_keeps_all_rows(tmp_path):
    cell = cell_fixture(tmp_path)
    context = SimpleNamespace(directory=tmp_path)
    result = study.run(cell, context)
    assert result["model_profiles"] == 12 and result["native_steps"] == 36
    assert result["actual_tapes_generated"] == 6 and result["group_seed_rows"] == 6
    assert (
        result["zero_all_demand_revenue_pass"]
        and result["conversion_declared_class_pass"]
    )
    assert result["full_MR6_pass"] is None and result["scientific_budget"] is None
    exact = json.loads((tmp_path / "exact_conversion.json").read_text())
    assert exact["simulation_rows_processed_yet"] == 0
    rows = [
        json.loads(s) for s in (tmp_path / "cohort_rows.jsonl").read_text().splitlines()
    ]
    assert [r["physical_seed"] for r in rows] == list(range(209500, 209503)) * 2
    assert (
        file_hash(tmp_path / "cohort_rows.jsonl")
        == result["artifacts"]["rows"]["sha256"]
    )


def test_final_seed_partition_stays_closed(tmp_path):
    cell = cell_fixture(tmp_path)
    cell["parameters"]["seed_start"] = 300000
    with pytest.raises(ValueError, match="validation IDs"):
        study.run(cell, SimpleNamespace(directory=tmp_path))


def test_changed_frozen_input_fails_before_physics(tmp_path):
    cell = cell_fixture(tmp_path)
    cell["parameters"]["design_registration_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen input"):
        study.run(cell, SimpleNamespace(directory=tmp_path))
