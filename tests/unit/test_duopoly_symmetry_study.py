"""Independent moments, native fixed-policy roots and paired-equivalence semantics."""

import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from airline_market.campaign.artifacts import file_hash
from airline_market.spec import compile_market, load_spec

if os.environ.get("SYMMETRY_STAGE"):
    spec = importlib.util.spec_from_file_location(
        "airline_market.experiments.duopoly_symmetry_study",
        Path(os.environ["SYMMETRY_STAGE"]) / "duopoly_symmetry_study.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
from airline_market.experiments import duopoly_symmetry_study as study


def fixture():
    d = json.loads(
        Path("config/benchmarks/V3_MR4_symmetric_policy_T4.json").read_text()
    )
    d["horizon"]["n_periods"] = 2
    for leg in d["legs"]:
        leg["capacity"]["support"] = [3]
    return compile_market(load_spec(d))


def test_independent_moments_match_closed_no_binding_two_periods(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("native call")

    monkeypatch.setattr(study, "ExactBeliefDP", forbidden)
    c = fixture()
    r = study.independent_moments(c, 100)
    prob = 0.8 * math.exp(0.5) / (1 + 2 * math.exp(0.5))
    assert r["root_mean_profit_USD"] == pytest.approx([2 * 80 * prob] * 2, abs=1e-12)
    assert r["root_paired_variance_USD2"] == pytest.approx(
        2 * 2 * prob * 80**2, abs=1e-10
    )
    assert abs(r["root_difference_USD"]) <= 1e-12 and r[
        "exact_focal_revenue_USD"
    ] == pytest.approx(2 * 100 * prob, abs=1e-12)


def test_native_general_reference_roots_match_independent_fixture():
    c = fixture()
    a = study.independent_moments(c, 100)
    n = study.native_expectations(c, 10000)
    assert all(
        x["backend"] == "reference" and x["kernel_type"] == "ExactKernel" for x in n
    )
    assert [x["value_USD"] for x in n] == pytest.approx(
        a["root_mean_profit_USD"], abs=1e-10
    )
    assert all(abs(m - 1) <= 1e-12 for x in n for m in x["raw_successor_masses"])


def test_TOST_can_pass_with_nonzero_variance_and_preserves_boot_namespace():
    r = study.paired_statistics(np.tile([-0.01, 0.01], 100), 0.1)
    assert r["TOST_equivalent"] and r["TOST_p_max"] < 0.05
    assert (
        r["bootstrap_namespace"] == "boot"
        and r["bootstrap_root"] == 773
        and r["bootstrap_replicates"] == 2000
    )
    assert r["normal_95_CI_USD"][0] < 0 < r["normal_95_CI_USD"][1]


def test_non_significance_is_not_equivalence_and_zero_variance_is_inconclusive():
    r = study.paired_statistics([-80.0, 80.0] * 4, 0.1)
    assert (
        r["normal_95_CI_USD"][0] < 0 < r["normal_95_CI_USD"][1]
        and not r["TOST_equivalent"]
    )
    z = study.paired_statistics([0.0] * 8, 0.1)
    assert (
        z["TOST_equivalent"] is None
        and z["TOST_p_max"] is None
        and z["TOST_status"] == "inconclusive_zero_sample_variance"
    )


def test_skewed_sample_uses_BCa_without_dropping_outlier():
    r = study.paired_statistics([0.0] * 99 + [50.0], 0.1)
    assert (
        r["sample_skewness"] > 1
        and r["reported_bootstrap_method"] == "BCa"
        and r["BCa_95_USD"] is not None
    )
    assert (
        r["strict_A_greater_B"] == 1
        and r["ties"] == 99
        and r["mean_difference_USD"] == 0.5
    )


def test_asymmetric_model_rejected_before_native_mean():
    d = study.jsonable(fixture().spec)
    d["products"][1]["service_cost"] = 21
    with pytest.raises(AssertionError):
        study.independent_moments(compile_market(load_spec(d)), 100)


def parameters(tmp_path):
    c = fixture()
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(study.jsonable(c.spec)))
    reg = tmp_path / "design.json"
    reg.write_text("{}")
    p = json.loads(Path("config/campaigns/V3_MR4_symmetric_policy.json").read_text())[
        "fixed"
    ]
    return {
        **p,
        "source_spec": str(path),
        "source_spec_sha256": file_hash(path),
        "compiled_spec_hash": c.spec_hash,
        "design_registration": str(reg),
        "design_registration_sha256": file_hash(reg),
        "n_cohorts": 8,
        "seed_start": 200980,
    }


def test_driver_saves_every_full_fixture_and_exact_target_before_simulation(tmp_path):
    p = parameters(tmp_path)
    r = study.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
    assert (
        r["status"] == "completed"
        and r["n_cohorts"] == 8
        and r["native_period_steps"] == 16
        and r["exact_symmetry_pass"]
    )
    rows = [
        json.loads(x)
        for x in (tmp_path / "native_cohorts.jsonl").read_text().splitlines()
    ]
    assert [x["physical_seed"] for x in rows] == list(range(200980, 200988)) and all(
        len(x["categories"]) == 2 for x in rows
    )
    assert (
        json.loads((tmp_path / "exact_reference.json").read_text())[
            "simulation_rows_processed_yet"
        ]
        == 0
    )
    assert r["full_V3_pass"] is None and r["scientific_budget"] is None
    assert all(
        file_hash(tmp_path / name) == h for name, h in r["artifact_hashes"].items()
    )


def test_hash_and_final_seed_guards_precede_native_work(tmp_path, monkeypatch):
    p = parameters(tmp_path)
    p["source_spec_sha256"] = "0" * 64

    def forbidden(*a, **k):
        raise AssertionError("native call")

    monkeypatch.setattr(study, "MarketEngine", forbidden)
    with pytest.raises(ValueError, match="frozen symmetry"):
        study.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
    p["seed_start"] = 300000
    with pytest.raises(ValueError, match="validation cohort"):
        study.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
    assert not (tmp_path / "native_cohorts.jsonl").exists()
