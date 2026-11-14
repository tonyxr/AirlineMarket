"""Exact price-map ambiguity, source replay and honest DLP diagnostics."""

import importlib.util
import json
import os
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from airline_market.agents.bidprice import solve_dlp
from airline_market.campaign.artifacts import file_hash
from airline_market.campaign.instances import build_instance
from airline_market.core.hashing import jsonable
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.spec import compile_market, dump_spec

if os.environ.get("LEG_STAGE"):
    name = "airline_market.experiments.leg_identification_study"
    spec = importlib.util.spec_from_file_location(
        name, Path(os.environ["LEG_STAGE"]) / "leg_identification_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
from airline_market.experiments import leg_identification_study as study


def fixture(generator="hub_feeder_small"):
    c = compile_market(build_instance(generator, {"T": 3, "actions": 4}))
    env = FocalCarrierEnv(c, "A", "direct")
    obs, _ = env.reset(203008)
    adapter = study.price_view(c)
    physical = np.r_[np.zeros(adapter.layout.n_legs), [0.0, 0.0, 7.5]]
    return c, adapter, obs, physical


def test_fraction_linear_system_is_exact_and_never_rounds_inputs():
    assert study.exact_solve([[1, 0], [0, 1], [1, 1]], [1, 1, 2]) == [
        Fraction(1),
        Fraction(1),
    ]
    with pytest.raises(ValueError):
        study.exact_solve([[1, 1]], [1])
    with pytest.raises(ValueError):
        study.exact_solve([[1], [1]], [0, 1])
    with pytest.raises(ValueError):
        study.exact_solve([[0.5]], [1])


@pytest.mark.parametrize("generator", ["toy_hub", "hub_feeder_small"])
def test_native_features_have_three_exact_price_null_directions(generator):
    _, adapter, obs, physical = fixture(generator)
    A, F, U = study.feature_leg_basis(adapter, obs)
    np.testing.assert_allclose(A.T @ U, F, atol=1e-14, rtol=0)
    assert np.linalg.matrix_rank(np.c_[A.T, F]) == A.shape[0]
    rows = study.shifted_controls(adapter, obs, physical)
    assert len(rows) == 6 and all(r["cent_prices_equal"] for r in rows)
    assert all(r["raw_price_difference_max"] < 1e-13 for r in rows)
    assert max(r["max_leg_value_change"] for r in rows) > 0
    assert not hasattr(adapter.env, "engine") and not hasattr(adapter.env, "world")


def test_bounds_limit_amplitude_without_dropping_or_clipping_the_row():
    _, adapter, obs, physical = fixture()
    physical[-3] = -30.0
    rows = study.shifted_controls(adapter, obs, physical)
    assert (
        rows[0]["status"] == "amplitude_limited_not_checked"
        and rows[0]["amplitude"] == 0
    )
    assert all(x["cent_prices_equal"] for x in rows if x["status"] == "checked")


def test_saved_observation_roundtrip_preserves_original_native_prices():
    _, adapter, obs, physical = fixture()
    restored = study.decode_observation(jsonable(obs))
    assert adapter.price_action(physical, restored) == adapter.price_action(
        physical, obs
    )
    data = jsonable(obs)
    data["privileged"] = {"world": "forbidden"}
    with pytest.raises(ValueError, match="nonprivileged"):
        study.decode_observation(data)


def test_dlp_unique_positive_units_and_nonunique_leg_references():
    A = np.array([[1, 0, 1], [0, 1, 1]])
    x, dual, value = solve_dlp(A, [3, 4], [100, 70, 180], [5, 5, 2])
    np.testing.assert_allclose(x, [1, 2, 2], atol=1e-9)
    np.testing.assert_allclose(dual, [100, 70], atol=1e-9)
    assert value == pytest.approx(600)
    # Both vectors certify the same separate deterministic LP; no unique leg label.
    A = np.ones((2, 2))
    c = np.ones(2)
    revenues = np.array([100, 50])
    d = np.array([2, 2])
    _, _, value = solve_dlp(A, c, revenues, d)
    for candidate in [np.array([100, 0]), np.array([0, 100])]:
        assert (A.T @ candidate >= revenues).all() and c @ candidate == pytest.approx(
            value
        )


def test_zero_constant_duals_remain_undefined_not_a_pass():
    result = study.descriptive_metrics([1.0, 2.0], [0.0, 0.0])
    assert (
        result["Pearson"] is None
        and result["median_relative_error_all_components"] is None
    )
    assert result["positive_target_median_relative_error"] is None
    assert result["zero_target_components"] == 2 and result["R1_acceptance"] is None
    other = study.descriptive_metrics([2.0, 4.0], [1.0, 2.0])
    assert (
        other["Pearson"] == pytest.approx(1)
        and other["median_relative_error_all_components"] == 1
    )


def test_actual_original_forecast_lp_has_a_checked_primal_dual_certificate():
    _, adapter, obs, physical = fixture()
    r = study.lp_comparison(adapter, obs, physical)
    assert abs(r["DLP_dual_gap"]) < 1e-8
    assert len(r["sampled_leg_values"]) == adapter.layout.n_legs


def test_read_only_driver_replays_sealed_rows_and_rejects_input_tampering(tmp_path):
    c, adapter, obs, physical = fixture()
    source = tmp_path / "source"
    source.mkdir()
    out = tmp_path / "output"
    out.mkdir()
    dump_spec(c.spec, source / "spec.json")
    prices = adapter.price_action(physical, obs).prices
    row = {
        "seed": 203008,
        "physical_action": physical.tolist(),
        "before_observation": jsonable(obs),
        "prices": prices,
    }
    (source / "predictions_final.jsonl").write_text(json.dumps(row) + "\n")
    r = {
        "status": "completed",
        "spec_hash": c.spec_hash,
        "artifact_hashes": {
            "predictions_final.jsonl": file_hash(source / "predictions_final.jsonl")
        },
    }
    (source / "results.json").write_text(json.dumps(r))
    (source / "COMPLETE.json").write_text(
        json.dumps({"results_sha256": file_hash(source / "results.json")})
    )
    p = {
        "input_hashes": {str(x): file_hash(x) for x in source.iterdir()},
        "source_receipt": str(source / "COMPLETE.json"),
        "source_results": str(source / "results.json"),
        "source_spec": str(source / "spec.json"),
        "source_rows": str(source / "predictions_final.jsonl"),
        "compiled_spec_hash": c.spec_hash,
        "kind": "event",
        "observation_field": "before_observation",
        "rows": 1,
    }
    result = study.run({"parameters": p}, SimpleNamespace(directory=out))
    assert result["checked_transforms"] == 6 and result["cent_price_mismatches"] == 0
    assert (
        result["R1_acceptance"] is None
        and not result["neural_policy_density_equivalence"]
    )
    assert result["checks_sha256"] == file_hash(out / "control_checks.jsonl")
    (source / "predictions_final.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="source changed"):
        study.run({"parameters": p}, SimpleNamespace(directory=out))
