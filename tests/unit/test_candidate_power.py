"""Synthetic receipt/accounting fixtures and section2.4 sample-size boundaries."""

import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.spec import compile_market, dump_spec, load_spec

module = importlib.import_module(
    os.environ.get(
        "AIRLINE_POWER_TEST_MODULE", "airline_market.experiments.candidate_power_study"
    )
)


def synthetic(tmp_path, name="one", cpu=2):
    d = tmp_path / name
    d.mkdir()
    spec = load_spec("airline_market/spec/library/toy_hub.json")
    dump_spec(spec, d / "spec.json")
    atomic_json(d / "tails.json", {"fixture": "synthetic, not a fitted result"})
    parameters = {
        "seed_start": 200000,
        "n_cohorts": 256,
        "tails_sha256": file_hash(d / "tails.json"),
        "instance": "SYNTHETIC",
        "method": "all",
        "tail": "fixed",
        "scenarios": 32,
    }
    atomic_json(
        d / "config.json", {"config_hash": "synthetic-config", "parameters": parameters}
    )
    delta = np.array([-1.0, 1.0] * 128)
    rows = [
        {
            "seed": 200000 + i,
            "profit": 300 + x,
            "fixed_profit": 300.0,
            "revenue": 400.0,
            "fixed_revenue": 399.0,
            "direct_minus_fixed_profit": x,
            "value": 99999.0,
        }
        for i, x in enumerate(delta)
    ]
    atomic_json(d / "episodes.json", rows)
    atomic_json(
        d / "results.json",
        {
            "config_hash": "synthetic-config",
            "status": "completed",
            "n_cohorts": 256,
            "spec_hash": compile_market(spec).spec_hash,
            "mean_direct_minus_fixed_profit": float(delta.mean()),
            "paired_standard_error": float(delta.std(ddof=1) / 16),
            "mean_cohort_revenue": 400.0,
            "wall_seconds": 4.0,
            "cpu_seconds": cpu,
        },
    )
    atomic_json(
        d / "COMPLETE.json",
        {
            "config_hash": "synthetic-config",
            "status": "completed",
            "results_sha256": file_hash(d / "results.json"),
        },
    )
    return {
        "cell_id": name,
        "config_hash": "synthetic-config",
        "files": {
            n: {"path": str(d / n), "sha256": file_hash(d / n)}
            for n in [
                "config.json",
                "results.json",
                "COMPLETE.json",
                "episodes.json",
                "spec.json",
                "tails.json",
            ]
        },
    }


def alter(entry, name, function):
    p = Path(entry["files"][name]["path"])
    data = json.loads(p.read_text())
    function(data)
    atomic_json(p, data)
    entry["files"][name]["sha256"] = file_hash(p)
    if name == "results.json":
        alter(entry, "COMPLETE.json", lambda d: d.update(results_sha256=file_hash(p)))


def parameters():
    return {
        "alpha": 0.05,
        "power": 0.8,
        "revenue_fraction": 0.005,
        "min_n": 512,
        "max_n": 8192,
        "round_multiple": 256,
    }


def test_plan_example_rounds_to1280_and_cap_failure_is_explicit():
    r = module.power_projection(20.5, 1.75)
    assert r["capped_candidate_n"] == 1280 and r["rounded_unbounded_n"] == 1280
    assert (
        r["raw_n"] == 1078
    )  # exact normal quantiles; printed plan example is approximate
    assert r["target_met_at_candidate_n"] is True
    r = module.power_projection(100, 0.01)
    assert r["cap_binds"] is True and r["capped_candidate_n"] == 8192
    assert r["target_met_at_candidate_n"] is False and r["mde_at_candidate_n"] > 0.01


def test_zero_pilot_variance_retains_minimum_without_equivalence_claim():
    r = module.power_projection(0, 1)
    assert r["raw_n"] == 0 and r["capped_candidate_n"] == 512
    assert r["pilot_variance_is_zero"] is True


@pytest.mark.parametrize("sd,delta", [(-1, 1), (np.nan, 1), (np.inf, 1), (1, 0)])
def test_invalid_numeric_power_inputs(sd, delta):
    with pytest.raises(ValueError):
        module.power_projection(sd, delta)


def test_power_probability_and_block_guards():
    for options in [{"alpha": 1}, {"power": 0.5}, {"minimum": 600}, {"maximum": 511}]:
        with pytest.raises(ValueError):
            module.power_projection(1, 1, **options)


def test_real_dollar_pairing_is_used_instead_of_normalized_J(tmp_path):
    entry = synthetic(tmp_path)
    _p, _result, difference, revenue, baseline = module.audit_entry(entry)
    assert revenue == 400 and len(baseline) == 256
    np.testing.assert_array_equal(difference, [-1.0, 1.0] * 128)
    out = tmp_path / "out"
    out.mkdir()
    report = module.catalog(
        {"entries": [entry], "not_sealed_at_snapshot": ["later"]}, parameters(), out
    )
    row = report["rows"][0]
    assert row["paired_profit_sd_usd"] == pytest.approx(np.sqrt(256 / 255))
    assert row["target_delta_usd"] == 2 and row["mean_profit_difference_usd"] == 0
    assert row["F5_comparison_wall_seconds_per_cohort"] == 4 / 256
    assert (
        row["phase4_primary_core_hours"] is None
        and report["phase4_primary_n_selected"] is None
    )


def test_changed_registered_episode_bytes_are_rejected(tmp_path):
    entry = synthetic(tmp_path)
    Path(entry["files"]["episodes.json"]["path"]).write_text("[]")
    with pytest.raises(ValueError, match="changed frozen"):
        module.audit_entry(entry)


@pytest.mark.parametrize("kind", ["duplicate", "missing", "final"])
def test_exact256_validation_pairing_guards(tmp_path, kind):
    entry = synthetic(tmp_path)
    if kind == "duplicate":
        alter(entry, "episodes.json", lambda r: r[1].update(seed=200000))
    elif kind == "missing":
        alter(entry, "episodes.json", lambda r: r.pop())
    else:
        alter(entry, "config.json", lambda c: c["parameters"].update(seed_start=300000))
    with pytest.raises(ValueError, match="cohort"):
        module.audit_entry(entry)


def test_accounting_and_original_sealed_aggregate_mismatch_rejected(tmp_path):
    entry = synthetic(tmp_path)
    alter(entry, "episodes.json", lambda r: r[0].update(direct_minus_fixed_profit=100))
    with pytest.raises(ValueError, match="accounting"):
        module.audit_entry(entry)
    entry = synthetic(tmp_path, "second")
    alter(entry, "results.json", lambda d: d.update(paired_standard_error=999))
    with pytest.raises(ValueError, match="aggregates"):
        module.audit_entry(entry)


def test_shared_fitted_baseline_mismatch_is_not_silently_pooled(tmp_path):
    one, two = synthetic(tmp_path), synthetic(tmp_path, "two")

    def change(rows):
        for r in rows:
            r["profit"] += 1
            r["fixed_profit"] += 1

    alter(two, "episodes.json", change)
    with pytest.raises(ValueError, match="baseline differs"):
        module.catalog(
            {"entries": [one, two], "not_sealed_at_snapshot": []},
            parameters(),
            tmp_path,
        )


def test_missing_cpu_cost_stays_null(tmp_path):
    entry = synthetic(tmp_path, cpu=None)
    report = module.catalog(
        {"entries": [entry], "not_sealed_at_snapshot": []}, parameters(), tmp_path
    )
    assert report["rows"][0]["projected_same_F5_comparison_cpu_hours"] is None


@pytest.mark.parametrize("kind", ["physical", "receipt"])
def test_original_physical_identity_and_result_seal_are_required(tmp_path, kind):
    entry = synthetic(tmp_path)
    if kind == "physical":
        alter(entry, "results.json", lambda d: d.update(spec_hash="different-model"))
        pattern = "physical"
    else:
        alter(entry, "COMPLETE.json", lambda d: d.update(results_sha256="wrong"))
        pattern = "receipt"
    with pytest.raises(ValueError, match=pattern):
        module.audit_entry(entry)


def test_actual_wrapper_preserves_all_candidates_and_manifest_identity(tmp_path):
    entry = synthetic(tmp_path)
    path = tmp_path / "manifest.json"
    atomic_json(path, {"entries": [entry], "not_sealed_at_snapshot": []})
    p = {
        "manifest": str(path),
        "manifest_sha256": file_hash(path),
        "implementation_sha256": file_hash(module.__file__),
        "power": parameters(),
    }
    result = module.study({"parameters": p}, SimpleNamespace(directory=tmp_path))
    assert result["candidates"] == 1 and result["simulation_cohorts_rerun"] == 0
    with pytest.raises(ValueError, match="changed registered"):
        module.study(
            {"parameters": {**p, "manifest_sha256": "wrong"}},
            SimpleNamespace(directory=tmp_path),
        )
