"""Small analytical identification fixtures, no production source bank or RNG."""

import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.campaign.worker import CellContext
from airline_market.spec import compile_market, load_spec

if os.environ.get("DEMAND_IDENT_STAGE"):
    name = "airline_market.experiments.demand_identification_study"
    spec = importlib.util.spec_from_file_location(
        name, Path(os.environ["DEMAND_IDENT_STAGE"]) / "demand_identification_study.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
from airline_market.experiments import demand_identification_study as study


def fixture():
    d = json.loads(Path("airline_market/spec/library/toy_hub.json").read_text())
    d["demand"]["mode"] = "poisson"
    d["demand"]["arrival_probability"] = None
    d["demand"]["intensity"] = {m["id"]: {"constant": 3.0} for m in d["markets"]}
    d["segments"] = [
        {
            "id": "one",
            "classes": [
                {
                    "weight": 1.0,
                    "alpha": 0.02,
                    "market_value": {"*": 1.0},
                    "carrier_pref": {"A": 0.3, "B": -0.1},
                }
            ],
        },
        {
            "id": "two",
            "classes": [
                {
                    "weight": 1.0,
                    "alpha": 0.01,
                    "market_value": {"*": 1.2},
                    "carrier_pref": {"A": -0.2, "B": 0.2},
                }
            ],
        },
    ]
    d["demand"]["segment_weights"] = {"constant": [0.7, 0.3]}
    d["choice"]["model"] = "latent_class_mnl"
    return d


def test_closed_form_keeps_every_inside_intensity_and_changes_nonpurchase():
    u = np.array([0.2, -0.4, 0.0])
    p = np.exp(u) / np.exp(u).sum()
    delta = study.compensating_shift(p[-1], 2.0)
    v = u.copy()
    v[:-1] += delta
    q = np.exp(v) / np.exp(v).sum()
    np.testing.assert_allclose(2 * q[:-1], p[:-1], atol=1e-14, rtol=0)
    assert q[-1] > p[-1]
    assert delta == pytest.approx(math.log(p[-1]) - math.log(2 - (1 - p[-1])))


@pytest.mark.parametrize("p0,r", [(0, 2), (1, 2), (0.2, 0.8), (0.2, float("inf"))])
def test_invalid_or_boundary_reparameterizations_are_errors(p0, r):
    with pytest.raises(ValueError):
        study.compensating_shift(p0, r)


def test_native_mixture_reference_equality_null_and_intervention_boundary():
    c = compile_market(load_spec(fixture()))
    before = c.spec_hash
    rows, witnesses, calls = study.analyze(c, [1.25, 2, 4], 10, 1e-10, 1e-10)
    assert calls == len(c.spec.markets) * 2 * 12
    assert all(
        r["rank"] < r["parameter_count"] and r["null_residual"] < 1e-12 for r in rows
    )
    errors = [e for r in rows for e in r["scenario_errors"]]
    assert (
        max(
            e["max_absolute_purchase_intensity_difference"]
            for e in errors
            if e["scenario"] == "reference_all_offered"
        )
        < 1e-12
    )
    assert (
        max(
            e["max_absolute_purchase_intensity_difference"]
            for e in errors
            if e["scenario"] != "reference_all_offered"
        )
        > 1e-5
    )
    assert c.spec_hash == before
    assert all(w["transformed_utilities"][0][-1] == 0 for w in witnesses)
    assert any(None in w["original_utilities"][0] for w in witnesses)


def test_actual_worker_preserves_source_and_emits_transitive_artifact_hashes(tmp_path):
    p = tmp_path / "source.json"
    atomic_json(p, fixture())
    reg = tmp_path / "registration.json"
    atomic_json(reg, {"tiny_fixture": True})
    params = {
        "source_spec": str(p),
        "source_spec_sha256": file_hash(p),
        "design_registration": str(reg),
        "design_registration_sha256": file_hash(reg),
        "arrival_multipliers": [1.25, 2, 4],
        "core_price_increment": 10,
        "intensity_tolerance": 1e-10,
        "rank_rtol": 1e-10,
    }
    out = tmp_path / "worker"
    out.mkdir()
    result = study.run({"parameters": params}, CellContext(out))
    assert result["reference_equivalences"] == len(fixture()["markets"]) * 3
    assert (
        result["new_simulation_cohorts"]
        == result["new_RNG_calls"]
        == result["parameter_fits"]
        == 0
    )
    assert result["calibration_at_optimum_Jacobian"] is None
    assert file_hash(p) == params["source_spec_sha256"]
    assert all(file_hash(out / n) == h for n, h in result["artifact_hashes"].items())


def test_nonpurchase_is_distinguishable_and_jacobian_matches_finite_difference():
    c = compile_market(load_spec(fixture()))
    rows, witnesses, _ = study.analyze(c, [2], 10, 1e-10, 1e-10)
    for w in witnesses:
        if w["scenario"] == "reference_all_offered":
            assert w["transformed_nonpurchase_intensity"] - w[
                "original_nonpurchase_intensity"
            ] == pytest.approx(w["original_potential_intensity"])
    row = rows[0]
    w = witnesses[0]
    u = np.array(w["original_utilities"])
    weights = np.array(row["segment_weights"])
    lam = row["arrival_intensity"]
    h = 1e-5

    def moment(parameters):
        util = u.copy()
        util[:, :-1] += parameters[1:, None]
        prob = np.exp(util - util.max(axis=1, keepdims=True))
        prob /= prob.sum(axis=1, keepdims=True)
        return lam * np.exp(parameters[0]) * (weights @ prob[:, :-1])

    numerical = []
    for k in range(3):
        step = np.zeros(3)
        step[k] = h
        numerical.append((moment(step) - moment(-step)) / (2 * h))
    np.testing.assert_allclose(
        np.column_stack(numerical), row["jacobian"], atol=1e-9, rtol=1e-7
    )
