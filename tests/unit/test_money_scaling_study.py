"""Native monetary dimensions, cent counterexample and shared-tape control paths."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from airline_market.campaign.artifacts import file_hash
from airline_market.core.hashing import schema_hash
from airline_market.core.types import CoefficientAction
from airline_market.spec import compile_market, load_spec

if os.environ.get("MONEY_STAGE"):
    spec = importlib.util.spec_from_file_location(
        "airline_market.experiments.money_scaling_study",
        Path(os.environ["MONEY_STAGE"]) / "money_scaling_study.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
from airline_market.experiments import money_scaling_study as study


def unit_data():
    source = json.loads(Path("config/campaigns/V7_terminal_credit.json").read_text())[
        "cells"
    ][0]["source_spec"]
    d = json.loads(Path(source).read_text())
    # Distinct physical fixture; no registered production cohort is replayed.
    for leg in d["legs"]:
        if leg["carrier"] == "A":
            leg["capacity"]["support"] = [6]
    return d


def test_transform_respects_dimensioned_and_dimensionless_coefficients():
    d = unit_data()
    original = schema_hash(d)
    s = study.scaled_spec(d, 2)
    assert schema_hash(d) == original
    assert (
        s["segments"][0]["classes"][0]["alpha"]
        == d["segments"][0]["classes"][0]["alpha"] / 2
    )
    assert (
        s["pricing"][0]["fixed_features"][0][1]
        == d["pricing"][0]["fixed_features"][0][1]
    )
    assert s["pricing"][0]["theta_grid"] == [
        [2 * x for x in row] for row in d["pricing"][0]["theta_grid"]
    ]
    assert (
        s["background"][0]["purchase"]["slope"]
        == d["background"][0]["purchase"]["slope"] / 2
    )
    assert (
        s["controllers"][1]["params"]["rules"]["B_OH"]["responses"][0]["coefficient"]
        == 0.1
    )
    assert s["objectives"] == d["objectives"]


def test_unknown_transform_is_rejected_without_fallback():
    d = unit_data()
    d["controllers"][1]["controller"] = "unknown"
    with pytest.raises(ValueError, match="undeclared controller"):
        study.scaled_spec(d, 2)
    with pytest.raises(ValueError, match="integer monetary"):
        study.scaled_spec(unit_data(), 3)


def test_permitted_observation_is_lifted_without_mutation_or_hidden_input():
    c = compile_market(load_spec(unit_data()))
    e = study.MarketEngine(c)
    e.reset(299902)
    obs = e.observe("A", "direct")
    before = schema_hash(obs)
    scaled = study.scale_observation(obs, 4)
    assert schema_hash(obs) == before and scaled.privileged is None
    assert scaled.own_inventory == obs.own_inventory
    assert scaled.own_previous_prices == tuple(4 * p for p in obs.own_previous_prices)
    assert all(
        q.fare == 4 * p.fare
        for p, q in zip(obs.quotes, scaled.quotes)
        if p.fare is not None
    )


def test_native_cent_rounding_counterexample_is_reported_not_nudged():
    d = unit_data()
    c = compile_market(load_spec(d))
    v = compile_market(load_spec(study.scaled_spec(d, 2)))
    e = study.MarketEngine(c)
    e.reset(299903)
    e.step({0: CoefficientAction((0.0, 0.0, 0.0))})
    obs = e.observe("A", "direct")
    before = schema_hash(obs)
    base = study.AdditivePriceRule(c, c.spec.pricing[0]).prices((0.0, 0.0, 0.0), obs)
    shadow = study.AdditivePriceRule(v, v.spec.pricing[0]).prices(
        (0.0, 0.0, 0.0), study.scale_observation(obs, 2)
    )
    assert np.max(np.abs(shadow - 2 * base)) > 1e-8
    assert schema_hash(obs) == before


def test_observer_subclass_calls_actual_native_rule_once(monkeypatch):
    c = compile_market(load_spec(unit_data()))
    e = study.ObservedPriceEngine(c)
    e.reset(299904)
    count = []
    old = e.rules[0].prices

    def call(*args):
        count.append(1)
        return old(*args)

    monkeypatch.setattr(e.rules[0], "prices", call)
    e.step({0: CoefficientAction((0.0, 0.0, 0.0))})
    assert len(count) == 1 and 0 in e.actual_price_inputs
    assert study.profile_structure(e)["t"] == 1


def test_full_driver_retains_negative_rule_rows_and_all_shared_control_paths(tmp_path):
    data = unit_data()
    source = tmp_path / "unit.json"
    source.write_text(json.dumps(data))
    registration = tmp_path / "registration.json"
    registration.write_text("{}")
    variants = []
    for factor in [2, 4]:
        path = tmp_path / f"x{factor}.json"
        path.write_text(json.dumps(study.scaled_spec(data, factor)))
        variants.append(
            {
                "factor": factor,
                "source_spec": str(path),
                "source_spec_sha256": file_hash(path),
                "compiled_spec_hash": compile_market(load_spec(path)).spec_hash,
            }
        )
    p = {
        "instance": "distinct_unit",
        "source_spec": str(source),
        "source_spec_sha256": file_hash(source),
        "compiled_spec_hash": compile_market(load_spec(source)).spec_hash,
        "scaled": variants,
        "design_registration": str(registration),
        "design_registration_sha256": file_hash(registration),
        "n_cohorts": 2,
        "seed_start": 299905,
        "probability_tolerance": 1e-12,
        "monetary_tolerance": 1e-8,
    }
    r = study.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
    assert (
        r["status"] == "completed"
        and r["physical_cohorts"] == 2
        and r["native_profile_cohorts"] == 6
    )
    assert all(
        row["aligned_price_choice_accounting_pass"] for row in r["results_by_factor"]
    )
    assert any(
        not row["native_same_history_coefficient_scaling_pass"]
        for row in r["results_by_factor"]
    )
    rows = [
        json.loads(x) for x in (tmp_path / "cohort_rows.jsonl").read_text().splitlines()
    ]
    assert [row["seed"] for row in rows] == [299905, 299906] and all(
        row["shared_actual_tape_object"] for row in rows
    )
    assert all(len(row["periods"]) == 10 for row in rows)
    assert all(
        file_hash(tmp_path / name) == h for name, h in r["artifact_hashes"].items()
    )
    assert r["full_V3_pass"] is None


def test_seed_guard_before_any_physical_output(tmp_path):
    with pytest.raises(ValueError, match="validation"):
        study.run(
            {"parameters": {"seed_start": 299999, "n_cohorts": 2}},
            SimpleNamespace(directory=tmp_path),
        )
    assert list(tmp_path.iterdir()) == []
