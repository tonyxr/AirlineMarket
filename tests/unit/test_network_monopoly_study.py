"""Independent network recursion, native deferred-cost Q and constrained capacity."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from airline_market.campaign.artifacts import file_hash
from airline_market.spec import compile_market, load_spec

if os.environ.get("NETWORK_STAGE"):
    spec = importlib.util.spec_from_file_location(
        "airline_market.experiments.network_monopoly_study",
        Path(os.environ["NETWORK_STAGE"]) / "network_monopoly_study.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
from airline_market.experiments import network_monopoly_study as study


def config():
    return json.loads(Path("config/campaigns/V3_network_MR2_MR8.json").read_text())


def unit_compiled(capacity=3):
    cell = config()["cells"][0]
    data = json.loads(Path(cell["variants"][0]["monopoly"]["path"]).read_text())
    data["horizon"]["n_periods"] = 2
    data["legs"][0]["capacity"]["support"] = [capacity]
    return compile_market(load_spec(data))


def test_independent_recursion_does_not_call_native_DP(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("nativeDP")

    monkeypatch.setattr(study, "ExactBeliefDP", forbidden)
    r = study.independent_network(unit_compiled())
    assert r["root_value_USD"] > 0 and len(r["rows"]) == 8
    assert all(
        abs(x["no_sale_probability"] + sum(x["product_probabilities"]) - 1) <= 1e-12
        for row in r["rows"]
        for x in row["actions"]
    )


def test_zero_inventory_has_zero_future_value():
    r = study.independent_network(unit_compiled(0))
    assert r["root_value_USD"] == 0 and all(row["V_net_USD"] == 0 for row in r["rows"])


def test_native_reference_matches_all_Q_with_pending_service_adjustment():
    c = unit_compiled(1)
    reference = study.independent_network(c)
    r = study.native_reference(c, reference, 10000, {})
    assert r["backend"] == "reference" and r["kernel_type"] == "ExactKernel"
    assert abs(r["value_USD"] - reference["root_value_USD"]) <= 1e-10
    assert (
        r["all_native_cached_Q_max_abs_error"] <= 1e-10
        and r["max_raw_successor_mass_error"] <= 1e-12
    )
    assert any(row["existing_pending_service_cost_USD"] > 0 for row in r["rows"])
    assert all(row["privileged"] is None for row in r["rows"])


def test_constrained_cheap_itinerary_can_violate_capacity_monotonicity():
    cell = config()["cells"][2]

    # Distinct T2 fixture; the production T3 class is not solved in unit verification.
    def value(variant):
        d = json.loads(Path(variant["monopoly"]["path"]).read_text())
        d["horizon"]["n_periods"] = 2
        return study.independent_network(compile_market(load_spec(d)))["root_value_USD"]

    base = value(cell["variants"][0])
    extra = value(next(x for x in cell["variants"] if x["changed_leg"] == "HX"))
    assert extra < base


def test_contract_rejects_changed_objective_instead_of_rescaling():
    c = unit_compiled()
    d = json.loads(
        Path(config()["cells"][0]["variants"][0]["monopoly"]["path"]).read_text()
    )
    d["objectives"][0]["share_weight"] = 1.0
    with pytest.raises(ValueError, match="declared fully observed"):
        study.independent_network(compile_market(load_spec(d)))
    assert c.spec.objectives[0].profit_scale == 1.0


def test_driver_retains_general_parent_and_each_capacity_variant(tmp_path):
    cfg = config()
    cell = cfg["cells"][0]
    variants = []
    for i, original in enumerate(cell["variants"]):
        item = {"capacity": [3 + i], "changed_leg": original["changed_leg"]}
        for kind in ["monopoly", "unavailable_rival"]:
            d = json.loads(Path(original[kind]["path"]).read_text())
            d["horizon"]["n_periods"] = 2
            d["legs"][0]["capacity"]["support"] = [3 + i]
            path = tmp_path / f"{kind}_{i}.json"
            path.write_text(json.dumps(d))
            item[kind] = {
                "path": str(path),
                "sha256": file_hash(path),
                "compiled_spec_hash": compile_market(load_spec(path)).spec_hash,
            }
        variants.append(item)
    reg = tmp_path / "registration.json"
    reg.write_text("{}")
    p = {
        **cfg["fixed"],
        "network": "distinct_unit_fixture",
        "variants": variants,
        "design_registration": str(reg),
        "design_registration_sha256": file_hash(reg),
    }
    r = study.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
    assert (
        r["status"] == "completed"
        and r["MR8_declared_network_reference_pass"]
        and r["MR2_declared_capacity_edges_pass"]
    )
    assert (
        r["physical_cohorts"] == r["optimizer_updates"] == 0 and len(r["variants"]) == 2
    )
    assert all(
        file_hash(tmp_path / name) == h for name, h in r["artifact_hashes"].items()
    )
    assert r["full_V3_pass"] is None and r["full_V4_pass"] is None


def test_hash_guard_before_any_output(tmp_path):
    p = {**config()["fixed"], "design_registration_sha256": "0" * 64}
    with pytest.raises(ValueError, match="frozen network design"):
        study.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
    assert list(tmp_path.iterdir()) == []
