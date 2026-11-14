"""Permutation coordinates, saved-original reuse and distinct native fixtures."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from airline_market.campaign.artifacts import file_hash
from airline_market.spec import compile_market, load_spec

if os.environ.get("RELABEL_STAGE"):
    spec = importlib.util.spec_from_file_location(
        "airline_market.experiments.relabeling_study",
        Path(os.environ["RELABEL_STAGE"]) / "relabeling_study.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
from airline_market.experiments import relabeling_study as study


def old_case(index=1):
    cfg = json.loads(Path("config/campaigns/V3_MR3_relabeling.json").read_text())
    p = cfg["cells"][index]
    return cfg, p, json.loads(Path(p["original_source"]).read_text())


def test_source_incidence_carrier_and_market_weights_are_permuted():
    _, _, old = old_case()
    new, m = study.permute_spec(old)
    c = compile_market(load_spec(new))
    assert (
        new["products"][0]["id"]
        == m["old_to_new"]["products"][old["products"][-1]["id"]]
    )
    assert new["demand"]["market_weights"]["constant"] == list(
        reversed(old["demand"]["market_weights"]["constant"])
    )
    for j, previous in enumerate(m["new_to_old_products"]):
        for l, oldleg in enumerate(m["new_to_old_legs"]):
            assert int(c.A_global[l, j]) == int(
                old["legs"][oldleg]["id"] in old["products"][previous]["legs"]
            )
    assert old == old_case()[2]


def test_per_product_columns_and_action_rows_both_reverse():
    _, _, old = old_case()
    new, _ = study.permute_spec(old)
    assert new["pricing"][0]["theta_grid"] == [
        list(reversed(row)) for row in reversed(old["pricing"][0]["theta_grid"])
    ]
    assert new["pricing"][0]["theta_bounds"] == list(
        reversed(old["pricing"][0]["theta_bounds"])
    )


def test_duopoly_asymmetric_costs_preferences_and_schedule_survive():
    _, p, old = old_case(4)
    new, m = study.permute_spec(old)
    c = compile_market(load_spec(new))
    assert new["demand"]["market_weights"]["values"] == [[1.0]]
    assert c.product_cost.tolist() == [
        old["products"][i]["service_cost"] for i in m["new_to_old_products"]
    ]
    taste = new["segments"][0]["classes"][0]
    assert taste["carrier_pref"] == {
        m["old_to_new"]["carriers"][k]: v
        for k, v in old["segments"][0]["classes"][0]["carrier_pref"].items()
    }
    assert file_hash(p["source_spec"]) == p["source_spec_sha256"]


def test_rejects_unmapped_quote_or_history_feature_instead_of_guessing():
    _, _, old = old_case()
    old["quotes"]["requests"] = [{"id": "unmapped"}]
    with pytest.raises(ValueError, match="noevent/noquote"):
        study.permute_spec(old)
    _, _, old = old_case()
    old["pricing"][0]["fixed_features"] = [[{"name": "rival_gap"}, 1]]
    with pytest.raises(ValueError, match="static intercept"):
        study.permute_spec(old)


def unit_case(tmp_path, kind):
    from airline_market.experiments.network_monopoly_study import independent_network

    index = 0 if kind == "network" else 4
    cfg, p, old = old_case(index)
    # Distinct fixtures: T2/three-seat network or lower-utility logit. No original solve/sampler.
    if kind == "network":
        old["horizon"]["n_periods"] = 2
        old["legs"][0]["capacity"]["support"] = [3]
    else:
        old["segments"][0]["classes"][0]["market_value"] = {
            old["markets"][0]["id"]: 2.0
        }
    original = tmp_path / "original_fixture.json"
    original.write_text(json.dumps(old))
    new, mapping = study.permute_spec(old)
    path = tmp_path / "permuted_fixture.json"
    path.write_text(json.dumps(new))
    c = compile_market(load_spec(path))
    refs = []
    result = tmp_path / "reference_results.json"
    receipt = tmp_path / "reference_receipt.json"
    if kind == "network":
        reference = independent_network(compile_market(load_spec(old)))
        table = tmp_path / "classical.json"
        table.write_text(
            json.dumps(
                {
                    "root_value_USD": reference["root_value_USD"],
                    "rows": reference["rows"],
                }
            )
        )
        r = {"variants": [{"native_monopoly_value_USD": reference["root_value_USD"]}]}
        extra = {"reference_variant": 0, "reference_classical": str(table)}
        refs.append(table)
    else:
        import math

        analytical = tmp_path / "analytical.json"
        values = [
            2 + old["segments"][0]["classes"][0]["carrier_pref"].get(prod["carrier"], 0)
            for prod in old["products"]
        ]
        price = [100.0, 110.0]
        weights = [math.exp(v - 0.025 * x) for v, x in zip(values, price)]
        denom = 1 + sum(weights)
        probs = [0.85 * w / denom for w in weights] + [0.85 / denom, 0.15]
        analytical.write_text(
            json.dumps({"points": [{"prices": price, "native_raw_categories": probs}]})
        )
        r = {}
        extra = {"reference_analytical": str(analytical)}
        refs.append(analytical)
    result.write_text(json.dumps(r))
    receipt.write_text(json.dumps({"results_sha256": file_hash(result)}))
    refs.extend([result, receipt])
    reg = tmp_path / "design.json"
    reg.write_text("{}")
    return {
        **cfg["fixed"],
        **p,
        **extra,
        "original_source": str(original),
        "original_source_sha256": file_hash(original),
        "source_spec": str(path),
        "source_spec_sha256": file_hash(path),
        "compiled_spec_hash": c.spec_hash,
        "reference_results": str(result),
        "reference_receipt": str(receipt),
        "mapping": mapping,
        "saved_references": [{"path": str(x), "sha256": file_hash(x)} for x in refs],
        "design_registration": str(reg),
        "design_registration_sha256": file_hash(reg),
    }


def test_driver_permuted_network_maps_all_Q_and_prices_without_old_solve(tmp_path):
    p = unit_case(tmp_path, "network")
    r = study.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
    w = json.loads((tmp_path / "permutation_witnesses.json").read_text())
    assert (
        r["declared_MR3_pass"]
        and w["root_error_USD"] <= 1e-12
        and w["max_Q_error_USD"] <= 1e-12
        and w["max_price_error_USD"] == 0
    )
    assert r["physical_cohorts"] == r[
        "original_source_solver_or_sampler_reruns"
    ] == 0 and all(x["privileged"] is None for x in w["rows"])
    assert all(
        file_hash(tmp_path / name) == h for name, h in r["artifact_hashes"].items()
    )


def test_driver_permuted_asymmetric_logit_maps_output_profit_vector(tmp_path):
    p = unit_case(tmp_path, "logit")
    r = study.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
    w = json.loads((tmp_path / "permutation_witnesses.json").read_text())
    assert (
        r["declared_MR3_pass"]
        and w["kernel_queries"] == 1
        and w["max_probability_or_profit_error"] <= 1e-12
    )
    assert (
        w["rows"][0]["new_expected_profit_USD"][0]
        != w["rows"][0]["new_expected_profit_USD"][1]
        and r["full_V3_pass"] is None
    )


def test_hash_guard_before_any_native_call_or_output(tmp_path, monkeypatch):
    p = unit_case(tmp_path, "network")
    p["original_source_sha256"] = "0" * 64

    def forbidden(*a, **k):
        raise AssertionError("new native call")

    monkeypatch.setattr(study, "ExactBeliefDP", forbidden)
    with pytest.raises(ValueError, match="frozen relabel"):
        study.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
    assert not (tmp_path / "permutation_witnesses.json").exists()
