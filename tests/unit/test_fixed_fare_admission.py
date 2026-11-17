"""Distinct T2/101-and41 fare fixtures for the new native-ledger RM mode."""

import copy
import importlib.util
import json
import os
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.hashing import jsonable
from airline_market.sim.accounting import carrier_metrics
from airline_market.spec import compile_market, load_spec

if os.environ.get("FIXED_FARE_STAGE"):
    for package, name in [
        ("sim", "fixed_fare_admission"),
        ("experiments", "fixed_fare_study"),
    ]:
        qualified = "airline_market." + package + "." + name
        spec = importlib.util.spec_from_file_location(
            qualified, Path(os.environ["FIXED_FARE_STAGE"]) / (name + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
from airline_market.sim.fixed_fare_admission import FixedFareAdmissionEngine

from airline_market.experiments import fixed_fare_study as study


def fixture_source():
    source = json.loads(
        Path("config/benchmarks/V4_fixed_fare_single_two_class.json").read_text()
    )
    source["case"] = "unit_T2_101_41"
    source["physical_spec"]["horizon"]["n_periods"] = 2
    source["physical_spec"]["legs"][0]["capacity"]["support"] = [1]
    source["class_probabilities"] = [[0.5, 0.5]] * 2
    for product, fare in zip(source["physical_spec"]["products"], [101, 41]):
        product["reference_fare"] = fare
        source["physical_spec"]["pricing"][0]["base"][product["id"]] = fare
    return source


def engine(source=None):
    source = source or fixture_source()
    return FixedFareAdmissionEngine(
        compile_market(load_spec(source["physical_spec"])),
        source["class_probabilities"],
    )


def snapshot(e):
    return (
        e.world.t,
        e.world.next_record_id,
        copy.deepcopy(e.world.request_log),
        jsonable(e.world.carriers[0]),
        carrier_metrics(e.world, e.compiled, 0),
    )


def test_real_rejection_keeps_stock_and_cash_then_native_booking_and_cost():
    e = engine()
    obs = e.observe(1)
    assert (
        obs.own_inventory == (1,)
        and obs.requested_product == 1
        and obs.fixed_fare_USD == 41
    )
    with pytest.raises(FrozenInstanceError):
        obs.period = 99
    e.step(1, False)
    assert e.world.carriers[0].cash == 0 and e.observe(0).own_inventory == (1,)
    e.step(0, True)
    m = carrier_metrics(e.world, e.compiled, 0)
    assert (
        m["inventory"] == [0]
        and m["revenue"] == 101
        and m["service_cost"] == 5
        and m["profit"] == 96
    )
    assert (
        len(e.world.carriers[0].bookings) == 1
        and e.world.request_log[0]["admitted"] is False
    )


def test_oversell_fails_without_partial_mutation():
    e = engine()
    e.step(0, True)
    before = snapshot(e)
    with pytest.raises(ValueError, match="available leg inventory"):
        e.step(1, True)
    assert snapshot(e) == before


def test_cloned_native_ledgers_do_not_change_parent():
    e = engine()
    before = snapshot(e)
    child = e.clone()
    child.step(0, True)
    assert snapshot(e) == before and len(child.world.carriers[0].bookings) == 1


def test_native_history_tree_matches_hand_value_and_every_classical_state():
    source = fixture_source()
    e = engine(source)
    reference = study.classical_reference(source)
    value, nodes = study.native_ledger_tree(e)
    assert value == 81.75 and str(reference["table"][0, (1,)]) == "327/4"
    assert all(
        abs(
            n["native_final_value_USD"]
            - n["already_booked_net_USD"]
            - float(reference["table"][n["t"], tuple(n["stock"])])
        )
        < 1e-10
        for n in nodes
    )
    assert e.world.t == 0 and e.world.carriers[0].bookings == []


def test_negative_margin_can_be_rejected():
    source = fixture_source()
    source["physical_spec"]["products"][1]["service_cost"] = 60
    value, _ = study.native_ledger_tree(engine(source))
    ref = study.classical_reference(source)
    assert value == 72 and float(ref["table"][0, (1,)]) == 72


@pytest.mark.parametrize("bad", [[-0.1, 0.5], [0.8, 0.8], [0.5]])
def test_invalid_class_law_rejected(bad):
    source = fixture_source()
    source["class_probabilities"][0] = bad
    with pytest.raises(ValueError):
        engine(source)


def test_terminal_and_unknown_class_are_errors():
    e = engine()
    with pytest.raises(ValueError, match="unknown request class"):
        e.observe(True)
    e.step(None, False)
    e.step(None, False)
    with pytest.raises(ValueError, match="terminal"):
        e.step(None, False)


def test_complete_driver_and_nonunique_dual_face(tmp_path):
    problem = tmp_path / "problem.json"
    atomic_json(problem, fixture_source())
    reg = tmp_path / "registration.json"
    atomic_json(reg, {"unit_T2": True})
    p = {
        "problem": str(problem),
        "problem_sha256": file_hash(problem),
        "design_registration": str(reg),
        "design_registration_sha256": file_hash(reg),
        "tolerance": 1e-10,
        "max_nodes": 100000,
    }
    result = study.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
    assert result["root_native_value_USD"] == 81.75 and result["DLP_bound_pass"]
    assert result["root_LP_USD"] == 96 and result["unique_vector_agreement"] is None
    assert result["simulation_cohorts"] == 0 and result["full_V4_pass"] is None
    assert all(
        file_hash(tmp_path / n) == h for n, h in result["artifact_hashes"].items()
    )


def test_published_settled_study_adapter_two_new_cases(tmp_path):
    if os.environ.get("FIXED_FARE_STAGE"):
        pytest.skip(
            "published entrypoint is exercised in default subset after publication"
        )
    import subprocess

    from airline_market.campaign.runner import THREAD_ENV

    problems = []
    for increment in [2, 4]:
        source = fixture_source()
        source["case"] = "new_adapter_" + str(increment)
        for p in source["physical_spec"]["products"]:
            p["reference_fare"] += increment
            source["physical_spec"]["pricing"][0]["base"][p["id"]] = p["reference_fare"]
        path = tmp_path / ("problem" + str(increment) + ".json")
        atomic_json(path, source)
        problems.append(path)
    reg = tmp_path / "reg.json"
    atomic_json(reg, {"new_adapter_entrypoint": True})
    cells = [
        {
            "problem": str(path),
            "problem_sha256": file_hash(path),
            "design_registration": str(reg),
            "design_registration_sha256": file_hash(reg),
            "tolerance": 1e-10,
            "max_nodes": 100000,
        }
        for path in problems
    ]
    root = tmp_path / "new_study_root"
    cfg = tmp_path / "campaign.json"
    atomic_json(
        cfg,
        {
            "schema": "airline-campaign/1",
            "id": "new_fixed_fare_study_adapter",
            "task": "study:airline_market.experiments.fixed_fare_study:run",
            "design": "list",
            "cells": cells,
            "workers": 1,
            "out": str(root),
            "log_path": str(tmp_path / "log.md"),
        },
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "airline_market.experiments.settled_study_campaign",
            "--config",
            str(cfg),
        ],
        env={**os.environ, **THREAD_ENV},
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    (tmp_path / "child.stderr.log").write_text(result.stderr)
    assert result.returncode == 0, (
        f"study adapter exited{result.returncode}; see retained stderr"
    )
    assert json.loads((root / "status.json").read_text())["counts"] == {"completed": 2}
    assert not (root / "publication_protocol.json").exists()
    assert len(json.loads((root / "settled_cells.json").read_text())["cell_ids"]) == 2
    for p in root.glob("cells/*/attempt-*/results.json"):
        data = json.loads(p.read_text())
        assert data["fixed_fare_native_ledger_DP_pass"] and data["DLP_bound_pass"]
