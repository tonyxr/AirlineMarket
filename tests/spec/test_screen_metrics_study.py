"""S1/R4: new resource-bound saved-policy adapter, including stable failures."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.campaign.worker import CellContext

if os.environ.get("S1_METRICS_STUDY_SOURCE"):
    spec = importlib.util.spec_from_file_location(
        "staged_metrics_study", os.environ["S1_METRICS_STUDY_SOURCE"]
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
else:
    from airline_market.experiments import screen_metrics_study as module

ARMS = ["direct", "linked", "unrelated", "oracle", "full_state"]


def metadata(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    source = root / "physical.json"
    source.write_text('{"local":"fixture"}\n')
    atomic_json(root / "config.json", {"plan_id": "S1", "fixture": "new_adapter"})
    cells = []
    for arm in ARMS:
        p = {
            "setting_id": "new_fixture",
            "arm": arm,
            "source_spec": str(source),
            "source_spec_sha256": file_hash(source),
        }
        cell = {
            "cell_id": arm,
            "config_hash": hashlib.sha256(arm.encode()).hexdigest(),
            "parameters": p,
        }
        a = root / "cells" / arm / "attempt-0001"
        atomic_json(
            a / "results.json",
            {"status": "solved", "parameters": p, "config_hash": cell["config_hash"]},
        )
        atomic_json(
            a / "COMPLETE.json",
            {
                "status": "solved",
                "config_hash": cell["config_hash"],
                "results_sha256": file_hash(a / "results.json"),
            },
        )
        (a / "optimal_policy.pkl.gz").write_bytes(b"metadata-only-pinning-fixture")
        cells.append(cell)
    atomic_json(root / "cells.json", cells)
    atomic_json(root / "settled_cells.json", {"cell_ids": ARMS})
    return root, cells


def task(parameters):
    return {"parameters": parameters, "limits": {"rss_gb": 0.5, "reservation_gb": 0.5}}


def test_pinning_requires_all_five_returned_publishers_and_solved_results(tmp_path):
    root, cells = metadata(tmp_path)
    p = module.pin_setting(root, "new_fixture")
    assert len(p["pinned_files"]) == 18
    assert p["reservation_gb"] == 0.5 and "post-completion" in p["policy_hash_scope"]
    atomic_json(root / "settled_cells.json", {"cell_ids": ARMS[:-1]})
    with pytest.raises(ValueError, match="acknowledged"):
        module.pin_setting(root, "new_fixture")
    atomic_json(root / "settled_cells.json", {"cell_ids": ARMS})
    a = root / "cells" / "linked" / "attempt-0001"
    result = json.loads((a / "results.json").read_text())
    result.update(status="memory_limit", value=None)
    atomic_json(a / "results.json", result)
    atomic_json(
        a / "COMPLETE.json",
        {
            "status": "memory_limit",
            "config_hash": cells[1]["config_hash"],
            "results_sha256": file_hash(a / "results.json"),
        },
    )
    with pytest.raises(ValueError, match="solved"):
        module.pin_setting(root, "new_fixture")


def test_wrapper_preserves_pinned_inputs_and_seals_output_without_native_calls(
    tmp_path, monkeypatch
):
    root, _ = metadata(tmp_path)
    p = module.pin_setting(root, "new_fixture")
    output = tmp_path / "output"
    output.mkdir()
    calls = []

    def saved_analysis(source, setting):
        calls.append((source, setting))
        return {
            "plan_id": "S1",
            "pairwise_information": {},
            "policy_divergence": {"first_period": None},
        }

    monkeypatch.setattr(module, "analyze_settled_setting", saved_analysis)
    context = CellContext(output)
    before = {s["path"]: file_hash(s["path"]) for s in p["pinned_files"]}
    result = module.run(task(p), context)
    assert calls == [(str(root.resolve()), "new_fixture")]
    assert result["status"] == "completed" and result["value"] is None
    assert result["metrics_sha256"] == file_hash(output / result["metrics_path"])
    assert all(file_hash(path) == h for path, h in before.items())
    assert context.counters()["stage"] == "complete"
    assert (
        result["sampled_physical_cohorts"]
        == result["bellman_optimizations"]
        == result["new_rng_calls"]
        == 0
    )


@pytest.mark.parametrize("when", ["before", "after"])
def test_stable_input_change_fails_without_repair_or_output(
    tmp_path, monkeypatch, when
):
    root, _ = metadata(tmp_path)
    p = module.pin_setting(root, "new_fixture")
    output = tmp_path / "output"
    output.mkdir()
    source = Path(p["pinned_files"][0]["path"])

    def corrupt(*args):
        source.write_bytes(source.read_bytes() + b"\n")
        return {}

    monkeypatch.setattr(module, "analyze_settled_setting", corrupt)
    if when == "before":
        source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="input changed"):
        module.run(task(p), CellContext(output))
    assert not (output / "saved_policy_metrics.json").exists()


@pytest.mark.parametrize("field", ["rss_gb", "reservation_gb"])
def test_caller_cannot_drop_memory_controls(tmp_path, field):
    root, _ = metadata(tmp_path)
    cell = task(module.pin_setting(root, "new_fixture"))
    cell["limits"][field] = 1
    with pytest.raises(ValueError, match="0.5 GiB"):
        module.run(cell, CellContext(tmp_path))


@pytest.mark.skipif(
    bool(os.environ.get("S1_METRICS_STUDY_SOURCE")),
    reason="Published adapter integration runs once in default subset",
)
def test_new_saved_metrics_adapter_and_original_rss_guard(tmp_path, monkeypatch):
    from airline_market.campaign import runner
    from airline_market.campaign.settled_runner import SettledCampaignRunner
    from airline_market.experiments.screen_study import run as produce
    from airline_market.experiments.study_campaign import StudyProcesses
    from airline_market.pipeline.screen_generators import screen_spec
    from airline_market.spec import dump_spec

    root = tmp_path / "native_source"
    root.mkdir()
    source = root / "physical.json"
    dump_spec(
        screen_spec(
            {
                "T": 2,
                "actions": 2,
                "n_spokes": 1,
                "capacity": 3,
                "capacity_support": [1, 2],
                "price_scale": 1.343,
                "outside_utility_shift": 0.241,
                "arrival_probability": 0.684,
            }
        ),
        source,
    )
    atomic_json(
        root / "config.json", {"plan_id": "S1", "fixture": "new_resource_adapter"}
    )
    cells = []
    for arm in ARMS:
        a = root / "cells" / arm / "attempt-0001"
        a.mkdir(parents=True)
        p = {
            "setting_id": "resource_fixture",
            "arm": arm,
            "source_spec": str(source),
            "source_spec_sha256": file_hash(source),
        }
        cell = {
            "cell_id": arm,
            "config_hash": hashlib.sha256((arm + "resource").encode()).hexdigest(),
            "parameters": p,
            "limits": {"max_nodes": 100000},
        }
        result = {
            "config_hash": cell["config_hash"],
            "parameters": p,
            **produce(cell, CellContext(a)),
        }
        atomic_json(a / "results.json", result)
        atomic_json(
            a / "COMPLETE.json",
            {
                "config_hash": cell["config_hash"],
                "status": result["status"],
                "results_sha256": file_hash(a / "results.json"),
            },
        )
        cells.append(cell)
    atomic_json(root / "cells.json", cells)
    atomic_json(root / "settled_cells.json", {"cell_ids": ARMS})
    parameters = module.pin_setting(root, "resource_fixture")
    cfg = {
        "schema": "airline-campaign/1",
        "id": "S1_new_saved_metrics_adapter",
        "plan_id": "S1/R4",
        "phase": "S",
        "generator": "saved_policy",
        "task": "study:airline_market.experiments.screen_metrics_study:run",
        "implementation_revision": "metrics_adapter_v1",
        "design": "list",
        "cells": [parameters],
        "workers": 1,
        "memory_budget_gb": 0.5,
        "limits": {"seconds": 120, "rss_gb": 0.5, "reservation_gb": 0.5},
        "log_path": str(tmp_path / "log.md"),
    }
    monkeypatch.setattr(runner, "subprocess", StudyProcesses())
    out = tmp_path / "analysis"
    SettledCampaignRunner(cfg, out).run()
    ack = json.loads((out / "settled_cells.json").read_text())
    assert len(ack["cell_ids"]) == 1
    a = next((out / "cells").glob("*/attempt-0001"))
    r = json.loads((a / "results.json").read_text())
    assert r["status"] == "completed"
    metrics = json.loads((a / r["metrics_path"]).read_text())
    assert len(metrics["pairwise_information"]) == 5
    assert metrics["policy_divergence"]["total_mass"] == pytest.approx(1, abs=1e-12)
    assert file_hash(a / r["metrics_path"]) == r["metrics_sha256"]
    # A different metadata-only fixture verifies the actual parent RSS branch,
    # with a deliberately tiny 1 MiB limit, not a scientific memory conclusion.
    guard = {
        **cfg,
        "id": "S1_new_RSS_guard_fixture",
        "task": "smoke",
        "implementation_revision": "guard_metadata_only",
        "cells": [{"plan_id": "S1/R4", "x": 77, "delay": 3, "reservation_gb": 0.5}],
        "limits": {"seconds": 30, "rss_gb": 0.001, "reservation_gb": 0.5},
    }
    import subprocess

    monkeypatch.setattr(runner, "subprocess", subprocess)
    guard_out = tmp_path / "guard"
    SettledCampaignRunner(guard, guard_out).run()
    a = next((guard_out / "cells").glob("*/attempt-0001"))
    r = json.loads((a / "results.json").read_text())
    assert r["status"] == "memory_limit" and r["value"] is None
    receipt = json.loads((a / "COMPLETE.json").read_text())
    assert receipt["results_sha256"] == file_hash(a / "results.json")
