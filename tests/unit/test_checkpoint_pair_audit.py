"""Reporting-failure preservation, complete-row integrity and zero-rollout audit."""

import importlib.util
import json
import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.experiments import checkpoint_pair_study as source

if os.environ.get("PAIR_AUDIT_STAGE"):
    spec = importlib.util.spec_from_file_location(
        "airline_market.experiments.checkpoint_pair_audit",
        Path(os.environ["PAIR_AUDIT_STAGE"]) / "checkpoint_pair_audit.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
from airline_market.experiments import checkpoint_pair_audit as audit


def saved(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    make = runpy.run_path(str(Path(__file__).with_name("test_checkpoint_pair.py")))[
        "fixture"
    ]
    p, _, _, _ = make(parent)
    directory = tmp_path / "failed_source"
    directory.mkdir()
    result = source.run({"parameters": p}, SimpleNamespace(directory=directory))
    assert (
        "status" not in result
    )  # Preserved old reporting contract defect, not patched.
    atomic_json(
        directory / "config.json", {"parameters": p, "config_hash": "unit-snapshot"}
    )
    atomic_json(
        directory / "results.json",
        {"status": "failed", "value": None, "wall_seconds": 1.0, "peak_rss_bytes": 1},
    )
    atomic_json(
        directory / "COMPLETE.json",
        {
            "status": "failed",
            "config_hash": "unit-snapshot",
            "results_sha256": file_hash(directory / "results.json"),
        },
    )
    (directory / "run.log").write_text("KeyError: 'status'\n")
    atomic_json(directory / "environment.json", {})
    atomic_json(directory / "progress.json", {"completed": 3})
    names = [
        "COMPLETE.json",
        "results.json",
        "config.json",
        "spec.json",
        "paired_summary.json",
        "pairs.jsonl",
        "run.log",
        "environment.json",
        "progress.json",
    ]
    return {
        "source_directory": str(directory),
        "seed_start": p["seed_start"],
        "n_cohorts": 3,
        "source_hashes": {str(directory / x): file_hash(directory / x) for x in names},
    }


def test_readonly_audit_returns_required_worker_status_and_never_runs_policy(
    tmp_path, monkeypatch
):
    p = saved(tmp_path)
    original = dict(p["source_hashes"])

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "read-only audit constructed a policy/environment or loaded weights"
        )

    monkeypatch.setattr(source, "collect", forbidden)
    monkeypatch.setattr(source.FocalCarrierEnv, "__init__", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    output = tmp_path / "derived"
    output.mkdir()
    r = audit.run({"parameters": p}, SimpleNamespace(directory=output))
    assert r["status"] == "completed" and r["source_worker_status"] == "failed"
    assert r["source_worker_value"] is None and r["new_physical_cohorts"] == 0
    assert r["ordered_physical_cohorts"] == 3 and r["profile_measurements"] == 6
    assert r["saved_pairs_audit_sha256"] == file_hash(output / "saved_pairs_audit.json")
    assert all(file_hash(Path(k)) == v for k, v in original.items())


def test_changed_source_bytes_rejected_before_derivation(tmp_path):
    p = saved(tmp_path)
    p["source_hashes"][str(Path(p["source_directory"]) / "pairs.jsonl")] = "wrong"
    with pytest.raises(ValueError, match="snapshot mismatch"):
        audit.read_pairs(p)


def test_complete_source_sequence_cannot_be_reordered(tmp_path):
    p = saved(tmp_path)
    path = Path(p["source_directory"]) / "pairs.jsonl"
    path.write_text("\n".join(reversed(path.read_text().splitlines())) + "\n")
    p["source_hashes"][str(path)] = file_hash(path)
    summary = path.with_name("paired_summary.json")
    q = json.loads(summary.read_text())
    q["pairs_sha256"] = file_hash(path)
    atomic_json(summary, q)
    p["source_hashes"][str(summary)] = file_hash(summary)
    with pytest.raises(ValueError, match="reordered"):
        audit.read_pairs(p)


def test_failure_cannot_be_relabelled_completed(tmp_path):
    p = saved(tmp_path)
    path = Path(p["source_directory"]) / "COMPLETE.json"
    q = json.loads(path.read_text())
    q["status"] = "completed"
    atomic_json(path, q)
    p["source_hashes"][str(path)] = file_hash(path)
    with pytest.raises(ValueError, match="preserved original reporting failure"):
        audit.read_pairs(p)
