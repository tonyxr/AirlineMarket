"""End-to-end study accounting, isolated CLI and sealed-seed checks."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from airline_market.campaign.runner import THREAD_ENV


def run_study(tmp_path, seed=200000):
    config = {
        "schema": "airline-scaling-study/1",
        "id": "F3_test",
        "revision": "reference-v1",
        "out": str(tmp_path / "out"),
        "fixed": {"controller_mode": "external_all"},
        "cells": [{"n_spokes": 2, "n_carriers": 3, "periods": 3}],
        "n_cohorts": 2,
        "n_profile_cohorts": 1,
        "seed_start": seed,
        "limits": {"seconds": 30, "rss_gb": 2},
        "log_path": str(tmp_path / "log.md"),
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    command = [
        sys.executable,
        "-m",
        "airline_market.experiments.scaling_study",
        "--config",
        str(path),
    ]
    subprocess.run(
        command,
        env={**os.environ, **THREAD_ENV},
        capture_output=True,
        text=True,
        check=True,
        timeout=45,
    )
    return (
        config,
        command,
        json.loads((Path(config["out"]) / "results.json").read_text()),
    )


def test_profile_is_excluded_from_timing_and_resume_reuses_receipt(tmp_path):
    config, command, root = run_study(tmp_path)
    result = root["cells"][0]
    assert root["status"] == "finished" and result["status"] == "completed"
    assert result["n_cohorts"] == 2 and result["n_profile_cohorts_excluded"] == 1
    assert result["criterion_status"] == "pilot_only"
    attempt = next(Path(config["out"]).glob("cells/*/attempt-*"))
    rows = json.loads((attempt / "timing_rows.json").read_text())
    assert [r["seed"] for r in rows] == [200000, 200001]
    assert result["mean_seconds_per_cohort"] == pytest.approx(
        sum(r["seconds"] for r in rows) / 2
    )
    assert sum(result["profile"]["exclusive_shares"].values()) == pytest.approx(1)
    before = (attempt / "COMPLETE.json").read_bytes()
    subprocess.run(
        command,
        env={**os.environ, **THREAD_ENV},
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    assert (attempt / "COMPLETE.json").read_bytes() == before
    assert len(list(Path(config["out"]).glob("cells/*/attempt-*"))) == 1


def test_scaling_study_cannot_open_final_seeds(tmp_path):
    _, _, root = run_study(tmp_path, 300000)
    assert root["status"] == "finished_with_failures"
    assert root["cells"][0]["status"] == "failed"
    assert "validation cohorts only" in root["cells"][0]["reason"]
