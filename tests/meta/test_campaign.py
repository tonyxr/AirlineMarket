"""Campaign acceptance checks: real interruption/resume, hashes and limits."""

import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from airline_market.bench.harness import compare_baseline
from airline_market.campaign.design import expand_campaign, fractional_design
from airline_market.campaign.runner import THREAD_ENV


def invoke(config, directory):
    return [
        sys.executable,
        "-m",
        "airline_market",
        "campaign",
        "run",
        str(config),
        "--out",
        str(directory),
    ]


def test_resolution_four_design_has_no_main_two_factor_alias():
    factors = {str(k): [-1, 1] for k in range(7)}
    design = fractional_design(factors)
    x = np.asarray([[row[name] for name in factors] for row in design])
    assert x.shape == (16, 7)
    assert np.array_equal(x.T @ x, 16 * np.eye(7))
    for a in range(7):
        for b in range(a):
            assert np.array_equal(x.T @ (x[:, a] * x[:, b]), np.zeros(7))


def test_full_list_hashes_and_nested_action_grids():
    from airline_market.campaign.instances import grid_indices

    cfg = {
        "schema": "airline-campaign/1",
        "id": "test",
        "factors": {"a": [1, 2], "b": [3, 4]},
        "fixed": {"x": 9},
    }
    cells = expand_campaign(cfg)
    listed = expand_campaign(
        {**cfg, "design": "list", "cells": [row["parameters"] for row in cells]}
    )
    assert [row["config_hash"] for row in cells] == [
        row["config_hash"] for row in listed
    ]
    assert len({row["config_hash"] for row in cells}) == 4
    assert (
        set(grid_indices(2))
        < set(grid_indices(4))
        < set(grid_indices(6))
        < set(grid_indices(9))
        < set(grid_indices(18))
    )


@pytest.mark.parametrize("repetition", [1, 2, 3])
def test_twenty_cell_kill_resume_is_byte_identical(tmp_path, repetition):
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    cfg = {
        "schema": "airline-campaign/1",
        "id": "W4_R0.4_acceptance_" + str(repetition),
        "task": "smoke",
        "workers": 2,
        "factors": {"x": list(range(20))},
        "fixed": {"delay": 0.15},
        "limits": {"seconds": 60},
        "log_path": str(tmp_path / "acceptance_log.md"),
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    reference, resumed = tmp_path / "reference", tmp_path / "resumed"
    env = {**os.environ, **THREAD_ENV}
    subprocess.run(
        invoke(path, reference),
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    process = subprocess.Popen(
        invoke(path, resumed),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 60
    try:
        while len(list(resumed.glob("cells/*/attempt-*/COMPLETE.json"))) < 3:
            if process.poll() is not None:
                raise AssertionError(process.stderr.read())
            assert time.monotonic() < deadline, "campaign did not make progress"
            time.sleep(0.1)
        receipts = {
            str(p.relative_to(resumed)): p.read_bytes()
            for p in resumed.glob("cells/*/attempt-*/COMPLETE.json")
        }
        process.kill()
        process.wait(timeout=10)
        subprocess.run(
            invoke(path, resumed),
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if process.stderr:
            process.stderr.close()
    for relative, content in receipts.items():
        assert (resumed / relative).read_bytes() == content
    for name in ["cells.csv", "cells.parquet"]:
        assert (reference / name).read_bytes() == (resumed / name).read_bytes()
    status = json.loads((resumed / "status.json").read_text())
    assert status["counts"] == {"completed": 20}


def test_timeout_has_no_fabricated_value(tmp_path):
    cfg = {
        "schema": "airline-campaign/1",
        "id": "timeout_test",
        "task": "smoke",
        "workers": 1,
        "fixed": {"delay": 10},
        "limits": {"seconds": 0.5},
        "log_path": str(tmp_path / "timeout_log.md"),
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    output = tmp_path / "run"
    subprocess.run(
        invoke(path, output),
        env={**os.environ, **THREAD_ENV},
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    result = json.loads(next(output.glob("cells/*/attempt-*/results.json")).read_text())
    assert result["status"] == "time_limit"
    assert result["value"] is None


def test_performance_regression_detects_synthetic_slowdown():
    baseline = {
        "machine_id": "same",
        "benchmarks": [{"operation": "step", "median_seconds": 1.0}],
    }
    current = {
        "machine_id": "same",
        "benchmarks": [{"operation": "step", "median_seconds": 1.26}],
    }
    assert compare_baseline(current, baseline) == [
        {"operation": "step", "slowdown_ratio": 1.26}
    ]
    with pytest.raises(ValueError, match="same machine"):
        compare_baseline({**current, "machine_id": "other"}, baseline)


def test_timeout_stops_nested_worker(tmp_path):
    psutil = pytest.importorskip("psutil")
    cfg = {
        "schema": "airline-campaign/1",
        "id": "nested_timeout",
        "task": "smoke",
        "workers": 1,
        "fixed": {"spawn_child": True},
        "limits": {"seconds": 8},
        "log_path": str(tmp_path / "log.md"),
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    output = tmp_path / "run"
    subprocess.run(
        invoke(path, output),
        env={**os.environ, **THREAD_ENV},
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    child = int(next(output.glob("cells/*/attempt-*/child_pid.json")).read_text())
    deadline = time.monotonic() + 5
    while (
        psutil.pid_exists(child)
        and psutil.Process(child).status() != psutil.STATUS_ZOMBIE
    ):
        assert time.monotonic() < deadline, "nested worker survived the cell timeout"
        time.sleep(0.1)
