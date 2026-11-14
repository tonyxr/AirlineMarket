"""Metadata-only publication checks and new opt-in R0.4 protocol acceptance."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.campaign.runner import THREAD_ENV, aggregate

if os.environ.get("SETTLED_RUNNER_SOURCE"):
    spec = importlib.util.spec_from_file_location(
        "airline_market.campaign._staged_settled_runner",
        os.environ["SETTLED_RUNNER_SOURCE"],
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
else:
    from airline_market.campaign import settled_runner as module


def metadata_fixture(tmp_path):
    digest = "b" * 64
    cell = {
        "cell_id": "fresh_metadata",
        "config_hash": digest,
        "generator": "metadata",
        "task": "smoke",
        "parameters": {"case": 2342},
    }
    atomic_json(tmp_path / "cells.json", [cell])
    directory = tmp_path / "cells" / cell["cell_id"]
    attempt = directory / "attempt-0001"
    result = {
        "status": "completed",
        "config_hash": digest,
        "value": 17,
        "parameters": cell["parameters"],
        "peak_rss_bytes": 10,
    }
    atomic_json(attempt / "results.json", result)
    receipt = {
        "config_hash": digest,
        "status": "completed",
        "results_sha256": file_hash(attempt / "results.json"),
    }
    atomic_json(attempt / "COMPLETE.json", receipt)
    return cell, directory, attempt, result, receipt


def test_inflight_pair_is_not_read_and_settled_output_matches_native(tmp_path):
    cell, _, attempt, result, receipt = metadata_fixture(tmp_path)
    atomic_json(attempt / "results.json", {**result, "peak_rss_bytes": 20})
    with pytest.raises(ValueError, match="completed result is damaged"):
        aggregate(tmp_path)
    assert module.aggregate_settled(tmp_path, []) == []
    # The publisher finishes; caller may now acknowledge the future.
    atomic_json(
        attempt / "COMPLETE.json",
        {**receipt, "results_sha256": file_hash(attempt / "results.json")},
    )
    expected = aggregate(tmp_path)
    expected_bytes = {
        name: (tmp_path / name).read_bytes()
        for name in ["cells.csv", "cells.parquet", "resources.csv"]
    }
    assert module.aggregate_settled(tmp_path, [cell["cell_id"]]) == expected
    assert all(
        (tmp_path / name).read_bytes() == content
        for name, content in expected_bytes.items()
    )


@pytest.mark.parametrize("damage", ["hash", "config", "status", "missing_result"])
def test_stable_damage_is_never_suppressed(tmp_path, damage):
    cell, directory, attempt, result, receipt = metadata_fixture(tmp_path)
    if damage == "hash":
        atomic_json(attempt / "results.json", {**result, "value": 99})
    elif damage == "missing_result":
        # Simulate missing data without deleting the preserved fixture bytes.
        (attempt / "results.json").rename(attempt / "preserved_results.json")
    else:
        atomic_json(
            attempt / "COMPLETE.json",
            {**receipt, damage if damage == "status" else "config_hash": "wrong"},
        )
    with pytest.raises((ValueError, FileNotFoundError)):
        module.completed_snapshot(directory, cell["config_hash"])
    with pytest.raises((ValueError, FileNotFoundError)):
        module.aggregate_settled(tmp_path, [cell["cell_id"]])


def test_hash_and_decoding_use_one_result_buffer(tmp_path, monkeypatch):
    cell, directory, attempt, result, _ = metadata_fixture(tmp_path)
    read = Path.read_bytes
    counts = []

    def replace_after_read(path):
        contents = read(path)
        if path == attempt / "results.json":
            counts.append(path)
            atomic_json(path, {**result, "value": 123})
        return contents

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    assert module.completed_snapshot(directory, cell["config_hash"])[1] == result
    assert len(counts) == 1


def test_unknown_settled_id_is_an_error(tmp_path):
    metadata_fixture(tmp_path)
    with pytest.raises(ValueError, match="unknown settled"):
        module.aggregate_settled(tmp_path, ["not_a_cell"])


def test_legacy_root_cannot_be_adopted(tmp_path):
    config = {
        "schema": "airline-campaign/1",
        "id": "new_protocol_only",
        "task": "smoke",
    }
    atomic_json(tmp_path / "config.json", config)
    with pytest.raises(ValueError, match="new settled-protocol output"):
        module.SettledCampaignRunner(config, tmp_path).run()


def invoke_recorded(command, label, directory):
    """Retain child stderr in local fixtures; never print a subprocess env repr."""
    result = subprocess.run(
        command,
        env={**os.environ, **THREAD_ENV},
        capture_output=True,
        text=True,
        check=False,
        timeout=150,
    )
    (directory / (label + ".stdout.log")).write_text(result.stdout)
    (directory / (label + ".stderr.log")).write_text(result.stderr)
    atomic_json(
        directory / (label + ".execution.json"),
        {
            "returncode": result.returncode,
            "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
            "environment_mapping_serialized_by_harness": False,
            "raw_child_streams_retained_locally": True,
        },
    )
    assert result.returncode == 0, (
        f"child exit {result.returncode}; stderr retained at {directory / (label + '.stderr.log')}"
    )
    return result


def test_new_settled_twenty_cell_kill_resume_with_stderr_retained(tmp_path):
    """New revision/config/output; metadata smoke only, no old acceptance replay."""
    config = {
        "schema": "airline-campaign/1",
        "id": "R0_4_settled_protocol_2342_v1",
        "implementation_revision": "returned_future_single_read_v1",
        "task": "smoke",
        "workers": 2,
        "factors": {"x": list(range(23420, 23440))},
        "fixed": {"delay": 0.09},
        "limits": {"seconds": 60},
        "log_path": str(tmp_path / "new_protocol_log.md"),
    }
    path = tmp_path / "config.json"
    atomic_json(path, config)
    reference, resumed = tmp_path / "new_reference", tmp_path / "new_interrupted"

    def command(out):
        if os.environ.get("SETTLED_RUNNER_SOURCE"):
            bootstrap = (
                "import importlib.util; "
                "s=importlib.util.spec_from_file_location('airline_market.campaign._staged_cli', "
                + repr(os.environ["SETTLED_RUNNER_SOURCE"])
                + "); "
                "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.main()"
            )
            return [sys.executable, "-c", bootstrap, str(path), "--out", str(out)]
        return [
            sys.executable,
            "-m",
            "airline_market.campaign.settled_runner",
            str(path),
            "--out",
            str(out),
        ]

    invoke_recorded(command(reference), "reference", tmp_path)
    stderr = tmp_path / "interrupted.stderr.log"
    stdout = tmp_path / "interrupted.stdout.log"
    with stdout.open("w") as out, stderr.open("w") as err:
        process = subprocess.Popen(
            command(resumed), env={**os.environ, **THREAD_ENV}, stdout=out, stderr=err
        )
        deadline = time.monotonic() + 80
        try:
            marker = resumed / "settled_cells.json"
            while (
                not marker.exists()
                or len(json.loads(marker.read_bytes())["cell_ids"]) < 4
            ):
                assert process.poll() is None, (
                    f"early exit {process.returncode}; stderr at {stderr}"
                )
                assert time.monotonic() < deadline
                time.sleep(0.05)
            settled = json.loads(marker.read_bytes())["cell_ids"]
            receipts = {
                str(p.relative_to(resumed)): p.read_bytes()
                for cell_id in settled
                for p in (resumed / "cells" / cell_id).glob("attempt-*/COMPLETE.json")
            }
            process.kill()
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    invoke_recorded(command(resumed), "resumed", tmp_path)
    for relative, contents in receipts.items():
        assert (resumed / relative).read_bytes() == contents
    for name in ["cells.csv", "cells.parquet"]:
        assert (reference / name).read_bytes() == (resumed / name).read_bytes()
    assert json.loads((resumed / "status.json").read_bytes())["counts"] == {
        "completed": 20
    }
    assert (
        len(json.loads((resumed / "settled_cells.json").read_bytes())["cell_ids"]) == 20
    )
