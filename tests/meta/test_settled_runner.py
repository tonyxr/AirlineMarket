"""W4/R0.4 immutable-result readers, resources, and real kill-resume checks."""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from airline_market.campaign import runner as module
from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.campaign.runner import THREAD_ENV


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


def test_resources_change_without_rewriting_successful_result(tmp_path):
    cell, directory, attempt, result, _receipt = metadata_fixture(tmp_path)
    before = {
        name: (attempt / name).read_bytes()
        for name in ("results.json", "COMPLETE.json")
    }
    atomic_json(attempt / "resources.json", {"peak_rss_bytes": 20})
    assert module.completed_attempt(directory, cell["config_hash"])[1] == result
    assert module.aggregate(tmp_path)[0]["value"] == 17
    assert all(
        (attempt / name).read_bytes() == contents for name, contents in before.items()
    )
    import csv

    with (tmp_path / "resources.csv").open() as stream:
        assert next(csv.DictReader(stream))["peak_rss_bytes"] == "20"


def test_coordinator_never_rewrites_a_published_result(tmp_path, monkeypatch):
    writes = []
    publish = module.atomic_json

    def watch(path, value):
        path = Path(path)
        if path.name in ("results.json", "COMPLETE.json"):
            assert not path.exists(), "coordinator rewrote an already published result"
        if path.name == "resources.json":
            before = {
                name: path.with_name(name).read_bytes()
                for name in ("results.json", "COMPLETE.json")
            }
            # Read during the coordinator's resource publication, before its future returns.
            assert module.aggregate(tmp_path)[0]["answer"] == 0
            publish(path, value)
            assert all(
                path.with_name(name).read_bytes() == data
                for name, data in before.items()
            )
            writes.append(value)
            return
        publish(path, value)

    monkeypatch.setattr(module, "atomic_json", watch)
    config = {
        "schema": "airline-campaign/1",
        "id": "W4_R04_immutable_result",
        "task": "smoke",
        "log_path": str(tmp_path / "log.md"),
    }
    assert module.CampaignRunner(config, tmp_path).run()["counts"] == {"completed": 1}
    assert len(writes) == 1 and writes[0]["peak_rss_bytes"] > 0


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
        module.completed_attempt(directory, cell["config_hash"])
    with pytest.raises((ValueError, FileNotFoundError)):
        module.aggregate(tmp_path)


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
    assert module.completed_attempt(directory, cell["config_hash"])[1] == result
    assert len(counts) == 1


def test_resume_rejects_a_different_campaign_config(tmp_path):
    config = {
        "schema": "airline-campaign/1",
        "id": "W4_config_identity",
        "task": "smoke",
    }
    atomic_json(tmp_path / "config.json", {**config, "id": "different"})
    with pytest.raises(ValueError, match="different campaign config"):
        module.CampaignRunner(config, tmp_path).run()


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
    """Exercise the canonical runner with retained stderr across kill and resume."""
    config = {
        "schema": "airline-campaign/1",
        "id": "W4_R04_canonical_2342",
        "implementation_revision": "immutable_results_separate_resources",
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
        return [
            sys.executable,
            "-m",
            "airline_market",
            "campaign",
            "run",
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
