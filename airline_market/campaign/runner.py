"""Bounded subprocess campaigns with atomic resume and scientific aggregates."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path
import csv
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time

from ..core.hashing import canonical_json, config_hash
from .artifacts import atomic_json, environment, file_hash, log_run, utc_now
from .design import expand_campaign, load_campaign


TERMINAL = {
    "solved",
    "completed",
    "unsupported",
    "node_limit",
    "time_limit",
    "memory_limit",
    "failed",
}
THREAD_ENV = {
    key: "1"
    for key in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
}


def completed_attempt(cell_directory: Path, digest: str) -> tuple[Path, dict] | None:
    """Trust a completed cell only after checking its config and result digest."""
    for attempt in sorted(cell_directory.glob("attempt-*"), reverse=True):
        receipt_path = attempt / "COMPLETE.json"
        if not receipt_path.exists():
            continue
        receipt = json.loads(receipt_path.read_text())
        if receipt["config_hash"] != digest:
            raise ValueError(f"cell config mismatch: {attempt}")
        if file_hash(attempt / "results.json") != receipt["results_sha256"]:
            raise ValueError(f"completed result is damaged: {attempt}")
        return attempt, json.loads((attempt / "results.json").read_text())
    return None


def aggregate(directory: str | Path) -> list[dict]:
    """Write byte-stable scientific CSV/parquet; volatile resources are separate.

    Timing, host and attempt identifiers vary across reruns and therefore live
    in resources.csv and the full cell results, not the resume identity table.
    """
    import pandas as pd

    root = Path(directory)
    cells = json.loads((root / "cells.json").read_text())
    rows, resources = [], []
    omitted = {
        "completed_at",
        "wall_seconds",
        "cpu_seconds",
        "peak_rss_bytes",
        "solve_seconds",
        "ms_per_node",
        "measurement_note",
        "error_type",
        "reason",
    }
    for cell in cells:
        found = completed_attempt(root / "cells" / cell["cell_id"], cell["config_hash"])
        if not found:
            continue
        attempt, result = found
        row = {
            "cell_id": cell["cell_id"],
            "config_hash": cell["config_hash"],
            "generator": cell["generator"],
            "task": cell["task"],
        }
        row.update(
            {
                "parameter/" + key: canonical_json(value)
                if isinstance(value, (dict, list))
                else value
                for key, value in cell["parameters"].items()
            },
        )
        row.update(
            {
                key: canonical_json(value) if isinstance(value, (dict, list)) else value
                for key, value in result.items()
                if key not in omitted | {"config_hash", "parameters"}
            }
        )
        rows.append(row)
        resources.append(
            {
                "cell_id": cell["cell_id"],
                "attempt": attempt.name,
                **{
                    key: result.get(key)
                    for key in [
                        "wall_seconds",
                        "cpu_seconds",
                        "peak_rss_bytes",
                        "solve_seconds",
                        "ms_per_node",
                        "reason",
                    ]
                },
            }
        )
    rows.sort(key=lambda row: row["cell_id"])
    for name, records in [
        ("cells", rows),
        ("resources", sorted(resources, key=lambda row: row["cell_id"])),
    ]:
        frame = pd.DataFrame(records)
        frame = frame.reindex(sorted(frame.columns), axis=1)
        temporary = root / (name + ".csv.tmp")
        frame.to_csv(temporary, index=False, float_format="%.17g", lineterminator="\n")
        temporary.replace(root / (name + ".csv"))
        if name == "cells":
            temporary = root / "cells.parquet.tmp"
            frame.to_parquet(temporary, index=False)
            temporary.replace(root / "cells.parquet")
    return rows


def campaign_status(directory: str | Path) -> dict:
    """Read durable progress, including live per-cell counters when present."""
    root = Path(directory)
    state = (
        json.loads((root / "status.json").read_text())
        if (root / "status.json").exists()
        else {"status": "not_started"}
    )
    active = []
    for path in root.glob("cells/*/attempt-*/status.json"):
        value = json.loads(path.read_text())
        if value.get("status") == "running":
            progress = path.with_name("progress.json")
            active.append(
                {
                    **value,
                    "directory": str(path.parent),
                    "progress": json.loads(progress.read_text())
                    if progress.exists()
                    else None,
                }
            )
    return {**state, "active_cells": active}


class CampaignRunner:
    """Own one campaign coordinator and its explicitly bounded cell processes."""

    def __init__(self, config: dict, directory: Path, *, retry_failed: bool = False):
        """Resolve cells and prepare memory-aware coordination without launching."""
        self.config, self.directory = config, directory
        self.cells = expand_campaign(config)
        self.retry_failed = retry_failed
        self.stop = threading.Event()
        self.condition = threading.Condition()
        self.reserved = 0.0
        self.memory_queue = []
        self.memory_budget = float(config.get("memory_budget_gb", 12))
        self.states = {}
        self.children = {}

    def terminate(self, process: subprocess.Popen) -> None:
        """Stop only this coordinator's own cell after interruption or a limit."""
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                return
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    def run_cell(self, cell: dict) -> dict:
        """Execute or reuse one hashed cell and preserve every attempt."""
        import psutil

        directory = self.directory / "cells" / cell["cell_id"]
        directory.mkdir(parents=True, exist_ok=True)
        found = completed_attempt(directory, cell["config_hash"])
        if found and not (self.retry_failed and found[1]["status"] == "failed"):
            self.states[cell["cell_id"]] = found[1]["status"]
            return found[1]
        reservation = min(
            self.memory_budget,
            float(
                cell["parameters"].get(
                    "reservation_gb", cell["limits"].get("reservation_gb", 1)
                )
            ),
        )
        with self.condition:
            self.memory_queue.append(cell["cell_id"])
            while (
                self.memory_queue[0] != cell["cell_id"]
                or self.reserved + reservation > self.memory_budget
            ) and not self.stop.is_set():
                self.condition.wait(timeout=1)
            if self.stop.is_set():
                self.memory_queue.remove(cell["cell_id"])
                return {"status": "interrupted"}
            self.memory_queue.pop(0)
            self.reserved += reservation
            self.condition.notify_all()
        try:
            return self._run_cell(cell, directory, psutil)
        finally:
            with self.condition:
                self.reserved -= reservation
                self.condition.notify_all()

    def _run_cell(self, cell: dict, directory: Path, psutil) -> dict:
        """Watch an isolated child for process exit, wall time and actual RSS."""
        # A killed coordinator's child exits when its parent-watch heartbeat runs.
        for previous in directory.glob("attempt-*/status.json"):
            record = json.loads(previous.read_text())
            if record.get("status") != "running":
                continue
            pid = record.get("pid")
            for _ in range(35):
                if not pid or not psutil.pid_exists(pid):
                    break
                old = psutil.Process(pid)
                if abs(old.create_time() - record.get("process_created_at", 0)) > 0.1:
                    break
                time.sleep(0.2)
            else:
                raise RuntimeError(
                    f"matching cell still active at PID {pid}; refusing a duplicate"
                )
            atomic_json(
                previous, {**record, "status": "interrupted", "observed_at": utc_now()}
            )
        attempt = (
            directory / f"attempt-{len(list(directory.glob('attempt-*'))) + 1:04d}"
        )
        attempt.mkdir()
        atomic_json(attempt / "config.json", cell)
        started = time.monotonic()
        with (attempt / "run.log").open("w") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "airline_market.campaign.worker",
                    str(attempt.resolve()),
                    "--parent",
                    str(os.getpid()),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                env={**os.environ, **THREAD_ENV},
                start_new_session=True,
            )
            self.children[cell["cell_id"]] = process
            self.states[cell["cell_id"]] = "running"
            atomic_json(
                attempt / "status.json",
                {
                    "status": "running",
                    "pid": process.pid,
                    "process_created_at": psutil.Process(process.pid).create_time(),
                    "started_at": utc_now(),
                    "cell_id": cell["cell_id"],
                },
            )
            limit_seconds = float(cell["limits"].get("seconds", 14400))
            rss_limit = float(cell["limits"].get("rss_gb", 12)) * 1024**3
            peak_rss = 0
            status = None
            while process.poll() is None:
                elapsed = time.monotonic() - started
                try:
                    observed = psutil.Process(process.pid)
                    rss = observed.memory_info().rss
                    progress_path = attempt / "progress.json"
                    if progress_path.exists():
                        progress = json.loads(progress_path.read_text())
                        rss += sum(
                            child["rss_bytes"]
                            for child in progress.get("owned_children", [])
                        )
                    peak_rss = max(peak_rss, rss)
                except psutil.NoSuchProcess:
                    pass
                if self.stop.is_set():
                    status = "interrupted"
                elif elapsed >= limit_seconds:
                    status = "time_limit"
                elif peak_rss > rss_limit:
                    status = "memory_limit"
                if status:
                    self.terminate(process)
                    break
                time.sleep(0.25)
        self.children.pop(cell["cell_id"], None)
        found = completed_attempt(directory, cell["config_hash"])
        if found and found[0] == attempt:
            result = found[1]
            result["peak_rss_bytes"] = max(peak_rss, result.get("peak_rss_bytes", 0))
            atomic_json(attempt / "results.json", result)
            atomic_json(
                attempt / "COMPLETE.json",
                {
                    "config_hash": cell["config_hash"],
                    "status": result["status"],
                    "results_sha256": file_hash(attempt / "results.json"),
                },
            )
        else:
            result = {
                "status": status or "failed",
                "config_hash": cell["config_hash"],
                "parameters": cell["parameters"],
                "phase": cell["phase"],
                "value": None,
                "wall_seconds": time.monotonic() - started,
                "peak_rss_bytes": peak_rss,
                "exit_code": process.returncode,
                "reason": status or "worker exited without a completed receipt",
            }
            atomic_json(attempt / "results.json", result)
            if result["status"] != "interrupted":
                atomic_json(
                    attempt / "COMPLETE.json",
                    {
                        "config_hash": cell["config_hash"],
                        "status": result["status"],
                        "results_sha256": file_hash(attempt / "results.json"),
                    },
                )
        atomic_json(
            attempt / "status.json",
            {
                "status": result["status"],
                "exit_code": process.returncode,
                "completed_at": utc_now(),
                "cell_id": cell["cell_id"],
            },
        )
        self.states[cell["cell_id"]] = result["status"]
        log_run(
            {
                "date": utc_now(),
                "id": self.config["id"],
                "run_id": str(attempt),
                "git_sha": self.git_sha,
                "status": result["status"],
                "seconds": round(result["wall_seconds"], 3),
                "headline": "value=" + str(result.get("value")),
                "deviations": result.get("reason", ""),
            },
            path=self.config.get("log_path", "EXPERIMENT_LOG.md"),
        )
        return result

    def snapshot(self, status: str) -> None:
        """Publish campaign counts and the identities of active cells."""
        counts = dict(Counter(self.states.copy().values()))
        atomic_json(
            self.directory / "status.json",
            {
                "schema": "airline-campaign-status/1",
                "id": self.config["id"],
                "status": status,
                "updated_at": utc_now(),
                "coordinator_pid": os.getpid(),
                "cells_total": len(self.cells),
                "counts": counts,
                "unstarted": len(self.cells) - len(self.states),
                "reserved_gb": self.reserved,
                "memory_budget_gb": self.memory_budget,
            },
        )

    def run(self) -> dict:
        """Run a resumable campaign under one exclusive coordinator lock."""
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".coordinator.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "campaign already has an active coordinator"
                ) from error
            existing = self.directory / "config.json"
            if existing.exists() and json.loads(existing.read_text()) != self.config:
                raise ValueError(
                    "resume directory belongs to a different campaign config"
                )
            provenance = environment()
            previous_environment = self.directory / "environment.json"
            if previous_environment.exists():
                previous_sources = json.loads(previous_environment.read_text()).get(
                    "source_hashes", {}
                )
                changed = [
                    name
                    for name, digest in previous_sources.items()
                    if name in provenance["source_hashes"]
                    and provenance["source_hashes"][name] != digest
                ]
                if changed:
                    raise ValueError(
                        "implementation changed since this campaign started; use a new implementation_revision and output directory: "
                        + ", ".join(changed)
                    )
            atomic_json(existing, self.config)
            atomic_json(self.directory / "cells.json", self.cells)
            self.git_sha = provenance["git_sha"]
            atomic_json(self.directory / "environment.json", provenance)
            previous = {}

            def interrupt(signum, frame):
                self.stop.set()

            for signum in [signal.SIGTERM, signal.SIGINT]:
                previous[signum] = signal.signal(signum, interrupt)
            self.snapshot("running")
            try:
                with ThreadPoolExecutor(
                    max_workers=int(self.config.get("workers", 1))
                ) as pool:
                    pending = {pool.submit(self.run_cell, cell) for cell in self.cells}
                    try:
                        while pending:
                            done, pending = wait(
                                pending, timeout=5, return_when=FIRST_COMPLETED
                            )
                            for future in done:
                                future.result()
                            self.snapshot(
                                "interrupting" if self.stop.is_set() else "running"
                            )
                            if done:
                                aggregate(self.directory)
                    except BaseException:
                        self.stop.set()
                        raise
                aggregate(self.directory)
                self.snapshot("interrupted" if self.stop.is_set() else "finished")
            except Exception:
                self.snapshot("failed")
                raise
            finally:
                self.stop.set()
                for process in list(self.children.copy().values()):
                    self.terminate(process)
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
        return campaign_status(self.directory)


def run_campaign(
    path: str | Path, out: str | Path | None = None, *, retry_failed: bool = False
) -> dict:
    """Resolve a saved config and run or resume its named campaign directory."""
    config = load_campaign(path)
    if out is None:
        out = config.get("out")
    if out is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out = Path("results") / config["id"] / (stamp + "_" + config_hash(config)[:8])
    return CampaignRunner(config, Path(out).resolve(), retry_failed=retry_failed).run()
