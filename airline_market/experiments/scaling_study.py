"""Isolated Poisson-oligopoly throughput/profile study (validation plan F3).

Each timing cohort includes engine construction, random-policy reset, tape
generation, simulation and accounting. Model compilation is separate.
Profiling uses additional declared validation cohorts and never contaminates
the unprofiled timing estimate. This module does not alter the live F1 code.
"""

from collections import defaultdict
from pathlib import Path
import argparse
import cProfile
import csv
import fcntl
import gzip
import io
import json
import os
import pstats
import resource
import signal
import subprocess
import sys
import threading
import time

import numpy as np

from ..agents.random_coefficient import RandomCoefficientController
from ..campaign.artifacts import atomic_json, environment, file_hash, log_run, utc_now
from ..campaign.runner import THREAD_ENV, completed_attempt
from ..campaign.worker import CellContext
from ..core.hashing import config_hash
from ..pipeline.scaling import scaling_network
from ..sim.engine import MarketEngine
from ..spec import compile_market, dump_spec

MODULE = "airline_market.experiments.scaling_study"


def cohort(compiled, seed: int, context: CellContext, label: str) -> dict:
    """Measure a complete cohort with carrier-local random controls (Eqs. 5, 18)."""
    started, cpu_started = time.perf_counter(), time.process_time()
    engine = MarketEngine(
        compiled,
        {
            carrier.code: RandomCoefficientController()
            for carrier in compiled.spec.carriers
        },
    )
    engine.reset(seed)
    for _ in range(compiled.spec.horizon.n_periods):
        engine.step()
        context.update(
            {
                "stage": label,
                "seed": seed,
                "period": engine.world.t,
                "periods": compiled.spec.horizon.n_periods,
                "requests_so_far": len(engine.world.request_log),
            }
        )
    metrics = engine.metrics()
    elapsed = time.perf_counter() - started
    return {
        "seed": seed,
        "seconds": elapsed,
        "cpu_seconds": time.process_time() - cpu_started,
        "period_transitions": compiled.spec.horizon.n_periods,
        "shopping_requests": len(engine.world.request_log),
        "profit": metrics["profit"],
        "share": metrics["share"],
    }


def profile_summary(path: Path) -> dict:
    """Attribute exclusive profile time; cumulative hotspots may overlap."""
    stats = pstats.Stats(str(path))
    stages = defaultdict(float)
    functions = []
    groups = {
        "arrivals_and_tapes": {"arrivals.py", "tape.py", "exogenous.py"},
        "choice": {"choice.py"},
        "ledger_and_accounting": {"ledger.py", "capacity.py", "accounting.py"},
        "quotes": {"quotes.py", "background.py"},
        "observation": {"observation.py"},
        "pricing_and_controls": {"price_rule.py", "random_coefficient.py"},
        "engine_orchestration": {"engine.py"},
    }
    for (filename, line, name), (
        primitive,
        calls,
        exclusive,
        cumulative,
        _,
    ) in stats.stats.items():
        stage = next(
            (group for group, files in groups.items() if Path(filename).name in files),
            "other_and_builtins",
        )
        stages[stage] += exclusive
        functions.append(
            {
                "file": filename,
                "line": line,
                "function": name,
                "calls": calls,
                "exclusive_seconds": exclusive,
                "cumulative_seconds": cumulative,
            }
        )
    total = sum(stages.values())
    return {
        "exclusive_seconds": dict(stages),
        "exclusive_shares": {key: value / total for key, value in stages.items()},
        "top_cumulative_functions": sorted(
            functions, key=lambda row: row["cumulative_seconds"], reverse=True
        )[:30],
        "note": "Exclusive shares partition profiled time; builtins are not assigned back to their Python caller. Profiled cohorts are excluded from timing estimates.",
    }


def run_worker(config: dict, directory: Path, parent: int) -> int:
    """Run a declared cell, preserving live counters and all completed evidence."""
    if parent is None or os.getpgrp() != os.getpid():
        raise ValueError("scaling worker requires its own process group")
    context = CellContext(directory)
    stop = threading.Event()

    def heartbeat():
        """Publish owned-process counters and stop after coordinator loss."""
        while not stop.wait(1):
            if os.getppid() != parent:
                os.killpg(os.getpgrp(), signal.SIGTERM)
            context.publish()

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    atomic_json(directory / "environment.json", environment())
    started = time.perf_counter()
    try:
        count = int(config["n_cohorts"])
        profile_count = int(config.get("n_profile_cohorts", 0))
        seed_start = int(config.get("seed_start", 200000))
        if (
            count < 1
            or profile_count < 0
            or not (
                200000 <= seed_start and seed_start + count + profile_count <= 300000
            )
        ):
            raise ValueError("F3 uses nonempty validation cohorts only")
        context.update({"stage": "compile"})
        setup = time.perf_counter()
        spec = scaling_network(config["parameters"])
        compiled = compile_market(spec)
        compile_seconds = time.perf_counter() - setup
        dump_spec(spec, directory / "spec.json")
        rows = []
        for k in range(count):
            rows.append(cohort(compiled, seed_start + k, context, "timing"))
            context.update({"cohorts_completed": k + 1, "cohorts_total": count})
            atomic_json(directory / "timing_rows.json", rows)
        profile = None
        if profile_count:
            profiler = cProfile.Profile()
            profiler.enable()
            for k in range(profile_count):
                cohort(
                    compiled,
                    seed_start + count + k,
                    context,
                    "profile_excluded_from_timing",
                )
            profiler.disable()
            profiler.dump_stats(str(directory / "profile.pstats"))
            profile = profile_summary(directory / "profile.pstats")
            atomic_json(directory / "profile_summary.json", profile)
        durations = np.asarray([row["seconds"] for row in rows])
        text = io.StringIO()
        writer = csv.DictWriter(text, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        with (directory / "episodes.csv.gz").open("wb") as stream:
            with gzip.GzipFile(
                fileobj=stream, mode="wb", filename="", mtime=0
            ) as archive:
                archive.write(text.getvalue().encode())
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if sys.platform == "darwin" else 1024
        )
        result = {
            "schema": "airline-scaling-cell/1",
            "status": "completed",
            "config_hash": config["config_hash"],
            "spec_hash": compiled.spec_hash,
            "parameters": config["parameters"],
            "n_cohorts": count,
            "n_profile_cohorts_excluded": profile_count,
            "compile_seconds": compile_seconds,
            "mean_seconds_per_cohort": float(durations.mean()),
            "median_seconds_per_cohort": float(np.median(durations)),
            "iqr_seconds_per_cohort": float(
                np.subtract(*np.percentile(durations, [75, 25]))
            ),
            "period_transitions_per_second": sum(
                row["period_transitions"] for row in rows
            )
            / durations.sum(),
            "shopping_requests_per_second": sum(
                row["shopping_requests"] for row in rows
            )
            / durations.sum(),
            "peak_rss_bytes": peak,
            "wall_seconds": time.perf_counter() - started,
            "profile": profile,
            "target_seconds_per_cohort": 2.0,
            "target_scope": "50 spokes, 3 carriers, T=60, 64 cohorts; compilation excluded",
            "criterion_status": "pilot_only" if count < 64 else "measured_64_cohorts",
            "engine": "reference MarketEngine; invariants enabled; random coefficient policies for every carrier",
        }
    except Exception as error:
        import traceback

        traceback.print_exc()
        result = {
            "status": "failed",
            "config_hash": config["config_hash"],
            "reason": str(error),
            "wall_seconds": time.perf_counter() - started,
        }
    finally:
        stop.set()
        thread.join(timeout=2)
    result.setdefault("parameters", config["parameters"])
    atomic_json(directory / "results.json", result)
    atomic_json(
        directory / "COMPLETE.json",
        {
            "config_hash": config["config_hash"],
            "status": result["status"],
            "results_sha256": file_hash(directory / "results.json"),
        },
    )
    return int(result["status"] == "failed")


def run(config: dict) -> dict:
    """Run/resume declared F3 cells sequentially while F1 uses another worker."""
    import psutil

    if config.get("schema") != "airline-scaling-study/1":
        raise ValueError("unknown scaling study schema")
    root = Path(config["out"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config_path = root / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != config:
            raise ValueError("output belongs to a different scaling configuration")
        provenance = environment()
        previous = root / "environment.json"
        if previous.exists():
            old = json.loads(previous.read_text())["source_hashes"]
            if any(
                provenance["source_hashes"].get(name) != digest
                for name, digest in old.items()
            ):
                raise ValueError("scaling source changed; use a new revision/output")
        atomic_json(config_path, config)
        atomic_json(previous, provenance)
        results = []
        for parameters in config["cells"]:
            selected = config.get("profile_filter")
            profile_count = (
                config.get("n_profile_cohorts", 0)
                if not selected
                or all(parameters.get(key) == value for key, value in selected.items())
                else 0
            )
            cell = {
                "parameters": {**config.get("fixed", {}), **parameters},
                "n_cohorts": config["n_cohorts"],
                "n_profile_cohorts": profile_count,
                "seed_start": config.get("seed_start", 200000),
                "revision": config["revision"],
            }
            cell["config_hash"] = config_hash(cell)
            directory = root / "cells" / cell["config_hash"][:16]
            directory.mkdir(parents=True, exist_ok=True)
            found = completed_attempt(directory, cell["config_hash"])
            if found:
                results.append(found[1])
                continue
            for path in directory.glob("attempt-*/status.json"):
                old_status = json.loads(path.read_text())
                if old_status.get("status") != "running":
                    continue
                for _ in range(20):
                    try:
                        old_process = psutil.Process(old_status["pid"])
                        if (
                            abs(old_process.create_time() - old_status["created_at"])
                            > 0.1
                        ):
                            break
                    except psutil.NoSuchProcess:
                        break
                    time.sleep(0.2)
                else:
                    raise RuntimeError(
                        "an orphaned matching scaling cell is still active"
                    )
                atomic_json(path, {**old_status, "status": "interrupted"})
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
                        MODULE,
                        "--worker",
                        "--config",
                        str(attempt / "config.json"),
                        "--out",
                        str(attempt),
                        "--parent",
                        str(os.getpid()),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, **THREAD_ENV},
                    start_new_session=True,
                )
                atomic_json(
                    attempt / "status.json",
                    {
                        "status": "running",
                        "pid": process.pid,
                        "created_at": psutil.Process(process.pid).create_time(),
                    },
                )
                status = None
                try:
                    while process.poll() is None:
                        if time.monotonic() - started > config["limits"]["seconds"]:
                            status = "time_limit"
                        try:
                            if (
                                psutil.Process(process.pid).memory_info().rss
                                > config["limits"]["rss_gb"] * 1024**3
                            ):
                                status = "memory_limit"
                        except psutil.NoSuchProcess:
                            pass
                        atomic_json(
                            root / "status.json",
                            {
                                "status": "running",
                                "updated_at": utc_now(),
                                "coordinator_pid": os.getpid(),
                                "worker_pid": process.pid,
                                "active_parameters": cell["parameters"],
                                "completed_cells": len(results),
                                "cells_total": len(config["cells"]),
                                "attempt": str(attempt),
                            },
                        )
                        if status:
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                            break
                        time.sleep(1)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            receipt = completed_attempt(directory, cell["config_hash"])
            result = (
                receipt[1]
                if receipt and receipt[0] == attempt
                else {
                    "status": status or "failed",
                    "config_hash": cell["config_hash"],
                    "wall_seconds": time.monotonic() - started,
                    "reason": "worker exited without a receipt"
                    if status is None
                    else status,
                }
            )
            result.setdefault("parameters", cell["parameters"])
            if receipt is None:
                atomic_json(attempt / "results.json", result)
                atomic_json(
                    attempt / "COMPLETE.json",
                    {
                        "config_hash": cell["config_hash"],
                        "status": result["status"],
                        "results_sha256": file_hash(attempt / "results.json"),
                    },
                )
            results.append(result)
            atomic_json(
                attempt / "status.json",
                {"status": result["status"], "completed_at": utc_now()},
            )
            log_run(
                {
                    "date": utc_now(),
                    "id": config["id"],
                    "run_id": str(attempt),
                    "git_sha": provenance["git_sha"],
                    "status": result["status"],
                    "seconds": result["wall_seconds"],
                    "headline": str(result.get("mean_seconds_per_cohort"))
                    + " s/cohort",
                    "deviations": "profile cohorts excluded from timing; pilot is not the 64-cohort F3 criterion"
                    if config["n_cohorts"] < 64
                    else "",
                },
                path=config.get("log_path", "EXPERIMENT_LOG.md"),
            )
        fields = [
            "n_spokes",
            "n_carriers",
            "periods",
            "status",
            "n_cohorts",
            "mean_seconds_per_cohort",
            "period_transitions_per_second",
            "shopping_requests_per_second",
            "peak_rss_bytes",
        ]
        with (root / "scaling.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                {
                    field: result.get("parameters", {}).get(field, result.get(field))
                    for field in fields
                }
                for result in results
            )
        status = (
            "finished"
            if all(row["status"] == "completed" for row in results)
            else "finished_with_failures"
        )
        atomic_json(root / "results.json", {"status": status, "cells": results})
        atomic_json(
            root / "status.json",
            {
                "status": status,
                "updated_at": utc_now(),
                "completed_cells": len(results),
                "cells_total": len(config["cells"]),
            },
        )
        return {"status": status, "output": str(root), "cells": len(results)}


def main() -> None:
    """Launch a declared scaling pilot/campaign or execute its isolated worker."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--parent", type=int)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.worker:
        raise SystemExit(run_worker(config, Path(args.out), args.parent))
    if args.detach:
        import psutil

        root = Path(config["out"]).resolve()
        root.mkdir(parents=True, exist_ok=True)
        supervisor = root / "supervisor.json"
        if supervisor.exists():
            prior = json.loads(supervisor.read_text())
            if (
                psutil.pid_exists(prior["pid"])
                and abs(
                    psutil.Process(prior["pid"]).create_time() - prior["created_at"]
                )
                < 0.1
            ):
                print(json.dumps({"already_running": True, **prior}, indent=2))
                return
        command = [
            sys.executable,
            "-u",
            "-m",
            "airline_market.experiments.scaling_study",
            "--config",
            str(Path(args.config).resolve()),
        ]
        if sys.platform == "darwin":
            command = ["/usr/bin/caffeinate", "-is", *command]
        with (root / "supervisor.log").open("a") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ, **THREAD_ENV},
            )
        record = {
            "pid": process.pid,
            "created_at": psutil.Process(process.pid).create_time(),
            "started_at": utc_now(),
            "command": command,
        }
        atomic_json(supervisor, record)
        print(json.dumps(record, indent=2))
    else:
        print(json.dumps(run(config), indent=2))


if __name__ == "__main__":
    main()
