"""Resumable reduced-data processing and F8/V1 evidence, independent of live F1."""

from pathlib import Path
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time

from .scoped_io import atomic_json, sha256


MODULE = "airline_market.pipeline.scoped_processing"


def now():
    """Return a timezone-explicit execution timestamp."""
    return datetime.now(timezone.utc).isoformat()


def processing_hash(config):
    """Hash declared scope, input identities and this processing implementation."""
    source = {
        str(p): sha256(p) for p in sorted(Path(__file__).parent.glob("scoped*.py"))
    }
    p = Path(__file__).with_name("regional.py")
    source[str(p)] = sha256(p)
    scope = json.loads(Path(config["scope"]).read_text())
    receipts = {}
    for requirement in scope["requirements"]:
        for id in requirement["ids"]:
            r = json.loads(
                (
                    Path(
                        config.get("source_receipts", "results/data_acquisition/files")
                    )
                    / (id + ".json")
                ).read_text()
            )
            if r.get("status") != "downloaded":
                raise ValueError(f"required source is not acquired: {id}")
            if sha256(r["path"]) != r["sha256"]:
                raise ValueError(f"source hash mismatch: {id}")
            receipts[id] = {
                "path": r["path"],
                "sha256": r["sha256"],
                "source_url": r["source_url"],
            }
    payload = {"config": config, "scope": scope, "source": source, "inputs": receipts}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest, payload


def run(config):
    """Execute independent stages, retaining failed attempts and matching receipts."""
    import pandas as pd
    import psutil
    from ..campaign.artifacts import environment
    from .scoped_tables import (
        process_flights,
        process_capacity,
        process_markets,
        process_weather,
    )
    from .scoped_quotes import process_quotes

    root, output = Path(config["evidence_output"]), Path(config["processed_output"])
    root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    with (root / ".coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        digest, manifest = processing_hash(config)
        previous = root / "processing_identity.json"
        if previous.exists() and json.loads(previous.read_text())["hash"] != digest:
            raise ValueError(
                "processing source/scope changed: preserve the old run and use a new revision/output"
            )
        atomic_json(previous, {"hash": digest, **manifest})
        atomic_json(root / "environment.json", environment())
        atomic_json(root / "config.json", config)
        state = {
            "status": "running",
            "pid": os.getpid(),
            "process_created_at": psutil.Process().create_time(),
            "started_at": now(),
            "stages": {},
        }
        started = time.monotonic()
        stopped = threading.Event()
        state_lock = threading.RLock()

        def publish(**values):
            """Refresh actual progress and owned-process resources without global scans."""
            with state_lock:
                state.update(
                    values,
                    updated_at=now(),
                    elapsed_seconds=time.monotonic() - started,
                    rss_bytes=psutil.Process().memory_info().rss,
                )
                atomic_json(root / "status.json", state)

        def monitor():
            """Enforce only the declared wall/RSS limits, retaining partial stages."""
            while not stopped.wait(2):
                rss = psutil.Process().memory_info().rss
                reason = (
                    "time_limit"
                    if time.monotonic() - started > config["limits"]["seconds"]
                    else "memory_limit"
                    if rss > config["limits"]["rss_gb"] * 1024**3
                    else None
                )
                if reason:
                    publish(status=reason)
                    os._exit(124)
                publish()

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        inputs = manifest["inputs"]
        scope = manifest["scope"]
        artifacts = {}

        def stage(name, dependencies, execute):
            """Resume a completed matching stage; otherwise keep a new durable attempt."""
            directory = root / "stages" / name
            directory.mkdir(parents=True, exist_ok=True)
            for receipt in sorted(directory.glob("attempt-*/COMPLETE.json")):
                r = json.loads(receipt.read_text())
                if r["identity_hash"] == digest and all(
                    Path(p).exists() and sha256(p) == h
                    for p, h in r["artifact_hashes"].items()
                ):
                    artifacts[name] = Path(r["output"])
                    state["stages"][name] = {
                        "status": "completed_reused",
                        "output": r["output"],
                        "result": r["result"],
                    }
                    publish()
                    return
            if any(name not in artifacts for name in dependencies):
                state["stages"][name] = {
                    "status": "blocked_dependency",
                    "dependencies": dependencies,
                }
                publish()
                return
            number = len(list(directory.glob("attempt-*"))) + 1
            attempt = directory / f"attempt-{number:04d}"
            attempt.mkdir()
            target = output / name / f"attempt-{number:04d}"
            target.mkdir(parents=True, exist_ok=True)
            state["stages"][name] = {"status": "running", "output": str(target)}
            publish(active_stage=name)
            timer = time.perf_counter()
            try:
                result = execute(target)
                record = {
                    "identity_hash": digest,
                    "output": str(target),
                    "result": result,
                    "seconds": time.perf_counter() - timer,
                    "completed_at": now(),
                    "artifact_hashes": {
                        str(p): sha256(p)
                        for p in sorted(target.glob("*"))
                        if p.is_file()
                    },
                }
                atomic_json(attempt / "COMPLETE.json", record)
                artifacts[name] = target
                state["stages"][name] = {
                    "status": "completed",
                    "output": str(target),
                    "result": result,
                    "seconds": record["seconds"],
                }
            except Exception as error:
                import traceback

                traceback.print_exc()
                state["stages"][name] = {
                    "status": "failed",
                    "reason": str(error),
                    "output": str(target),
                    "seconds": time.perf_counter() - timer,
                }
                atomic_json(attempt / "failure.json", state["stages"][name])
            publish()

        try:

            def flights(target):
                """Normalize regional-inclusive dates and count actual physical flights."""
                _, checks = process_flights(
                    [inputs[f"ontime_mkt_2022_{m}"]["path"] for m in range(4, 10)],
                    scope,
                    target,
                    publish,
                )
                return checks

            def capacity(target):
                """Allocate totals while preserving operating-carrier source quantities."""
                f = pd.read_parquet(artifacts["flights"] / "flights.parquet")
                _, checks = process_capacity(
                    inputs["t100_segment_2022"]["path"], f, scope, target, publish
                )
                return checks

            def markets(target):
                """Derive scope from Q2 only and keep Q3 separately."""
                _, checks = process_markets(
                    {q: inputs[f"DB1BMarket_2022_{q}"]["path"] for q in (2, 3)},
                    scope,
                    target,
                    publish,
                )
                return checks

            def quotes(target):
                """Read the original ZIP once for selected markets, not a 31GB extraction."""
                market_names = pd.read_csv(
                    artifacts["markets"] / "model_markets.csv"
                ).market.tolist()
                f = pd.read_parquet(artifacts["flights"] / "flights.parquet")
                return process_quotes(
                    inputs["flightprices_v1"]["path"],
                    market_names,
                    scope,
                    f,
                    target,
                    publish,
                    config["booking_window_days"],
                    state["stages"]["flights"]["result"][
                        "national_marketing_network_codes"
                    ],
                )

            stage("flights", [], flights)
            stage("capacity", ["flights"], capacity)
            stage("markets", [], markets)
            stage(
                "weather",
                [],
                lambda target: process_weather(
                    inputs["noaa_daily_2022"]["path"], target
                ),
            )
            stage("quotes", ["flights", "markets"], quotes)
            if all(
                name in artifacts
                for name in ("flights", "capacity", "markets", "weather", "quotes")
            ):
                q2 = state["stages"]["markets"]["result"]["2"][
                    "all_pool_hub_connecting_passengers_by_network"
                ]
                total = sum(q2.values())
                shares = {
                    carrier: q2.get(carrier, 0) / max(1, total)
                    for carrier in scope["carriers"]
                }
                quote_stats = state["stages"]["quotes"]["result"]
                f8 = {
                    "hub": scope["hub_candidate"],
                    "selection_period": "2022 Q2 departure cohorts only",
                    "connecting_passenger_shares": shares,
                    "qualifying_quote_markets": quote_stats[
                        "q2_competitive_markets_at_least_200_quotes"
                    ],
                    "max_shared_rival_leg_markets": quote_stats[
                        "max_rival_leg_competitive_markets"
                    ],
                    "thresholds": {
                        "min_each_carrier_share": 0.2,
                        "min_markets_200_quotes": 30,
                        "min_shared_leg_markets": 3,
                    },
                    "eligible": all(value >= 0.2 for value in shares.values())
                    and quote_stats["q2_competitive_markets_at_least_200_quotes"] >= 30
                    and quote_stats["max_rival_leg_competitive_markets"] >= 3,
                    "calibration_status": "not_run",
                    "note": "F8 eligibility is not evidence of a positive information value or empirical causal effect.",
                }
                atomic_json(root / "F8_scope_summary.json", f8)
                pd.DataFrame(
                    [
                        {
                            **{k: v for k, v in f8.items() if not isinstance(v, dict)},
                            **{
                                carrier + "_connecting_share": value
                                for carrier, value in shares.items()
                            },
                        }
                    ]
                ).to_csv(root / "F8_scope_summary.csv", index=False)
                state["F8"] = f8
            status = "completed" if len(artifacts) == 5 else "completed_with_failures"
            atomic_json(
                root / "results.json",
                {
                    "status": status,
                    "identity_hash": digest,
                    "stages": state["stages"],
                    "F8": state.get("F8"),
                },
            )
            publish(status=status, active_stage=None)
        finally:
            stopped.set()
            thread.join(timeout=3)
        return {"status": state["status"], "output": str(root)}


def main():
    """Launch or resume the declared processing recipe with durable supervision."""
    import psutil
    from ..campaign.runner import THREAD_ENV

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.detach:
        root = Path(config["evidence_output"])
        root.mkdir(parents=True, exist_ok=True)
        supervisor = root / "supervisor.json"
        if supervisor.exists():
            r = json.loads(supervisor.read_text())
            try:
                p = psutil.Process(r["pid"])
                if abs(p.create_time() - r["created_at"]) < 0.1:
                    print(json.dumps({"already_running": True, **r}, indent=2))
                    return
            except psutil.NoSuchProcess:
                pass
        command = [
            sys.executable,
            "-u",
            "-m",
            MODULE,
            "--config",
            str(Path(args.config).resolve()),
        ]
        if sys.platform == "darwin":
            command = ["/usr/bin/caffeinate", "-is", *command]
        with (root / "run.log").open("a") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ, **THREAD_ENV},
            )
        r = {
            "pid": process.pid,
            "created_at": psutil.Process(process.pid).create_time(),
            "started_at": now(),
            "command": command,
        }
        atomic_json(supervisor, r)
        print(json.dumps(r, indent=2))
    else:
        print(json.dumps(run(config), indent=2))


if __name__ == "__main__":
    main()
