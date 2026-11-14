"""Run F3 with an exact counted ledger in isolated, source-attributed workers."""

from pathlib import Path
import json
import os
import subprocess
import sys

from . import scaling_study
from ..campaign.artifacts import atomic_json, file_hash
from ..sim.count_ledger import counted_ledger


def detach():
    """Keep the counted worker module in the detached coordinator command."""
    import psutil

    config_path = Path(sys.argv[sys.argv.index("--config") + 1]).resolve()
    config = json.loads(config_path.read_text())
    root = Path(config["out"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    supervisor = root / "supervisor.json"
    if supervisor.exists():
        old = json.loads(supervisor.read_text())
        try:
            process = psutil.Process(old["pid"])
            if abs(process.create_time() - old["created_at"]) < 0.1:
                print(json.dumps({"already_running": True, **old}, indent=2))
                return
        except psutil.NoSuchProcess:
            pass
    command = [
        sys.executable,
        "-u",
        "-m",
        "airline_market.experiments.scaling_counted",
        "--config",
        str(config_path),
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
            env={**os.environ, **scaling_study.THREAD_ENV},
        )
    record = {
        "pid": process.pid,
        "created_at": psutil.Process(process.pid).create_time(),
        "started_at": scaling_study.utc_now(),
        "command": command,
    }
    atomic_json(supervisor, record)
    print(json.dumps(record, indent=2))


def main():
    """Reuse validated campaign controls while changing only the counted primitives."""
    if "--detach" in sys.argv:
        detach()
        return
    if "--worker" in sys.argv:
        original_worker = scaling_study.run_worker
        original_profile = scaling_study.profile_summary

        def profile(path):
            """Attribute the new ledger module to the same nonoverlapping stage."""
            import pstats

            result = original_profile(path)
            extra = sum(
                row[2]
                for (file, _, _), row in pstats.Stats(str(path)).stats.items()
                if Path(file).name == "count_ledger.py"
            )
            seconds = result["exclusive_seconds"]
            seconds["other_and_builtins"] -= extra
            seconds["ledger_and_accounting"] = (
                seconds.get("ledger_and_accounting", 0) + extra
            )
            total = sum(seconds.values())
            result["exclusive_shares"] = {
                key: value / total for key, value in seconds.items()
            }
            result["note"] += (
                " Counted-ledger exclusive time is included in ledger_and_accounting."
            )
            return result

        def worker(config, directory, parent):
            """Finish the declared cell before sealing its implementation label."""
            with counted_ledger():
                code = original_worker(config, directory, parent)
            path = Path(directory) / "results.json"
            result = json.loads(path.read_text())
            result["engine"] = (
                "MarketEngine with isolated counted product/resource ledger; invariants enabled"
            )
            result["ledger_implementation"] = "airline_market.sim.count_ledger"
            result["ledger_source_sha256"] = file_hash(
                Path(__file__).parents[1] / "sim/count_ledger.py"
            )
            atomic_json(path, result)
            receipt_path = Path(directory) / "COMPLETE.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["results_sha256"] = file_hash(path)
            atomic_json(receipt_path, receipt)
            return code

        scaling_study.run_worker = worker
        scaling_study.profile_summary = profile
    scaling_study.MODULE = "airline_market.experiments.scaling_counted"
    scaling_study.main()


if __name__ == "__main__":
    main()
