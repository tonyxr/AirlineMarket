"""F3 isolated sparse-access revision, keeping both earlier pilots immutable."""

from pathlib import Path
import json
import sys

from . import scaling_counted, scaling_study
from ..campaign.artifacts import atomic_json, file_hash
from ..sim.sparse_access import sparse_access


def main():
    """Use the tested study controls and install sparse primitives only in workers."""
    # The detach helper derives the child module from its own module namespace.
    # Reuse its process controls with an explicit command rather than altering it.
    if "--detach" in sys.argv:
        import os
        import psutil
        import subprocess

        path = Path(sys.argv[sys.argv.index("--config") + 1]).resolve()
        config = json.loads(path.read_text())
        root = Path(config["out"]).resolve()
        root.mkdir(parents=True, exist_ok=True)
        supervisor = root / "supervisor.json"
        if supervisor.exists():
            old = json.loads(supervisor.read_text())
            try:
                process = psutil.Process(old["pid"])
                if abs(process.create_time() - old["created_at"]) < 0.1:
                    print(json.dumps({"already_running": True, **old}))
                    return
            except psutil.NoSuchProcess:
                pass
        command = [sys.executable, "-u", "-m", __spec__.name, "--config", str(path)]
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
        return
    if "--worker" in sys.argv:
        original_worker = scaling_study.run_worker
        original_profile = scaling_study.profile_summary

        def profile(path):
            """Attribute each new module exclusively to its original scientific stage."""
            import pstats

            result = original_profile(path)
            stages = result["exclusive_seconds"]
            for (file, _, name), row in pstats.Stats(str(path)).stats.items():
                filename = Path(file).name
                group = None
                if filename == "count_ledger.py":
                    group = "ledger_and_accounting"
                elif filename == "sparse_access.py":
                    group = (
                        "quotes"
                        if name == "sparse_posted_quotes"
                        else "pricing_and_controls"
                    )
                if group:
                    stages["other_and_builtins"] -= row[2]
                    stages[group] = stages.get(group, 0) + row[2]
            total = sum(stages.values())
            result["exclusive_shares"] = {
                key: value / total for key, value in stages.items()
            }
            return result

        def worker(config, directory, parent):
            """Seal the actual sparse implementation and source hashes after completion."""
            with sparse_access():
                code = original_worker(config, directory, parent)
            path = Path(directory) / "results.json"
            result = json.loads(path.read_text())
            result["engine"] = (
                "MarketEngine with isolated exact counted ledger and sparse incidence access; invariants enabled"
            )
            result["sparse_implementation"] = "airline_market.sim.sparse_access"
            result["implementation_source_sha256"] = {
                name: file_hash(Path(__file__).parents[1] / "sim" / name)
                for name in ("count_ledger.py", "sparse_access.py")
            }
            atomic_json(path, result)
            receipt_path = Path(directory) / "COMPLETE.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["results_sha256"] = file_hash(path)
            atomic_json(receipt_path, receipt)
            return code

        scaling_study.run_worker = worker
        scaling_study.profile_summary = profile
    scaling_study.MODULE = "airline_market.experiments.scaling_sparse"
    scaling_study.main()


if __name__ == "__main__":
    main()
