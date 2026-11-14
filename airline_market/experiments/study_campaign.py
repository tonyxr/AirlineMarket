"""Run independent study tasks through the tested isolated campaign controls.

Only this coordinator redirects its worker command. Existing F1/F3 process
code and task registries remain unchanged. A child imports the declared
study function, then uses the original owned-worker heartbeat and receipts.
"""

from pathlib import Path
import argparse
import importlib
import json
import os
import subprocess
import sys

from ..campaign import runner, worker, tasks
from ..campaign.artifacts import atomic_json, utc_now
from ..campaign.design import load_campaign

MODULE = "airline_market.experiments.study_campaign"


class StudyProcesses:
    """Redirect only the campaign worker invocation in this local coordinator."""

    def __getattr__(self, name):
        """Keep all unrelated process APIs identical to the tested runner."""
        return getattr(subprocess, name)

    def Popen(self, command, *args, **kwargs):
        """Launch the study worker with the same process-group ownership."""
        command = list(command)
        if "airline_market.campaign.worker" in command:
            index = command.index("airline_market.campaign.worker")
            command[index : index + 1] = [MODULE, "--worker"]
        return subprocess.Popen(command, *args, **kwargs)


def execute(cell, context):
    """Import the explicit maintained study task, without editing the registry."""
    prefix, module, name = cell["task"].split(":")
    if prefix != "study" or not module.startswith("airline_market.experiments."):
        raise ValueError("study task must identify a maintained experiment function")
    return getattr(importlib.import_module(module), name)(cell, context)


def main():
    """Run, detach or execute one isolated study cell using existing controls."""
    if "--worker" in sys.argv:
        sys.argv.remove("--worker")
        tasks.execute = execute
        raise SystemExit(worker.main())
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.detach:
        import psutil

        path = Path(args.config).resolve()
        root = Path(load_campaign(path)["out"]).resolve()
        root.mkdir(parents=True, exist_ok=True)
        supervisor = root / "supervisor.json"
        if supervisor.exists():
            old = json.loads(supervisor.read_text())
            try:
                process = psutil.Process(old["pid"])
                if abs(process.create_time() - old["process_created_at"]) < 0.1:
                    print(json.dumps({"already_running": True, **old}))
                    return
            except psutil.NoSuchProcess:
                pass
        command = [sys.executable, "-u", "-m", MODULE, "--config", str(path)]
        if sys.platform == "darwin":
            command = ["/usr/bin/caffeinate", "-is", *command]
        with (root / "supervisor.log").open("a") as log:
            process = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ, **runner.THREAD_ENV},
            )
        record = {
            "pid": process.pid,
            "process_created_at": psutil.Process(process.pid).create_time(),
            "started_at": utc_now(),
            "command": command,
            "config": str(path),
        }
        atomic_json(supervisor, record)
        print(json.dumps(record, indent=2))
    else:
        runner.subprocess = StudyProcesses()
        print(json.dumps(runner.run_campaign(args.config), indent=2))


if __name__ == "__main__":
    main()
