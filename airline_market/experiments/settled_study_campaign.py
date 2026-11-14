"""Opt-in custom-study adapter for returned-future campaign aggregation."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from airline_market.campaign import runner
from airline_market.campaign.artifacts import atomic_json, utc_now
from airline_market.campaign.design import load_campaign
from airline_market.campaign.settled_runner import SettledCampaignRunner
from airline_market.experiments.study_campaign import StudyProcesses

MODULE = "airline_market.experiments.settled_study_campaign"


def main():
    """Launch explicit new study roots using the unchanged study worker/ownership."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    config = load_campaign(args.config)
    root = Path(config["out"]).resolve()
    if args.detach:
        import psutil

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
        command = [
            sys.executable,
            "-u",
            "-m",
            MODULE,
            "--config",
            str(args.config.resolve()),
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
                env={**os.environ, **runner.THREAD_ENV},
            )
        record = {
            "pid": process.pid,
            "process_created_at": psutil.Process(process.pid).create_time(),
            "started_at": utc_now(),
            "command": command,
            "config": str(args.config.resolve()),
        }
        atomic_json(supervisor, record)
        print(json.dumps(record))
        return
    # Only this coordinator's imported runner module is redirected. The shared
    # source and every other live Python process remain unchanged.
    runner.subprocess = StudyProcesses()
    print(json.dumps(SettledCampaignRunner(config, root).run()))


if __name__ == "__main__":
    main()
