"""Detach one authorized local campaign with a durable log and sleep inhibition."""

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time


def main() -> None:
    """Launch once; a matching live supervisor is reported instead of duplicated."""
    import psutil
    from airline_market.campaign.artifacts import atomic_json, utc_now
    from airline_market.campaign.design import load_campaign
    from airline_market.campaign.runner import THREAD_ENV

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_campaign(config_path)
    directory = Path(config["out"]).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "supervisor.json"
    if path.exists():
        prior = json.loads(path.read_text())
        try:
            process = psutil.Process(prior["pid"])
            if abs(process.create_time() - prior["process_created_at"]) < 0.1:
                print(json.dumps({"already_running": True, **prior}, indent=2))
                return
        except psutil.NoSuchProcess:
            pass
    command = [
        sys.executable,
        "-u",
        "-m",
        "airline_market",
        "campaign",
        "run",
        str(config_path),
    ]
    if sys.platform == "darwin":
        command = ["/usr/bin/caffeinate", "-is", *command]
    with (directory / "supervisor.log").open("a") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, **THREAD_ENV},
            start_new_session=True,
            close_fds=True,
        )
    record = {
        "pid": process.pid,
        "process_created_at": psutil.Process(process.pid).create_time(),
        "started_at": utc_now(),
        "command": command,
        "config": str(config_path),
        "directory": str(directory),
        "thread_settings": THREAD_ENV,
    }
    atomic_json(path, record)
    time.sleep(1)
    if process.poll() is not None:
        raise RuntimeError(
            f"campaign exited immediately; see {directory / 'supervisor.log'}"
        )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
