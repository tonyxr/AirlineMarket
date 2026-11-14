"""Write a compact progress snapshot of AirlineMarket jobs and artifacts only."""

from pathlib import Path
import json
import os
import time


def main() -> None:
    """Inspect project-local processes without signaling or restarting anything."""
    import psutil
    from airline_market.campaign.artifacts import atomic_json, utc_now
    from airline_market.campaign.runner import campaign_status

    root = Path.cwd().resolve()
    processes = []
    for process in psutil.process_iter(["pid", "cwd", "name"]):
        try:
            if (
                process.pid == os.getpid()
                or not process.info["cwd"]
                or Path(process.info["cwd"]).resolve() != root
            ):
                continue
            command = process.cmdline()
            if not any(
                token in " ".join(command)
                for token in ["airline_market", "pytest", "start_validation_campaign"]
            ):
                continue
            processes.append(
                {
                    "pid": process.pid,
                    "ppid": process.ppid(),
                    "name": process.info["name"],
                    "command": command,
                    "elapsed_seconds": time.time() - process.create_time(),
                    "rss_bytes": process.memory_info().rss,
                    "cpu_seconds": sum(process.cpu_times()[:2]),
                    "status": process.status(),
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    state_path = Path("config/campaigns/execution_state.json")
    state = json.loads(state_path.read_text())
    path = Path(state["phase1"]["F1"]["out"])
    result = {
        "updated_at": utc_now(),
        "processes": processes,
        "memory": {
            "available_bytes": psutil.virtual_memory().available,
            "total_bytes": psutil.virtual_memory().total,
            "swap_used_bytes": psutil.swap_memory().used,
        },
        "F1": campaign_status(path)
        if (path / "status.json").exists()
        else {"status": "not_started", "directory": str(path)},
    }
    atomic_json("results/monitor/status.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
