"""One campaign cell in an isolated, monitored process (A.1)."""

from pathlib import Path
import argparse
import json
import os
import signal
import threading
import time
import traceback
import multiprocessing

from .artifacts import atomic_json, environment, file_hash, utc_now


class CellContext:
    """Expose counters and atomically publish periodic cell progress."""

    def __init__(self, directory: Path):
        """Bind the cell attempt's output path and latest declared counters."""
        self.directory = directory
        self.values = {}
        self.counters = lambda: {}
        self.child_pids = []
        self.started = time.monotonic()

    def update(self, values: dict) -> None:
        """Retain measured progress; the heartbeat handles durable publication."""
        self.values.update(values)

    def publish(self) -> dict:
        """Snapshot counters and resource use without mutating model state."""
        import psutil

        process = psutil.Process()
        children = []
        pids = set(self.child_pids) | {
            child.pid for child in multiprocessing.active_children()
        }
        for pid in pids:
            try:
                child = psutil.Process(pid)
                children.append(
                    {
                        "pid": pid,
                        "created_at": child.create_time(),
                        "rss_bytes": child.memory_info().rss,
                    }
                )
            except psutil.NoSuchProcess:
                pass
        record = {
            "updated_at": utc_now(),
            "pid": os.getpid(),
            "elapsed_seconds": time.monotonic() - self.started,
            "rss_bytes": process.memory_info().rss,
            "owned_children": children,
            **self.values,
            **self.counters(),
        }
        atomic_json(self.directory / "progress.json", record)
        return record


def main() -> int:
    """Run exactly one resolved cell and write a receipt after all artifacts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--parent", type=int)
    args = parser.parse_args()
    if args.parent and os.getpgrp() != os.getpid():
        raise ValueError("a monitored cell must have its own process group")
    directory = args.directory.resolve()
    cell = json.loads((directory / "config.json").read_text())
    context = CellContext(directory)
    stop = threading.Event()

    def heartbeat() -> None:
        """Exit if the coordinator dies; report owned-worker progress each second."""
        while not stop.wait(1):
            if args.parent and os.getppid() != args.parent:
                os.killpg(os.getpgrp(), signal.SIGTERM)
            context.publish()

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    atomic_json(directory / "environment.json", environment())
    started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        from .tasks import execute

        outcome = execute(cell, context)
    except Exception as error:
        traceback.print_exc()
        outcome = {
            "status": "failed",
            "error_type": type(error).__name__,
            "reason": str(error),
        }
    stop.set()
    thread.join(timeout=6)
    import resource

    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.sys.platform != "darwin":
        maximum_rss *= 1024
    result = {
        "schema": "airline-campaign-cell/1",
        "config_hash": cell["config_hash"],
        "parameters": cell["parameters"],
        "phase": cell["phase"],
        "completed_at": utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu_started,
        "peak_rss_bytes": maximum_rss,
        "planning_crn": True,
        **outcome,
    }
    atomic_json(directory / "results.json", result)
    (directory / "report.md").write_text(
        f"# {cell['experiment_id']} cell {cell['cell_id']}\n\nStatus: {result['status']}\n\n"
        + "```json\n"
        + json.dumps(result, indent=2, allow_nan=False)
        + "\n```\n"
    )
    atomic_json(
        directory / "COMPLETE.json",
        {
            "config_hash": cell["config_hash"],
            "status": result["status"],
            "results_sha256": file_hash(directory / "results.json"),
        },
    )
    return int(result["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
