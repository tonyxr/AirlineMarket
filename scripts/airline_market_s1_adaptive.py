"""S1/R2: start after the large reservation; share memory with the finishing F5 cell.

Revision: 2 (Nov 16, 2026): W4/R0.4 uses the canonical campaign runner.
"""

import argparse
import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from airline_market.campaign import runner
from airline_market.campaign.artifacts import atomic_json
from airline_market.campaign.design import load_campaign
from airline_market.campaign.runner import CampaignRunner
from airline_market.experiments.study_campaign import StudyProcesses

ROOT = Path(__file__).resolve().parents[1]
F1 = ROOT / "results/v2_transition/drain_F1.state.json"
F5 = ROOT / "results/v2_transition/drain_F5.state.json"
STATE = ROOT / "results/v2_transition/S1_adaptive_launch.json"


def available_budget(f1_status, f5_status):
    """Reserve F4=1GiB and event streaming=.5GiB; F5 retains2GiB until drained."""
    if f1_status != "stopped_v2":
        return 0.0
    return 10.5 if f5_status == "stopped_v2" else 8.5


class AdaptiveScreenRunner(CampaignRunner):
    """Keep native worker/settled aggregation; change only admission reservation."""

    def update_budget(self):
        budget = available_budget(
            json.loads(F1.read_text())["status"], json.loads(F5.read_text())["status"]
        )
        if budget <= 0:
            raise RuntimeError("large legacy reservation has not released")
        budget = min(float(self.config["memory_budget_gb"]), budget)
        if getattr(self, "_reported_budget", None) == budget:
            return
        with self.condition:
            self.memory_budget = budget
            self.condition.notify_all()
        self._reported_budget = budget
        atomic_json(
            self.directory / "runtime_admission.json",
            {
                "plan_id": "S1/R2",
                "effective_budget_gb": self.memory_budget,
                "maximum_configured_budget_gb": self.config["memory_budget_gb"],
                "external_reservations_gb": 12.0 - budget,
                "worker_pool": 8,
                "maximum_admitted_cells": int(self.memory_budget // 1.5),
                "control_source": str(Path(__file__).resolve()),
                "control_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
            },
        )

    def run(self):
        self.update_budget()
        stop = threading.Event()

        def watch():
            while not stop.wait(1):
                self.update_budget()

        thread = threading.Thread(target=watch, daemon=True)
        thread.start()
        try:
            return super().run()
        finally:
            stop.set()
            thread.join(timeout=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_campaign(config_path)
    state = {
        "plan_id": "S1/R2",
        "status": "waiting_for_F1_reservation_release",
        "pid": os.getpid(),
        "created": psutil.Process().create_time(),
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "settings": 52,
        "arm_jobs": 260,
        "workers": 8,
        "maximum_reservation_gb": 10.5,
    }
    atomic_json(STATE, state)
    while (
        available_budget(
            json.loads(F1.read_text())["status"], json.loads(F5.read_text())["status"]
        )
        == 0
    ):
        time.sleep(10)
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != state["config_sha256"]:
        raise ValueError("S1 config changed")
    root = Path(config["out"])
    root.mkdir(parents=True, exist_ok=True)
    supervisor = {
        "pid": os.getpid(),
        "process_created_at": psutil.Process().create_time(),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "command": psutil.Process().cmdline(),
        "config": str(config_path),
        "control_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    atomic_json(root / "supervisor.json", supervisor)
    runner.subprocess = StudyProcesses()
    state.update(status="running_adaptive_coordinator", launch_record=supervisor)
    atomic_json(STATE, state)
    result = AdaptiveScreenRunner(config, root).run()
    state.update(status="finished", campaign_status=result)
    atomic_json(STATE, state)


if __name__ == "__main__":
    main()
