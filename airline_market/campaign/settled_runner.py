"""Opt-in campaign aggregation after cell futures settle (plan §2.6/R0.4).

The original worker, cell launcher, limits and scientific tasks are inherited
unchanged. Only this coordinator's aggregation schedule and receipt reader
change. A settled future is the publication barrier; no hash retry or repair is
used. This module does not alter running campaigns or the default CLI.
"""

import argparse
import fcntl
import hashlib
import json
import signal
from collections.abc import Collection
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from ..core.hashing import canonical_json
from .artifacts import atomic_json, environment, utc_now
from .design import load_campaign
from .runner import CampaignRunner, campaign_status


def completed_snapshot(cell_directory: Path, digest: str) -> tuple[Path, dict] | None:
    """Validate and decode the same immutable byte buffer, without retries.

    Call only after that cell's publisher future has returned, or on a fully
    quiescent directory. Stable damage/config mismatch remains an error.
    """
    for attempt in sorted(cell_directory.glob("attempt-*"), reverse=True):
        receipt_path = attempt / "COMPLETE.json"
        if not receipt_path.exists():
            continue
        receipt = json.loads(receipt_path.read_bytes())
        if receipt["config_hash"] != digest:
            raise ValueError(f"cell config mismatch: {attempt}")
        result_bytes = (attempt / "results.json").read_bytes()
        if hashlib.sha256(result_bytes).hexdigest() != receipt["results_sha256"]:
            raise ValueError(f"completed result is damaged: {attempt}")
        result = json.loads(result_bytes)
        if result["config_hash"] != digest or result["status"] != receipt["status"]:
            raise ValueError(f"completed result/receipt identity mismatch: {attempt}")
        return attempt, result
    return None


def aggregate_settled(directory: Path, settled_ids: Collection[str]) -> list[dict]:
    """Aggregate only futures returned to this coordinator, using native columns.

    Caller owns the coordinator lock and supplies its completed-future IDs.
    Active/unsubmitted publishers are never opened. No saved result is modified.
    """
    import pandas as pd

    cells = json.loads((directory / "cells.json").read_bytes())
    eligible = set(settled_ids)
    if eligible - {cell["cell_id"] for cell in cells}:
        raise ValueError("unknown settled cell ID")
    rows, resources = [], []
    omitted = {
        "completed_at",
        "wall_seconds",
        "cpu_seconds",
        "peak_rss_bytes",
        "solve_seconds",
        "ms_per_node",
        "measurement_note",
        "error_type",
        "reason",
    }
    for cell in cells:
        if cell["cell_id"] not in eligible:
            continue
        found = completed_snapshot(
            directory / "cells" / cell["cell_id"], cell["config_hash"]
        )
        if not found:
            continue
        attempt, result = found
        row = {
            key: cell[key] for key in ["cell_id", "config_hash", "generator", "task"]
        }
        row.update(
            {
                "parameter/" + key: canonical_json(value)
                if isinstance(value, (dict, list))
                else value
                for key, value in cell["parameters"].items()
            }
        )
        row.update(
            {
                key: canonical_json(value) if isinstance(value, (dict, list)) else value
                for key, value in result.items()
                if key not in omitted | {"config_hash", "parameters"}
            }
        )
        rows.append(row)
        resources.append(
            {
                "cell_id": cell["cell_id"],
                "attempt": attempt.name,
                **{
                    key: result.get(key)
                    for key in [
                        "wall_seconds",
                        "cpu_seconds",
                        "peak_rss_bytes",
                        "solve_seconds",
                        "ms_per_node",
                        "reason",
                    ]
                },
            }
        )
    rows.sort(key=lambda row: row["cell_id"])
    for name, records in [
        ("cells", rows),
        ("resources", sorted(resources, key=lambda row: row["cell_id"])),
    ]:
        frame = pd.DataFrame(records)
        frame = frame.reindex(sorted(frame.columns), axis=1)
        temporary = directory / (name + ".csv.tmp")
        frame.to_csv(temporary, index=False, float_format="%.17g", lineterminator="\n")
        temporary.replace(directory / (name + ".csv"))
        if name == "cells":
            temporary = directory / "cells.parquet.tmp"
            frame.to_parquet(temporary, index=False)
            temporary.replace(directory / "cells.parquet")
    return rows


class SettledCampaignRunner(CampaignRunner):
    """Use returned futures as the result/receipt publication barrier.

    Inherited run_cell/_run_cell retain native bounded subprocesses, owned-child
    termination, memory reservations, strict duplicate checks and seal writes.
    """

    def run(self) -> dict:
        """Run only a new opt-in protocol root, retaining native resume checks."""
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / ".coordinator.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "campaign already has an active coordinator"
                ) from error
            protocol_path = self.directory / "publication_protocol.json"
            protocol = {
                "schema": "airline-settled-campaign/1",
                "aggregation_barrier": "returned_cell_future",
                "reader": "same_bytes_hash_and_decode",
                "hash_retries": 0,
            }
            existing = self.directory / "config.json"
            if existing.exists():
                if (
                    not protocol_path.exists()
                    or json.loads(protocol_path.read_bytes()) != protocol
                ):
                    raise ValueError("use a new settled-protocol output directory")
                if json.loads(existing.read_bytes()) != self.config:
                    raise ValueError(
                        "resume directory belongs to a different campaign config"
                    )
            provenance = environment()
            previous_environment = self.directory / "environment.json"
            if previous_environment.exists():
                previous_sources = json.loads(previous_environment.read_bytes()).get(
                    "source_hashes", {}
                )
                changed = [
                    name
                    for name, digest in previous_sources.items()
                    if name in provenance["source_hashes"]
                    and provenance["source_hashes"][name] != digest
                ]
                if changed:
                    raise ValueError(
                        "implementation changed; use a new implementation_revision and output directory: "
                        + ", ".join(changed)
                    )
            atomic_json(protocol_path, protocol)
            atomic_json(existing, self.config)
            atomic_json(self.directory / "cells.json", self.cells)
            self.git_sha = provenance["git_sha"]
            atomic_json(self.directory / "environment.json", provenance)
            previous = {}

            def interrupt(signum, frame):
                self.stop.set()

            for signum in [signal.SIGTERM, signal.SIGINT]:
                previous[signum] = signal.signal(signum, interrupt)
            settled = set()
            self.snapshot("running")
            try:
                with ThreadPoolExecutor(
                    max_workers=int(self.config.get("workers", 1))
                ) as pool:
                    pending = {
                        pool.submit(self.run_cell, cell): cell["cell_id"]
                        for cell in self.cells
                    }
                    try:
                        while pending:
                            done, _ = wait(
                                pending, timeout=5, return_when=FIRST_COMPLETED
                            )
                            for future in done:
                                future.result()
                                settled.add(pending.pop(future))
                            self.snapshot(
                                "interrupting" if self.stop.is_set() else "running"
                            )
                            if done:
                                aggregate_settled(self.directory, settled)
                                atomic_json(
                                    self.directory / "settled_cells.json",
                                    {
                                        "updated_at": utc_now(),
                                        "cell_ids": sorted(settled),
                                    },
                                )
                    except BaseException:
                        self.stop.set()
                        raise
                aggregate_settled(self.directory, settled)
                atomic_json(
                    self.directory / "settled_cells.json",
                    {"updated_at": utc_now(), "cell_ids": sorted(settled)},
                )
                self.snapshot("interrupted" if self.stop.is_set() else "finished")
            except Exception:
                self.snapshot("failed")
                raise
            finally:
                self.stop.set()
                for process in list(self.children.copy().values()):
                    self.terminate(process)
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
        return campaign_status(self.directory)


def main() -> None:
    """Run/resume an explicit new protocol root; leave default CLI unchanged."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    config = load_campaign(args.config)
    print(json.dumps(SettledCampaignRunner(config, args.out).run()))


if __name__ == "__main__":
    main()
