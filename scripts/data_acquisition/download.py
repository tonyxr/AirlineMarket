"""Resumable public-data acquisition, independent of live experiment code."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import csv
import fcntl
import hashlib
import io
import json
import os
import shutil
import subprocess
import threading
import zipfile

ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT / "results/data_acquisition"
LOCK = threading.Lock()


def now():
    """UTC acquisition timestamp."""
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    """Publish complete JSON records atomically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def digest(path):
    """Hash an archive without loading it into memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def describe(path, suffix=None):
    """Inspect actual archive members/CSV headers; full row validation is separate."""
    path = Path(path)
    members = []
    suffix = (suffix or path.suffix).lower()
    if suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                row = {
                    "member": info.filename,
                    "bytes_uncompressed": info.file_size,
                    "crc32": info.CRC,
                }
                if info.filename.lower().endswith((".csv", ".txt")):
                    with archive.open(info) as stream:
                        header = (
                            stream.readline(1024 * 1024)
                            .decode("utf-8-sig", errors="replace")
                            .strip()
                        )
                    row["header"] = next(csv.reader([header]))
                members.append(row)
    elif suffix == ".csv":
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
            members = [{"member": path.name, "header": next(csv.reader(f))}]
    elif suffix == ".pdf":
        with path.open("rb") as f:
            if f.read(5) != b"%PDF-":
                raise ValueError("download is not a PDF")
    else:
        with path.open("rb") as f:
            if f.read(1000).lstrip().lower().startswith((b"<!doctype html", b"<html")):
                raise ValueError("server returned HTML instead of data")
    return members


def fetch(job):
    """Keep original compressed data, resume partial bytes, and preserve failures."""
    dest = ROOT / job["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    receipt = RECORDS / "files" / (job["id"] + ".json")
    if receipt.exists() and dest.exists():
        old = json.loads(receipt.read_text())
        if (
            old.get("status") == "downloaded"
            and old.get("bytes") == dest.stat().st_size
            and old.get("source_url") == job["url"]
        ):
            return old
    started = now()
    partial = dest.with_suffix(dest.suffix + ".part")
    record = {
        **job,
        "source_url": job["url"],
        "started_at": started,
        "status": "downloading",
        "rows": None,
        "date_range": None,
    }
    atomic(receipt, record)
    try:
        expected = job.get("expected_bytes", 0)
        need = max(0, expected - (partial.stat().st_size if partial.exists() else 0))
        free = shutil.disk_usage(dest.parent).free
        if free - need < 8 * 1024**3:
            record.update(
                status="storage_blocked",
                reason="Less than 8 GiB would remain after this archive",
                free_bytes=free,
                expected_additional_bytes=need,
            )
            atomic(receipt, record)
            return record
        if not dest.exists():
            cmd = [
                "curl",
                "-fLSs",
                "--retry",
                "2",
                "--retry-delay",
                "3",
                "--connect-timeout",
                "25",
                "--max-time",
                "14400",
                "--speed-time",
                "120",
                "--speed-limit",
                "1024",
                "-C",
                "-",
                "-A",
                "AirlineMarket-research/1.0",
                "-o",
                str(partial),
                job["url"],
            ]
            with (RECORDS / (job["id"] + ".curl.log")).open("a") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            # A normal HTML login/error page must never become a completed archive.
            members = describe(partial, dest.suffix)
            partial.replace(dest)
        else:
            members = describe(dest)
        if dest.stat().st_size == 0:
            raise ValueError("empty download")
        record.update(
            status="downloaded",
            completed_at=now(),
            bytes=dest.stat().st_size,
            sha256=digest(dest),
            members=members,
            integrity="HTTP completed; archive directory and member headers inspected; full stream CRC/row validation pending",
        )
        print(
            f"downloaded {job['id']}: {dest.stat().st_size / 1024**2:.1f} MiB",
            flush=True,
        )
    except Exception as error:
        record.update(status="failed", completed_at=now(), reason=str(error))
        print(f"failed {job['id']}: {error}", flush=True)
    atomic(receipt, record)
    return record


def inventory():
    """Keep raw acquisition separate from loader/calibration readiness."""
    files = [
        json.loads(p.read_text()) for p in sorted((RECORDS / "files").glob("*.json"))
    ]
    counts = {}
    for r in files:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    result = {
        "schema": "airline-data-inventory/2",
        "updated_at": now(),
        "status": "acquisition_in_progress",
        "counts": counts,
        "files": files,
        "note": "Downloaded is not loader-verified or calibrated. Archives are retained compressed. Null rows/date_range await streaming validation; unavailable releases and failures remain explicit.",
    }
    atomic(ROOT / "data/INVENTORY.json", result)
    return result


def run(manifest, workers=2):
    """One coordinator per immutable manifest; completed URL/path pairs are reused."""
    RECORDS.mkdir(parents=True, exist_ok=True)
    name = Path(manifest).stem
    with (RECORDS / (name + ".lock")).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = json.loads(Path(manifest).read_text())["jobs"]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, job) for job in jobs]
            for future in as_completed(futures):
                future.result()
                with LOCK:
                    result = inventory()
                    atomic(
                        RECORDS / (name + ".status.json"),
                        {
                            "pid": os.getpid(),
                            "updated_at": now(),
                            "configured_jobs": len(jobs),
                            "counts": result["counts"],
                        },
                    )
        result = inventory()
        atomic(
            RECORDS / (name + ".status.json"),
            {
                "status": "finished",
                "updated_at": now(),
                "configured_jobs": len(jobs),
                "counts": result["counts"],
            },
        )


def main():
    """Execute only the specified acquisition manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    run(args.manifest, args.workers)


if __name__ == "__main__":
    main()
