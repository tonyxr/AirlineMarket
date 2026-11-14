"""Measure advertised compressed bytes without downloading duplicate bodies."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).parent))
from download import atomic, now


def probe(job):
    """Record available lengths; a missing HEAD response stays unknown."""
    p = subprocess.run(
        ["curl", "-sSIL", "--connect-timeout", "15", "--max-time", "40", job["url"]],
        capture_output=True,
        text=True,
    )
    lengths = re.findall(r"^content-length:\s*(\d+)", p.stdout, re.I | re.M)
    codes = re.findall(r"^HTTP/\S+\s+(\d+)", p.stdout, re.M)
    return {
        "id": job["id"],
        "source": job["source"],
        "status": int(codes[-1]) if codes else None,
        "advertised_bytes": int(lengths[-1])
        if lengths and codes and codes[-1] == "200"
        else job.get("expected_bytes"),
    }


if __name__ == "__main__":
    jobs = []
    for name in ("priority_2022", "db1c", "historical_archives"):
        jobs.extend(
            json.loads(Path(f"results/data_acquisition/{name}.json").read_text())[
                "jobs"
            ]
        )
    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(probe, jobs))
    total = sum(r["advertised_bytes"] or 0 for r in results)
    groups = {}
    for r in results:
        groups[r["source"]] = groups.get(r["source"], 0) + (r["advertised_bytes"] or 0)
    atomic(
        "results/data_acquisition/size_preflight.json",
        {
            "measured_at": now(),
            "bytes_known": total,
            "gib_known": total / 1024**3,
            "unknown": sum(r["advertised_bytes"] is None for r in results),
            "groups_bytes": groups,
            "files": results,
            "note": "Excludes form-generated extracts and derived parquet; unknown lengths are not zero-sized files.",
        },
    )
    print(
        json.dumps(
            {
                "gib_known": total / 1024**3,
                "unknown": sum(r["advertised_bytes"] is None for r in results),
                "groups_bytes": groups,
            },
            indent=2,
        )
    )
