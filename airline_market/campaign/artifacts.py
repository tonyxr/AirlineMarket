"""Durable experiment artifacts and measured provenance (validation plan §2.6)."""

from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import fcntl
import hashlib
import json
import os
import platform
import subprocess
import tempfile

from ..core.hashing import jsonable


def utc_now() -> str:
    """Return an explicit UTC timestamp for an observed event."""
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: str | Path, data: object) -> None:
    """Publish complete finite JSON without exposing a partial cell receipt."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(jsonable(data), indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=target.parent, delete=False) as out:
        temporary = Path(out.name)
        out.write(encoded)
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(target)


def file_hash(path: str | Path) -> str:
    """Hash bytes by streaming, including large source-data files."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment() -> dict:
    """Capture actual code, software, hardware and per-worker thread settings."""
    import psutil

    def command(args: list[str]) -> str | None:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    packages = {}
    for name in [
        "numpy",
        "scipy",
        "torch",
        "pandas",
        "pyarrow",
        "hypothesis",
        "psutil",
        "matplotlib",
    ]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    try:
        cores = psutil.cpu_count(logical=False)
        memory = psutil.virtual_memory().total
    except (PermissionError, OSError) as error:
        cores, memory = None, None
        packages["hardware_probe_error"] = str(error)
    sources = {
        str(path): file_hash(path)
        for path in sorted(Path("airline_market").rglob("*.py"))
    }
    return {
        "captured_at": utc_now(),
        "git_sha": command(["git", "rev-parse", "HEAD"]),
        "tracked_dirty": bool(command(["git", "diff", "--name-only", "HEAD"])),
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": command(["sysctl", "-n", "machdep.cpu.brand_string"])
        if platform.system() == "Darwin"
        else platform.processor(),
        "physical_cores": cores,
        "logical_cores": os.cpu_count(),
        "ram_bytes": memory,
        "packages": packages,
        "thread_settings": {
            key: os.environ.get(key)
            for key in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ]
        },
        "pip_freeze": command([os.sys.executable, "-m", "pip", "freeze"]),
        "source_hashes": sources,
    }


def log_run(record: dict, path: str | Path = "EXPERIMENT_LOG.md") -> None:
    """Append one completed attempt under a cross-process file lock."""
    fields = [
        record.get(key, "")
        for key in [
            "date",
            "id",
            "run_id",
            "git_sha",
            "status",
            "seconds",
            "headline",
            "deviations",
        ]
    ]
    line = (
        "| "
        + " | ".join(
            str(value).replace("|", "/").replace("\n", " ") for value in fields
        )
        + " |\n"
    )
    with Path(path).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(line)
        handle.flush()
