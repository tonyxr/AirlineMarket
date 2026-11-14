"""Enforce the plan's prospective test split before any cohort is opened (§2.2)."""

from pathlib import Path
import json
import re
import subprocess


def validate_seed_release(
    seeds, phase: str = "development", *, check_freeze: bool = True
) -> None:
    """Reject reserved test seeds outside a clean, tagged final experiment."""
    if phase not in ("development", "final"):
        raise ValueError("experiment phase must be development or final")
    reserved = [
        seed
        for split in [seeds.train, seeds.validation, seeds.test]
        for seed in split
        if 300000 <= seed < 400000
    ]
    if not reserved:
        return
    if phase != "final":
        raise ValueError("test seeds 300000–399999 are sealed until phase='final'")
    bounds = [
        (seeds.train, 100000, 200000),
        (seeds.validation, 200000, 300000),
        (seeds.test, 300000, 400000),
    ]
    if any(
        not all(low <= seed < high for seed in split) for split, low, high in bounds
    ):
        raise ValueError("final experiment seeds violate the registered split ranges")
    if not check_freeze:
        return
    tags = subprocess.check_output(
        ["git", "tag", "--points-at", "HEAD"], text=True
    ).splitlines()
    if not any(
        tag == "v1.0-experiments" or re.fullmatch(r"v1\.0\.\d+", tag) for tag in tags
    ):
        raise ValueError("final experiments require the experiment freeze tag at HEAD")
    frozen_paths = [
        "airline_market",
        "config",
        "scripts",
        "analysis",
        "tests",
        "pyproject.toml",
        "requirements-lock.txt",
    ]
    dirty = subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD", "--", *frozen_paths], text=True
    )
    untracked_code = subprocess.check_output(
        [
            "git",
            "ls-files",
            "--others",
            "--exclude-standard",
            "airline_market",
            "config",
            "scripts",
            "analysis",
            "tests",
        ],
        text=True,
    )
    if dirty or untracked_code:
        raise ValueError(
            "final experiments require clean frozen source/config files and no untracked implementation/config files"
        )


def audit_manifest_sealing(root: str | Path = "results") -> list[dict]:
    """Find registered cohort results that used reserved seeds before final release."""
    violations = []
    visited = set()
    for manifest in Path(root).rglob("*manifest.jsonl"):
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            output = record.get("output")
            if not output:
                continue
            path = Path(output) / "results.json"
            if path in visited or not path.exists():
                continue
            visited.add(path)
            result = json.loads(path.read_text())
            seeds = result.get("seeds", {})
            reserved = [
                seed
                for key in ("train", "validation", "test")
                for seed in seeds.get(key, [])
                if 300000 <= seed < 400000
            ]
            if reserved and result.get("phase") != "final":
                violations.append(
                    {
                        "path": str(path),
                        "reason": "reserved seeds in a non-final registered run",
                        "reserved_seed_count": len(reserved),
                    }
                )
    return violations
