"""Registered engineering curriculum, not calibrated/full-budget TR training."""

import json
from pathlib import Path

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.learn.curriculum_ppo import load_domains, train_curriculum


def run(cell, context):
    p = cell["parameters"]
    path = Path(p["manifest"])
    if file_hash(path) != p["manifest_sha256"]:
        raise ValueError("changed frozen curriculum manifest")
    members = json.loads(path.read_text())["members"]
    domains = load_domains(members)
    latest = {"stage": "setup"}
    context.counters = lambda: latest.copy()
    result = train_curriculum(domains, p["training"], context.directory, latest.update)
    checkpoints = [
        {
            "file": name,
            "bytes": (context.directory / name).stat().st_size,
            "sha256": file_hash(context.directory / name),
            "storage": "local Git-ignored weights",
        }
        for name in ("best.pt", "last.pt")
    ]
    atomic_json(
        context.directory / "checkpoint_manifest.json",
        {
            "checkpoints": checkpoints,
            "second_machine_reproduction": False,
            "note": "New checkpoint snapshot; source game receipts seal their results separately.",
        },
    )
    learning = json.loads((context.directory / "learning.json").read_text())
    atomic_json(
        context.directory / "curriculum_audit.json",
        {
            "assignments": learning["assignments"],
            "validation_assignments": [
                v["assignments"] for v in learning["validation"]
            ],
            "training_members": [d["id"] for d in domains if d["role"] == "train"],
            "heldout_members": [d["id"] for d in domains if d["role"] == "heldout"],
            "scope": "Synthetic engineering member set; not the calibrated24train/8heldout ensemble or held-out rival evaluation.",
        },
    )
    return {
        **result,
        "manifest_sha256": file_hash(path),
        "learning_sha256": file_hash(context.directory / "learning.json"),
        "checkpoint_manifest_sha256": file_hash(
            context.directory / "checkpoint_manifest.json"
        ),
        "curriculum_audit_sha256": file_hash(
            context.directory / "curriculum_audit.json"
        ),
    }
