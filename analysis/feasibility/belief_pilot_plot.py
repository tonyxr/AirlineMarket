"""Plot saved validation contrasts without rerunning learning or validation."""

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from airline_market.campaign.artifacts import file_hash
from airline_market.campaign.runner import completed_attempt


def plot(root):
    root = Path(root)
    cell, = json.loads((root / "cells.json").read_text())
    directory, result = completed_attempt(root / "cells" / cell["cell_id"], cell["config_hash"])
    assert result["status"] == "completed"
    path = directory / "pilot.json"
    assert file_hash(path) == result["pilot_sha256"]
    pilot = json.loads(path.read_text())
    rows = []
    for row in pilot["curve"]:
        d = row["deterministic_minus_fixed"]
        rows.append({"training_cohorts": row["cohorts"], "validation_n": d["n"], "mean": d["mean"], "lower": d["bootstrap95"][0], "upper": d["bootstrap95"][1]})
    with (root / "validation_curve.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    x = [r["training_cohorts"] for r in rows]
    ax.plot(x, [r["mean"] for r in rows], marker="o", color="#0072B2", label="Checkpoint at each validation cadence")
    ax.fill_between(x, [r["lower"] for r in rows], [r["upper"] for r in rows], color="#0072B2", alpha=.18, label="Paired pointwise 95% bootstrap interval")
    ax.axhline(0, color="#444444", linestyle="--", label="Frozen fitted fixed baseline")
    ax.set(xlabel="Completed training cohorts", ylabel="Validation objective minus fixed baseline", title="I0 exact-belief PPO precursor: one initialization seed")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, loc="best")
    fig.text(.5, .012, "256 reused validation cohorts; 2,000 paired bootstrap draws. Descriptive curve, no multi-seed ranking or tuning claim.", ha="center", fontsize=7.7)
    fig.tight_layout(rect=(0, .035, 1, 1))
    fig.savefig(root / "validation_curve.pdf")
    fig.savefig(root / "validation_curve.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    plot(sys.argv[1])
