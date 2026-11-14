"""Collect the F1 frontier and fit only observed, solved exact cells."""

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from airline_market.campaign.artifacts import atomic_json
from airline_market.campaign.runner import completed_attempt


def collect(directory: Path) -> pd.DataFrame:
    """Retain pending/censored/unsupported cells; never impute an exact value."""
    rows = []
    for cell in json.loads((directory / "cells.json").read_text()):
        parameters = cell["parameters"]
        found = completed_attempt(
            directory / "cells" / cell["cell_id"], cell["config_hash"]
        )
        row = {
            "cell_id": cell["cell_id"],
            "generator": cell["generator"],
            **parameters,
            "status": "pending",
            "value": None,
            "nodes": None,
            "seconds": None,
            "peak_rss_bytes": None,
        }
        if found:
            _, result = found
            row.update(
                {
                    key: result.get(key)
                    for key in [
                        "status",
                        "value",
                        "nodes",
                        "peak_rss_bytes",
                        "backend",
                        "reason",
                    ]
                }
            )
            row["seconds"] = result["wall_seconds"]
        else:
            attempts = sorted((directory / "cells" / cell["cell_id"]).glob("attempt-*"))
            if attempts:
                status = attempts[-1] / "status.json"
                progress = attempts[-1] / "progress.json"
                if status.exists():
                    row["status"] = json.loads(status.read_text())["status"]
                if progress.exists():
                    observed = json.loads(progress.read_text())
                    row.update(
                        nodes=observed.get("nodes"),
                        seconds=observed.get("elapsed_seconds"),
                        peak_rss_bytes=observed.get("rss_bytes"),
                    )
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(
        ["generator", "arm", "T", "actions", "cell_id"]
    )
    frame.to_csv(directory / "frontier.csv", index=False, float_format="%.17g")
    return frame


def fit_scaling(frame: pd.DataFrame) -> dict:
    """Fit log nodes = intercept + b*T + c*log(actions), with censoring disclosed."""
    fits = {}
    for (family, arm), group in frame.groupby(["generator", "arm"]):
        solved = group[(group.status == "solved") & (group.nodes > 0)]
        result = {
            "solved": len(solved),
            "total": len(group),
            "status_counts": group.status.value_counts().to_dict(),
            "model": "log(nodes) = intercept + b*T + c*log(actions)",
            "warning": "Fit excludes censored/unsupported/pending cells and can underestimate the unsolved tail; extrapolation is not a solved exact result.",
        }
        if len(solved) >= 4:
            x = np.column_stack(
                [
                    np.ones(len(solved)),
                    solved["T"],
                    np.log(solved.actions),
                ]
            )
            y = np.log(solved.nodes.to_numpy(dtype=float))
            if np.linalg.matrix_rank(x) == 3:
                coefficient = np.linalg.lstsq(x, y, rcond=None)[0]
                residual = y - x @ coefficient
                covariance = (
                    float(residual @ residual)
                    / (len(solved) - 3)
                    * np.linalg.inv(x.T @ x)
                )
                critical = student_t.ppf(0.975, len(solved) - 3)
                result.update(
                    coefficients=coefficient.tolist(),
                    coefficient_95_ci=np.column_stack(
                        [
                            coefficient - critical * np.sqrt(np.diag(covariance)),
                            coefficient + critical * np.sqrt(np.diag(covariance)),
                        ]
                    ).tolist(),
                )
                if family == "toy_hub":
                    target = np.array([1.0, 10.0, np.log(18.0)])
                    result["predicted_nodes_T10_A18"] = float(
                        np.exp(target @ coefficient)
                    )
                    result["prediction_note"] = (
                        "Scheduling must also use measured cost/node and a prediction interval, not this point alone."
                    )
            else:
                result["fit_status"] = "insufficient_design_rank"
        else:
            result["fit_status"] = "insufficient_solved_cells"
        fits[family + "/" + arm] = result
    return fits


def plot(frame: pd.DataFrame, directory: Path) -> None:
    """Draw the observed frontier; pending cells remain visibly pending."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    families = list(frame.generator.unique())
    arms = ["direct", "linked", "unrelated", "oracle", "full_state"]
    colors = {
        "pending": "#d9d9d9",
        "running": "#56B4E9",
        "solved": "#009E73",
        "unsupported": "#999999",
        "node_limit": "#E69F00",
        "time_limit": "#D55E00",
        "memory_limit": "#CC79A7",
        "failed": "#000000",
    }
    figure, axes = plt.subplots(
        len(families), len(arms), figsize=(15, 3 * len(families)), squeeze=False
    )
    for row, family in enumerate(families):
        for column, arm in enumerate(arms):
            axis = axes[row, column]
            group = frame[(frame.generator == family) & (frame.arm == arm)]
            # Several I4 factor corners occupy the same grid coordinate; counts
            # are shown rather than hiding unsolved cases beneath solved ones.
            for (T, actions), cell in group.groupby(["T", "actions"]):
                solved = int((cell.status == "solved").sum())
                complete = solved == len(cell)
                status = (
                    "solved"
                    if complete
                    else "running"
                    if (cell.status == "running").any()
                    else "unsupported"
                    if (cell.status == "unsupported").any()
                    else next(
                        (
                            s
                            for s in [
                                "failed",
                                "memory_limit",
                                "time_limit",
                                "node_limit",
                            ]
                            if (cell.status == s).any()
                        ),
                        "pending",
                    )
                )
                axis.scatter(
                    T,
                    actions,
                    color=colors[status],
                    s=110,
                    edgecolors="white",
                    linewidths=0.5,
                )
                if len(cell) > 1:
                    axis.annotate(
                        f"{solved}/{len(cell)}",
                        (T, actions),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=6,
                    )
            axis.set(
                title=arm, xlabel="Periods T", ylabel="Actions" if column == 0 else ""
            )
            axis.set_xticks(sorted(group["T"].unique()))
            axis.set_yticks(sorted(group["actions"].unique()))
            axis.margins(x=0.12, y=0.18)
            axis.grid(alpha=0.15)
        axes[row, 0].text(
            -0.35,
            0.5,
            family,
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            fontsize=8,
        )
    figure.suptitle(
        "F1 exact-DP frontier — observed status; partial campaign", fontsize=12
    )
    figure.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", color=color, label=status)
            for status, color in colors.items()
        ],
        loc="lower center",
        ncol=4,
        fontsize=8,
    )
    figure.tight_layout(rect=(0.02, 0.07, 1, 0.95))
    figure.savefig(directory / "frontier.pdf")
    figure.savefig(directory / "frontier.png", dpi=140)
    plt.close(figure)


def main() -> None:
    """Refresh CSV/fits and optionally the vector figure from existing evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    frame = collect(args.directory)
    atomic_json(args.directory / "fit.json", fit_scaling(frame))
    if args.plot:
        plot(frame, args.directory)


if __name__ == "__main__":
    main()
