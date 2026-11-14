from pathlib import Path
import csv, json, platform, hashlib
import numpy as np
from ..core.hashing import jsonable
from ..core.registry import append_manifest
from ..vendor.ride_response.paired import PairedEstimate
from ..core.rng import StreamRegistry


def statistics(values, *, seed=0, bootstrap=2000):
    """
    Summarize cohort values with normal and seed-cluster bootstrap intervals (Eqs.
    20–22, 29).
    """
    e = PairedEstimate.from_values(values)
    se = e.standard_error
    finite = np.isfinite(se)
    a = np.asarray(values)
    rng = StreamRegistry(seed, "boot").generator()
    means = a[rng.integers(len(a), size=(bootstrap, len(a)))].mean(1)
    return {
        "n": len(a),
        "mean": e.mean,
        "standard_error": se if finite else None,
        "ci95_normal": [e.mean - 1.96 * se, e.mean + 1.96 * se] if finite else None,
        "ci95_bootstrap": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def paired_statistics(left, right, metric, **kwargs):
    """Estimate paired differences after verifying cohort alignment (Eqs. 20–22, 29)."""
    if [r["seed"] for r in left] != [r["seed"] for r in right]:
        raise ValueError("paired cohorts are misaligned")
    return statistics([a[metric] - b[metric] for a, b in zip(left, right)], **kwargs)


def save_results(result, out):
    """
    Write results, episode tables, a report and provenance manifest (Eqs. 20–22, 29).
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).parents[1]
    result["source_sha256"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }
    result["runtime"] = {"python": platform.python_version(), "numpy": np.__version__}
    (out / "results.json").write_text(
        json.dumps(jsonable(result), indent=2, allow_nan=False) + "\n"
    )
    rows = [
        {
            "method": m,
            **{k: v for k, v in r.items() if isinstance(v, (int, float, str))},
        }
        for m, rr in result["episodes"].items()
        for r in rr
    ]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out / "episodes.csv").open("w") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)
    lines = [
        "# Airline market experiment",
        "",
        "Material Passport: synthetic computational experiment; common cohort tapes; separate training, validation and test seeds. No empirical or equilibrium claim.",
        "",
        "| Method | Profit | Share | Objective |",
        "| --- | ---: | ---: | ---: |",
    ]
    for m, s in result["summary"].items():
        lines.append(
            f"| {m} | {s['profit']['mean']:.4f} | {s['share']['mean']:.4f} | {s['objective']['mean']:.6f} |"
        )
    lines += [
        "",
        "| Paired contrast | Objective difference | Normal 95% CI | Bootstrap 95% CI |",
        "| --- | ---: | --- | --- |",
    ]
    for m, s in result["contrasts"].items():
        a = s["objective"]
        lines.append(
            f"| {m} | {a['mean']:.6f} | {a['ci95_normal']} | {a['ci95_bootstrap']} |"
        )
    lines += [
        "",
        result["scope"],
        "",
        "Exact values are reported only for successfully enumerated finite instances. Scenario rollout estimates are policy comparisons, not optimal information values. Split selection estimates the selected candidate on held-out scenarios; it does not guarantee improvement. Consumer surplus uses the declared random-utility scale.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    if result["config"].get("trace"):
        with (out / "trace.jsonl").open("w") as f:
            for method, episodes in result["episodes"].items():
                for row in episodes:
                    for decision in row.get("decisions", []):
                        f.write(
                            json.dumps(
                                {"method": method, "seed": row["seed"], **decision}
                            )
                            + "\n"
                        )
    append_manifest(
        str(out.parent),
        {
            "kind": "experiment",
            "output": str(out.resolve()),
            "spec_hash": result["spec_hash"],
            "config_hash": result["config_hash"],
        },
    )
    return out
