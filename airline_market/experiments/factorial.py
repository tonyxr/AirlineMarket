"""Generate the declared information-value factorial and run explicit cells."""

from itertools import product
from pathlib import Path
import json
from ..pipeline.generators import hub_feeder_family
from ..spec import dump_spec, compile_market
from ..core.hashing import jsonable, spec_hash
from ..core.errors import KernelLimitError
from ..eval import ExperimentRunner
from ..solvers import ExactBeliefDP
from .information_value import toy_config

FACTORS = {
    "connecting_share": [0.1, 0.3, 0.5, 0.7],
    "bottleneck": ["focal", "rival", "both"],
    "capacity_support": [[1, 3], [2, 3], [1, 2, 3]],
    "fare_bin": [0, 10, 25],
    "lag_periods": [0, 1, 2],
    "availability": ["flag", "seats_capped"],
    "rival_controller": ["threshold_linear_rule", "matching", "bid_price_dlp"],
    "demand_factor": ["independent", "latent"],
}


def factorial_parameters():
    """
    Enumerate the declared synthetic factor combinations deterministically (Eqs. 28–32).
    """
    names = list(FACTORS)
    for values in product(*(FACTORS[n] for n in names)):
        yield dict(zip(names, values))


def write_factorial(root="config/factorial"):
    """
    Write reproducible specifications and the full factorial manifest (Eqs. 28–32).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for params in factorial_parameters():
        spec = hub_feeder_family(params)
        name = spec_hash(spec)[:16]
        path = root / (name + ".json")
        dump_spec(spec, path)
        manifest.append({"name": name, "parameters": params, "spec": str(path)})
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run_cells(
    cells,
    out,
    *,
    scenarios=16,
    train_count=8,
    test_count=8,
    periods=4,
    max_exact_nodes=0,
):
    """
    Execute explicit factorial cells with paired evaluation and labeled exact limits
    (Eqs. 28–32).
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table = []
    for index, params in enumerate(cells):
        spec = hub_feeder_family({"periods": periods, **params})
        path = out / f"cell_{index}_spec.json"
        dump_spec(spec, path)
        cfg = toy_config(scenarios=scenarios)
        cfg.update(spec=str(path), name="factorial_" + str(index))
        cfg["seeds"] = {
            "train": [100, train_count],
            "validation": [],
            "test": [10000, test_count],
            "planning_master": 773,
        }
        result = ExperimentRunner(cfg).run(out / f"cell_{index}")
        row = {
            "parameters": params,
            "linked_minus_direct": result["contrasts"]["linked_minus_direct"][
                "objective"
            ],
            "unrelated_minus_direct": result["contrasts"]["unrelated_minus_direct"][
                "objective"
            ],
            "oracle_minus_direct": result["contrasts"]["oracle_minus_direct"][
                "objective"
            ],
            "exact_status": "not_requested",
            "factorization_loss": None,
        }
        if max_exact_nodes:
            try:
                c = compile_market(spec)
                values = {
                    a: ExactBeliefDP(
                        c, "A", a, max_nodes=max_exact_nodes, node_key="belief"
                    )
                    .solve()
                    .value
                    for a in ["direct", "linked"]
                }
                row.update(
                    exact_status="completed",
                    exact_delta=values["linked"] - values["direct"],
                )
            except KernelLimitError as error:
                row.update(exact_status="enumeration_limit", exact_error=str(error))
        table.append(row)
        (out / "table.json").write_text(json.dumps(jsonable(table), indent=2) + "\n")
    return table


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--write-full", action="store_true")
    p.add_argument("--out", default="results/factorial_smoke")
    a = p.parse_args()
    if a.write_full:
        print(len(write_factorial(a.out)))
    else:
        run_cells(
            [{"connecting_share": x, "n_spokes": 1} for x in [0.1, 0.3, 0.5, 0.7]],
            a.out,
            scenarios=4,
            train_count=4,
            test_count=4,
            periods=3,
        )
