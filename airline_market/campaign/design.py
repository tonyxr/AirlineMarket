"""Deterministic full/list/fractional designs and resolved cell hashes (A.1)."""

from itertools import combinations, product
from pathlib import Path
import copy
import json

from ..core.hashing import config_hash


def load_campaign(path: str | Path) -> dict:
    """Load one campaign and resolve only explicitly named budget references."""
    value = json.loads(Path(path).read_text())
    if value.get("schema") != "airline-campaign/1":
        raise ValueError("unknown campaign schema")
    budget = None

    def resolve(item: object) -> object:
        nonlocal budget
        if isinstance(item, str) and item.startswith("budget:"):
            if budget is None:
                budget = json.loads(
                    Path(
                        value.get("budget_file", "config/campaigns/budget.json")
                    ).read_text()
                )
            found = budget
            for key in item.removeprefix("budget:").split("."):
                found = found[key]
            if found is None:
                raise ValueError(f"unmeasured budget reference: {item}")
            return copy.deepcopy(found)
        if isinstance(item, dict):
            return {key: resolve(child) for key, child in item.items()}
        if isinstance(item, list):
            return [resolve(child) for child in item]
        return item

    return resolve(value)


def fractional_design(factors: dict, base_factors: int = 4) -> list[dict]:
    """Generate a two-level resolution-IV design using odd-parity columns.

    Distinct nonzero odd-parity columns cannot have a product relation of
    length three, so the defining relation has minimum length four.
    Three-level factors require an explicitly supplied screening design.
    """
    names = list(factors)
    if any(len(factors[name]) != 2 for name in names):
        raise ValueError("fractional design requires two levels per factor")
    columns = [(k,) for k in range(base_factors)]
    columns += [
        subset
        for size in range(3, base_factors + 1, 2)
        for subset in combinations(range(base_factors), size)
    ]
    if len(names) > len(columns):
        raise ValueError("not enough resolution-IV columns; increase base_factors")
    rows = []
    for signs in product((-1, 1), repeat=base_factors):
        row = {}
        for name, subset in zip(names, columns):
            sign = 1
            for k in subset:
                sign *= signs[k]
            row[name] = factors[name][int(sign > 0)]
        rows.append(row)
    return rows


def expand_campaign(config: dict) -> list[dict]:
    """Resolve each scientific cell once, preserving declared iteration order."""
    design = config.get("design", "full")
    factors = config.get("factors", {})
    if design == "list":
        rows = config["cells"]
    elif design == "full":
        names = list(factors)
        rows = [
            dict(zip(names, values))
            for values in product(*(factors[name] for name in names))
        ]
    elif design == "fractional" or design.startswith("fractional:"):
        rows = fractional_design(factors, int(config.get("base_factors", 4)))
        rows += config.get("center_points", [])
    else:
        raise ValueError(f"unsupported campaign design {design}")
    cells = []
    seen = set()
    for row in rows:
        row = copy.deepcopy(row)
        generator = row.pop("instance_generator", config.get("generator", "toy_hub"))
        cell = {
            "experiment_id": config["id"],
            "phase": config.get("phase", 1),
            "hypothesis": config.get("hypothesis"),
            "implementation_revision": config.get("implementation_revision"),
            "task": config.get("task", "exact_dp"),
            "generator": generator,
            "parameters": {**config.get("fixed", {}), **copy.deepcopy(row)},
            "per_cell": copy.deepcopy(config.get("per_cell", {})),
            "limits": copy.deepcopy(
                config.get("limits", config.get("per_cell", {}).get("limits", {}))
            ),
        }
        digest = config_hash(cell)
        if digest in seen:
            continue
        seen.add(digest)
        cells.append({**cell, "config_hash": digest, "cell_id": digest[:16]})
    return cells
