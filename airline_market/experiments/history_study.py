"""Isolated history-state frontier and independent random-micro reference checks."""

from pathlib import Path
import time
import numpy as np

from ..campaign.artifacts import atomic_json, file_hash
from ..campaign.instances import build_instance
from ..core.errors import KernelLimitError
from ..core.rng import StreamRegistry
from ..sim.finite import UnsupportedFiniteModel
from ..solvers.exact_dp import ExactBeliefDP
from ..solvers.oracle_dp import OracleDP
from ..solvers.lagged_dp import history_dp
from ..spec import compile_market, dump_spec, load_spec


def solve(cell, context):
    """Solve the original frozen physical spec with the new exact history state."""
    p = cell["parameters"]
    path = Path(p["source_spec"])
    if file_hash(path) != p["source_spec_sha256"]:
        raise ValueError("history frontier frozen physical spec changed")
    compiled = compile_market(load_spec(path))
    dump_spec(compiled.spec, context.directory / "spec.json")
    max_nodes = int(cell["limits"]["max_nodes"])
    try:
        dp = history_dp(compiled, p.get("focal", "A"), p["arm"], max_nodes=max_nodes)
    except UnsupportedFiniteModel as error:
        return {"status": "unsupported", "value": None, "reason": str(error),
                "spec_hash": compiled.spec_hash}
    finite = dp.finite

    def counters():
        """Separate memo entries and conditional vectors, retaining per-period work."""
        info = finite.tail_vectors.cache_info()
        return {"nodes": finite.memo_entries(), "nodes_per_period": dict(finite.counts),
                "bellman_entries": len(finite.cache), "tail_vector_entries": info.currsize,
                "history_targets": sorted(finite.model.history_targets),
                "backend": "compact_history_lag1"}

    context.counters = counters
    started = time.perf_counter()
    try:
        value = dp.solve().value
        outcome = {"status": "solved", "value": value, "exact": True}
    except KernelLimitError as error:
        outcome = {"status": "node_limit", "value": None, "exact": False,
                   "reason": str(error)}
    return {**outcome, **counters(), "spec_hash": compiled.spec_hash,
            "T": compiled.spec.horizon.n_periods, "actions": len(dp.grid),
            "solve_seconds": time.perf_counter() - started,
            "tail_certificate_enabled": finite.tail_enabled,
            "scope": "Exact stock/journey plus declared last-posted quote memory; no belief rounding/pruning. Original lag-zero cells are reused separately, not rerun. Resource-limited cells are not exact values."}


def microcheck(cell, context):
    """Check 50 named-stream micro networks against the unchanged general kernel."""
    p = cell["parameters"]
    rng = StreamRegistry(p.get("seed", 200000), "validation/history_micro").generator()
    rows = []
    current = {"models_completed": 0, "models_total": p["models"]}
    context.counters = lambda: dict(current)
    for index in range(p["models"]):
        arm = ["direct", "linked", "unrelated", "oracle", "full_state"][index % 5]
        params = {"T": 2, "actions": 2, "n_spokes": 1, "lag_periods": index // 5 % 2,
                  "capacity": int(rng.integers(1, 4)),
                  "capacity_support": [[1, 2], [1, 3], [2, 3]][int(rng.integers(3))],
                  "connecting_share": float(rng.uniform(0.1, 0.9)),
                  "arrival_probability": float(rng.uniform(0.3, 1.0)),
                  "availability": "flag" if rng.random() < 0.5 else "seats_capped",
                  "fare_bin": 0 if rng.random() < 0.5 else 10,
                  "share_weight": 0 if rng.random() < 0.7 else 0.3}
        compiled = compile_market(build_instance("hub_feeder_family", params))
        reference = (OracleDP(compiled, "A", backend="reference", max_nodes=p["max_nodes"])
                     if arm == "full_state" else ExactBeliefDP(compiled, "A", arm,
                                                               backend="reference", max_nodes=p["max_nodes"]))
        optimized = history_dp(compiled, "A", arm, max_nodes=p["max_nodes"])
        current.update(model=index + 1, arm=arm, stage="root_values")
        actual, expected = optimized.solve().value, reference.solve().value
        errors = [abs(actual - expected)]
        roots = reference.roots()
        node = roots[0][1]
        errors.append(float(np.max(np.abs(optimized.q_values(node) - reference.q_values(node)))))
        children = reference.successors(node, index % 2)
        if children:
            child = children[index % len(children)][2]
            errors.append(float(np.max(np.abs(optimized.q_values(child) - reference.q_values(child)))))
        row = {"index": index, "arm": arm, "params": params, "spec_hash": compiled.spec_hash,
               "reference_value": expected, "history_value": actual,
               "maximum_absolute_error": max(errors), "passed": max(errors) <= 1e-10}
        rows.append(row)
        atomic_json(context.directory / "micro_checks.json", rows)
        if not row["passed"]:
            raise AssertionError("history-state microcheck differs from general reference")
        current.update(models_completed=index + 1)
    return {"status": "completed", "models": len(rows),
            "maximum_absolute_error": max(row["maximum_absolute_error"] for row in rows),
            "checks_sha256": file_hash(context.directory / "micro_checks.json"),
            "scope": "Random T2/A2 single-spoke networks, root and reachable Q comparisons, all5 arms, lag0/1, binned/seat channels and nonzero share cases. Not a complete frontier or brute-force policy-enumeration certificate."}
