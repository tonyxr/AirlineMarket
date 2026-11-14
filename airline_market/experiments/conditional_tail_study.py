"""Isolated F1 revision using exact certified conditional-tail vectors."""

import time
from pathlib import Path

from ..campaign.artifacts import file_hash
from ..campaign.instances import build_instance
from ..core.errors import KernelLimitError
from ..solvers.conditional_tail import ConditionalTailSolver
from ..solvers.exact_dp import ExactBeliefDP
from ..solvers.oracle_dp import OracleDP
from ..spec import compile_market, dump_spec, load_spec
from ..sim.finite import UnsupportedFiniteModel


def solve(cell, context):
    """Solve the unchanged physical model and label the new memo-count definition."""
    p = cell["parameters"]
    if "source_spec" in p:
        path = Path(p["source_spec"])
        if file_hash(path) != p["source_spec_sha256"]:
            raise ValueError("frozen physical specification changed")
        spec = load_spec(path)
    else:
        spec = build_instance(cell["generator"], p)
    dump_spec(spec, context.directory / "spec.json")
    compiled = compile_market(spec)
    focal = p.get("focal", spec.carriers[0].code)
    arm = p["arm"]
    budget = int(cell["limits"]["max_nodes"])
    try:
        dp = (
            OracleDP(compiled, focal, backend="compact", max_nodes=budget)
            if arm == "full_state"
            else ExactBeliefDP(
                compiled, focal, arm, backend="compact", max_nodes=budget
            )
        )
    except UnsupportedFiniteModel as error:
        return {
            "status": "unsupported",
            "value": None,
            "reason": str(error),
            "spec_hash": compiled.spec_hash,
        }
    old = dp.finite
    solver = ConditionalTailSolver(old.model, old.grid, budget)
    dp.finite = solver

    def counters():
        """Expose actual work without comparing vector entries to raw belief nodes."""
        info = solver.tail_vectors.cache_info()
        return {
            "nodes": solver.memo_entries(),
            "prefix_and_final_nodes": len(solver.cache),
            "nodes_per_period": dict(solver.counts),
            "tail_vector_entries": info.currsize,
            "tail_vector_evaluations": solver.tail_vector_evaluations,
            "tail_vector_cache_hits": info.hits,
            "penultimate_belief_queries": solver.tail_belief_queries,
            "backend": "compact_conditional_tail",
        }

    context.counters = counters
    started = time.perf_counter()
    try:
        result = dp.solve()
        outcome = {"status": "solved", "value": result.value, "exact": True}
    except KernelLimitError as error:
        outcome = {
            "status": "node_limit",
            "value": None,
            "exact": False,
            "reason": str(error),
        }
    return {
        **outcome,
        **counters(),
        "spec_hash": compiled.spec_hash,
        "T": spec.horizon.n_periods,
        "actions": len(dp.grid),
        "solve_seconds": time.perf_counter() - started,
        "tail_certificate_enabled": solver.tail_enabled,
        "memo_definition": "Retained prefix/final Bellman entries plus conditional physical-state tail vectors. Per-period counts cover prefix/final entries only; vector work/query counts are separate. These counts differ from the original belief-node frontier.",
        "information": "The current posterior weights all conditional vectors; no hidden evaluation state or future draw is observed. Noncertified models use the unchanged exact recursion.",
    }
