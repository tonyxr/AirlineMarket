"""Executable campaign tasks; each task preserves the requested model (A.1)."""

from collections import Counter
import csv
import gzip
import io
import time
import subprocess
import sys

from ..core.hashing import jsonable
from ..spec import compile_market, dump_spec
from .instances import build_instance


def exact_dp(cell: dict, context) -> dict:
    """Measure an exact Bellman solve without treating a limit as a value."""
    from ..core.errors import KernelLimitError
    from ..sim.finite import UnsupportedFiniteModel
    from ..solvers.exact_dp import ExactBeliefDP
    from ..solvers.oracle_dp import OracleDP

    parameters = cell["parameters"]
    spec = build_instance(cell["generator"], parameters)
    dump_spec(spec, context.directory / "spec.json")
    compiled = compile_market(spec)
    focal = parameters.get("focal", spec.carriers[0].code)
    arm = parameters.get("arm", "direct")
    kwargs = {
        "max_nodes": int(cell["limits"].get("max_nodes", 20_000_000)),
        "backend": parameters.get("backend", "compact"),
    }
    try:
        dp = (
            OracleDP(compiled, focal, **kwargs)
            if arm == "full_state"
            else ExactBeliefDP(compiled, focal, arm, **kwargs)
        )
    except UnsupportedFiniteModel as error:
        return {
            "status": "unsupported",
            "reason": str(error),
            "value": None,
            "spec_hash": compiled.spec_hash,
        }

    def counters() -> dict:
        solver = dp.finite or dp
        return {
            "nodes": len(solver.cache) if dp.finite else len(dp.nodes),
            "nodes_per_period": dict(solver.counts),
            "backend": dp.backend,
        }

    context.counters = counters
    started = time.perf_counter()
    try:
        result = dp.solve()
        outcome = {
            "status": "solved",
            "value": result.value,
            "nodes": result.nodes,
            "nodes_per_period": result.nodes_per_period,
            "exact": True,
        }
    except KernelLimitError as error:
        outcome = {
            "status": "node_limit",
            "reason": str(error),
            "value": None,
            "exact": False,
            **counters(),
        }
    seconds = time.perf_counter() - started
    return {
        **outcome,
        "backend": dp.backend,
        "backend_reason": dp.backend_reason,
        "spec_hash": compiled.spec_hash,
        "actions": len(dp.grid),
        "T": spec.horizon.n_periods,
        "solve_seconds": seconds,
        "ms_per_node": 1000 * seconds / max(1, outcome.get("nodes", 0)),
        "distinct_belief_supports": None,
        "raw_history_merge_ratio": None,
        "measurement_note": "Support/complete raw-history counts not instrumented; no inferred values are substituted.",
    }


def simulate(cell: dict, context) -> dict:
    """Run fixed-coefficient development cohorts on named exogenous tapes."""
    from ..agents.fixed import FixedCoefficientController
    from ..sim.engine import MarketEngine
    from ..sim.invariants import check_world

    parameters = cell["parameters"]
    spec = build_instance(cell["generator"], parameters)
    dump_spec(spec, context.directory / "spec.json")
    compiled = compile_market(spec)
    focal = spec.carriers[0].code
    theta = tuple(parameters.get("theta", spec.pricing[0].theta_grid[0]))
    start = int(parameters.get("seed_start", 200000))
    count = int(parameters.get("n_cohorts", 64))
    if any(300000 <= seed < 400000 for seed in range(start, start + count)):
        raise ValueError("simulation task cannot open sealed final seeds")
    rows = []
    started = time.perf_counter()
    for k, seed in enumerate(range(start, start + count)):
        engine = MarketEngine(compiled, {focal: FixedCoefficientController(theta)})
        engine.reset(seed)
        while not engine.terminal():
            engine.step()
            check_world(engine.world, compiled)
        metrics = engine.metrics()
        rows.append(
            {
                "seed": seed,
                **{
                    name: metrics[name]
                    for name in ["profit", "revenue", "objective", "share"]
                    if name in metrics
                },
            }
        )
        context.update({"cohorts_completed": k + 1, "cohorts_total": count})
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    with (context.directory / "episodes.csv.gz").open("wb") as handle:
        with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0, filename="") as archive:
            archive.write(text.getvalue().encode())
    elapsed = time.perf_counter() - started
    return {
        "status": "completed",
        "n_cohorts": count,
        "spec_hash": compiled.spec_hash,
        "mean_profit": sum(r["profit"] for r in rows) / count,
        "seconds_per_cohort": elapsed / count,
    }


def smoke(cell: dict, context) -> dict:
    """Deterministic scheduling fixture for crash/resume acceptance tests."""
    parameters = cell["parameters"]
    if parameters.get("spawn_child"):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        context.child_pids.append(child.pid)
        (context.directory / "child_pid.json").write_text(str(child.pid))
        child.wait()
    time.sleep(float(parameters.get("delay", 0)))
    return {"status": "completed", "answer": int(parameters.get("x", 0)) ** 2}


def execute(cell: dict, context) -> dict:
    """Dispatch a registered experiment; unknown tasks fail explicitly."""
    tasks = {"exact_dp": exact_dp, "simulate": simulate, "smoke": smoke}
    if cell["task"] == "benchmark":
        from ..bench.harness import run_benchmarks

        return run_benchmarks(cell["parameters"], context.directory)
    if cell["task"] == "experiment":
        from ..eval import ExperimentRunner

        phase = "final" if cell["phase"] in (4, "final") else "development"
        runner = ExperimentRunner(cell["parameters"]["config"], phase=phase)
        result = runner.run(context.directory)
        return {
            "status": "completed",
            "summary": result["summary"],
            "contrasts": result["contrasts"],
        }
    return tasks[cell["task"]](cell, context)
