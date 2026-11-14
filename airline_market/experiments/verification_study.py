"""Isolated full-history policy enumeration and instrumented empirical profiling."""

import cProfile
import pstats
from pathlib import Path
import json

from ..campaign.artifacts import atomic_json, file_hash
from ..solvers.brute_history import binary_sale_micro, BruteHistoryEnumerator
from ..solvers.exact_dp import ExactBeliefDP
from ..spec import compile_market, dump_spec
from .empirical_preflight import measure


def enumerate_micro(cell, context):
    """Compare separately optimized DPs with all128 explicit T3 policies."""
    p = cell["parameters"]
    compiled = compile_market(binary_sale_micro(p))
    dump_spec(compiled.spec, context.directory / "spec.json")
    enum = BruteHistoryEnumerator(compiled)
    context.counters = lambda: {"stage": "explicit_policy_enumeration",
                                "physical_transition_cache": enum.transitions.cache_info().currsize}
    enumerated = enum.enumerate()
    atomic_json(context.directory / "all_policy_values.json", enumerated)
    reference = ExactBeliefDP(compiled, "A", "direct", backend="reference").solve().value
    compact = ExactBeliefDP(compiled, "A", "direct", backend="compact").solve().value
    error = max(abs(reference - enumerated["maximum"]), abs(compact - enumerated["maximum"]))
    report = {"status": "completed" if error <= 1e-10 else "failed",
              "reference_value": reference, "compact_value": compact,
              "enumeration_maximum": enumerated["maximum"], "maximum_absolute_error": error,
              "policies_enumerated": len(enumerated["values"]), "spec_hash": compiled.spec_hash,
              "policy_values_sha256": file_hash(context.directory / "all_policy_values.json"),
              "scope": "T3 binary-sale micro family with hidden rival inventory; not every possible T3 market or all V7 criteria."}
    return report


def profile_empirical(cell, context):
    """Profile the unchanged prepared-cohort path; never publish profiled cost as timing."""
    atomic_json(context.directory / "PROFILING_ONLY.json", {
        "all_times_include_profiler_overhead": True,
        "benchmark_timing_eligible": False,
        "scope": "A separately declared profiling operation; completed cold timing probe is preserved."})
    profiler = cProfile.Profile()
    try:
        result = profiler.runcall(measure, cell, context)
    finally:
        profiler.dump_stats(str(context.directory / "cohort_profile.pstats"))
    stats = pstats.Stats(profiler)
    rows = [{"file": key[0], "line": key[1], "function": key[2],
             "primitive_calls": v[0], "calls": v[1], "exclusive_seconds": v[2],
             "cumulative_seconds": v[3]} for key, v in stats.stats.items()]
    rows.sort(key=lambda r: r["exclusive_seconds"], reverse=True)
    atomic_json(context.directory / "profile_top.json", rows[:60])
    result["profiled_full_cohort_seconds"] = result.pop("native_full_cohort_seconds")
    result["profiled_tape_seconds"] = result.pop("tape_seconds")
    result.update(benchmark_timing_eligible=False, profiler_total_seconds=stats.total_tt,
                  profile_sha256=file_hash(context.directory / "cohort_profile.pstats"),
                  scope="Instrumented unchanged empirical cohort; all durations include profiler overhead and cannot replace the cold resource measurement.")
    return result
