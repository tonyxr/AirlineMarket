"""A separately versioned physical-kernel optimization and matched resource probe."""

from pathlib import Path
import json

from ..campaign.artifacts import file_hash
from ..core.hashing import canonical_json
from ..sim.sellability_cache import cached_sellability
from .empirical_preflight import measure


def measure_cached(cell, context):
    """Reuse the unchanged native cohort and compare its saved scientific outputs."""
    p = cell["parameters"]
    path = Path(p["baseline_result"])
    if file_hash(path) != p["baseline_result_sha256"]:
        raise ValueError("saved cold baseline result changed")
    baseline = json.loads(path.read_text())
    if (baseline["parameters"]["spec_sha256"] != p["spec_sha256"]
            or baseline["parameters"]["seed"] != p["seed"]):
        raise ValueError("optimized resource probe must match the frozen physical spec and tape seed")
    with cached_sellability() as cache:
        result = measure(cell, context)
        info = cache.info()
    fields = ["spec_hash", "potential_requests", "tape_layout", "contribution_proxy",
              "nonpurchase_requests", "outside_network_bookings"]
    matching = canonical_json({k: result[k] for k in fields}) == canonical_json(
        {k: baseline[k] for k in fields})
    result.update(status="completed" if matching else "failed",
                  matched_scientific_outputs=matching, sellability_cache=info,
                  engine_mode="counted_sparse+cached_binary_sellability",
                  cold_baseline_cohort_seconds=baseline["native_full_cohort_seconds"],
                  baseline_result_sha256=p["baseline_result_sha256"],
                  scope="New cached-availability revision on the same frozen cohort/tape; scientific outputs must match the saved baseline. One cold timing under concurrent workloads is not a final throughput estimate or calibration budget.")
    return result
