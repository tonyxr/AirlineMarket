"""Exercise the1000-cohort counter boundary with actual CellContext, C2fixture."""

import importlib.util
import json
import os
import sys
from pathlib import Path

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.campaign.worker import CellContext

if os.environ.get("LITTLEWOOD_V2_STAGE"):
    qualified = "airline_market.experiments.littlewood_worker_v2"
    spec = importlib.util.spec_from_file_location(
        qualified, Path(os.environ["LITTLEWOOD_V2_STAGE"]) / "littlewood_worker_v2.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
from airline_market.experiments.littlewood_worker_v2 import run


def test_actual_worker_context_crosses_counter_boundary(tmp_path):
    s = json.loads(Path("config/benchmarks/V4_Littlewood_C8_binomial.json").read_text())
    s["physical_spec"]["name"] = "unit_v2_C2_65_135"
    s["physical_spec"]["legs"][0]["capacity"]["support"] = [2]
    for p, f in zip(s["physical_spec"]["products"], [65, 135]):
        p["reference_fare"] = f
        s["physical_spec"]["pricing"][0]["base"][p["id"]] = f
    s["demand_counts"] = [{"n": 3, "p": ".4"}, {"n": 3, "p": ".75"}]
    problem = tmp_path / "problem.json"
    reg = tmp_path / "reg.json"
    atomic_json(problem, s)
    atomic_json(reg, {"unit_v2_actual_context": True})
    p = {
        "problem": str(problem),
        "problem_sha256": file_hash(problem),
        "design_registration": str(reg),
        "design_registration_sha256": file_hash(reg),
        "cohorts": 1000,
        "validation_seed_start": 201128,
    }
    context = CellContext(tmp_path)
    result = run({"parameters": p}, context)
    assert result["n_physical_cohorts"] == 1000 and result["n_native_profiles"] == 3000
    assert context.counters()["cohorts_completed"] == 1000
    assert context.counters()["cohorts_total"] == 1000
    assert context.counters()["protection_profiles"] == 3000
    context.publish()
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert (
        progress["cohorts_completed"] == 1000
        and progress["native_bookings"] == result["native_bookings"]
    )
