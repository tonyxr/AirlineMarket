"""Bounded, isolated preparation and resource measurements, without CEM fitting."""

from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
import json
import time

import numpy as np

from ..campaign.artifacts import atomic_json, file_hash
from ..pipeline.behavioral_network import initialize_cohort, read_q2_inputs
from ..sim.revenue_proxy import fit_fee_moments, net_proxy_metrics
from ..sim.engine import MarketEngine
from ..sim.tape import ExogenousTape
from ..sim.market_local_tape import generate_market_local
from ..sim.sparse_access import sparse_access
from ..core.rng import StreamRegistry
from ..spec import compile_market, dump_spec, load_spec


def prepare(cell, context):
    """Build a Q2-initialized behavioral spec and fit the observed fare-gap proxy."""
    import pyarrow.parquet as pq

    p = cell["parameters"]
    config = json.loads(Path(p["initializer_config"]).read_text())
    if file_hash(p["initializer_config"]) != p["initializer_sha256"]:
        raise ValueError("initializer configuration changed")
    context.counters = lambda: {"stage": "preparing_behavioral_spec"}
    spec = initialize_cohort(*read_q2_inputs(config, p["date"]), config, p["date"])
    compiled = compile_market(spec)
    dump_spec(spec, context.directory / "spec.json")
    moments, excluded, rows = defaultdict(lambda: np.zeros(6)), 0, 0
    source = Path(config["Q2_quotes"])
    context.counters = lambda: {"stage": "Q2_fee_proxy", "rows_scanned": rows,
                                "invalid_core_rows": excluded}
    columns = ["quarter", "split", "model_path_eligible", "path_network", "isNonStop",
               "totalFare", "baseFare"]
    for batch in pq.ParquetFile(source).iter_batches(batch_size=100000, columns=columns):
        frame = batch.to_pandas()
        if not ((frame.quarter == 2) & (frame.split == "calibration")).all():
            raise ValueError("proxy estimation input contains non-Q2 data")
        rows += len(frame)
        core = frame[frame.model_path_eligible].copy()
        x, base = core.totalFare.to_numpy(), core.baseFare.to_numpy()
        valid = np.isfinite(x) & np.isfinite(base) & (x > 0) & (base >= 0) & (base <= x)
        excluded += int((~valid).sum())
        core = core.loc[valid].copy()
        core["n_legs"] = np.where(core.isNonStop.isin([True, "true", "True"]), 1, 2)
        for (carrier, legs), group in core.groupby(["path_network", "n_legs"]):
            x = group.totalFare.to_numpy()
            y = x - group.baseFare.to_numpy()
            moments[f"{carrier}|{legs}"] += [len(x), x.sum(), x @ x, y.sum(), x @ y, y @ y]
    fits = {key: fit_fee_moments(*values) for key, values in moments.items()}
    atomic_json(context.directory / "fee_proxy.json", fits)
    physical_legs = len(spec.legs) - len(spec.markets)
    physical_products = sum(p.carrier != "OUTSIDE" for p in spec.products)
    report = {"status": "completed", "spec_hash": compiled.spec_hash,
        "spec_sha256": file_hash(context.directory / "spec.json"),
        "fee_proxy_sha256": file_hash(context.directory / "fee_proxy.json"),
        "physical_legs": physical_legs, "physical_products": physical_products,
        "synthetic_outside_resources": len(spec.markets), "total_products": len(spec.products),
        "markets": len(spec.markets), "Q2_quote_rows_scanned": rows,
        "invalid_base_total_core_rows_excluded": excluded,
        "calibrated": False, "CEM_fit_run": False, "Q3_targets_used": False,
        "fee_proxy_is_verified_tax": False,
        "input_hashes": {str(path): file_hash(path) for path in [
            Path(p["initializer_config"]), source,
            Path(config["Q2_targets"]) / "q2_passenger_targets.csv",
            Path(config["Q2_targets"]) / "q2_quote_targets_by_market.csv",
            Path(config["dated_network"]) / "legs.parquet",
            Path(config["dated_network"]) / "products.parquet"]}}
    atomic_json(context.directory / "initialization.json", report)
    return report


def measure(cell, context):
    """Measure actual full dense tape and a bounded native-engine cohort separately."""
    p = cell["parameters"]
    source = Path(p["prepared_attempt"])
    if file_hash(source / "spec.json") != p["spec_sha256"]:
        raise ValueError("prepared physical/behavioral spec changed")
    compiled = compile_market(load_spec(source / "spec.json"))
    context.counters = lambda: {"stage": "full_dense_tape_generation"}
    started = time.perf_counter()
    generator = generate_market_local if p.get("tape_layout") == "market_local_v1" else ExogenousTape.generate
    tape = generator(compiled, p["seed"],
        StreamRegistry(p["seed"], "calibration/empirical_resource/full_tape"))
    tape_seconds = time.perf_counter() - started
    requests = sum(map(len, tape.arrival_marks))
    capacity = compiled.spec.metadata["outside_proxy"]["capacity"]
    if requests >= capacity:
        raise ValueError("technical outside proxy could exhaust on this tape")
    result = {"status": "completed", "spec_hash": compiled.spec_hash,
              "tape_seconds": tape_seconds, "potential_requests": requests,
              "dense_gumbel_shape": list(tape.gumbel.shape),
              "tape_layout": p.get("tape_layout", "dense"),
              "stored_gumbel_bytes": tape.gumbel.nbytes,
              "array_bytes": sum(v.nbytes for v in vars(tape).values()
                                 if isinstance(v, np.ndarray))
                  + (tape.gumbel.nbytes if p.get("tape_layout") == "market_local_v1" else 0),
              "tape_read_only": all(not v.flags.writeable for v in vars(tape).values()
                                    if isinstance(v, np.ndarray)),
              "native_full_cohort_seconds": None, "calibrated": False,
              "scope": "One actual uncalibrated Q2-initialized resource observation under concurrent workloads; not a512-cohort CEM evaluation or final throughput estimate."}
    atomic_json(context.directory / "tape_measurement.json", result)
    if not p.get("run_cohort", False):
        return result
    access = sparse_access() if p.get("engine_mode") == "counted_sparse" else nullcontext()
    with access:
        engine = MarketEngine(compiled)
        started = time.perf_counter()
        context.counters = lambda: {"stage": "native_engine_reset"}
        engine.reset(p["seed"], tape=tape)
        context.counters = lambda: {"stage": "engine_cohort", "period": engine.world.t,
                                    "periods": compiled.spec.horizon.n_periods,
                                    "requests_completed": len(engine.world.request_log)}
        while not engine.terminal():
            engine.step()
        result["native_full_cohort_seconds"] = time.perf_counter() - started
        fits = json.loads((source / "fee_proxy.json").read_text())
        result["contribution_proxy"] = net_proxy_metrics(engine.world, compiled, fits)
    result["engine_mode"] = p.get("engine_mode", "reference")
    chosen = engine.world.request_log
    result["nonpurchase_requests"] = sum(r["chosen"] is None for r in chosen)
    outside = compiled.carrier_index["OUTSIDE"]
    result["outside_network_bookings"] = len(engine.world.carriers[outside].bookings)
    result["native_optimization_basis"] = "gross all-in fares; proxy not used by native planners"
    return result
