"""Bounded resource-load sensitivity on the frozen Q2 behavioral initialization."""

from collections import Counter
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import time

import numpy as np

from ..campaign.artifacts import atomic_json, file_hash
from ..core.hashing import canonical_json
from ..core.rng import StreamRegistry
from ..pipeline.background_load import generate_load_events
from ..sim.market_local_tape import generate_market_local
from ..sim.shared_background import SharedLoadEngine, augment_load_products, PREFIX
from ..sim.sparse_access import sparse_access
from ..sim.sellability_cache import cached_sellability
from ..sim.ledger import inventory
from ..spec import compile_market, dump_spec, load_spec


def measure(cell, context):
    """Measure one declared load fraction; no fitting, residual imputation or tax claim."""
    p = cell["parameters"]
    for key in ("source_spec", "source_patterns"):
        if file_hash(p[key]) != p[key + "_sha256"]:
            raise ValueError("frozen empirical input changed: " + key)
    context.counters = lambda: {"stage": "shared_load_setup"}
    original = load_spec(p["source_spec"])
    augmented = augment_load_products(original, p["core_carriers"])
    compiled = compile_market(augmented)
    patterns = json.loads(Path(p["source_patterns"]).read_text())
    events, mapping = generate_load_events(compiled, patterns, p["seed"],
        quarter_days=p["quarter_days"], period_weights=p.get("period_weights"))
    mapping["event_tape_sha256"] = hashlib.sha256(
        canonical_json([asdict(e) for e in events]).encode()).hexdigest()
    atomic_json(context.directory / "mapping.json", mapping)
    dump_spec(augmented, context.directory / "spec.json")
    started = time.perf_counter()
    tape = generate_market_local(compiled, p["seed"],
        StreamRegistry(p["seed"], "calibration/empirical_resource/full_tape"))
    tape_seconds = time.perf_counter() - started
    with sparse_access(), cached_sellability() as cache:
        engine = SharedLoadEngine(compiled, events, fraction=p["fraction"])
        started = time.perf_counter()
        engine.reset(p["seed"], tape=tape)
        context.counters = lambda: {"stage": "shared_load_cohort", "period": engine.world.t,
            "periods": compiled.spec.horizon.n_periods,
            "strategic_requests_completed": len(engine.world.request_log),
            **dict(engine.load_counts)}
        while not engine.terminal():
            engine.step()
        cohort_seconds = time.perf_counter() - started
        result = engine.metrics()
        availability = cache.info()
        resources = []
        for code in p["core_carriers"]:
            i = compiled.carrier_index[code]
            state = engine.world.carriers[i]
            background = Counter()
            strategic = Counter()
            for booking in state.bookings:
                product = compiled.spec.products[booking.product]
                target = background if product.id.startswith(PREFIX) else strategic
                target.update(product.legs)
            inv = inventory(state, compiled, i)
            for k, global_leg in enumerate(compiled.legs_of[i]):
                leg = compiled.spec.legs[global_leg]
                if state.capacity[k] - strategic[leg.id] - background[leg.id] != inv[k]:
                    raise ValueError("shared resource accounting mismatch")
                resources.append({"leg": leg.id, "carrier": code,
                    "capacity": int(state.capacity[k]), "strategic_usage": strategic[leg.id],
                    "external_usage": background[leg.id], "inventory": int(inv[k])})
    atomic_json(context.directory / "resource_usage.json", resources)
    counts = engine.load_counts
    if counts["attempted_groups"] != counts["accepted_groups"] + counts["capacity_rejected_groups"]:
        raise ValueError("load group flow does not conserve attempts")
    core = {code: result["carriers"][code] for code in p["core_carriers"]}
    report = {"status": "completed", "spec_hash": compiled.spec_hash,
        "fraction": p["fraction"], "seed": p["seed"], "mapping": mapping,
        "tape_seconds": tape_seconds, "cohort_seconds": cohort_seconds,
        "potential_strategic_requests": sum(map(len, tape.arrival_marks)),
        "core_carriers": core, "shared_core_load": result["shared_core_load"],
        "availability_cache": availability,
        "nonpurchase_requests": sum(r["chosen"] is None for r in engine.world.request_log),
        "outside_network_bookings": len(engine.world.carriers[compiled.carrier_index[p["outside_carrier"]]].bookings),
        "resource_accounting_exact": True, "calibrated": False,
        "remaining_residual_imputed": False, "Q3_targets_used": False,
        "input_hashes": {p[key]: file_hash(p[key]) for key in ("source_spec", "source_patterns")},
        "scope": "One uncalibrated matched-seed resource-footprint sensitivity. Uniform assumed booking periods and within-route flight assignment do not reconstruct legal dated itineraries. External groups consume core capacity atomically but receive no attributed revenue/cost or competitive share. Strategic gross contribution is conditional on this load; not total airline/net profit, a calibrated background process, a CEM evaluation or V1/V5 acceptance."}
    atomic_json(context.directory / "cohort_summary.json", report)
    return report
