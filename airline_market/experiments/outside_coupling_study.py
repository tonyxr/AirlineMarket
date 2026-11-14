"""One matched Q2 resource-coupling probe; no new passenger population or CEM fit."""

from collections import Counter
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import time

from ..campaign.artifacts import atomic_json, file_hash
from ..core.hashing import canonical_json
from ..core.rng import StreamRegistry
from ..pipeline.background_load import generate_load_events
from ..pipeline.outside_footprints import build_catalogue, generate_purchase_tape
from ..sim.coupled_outside import CoupledOutsideEngine
from ..sim.shared_background import augment_load_products, PREFIX
from ..sim.market_local_tape import generate_market_local
from ..sim.sparse_access import sparse_access
from ..sim.sellability_cache import cached_sellability
from ..sim import ledger
from ..spec import compile_market, load_spec, dump_spec


def measure(cell, context):
    """Attach core footprints to existing outside choices and audit passenger flow."""
    import pandas as pd

    p = cell["parameters"]
    sources = ["source_spec", "source_patterns", "Q2_scoped_markets",
               "Q2_capacity_targets", "baseline_result"]
    for key in sources:
        if file_hash(p[key]) != p[key + "_sha256"]:
            raise ValueError("frozen outside-coupling input changed: " + key)
    baseline = json.loads(Path(p["baseline_result"]).read_text())
    original = load_spec(p["source_spec"])
    context.counters = lambda: {"stage": "Q2_outside_footprint_catalogue"}
    frame = pd.read_parquet(p["Q2_scoped_markets"])
    capacity = pd.read_csv(p["Q2_capacity_targets"])
    routes = set(capacity[["marketing_carrier", "origin", "dest"]].itertuples(index=False, name=None))
    catalogue = build_catalogue(frame, original, routes, p["core_carriers"])
    catalogue["input_rows"] = len(frame)
    atomic_json(context.directory / "catalogue.json", catalogue)
    del frame, capacity
    augmented = augment_load_products(original, p["core_carriers"])
    compiled = compile_market(augmented)
    if compiled.spec_hash != baseline["spec_hash"] or p["seed"] != baseline["seed"]:
        raise ValueError("reused outside-coupling baseline has a different physical spec or seed")
    patterns = json.loads(Path(p["source_patterns"]).read_text())
    events, mapping = generate_load_events(compiled, patterns, p["seed"], quarter_days=p["quarter_days"])
    event_hash = hashlib.sha256(canonical_json([asdict(e) for e in events]).encode()).hexdigest()
    if event_hash != baseline["mapping"]["event_tape_sha256"] or baseline["fraction"] != 1.0:
        raise ValueError("reused baseline must have the same full external-load tape")
    context.counters = lambda: {"stage": "physical_and_purchase_tapes"}
    tape = generate_market_local(compiled, p["seed"],
        StreamRegistry(p["seed"], "calibration/empirical_resource/full_tape"))
    potential = sum(map(len, tape.arrival_marks))
    if potential != baseline["potential_strategic_requests"]:
        raise ValueError("paired potential passenger count changed")
    purchase_tape, purchase_info = generate_purchase_tape(compiled, tape, catalogue, p["seed"])
    purchase_info["tape_sha256"] = hashlib.sha256(canonical_json([
        {"period": t, "potential_row": row, **asdict(mark)}
        for (t, row), mark in sorted(purchase_tape.items())]).encode()).hexdigest()
    atomic_json(context.directory / "purchase_tape_summary.json", purchase_info)
    dump_spec(augmented, context.directory / "spec.json")
    with sparse_access(), cached_sellability() as cache:
        engine = CoupledOutsideEngine(compiled, events, footprint_tape=purchase_tape,
                                      outside_carrier=p["outside_carrier"])
        start = time.perf_counter()
        engine.reset(p["seed"], tape=tape)
        context.counters = lambda: {"stage": "purchase_coupled_cohort", "period": engine.world.t,
            "periods": compiled.spec.horizon.n_periods,
            "requests_completed": len(engine.world.request_log), **dict(engine.purchase_counts)}
        while not engine.terminal():
            engine.step()
        seconds = time.perf_counter() - start
        metrics = engine.metrics()
        resources = []
        attached = {key for group in engine.purchase_links for key in engine.groups[group]}
        for code in p["core_carriers"]:
            i = compiled.carrier_index[code]
            foreground, external, purchased = Counter(), Counter(), Counter()
            state = engine.world.carriers[i]
            for b in state.bookings:
                product = compiled.spec.products[b.product]
                target = (foreground if not product.id.startswith(PREFIX)
                          else purchased if (i, b.record_id) in attached else external)
                target.update(product.legs)
            stock = ledger.inventory(state, compiled, i)
            for local, global_leg in enumerate(compiled.legs_of[i]):
                leg = compiled.spec.legs[global_leg].id
                if state.capacity[local] - foreground[leg] - external[leg] - purchased[leg] != stock[local]:
                    raise ValueError("purchase-coupled core resource identity failed")
                resources.append({"leg": leg, "carrier": code, "capacity": int(state.capacity[local]),
                    "strategic_usage": foreground[leg], "external_usage": external[leg],
                    "outside_purchase_usage": purchased[leg], "inventory": int(stock[local])})
        active_tickets = sum(not compiled.spec.products[b.product].id.startswith(PREFIX)
            for s in engine.world.carriers for b in s.bookings)
        cancellations = sum(int(row["cancellations"]) for row in metrics["carriers"].values())
        purchases = sum(row["chosen"] is not None for row in engine.world.request_log)
        if purchases != active_tickets + cancellations:
            raise ValueError("one-choice/one-ticket passenger accounting failed")
        availability = cache.info()
    atomic_json(context.directory / "resource_usage.json", resources)
    report = {"status": "completed", "spec_hash": compiled.spec_hash,
        "seed": p["seed"], "potential_requests": potential, "realized_requests": len(engine.world.request_log),
        "active_logical_tickets": active_tickets, "canceled_logical_tickets": cancellations,
        "chosen_ticket_requests": purchases, "one_choice_one_ticket": True,
        "core_resource_accounting_exact": True, "catalogue_sha256": file_hash(context.directory / "catalogue.json"),
        "conditional_core_coupon_edges": catalogue["core_coupon_edges_by_carrier"],
        "purchase_tape": purchase_info, "external_event_tape_sha256": event_hash,
        "outside_resource_coupling": metrics["outside_resource_coupling"],
        "shared_core_load": metrics["shared_core_load"],
        "core_carriers": {code: metrics["carriers"][code] for code in p["core_carriers"]},
        "nonpurchase_requests": sum(row["chosen"] is None for row in engine.world.request_log),
        "outside_ticket_bookings": len(engine.world.carriers[compiled.carrier_index[p["outside_carrier"]]].bookings),
        "cohort_seconds": seconds, "availability_cache": availability,
        "baseline_reused": {"path": p["baseline_result"], "sha256": p["baseline_result_sha256"]},
        "input_hashes": {p[key]: file_hash(p[key]) for key in sources},
        "calibrated": False, "new_passenger_population": False, "Q3_targets_used": False,
        "scope": "Single matched-seed physical resource-footprint sensitivity. Q2 conditional purchase shares initialize a latent subtype drawn before choice; its capacity can censor the outside alternative. Dated routes remain assumed, not legal full-itinerary reconstruction. Core receipts for outside paths are not attributed; economic outcomes are strategic gross contribution, not full marketing-carrier shares or total/net airline profit. No CEM, predictive validation or exact/PF/MPC/learning certificate."}
    atomic_json(context.directory / "cohort_summary.json", report)
    return report
