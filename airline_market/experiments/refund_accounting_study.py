"""MR10 supported full-refund ledger witnesses (Eq.15); zero-rate arm stays open."""

from pathlib import Path

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.hashing import canonical_json, jsonable, schema_hash
from airline_market.core.types import PriceAction
from airline_market.sim.accounting import carrier_metrics
from airline_market.sim.engine import MarketEngine
from airline_market.sim.tape import ExogenousTape
from airline_market.spec import compile_market, load_spec


def ledger_view(engine):
    """Analyst-only full ledger; never provided to a controller or action rule."""
    s = engine.world.carriers[0]
    return {
        "t": engine.world.t,
        "capacity": s.capacity.tolist(),
        "bookings": jsonable(s.bookings),
        "cash": s.cash,
        "revenue": s.revenue,
        "refunds": s.refunds,
        "penalties": s.penalties,
        "canceled": s.canceled,
    }


def independent_resolution(before, purchases, after, compiled, penalty):
    """Source-path fixed-priority accounting, independent of native resolve/counts/A."""
    products = compiled.spec.products
    legs = compiled.spec.legs
    incidence = [[int(leg.id in product.legs) for leg in legs] for product in products]
    all_bookings = [*before["bookings"], *purchases]
    if len({b["record_id"] for b in all_bookings}) != len(all_bookings):
        raise AssertionError("duplicate live/new booking ID")
    remaining = list(after["capacity"])
    kept, canceled = [], []
    for b in sorted(
        all_bookings, key=lambda b: (products[b["product"]].priority, b["record_id"])
    ):
        use = incidence[b["product"]]
        if all(x >= u for x, u in zip(remaining, use)):
            kept.append(b)
            remaining = [x - u for x, u in zip(remaining, use)]
        else:
            canceled.append(b)
    assert {b["record_id"] for b in kept} == {b["record_id"] for b in after["bookings"]}
    assert all(b in all_bookings for b in after["bookings"])
    assert min(remaining) >= 0
    assert all(b["period"] < after["t"] for b in all_bookings)
    revenue = sum(b["fare"] for b in purchases)
    refund = sum(b["fare"] for b in canceled)
    penalties = penalty * len(canceled)
    errors = [
        abs(after["revenue"] - before["revenue"] - revenue),
        abs(after["refunds"] - before["refunds"] - refund),
        abs(after["penalties"] - before["penalties"] - penalties),
        abs(after["cash"] - before["cash"] - revenue + refund + penalties),
        abs(after["cash"] - after["revenue"] + after["refunds"] + after["penalties"]),
    ]
    assert after["canceled"] - before["canceled"] == len(canceled)
    return {
        "canceled_bookings": canceled,
        "kept_ids": [b["record_id"] for b in kept],
        "manual_inventory": remaining,
        "new_fares_USD": revenue,
        "original_fare_refund_USD": refund,
        "penalty_USD": penalties,
        "max_accounting_error_USD": max(errors),
    }


def run(cell, context):
    p = cell["parameters"]
    if not 200000 <= p["seed_start"] < p["seed_start"] + p["n_cohorts"] <= 300000:
        raise ValueError("full validation cohort range required")
    guards = [
        (p["design_registration"], p["design_registration_sha256"]),
        ("airline_market/sim/ledger.py", p["native_ledger_sha256"]),
    ]
    guards += [(v["path"], v["sha256"]) for v in p["variants"]]
    if any(file_hash(Path(path)) != h for path, h in guards):
        raise ValueError("frozen refund input changed")
    compiled = [compile_market(load_spec(v["path"])) for v in p["variants"]]
    for c, v in zip(compiled, p["variants"]):
        if (
            c.spec_hash != v["compiled_spec_hash"]
            or c.spec.resolution.refund != "full_original_fare"
        ):
            raise ValueError("native supported full-refund contract required")
        if len(c.spec.carriers) != 1 or c.spec.quotes.requests or c.spec.background:
            raise ValueError(
                "declared single-carrier noquote/background contract required"
            )
        assert c.spec.resolution.penalty_per_cancellation == v["penalty"]
    structural = []
    for c in compiled:
        d = jsonable(c.spec)
        d["resolution"]["penalty_per_cancellation"] = 0
        structural.append(d)
    if structural[0] != structural[1] or [v["penalty"] for v in p["variants"]] != [
        0,
        17,
    ]:
        raise ValueError("paired models differ beyond declared penalty")
    engines = [MarketEngine(c) for c in compiled]
    current = {
        "stage": "native_full_refund_accounting",
        "cohorts_completed": 0,
        "cohorts_total": p["n_cohorts"],
        "native_period_steps": 0,
    }
    context.counters = lambda: dict(current)
    totals = [
        {
            "penalty": v["penalty"],
            "purchases": 0,
            "cancellations": 0,
            "local_cancellations": 0,
            "connecting_cancellations": 0,
            "zero_cancel_cohorts": 0,
            "revenue_USD": 0.0,
            "refunds_USD": 0.0,
            "penalties_USD": 0.0,
            "profit_USD": 0.0,
            "max_accounting_error_USD": 0.0,
            "event_cancellations": [0, 0],
        }
        for v in p["variants"]
    ]
    with (context.directory / "cohort_rows.jsonl").open("x") as out:
        for k in range(p["n_cohorts"]):
            seed = p["seed_start"] + k
            tape = ExogenousTape.generate(compiled[0], seed)
            tape_sha = schema_hash(tape)
            for e in engines:
                e.reset(seed, tape=tape)
                assert e.tape is tape
            periods = []
            while not engines[0].terminal():
                paired = []
                for e, c, v, total in zip(engines, compiled, p["variants"], totals):
                    before = ledger_view(e)
                    assert e.observe("A", "direct").privileged is None
                    result = e.step({"A": PriceAction(dict(enumerate(p["prices"])))})
                    after = ledger_view(e)
                    purchases = jsonable(result.purchases)
                    witness = independent_resolution(
                        before, purchases, after, c, v["penalty"]
                    )
                    error = max(
                        witness["max_accounting_error_USD"],
                        abs(result.cash_changes[0] - after["cash"] + before["cash"]),
                    )
                    assert error <= p["tolerance"]
                    total["max_accounting_error_USD"] = max(
                        total["max_accounting_error_USD"], error
                    )
                    total["purchases"] += len(purchases)
                    total["cancellations"] += len(witness["canceled_bookings"])
                    for b in witness["canceled_bookings"]:
                        key = (
                            "connecting_cancellations"
                            if len(c.spec.products[b["product"]].legs) > 1
                            else "local_cancellations"
                        )
                        total[key] += 1
                    if result.resolutions:
                        assert len(result.resolutions) == 1
                        event = result.resolutions[0]["event"]
                        total["event_cancellations"][event] += len(
                            witness["canceled_bookings"]
                        )
                    paired.append(
                        {
                            "penalty": v["penalty"],
                            "before": before,
                            "purchases": purchases,
                            "after": after,
                            "step": jsonable(result),
                            "witness": witness,
                        }
                    )
                    current["native_period_steps"] += 1
                for key in ["capacity", "bookings", "canceled"]:
                    assert paired[0]["after"][key] == paired[1]["after"][key]
                assert paired[0]["purchases"] == paired[1]["purchases"]
                assert (
                    paired[0]["step"]["resolutions"] == paired[1]["step"]["resolutions"]
                )
                periods.append(paired)
            terminal = []
            for e, c, total in zip(engines, compiled, totals):
                m = carrier_metrics(e.world, c, 0)
                assert not e.world.carriers[0].bookings and m["service_cost"] == 0
                error = abs(m["profit"] + total["penalty"] * m["canceled"])
                assert error <= p["tolerance"]
                if m["canceled"] == 0:
                    total["zero_cancel_cohorts"] += 1
                for key in ["revenue", "refunds", "penalties", "profit"]:
                    total[key + "_USD"] += m[key]
                terminal.append(m)
            assert schema_hash(tape) == tape_sha
            out.write(
                canonical_json(
                    {
                        "physical_seed": seed,
                        "tape_sha256": tape_sha,
                        "periods": periods,
                        "terminal": terminal,
                    }
                )
                + "\n"
            )
            current["cohorts_completed"] = k + 1
    coverage = all(
        t["local_cancellations"] > 0
        and t["connecting_cancellations"] > 0
        and min(t["event_cancellations"]) > 0
        for t in totals
    )
    summary = {
        "totals": totals,
        "nonzero_cancellation_coverage": coverage,
        "supported_full_refund_accounting_pass": coverage,
        "zero_refund_rate_arm": "UNSUPPORTED_UNRUN",
        "full_MR10_pass": None,
        "full_V3_pass": None,
        "scientific_budget": None,
    }
    atomic_json(context.directory / "refund_summary.json", summary)
    assert all(file_hash(Path(path)) == h for path, h in guards)
    return {
        "status": "completed",
        **summary,
        "physical_cohorts": p["n_cohorts"],
        "native_profile_cohorts": 2 * p["n_cohorts"],
        "native_period_steps": current["native_period_steps"],
        "actor_forwards": 0,
        "optimizer_updates": 0,
        "artifact_hashes": {
            name: file_hash(context.directory / name)
            for name in ["cohort_rows.jsonl", "refund_summary.json"]
        },
        "scope": "Supported native full-original-fare refunds only; no implemented zero-refund rate. Full-refund lifecycle sale+refund0, immediate refund reducescash; penaltyseparate. No forced requests/choices/retries. Analyst ledger evidence neveractorinput.",
    }
