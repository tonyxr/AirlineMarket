"""Controlled native ledger accounting under declared refundable fee objectives."""

import hashlib
import json
import pickle
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
from airline_market.sim.refundable_net_objective import (
    RefundableFee,
    net_contribution_metrics,
)

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.hashing import jsonable
from airline_market.sim.fixed_fare_admission import FixedFareAdmissionEngine
from airline_market.sim.ledger import resolve
from airline_market.spec import compile_market, load_spec


def fingerprint(world):
    """Hash the complete trusted constructed world in memory; never unpickle data."""
    return hashlib.sha256(pickle.dumps(world, protocol=5)).hexdigest()


def rational_fee(fare, rule):
    """Independent exact transaction-fee expression for saved arithmetic."""
    p = Fraction(str(fare))
    return min(
        p,
        max(
            Fraction(0),
            Fraction(str(rule["rate"])) * p + Fraction(str(rule["flat_USD"])),
        ),
    )


def run(cell, context):
    p = cell["parameters"]
    guards = [(p[x], p[x + "_sha256"]) for x in ["problem", "design_registration"]]
    assert all(file_hash(n) == h for n, h in guards)
    source = json.loads(Path(p["problem"]).read_text())
    c = compile_market(load_spec(source["physical_spec"]))
    capacity = source["initial_capacity"]
    rows = []
    native_branches = []
    base_count = 0
    book_calls = 0
    view_calls = 0
    maxerror = 0.0
    for counts in product(*(range(k + 1) for k in [3] * len(c.spec.products))):
        if any(
            sum(int(c.A[0][i, j]) * counts[j] for j in range(len(counts))) > capacity[i]
            for i in range(len(capacity))
        ):
            continue
        base = FixedFareAdmissionEngine(
            c, [[0.0] * len(counts)] * c.spec.horizon.n_periods
        )
        for j, n in enumerate(counts):
            for _ in range(n):
                base.step(j, True)
                book_calls += 1
        base_count += 1
        before_books = jsonable(base.world.carriers[0].bookings)
        before = {}
        rules = {}
        for plan in source["fee_plans"]:
            rules[plan["name"]] = {
                k: RefundableFee(**v) for k, v in plan["rules"].items()
            }
            stamp = fingerprint(base.world)
            before[plan["name"]] = net_contribution_metrics(
                base.world,
                c,
                0,
                rules[plan["name"]],
                fees_fully_refundable=source["fees_fully_refundable"],
            )
            view_calls += 1
            assert stamp == fingerprint(base.world)
        for target in source["capacity_interventions"]:
            engine = base.clone()
            engine.world.carriers[0].capacity = np.array(target, dtype=np.int64)
            outcome = resolve(engine.world, 0, c)
            state = engine.world.carriers[0]
            clone = engine.world.clone()
            clone.carriers[0].posted_prices = np.minimum(
                np.array([x.fare_cap for x in c.spec.products]),
                np.maximum(
                    np.array([x.fare_floor for x in c.spec.products]),
                    clone.carriers[0].posted_prices + source["quote_shift_USD"],
                ),
            )
            branch_id = len(native_branches)
            native_branches.append(
                {
                    "id": branch_id,
                    "base_counts": list(counts),
                    "period": engine.world.t,
                    "target_capacity": target,
                    "before_bookings": before_books,
                    "active_bookings": jsonable(state.bookings),
                    "canceled_bookings": jsonable(outcome.canceled),
                    "native_resolution": jsonable(outcome),
                    "native_state": jsonable(state),
                    "current_quotes_after_change": clone.carriers[
                        0
                    ].posted_prices.tolist(),
                }
            )
            for plan in source["fee_plans"]:
                stamp = fingerprint(engine.world)
                quote_stamp = fingerprint(clone)
                after = net_contribution_metrics(
                    engine.world,
                    c,
                    0,
                    rules[plan["name"]],
                    fees_fully_refundable=source["fees_fully_refundable"],
                )
                view_calls += 1
                changed = net_contribution_metrics(
                    clone,
                    c,
                    0,
                    rules[plan["name"]],
                    fees_fully_refundable=source["fees_fully_refundable"],
                )
                view_calls += 1
                assert (
                    after == changed
                    and stamp == fingerprint(engine.world)
                    and quote_stamp == fingerprint(clone)
                )
                paid = sum(
                    (
                        rational_fee(
                            b["fare"], plan["rules"][c.spec.products[b["product"]].id]
                        )
                        for b in before_books
                    ),
                    Fraction(0),
                )
                released = sum(
                    (
                        rational_fee(
                            b.fare, plan["rules"][c.spec.products[b.product].id]
                        )
                        for b in outcome.canceled
                    ),
                    Fraction(0),
                )
                active_fee = paid - released
                service = sum(
                    (Fraction(str(c.product_cost[b.product])) for b in state.bookings),
                    Fraction(0),
                )
                netcash = Fraction(str(state.cash)) - active_fee
                net = netcash - service
                expected = {
                    "active_refundable_fee_liability_USD": active_fee,
                    "net_cash_USD": netcash,
                    "native_service_cost_USD": service,
                    "declared_net_contribution_USD": net,
                }
                errors = [abs(after[k] - float(v)) for k, v in expected.items()]
                errors += [
                    abs(
                        before[plan["name"]]["active_refundable_fee_liability_USD"]
                        - after["active_refundable_fee_liability_USD"]
                        - float(released)
                    )
                ]
                maxerror = max(maxerror, max(errors))
                assert max(errors) <= p["tolerance"]
                if plan["name"] == "zero":
                    assert (
                        after["declared_net_objective"]
                        == after["native_default_objective"]
                    )
                rows.append(
                    {
                        "branch_id": branch_id,
                        "fee_plan": plan["name"],
                        "before": before[plan["name"]],
                        "after": after,
                        "fee_paid_fraction_USD": str(paid),
                        "fee_released_fraction_USD": str(released),
                        "active_fee_fraction_USD": str(active_fee),
                        "net_cash_fraction_USD": str(netcash),
                        "net_contribution_fraction_USD": str(net),
                        "max_error_USD": max(errors),
                        "native_world_before_query_sha256": stamp,
                        "native_world_after_query_sha256": fingerprint(engine.world),
                        "quote_world_before_query_sha256": quote_stamp,
                        "quote_world_after_query_sha256": fingerprint(clone),
                        "quote_invariant": True,
                    }
                )
    assert all(file_hash(n) == h for n, h in guards)
    atomic_json(
        context.directory / "native_branches.json",
        {"branches": native_branches, "physical_spec_sha256": p["problem_sha256"]},
    )
    atomic_json(context.directory / "net_objective_rows.json", rows)
    return {
        "status": "completed",
        "base_booking_count_states": base_count,
        "native_booking_calls": book_calls,
        "native_resolution_branches": len(native_branches),
        "net_objective_rows": len(rows),
        "net_objective_view_calls": view_calls,
        "max_independent_transaction_error_USD": maxerror,
        "actual_canceled_bookings_across_branches": sum(
            len(b["canceled_bookings"]) for b in native_branches
        ),
        "negative_net_rows": sum(
            x["after"]["declared_net_contribution_USD"] < 0 for x in rows
        ),
        "empirical_net_profit_identified": False,
        "default_engine_objective_changed": False,
        "new_sampled_physical_cohorts": 0,
        "RNG_calls": 0,
        "parameter_fits": 0,
        "full_P3_acceptance": None,
        "scientific_budget": None,
        "artifact_hashes": {
            n: file_hash(context.directory / n)
            for n in ["native_branches.json", "net_objective_rows.json"]
        },
        "scope": "Explicit opt-in fully refundable-fee objective on constructed native ledgers; not empirical fee/cost identification, default optimizer wiring, original refund-rate support or MR10 replay.",
    }
