"""Distinct native-ledger objective fixtures, no registered production bank."""

import importlib.util
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.campaign.worker import CellContext
from airline_market.sim.fixed_fare_admission import FixedFareAdmissionEngine
from airline_market.sim.ledger import resolve
from airline_market.spec import compile_market, load_spec

if os.environ.get("NET_OBJECTIVE_STAGE"):
    for package, name in [
        ("sim", "refundable_net_objective"),
        ("experiments", "net_objective_study"),
    ]:
        q = "airline_market." + package + "." + name
        spec = importlib.util.spec_from_file_location(
            q, Path(os.environ["NET_OBJECTIVE_STAGE"]) / (name + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[q] = module
        spec.loader.exec_module(module)
from airline_market.sim.refundable_net_objective import (
    RefundableFee,
    net_contribution_metrics,
)

from airline_market.experiments import net_objective_study as study


def fixture():
    s = json.loads(
        Path("config/benchmarks/V4_fixed_fare_two_leg_network.json").read_text()
    )["physical_spec"]
    s["horizon"]["n_periods"] = 6
    for leg in s["legs"]:
        leg["capacity"]["support"] = [2]
    for prod, fare in zip(s["products"], [71, 67, 121]):
        prod["reference_fare"] = fare
        s["pricing"][0]["base"][prod["id"]] = fare
    s["resolution"]["penalty_per_cancellation"] = 9
    c = compile_market(load_spec(s))
    return c, FixedFareAdmissionEngine(c, [[0, 0, 0]] * 6)


def rules(c, rate=0.1, flat=3):
    return {p.id: RefundableFee(rate, flat) for p in c.spec.products}


def test_original_booking_fare_basis_full_refund_and_no_double_service_cost():
    c, e = fixture()
    e.step(2, True)
    e.step(0, True)
    r = rules(c)
    before = net_contribution_metrics(e.world, c, 0, r, fees_fully_refundable=True)
    old = study.fingerprint(e.world)
    e.world.carriers[0].posted_prices += 5
    assert (
        net_contribution_metrics(e.world, c, 0, r, fees_fully_refundable=True) == before
    )
    assert study.fingerprint(e.world) != old
    e.world.carriers[0].capacity = np.array([1, 1])
    out = resolve(e.world, 0, c)
    after = net_contribution_metrics(e.world, c, 0, r, fees_fully_refundable=True)
    assert [b.product for b in out.canceled] == [2] and after[
        "native_penalties_USD"
    ] == 9
    assert after["active_refundable_fee_liability_USD"] == pytest.approx(10.1)
    assert after["declared_net_contribution_USD"] == pytest.approx(71 - 9 - 2 - 10.1)
    assert before["active_refundable_fee_liability_USD"] - after[
        "active_refundable_fee_liability_USD"
    ] == pytest.approx(15.1)


def test_zero_fee_parity_clipping_negative_outcomes_and_complete_query_purity():
    c, e = fixture()
    e.step(2, True)
    stamp = study.fingerprint(e.world)
    zero = net_contribution_metrics(
        e.world, c, 0, rules(c, 0, 0), fees_fully_refundable=True
    )
    assert zero["declared_net_objective"] == zero["native_default_objective"]
    clipped = net_contribution_metrics(
        e.world, c, 0, rules(c, 0, 500), fees_fully_refundable=True
    )
    assert (
        clipped["active_refundable_fee_liability_USD"] == 121
        and clipped["net_cash_USD"] == 0
        and clipped["declared_net_contribution_USD"] < 0
    )
    assert study.fingerprint(e.world) == stamp


def test_objective_keeps_declared_scale_and_share_term():
    c, e = fixture()
    e.step(0, True)
    modified = replace(
        c,
        spec=replace(
            c.spec,
            objectives=(
                replace(c.spec.objectives[0], profit_scale=10, share_weight=0.7),
            ),
        ),
    )
    x = net_contribution_metrics(
        e.world, modified, 0, rules(c), fees_fully_refundable=True
    )
    assert x["native_share"] == 1 and x["declared_net_objective"] == pytest.approx(
        x["declared_net_contribution_USD"] / 10 + 0.7
    )


@pytest.mark.parametrize("r,f", [(-0.1, 0), (1.1, 0), (0.1, -1), (float("nan"), 0)])
def test_invalid_fee_contract_rejected(r, f):
    with pytest.raises(ValueError):
        RefundableFee(r, f)


def test_unknown_contract_or_missing_rules_do_not_infer_zero_fees():
    c, e = fixture()
    with pytest.raises(ValueError):
        net_contribution_metrics(e.world, c, 0, rules(c), fees_fully_refundable=False)
    with pytest.raises(ValueError):
        net_contribution_metrics(e.world, c, 0, {}, fees_fully_refundable=True)


def test_actual_worker_context_with_distinct_small_capacity_and_one_fee(tmp_path):
    c, _ = fixture()
    physical = json.loads(
        Path("config/benchmarks/V4_fixed_fare_two_leg_network.json").read_text()
    )["physical_spec"]
    physical["horizon"]["n_periods"] = 8
    for leg in physical["legs"]:
        leg["capacity"]["support"] = [1]
    problem = tmp_path / "problem.json"
    reg = tmp_path / "reg.json"
    atomic_json(
        problem,
        {
            "physical_spec": physical,
            "initial_capacity": [1, 1],
            "capacity_interventions": [[1, 1], [0, 0]],
            "fee_plans": [
                {
                    "name": "zero",
                    "rules": {
                        p.id: {"rate": 0, "flat_USD": 0} for p in c.spec.products
                    },
                }
            ],
            "quote_shift_USD": 5,
            "fees_fully_refundable": True,
        },
    )
    atomic_json(reg, {"unit_fixture": True})
    out = tmp_path / "out"
    out.mkdir()
    r = study.run(
        {
            "parameters": {
                "problem": str(problem),
                "problem_sha256": file_hash(problem),
                "design_registration": str(reg),
                "design_registration_sha256": file_hash(reg),
                "tolerance": 1e-10,
            }
        },
        CellContext(out),
    )
    assert (
        r["base_booking_count_states"] == 5
        and r["native_resolution_branches"] == 10
        and r["net_objective_rows"] == 10
    )
    assert (
        r["net_objective_view_calls"] == 25
        and r["new_sampled_physical_cohorts"] == r["RNG_calls"] == 0
    )
    assert all(file_hash(out / n) == h for n, h in r["artifact_hashes"].items())
