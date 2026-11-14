"""Distinct unit models C2/a7/beta.03, not the five production scales."""

import copy
import importlib.util
import json
import math
import os
import sys
from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
import pytest

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.campaign.worker import CellContext
from airline_market.spec import compile_market, load_spec

if os.environ.get("EXPONENTIAL_STAGE"):
    for package, name in [
        ("sim", "exponential_pricing"),
        ("experiments", "exponential_study"),
    ]:
        q = "airline_market." + package + "." + name
        spec = importlib.util.spec_from_file_location(
            q, Path(os.environ["EXPONENTIAL_STAGE"]) / (name + ".py")
        )
        m = importlib.util.module_from_spec(spec)
        sys.modules[q] = m
        spec.loader.exec_module(m)
from airline_market.experiments import exponential_study as study
from airline_market.sim.exponential_pricing import (
    OpportunityTape,
    PricingObservation,
    draw_tape,
    dynamic_price,
    fixed_price,
    optimal_value,
    polynomial,
    run_profile,
)


def source():
    s = json.loads(Path("config/benchmarks/V4_exponential_k1.json").read_text())
    s.update(scale=3, capacity=2, a=7, beta=0.03, horizon=0.7)
    s["physical_spec"]["name"] = "unit_continuous_C2_a7"
    s["physical_spec"]["legs"][0]["capacity"]["support"] = [2]
    return s


def test_polynomial_and_price_against_high_precision_arithmetic():
    with localcontext() as c:
        c.prec = 80
        mu = Decimal("2.3")
        total = sum(mu**j / Decimal(math.factorial(j)) for j in range(7))
        assert abs(polynomial(6, float(mu))[0] - float(total)) < 1e-12
        remaining = float(mu) * math.e / 7
        price = dynamic_price(PricingObservation(0, remaining, 6), 7, 0.03)
        prev = sum(mu**j / Decimal(math.factorial(j)) for j in range(6))
        expect = (Decimal(1) + (total / prev).ln()) / Decimal(".03")
        assert abs(price - float(expect)) < 1e-12


def test_exact_value_agrees_with_independent_HJB_and_terminal_boundaries():
    s = source()
    r = study.reference(s, 1e-8)
    assert r["HJB_agreement"] and r["rows_processed"] == 0
    assert optimal_value(0, 1, 7, 0.03) == 0 and optimal_value(2, 0, 7, 0.03) == 0
    assert all(row[0] <= row[1] for row in r["formula_values_USD"])


def test_fixed_expected_poisson_revenue_hand_C1():
    a, b, T = 7, 0.03, 0.7
    p = fixed_price(1, T, a, b)
    mu = a * T * math.exp(-b * p)
    assert abs(study.fixed_expectation(1, T, a, b) - p * (1 - math.exp(-mu))) < 1e-12


def test_native_ledger_continuous_prices_and_full_stockout_events():
    s = source()
    c = compile_market(load_spec(s["physical_spec"]))
    rows = np.array(
        [[0.1, 0.1, 0.0], [0.2, 0.1, 0.0], [0.3, 0.1, 0.0], [0.4, 0.1, 0.9]]
    )
    rows.flags.writeable = False
    tape = OpportunityTape(209360, rows)
    world, events = run_profile(c, tape, 1, s["a"], s["beta"], s["horizon"], "fixed")
    assert len(world.carriers[0].bookings) == 2 and len(events) == 4
    assert [b.record_id for b in world.carriers[0].bookings] == [0, 1]
    assert [b.arrival_rank for b in world.carriers[0].bookings] == [0, 1]
    assert all(b.period == 0 for b in world.carriers[0].bookings)
    assert math.isnan(events[2, 1]) and events[2, 3] == 0
    assert world.carriers[0].cash == 2 * fixed_price(2, 0.7, 7, 0.03)


def test_only_current_observation_controls_price_and_tape_readonly():
    obs = PricingObservation(0.1, 0.6, 2)
    with pytest.raises(FrozenInstanceError):
        obs.stock = 8
    tape = draw_tape(209360, 20, 0.7)
    with pytest.raises(ValueError):
        tape.rows[0, 0] = 0.9
    assert dynamic_price(obs, 7, 0.03) == dynamic_price(copy.copy(obs), 7, 0.03)
    assert dynamic_price(PricingObservation(0.1, 0.6, 0), 7, 0.03) is None


def test_scale_thinning_shares_original_mark_and_no_parent_mutation():
    s = source()
    c = compile_market(load_spec(s["physical_spec"]))
    rows = np.array([[0.1, 0.1, 0.01], [0.2, 0.8, 0.01], [0.3, 0.2, 0.99]])
    rows.flags.writeable = False
    tape = OpportunityTape(209360, rows)
    before = rows.copy()
    _, events = run_profile(c, tape, 0.5, 7, 0.03, 0.7, "dynamic")
    assert events[:, 0].tolist() == [0, 2] and np.array_equal(rows, before)


def test_actual_CellContext_driver_and_transitive_outputs(tmp_path):
    models = []
    for j in [1, 2]:
        s = source()
        s["capacity"] *= j
        s["a"] *= j
        s["scale"] = j
        s["physical_spec"]["legs"][0]["capacity"]["support"] = [s["capacity"]]
        path = tmp_path / f"model{j}.json"
        atomic_json(path, s)
        models.append({"path": str(path), "sha256": file_hash(path)})
    reg = tmp_path / "reg.json"
    atomic_json(reg, {"unit_actual_context": True})
    p = {
        "models": models,
        "design_registration": str(reg),
        "design_registration_sha256": file_hash(reg),
        "cohorts": 16,
        "validation_seed_start": 209360,
        "HJB_tolerance_USD": 1e-8,
    }
    context = CellContext(tmp_path)
    r = study.run({"parameters": p}, context)
    assert (
        r["physical_cohorts"] == 16
        and r["native_profiles"] == 64
        and r["all_HJB_agree"]
    )
    assert r["full_V4_pass"] is None and r["scientific_budget"] is None
    assert context.publish()["physical_cohorts_completed"] == 16
    assert all(file_hash(tmp_path / n) == h for n, h in r["artifact_hashes"].items())
    with np.load(tmp_path / "bootstrap.npz") as z:
        assert z["physical_indices"].shape == (2000, 16)


def test_hidden_purchase_mark_cannot_change_prechoice_price():
    s = source()
    c = compile_market(load_spec(s["physical_spec"]))
    rows = np.array([[0.1, 0.1, 0.0], [0.2, 0.1, 0.99]])
    changed = rows.copy()
    changed[0, 2] = 0.99
    rows.flags.writeable = changed.flags.writeable = False
    _, a = run_profile(c, OpportunityTape(209360, rows), 1, 7, 0.03, 0.7, "dynamic")
    _, b = run_profile(c, OpportunityTape(209360, changed), 1, 7, 0.03, 0.7, "dynamic")
    assert a[0, 1] == b[0, 1] and a[0, 3] == 1 and b[0, 3] == 0
    with pytest.raises(ValueError, match="unknown exponential"):
        run_profile(c, OpportunityTape(209360, rows), 1, 7, 0.03, 0.7, "misspelled")
