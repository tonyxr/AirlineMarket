"""Distinct C2/130-and60/binomial3 unit fixtures, never the production100k grid."""

import importlib.util
import json
import os
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.spec import compile_market, load_spec

if os.environ.get("LITTLEWOOD_STAGE"):
    for package, name in [
        ("sim", "batch_admission"),
        ("experiments", "littlewood_study"),
    ]:
        qualified = "airline_market." + package + "." + name
        spec = importlib.util.spec_from_file_location(
            qualified, Path(os.environ["LITTLEWOOD_STAGE"]) / (name + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)

from airline_market.sim.batch_admission import (
    BatchAdmissionEngine,
    DemandTape,
    draw_tape,
    protection_action,
    run_tape,
)

from airline_market.experiments import littlewood_study as study


def source():
    s = json.loads(Path("config/benchmarks/V4_Littlewood_C8_binomial.json").read_text())
    s["physical_spec"]["name"] = "unit_C2_fares60_130"
    s["physical_spec"]["legs"][0]["capacity"]["support"] = [2]
    for p, f in zip(s["physical_spec"]["products"], [60, 130]):
        p["reference_fare"] = f
        s["physical_spec"]["pricing"][0]["base"][p["id"]] = f
    s["demand_counts"] = [{"n": 3, "p": "0.4"}, {"n": 3, "p": "0.75"}]
    return s


def engine():
    return BatchAdmissionEngine(compile_market(load_spec(source()["physical_spec"])))


def test_native_rejection_booking_period_rank_and_observation():
    e = engine()
    obs = e.observe(0)
    with pytest.raises(FrozenInstanceError):
        obs.period = 8
    e.take_request(False)
    assert e.world.carriers[0].cash == 0 and e.observe(0).own_inventory == (2,)
    b = e.take_request(True)
    assert (
        b.record_id,
        b.product,
        b.period,
        b.arrival_rank,
        b.segment,
        b.taste_class,
    ) == (0, 0, 0, 1, 0, 0)
    e.next_stage()
    b = e.take_request(True)
    assert (b.product, b.period, b.arrival_rank, b.fare) == (1, 1, 0, 130)
    assert e.world.carriers[0].cash == 190
    e.next_stage()
    with pytest.raises(ValueError, match="terminal"):
        e.take_request(False)


def test_exact_littlewood_discrete_tail_and_rational_optimum():
    d = study.exact_design(source())
    assert d["Littlewood_protection"] == 2 and d["exact_optimal_set"] == [2]
    from fractions import Fraction

    assert sum(map(Fraction, d["high_pmf"])) == 1
    assert d["rows_processed"] == 0


def test_all_small_demand_pairs_and_grid_use_native_ledger():
    e = engine()
    for l in range(4):
        for h in range(4):
            tape = DemandTape(200128, l, h)
            for y in range(3):
                w = run_tape(e, tape, y)
                a = min(l, 2 - y)
                b = min(h, 2 - a)
                assert w.carriers[0].cash == 60 * a + 130 * b
                assert [x.product for x in w.carriers[0].bookings] == [0] * a + [1] * b
                assert len(w.request_log) == l + h and w.t == 2


def test_policy_only_sees_current_frozen_observation():
    e = engine()
    assert protection_action(e.observe(0), 2) is False
    assert protection_action(e.observe(0), 1) is True
    e.next_stage()
    assert protection_action(e.observe(1), 2) is True


def test_immutable_tape_and_class_streams():
    a = draw_tape(200128, 3, 0.4, 3, 0.75)
    with pytest.raises(FrozenInstanceError):
        a.high = 7
    b = draw_tape(200128, 3, 0.4, 9, 0.75)
    assert a.low == b.low and a == draw_tape(200128, 3, 0.4, 3, 0.75)


def test_overbooking_raises_without_partial_ledger():
    e = engine()
    e.take_request(True)
    e.take_request(True)
    before = (
        e.world.next_record_id,
        e.world.carriers[0].cash,
        len(e.world.request_log),
    )
    with pytest.raises(ValueError, match="available leg inventory"):
        e.take_request(True)
    assert (
        e.world.next_record_id,
        e.world.carriers[0].cash,
        len(e.world.request_log),
    ) == before


def test_histogram_bootstrap_preserves_common_cohort_controls():
    counts = np.array([2, 3, 4])
    payoff = np.array([[10, 20], [30, 60], [50, 100]])
    means, draws = study.paired_grid_statistics(counts, payoff)
    assert means[1] == 2 * means[0] and draws.shape == (2000, 2)
    assert np.array_equal(draws[:, 1], 2 * draws[:, 0])


def test_driver_transitive_evidence_on_distinct_small_fixture(tmp_path):
    problem = tmp_path / "problem.json"
    reg = tmp_path / "reg.json"
    atomic_json(problem, source())
    atomic_json(reg, {"unit_C2": True})
    p = {
        "problem": str(problem),
        "problem_sha256": file_hash(problem),
        "design_registration": str(reg),
        "design_registration_sha256": file_hash(reg),
        "cohorts": 16,
        "validation_seed_start": 200128,
    }
    counter = SimpleNamespace(update=lambda **kwargs: None)
    result = study.run(
        {"parameters": p}, SimpleNamespace(directory=tmp_path, counter=counter)
    )
    assert result["n_physical_cohorts"] == 16 and result["n_native_profiles"] == 48
    assert result["full_V4_pass"] is None
    assert all(
        file_hash(tmp_path / n) == h for n, h in result["artifact_hashes"].items()
    )
    with np.load(tmp_path / "native_cohorts.npz") as z:
        assert z["native_bookings"].shape == (16, 3, 2, 4)
        assert len(z["seeds"]) == len(set(z["seeds"])) == 16


def test_clone_keeps_clock_and_mutates_only_child():
    e = engine()
    e.take_request(False)
    child = e.clone()
    child.take_request(True)
    assert child.world.carriers[0].bookings[0].arrival_rank == 1
    assert e.rank == 1 and e.world.carriers[0].bookings == []
