"""Refund accounting witnesses and driver verification; no production cohorts."""

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from airline_market.campaign.artifacts import file_hash
from airline_market.spec import compile_market, load_spec

if os.environ.get("REFUND_STAGE"):
    spec = importlib.util.spec_from_file_location(
        "airline_market.experiments.refund_accounting_study",
        Path(os.environ["REFUND_STAGE"]) / "refund_accounting_study.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
from airline_market.experiments import refund_accounting_study as study


def fixture_compiled():
    d = json.loads(
        Path("config/benchmarks/V3_MR10_full_refund_penalty0.json").read_text()
    )
    d["horizon"]["n_periods"] = 3
    d["legs"][0]["capacity"]["support"] = [2]
    d["products"][0].update(reference_fare=101, fare_floor=81, fare_cap=121)
    d["pricing"][0]["base"]["LOCAL"] = 101
    return compile_market(load_spec(d))


def booking(record, product, fare):
    return {
        "record_id": record,
        "carrier": 0,
        "product": product,
        "fare": fare,
        "period": 0,
        "arrival_rank": record,
        "segment": 0,
        "taste_class": 0,
    }


def empty():
    return {
        "t": 0,
        "capacity": [2, 2],
        "bookings": [],
        "cash": 0.0,
        "revenue": 0.0,
        "refunds": 0.0,
        "penalties": 0.0,
        "canceled": 0,
    }


def test_independent_whole_itinerary_refund_and_penalty():
    c = fixture_compiled()
    before = empty()
    p = [booking(0, 0, 101), booking(1, 1, 150)]
    after = {
        "t": 1,
        "capacity": [2, 0],
        "bookings": [p[0]],
        "cash": 84.0,
        "revenue": 251.0,
        "refunds": 150.0,
        "penalties": 17.0,
        "canceled": 1,
    }
    r = study.independent_resolution(before, p, after, c, 17)
    assert r["canceled_bookings"] == [p[1]] and r["manual_inventory"] == [1, 0]
    assert r["original_fare_refund_USD"] == 150 and r["max_accounting_error_USD"] == 0


def test_fixed_priority_preserves_local_even_if_connecting_record_earlier():
    c = fixture_compiled()
    before = empty()
    p = [booking(0, 1, 150), booking(1, 0, 101)]
    after = {
        "t": 1,
        "capacity": [1, 2],
        "bookings": [p[1]],
        "cash": 101.0,
        "revenue": 251.0,
        "refunds": 150.0,
        "penalties": 0.0,
        "canceled": 1,
    }
    r = study.independent_resolution(before, p, after, c, 0)
    assert r["kept_ids"] == [1] and r["canceled_bookings"] == [p[0]]


def test_full_refund_lifecycle_zero_and_immediate_cash_decrease():
    c = fixture_compiled()
    b = booking(0, 1, 150)
    before = empty()
    before.update(t=1, bookings=[b], cash=150.0, revenue=150.0)
    after = {
        "t": 2,
        "capacity": [2, 0],
        "bookings": [],
        "cash": 0.0,
        "revenue": 150.0,
        "refunds": 150.0,
        "penalties": 0.0,
        "canceled": 1,
    }
    r = study.independent_resolution(before, [], after, c, 0)
    assert r["max_accounting_error_USD"] == 0 and after["cash"] - before["cash"] == -150


def test_bad_refund_arithmetic_is_retained_as_error():
    c = fixture_compiled()
    before = empty()
    p = [booking(0, 1, 150)]
    after = {
        "t": 1,
        "capacity": [2, 0],
        "bookings": [],
        "cash": 1.0,
        "revenue": 150.0,
        "refunds": 149.0,
        "penalties": 0.0,
        "canceled": 1,
    }
    r = study.independent_resolution(before, p, after, c, 0)
    assert r["max_accounting_error_USD"] == 1


def test_wrong_canceled_ticket_is_rejected():
    c = fixture_compiled()
    before = empty()
    p = [booking(0, 0, 101), booking(1, 1, 150)]
    after = {
        "t": 1,
        "capacity": [2, 0],
        "bookings": [p[1]],
        "cash": 150.0,
        "revenue": 251.0,
        "refunds": 101.0,
        "penalties": 0.0,
        "canceled": 1,
    }
    with pytest.raises(AssertionError):
        study.independent_resolution(before, p, after, c, 0)


def unit_parameters(tmp_path):
    cfg = json.loads(
        Path("config/campaigns/V3_MR10_full_refund_accounting.json").read_text()
    )["fixed"]
    variants = []
    for penalty in [0, 17]:
        d = study.jsonable(fixture_compiled().spec)
        d["resolution"]["penalty_per_cancellation"] = penalty
        path = tmp_path / f"model{penalty}.json"
        path.write_text(json.dumps(d))
        variants.append(
            {
                "path": str(path),
                "sha256": file_hash(path),
                "compiled_spec_hash": compile_market(load_spec(path)).spec_hash,
                "penalty": penalty,
            }
        )
    reg = tmp_path / "design.json"
    reg.write_text("{}")
    return {
        **cfg,
        "variants": variants,
        "n_cohorts": 8,
        "seed_start": 200960,
        "prices": [101, 150],
        "design_registration": str(reg),
        "design_registration_sha256": file_hash(reg),
    }


def test_actual_native_driver_shares_tape_and_saves_each_period(tmp_path):
    p = unit_parameters(tmp_path)
    r = study.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
    assert (
        r["status"] == "completed"
        and r["physical_cohorts"] == 8
        and r["native_profile_cohorts"] == 16
        and r["native_period_steps"] == 48
    )
    assert (
        r["zero_refund_rate_arm"] == "UNSUPPORTED_UNRUN" and r["full_MR10_pass"] is None
    )
    rows = [
        json.loads(x) for x in (tmp_path / "cohort_rows.jsonl").read_text().splitlines()
    ]
    assert [x["physical_seed"] for x in rows] == list(range(200960, 200968)) and all(
        len(x["periods"]) == 3 for x in rows
    )
    assert r["totals"][0]["cancellations"] > 0 and all(
        t["max_accounting_error_USD"] == 0 for t in r["totals"]
    )
    assert all(
        file_hash(tmp_path / name) == h for name, h in r["artifact_hashes"].items()
    )


def test_hash_failure_precedes_native_engine_and_tape(tmp_path, monkeypatch):
    p = unit_parameters(tmp_path)
    p["design_registration_sha256"] = "0" * 64

    def forbidden(*a, **k):
        raise AssertionError("new native call")

    monkeypatch.setattr(study, "MarketEngine", forbidden)
    with pytest.raises(ValueError, match="frozen refund"):
        study.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
    assert not (tmp_path / "cohort_rows.jsonl").exists()


def test_no_final_test_seed_opening(tmp_path):
    p = unit_parameters(tmp_path)
    p["seed_start"] = 300000
    with pytest.raises(ValueError, match="validation cohort"):
        study.run({"parameters": p}, SimpleNamespace(directory=tmp_path))
