"""Tiny dated/quote fixtures; no full saved snapshot or real schedules in tests."""

import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.campaign.worker import CellContext

if os.environ.get("QUOTE_ROUTE_STAGE"):
    for package, name in [
        ("pipeline", "quote_route_binding"),
        ("experiments", "quote_route_study"),
    ]:
        q = "airline_market." + package + "." + name
        spec = importlib.util.spec_from_file_location(
            q, Path(os.environ["QUOTE_ROUTE_STAGE"]) / (name + ".py")
        )
        m = importlib.util.module_from_spec(spec)
        sys.modules[q] = m
        spec.loader.exec_module(m)
from airline_market.experiments import quote_route_study as study
from airline_market.pipeline.quote_route_binding import BindingIndex, bind_reported

ZONES = {"BOS": "America/New_York", "ORD": "America/Chicago", "DEN": "America/Denver"}
T = 1655294400


def leg(i, o, d, t, u, operator="OO"):
    return {
        "id": i,
        "marketing_carrier": "UA",
        "operating_carrier": operator,
        "origin": o,
        "dest": d,
        "departure_utc_seconds": t,
        "arrival_utc_seconds": u,
        "cohort_date": "2022-06-15",
    }


def values(wait=60):
    return [
        "2022-06-15",
        "BOS",
        "DEN",
        "UA",
        f"{T}||{T + 7200 + wait * 60}",
        f"{T + 7200}||{T + 14400 + wait * 60}",
        "BOS||ORD",
        "ORD||DEN",
        "UA||OO",
    ]


def index(extra=False):
    ls = [
        leg("L1", "BOS", "ORD", T, T + 7200),
        leg("L2", "ORD", "DEN", T + 10800, T + 18000),
    ]
    products = [{"id": "P1", "leg_ids": ["L1", "L2"]}]
    if extra:
        ls.append(leg("L3", "ORD", "DEN", T + 10800, T + 18000))
        products.append({"id": "P2", "leg_ids": ["L1", "L3"]})
    return BindingIndex(ls, products)


def test_exact_operator_or_marketer_compatibility_and_all_ambiguous_ids():
    r = bind_reported(values(), index(), ZONES, [40, 300], "ORD")
    assert r["status"] == "unique_exact_dated_product" and r["dated_product_ids"] == [
        "P1"
    ]
    r = bind_reported(values(), index(True), ZONES, [40, 300], "ORD")
    assert r["status"] == "ambiguous_exact_dated_products" and r[
        "dated_product_ids"
    ] == ["P1", "P2"]
    assert r["leg_candidates"] == [["L1"], ["L2", "L3"]]


def test_near_times_never_become_exact_matches():
    v = values()
    v[5] = f"{T + 7201}||{T + 18000}"
    r = bind_reported(v, index(), ZONES, [40, 300], "ORD")
    assert r["status"] == "missing_exact_leg" and r["departure_candidate_counts"] == [
        1,
        1,
    ]
    assert r["route_day_candidate_counts"] == [1, 1] and r["leg_candidates"][0] == []


def test_fractional_source_clock_is_not_truncated_to_match():
    ls = [
        leg("L1", "BOS", "ORD", T, T + 7200.25),
        leg("L2", "ORD", "DEN", T + 10800, T + 18000),
    ]
    r = bind_reported(
        values(),
        BindingIndex(ls, [{"id": "P1", "leg_ids": ["L1", "L2"]}]),
        ZONES,
        [40, 300],
        "ORD",
    )
    assert r["status"] == "missing_exact_leg"


def test_window_rule_does_not_claim_legality_or_change_thresholds():
    r = bind_reported(values(30), index(), ZONES, [40, 300], "ORD")
    assert r["structural_valid"] and not r["connection_window_ok"]
    assert (
        r["status"] == "outside_declared_connection_window" and r["wait_minutes"] == 30
    )


def test_invalid_structure_retained_with_reason():
    v = values()
    v[6] = "BOS||DEN"
    r = bind_reported(v, index(), ZONES, [40, 300], "ORD")
    assert not r["structural_valid"] and r["reason"] == "disconnected_airports"
    v = values()
    v[0] = "2022-06-16"
    assert (
        bind_reported(v, index(), ZONES, [40, 300], "ORD")["reason"]
        == "first_local_departure_date_mismatch"
    )


def test_all_legs_exist_but_missing_product_stays_unbound():
    ix = index()
    ix.products.clear()
    assert (
        bind_reported(values(), ix, ZONES, [40, 300], "ORD")["status"]
        == "no_existing_dated_product"
    )


def test_actual_worker_driver_keeps_all_keys_and_source_bytes(tmp_path):
    ix = index()
    legs = tmp_path / "legs.parquet"
    products = tmp_path / "products.parquet"
    pq.write_table(pa.Table.from_pylist(list(ix.legs.values())), legs)
    pq.write_table(
        pa.Table.from_pylist([{"id": "P1", "leg_ids": ["L1", "L2"]}]), products
    )
    dbpath = tmp_path / "source.sqlite"
    db = sqlite3.connect(dbpath)
    db.execute(
        "CREATE TABLE products(id INTEGER PRIMARY KEY,payload TEXT,quarter INTEGER,departure TEXT,carrier TEXT,kind TEXT,market TEXT)"
    )
    v = values()
    for i in [1, 2, 3]:
        key = v + ["coach", f"FARE{i}", "false", "false", "false"]
        if i == 3:
            key[5] = f"{T + 7201}||{T + 18000}"
        db.execute(
            "INSERT INTO products VALUES(?,?,?,?,?,?,?)",
            (i, json.dumps(key), 2, key[0], key[3], "connecting", "BOS-DEN"),
        )
    db.commit()
    db.close()
    config = tmp_path / "net.json"
    reg = tmp_path / "reg.json"
    atomic_json(
        config, {"airport_timezones": ZONES, "connection_window_minutes": [40, 300]}
    )
    atomic_json(reg, {"tiny_fixture": True})
    p = {
        n: str(x)
        for n, x in [
            ("quote_db", dbpath),
            ("legs", legs),
            ("products", products),
            ("network_config", config),
            ("design_registration", reg),
        ]
    }
    p.update({n + "_sha256": file_hash(x) for n, x in list(p.items())})
    p["batch_size"] = 1
    out = tmp_path / "out"
    out.mkdir()
    context = CellContext(out)
    r = study.run({"parameters": p}, context)
    assert r["reported_keys_checked"] == 3 and r["unique_physical_paths"] == 2
    links = pq.read_table(out / "reported_key_links.parquet").to_pylist()
    assert links == [
        {"reported_key_id": 1, "physical_path_id": 1, "quarter": 2},
        {"reported_key_id": 2, "physical_path_id": 1, "quarter": 2},
        {"reported_key_id": 3, "physical_path_id": 2, "quarter": 2},
    ]
    assert (
        r["new_full_quote_scans"]
        == r["new_parameter_fits"]
        == r["new_simulation_cohorts"]
        == 0
    )
    assert (
        r["V1_V5_V6_acceptance"] is None and file_hash(dbpath) == p["quote_db_sha256"]
    )
    assert context.publish()["reported_keys_completed"] == 3
    assert all(file_hash(out / n) == h for n, h in r["artifact_hashes"].items())
