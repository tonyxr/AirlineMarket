"""A.15/V10: pre-event selection, dated identity, folds and exact cancellation counts."""

import csv
import hashlib
import importlib.util
import json
import os
import sys
import zipfile

import pyarrow.parquet as pq
import pytest

if os.environ.get("EVENT_NETWORK_SOURCE"):
    s = importlib.util.spec_from_file_location(
        "staged_event_network", os.environ["EVENT_NETWORK_SOURCE"]
    )
    m = importlib.util.module_from_spec(s)
    sys.modules[s.name] = m
    s.loader.exec_module(m)
else:
    from airline_market.pipeline import event_network as m


def scope():
    return {
        "periods": [(2026, x) for x in range(1, 5)],
        "airport_pairs": [["AAA", "BBB"]],
        "directed_markets": [["AAA", "BBB"], ["BBB", "AAA"]],
        "route_counts": [
            {
                "airports": ["AAA", "BBB"],
                "marketing_carrier": c,
                "operating_carrier": o,
                "completed_proxy": n,
            }
            for c, o, n in [
                ("NK", "NK", 10),
                ("AA", "YX", 20),
                ("AA", "AA", 15),
                ("UA", "G7", 30),
                ("DL", "DL", 30),
            ]
        ],
    }


def row(**kw):
    r = {
        "FL_DATE": "01/01/2026 12:00:00 AM",
        "MKT_UNIQUE_CARRIER": "AA",
        "OP_UNIQUE_CARRIER": "YX",
        "OP_CARRIER_AIRLINE_ID": "123",
        "OP_CARRIER_FL_NUM": "45",
        "MKT_CARRIER_FL_NUM": "456",
        "ORIGIN": "AAA",
        "DEST": "BBB",
        "CRS_DEP_TIME": "1000",
        "CRS_ARR_TIME": "1200",
        "CRS_ELAPSED_TIME": "120",
        "CANCELLED": "0",
        "DIVERTED": "0",
        "DUP": "N",
        "DISTANCE": "500",
    }
    r.update(kw)
    return r


def source():
    return {"path": "fixture", "sha256": "a" * 64, "year": 2026, "month": 1}


ZONES = {"AAA": "America/New_York", "BBB": "America/New_York"}


def test_incumbents_use_all_pre_event_completed_marketing_counts_and_keep_ties():
    r = m.select_incumbents(scope())[0]
    assert r["strategic_carriers"] == ["NK", "AA", "DL"] and r["tied_boundary"]
    assert r["incumbent_ranking"][0]["completed_service"] == 35
    with pytest.raises(ValueError):
        m.select_incumbents({**scope(), "periods": [(2026, 5)]})


def test_local_clock_rollover_and_exact_elapsed_arrival():
    r = m.scheduled_times("2026-01-01", "2400", "0200", 120, "AAA", "BBB", ZONES)
    assert r["timing_status"] == "ok" and r["departure_local"].startswith(
        "2026-01-02T00:00"
    )
    assert r["arrival_utc"] - r["departure_utc"] == 7200
    assert (
        m.scheduled_times("2026-01-01", "1000", "1201", 120, "AAA", "BBB", ZONES)[
            "timing_status"
        ]
        == "arrival_clock_disagreement"
    )


def test_dst_nonexistent_and_ambiguous_are_not_silently_shifted():
    assert (
        m.scheduled_times("2026-03-08", "0230", "0430", 120, "AAA", "BBB", ZONES)[
            "timing_status"
        ]
        == "nonexistent_local_departure"
    )
    assert (
        m.scheduled_times("2026-11-01", "0130", "0330", 120, "AAA", "BBB", ZONES)[
            "timing_status"
        ]
        == "ok"
    )  # arrival clock selects fold1
    assert (
        m.scheduled_times(
            "2026-11-01",
            "0130",
            "0230",
            120,
            "AAA",
            "BBB",
            {"AAA": "America/New_York", "BBB": "America/Phoenix"},
        )["timing_status"]
        == "arrival_clock_disagreement"
    )
    # Both folds can agree when destination also rolls clocks before arrival.
    r = m.scheduled_times("2026-11-01", "0030", "0130", 60, "AAA", "BBB", ZONES)
    assert r["timing_status"] == "ok"


@pytest.mark.parametrize(
    "key,value",
    [
        ("CRS_DEP_TIME", "1061"),
        ("CRS_ARR_TIME", "2500"),
        ("CRS_ELAPSED_TIME", "-1"),
        ("CRS_ELAPSED_TIME", "NaN"),
        ("OP_CARRIER_FL_NUM", None),
    ],
)
def test_bad_clock_or_identity_retains_raw_row_without_fake_resource(key, value):
    raw = row(**{key: value})
    r = m.normalized_row(raw, source(), 3, m.select_incumbents(scope())[0], ZONES)
    assert (
        r[key] == value
        and r["physical_id"] is None
        and r["source_row_id"] == "a" * 64 + ":3"
    )


def test_duplicate_marketing_rows_share_physical_resource_and_preserve_raw_codes():
    first = row()
    second = row(MKT_UNIQUE_CARRIER=" UA ", MKT_CARRIER_FL_NUM="987")
    a, b = [
        m.normalized_row(r, source(), i, m.select_incumbents(scope())[0], ZONES)
        for i, r in enumerate([first, second])
    ]
    assert (
        a["physical_id"] == b["physical_id"]
        and a["source_row_id"] != b["source_row_id"]
    )
    assert (
        b["MKT_UNIQUE_CARRIER"] == " UA "
        and b["marketing_carrier"] == "UA"
        and not b["strategic"]
    )
    assert a["strategic"] and a["OP_UNIQUE_CARRIER"] == "YX"


def test_missing_timezone_and_unknown_cancellation_are_retained():
    r = m.normalized_row(
        row(CANCELLED="", DIVERTED="bad"),
        source(),
        0,
        m.select_incumbents(scope())[0],
        {},
    )
    assert (
        r["timing_status"] == "missing_timezone"
        and r["cancelled_flag"] is None
        and r["diverted_flag"] is None
    )
    assert r["physical_id"] is None


def archives(tmp, wrong=False):
    result = []
    for month in range(1, 9):
        rows = [
            row(FL_DATE=f"{month:02d}/01/2026 12:00:00 AM"),
            row(
                FL_DATE=f"{month:02d}/01/2026 12:00:00 AM",
                MKT_UNIQUE_CARRIER="UA",
                CANCELLED="1",
            ),
            row(
                FL_DATE=f"{month:02d}/01/2026 12:00:00 AM",
                OP_CARRIER_FL_NUM="46",
                DIVERTED="1",
            ),
            row(FL_DATE=f"{month:02d}/01/2026 12:00:00 AM", ORIGIN="OUT", DEST="OTHER"),
        ]
        if wrong and month == 8:
            rows[0]["FL_DATE"] = "09/01/2026 12:00:00 AM"
        p = tmp / f"{month}.zip"
        csvpath = tmp / f"{month}.csv"
        with csvpath.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=m.FIELDS)
            w.writeheader()
            w.writerows(rows)
        with zipfile.ZipFile(p, "w") as z:
            z.write(csvpath, "table.csv")
        result.append(
            {
                "path": str(p),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "year": 2026,
                "month": month,
            }
        )
    return result


def test_full_stream_keeps_rows_daily_cancels_and_one_resource_per_operating_flight(
    tmp_path,
):
    out = tmp_path / "new"
    sources = archives(tmp_path)
    r = m.compile_service(sources, scope(), ZONES, out)
    assert (
        r["rows_read"] == 32
        and r["rows_kept"] == 24
        and r["physical_resources"] == 16
        and r["nonstop_products"] == 24
    )
    assert (
        r["scoped_source_totals"]["cancelled"] == 8
        and r["scoped_source_totals"]["diverted"] == 8
    )
    assert r["ambiguous_marketing_resources"] == 8 and r["conflicting_resources"] == 8
    raw = pq.read_table(out / "source_rows.parquet").to_pylist()
    assert len(raw) == 24 and len({x["source_row_id"] for x in raw}) == 24
    daily = pq.read_table(out / "daily_marketing.parquet").to_pylist()
    assert (
        sum(x["source_rows"] for x in daily) == 24
        and sum(x["cancelled"] for x in daily) == 8
    )
    phys = pq.read_table(out / "physical_resources.parquet").to_pylist()
    assert len({x["physical_id"] for x in phys}) == 16
    products = pq.read_table(out / "nonstop_products.parquet").to_pylist()
    assert len({x["product_id"] for x in products}) == 24
    assert len({x["physical_id"] for x in products}) == 16
    assert r["full_V10_pass"] is False and r["scientific_budget"] is None


def test_post_event_rows_cannot_change_pre_event_incumbent_selection(tmp_path):
    s = scope()
    before = m.select_incumbents(s)
    r = m.compile_service(archives(tmp_path), s, ZONES, tmp_path / "new")
    assert r["incumbents"] == before
    assert "UA" not in r["incumbents"][0]["strategic_carriers"]


def test_wrong_reporting_period_fails_in_new_output_and_retains_partial_files(tmp_path):
    out = tmp_path / "new"
    with pytest.raises(ValueError, match="reporting period"):
        m.compile_service(archives(tmp_path, True), scope(), ZONES, out)
    assert (out / "source_rows.parquet").exists() and (
        out / "service_index.sqlite"
    ).exists()


def test_reduction_of_required_months_is_not_accepted(tmp_path):
    with pytest.raises(ValueError, match="eight"):
        m.compile_service([], scope(), ZONES, tmp_path / "new")


def test_true_dst_fold_ambiguity_is_retained():
    assert (
        m.scheduled_times("2026-11-01", "0115", "0145", 30, "AAA", "BBB", ZONES)[
            "timing_status"
        ]
        == "ambiguous_local_departure"
    )


def test_2400_and_next_day_zero_share_physical_id_across_source_months():
    selection = m.select_incumbents(scope())[0]
    a = m.normalized_row(
        row(FL_DATE="04/30/2026 12:00:00 AM", CRS_DEP_TIME="2400", CRS_ARR_TIME="0200"),
        {**source(), "month": 4},
        0,
        selection,
        ZONES,
    )
    b = m.normalized_row(
        row(FL_DATE="05/01/2026 12:00:00 AM", CRS_DEP_TIME="0000", CRS_ARR_TIME="0200"),
        {**source(), "month": 5, "sha256": "b" * 64},
        0,
        selection,
        ZONES,
    )
    assert (
        a["physical_id"] == b["physical_id"]
        and a["source_row_id"] != b["source_row_id"]
    )
    assert a["service_date"] == "2026-04-30" and b["service_date"] == "2026-05-01"


def test_unresolved_times_do_not_remove_scoped_rows_or_cancellation_counts(tmp_path):
    r = m.compile_service(archives(tmp_path), scope(), {}, tmp_path / "new")
    assert (
        r["rows_kept"] == 24 and r["physical_resources"] == r["nonstop_products"] == 0
    )
    assert (
        r["scoped_source_totals"]["unresolved_physical_identity"] == 24
        and r["scoped_source_totals"]["cancelled"] == 8
    )


def test_task_uses_actual_cell_context_and_seals_all_closed_outputs(
    tmp_path, monkeypatch
):
    from airline_market.campaign.worker import CellContext

    monkeypatch.setitem(sys.modules, "airline_market.pipeline.event_network", m)
    if os.environ.get("EVENT_NETWORK_TASK_SOURCE"):
        spec = importlib.util.spec_from_file_location(
            "staged_network_task", os.environ["EVENT_NETWORK_TASK_SOURCE"]
        )
        task = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task)
    else:
        from airline_market.experiments import event_network_study as task
    sources = archives(tmp_path)
    sp = tmp_path / "scope.json"
    sp.write_text(json.dumps(scope()))
    zp = tmp_path / "zones.json"
    zp.write_text(json.dumps({"airport_timezones": ZONES}))
    out = tmp_path / "actual_context"
    out.mkdir()
    p = {
        "marketing_sources": sources,
        "candidate_scope": str(sp),
        "candidate_scope_sha256": hashlib.sha256(sp.read_bytes()).hexdigest(),
        "timezone_config": str(zp),
        "timezone_config_sha256": hashlib.sha256(zp.read_bytes()).hexdigest(),
        "candidate_scope_hash_scope": "unit saved snapshot",
    }
    ctx = CellContext(out)
    result = task.run({"parameters": p}, ctx)
    assert (
        result["status"] == "completed"
        and result["rows_kept"] == 24
        and ctx.counters()["stage"] == "complete"
    )
    assert len(result["artifact_hashes"]) == 6
    for name, h in result["artifact_hashes"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == h
    assert (
        result["simulated_physical_cohorts"] == result["fits"] == 0
        and result["raw_rng_call_count"] is None
    )
