"""A.15/V10: source/calendar counts, classes, outer joins and unadopted seat means."""

import csv
import hashlib
import importlib.util
import json
import os
import sys
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

if os.environ.get("EVENT_SEATS_SOURCE"):
    s = importlib.util.spec_from_file_location(
        "staged_event_seats", os.environ["EVENT_SEATS_SOURCE"]
    )
    m = importlib.util.module_from_spec(s)
    sys.modules[s.name] = m
    s.loader.exec_module(m)
else:
    from airline_market.pipeline import event_seats as m


def flight(n, **kw):
    r = {
        "source_row_id": str(n),
        "source_year": 2026,
        "source_month": 1,
        "OP_CARRIER_AIRLINE_ID": "123",
        "OP_CARRIER_FL_NUM": str(n),
        "CRS_DEP_TIME": "1000",
        "service_date": "2026-01-01",
        "origin": "AAA",
        "dest": "BBB",
        "operating_carrier": "YX",
        "marketing_carrier": "AA",
        "cancelled_flag": 0,
        "diverted_flag": 0,
        "physical_id": "utc" + str(n),
    }
    r.update(kw)
    return r


def trow(**kw):
    r = {
        "YEAR": "2026",
        "MONTH": "1",
        "AIRLINE_ID": "123",
        "UNIQUE_CARRIER": "YX",
        "ORIGIN": "AAA",
        "DEST": "BBB",
        "CLASS": "F",
        "AIRCRAFT_TYPE": "1",
        "AIRCRAFT_CONFIG": "1",
        "DEPARTURES_SCHEDULED": "3.00",
        "DEPARTURES_PERFORMED": "2.00",
        "SEATS": "300.00",
        "PASSENGERS": "210.00",
    }
    r.update(kw)
    return r


def fixture(tmp, flights=None, trows=None):
    flights = (
        flights
        if flights is not None
        else [
            flight(1),
            flight(
                2, source_row_id="alias", OP_CARRIER_FL_NUM="1", marketing_carrier="UA"
            ),
            flight(3, diverted_flag=1, physical_id=None),
            flight(4, cancelled_flag=1),
            flight(5, source_month=8, service_date="2026-08-01"),
        ]
    )
    # Explicit schema remains stable when all fixture IDs are unresolved.
    schema = pa.schema(
        [
            (
                k,
                pa.int64()
                if k
                in ["source_year", "source_month", "cancelled_flag", "diverted_flag"]
                else pa.string(),
            )
            for k in flights[0]
        ]
    )
    pq.write_table(
        pa.Table.from_pylist(flights, schema=schema), tmp / "prepared.parquet"
    )
    trows = (
        trows
        if trows is not None
        else [
            trow(),
            trow(
                CLASS="L",
                SEATS="20",
                DEPARTURES_SCHEDULED="0",
                DEPARTURES_PERFORMED="1",
            ),
            trow(AIRLINE_ID="999", UNIQUE_CARRIER="ZZ", ORIGIN="BBB", DEST="AAA"),
            trow(ORIGIN="OUT"),
        ]
    )
    p = tmp / "data.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=m.FIELDS)
        w.writeheader()
        w.writerows(trows)
    archive = tmp / "t100.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.write(p, "T100.csv")
        z.writestr(
            "documentation.csv", "SYS_FIELD_NAME,FIELD_DESC\nCLASS,Service class\n"
        )
    return archive, tmp / "prepared.parquet", {"airport_pairs": [["AAA", "BBB"]]}


def run(tmp, **kwargs):
    a, p, s = fixture(tmp, **kwargs)
    return m.run_reconciliation(a, p, s, set(range(1, 8)), tmp / "new")


def test_reported_identity_counts_utc_exceptions_and_deduplicates_shared_marketing(
    tmp_path,
):
    r = run(tmp_path)
    x = next(
        x
        for x in r["operator_route_months"]
        if x["month"] == 1 and x["join_status"] == "matched"
    )
    assert (
        r["source_rows"] == 5
        and x["on_time_source_rows"] == 4
        and x["on_time_scheduled"] == 3
    )
    assert (
        x["on_time_non_cancelled"] == 2
        and x["on_time_diverted_known"] == 1
        and x["on_time_completed_proxy_known"] == 1
    )
    assert (
        x["on_time_shared_marketing_flights"] == 1
        and x["on_time_utc_unresolved_rows"] == 1
    )
    assert (
        x["performed_difference"] == 0
        and x["seats_per_performed_departure_mean"] == 150
        and x["implied_on_time_seats"] == 300
    )
    assert (
        x["V10_native_seat_check"] is None
        and not x["native_per_flight_capacity_adopted"]
    )


def test_all_service_classes_retained_but_F_primary_and_outside_scope_not_joined(
    tmp_path,
):
    r = run(tmp_path)
    assert (
        r["T100_rows_read"] == 4
        and r["T100_scoped_rows"] == 3
        and r["class_counts"] == {"F": 2, "L": 1}
    )
    assert all(x["T100_class"] == "F" for x in r["operator_route_months"])
    assert pq.read_table(tmp_path / "new/t100_scoped_rows.parquet").num_rows == 3


def test_missing_month_and_missing_sides_are_null_not_zero(tmp_path):
    r = run(tmp_path)
    august = next(x for x in r["operator_route_months"] if x["month"] == 8)
    assert (
        august["coverage"] == "missing_source_month"
        and august["T100_SEATS"] is None
        and august["implied_on_time_seats"] is None
    )
    only = next(
        x for x in r["operator_route_months"] if x["join_status"] == "T100_only"
    )
    assert only["on_time_scheduled"] is None and only["performed_difference"] is None


def test_departure_and_implied_seat_mismatches_remain_signed_without_rescaling(
    tmp_path,
):
    r = run(tmp_path, trows=[trow(DEPARTURES_PERFORMED="3")])
    x = next(x for x in r["operator_route_months"] if x["month"] == 1)
    assert (
        x["performed_difference"] == -1
        and x["implied_on_time_seats"] == 200
        and x["implied_seat_difference"] == -100
    )
    assert (
        x["T100_SEATS"] == 300
        and r["no_seat_rescaling_or_allocation"]
        and not r["full_V10_pass"]
    )


def test_unknown_or_conflicting_cancellation_prevents_false_complete_counts(tmp_path):
    fs = [flight(1), flight(2, OP_CARRIER_FL_NUM="1", cancelled_flag=1), flight(3)]
    r = run(tmp_path, flights=fs, trows=[trow()])
    x = r["operator_route_months"][0]
    assert (
        x["on_time_status_conflict_flights"] == 1
        and x["on_time_non_cancelled_known"] == 1
    )
    assert x["on_time_non_cancelled"] is None and x["implied_on_time_seats"] is None


def test_missing_operator_ids_never_join_one_another(tmp_path):
    r = run(
        tmp_path,
        flights=[flight(1, OP_CARRIER_AIRLINE_ID=None)],
        trows=[trow(AIRLINE_ID="")],
    )
    xs = r["operator_route_months"]
    assert len(xs) == 2 and all(x["operator_id_unresolved"] for x in xs)
    assert {x["join_status"] for x in xs} == {"T100_only", "on_time_only"}


@pytest.mark.parametrize("value", ["", None, "NaN", "-1", "1.5"])
def test_invalid_measure_retained_but_aggregate_null(tmp_path, value):
    r = run(tmp_path, trows=[trow(SEATS=value)])
    assert (
        r["T100_strata"][0]["SEATS"] is None
        and r["T100_strata"][0]["SEATS_unknown_rows"] == 1
    )
    assert r["operator_route_months"][0]["implied_on_time_seats"] is None


def test_zero_departure_denominator_has_no_mean_or_ratio(tmp_path):
    r = run(tmp_path, trows=[trow(SEATS="0", DEPARTURES_PERFORMED="0")])
    x = r["operator_route_months"][0]
    assert (
        x["performed_difference"] == 2
        and x["seats_per_performed_departure_mean"] is None
        and x["implied_on_time_seats"] is None
    )
    assert x["performed_relative_difference"] is None


def test_multiple_aircraft_strata_sum_exactly(tmp_path):
    r = run(
        tmp_path,
        trows=[
            trow(SEATS="100", DEPARTURES_PERFORMED="1", AIRCRAFT_TYPE="1"),
            trow(SEATS="200", DEPARTURES_PERFORMED="1", AIRCRAFT_TYPE="2"),
        ],
    )
    x = r["T100_strata"][0]
    assert (
        x["SEATS"] == 300
        and x["DEPARTURES_PERFORMED"] == 2
        and x["aircraft_types"] == ["1", "2"]
    )


def test_period_outside_pinned_local_coverage_is_not_silently_used(tmp_path):
    with pytest.raises(ValueError, match="coverage"):
        run(tmp_path, trows=[trow(MONTH="8")])


def test_real_cell_context_and_transitive_artifact_hashes(tmp_path, monkeypatch):
    from airline_market.campaign.worker import CellContext

    monkeypatch.setitem(sys.modules, "airline_market.pipeline.event_seats", m)
    if os.environ.get("EVENT_SEATS_TASK_SOURCE"):
        s = importlib.util.spec_from_file_location(
            "staged_seats_task", os.environ["EVENT_SEATS_TASK_SOURCE"]
        )
        task = importlib.util.module_from_spec(s)
        s.loader.exec_module(task)
    else:
        from airline_market.experiments import event_seats_study as task
    a, p, scope = fixture(tmp_path)
    sp = tmp_path / "scope.json"
    sp.write_text(json.dumps(scope))
    out = tmp_path / "context"
    out.mkdir()
    params = {
        "T100_archive": str(a),
        "source_parquet": str(p),
        "candidate_scope": str(sp),
        "coverage_months": list(range(1, 8)),
        "pinned_files": [
            {"path": str(q), "sha256": hashlib.sha256(q.read_bytes()).hexdigest()}
            for q in [a, p, sp]
        ],
    }
    ctx = CellContext(out)
    r = task.run({"parameters": params}, ctx)
    assert (
        r["status"] == "completed"
        and ctx.counters()["stage"] == "complete"
        and len(r["artifact_hashes"]) == 5
    )
    for f, h in r["artifact_hashes"].items():
        assert hashlib.sha256((out / f).read_bytes()).hexdigest() == h
    assert (
        r["simulated_physical_cohorts"] == r["fits"] == 0
        and r["raw_rng_call_count"] is None
    )
