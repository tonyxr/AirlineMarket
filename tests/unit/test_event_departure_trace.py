"""A.15/V10: source membership, signed decompositions and unadopted basis checks."""

import hashlib
import importlib.util
import json
import os
import sqlite3
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

if os.environ.get("EVENT_TRACE_SOURCE"):
    s = importlib.util.spec_from_file_location(
        "staged_trace", os.environ["EVENT_TRACE_SOURCE"]
    )
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
else:
    from airline_market.pipeline import event_departure_trace as m


def row(os=10, op=8, ts=10, tp=8, **kw):
    x = dict(
        year=2026,
        month=1,
        operating_airline_id=123,
        origin="AAA",
        dest="BBB",
        join_status="matched",
        coverage="available",
        on_time_scheduled=os,
        on_time_non_cancelled=op,
        T100_DEPARTURES_SCHEDULED=ts,
        T100_DEPARTURES_PERFORMED=tp,
        scheduled_difference=None if os is None or ts is None else os - ts,
        performed_difference=None if op is None or tp is None else op - tp,
        on_time_completed_proxy_known=op,
        on_time_diverted_known=0,
        on_time_source_rows=os,
        T100_source_rows=1,
    )
    x.update(kw)
    return x


@pytest.mark.parametrize(
    "x,expected",
    [
        (row(), "both_components_agree"),
        (row(op=7), "schedule_agrees_net_shortfall_differs"),
        (row(os=11, op=9), "schedule_differs_net_shortfall_agrees"),
        (row(os=11, op=7), "both_components_differ"),
        (row(os=11, op=8), "nonzero_components_offset"),
        (
            row(
                op=None, on_time_completed_proxy_known=None, on_time_diverted_known=None
            ),
            "matched_missing_count",
        ),
    ],
)
def test_signed_arithmetic_cases(x, expected):
    z = m.trace_row(x)
    assert z["trace_category"] == expected
    assert z["decomposition_residual"] in [None, 0]
    assert (
        not z["primary_count_basis_changed"] and not z["cancellation_cause_identified"]
    )


def test_negative_T100_net_shortfall_never_called_cancellations():
    z = m.trace_row(row(os=4, op=4, ts=0, tp=4))
    assert (
        z["T100_schedule_minus_performed"] == -4
        and z["T100_performed_exceeds_scheduled"]
    )
    assert z["trace_category"] == "nonzero_components_offset"
    assert not z["cancellation_cause_identified"]


def test_diversion_sensitivity_does_not_replace_primary():
    z = m.trace_row(row(on_time_completed_proxy_known=5, on_time_diverted_known=3))
    assert (
        z["performed_difference"] == 0
        and z["completed_proxy_difference_sensitivity"] == -3
    )
    assert (
        z["completed_basis_change_sensitivity"] == -3
        and z["primary_count_basis_changed"] is False
    )


def test_missing_August_counterpart_stays_null():
    z = m.trace_row(
        row(
            ts=None,
            tp=None,
            month=8,
            coverage="missing_source_month",
            join_status="on_time_only",
            T100_source_rows=None,
        )
    )
    assert (
        z["performed_difference"]
        is z["decomposition_residual"]
        is z["net_shortfall_difference"]
        is None
    )
    assert (
        z["trace_category"] == "on_time_only"
        and z["coverage"] == "missing_source_month"
    )


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("performed_difference", 91, "saved count arithmetic"),
        ("scheduled_difference", 91, "saved count arithmetic"),
        ("on_time_diverted_known", 2, "completed plus diverted"),
    ],
)
def test_inconsistent_saved_fields_raise(field, value, message):
    with pytest.raises(ValueError, match=message):
        m.trace_row(row(**{field: value}))


def banks(tmp_path):
    dbpath = tmp_path / "prepared.sqlite"
    db = sqlite3.connect(dbpath)
    db.execute("CREATE TABLE rows(y,m,opid,o,d,sid,k,c,v,utc_id)")
    db.executemany(
        "INSERT INTO rows VALUES(?,?,?,?,?,?,?,?,?,?)",
        [
            (2026, 1, 123, "AAA", "BBB", "a", "flight1", 0, 0, "utc1"),
            (2026, 1, 123, "AAA", "BBB", "alias", "flight1", 0, 0, "utc1"),
            (2026, 1, 123, "AAA", "BBB", "b", "flight2", 0, 1, None),
        ],
    )
    db.commit()
    db.close()
    projection = tmp_path / "t100.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                dict(
                    YEAR="2026",
                    MONTH="1",
                    AIRLINE_ID="123.00",
                    ORIGIN="AAA",
                    DEST="BBB",
                    CLASS="F",
                    source_row_ordinal=7,
                ),
                dict(
                    YEAR="2026",
                    MONTH="1",
                    AIRLINE_ID="123",
                    ORIGIN="AAA",
                    DEST="BBB",
                    CLASS="L",
                    source_row_ordinal=8,
                ),
            ]
        ),
        projection,
    )
    parent = dict(
        operator_route_months=[
            row(
                os=2,
                op=2,
                ts=3,
                tp=2,
                on_time_completed_proxy_known=1,
                on_time_diverted_known=1,
                on_time_source_rows=3,
            )
        ]
    )
    return dbpath, projection, parent


def test_every_group_keeps_shared_marketing_rows_and_UTC_exceptions(tmp_path):
    db, p, parent = banks(tmp_path)
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    z = m.run_trace(parent, db, p)
    assert (
        z["source_rows"] == 3 and z["source_witnesses"][0]["reported_identities"] == 2
    )
    assert (
        z["source_witnesses"][0]["UTC_unresolved_rows"] == 1
        and z["T100_projection_rows"] == 2
    )
    assert len(z["T100_witnesses"]) == 2 and len(z["trace_rows"]) == 1
    assert z["trace_rows"][0]["trace_category"] == "nonzero_components_offset"
    assert (
        z["discrepancies_resolved"] == 0
        and z["scientific_budget"] is None
        and z["full_V10_pass"] is False
    )
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


def test_changed_source_membership_is_rejected(tmp_path):
    db, p, parent = banks(tmp_path)
    parent["operator_route_months"][0]["on_time_source_rows"] = 99
    with pytest.raises(ValueError, match="source group"):
        m.run_trace(parent, db, p)


def test_actual_worker_context_hashes_and_input_pin(tmp_path, monkeypatch):
    from airline_market.campaign.worker import CellContext

    monkeypatch.setitem(sys.modules, "airline_market.pipeline.event_departure_trace", m)
    if os.environ.get("EVENT_TRACE_TASK_SOURCE"):
        s = importlib.util.spec_from_file_location(
            "staged_trace_task", os.environ["EVENT_TRACE_TASK_SOURCE"]
        )
        task = importlib.util.module_from_spec(s)
        s.loader.exec_module(task)
    else:
        from airline_market.experiments import event_departure_trace_study as task
    db, p, parent = banks(tmp_path)
    j = tmp_path / "comparison.json"
    j.write_text(json.dumps(parent))
    out = tmp_path / "context"
    out.mkdir()
    ctx = CellContext(out)
    params = dict(
        prepared_index=str(db),
        T100_projection=str(p),
        comparison=str(j),
        pinned_files=[
            dict(path=str(j), sha256=hashlib.sha256(j.read_bytes()).hexdigest())
        ],
    )
    r = task.run(dict(parameters=params), ctx)
    assert r["status"] == "completed" and ctx.counters()["stage"] == "complete"
    for n, h in r["artifact_hashes"].items():
        assert hashlib.sha256((out / n).read_bytes()).hexdigest() == h
    assert r["raw_rng_call_count"] is None and r["simulated_physical_cohorts"] == 0
    j.write_text("{}")
    with pytest.raises(ValueError, match="changed departure-trace input"):
        task.run(dict(parameters=params), ctx)
