"""A.11/A.15/V10: exact weighted moments, all raw strata and missing mass."""

import importlib.util
import os
import sys
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

if os.environ.get("EVENT_MOMENT_SOURCE"):
    s = importlib.util.spec_from_file_location(
        "staged_moments", os.environ["EVENT_MOMENT_SOURCE"]
    )
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
else:
    from airline_market.pipeline import event_fare_moments as m


def row(**kw):
    x = dict(
        RpYear=2026,
        RpMonth=1,
        SchFlYear=2026,
        SchFlMonth=1,
        Origin="AAA",
        Dest="BBB",
        MktCarrier="AA",
        MktCarrierAirlineID=123,
        PurchaseWindowGroup="21AP",
        _purchase_window_status="reported_group",
        _event_partition="pre_event",
        Nonstop=1,
        V_Yield=1,
        Passengers=1.0,
        MktAmount=100.1,
        MktTax=20.2,
        _sample_expansion=2.5,
        _source_row=0,
    )
    x.update(kw)
    return x


def source(tmp_path, rows):
    path = tmp_path / "input.parquet"
    schema = pa.schema(
        [
            (
                k,
                pa.int64()
                if k
                in [
                    "RpYear",
                    "RpMonth",
                    "SchFlYear",
                    "SchFlMonth",
                    "MktCarrierAirlineID",
                    "Nonstop",
                    "V_Yield",
                    "_source_row",
                ]
                else pa.float64()
                if k in m.VALUES
                else pa.string(),
            )
            for k in rows[0]
        ]
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    pax = sum(
        (
            Decimal(str(z["Passengers"]))
            for z in rows
            if z["Passengers"] is not None
            and str(z["Passengers"]) not in ["nan", "inf", "-inf"]
        ),
        Decimal(0),
    )
    return dict(
        path=str(path),
        year=2026,
        month=1,
        expected_rows=len(rows),
        expected_observed_passengers=str(pax),
    )


def run(tmp_path, rows, batch=2, out="out"):
    result = m.aggregate([source(tmp_path, rows)], tmp_path / out, batch_rows=batch)
    bank = pq.read_table(tmp_path / out / "event_fare_moments.parquet").to_pylist()
    return result, bank


def test_known_truth_recovers_exact_decimal_numerators_and_expansion(tmp_path):
    r, b = run(tmp_path, [row(), row(MktAmount=200.2, MktTax=30.3, Passengers=2.0)])
    x = b[0]
    assert (
        Decimal(x["gross_numerator"]) == Decimal("500.5")
        and Decimal(x["gross_denominator"]) == 3
    )
    assert x["gross_complete_mean"] == float(Decimal("500.5") / 3)
    assert x["gross_expanded_complete_mean"] == x["gross_complete_mean"]
    assert Decimal(x["raw_expanded_passengers"]) == Decimal("7.5")
    assert (
        r["passenger_source_totals_match"]
        and not r["calibration_adopted"]
        and not r["full_A11_pass"]
    )


def test_all_records_preserved_with_zero_negative_fares_and_validity_flags(tmp_path):
    r, b = run(
        tmp_path,
        [
            row(MktAmount=-10, V_Yield=0),
            row(MktAmount=0, V_Yield=0),
            row(MktAmount=100, V_Yield=1),
        ],
    )
    assert r["rows"] == 3 and len(b) == 2
    x = next(x for x in b if x["V_Yield"] == 0)
    assert x["gross_complete_mean"] == -5 and x["gross_nonpositive_rows"] == 2


@pytest.mark.parametrize(
    "field,value,counter",
    [
        ("Passengers", None, "passenger_missing_rows"),
        ("Passengers", -1.0, "passenger_negative_rows"),
        ("MktAmount", None, "gross_missing_rows"),
        ("MktAmount", float("nan"), "gross_missing_rows"),
        ("MktAmount", float("inf"), "gross_missing_rows"),
    ],
)
def test_unknown_mass_makes_complete_mean_null_and_retains_supported_mean(
    tmp_path, field, value, counter
):
    _, b = run(tmp_path, [row(**{field: value}), row()])
    x = b[0]
    assert x[counter] == 1 and x["gross_complete_mean"] is None
    assert x["gross_known_support_mean"] == 100.1


def test_zero_weight_with_missing_money_does_not_invent_passenger_mass(tmp_path):
    _, b = run(tmp_path, [row(Passengers=0.0, MktAmount=None), row()])
    x = b[0]
    assert Decimal(x["gross_denominator"]) == 1 and x["gross_complete_mean"] == 100.1
    assert x["passenger_zero_rows"] == 1 and x["gross_missing_rows"] == 1


def test_all_zero_weights_have_no_mean(tmp_path):
    _, b = run(tmp_path, [row(Passengers=0.0)])
    assert b[0]["gross_known_support_mean"] is b[0]["gross_complete_mean"] is None


def test_invalid_expansion_affects_expanded_measure_only(tmp_path):
    _, b = run(tmp_path, [row(_sample_expansion=0)])
    x = b[0]
    assert (
        x["gross_complete_mean"] == 100.1 and x["gross_expanded_complete_mean"] is None
    )
    assert (
        x["complete_expanded_passengers"] is None and x["expansion_invalid_rows"] == 1
    )


def test_raw_null_empty_other_windows_and_nonstop_strata_survive(tmp_path):
    r, b = run(
        tmp_path,
        [
            row(PurchaseWindowGroup=None, _purchase_window_status="missing"),
            row(PurchaseWindowGroup="", _purchase_window_status="missing"),
            row(
                PurchaseWindowGroup="OTHER", _purchase_window_status="other", Nonstop=0
            ),
        ],
    )
    assert len(b) == 3 and {x["PurchaseWindowGroup"] for x in b} == {None, "", "OTHER"}
    assert sum(x["rows"] for x in b) == r["rows"] == 3


def test_reporting_and_scheduled_months_remain_distinct_no_target_adoption(tmp_path):
    _, b = run(tmp_path, [row(SchFlMonth=5, _event_partition="post_event"), row()])
    assert len(b) == 2 and all(
        x["RpMonth"] == 1 and x["calibration_adopted"] is False for x in b
    )
    assert {x["_event_partition"] for x in b} == {"pre_event", "post_event"}


def test_tax_missing_and_negative_are_separate_from_gross(tmp_path):
    _, b = run(tmp_path, [row(MktTax=-1.25), row(MktTax=None)])
    x = b[0]
    assert x["gross_complete_mean"] == 100.1 and x["tax_complete_mean"] is None
    assert x["tax_known_support_mean"] == -1.25 and x["tax_negative_rows"] == 1


def test_batch_partition_is_exactly_reproducible(tmp_path):
    rows = [row(MktAmount=v) for v in [0.1, 0.2, 0.3, 200000.45, 0.2] * 17]
    _, a = run(tmp_path, rows, batch=1, out="a")
    _, b = run(tmp_path, rows, batch=11, out="b")
    assert a == b


@pytest.mark.parametrize(
    "change,message",
    [
        (dict(expected_rows=99), "row count"),
        (dict(expected_observed_passengers="99"), "passengers"),
        (dict(month=2), "reporting period"),
    ],
)
def test_source_conservation_mismatch_is_explicit(tmp_path, change, message):
    s = source(tmp_path, [row()])
    s.update(change)
    with pytest.raises(ValueError, match=message):
        m.aggregate([s], tmp_path / "out")


def test_actual_worker_context_and_input_hashes(tmp_path, monkeypatch):
    from airline_market.campaign.artifacts import file_hash
    from airline_market.campaign.worker import CellContext

    monkeypatch.setitem(sys.modules, "airline_market.pipeline.event_fare_moments", m)
    if os.environ.get("EVENT_MOMENT_TASK_SOURCE"):
        s = importlib.util.spec_from_file_location(
            "staged_moment_task", os.environ["EVENT_MOMENT_TASK_SOURCE"]
        )
        task = importlib.util.module_from_spec(s)
        s.loader.exec_module(task)
    else:
        from airline_market.experiments import event_fare_moment_study as task
    src = source(tmp_path, [row(), row(MktAmount=200)])
    out = tmp_path / "context"
    out.mkdir()
    ctx = CellContext(out)
    p = dict(
        sources=[src],
        batch_rows=1,
        pinned_files=[dict(path=src["path"], sha256=file_hash(src["path"]))],
    )
    r = task.run(dict(parameters=p), ctx)
    assert (
        r["status"] == "completed"
        and ctx.counters()["stage"] == "complete"
        and r["simulated_physical_cohorts"] == r["fits"] == 0
    )
    for n, h in r["artifact_hashes"].items():
        assert file_hash(out / n) == h
    assert r["raw_rng_call_count"] is None
    Path(src["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed moment input"):
        task.run(dict(parameters=p), ctx)
