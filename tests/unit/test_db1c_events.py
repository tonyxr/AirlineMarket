"""A.15/V10: local archive fixtures, raw-row conservation and pre-event hygiene."""

import csv
import importlib.util
import io
import os
import struct
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def load(name):
    if os.environ.get("A15_STAGING_ROOT"):
        p = Path(os.environ["A15_STAGING_ROOT"]) / (name + ".py")
        spec = importlib.util.spec_from_file_location("staged_a15_" + name, p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    if name == "db1c":
        from airline_market.pipeline.sources import db1c

        return db1c
    from airline_market.pipeline import events

    return events


def table():
    return pa.table(
        {
            "Origin": ["AAA", "AAA", "BBB", "CCC"],
            "Dest": ["BBB", "CCC", "AAA", "DDD"],
            "RpYear": [2026] * 4,
            "RpMonth": [6] * 4,
            "SchFlYear": [2026] * 4,
            "SchFlMonth": [4, 5, 6, 8],
            "Passengers": [1.0, 1.0, None, 2.0],
            "MktAmount": [120.25, 130.0, None, -1.0],
            "MktTax": [12.0, None, None, 0.0],
            "PurchaseWindowGroup": ["21AP", " 2290 ", None, "bad"],
            "MktCarrier": ["NK", "AA", "UA", "WN"],
            "MktCoupons": [1, 2, 1, 1],
            "ExtraRawField": ["keep", "all", "source", "columns"],
        }
    )


def archive(tmp_path, data=None, count=1):
    path = tmp_path / "market.zip"
    buf = io.BytesIO()
    pq.write_table(table() if data is None else data, buf, row_group_size=2)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for k in range(count):
            z.writestr(f"market{k}.parquet", buf.getvalue())
        z.writestr("unused.csv.zip", b"not another representation to read")
    return path


def extract(tmp_path, source=None, **kwargs):
    m = load("db1c")
    return m.scope_market_archive(
        source or archive(tmp_path),
        [("AAA", "BBB"), ("BBB", "AAA")],
        tmp_path / "selected.parquet",
        expected_reporting_period=(2026, 6),
        sample_expansion=2.5,
        batch_rows=1,
        **kwargs,
    )


def test_scope_preserves_raw_nulls_fields_row_ids_and_both_periods(tmp_path):
    audit = extract(tmp_path)
    actual = pq.read_table(tmp_path / "selected.parquet")
    assert audit["rows_read"] == 4 and audit["rows_kept"] == 2
    assert audit["member_crc_verified"] and not audit["full_V10_pass"]
    assert actual["_source_row"].to_pylist() == [0, 2]
    assert actual["MktCarrier"].to_pylist() == ["NK", "UA"]
    assert actual["MktAmount"].to_pylist() == [120.25, None]
    assert actual["Passengers"].to_pylist() == [1.0, None]
    assert actual["MktTax"].to_pylist() == [12.0, None]
    assert actual["ExtraRawField"].to_pylist() == ["keep", "source"]
    assert actual["_purchase_window_status"].to_pylist() == [
        "reported_group",
        "missing",
    ]
    assert actual["_event_partition"].to_pylist() == ["pre_event", "post_event"]
    assert audit["national_purchase_window_status"] == {
        "reported_group": 2,
        "missing": 1,
        "other": 1,
    }


def test_same_rows_for_different_batch_sizes(tmp_path):
    src = archive(tmp_path)
    m = load("db1c")
    a = m.scope_market_archive(
        src,
        [("CCC", "DDD")],
        tmp_path / "a.parquet",
        expected_reporting_period=(2026, 6),
        sample_expansion=2.5,
        batch_rows=1,
    )
    b = m.scope_market_archive(
        src,
        [("CCC", "DDD")],
        tmp_path / "b.parquet",
        expected_reporting_period=(2026, 6),
        sample_expansion=2.5,
        batch_rows=99,
    )
    assert pq.read_table(a["output"]).equals(pq.read_table(b["output"]))
    assert pq.read_table(a["output"])["MktAmount"].to_pylist() == [-1.0]
    assert pq.read_table(a["output"])["_event_partition"].to_pylist() == [
        "outside_event_window"
    ]


def test_empty_match_writes_valid_empty_table(tmp_path):
    m = load("db1c")
    r = m.scope_market_archive(
        archive(tmp_path),
        [("XXX", "YYY")],
        tmp_path / "out.parquet",
        expected_reporting_period=(2026, 6),
        sample_expansion=2.5,
    )
    assert r["rows_kept"] == 0 and pq.read_table(r["output"]).num_rows == 0


def test_rejects_missing_window_field_and_ambiguous_parquet(tmp_path):
    src = archive(tmp_path, table().drop(["PurchaseWindowGroup"]))
    with pytest.raises(ValueError, match="missing DB1C fields"):
        extract(tmp_path, source=src)
    src = archive(tmp_path, count=2)
    with pytest.raises(ValueError, match="exactly one"):
        extract(tmp_path, source=src)


def test_crc_failure_is_not_repaired_or_published(tmp_path):
    source = archive(tmp_path)
    data = bytearray(source.read_bytes())
    at = data.index(b"PK\x01\x02")
    crc = struct.unpack_from("<I", data, at + 16)[0]
    struct.pack_into("<I", data, at + 16, crc ^ 1)
    source.write_bytes(data)
    with pytest.raises(zipfile.BadZipFile):
        extract(tmp_path, source=source)
    assert not (tmp_path / "selected.parquet").exists()


def test_existing_output_is_retained(tmp_path):
    extract(tmp_path)
    before = (tmp_path / "selected.parquet").read_bytes()
    with pytest.raises(FileExistsError):
        extract(tmp_path)
    assert (tmp_path / "selected.parquet").read_bytes() == before


def marketing_sources(tmp_path, bad_period=False):
    e = load("events")
    sources = []
    rows = [
        ["NK", "NK", "AAA", "BBB", "0", "0"],
        ["UA", "YX", "BBB", "AAA", "0", "0"],
        ["NK", "NK", "AAA", "CCC", "1", "0"],
        ["AA", "AA", "AAA", "CCC", "0", "0"],
        ["NK", "NK", "AAA", "DDD", "0", "0"],
        ["NK", "NK", "AAA", "EEE", "", "0"],
        ["AA", "AA", "AAA", "EEE", "0", "0"],
    ]
    for month in range(1, 5):
        text = io.StringIO()
        w = csv.writer(text)
        w.writerow(e.FIELDS)
        if month == 1:
            for row in rows:
                w.writerow([f"2026-{5 if bad_period else month:02d}-01", *row])
        source = tmp_path / f"mkt{month}.zip"
        with zipfile.ZipFile(source, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("T_ONTIME_MARKETING.csv", text.getvalue())
        sources.append((source, (2026, month)))
    return sources


def test_pre_event_scope_keeps_marketing_and_operating_codes(tmp_path):
    result = load("events").spirit_candidate_scope(marketing_sources(tmp_path))
    assert result["airport_pairs"] == [("AAA", "BBB")]
    assert result["directed_markets"] == [("AAA", "BBB"), ("BBB", "AAA")]
    assert any(
        x["marketing_carrier"] == "UA" and x["operating_carrier"] == "YX"
        for x in result["route_counts"]
    )
    assert any(x.get("unknown_status") == 1 for x in result["route_counts"])
    assert sum(x["rows_read"] for x in result["files"]) == 7
    assert not result["full_V10_pass"]


def test_post_event_rows_cannot_choose_pre_event_scope(tmp_path):
    with pytest.raises(ValueError, match="post-event or wrong-period"):
        load("events").spirit_candidate_scope(
            marketing_sources(tmp_path, bad_period=True)
        )


def test_all_four_pre_event_months_are_required(tmp_path):
    with pytest.raises(ValueError, match="each pre-event"):
        load("events").spirit_candidate_scope(marketing_sources(tmp_path)[:3])


def test_registered_event_task_with_native_context(tmp_path, monkeypatch):
    from airline_market.campaign.artifacts import file_hash
    from airline_market.campaign.worker import CellContext

    if os.environ.get("A15_STAGING_ROOT"):
        import sys

        monkeypatch.setitem(
            sys.modules, "airline_market.pipeline.events", load("events")
        )
        monkeypatch.setitem(
            sys.modules, "airline_market.pipeline.sources.db1c", load("db1c")
        )
        spec = importlib.util.spec_from_file_location(
            "staged_a15_task",
            Path(os.environ["A15_STAGING_ROOT"]) / "event_data_study.py",
        )
        task = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(task)
    else:
        from airline_market.experiments import event_data_study as task
    sources = marketing_sources(tmp_path)
    market = archive(tmp_path)
    p = {
        "pre_event_marketing_sources": [
            {
                "path": str(path),
                "sha256": file_hash(path),
                "year": period[0],
                "month": period[1],
            }
            for path, period in sources
        ],
        "db1c_sources": [
            {"path": str(market), "sha256": file_hash(market), "year": 2026, "month": 6}
        ],
        "sample_expansion": 2.5,
        "batch_rows": 1,
    }
    directory = tmp_path / "context"
    directory.mkdir()
    context = CellContext(directory)
    result = task.run({"parameters": p}, context)
    assert (
        result["status"] == "completed"
        and result["rows_read"] == 4
        and result["rows_kept"] == 2
    )
    assert result["simulated_physical_cohorts"] == 0 and result["fits"] == 0
    assert result["all_selected_member_crc_verified"] and not result["full_V10_pass"]
    assert context.publish()["files_completed"] == 1
