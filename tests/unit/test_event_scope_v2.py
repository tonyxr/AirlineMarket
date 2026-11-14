"""A.15/V10: actual BTS date serialization, unchanged pre-event scope and worker."""

import csv
import io
import zipfile
from datetime import date

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from airline_market.campaign.artifacts import file_hash
from airline_market.campaign.worker import CellContext
from airline_market.experiments.event_data_study_v2 import run
from airline_market.pipeline.event_scope_v2 import (
    FIELDS,
    marketing_calendar_date,
    spirit_candidate_scope,
)
from airline_market.pipeline.events import spirit_candidate_scope as original_scope


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2026-01-01", date(2026, 1, 1)),
        ("1/1/2026 12:00:00 AM", date(2026, 1, 1)),
        ("4/30/2026 12:00:00 AM", date(2026, 4, 30)),
    ],
)
def test_registered_date_serializations(raw, expected):
    assert marketing_calendar_date(raw)[0] == expected


def sources(tmp_path, us=True, future=False):
    result = []
    for month in range(1, 5):
        stamp = (
            f"{5 if future else month}/1/2026 12:00:00 AM"
            if us
            else f"2026-{5 if future else month:02d}-01"
        )
        data = io.StringIO()
        writer = csv.writer(data)
        writer.writerow(FIELDS)
        writer.writerow([stamp, "NK", "NK", "AAA", "BBB", "0", "0"])
        writer.writerow([stamp, "UA", "YX", "BBB", "AAA", "0", "0"])
        path = tmp_path / f"{month}_{us}_{future}.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("T_ONTIME_MARKETING.csv", data.getvalue())
        result.append((path, (2026, month)))
    return result


def test_old_rejection_reproduces_and_new_parser_preserves_original_ISO_scope(tmp_path):
    us, iso = sources(tmp_path), sources(tmp_path, us=False)
    with pytest.raises(ValueError, match="wrong-period"):
        original_scope(us)
    old, new = original_scope(iso), spirit_candidate_scope(us)
    for key in ["airport_pairs", "directed_markets", "route_counts", "periods"]:
        assert new[key] == old[key]
    assert all(f["date_serializations"] == {"US_timestamp": 2} for f in new["files"])
    assert new["route_counts"][1]["operating_carrier"] == "YX"


@pytest.mark.parametrize("us", [True, False])
def test_post_event_month_is_still_rejected(tmp_path, us):
    with pytest.raises(ValueError, match="wrong-period"):
        spirit_candidate_scope(sources(tmp_path, us=us, future=True))


def test_native_context_streams_all_seven_fixture_months(tmp_path):
    pre = sources(tmp_path)
    markets = []
    for month in range(1, 8):
        table = pa.table(
            {
                "Origin": ["AAA", "OUT"],
                "Dest": ["BBB", "SIDE"],
                "RpYear": [2026, 2026],
                "RpMonth": [month, month],
                "SchFlYear": [2026, 2026],
                "SchFlMonth": [month, month],
                "Passengers": [1.0, None],
                "MktAmount": [120.0, -1.0],
                "MktTax": [None, 0.0],
                "PurchaseWindowGroup": ["21AP", None],
                "MktCarrier": ["NK", "UA"],
                "ExtraRaw": ["keep", "offscope"],
            }
        )
        data = io.BytesIO()
        pq.write_table(table, data, row_group_size=1)
        path = tmp_path / f"market_{month}.zip"
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("market.parquet", data.getvalue())
        markets.append(
            {"path": str(path), "sha256": file_hash(path), "year": 2026, "month": month}
        )
    params = {
        "pre_event_marketing_sources": [
            {"path": str(path), "sha256": file_hash(path), "year": year, "month": month}
            for path, (year, month) in pre
        ],
        "db1c_sources": markets,
        "sample_expansion": 2.5,
        "batch_rows": 1,
    }
    directory = tmp_path / "run"
    directory.mkdir()
    context = CellContext(directory)
    result = run({"parameters": params}, context)
    assert (
        result["status"] == "completed"
        and result["months"] == 7
        and result["rows_read"] == 14
        and result["rows_kept"] == 7
    )
    assert (
        result["all_selected_member_crc_verified"]
        and result["simulated_physical_cohorts"] == 0
    )
    assert context.publish()["files_completed"] == 7
    assert len(list(directory.glob("db1c_2026_*.parquet"))) == 7
