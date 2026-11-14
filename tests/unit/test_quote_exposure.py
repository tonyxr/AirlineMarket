"""Exposure denominators, frozen holdout forecasting, identity and signed contrasts."""

import importlib.util
import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from airline_market.campaign.artifacts import atomic_json, file_hash

if os.environ.get("EXPOSURE_STAGE"):
    spec = importlib.util.spec_from_file_location(
        "airline_market.experiments.quote_exposure_study",
        Path(os.environ["EXPOSURE_STAGE"]) / "quote_exposure_study.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
from airline_market.experiments import quote_exposure_study as study


def row(flight, day, basis, fare, seats):
    q = date.fromisoformat(flight).month // 3 + (
        date.fromisoformat(flight).month % 3 != 0
    )
    return {
        "flightDate": flight,
        "searchDate": str(date.fromisoformat(flight) - timedelta(days=day)),
        "startingAirport": "ORD",
        "destinationAirport": "BOS",
        "path_network": "UA",
        "quarter": q,
        "split": "calibration" if q == 2 else "validation",
        "days_before_departure": day,
        "segmentsDepartureTimeEpochSeconds": "1655000000",
        "segmentsArrivalTimeEpochSeconds": "1655010000",
        "segmentsDepartureAirportCode": "ORD",
        "segmentsArrivalAirportCode": "BOS",
        "segmentsAirlineCode": "UA",
        "segmentsCabinCode": "coach",
        "fareBasisCode": basis,
        "isNonStop": "True",
        "isBasicEconomy": "False",
        "isRefundable": "False",
        "model_path_eligible": True,
        "left_truncated_by_collection_start": date.fromisoformat(flight)
        - timedelta(days=60)
        < date(2022, 4, 16),
        "totalFare": fare,
        "seatsRemaining": seats,
    }


def fixture(tmp_path, batch=3):
    q2 = []
    for flight in ["2022-06-15", "2022-06-16"]:
        q2 += [row(flight, 50, "A", 100.0, 9.0)] * 10 + [
            row(flight, 7, "A", 120.0, 10.0)
        ]
        q2 += [row(flight, 50, "B", 200.0, 9.0)] + [
            row(flight, 7, "B", 180.0, 10.0)
        ] * 10
        q2 += [row(flight, 7, "ONLY_LATE", 90.0, 10.0)]
    q2 += [row("2022-05-10", 7, "TRUNCATED", 999.0, 10.0)]
    missing = row("2022-06-16", 7, "MISSING", 100.0, 9.0)
    missing["segmentsCabinCode"] = None
    q2 += [missing]
    q3 = [
        row("2022-07-01", 50, "A", 100.0, 10.0),
        row("2022-07-01", 7, "A", 80.0, 9.0),
        row("2022-07-02", 50, "B", 150.0, 10.0),
        row("2022-07-02", 7, "B", 120.0, 9.0),
    ]
    files = {}
    for q, rows in [(2, q2), (3, q3)]:
        path = tmp_path / f"q{q}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        files[str(q)] = str(path)
    receipt = tmp_path / "source_receipt.json"
    atomic_json(
        receipt, {"artifact_hashes": {p: file_hash(Path(p)) for p in files.values()}}
    )
    return {
        "quarter_files": files,
        "source_receipt": str(receipt),
        "input_hashes": {
            **{p: file_hash(Path(p)) for p in files.values()},
            str(receipt): file_hash(receipt),
        },
        "expected_rows": {"2": len(q2), "3": len(q3)},
        "batch_size": batch,
        "collection_start": "2022-04-16",
        "prior_half_count": 0.5,
        "bootstrap_replicates": 2000,
    }


def test_display10_is_not9_and_noninventory_categories_retained():
    assert study.seat_category(9) == 9 and study.seat_category(10) == 10
    assert (
        study.seat_category(-1)
        == study.seat_category(1.5)
        == study.seat_category(11)
        == 11
    )
    assert study.seat_category(None) == study.seat_category(float("nan")) == 12


def test_identity_uses_reported_schedule_and_fare_product_not_prices_or_legId():
    a = row("2022-06-15", 50, "A", 100.0, 9.0)
    b = {**a, "totalFare": 120.0, "seatsRemaining": 10.0, "legId": "another"}
    assert study.reported_key(a) == study.reported_key(b)
    assert study.reported_key(a) != study.reported_key({**a, "fareBasisCode": "B"})
    assert study.reported_key({**a, "segmentsCabinCode": None}) is None


def test_real_driver_corrects_duplicate_exposure_without_zero_filling_or_Q3_refit(
    tmp_path,
):
    p = fixture(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    r = study.run({"parameters": p}, SimpleNamespace(directory=output))
    assert (
        r["status"] == "completed"
        and r["full_V5_pass"] is None
        and r["full_V6_pass"] is None
    )
    assert r["Q2"]["counts"]["nominal_left_truncated_in_window"] == 1
    assert r["Q2"]["counts"]["missing_reported_identity"] == 1
    q2 = r["matched_fares"]["2"]
    assert (
        q2["mean"] == pytest.approx(0.0) and q2["n_units"] == 4 and q2["clusters"] == 2
    )
    assert q2["coverage"]["unpaired_window_keys"] == 2
    assert r["matched_fares"]["3"]["mean"] == pytest.approx(-25.0)
    assert r["Q3_display"]["log_score_difference"]["mean"] < 0
    assert r["Q3_display"]["observed_keydays"] == 4
    model = json.loads((output / "frozen_Q2_display_model.json").read_text())
    freeze = json.loads((output / "Q2_model_freeze.json").read_text())
    assert model["fit_quarter"] == 2 and freeze["Q3_rows_processed_yet"] == 0
    assert (
        freeze["model_sha256"]
        == r["Q2_model_sha256"]
        == file_hash(output / "frozen_Q2_display_model.json")
    )
    assert r["artifact_hashes"]["observed_exposure.sqlite"] == file_hash(
        output / "observed_exposure.sqlite"
    )


def test_positive_fare_missing_does_not_delete_display_observation(tmp_path):
    p = fixture(tmp_path)
    path = Path(p["quarter_files"]["2"])
    rows = pq.read_table(path).to_pylist()
    rows.append(row("2022-06-15", 7, "A", None, 0.0))
    pq.write_table(pa.Table.from_pylist(rows), path)
    db = study.connect(tmp_path / "x.sqlite")
    p["expected_rows"]["2"] = len(rows)
    result = study.ingest(db, path, 2, p, lambda x: None)
    assert result["counts"]["invalid_fare_retained_for_display"] == 1
    assert db.execute("SELECT SUM(s0) FROM days").fetchone()[0] == 1
    assert (
        db.execute("SELECT SUM(n) FROM days").fetchone()[0]
        == result["counts"]["selected_offer_rows"]
    )
    db.close()


def test_Q3_changes_cannot_change_a_fitted_Q2_forecast():
    train = [
        (
            "2022-06-15",
            "UA",
            "nonstop_hub",
            "0-2",
            1,
            5,
            *([0.0] * 9 + [1.0, 0.0, 0.0, 0.0]),
        )
    ]
    a = study.fit_display(train)
    before = json.dumps(a, sort_keys=True)
    for cat in [9, 10]:
        counts = [0.0] * 13
        counts[cat] = 1
        study.score_display(
            [("2022-07-01", "UA", "nonstop_hub", "0-2", 1, 1, *counts)], a
        )
        assert json.dumps(a, sort_keys=True) == before
    assert all(x > 0 for x in a["pooled"])


def test_cluster_units_and_zero_sample_variance_are_not_independent_offers():
    r = study.cluster_summary([2, 8], [1, 4], "unit", B=50)
    assert r["mean"] == 2 and r["n_units"] == 5 and r["clusters"] == 2
    assert r["zero_sample_SD_flag"] and r["bootstrap95"]["percentile"] == [2, 2]
    one = study.cluster_summary([4], [2], "one", B=50)
    assert one["normal95"] is None and one["mean"] == 2


@pytest.mark.parametrize(
    "change",
    [
        {"quarter": 3},
        {"left_truncated_by_collection_start": True},
        {"days_before_departure": 8},
    ],
)
def test_partition_truncation_and_day_corruption_refused(tmp_path, change):
    bad = {**row("2022-06-15", 7, "A", 100.0, 9.0), **change}
    path = tmp_path / "bad.parquet"
    pq.write_table(pa.Table.from_pylist([bad]), path)
    p = {"batch_size": 3, "expected_rows": {"2": 1}, "collection_start": "2022-04-16"}
    db = study.connect(tmp_path / "bad.sqlite")
    with pytest.raises(ValueError):
        study.ingest(db, path, 2, p, lambda x: None)
    db.close()


def test_batch_partition_preserves_weighted_forecast_and_matched_statistic(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    results = []
    for directory, batch in [(a, 2), (b, 100)]:
        p = fixture(directory, batch)
        output = directory / "output"
        output.mkdir()
        r = study.run({"parameters": p}, SimpleNamespace(directory=output))
        results.append(r)
    assert results[0]["matched_fares"] == results[1]["matched_fares"]
    assert results[0]["Q3_display"] == results[1]["Q3_display"]
