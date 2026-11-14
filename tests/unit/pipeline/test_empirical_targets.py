"""Target preparation preserves passenger denominators and temporal separation."""

import json
from pathlib import Path
import pandas as pd
import pytest

from airline_market.pipeline.empirical_targets import (
    histogram_median,
    run,
    _quote_targets,
)


def test_exact_cent_frequency_median():
    assert histogram_median({10000: 1, 30000: 1}) == 200.0
    assert histogram_median({10000: 3, 30000: 1}) == 100.0
    assert histogram_median({10000: 10, 30000: 10}) == 200.0
    with pytest.raises(ValueError):
        histogram_median({10000: 0.5})


def test_q2_targets_all_carrier_denominators_and_complete_receipt(tmp_path):
    roots = {
        name: tmp_path / name
        for name in ["quotes", "markets", "capacity", "flights", "weather"]
    }
    for path in roots.values():
        path.mkdir()
    quote = {
        "quarter": 2,
        "split": "calibration",
        "flightDate": "2022-06-01",
        "market": "ORD-BOS",
        "path_network": "UA",
        "days_bin": "3-6",
        "isNonStop": "True",
        "totalFare": 100.0,
        "baseFare": 80.0,
        "model_path_eligible": True,
        "left_truncated_by_collection_start": True,
        "seatsRemaining": 9.0,
    }
    pd.DataFrame([quote, {**quote, "totalFare": 300.0}]).to_parquet(
        roots["quotes"] / "quotes_q2.parquet", index=False
    )
    traffic = {
        "quarter": 2,
        "split": "calibration",
        "market": "ORD-BOS",
        "path_network": "UA",
        "model_path_eligible": True,
        "hub_connecting": False,
        "passengers_expanded": 10.0,
        "passengers_sample": 1.0,
        "fare_eligible": True,
        "market_fare": 150.0,
        "n_coupons": 1,
    }
    pd.DataFrame(
        [
            traffic,
            {
                **traffic,
                "path_network": "DL",
                "model_path_eligible": False,
                "passengers_expanded": 30.0,
                "passengers_sample": 3.0,
            },
        ]
    ).to_parquet(roots["markets"] / "db1b_scoped_q2.parquet", index=False)
    pd.DataFrame(
        [
            {
                "month": 6,
                "allocated_passengers": 80.0,
                "allocated_seats": 100.0,
                "allocated_departures_performed": 1.0,
            },
            {
                "month": 7,
                "allocated_passengers": 1.0,
                "allocated_seats": 10000.0,
                "allocated_departures_performed": 1.0,
            },
        ]
    ).to_parquet(roots["capacity"] / "marketing_leg_month.parquet", index=False)
    pd.DataFrame(
        [
            {
                "split": "calibration",
                "flight_date": "2022-06-01",
                "marketing_carrier": "UA",
                "schedule_weight": 1.0,
                "performed_weight": 1.0,
            },
            {
                "split": "validation",
                "flight_date": "2022-07-01",
                "marketing_carrier": "UA",
                "schedule_weight": 1000.0,
                "performed_weight": 0.0,
            },
        ]
    ).to_parquet(roots["flights"] / "flights.parquet", index=False)
    pd.DataFrame(
        [{"DATE": "2022-06-01", "PRCP": 0.0}, {"DATE": "2022-07-01", "PRCP": 1000.0}]
    ).to_parquet(roots["weather"] / "weather_daily.parquet", index=False)
    source = tmp_path / "processing.json"
    source.write_text(
        json.dumps(
            {"stages": {name: {"output": str(root)} for name, root in roots.items()}}
        )
    )
    config = {"processing_results": str(source), "out": str(tmp_path / "result")}
    result = run(config)
    out = tmp_path / "result/attempt-0001"
    fares = pd.read_csv(out / "q2_quote_targets_by_market.csv")
    assert fares.median_total_fare.tolist() == [200.0]
    shares = pd.read_csv(out / "q2_passenger_targets.csv")
    assert (
        shares.loc[shares.path_network == "UA", "share_of_all_carrier_market"].item()
        == 0.25
    )
    assert result["capacity_rows"] == result["daily_network_rows"] == 1
    assert pd.read_csv(out / "q2_daily_supply_weather.csv").PRCP.tolist() == [0.0]
    assert result["quotes"]["left_truncated_core_offers"] == 2
    assert not result["Q3_read_for_target_construction"]
    assert result["P3_status"] == "not_run"
    assert run(config) == result
    assert len(list((tmp_path / "result").glob("attempt-*"))) == 1


def test_validation_departure_refused_in_quote_target_input(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "quarter": 3,
                "split": "validation",
                "flightDate": "2022-07-01",
                "market": "ORD-BOS",
                "path_network": "UA",
                "days_bin": "3-6",
                "isNonStop": "True",
                "totalFare": 100.0,
                "baseFare": 80.0,
                "model_path_eligible": True,
                "left_truncated_by_collection_start": False,
                "seatsRemaining": 9.0,
            }
        ]
    )
    source = tmp_path / "wrong_split.parquet"
    frame.to_parquet(source, index=False)
    with pytest.raises(ValueError, match="noncalibration"):
        _quote_targets(source, tmp_path)
