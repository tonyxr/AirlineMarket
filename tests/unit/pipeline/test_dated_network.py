"""Calendar and regional resource tests use hand-computed physical itineraries."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from airline_market.pipeline.dated_network import (
    capacity_prior,
    departure_time,
    timed_flight,
    run,
)

ZONES = {
    "ORD": "America/Chicago",
    "ATL": "America/New_York",
    "BOS": "America/New_York",
    "SFO": "America/Los_Angeles",
}


def test_elapsed_time_across_timezones_and_midnight():
    row = SimpleNamespace(
        flight_date="2022-06-01",
        crs_dep_time="0800",
        crs_arr_time="0900",
        origin="ATL",
        dest="ORD",
        scheduled_elapsed_minutes=120,
    )
    departure, arrival = timed_flight(row, ZONES)
    assert departure.isoformat() == "2022-06-01T08:00:00-04:00"
    assert arrival.isoformat() == "2022-06-01T09:00:00-05:00"
    assert arrival.timestamp() - departure.timestamp() == 7200
    row = SimpleNamespace(
        flight_date="2022-06-30",
        crs_dep_time="2300",
        crs_arr_time="0500",
        origin="SFO",
        dest="ORD",
        scheduled_elapsed_minutes=240,
    )
    departure, arrival = timed_flight(row, ZONES)
    assert arrival.isoformat() == "2022-07-01T05:00:00-05:00"
    assert (
        departure_time("2022-06-30", "2400", ZONES["ORD"]).date().isoformat()
        == "2022-07-01"
    )
    row.crs_arr_time = "0400"
    with pytest.raises(ValueError, match="disagreement"):
        timed_flight(row, ZONES)


def test_assumed_capacity_spread_preserves_mean():
    for mean in [0.0, 50.0, 76.3, 166.7]:
        low, high, p = capacity_prior(mean, 0.1)
        assert low * (1 - p) + high * p == pytest.approx(mean, abs=1e-12)
        assert 0 <= p <= 1
    with pytest.raises(ValueError):
        capacity_prior(76.0, 1.1)


def test_regional_dated_resources_and_utc_connections_do_not_force_capacity_fit(
    tmp_path,
):
    roots = {name: tmp_path / name for name in ["flights", "capacity", "markets"]}
    for path in roots.values():
        path.mkdir()
    common = {
        "year": 2022,
        "month": 6,
        "flight_date": "2022-06-01",
        "marketing_carrier": "UA",
        "operating_carrier": "OO",
        "operating_carrier_id": "20304",
        "physical_flight_id": "regional",
        "operating_flight_number": "1",
        "marketing_flight_number": "4001",
        "partner_operated": True,
        "marketing_ambiguous": False,
        "cancelled": 0,
        "origin": "ATL",
        "dest": "ORD",
        "crs_dep_time": "0800",
        "crs_arr_time": "0900",
        "scheduled_elapsed_minutes": 120.0,
        "distance": 606.0,
    }
    flights = [
        common,
        {
            **common,
            "operating_carrier": "UA",
            "operating_carrier_id": "19977",
            "partner_operated": False,
            "physical_flight_id": "main",
            "origin": "ORD",
            "dest": "BOS",
            "crs_dep_time": "1000",
            "crs_arr_time": "1300",
        },
        {
            **common,
            "marketing_carrier": "AA",
            "operating_carrier": "AA",
            "operating_carrier_id": "19805",
            "partner_operated": False,
            "physical_flight_id": "rival",
            "origin": "ORD",
            "dest": "BOS",
            "crs_dep_time": "1100",
            "crs_arr_time": "1400",
        },
        {
            **common,
            "physical_flight_id": "cancelled",
            "flight_date": "2022-06-02",
            "cancelled": 1,
        },
    ]
    pd.DataFrame(flights).to_parquet(roots["flights"] / "flights.parquet", index=False)
    capacities = [
        {
            "year": 2022,
            "month": 6,
            "operating_carrier_id": "20304",
            "origin": "ATL",
            "dest": "ORD",
            "seats": 100.0,
            "departures_performed": 2.0,
        },
        {
            "year": 2022,
            "month": 6,
            "operating_carrier_id": "19977",
            "origin": "ORD",
            "dest": "BOS",
            "seats": 200.0,
            "departures_performed": 1.0,
        },
        {
            "year": 2022,
            "month": 6,
            "operating_carrier_id": "19805",
            "origin": "ORD",
            "dest": "BOS",
            "seats": 80.0,
            "departures_performed": 1.0,
        },
    ]
    pd.DataFrame(capacities).to_parquet(
        roots["capacity"] / "operator_leg_month.parquet", index=False
    )
    pd.DataFrame(
        [
            {
                **row,
                "marketing_carrier": "AA"
                if row["operating_carrier_id"] == "19805"
                else "UA",
                "allocated_seats": row["seats"],
                "allocated_departures_performed": row["departures_performed"],
            }
            for row in capacities
        ]
    ).to_parquet(roots["capacity"] / "marketing_leg_month.parquet", index=False)
    pd.DataFrame({"market": ["ATL-ORD", "ORD-BOS", "ATL-BOS"]}).to_csv(
        roots["markets"] / "model_markets.csv", index=False
    )
    source = tmp_path / "processing.json"
    source.write_text(
        json.dumps(
            {"stages": {name: {"output": str(path)} for name, path in roots.items()}}
        )
    )
    scope = tmp_path / "scope.json"
    scope.write_text(
        json.dumps(
            {
                "airport_pool": ["ATL", "ORD", "BOS"],
                "carriers": ["UA", "AA"],
                "hub_candidate": "ORD",
            }
        )
    )
    config = {
        "scope": str(scope),
        "processing_results": str(source),
        "out": str(tmp_path / "out"),
        "airport_timezones": ZONES,
        "cohort_window": ["2022-04-01", "2022-09-30"],
        "capacity_prior_relative_spread": 0.1,
        "connection_window_minutes": [40, 300],
    }
    result = run(config)
    assert result["n_legs"] == 3
    assert result["n_products"] == 4
    assert result["n_connecting_products"] == 1
    assert result["partner_operated_legs"] == 1
    assert result["excluded_flights"] == {"cancelled_observed_service": 1}
    assert result["compiled_expected_seat_total"] == 330.0
    assert result["source_seat_total_for_core_networks"] == 380.0
    assert result["nonzero_departure_rows"] == 1
    assert result["seat_rows_above_0_1_percent"] == 1
    products = pd.read_parquet(tmp_path / "out/attempt-0001/products.parquet")
    connection = products[products.n_legs == 2].iloc[0]
    assert connection.carrier == "UA"
    assert connection.duration_h == 5.0
    assert connection.connection_wait_minutes == 60.0
    assert len(connection.leg_ids) == 2
    assert result["market_spec_status"] == "not_yet_built"
    assert run(config) == result
