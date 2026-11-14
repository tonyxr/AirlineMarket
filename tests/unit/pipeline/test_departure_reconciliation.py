"""Local diversion fixtures distinguish a scheduled flight from physical hops."""

import json
import pandas as pd

from airline_market.pipeline.reconcile_departures import COLUMNS, flown_path, run


def record(**values):
    """Use the actual reporting-carrier header and string field representation."""
    row = {name: "" for name in COLUMNS}
    row.update(
        Year="2022",
        Month="6",
        FlightDate="2022-06-01",
        Reporting_Airline="UA",
        DOT_ID_Reporting_Airline="19977",
        Flight_Number_Reporting_Airline="1",
        Origin="ORD",
        Dest="BOS",
        CRSDepTime="0800",
        Cancelled="0.00",
        Diverted="0.00",
        DivAirportLandings="0.00",
    )
    row.update(values)
    return row


def test_diversion_hops_and_unknown_destination():
    assert flown_path(record()) == (["ORD", "BOS"], None)
    assert flown_path(record(Cancelled="1.00")) == ([], None)
    assert flown_path(
        record(
            Diverted="1.00",
            DivAirportLandings="1.00",
            Div1Airport="DTW",
            DivReachedDest="1.00",
        )
    ) == (["ORD", "DTW", "BOS"], None)
    assert flown_path(
        record(
            Diverted="1.00",
            DivAirportLandings="1.00",
            Div1Airport="DTW",
            DivReachedDest="0.00",
        )
    ) == (["ORD", "DTW"], None)
    assert (
        flown_path(
            record(Diverted="1.00", DivAirportLandings="1.00", Div1Airport="DTW")
        )[1]
        == "unknown_diversion_destination_reached"
    )


def test_reporting_diversion_reconciles_without_changing_marketing_schedule(tmp_path):
    source = tmp_path / "reporting.csv"
    pd.DataFrame(
        [
            record(),
            record(
                Origin="SFO",
                Diverted="1.00",
                DivAirportLandings="1.00",
                Div1Airport="ORD",
                DivReachedDest="1.00",
            ),
        ]
    ).to_csv(source, index=False)
    capacity = tmp_path / "capacity"
    flights = tmp_path / "flights"
    capacity.mkdir()
    flights.mkdir()
    pd.DataFrame(
        [
            {
                "year": 2022,
                "month": 6,
                "operating_carrier_id": "19977",
                "origin": "ORD",
                "dest": "BOS",
                "departures_performed": 2,
            },
            {
                "year": 2022,
                "month": 6,
                "operating_carrier_id": "19977",
                "origin": "SFO",
                "dest": "ORD",
                "departures_performed": 1,
            },
        ]
    ).to_parquet(capacity / "operator_leg_month.parquet", index=False)
    pd.DataFrame(
        [
            {
                "year": 2022,
                "month": 6,
                "operating_carrier_id": "19977",
                "origin": "ORD",
                "dest": "BOS",
                "schedule_weight": 1.0,
                "performed_weight": 1.0,
            }
        ]
    ).to_parquet(flights / "flights.parquet", index=False)
    summary = tmp_path / "processing.json"
    summary.write_text(
        json.dumps(
            {
                "stages": {
                    "capacity": {"output": str(capacity)},
                    "flights": {"output": str(flights)},
                }
            }
        )
    )
    scope = tmp_path / "scope.json"
    scope.write_text(
        json.dumps({"hub_candidate": "ORD", "airport_pool": ["ORD", "BOS", "SFO"]})
    )
    config = {
        "scope": str(scope),
        "reporting_files": [str(source)],
        "processing_results": str(summary),
        "out": str(tmp_path / "out"),
    }
    result = run(config)
    assert result["marketing_scheduled"] == result["reporting_scheduled"] == 1
    assert result["mismatched_cells_before"] == 2
    assert result["mismatched_cells_after"] == 0
    assert result["reconstructed_flown_segments"] == result["t100_segments"] == 3
    assert result["V1_status"] == "not_yet_passed"
    assert run(config) == result
