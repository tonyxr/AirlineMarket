"""Local-fixture conservation, regional ambiguity and empirical split checks."""

import csv
import json
from pathlib import Path
import zipfile

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from airline_market.pipeline.regional import canonical_flights, allocate_capacity
from airline_market.pipeline.scoped_io import sha256
from airline_market.pipeline.scoped_quotes import (
    attribution_lookup,
    classify_quote,
    QUOTE_COLUMNS,
)
from airline_market.pipeline.scoped_tables import FLIGHT_COLUMNS, MARKET_COLUMNS
from airline_market.pipeline.scoped_processing import run


def flight(network="UA", operator="OO", number="1", cancelled=0):
    """Make a physical flight with explicit marketed and operating identities."""
    return dict(
        flight_date="2022-05-01",
        year=2022,
        month=5,
        marketing_carrier=network,
        marketing_carrier_id=network,
        operating_carrier=operator,
        operating_carrier_id=operator,
        operating_flight_number=number,
        origin="ORD",
        dest="D1",
        crs_dep_time="0900",
        cancelled=cancelled,
        diverted=0,
    )


def test_shared_marketing_and_duplicate_records_preserve_physical_capacity():
    frame = pd.DataFrame([flight("AA"), flight("UA"), flight("AA")])
    f, check = canonical_flights(frame)
    assert check["physical_scheduled_flights"] == 1
    assert check["duplicate_network_rows_removed"] == 1
    assert f.schedule_weight.tolist() == [0.5, 0.5]
    source = pd.DataFrame(
        [
            dict(
                year=2022,
                month=5,
                operating_carrier_id="OO",
                operating_carrier="OO",
                origin="ORD",
                dest="D1",
                seats=100.0,
                passengers=80.0,
                departures_scheduled=1.0,
                departures_performed=1.0,
                distance=500.0,
            )
        ]
    )
    allocated, checks = allocate_capacity(source, f)
    assert allocated.groupby("marketing_carrier").allocated_seats.sum().to_dict() == {
        "AA": 50.0,
        "UA": 50.0,
    }
    assert checks["seats"]["max_absolute_leg_month_error"] == 0
    assert checks["unmapped_seats"] == 0


def test_cancelled_flights_do_not_dilute_performed_flight_shares_and_unmapped_is_retained():
    f, _ = canonical_flights(
        pd.DataFrame(
            [
                flight("AA", number="1"),
                flight("UA", number="2"),
                flight("UA", number="3", cancelled=1),
            ]
        )
    )
    base = dict(
        year=2022,
        month=5,
        origin="ORD",
        dest="D1",
        seats=200.0,
        passengers=160.0,
        departures_scheduled=3.0,
        departures_performed=2.0,
        distance=500.0,
    )
    source = pd.DataFrame(
        [
            {**base, "operating_carrier_id": "OO", "operating_carrier": "OO"},
            {**base, "operating_carrier_id": "ZZ", "operating_carrier": "ZZ"},
        ]
    )
    allocated, checks = allocate_capacity(source, f)
    assert allocated.groupby("marketing_carrier").allocated_seats.sum().to_dict() == {
        "AA": 100.0,
        "UA": 100.0,
        "UNMAPPED": 200.0,
    }
    assert checks["unmapped_seat_share"] == 0.5
    assert checks["departures_performed"]["allocated_total"] == 4
    with pytest.raises(ValueError, match="conflicting"):
        canonical_flights(pd.DataFrame([flight(cancelled=0), flight(cancelled=1)]))


def test_regional_quote_attribution_requires_a_unique_observed_network():
    f, _ = canonical_flights(pd.DataFrame([flight("AA"), flight("UA")]))
    ambiguous = classify_quote(
        "OO",
        "ORD",
        "D1",
        "2022-05-01T09:00:00-05:00",
        "2022-05-01",
        "ORD",
        {"AA", "UA"},
        attribution_lookup(f),
    )
    assert ambiguous[0] == "UNMAPPED" and not ambiguous[2]
    f, _ = canonical_flights(pd.DataFrame([flight("UA")]))
    unique = classify_quote(
        "OO",
        "ORD",
        "D1",
        "2022-05-01T09:00:00-05:00",
        "2022-05-01",
        "ORD",
        {"AA", "UA"},
        attribution_lookup(f),
    )
    assert unique[0] == "UA" and unique[2] and unique[1] == "unique_operator_route_time"


def write_csv(path, fields, rows):
    """Create deterministic public-source-shaped local test inputs."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fixture_recipe(tmp_path):
    """Build a complete tiny offline recipe with a Q3-only candidate market."""
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    ids = []

    def register(id, fields, rows, zipped=False):
        path = tmp_path / (id + ".csv")
        write_csv(path, fields, rows)
        if zipped:
            archive = path.with_suffix(".zip")
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr(
                    "Documentation.csv", "SYS_FIELD_NAME,FIELD_DESC\nYEAR,Year\n"
                )
                z.write(path, path.name)
            path = archive
        (receipts / (id + ".json")).write_text(
            json.dumps(
                {
                    "status": "downloaded",
                    "path": str(path),
                    "sha256": sha256(path),
                    "source_url": "fixture://" + id,
                }
            )
        )
        ids.append(id)

    for month in range(4, 10):
        rows = []
        for network, operator in [("UA", "OO"), ("AA", "MQ")]:
            for number, (origin, dest) in enumerate([("O1", "ORD"), ("ORD", "D1")], 1):
                r = {name: "" for name in FLIGHT_COLUMNS}
                r.update(
                    YEAR="2022",
                    MONTH=str(month),
                    FL_DATE=f"{month}/15/2022 12:00:00 AM",
                    MKT_UNIQUE_CARRIER=network,
                    MKT_CARRIER_AIRLINE_ID=network,
                    OP_UNIQUE_CARRIER=operator,
                    OP_CARRIER_AIRLINE_ID=operator,
                    MKT_CARRIER_FL_NUM=str(number),
                    OP_CARRIER_FL_NUM=str(number),
                    TAIL_NUM="N1",
                    ORIGIN=origin,
                    DEST=dest,
                    CRS_DEP_TIME="0900" if number == 1 else "1200",
                    CRS_ARR_TIME="1000" if number == 1 else "1300",
                    CANCELLED="0",
                    DIVERTED="0",
                    DUP="N",
                    DISTANCE="500",
                    CRS_ELAPSED_TIME="60",
                )
                rows.append(r)
        register(f"ontime_mkt_2022_{month}", FLIGHT_COLUMNS, rows)
    cols = [
        "YEAR",
        "MONTH",
        "UNIQUE_CARRIER",
        "AIRLINE_ID",
        "ORIGIN",
        "DEST",
        "CLASS",
        "SEATS",
        "PASSENGERS",
        "DEPARTURES_SCHEDULED",
        "DEPARTURES_PERFORMED",
        "DISTANCE",
    ]
    rows = []
    for month in range(4, 10):
        for operator in ("OO", "MQ"):
            for origin, dest in [("O1", "ORD"), ("ORD", "D1")]:
                rows.append(
                    dict(
                        zip(
                            cols,
                            [
                                "2022",
                                str(month),
                                operator,
                                operator,
                                origin,
                                dest,
                                "F",
                                "100",
                                "80",
                                "1",
                                "1",
                                "500",
                            ],
                        )
                    )
                )
    register("t100_segment_2022", cols, rows, True)
    for q in (2, 3):
        rows = []
        for k, network in enumerate(["UA", "AA", "DL"]):
            r = {name: "" for name in MARKET_COLUMNS}
            path = "O1:ORD:D1" if network != "DL" else "O1:ATL:D1"
            r.update(
                ItinID=str(k),
                MktID=str(k),
                Year="2022",
                Quarter=str(q),
                Origin="O1",
                Dest="D1",
                AirportGroup=path,
                MktCoupons="2",
                TkCarrierGroup=f"{network}:{network}",
                OpCarrierGroup=f"{network}:{network}",
                TkCarrier=network,
                OpCarrier=network,
                RPCarrier=network,
                Passengers="5" if q == 2 else "2",
                MktFare="150",
                BulkFare="0",
                MktDistance="1000",
                NonStopMiles="800",
            )
            rows.append(r)
        if q == 3:
            r = rows[0].copy()
            r.update(
                ItinID="new",
                MktID="new",
                Origin="ORD",
                Dest="O1",
                AirportGroup="ORD:O1",
                MktCoupons="1",
                TkCarrierGroup="UA",
                OpCarrierGroup="OO",
            )
            rows.append(r)
        register(f"DB1BMarket_2022_{q}", MARKET_COLUMNS, rows)
    quotes = []
    for day, search in [("2022-06-15", "2022-04-16"), ("2022-07-15", "2022-06-15")]:
        for network in ["AA", "UA", "DL"]:
            r = {name: "" for name in QUOTE_COLUMNS}
            r.update(
                legId=network + day,
                searchDate=search,
                flightDate=day,
                startingAirport="O1",
                destinationAirport="D1",
                fareBasisCode="Y",
                totalFare="170",
                baseFare="150",
                seatsRemaining="9",
                isNonStop="False",
                isBasicEconomy="False",
                isRefundable="False",
                totalTravelDistance="1000",
                travelDuration="PT4H",
                segmentsDepartureTimeRaw=day
                + "T09:00:00-05:00||"
                + day
                + "T12:00:00-05:00",
                segmentsDepartureAirportCode="O1||ORD"
                if network != "DL"
                else "O1||ATL",
                segmentsArrivalAirportCode="ORD||D1" if network != "DL" else "ATL||D1",
                segmentsAirlineCode=network + "||" + network,
            )
            quotes.append(r)
    register("flightprices_v1", QUOTE_COLUMNS, quotes, True)
    register(
        "noaa_daily_2022",
        ["STATION", "DATE", "TAVG"],
        [
            {"STATION": "USW00094846", "DATE": d.strftime("%Y-%m-%d"), "TAVG": "10"}
            for d in pd.date_range("2022-04-01", "2022-09-30")
        ],
    )
    scope = {
        "hub_candidate": "ORD",
        "carriers": ["UA", "AA"],
        "airport_pool": ["O1", "ORD", "D1"],
        "quote_search_coverage": ["2022-04-16", "2022-10-05"],
        "requirements": [{"ids": ids}],
    }
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(scope))
    return {
        "schema": "airline-scoped-processing/1",
        "scope": str(path),
        "source_receipts": str(receipts),
        "processed_output": str(tmp_path / "processed"),
        "evidence_output": str(tmp_path / "results"),
        "booking_window_days": 60,
        "limits": {"seconds": 120, "rss_gb": 2},
    }


def test_scoped_offline_pipeline_keeps_competitors_quarters_and_resumes(tmp_path):
    config = fixture_recipe(tmp_path)
    assert run(config)["status"] == "completed"
    root = Path(config["processed_output"])
    q2 = pd.read_parquet(root / "markets/attempt-0001/db1b_scoped_q2.parquet")
    q3 = pd.read_parquet(root / "markets/attempt-0001/db1b_scoped_q3.parquet")
    assert len(q2) == len(q3) == 3
    assert set(q2.path_network) == {"AA", "UA", "DL"}
    assert q2.passengers_expanded.sum() == 150 and q3.passengers_expanded.sum() == 60
    assert pd.read_csv(
        root / "markets/attempt-0001/model_markets.csv"
    ).market.tolist() == ["O1-D1"]
    quotes = pd.read_parquet(root / "quotes/attempt-0001/quotes_q3.parquet")
    assert len(quotes) == 3 and set(quotes.searchDate) == {"2022-06-15"}
    assert set(quotes.split) == {"validation"}
    results = json.loads((Path(config["evidence_output"]) / "results.json").read_text())
    assert not results["F8"]["eligible"]
    before = sorted(str(p) for p in root.glob("*/*"))
    assert run(config)["status"] == "completed"
    assert sorted(str(p) for p in root.glob("*/*")) == before
