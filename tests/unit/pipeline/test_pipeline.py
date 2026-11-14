from pathlib import Path
import pytest

pytest.importorskip("pandas")
pytest.importorskip("pyarrow")
from airline_market.pipeline.catalog import MarketScope
from airline_market.pipeline.sources import t100, db1b, ontime, flightprices, noaa
from airline_market.pipeline.compiler import compile_scope
from airline_market.pipeline.observation_fit import fit_observation
from airline_market.pipeline.fieldmap import FieldMap
from airline_market.core.errors import DataValidationError
from airline_market.spec import compile_market


def scope():
    return MarketScope.from_dict(
        "fixture",
        {
            "hubs": ["SEA"],
            "carriers": ["AS", "DL"],
            "departure_date": "2024-08-14",
            "periods": [60, 30, 0],
            "sample_expansion": 10,
            "min_market_pax_per_day": 1,
            "sources": {
                k: str(Path("tests/fixtures") / k / "tiny.csv")
                for k in ["t100", "db1b", "ontime", "flightprices", "noaa", "coupon"]
            },
        },
    )


def test_sources_compiler_and_fit(tmp_path):
    s = scope()
    t = t100.load_segments(s, out=tmp_path)
    assert len(t) == 4 and t.iloc[0].seats_per_departure == 100
    f = db1b.load_markets(s, out=tmp_path)
    assert f.passengers.sum() == 6000
    assert len(db1b.load_coupons(s, out=False)) == 5
    o = ontime.load_schedule(s, out=tmp_path)
    assert len(o) == 5 and o.cancelled.sum() == 1
    panel = flightprices.load_panel(s, out=tmp_path)
    assert panel.iloc[0].segmentsAirlineCode == ["AS", "AS"]
    fit = fit_observation(panel)
    assert fit.seats_cap == 9 and fit.cap_verified and fit.fare_bin == 10
    assert len(noaa.load_weather(s, out=tmp_path)) == 5
    assert len(list(tmp_path.glob("*.json"))) == 5
    with pytest.raises(DataValidationError):
        FieldMap({"needed": []}).columns(["other"], ["needed"])
    spec = compile_scope(s)
    c = compile_market(spec)
    assert len(spec.legs) == 4 and len(spec.products) == 6
    assert len([p for p in spec.products if len(p.legs) == 2]) == 2
    assert c.spec.demand.mode == "poisson" and spec.metadata["warnings"]


def test_calibration_recovery_and_flat_direction():
    from airline_market.spec.library_builders import cx_example
    from airline_market.pipeline.calibrate import (
        SimulatedMomentCalibrator,
        MomentTarget,
    )
    from airline_market.vendor.ride_response.cem import CEMConfig
    from airline_market.agents.fixed import FixedCoefficientController

    fit = SimulatedMomentCalibrator(
        cx_example("C"),
        {"pricing.0.base.F_L1": (100.0, 200.0), "pricing.1.base.R_L1": (80.0, 120.0)},
        [MomentTarget("profit", 160.0)],
        {"F": FixedCoefficientController((0.0,))},
        seeds=[1, 2],
        cem=CEMConfig(population=12, iterations=4, seed=9),
    )
    result = fit.fit()
    assert abs(result.params["pricing.0.base.F_L1"] - 160) < 16
    assert any("pricing.1.base.R_L1" in w for w in result.warnings)
    ensemble = fit.ensemble(3)
    assert len(ensemble) == 3 and sum(w for s, w in ensemble) == pytest.approx(1.0)


def test_noaa_hourly_index_raw_fields(tmp_path):
    from datetime import datetime, timezone
    from dataclasses import replace

    p = tmp_path / "hourly.csv"
    p.write_text(
        'STATION,DATE,TMP,AA1,WND,VIS\nUS1,2024-08-14T12:00:00,"+0020,1","01,0010,1,1","180,1,N,0020,1","10000,1"\n'
    )
    s = scope()
    s = replace(
        s,
        sources={**s.sources, "noaa": str(p)},
        params={**s.params, "weather_stations": {"SEA": "US1"}},
    )
    index = noaa.hourly_indexes(s, out=False)["SEA"]
    record = index.nearest(datetime(2024, 8, 14, 12, 30, tzinfo=timezone.utc))
    assert (
        record.weather == "rain"
        and record.temperature_c == 2.0
        and record.precipitation_mm == 1.0
        and record.wind_speed_mps == 2.0
    )


def test_official_archive_headers_and_assumption_labels():
    import csv
    from airline_market.pipeline.fieldmap import MAPS

    required = {
        "t100": [
            "year",
            "month",
            "carrier",
            "origin",
            "dest",
            "departures_scheduled",
            "departures_performed",
            "seats",
            "passengers",
            "distance",
        ],
        "db1b": [
            "itin_id",
            "mkt_id",
            "origin",
            "dest",
            "ticket_carrier",
            "operating_carrier",
            "market_fare",
            "market_coupons",
            "distance",
            "nonstop_miles",
        ],
        "ontime": [
            "flight_date",
            "carrier",
            "flight_number",
            "origin",
            "dest",
            "crs_dep_time",
            "crs_arr_time",
            "cancelled",
        ],
    }
    for name, names in required.items():
        with open("tests/fixtures/" + name + "/official_header.csv") as handle:
            headers = next(csv.reader(handle))
        mapping = FieldMap(MAPS[name]).columns(headers, names)
        assert set(names) <= set(mapping.values())
    spec = compile_scope(scope())
    assert spec.metadata["identification"]["market_value"]["value"] == 0.0
    assert (
        spec.metadata["identification"]["carrier_preferences"]["status"]
        == "assumed_initialization"
    )
    assert "out-of-scope" in spec.metadata["identification"]["outside_option"]["scope"]
