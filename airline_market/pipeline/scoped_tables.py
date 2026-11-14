"""Scope 2022 schedules, capacity and demand while preserving source accounting."""

from pathlib import Path
from collections import defaultdict
import numpy as np

from .regional import (
    LEG_MONTH,
    canonical_flights,
    allocate_capacity,
    departure_comparison,
)
from .scoped_io import batches, ParquetSink, atomic_json


FLIGHT_COLUMNS = [
    "YEAR",
    "MONTH",
    "FL_DATE",
    "MKT_UNIQUE_CARRIER",
    "MKT_CARRIER_AIRLINE_ID",
    "OP_UNIQUE_CARRIER",
    "OP_CARRIER_AIRLINE_ID",
    "MKT_CARRIER_FL_NUM",
    "OP_CARRIER_FL_NUM",
    "TAIL_NUM",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "CANCELLED",
    "CANCELLATION_CODE",
    "DIVERTED",
    "DUP",
    "DISTANCE",
    "CRS_ELAPSED_TIME",
    "DEP_DELAY",
    "ARR_DELAY",
    "WEATHER_DELAY",
]
FLIGHT_NAMES = [
    "year",
    "month",
    "flight_date",
    "marketing_carrier",
    "marketing_carrier_id",
    "operating_carrier",
    "operating_carrier_id",
    "marketing_flight_number",
    "operating_flight_number",
    "tail_number",
    "origin",
    "dest",
    "crs_dep_time",
    "crs_arr_time",
    "cancelled",
    "cancellation_code",
    "diverted",
    "source_duplicate_flag",
    "distance",
    "scheduled_elapsed_minutes",
    "departure_delay_minutes",
    "arrival_delay_minutes",
    "weather_delay_minutes",
]
MARKET_COLUMNS = [
    "ItinID",
    "MktID",
    "Year",
    "Quarter",
    "Origin",
    "Dest",
    "AirportGroup",
    "MktCoupons",
    "TkCarrierGroup",
    "OpCarrierGroup",
    "TkCarrier",
    "OpCarrier",
    "RPCarrier",
    "Passengers",
    "MktFare",
    "BulkFare",
    "MktDistance",
    "NonStopMiles",
]


def normalize_flights(frame):
    """Keep both network and operator identities and canonical local dates."""
    import pandas as pd

    f = (
        frame[FLIGHT_COLUMNS]
        .rename(columns=dict(zip(FLIGHT_COLUMNS, FLIGHT_NAMES)))
        .copy()
    )
    f["flight_date"] = pd.to_datetime(
        f.flight_date, format="mixed", errors="raise"
    ).dt.strftime("%Y-%m-%d")
    for name in ("year", "month"):
        f[name] = pd.to_numeric(f[name], errors="raise").astype(int)
    for name in ("cancelled", "diverted"):
        f[name] = pd.to_numeric(f[name], errors="raise").astype(int)
    for name in (
        "distance",
        "scheduled_elapsed_minutes",
        "departure_delay_minutes",
        "arrival_delay_minutes",
        "weather_delay_minutes",
    ):
        f[name] = pd.to_numeric(f[name], errors="coerce").astype(float)
    for name in ("crs_dep_time", "crs_arr_time"):
        f[name] = f[name].str.zfill(4)
    dates = pd.to_datetime(f.flight_date)
    if not ((dates.dt.year == f.year) & (dates.dt.month == f.month)).all():
        raise ValueError("flight date disagrees with reported year/month")
    if not f.cancelled.isin([0, 1]).all():
        raise ValueError("cancelled indicator is not binary")
    return f


def normalize_markets(frame, quarter, hub, carriers):
    """Label representable paths but retain competitors and out-of-model alternatives."""
    import pandas as pd

    f = frame.rename(
        columns={
            "ItinID": "itin_id",
            "MktID": "market_record_id",
            "Year": "year",
            "Quarter": "quarter",
            "Origin": "origin",
            "Dest": "dest",
            "AirportGroup": "airport_sequence",
            "MktCoupons": "n_coupons",
            "TkCarrierGroup": "ticket_carrier_sequence",
            "OpCarrierGroup": "operating_carrier_sequence",
            "TkCarrier": "ticket_carrier",
            "OpCarrier": "aggregate_operating_carrier",
            "RPCarrier": "reporting_carrier",
            "Passengers": "passengers_sample",
            "MktFare": "market_fare",
            "BulkFare": "bulk_fare_flag",
            "MktDistance": "distance",
            "NonStopMiles": "nonstop_miles",
        }
    ).copy()
    for name in ("year", "quarter", "n_coupons"):
        f[name] = pd.to_numeric(f[name], errors="raise").astype(int)
    if not ((f.year == 2022) & (f.quarter == quarter)).all():
        raise ValueError("DB1B quarter does not match the declared input split")
    for name in ("passengers_sample", "market_fare", "distance", "nonstop_miles"):
        f[name] = pd.to_numeric(f[name], errors="coerce").astype(float)
    if f.passengers_sample.isna().any() or (f.passengers_sample < 0).any():
        raise ValueError("invalid sampled passenger count")
    networks, represented, hub_connections, integrity = [], [], [], []
    for row in f.itertuples(index=False):
        path = row.airport_sequence.split(":")
        ticket = row.ticket_carrier_sequence.split(":")
        operators = row.operating_carrier_sequence.split(":")
        valid = (
            len(path) == row.n_coupons + 1
            and path[0] == row.origin
            and path[-1] == row.dest
            and len(ticket) == row.n_coupons
            and len(operators) == row.n_coupons
            and all(ticket)
            and all(operators)
        )
        network = ticket[0] if ticket and len(set(ticket)) == 1 else "MIXED"
        connecting = row.n_coupons == 2 and len(path) == 3 and path[1] == hub
        local = row.n_coupons == 1 and hub in (row.origin, row.dest)
        networks.append(network)
        represented.append(valid and network in carriers and (local or connecting))
        hub_connections.append(valid and connecting)
        integrity.append(valid)
    f["sequence_valid"] = integrity
    f["path_network"] = networks
    f["model_path_eligible"] = represented
    f["hub_connecting"] = hub_connections
    f["passengers_expanded"] = 10.0 * f.passengers_sample
    f["market"] = f.origin + "-" + f.dest
    f["split"] = "calibration" if quarter == 2 else "validation"
    f["fare_eligible"] = (
        np.isfinite(f.market_fare)
        & (f.market_fare > 0)
        & f.bulk_fare_flag.isin(["0", "0.00", "N", "False"])
    )
    return f


def process_flights(paths, scope, out, progress):
    """Scope by airport geography before applying any marketing-network selection."""
    import pandas as pd

    frames, raw = [], 0
    national_networks = set()
    airport_set = set(scope["airport_pool"])
    for path in paths:
        for batch in batches(path, FLIGHT_COLUMNS):
            raw += batch.num_rows
            f = batch.to_pandas()
            national_networks.update(f.MKT_UNIQUE_CARRIER.unique())
            keep = (
                f.ORIGIN.isin(airport_set)
                & f.DEST.isin(airport_set)
                & (
                    (f.ORIGIN == scope["hub_candidate"])
                    | (f.DEST == scope["hub_candidate"])
                )
            )
            if keep.any():
                normalized = normalize_flights(f.loc[keep])
                if not (
                    (normalized.year == 2022) & normalized.month.between(4, 9)
                ).all():
                    raise ValueError("flight source outside declared 2022 Q2/Q3 window")
                frames.append(normalized)
            progress(stage="flights", rows_scanned=raw)
    combined = pd.concat(frames, ignore_index=True)
    flights, checks = canonical_flights(combined)
    checks["national_source_rows"] = raw
    checks["national_marketing_network_codes"] = sorted(national_networks)
    checks["all_marketing_networks_retained"] = sorted(
        flights.marketing_carrier.unique()
    )
    flights["split"] = np.where(flights.month <= 6, "calibration", "validation")
    flights.to_parquet(out / "flights.parquet", index=False)
    carrier_stats = (
        flights.groupby(["marketing_carrier", "operating_carrier"])
        .agg(
            scheduled_flights=("schedule_weight", "sum"),
            performed_flights=("performed_weight", "sum"),
            network_records=("physical_flight_id", "size"),
        )
        .reset_index()
    )
    carrier_stats.to_csv(out / "marketing_operator_counts.csv", index=False)
    modeled = flights[flights.marketing_carrier.isin(scope["carriers"])]
    checks["modeled_network_scheduled_flights"] = float(modeled.schedule_weight.sum())
    checks["modeled_network_partner_flights"] = float(
        modeled.loc[modeled.partner_operated, "schedule_weight"].sum()
    )
    checks["operating_only_filter_lost_flights"] = float(
        modeled.loc[
            ~modeled.operating_carrier.isin(scope["carriers"]), "schedule_weight"
        ].sum()
    )
    return flights, checks


def process_capacity(path, flights, scope, out, progress):
    """Keep scheduled passenger capacity and expose source/allocation reconciliation."""
    import pandas as pd

    columns = [
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
    frames, raw = [], 0
    for batch in batches(path, columns):
        raw += batch.num_rows
        f = batch.to_pandas()
        keep = (
            f.MONTH.astype(int).between(4, 9)
            & f.ORIGIN.isin(scope["airport_pool"])
            & f.DEST.isin(scope["airport_pool"])
            & (
                (f.ORIGIN == scope["hub_candidate"])
                | (f.DEST == scope["hub_candidate"])
            )
            & (f.CLASS == "F")
        )
        f = f.loc[keep].rename(
            columns={
                "YEAR": "year",
                "MONTH": "month",
                "UNIQUE_CARRIER": "operating_carrier",
                "AIRLINE_ID": "operating_carrier_id",
                "ORIGIN": "origin",
                "DEST": "dest",
                "SEATS": "seats",
                "PASSENGERS": "passengers",
                "DEPARTURES_SCHEDULED": "departures_scheduled",
                "DEPARTURES_PERFORMED": "departures_performed",
                "DISTANCE": "distance",
            }
        )
        if len(f):
            for name in ("year", "month"):
                f[name] = f[name].astype(int)
            for name in (
                "seats",
                "passengers",
                "departures_scheduled",
                "departures_performed",
                "distance",
            ):
                f[name] = pd.to_numeric(f[name], errors="raise")
            frames.append(f)
        progress(stage="capacity", rows_scanned=raw)
    data = pd.concat(frames, ignore_index=True)
    grouped = data.groupby(LEG_MONTH + ["operating_carrier"], as_index=False).agg(
        seats=("seats", "sum"),
        passengers=("passengers", "sum"),
        departures_scheduled=("departures_scheduled", "sum"),
        departures_performed=("departures_performed", "sum"),
        distance=("distance", "mean"),
    )
    grouped.to_parquet(out / "operator_leg_month.parquet", index=False)
    allocated, checks = allocate_capacity(grouped, flights)
    allocated.to_parquet(out / "marketing_leg_month.parquet", index=False)
    departure_comparison(grouped, flights).to_csv(
        out / "t100_ontime_departure_comparison.csv", index=False
    )
    checks["national_source_rows"] = raw
    return allocated, checks


def process_markets(paths, scope, out, progress):
    """Preserve quarter and all-carrier denominators; scope selection uses Q2 only."""
    import pandas as pd
    import pyarrow.parquet as pq

    outputs, checks, model_carriers = {}, {}, defaultdict(set)
    for quarter, path in paths.items():
        sink = ParquetSink(out / f"db1b_pool_q{quarter}.parquet")
        raw = kept = 0
        sampled = expanded = eligible_expanded = malformed = 0.0
        connecting_by_network = defaultdict(float)
        for batch in batches(path, MARKET_COLUMNS):
            raw += batch.num_rows
            f = batch.to_pandas()
            f = f[
                f.Origin.isin(scope["airport_pool"])
                & f.Dest.isin(scope["airport_pool"])
            ]
            if len(f):
                f = normalize_markets(
                    f, quarter, scope["hub_candidate"], scope["carriers"]
                )
                sink.write(f)
                kept += len(f)
                expanded += float(f.passengers_expanded.sum())
                sampled += float(f.passengers_sample.sum())
                for network, value in (
                    f.loc[f.hub_connecting]
                    .groupby("path_network")
                    .passengers_expanded.sum()
                    .items()
                ):
                    connecting_by_network[network] += float(value)
                eligible_expanded += float(
                    f.loc[f.model_path_eligible, "passengers_expanded"].sum()
                )
                malformed += int((~f.sequence_valid).sum())
                if quarter == 2:
                    for market, network in f.loc[
                        f.model_path_eligible, ["market", "path_network"]
                    ].itertuples(index=False, name=None):
                        model_carriers[market].add(network)
            progress(stage=f"db1b_q{quarter}", rows_scanned=raw, pool_rows=kept)
        outputs[quarter] = sink.close()
        checks[str(quarter)] = {
            "national_rows": raw,
            "pool_rows": kept,
            "pool_expanded_passengers": expanded,
            "represented_expanded_passengers": eligible_expanded,
            "malformed_pool_sequences": int(malformed),
            "sample_expansion": 10,
            "flow_scaling_error": expanded - 10 * sampled,
            "all_pool_hub_connecting_passengers_by_network": dict(
                connecting_by_network
            ),
        }
    markets = sorted(model_carriers)
    market_frame = pd.DataFrame(
        [
            {
                "market": market,
                "origin": market.split("-")[0],
                "dest": market.split("-")[1],
                "carriers": "|".join(sorted(model_carriers[market])),
                "competitive": len(model_carriers[market]) >= 2,
            }
            for market in markets
        ]
    )
    market_frame.to_csv(out / "model_markets.csv", index=False)
    summaries = []
    for quarter in paths:
        sink = ParquetSink(out / f"db1b_scoped_q{quarter}.parquet")
        for batch in pq.ParquetFile(out / f"db1b_pool_q{quarter}.parquet").iter_batches(
            batch_size=50000
        ):
            f = batch.to_pandas()
            f = f[f.market.isin(markets)]
            sink.write(f)
            g = (
                f.groupby(
                    ["market", "path_network", "model_path_eligible", "hub_connecting"],
                    dropna=False,
                )
                .agg(
                    passengers=("passengers_expanded", "sum"), rows=("itin_id", "size")
                )
                .reset_index()
            )
            g["quarter"] = quarter
            summaries.append(g)
        outputs[quarter]["scoped"] = sink.close()
    summary = (
        pd.concat(summaries, ignore_index=True)
        .groupby(
            [
                "quarter",
                "market",
                "path_network",
                "model_path_eligible",
                "hub_connecting",
            ],
            as_index=False,
        )
        .agg(passengers=("passengers", "sum"), rows=("rows", "sum"))
    )
    summary.to_csv(out / "market_passenger_summary.csv", index=False)
    checks["model_markets"] = len(markets)
    checks["competitive_q2_markets"] = int(market_frame.competitive.sum())
    checks["scope_selection"] = (
        "Only Q2 represented UA/AA local/one-connection paths; Q3 cannot add selected markets. All-carrier alternatives retained inside selected O&D markets."
    )
    return markets, checks


def process_weather(path, out, station="USW00094846"):
    """Retain the observed daily ORD window and declared source units."""
    import pandas as pd

    f = pd.read_csv(path, dtype={"STATION": str})
    f = f[(f.STATION == station) & f.DATE.between("2022-04-01", "2022-09-30")].copy()
    if f.DATE.duplicated().any():
        raise ValueError("duplicate daily weather station/date")
    f.sort_values("DATE").to_parquet(out / "weather_daily.parquet", index=False)
    return {
        "station": station,
        "rows": len(f),
        "date_min": f.DATE.min(),
        "date_max": f.DATE.max(),
        "expected_days": 183,
        "missing_days": 183 - len(f),
        "units": "NOAA API metric; raw file metadata retained",
    }
