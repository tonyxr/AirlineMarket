"""Bounded quote-panel processing and F8 coverage, using Q2-selected O&D scope."""

from collections import defaultdict
import numpy as np

from .scoped_io import batches, ParquetSink


QUOTE_COLUMNS = [
    "legId",
    "searchDate",
    "flightDate",
    "startingAirport",
    "destinationAirport",
    "fareBasisCode",
    "totalFare",
    "baseFare",
    "seatsRemaining",
    "isNonStop",
    "isBasicEconomy",
    "isRefundable",
    "totalTravelDistance",
    "travelDuration",
    "segmentsDepartureTimeEpochSeconds",
    "segmentsArrivalTimeEpochSeconds",
    "segmentsDepartureTimeRaw",
    "segmentsDepartureAirportCode",
    "segmentsArrivalAirportCode",
    "segmentsAirlineCode",
    "segmentsCabinCode",
    "segmentsDurationInSeconds",
    "segmentsDistance",
]
BIN_LABELS = ["0-2", "3-6", "7-13", "14-20", "21-29", "30-44", "45-59", "60+"]


def attribution_lookup(flights, known_network_codes=()):
    """Index observed marketing labels without assigning ambiguous regional offers."""
    exact, daily = defaultdict(set), defaultdict(set)
    network_codes = set(flights.marketing_carrier) | set(known_network_codes)
    for row in flights.itertuples():
        key = (row.flight_date, row.operating_carrier, row.origin, row.dest)
        daily[key].add(row.marketing_carrier)
        exact[(*key, row.crs_dep_time)].add(row.marketing_carrier)
    return exact, daily, network_codes


def classify_quote(
    codes,
    origins,
    destinations,
    departures,
    flight_date,
    hub,
    carriers,
    lookup,
    endpoints=None,
):
    """Resolve only reported or uniquely observed networks for one/two-leg offers.

    Offer flight numbers are unavailable. A regional label is assigned using
    an exact local departure-time key when unique, or a unique route/day
    network. Ambiguity remains explicit; seat-allocation shares are never used
    as passenger-level carrier assignments.
    """
    reported = codes.split("||")
    origins, destinations = origins.split("||"), destinations.split("||")
    stamps = departures.split("||")
    same_reported = reported[0] if len(set(reported)) == 1 else "MIXED"
    if (
        len(reported) not in (1, 2)
        or len(origins) != len(reported)
        or len(destinations) != len(reported)
    ):
        return same_reported, "outside_path_length", False, False, ()
    if any(destinations[k] != origins[k + 1] for k in range(len(reported) - 1)):
        return "UNMAPPED", "broken_path", False, False, ()
    if endpoints is not None and (origins[0], destinations[-1]) != endpoints:
        return "UNMAPPED", "endpoint_mismatch", False, False, ()
    exact, daily, known_networks = lookup
    resolved, methods = [], []
    for k, code in enumerate(reported):
        if code in known_networks or code in carriers:
            resolved.append(code)
            methods.append("reported_network")
            continue
        stamp = stamps[k] if k < len(stamps) else ""
        day = stamp[:10] if len(stamp) >= 16 else flight_date
        clock = stamp[11:13] + stamp[14:16] if len(stamp) >= 16 else ""
        key = (day, code, origins[k], destinations[k])
        candidates = exact.get((*key, clock), set())
        method = "unique_operator_route_time"
        if len(candidates) != 1:
            candidates = daily.get(key, set())
            method = "unique_operator_route_day"
        if len(candidates) != 1:
            return "UNMAPPED", "ambiguous_or_unmapped_operator", False, False, ()
        resolved.append(next(iter(candidates)))
        methods.append(method)
    network = resolved[0] if len(set(resolved)) == 1 else "MIXED"
    local = len(reported) == 1 and hub in (origins[0], destinations[-1])
    connection = len(reported) == 2 and destinations[0] == hub
    same_day = len(stamps) == len(reported) and all(
        stamp[:10] == flight_date for stamp in stamps
    )
    eligible = network in carriers and (local or connection) and same_day
    method = (
        "reported_network"
        if all(m == "reported_network" for m in methods)
        else "+".join(sorted(set(methods)))
    )
    legs = (
        tuple(f"{network}:{o}-{d}" for o, d in zip(origins, destinations))
        if eligible
        else ()
    )
    return network, method, eligible, not same_day, legs


def normalize_quotes(frame, scope, lookup, booking_window_days=60):
    """Keep nominal total/base fares separately and flag collection truncation."""
    import pandas as pd

    f = frame.copy()
    flight_dates = pd.to_datetime(f.flightDate, format="%Y-%m-%d", errors="raise")
    searches = pd.to_datetime(f.searchDate, format="%Y-%m-%d", errors="raise")
    f["days_before_departure"] = (flight_dates - searches).dt.days.astype(int)
    if (f.days_before_departure < 0).any():
        raise ValueError("quote search occurs after departure")
    f["quarter"] = flight_dates.dt.quarter.astype(int)
    f["split"] = np.where(f.quarter == 2, "calibration", "validation")
    f["market"] = f.startingAirport + "-" + f.destinationAirport
    f["days_bin"] = pd.cut(
        f.days_before_departure,
        [-1, 2, 6, 13, 20, 29, 44, 59, np.inf],
        labels=BIN_LABELS,
    ).astype(str)
    for name in ("totalFare", "baseFare", "seatsRemaining", "totalTravelDistance"):
        f[name] = pd.to_numeric(f[name], errors="coerce").astype(float)
    f["seats_reported"] = np.isfinite(f.seatsRemaining)
    f["seats_le_3"] = f.seats_reported & (f.seatsRemaining <= 3)
    start = pd.Timestamp(scope["quote_search_coverage"][0])
    f["left_truncated_by_collection_start"] = (
        flight_dates - pd.Timedelta(days=booking_window_days) < start
    )
    labels = [
        classify_quote(
            row.segmentsAirlineCode,
            row.segmentsDepartureAirportCode,
            row.segmentsArrivalAirportCode,
            row.segmentsDepartureTimeRaw,
            row.flightDate,
            scope["hub_candidate"],
            set(scope["carriers"]),
            lookup,
            (row.startingAirport, row.destinationAirport),
        )
        for row in f.itertuples(index=False)
    ]
    f["path_network"] = [row[0] for row in labels]
    f["network_attribution"] = [row[1] for row in labels]
    f["model_path_eligible"] = [row[2] for row in labels]
    f["cross_date_path"] = [row[3] for row in labels]
    return f, [row[4] for row in labels]


def process_quotes(
    path,
    markets,
    scope,
    flights,
    out,
    progress,
    booking_window_days=60,
    known_network_codes=(),
):
    """Write scoped Q2/Q3 panels and exact offer counts without using Q3 for selection."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc

    lookup = attribution_lookup(flights, known_network_codes)
    sinks = {q: ParquetSink(out / f"quotes_q{q}.parquet") for q in (2, 3)}
    counters, methods = (
        defaultdict(lambda: np.zeros(5, dtype=np.int64)),
        defaultdict(int),
    )
    networks, legs = defaultdict(set), defaultdict(set)
    raw = kept = 0
    for batch in batches(path, QUOTE_COLUMNS):
        raw += batch.num_rows
        table = pa.Table.from_batches([batch])
        market = pc.binary_join_element_wise(
            table["startingAirport"], table["destinationAirport"], "-"
        )
        keep = pc.and_(
            pc.is_in(market, value_set=pa.array(markets)),
            pc.and_(
                pc.greater_equal(table["flightDate"], "2022-04-01"),
                pc.less_equal(table["flightDate"], "2022-09-30"),
            ),
        )
        table = table.filter(keep)
        if table.num_rows:
            f, path_legs = normalize_quotes(
                table.to_pandas(), scope, lookup, booking_window_days
            )
            for q, sink in sinks.items():
                sink.write(f[f.quarter == q])
            kept += len(f)
            grouped = f.groupby(
                [
                    "quarter",
                    "market",
                    "days_bin",
                    "path_network",
                    "model_path_eligible",
                ],
                dropna=False,
            ).agg(
                quotes=("legId", "size"),
                seats_reported=("seats_reported", "sum"),
                seats_le_3=("seats_le_3", "sum"),
                left_truncated=("left_truncated_by_collection_start", "sum"),
                cross_date=("cross_date_path", "sum"),
            )
            for key, values in grouped.iterrows():
                counters[key] += values.to_numpy(dtype=np.int64)
            for method, count in f.network_attribution.value_counts().items():
                methods[method] += int(count)
            for row, path_ids in zip(f.itertuples(index=False), path_legs):
                if row.quarter == 2 and row.model_path_eligible:
                    networks[row.market].add(row.path_network)
                    for leg in path_ids:
                        legs[leg].add(row.market)
        progress(stage="quotes", rows_scanned=raw, scoped_rows=kept)
    artifacts = {str(q): sink.close() for q, sink in sinks.items()}
    rows = [
        {
            **dict(
                zip(
                    [
                        "quarter",
                        "market",
                        "days_bin",
                        "path_network",
                        "model_path_eligible",
                    ],
                    key,
                )
            ),
            **dict(
                zip(
                    [
                        "quotes",
                        "seats_reported",
                        "seats_le_3",
                        "left_truncated",
                        "cross_date",
                    ],
                    map(int, values),
                )
            ),
        }
        for key, values in sorted(counters.items())
    ]
    counts = pd.DataFrame(rows)
    counts.to_csv(out / "quote_coverage.csv", index=False)
    competitive = {
        market
        for market, carriers in networks.items()
        if set(scope["carriers"]) <= carriers
    }
    leg_rows = [
        {
            "leg": leg,
            "q2_eligible_markets": len(ms),
            "q2_competitive_markets": len(ms & competitive),
        }
        for leg, ms in sorted(legs.items())
    ]
    pd.DataFrame(leg_rows).to_csv(out / "shared_leg_coverage.csv", index=False)
    q2 = counts[counts.quarter == 2].groupby("market").quotes.sum()
    qualifying = [market for market in competitive if q2.get(market, 0) >= 200]
    rival = scope["carriers"][1]
    rival_max = max(
        [
            r["q2_competitive_markets"]
            for r in leg_rows
            if r["leg"].startswith(rival + ":")
        ]
        or [0]
    )
    return {
        "national_source_rows": raw,
        "scoped_rows": kept,
        "artifacts": artifacts,
        "q2_competitive_markets": len(competitive),
        "q2_competitive_markets_at_least_200_quotes": len(qualifying),
        "max_rival_leg_competitive_markets": rival_max,
        "network_attribution_counts": dict(methods),
        "quote_counts_are_offers_not_purchases": True,
        "booking_window_days_for_truncation_flag": booking_window_days,
        "fare_basis_note": "Raw totalFare and baseFare both retained. Tax/fee comparability with DB1B must be resolved before cross-source fare calibration.",
        "carrier_attribution_note": "Reported network labels retained. Regional codes inferred only when observed route/time or route/day marketing is unique; ambiguity remains UNMAPPED.",
    }
