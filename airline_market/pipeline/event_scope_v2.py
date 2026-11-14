"""A.15/V10: pre-event marketed service scope, with no post-event fare selection."""

import zipfile
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv

FIELDS = (
    "FL_DATE",
    "MKT_UNIQUE_CARRIER",
    "OP_UNIQUE_CARRIER",
    "ORIGIN",
    "DEST",
    "CANCELLED",
    "DIVERTED",
)


def marketing_calendar_date(value):
    """Parse the two registered FL_DATE serializations without changing calendar scope."""
    value = value.strip()
    if "/" in value:
        return datetime.strptime(value, "%m/%d/%Y %I:%M:%S %p").date(), "US_timestamp"
    return date.fromisoformat(value), "ISO_date"


def spirit_candidate_scope(sources):
    """Select unordered airport pairs with pre-event NK and incumbent service.

    sources is (path,(year,month)) for each of January–April 2026. A completed
    marketed-service proxy requires cancellation=0 and diversion=0. Operating
    codes are retained in route counts, including regional partners. This is a
    candidate O&D scope, not the final dated network or incumbent selection.
    """
    periods = [tuple(period) for _, period in sources]
    if sorted(periods) != [(2026, m) for m in range(1, 5)]:
        raise ValueError(
            "scope selection requires each pre-event Jan-Apr2026 month exactly once"
        )
    routes = defaultdict(Counter)
    files = []
    for path, period in sources:
        counts = Counter(
            {"rows_read": 0, "missing_date": 0, "missing_route_or_carrier": 0}
        )
        serializations = Counter()
        path = Path(path)
        with zipfile.ZipFile(path) as archive:
            members = [
                m
                for m in archive.infolist()
                if m.filename.lower().endswith(".csv")
                and m.filename.lower().split("/")[-1] != "documentation.csv"
            ]
            if len(members) != 1:
                raise ValueError("one marketing CSV required")
            info = members[0]
            with archive.open(info) as stream:
                reader = pacsv.open_csv(
                    stream,
                    read_options=pacsv.ReadOptions(
                        block_size=1024**2, use_threads=False
                    ),
                    convert_options=pacsv.ConvertOptions(
                        include_columns=FIELDS,
                        column_types={x: pa.string() for x in FIELDS},
                    ),
                )
                for batch in reader:
                    for row in batch.to_pylist():
                        counts["rows_read"] += 1
                        date = row["FL_DATE"]
                        if not date:
                            counts["missing_date"] += 1
                            continue
                        parsed, serialization = marketing_calendar_date(date)
                        serializations[serialization] += 1
                        if (parsed.year, parsed.month) != tuple(period):
                            raise ValueError(
                                "post-event or wrong-period row cannot select the pre-event scope"
                            )
                        origin = (row["ORIGIN"] or "").strip()
                        dest = (row["DEST"] or "").strip()
                        carrier = (row["MKT_UNIQUE_CARRIER"] or "").strip()
                        operator = (row["OP_UNIQUE_CARRIER"] or "").strip()
                        if not all((origin, dest, carrier, operator)):
                            counts["missing_route_or_carrier"] += 1
                            continue
                        pair = tuple(sorted((origin, dest)))
                        key = (*pair, carrier, operator)
                        counter = routes[key]
                        counter["scheduled"] += 1
                        try:
                            cancel = float(row["CANCELLED"])
                            divert = float(row["DIVERTED"])
                        except (TypeError, ValueError):
                            counter["unknown_status"] += 1
                            continue
                        if cancel not in (0.0, 1.0) or divert not in (0.0, 1.0):
                            counter["unknown_status"] += 1
                            continue
                        counter["cancelled"] += int(cancel == 1)
                        counter["diverted"] += int(divert == 1)
                        counter["completed_proxy"] += int(cancel == 0 and divert == 0)
        files.append(
            {
                "path": str(path),
                "period": list(period),
                "member": info.filename,
                "member_crc_verified": True,
                "date_serializations": dict(serializations),
                **counts,
            }
        )
    by_pair = defaultdict(set)
    for (a, b, carrier, _), counter in routes.items():
        if counter["completed_proxy"] > 0:
            by_pair[a, b].add(carrier)
    pairs = sorted(
        pair
        for pair, carriers in by_pair.items()
        if "NK" in carriers and len(carriers - {"NK"}) > 0
    )
    markets = sorted({direction for a, b in pairs for direction in [(a, b), (b, a)]})
    return {
        "plan_id": "A.15/V10",
        "status": "candidate_scope_complete",
        "periods": periods,
        "airport_pairs": pairs,
        "directed_markets": markets,
        "files": files,
        "route_counts": [
            {
                "airports": [a, b],
                "marketing_carrier": carrier,
                "operating_carrier": operator,
                **dict(counter),
            }
            for (a, b, carrier, operator), counter in sorted(routes.items())
        ],
        "full_V10_pass": False,
        "qualification": "Observed completed marketed-service proxy only; all carriers retained. Final dated network, strategic-incumbent selection and T100 reconciliation remain open.",
    }
