"""A.15/V10: full outer operator-route-month comparison, with explicit nulls.

All T100 classes retained; F is primary. Performed excludes cancellations but
keeps diversions as a separate diagnostic. No seat rescaling or allocation.
"""

import hashlib
import json
import sqlite3
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from airline_market.pipeline.scoped_io import batches

FIELDS = (
    "YEAR",
    "MONTH",
    "AIRLINE_ID",
    "UNIQUE_CARRIER",
    "ORIGIN",
    "DEST",
    "CLASS",
    "AIRCRAFT_TYPE",
    "AIRCRAFT_CONFIG",
    "DEPARTURES_SCHEDULED",
    "DEPARTURES_PERFORMED",
    "SEATS",
    "PASSENGERS",
)
MEASURES = ("DEPARTURES_SCHEDULED", "DEPARTURES_PERFORMED", "SEATS", "PASSENGERS")


def integer(value):
    try:
        x = Decimal(str(value))
        if not x.is_finite() or x < 0 or x != x.to_integral_value():
            return None
        return int(x)
    except (ValueError, TypeError, ArithmeticError):
        return None


def reported_key(row):
    """Source-calendar operating identity; UTC arrival errors do not erase counts."""
    opid = integer(row["OP_CARRIER_AIRLINE_ID"])
    flight = integer(row["OP_CARRIER_FL_NUM"])
    dep = integer(row["CRS_DEP_TIME"])
    if (
        opid is None
        or opid == 0
        or flight is None
        or flight == 0
        or not row["service_date"]
        or dep is None
    ):
        return None
    if dep != 2400 and (dep // 100 > 23 or dep % 100 > 59):
        return None
    return hashlib.sha256(
        json.dumps(
            [row["service_date"], opid, flight, row["origin"], row["dest"], dep],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def source_counts(path, out, progress):
    db = sqlite3.connect(out / "reported_counts.sqlite")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-8192")
    db.execute(
        "CREATE TABLE rows(sid TEXT PRIMARY KEY,k TEXT,y INTEGER,m INTEGER,opid INTEGER,op TEXT,o TEXT,d TEXT,mkt TEXT,c INTEGER,v INTEGER,utc_id TEXT)"
    )
    cols = [
        "source_row_id",
        "source_year",
        "source_month",
        "OP_CARRIER_AIRLINE_ID",
        "OP_CARRIER_FL_NUM",
        "CRS_DEP_TIME",
        "service_date",
        "origin",
        "dest",
        "operating_carrier",
        "marketing_carrier",
        "cancelled_flag",
        "diverted_flag",
        "physical_id",
    ]
    count = 0
    for b in pq.ParquetFile(path).iter_batches(batch_size=8192, columns=cols):
        rows = b.to_pylist()
        db.executemany(
            "INSERT INTO rows VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    r["source_row_id"],
                    reported_key(r),
                    r["source_year"],
                    r["source_month"],
                    integer(r["OP_CARRIER_AIRLINE_ID"]),
                    r["operating_carrier"],
                    r["origin"],
                    r["dest"],
                    r["marketing_carrier"],
                    r["cancelled_flag"],
                    r["diverted_flag"],
                    r["physical_id"],
                )
                for r in rows
            ],
        )
        db.commit()
        count += len(rows)
        progress(stage="prepared_source_counts", source_rows=count)
    db.execute("CREATE INDEX report_keys ON rows(k)")
    db.commit()
    result = {}

    def group(key):
        return result.setdefault(
            key,
            {
                "source_rows": 0,
                "reported_schedule_known": 0,
                "non_cancelled_known": 0,
                "completed_proxy_known": 0,
                "diverted_known": 0,
                "cancelled_known": 0,
                "unresolved_identity_rows": 0,
                "cancellation_unknown_flights": 0,
                "status_conflict_flights": 0,
                "utc_unresolved_rows": 0,
                "shared_marketing_flights": 0,
                "marketing_carriers": set(),
                "operating_codes": set(),
            },
        )

    for y, m, opid, o, d, n, utc, ops, mkts in db.execute(
        "SELECT y,m,opid,o,d,COUNT(*),SUM(utc_id IS NULL),GROUP_CONCAT(DISTINCT op),GROUP_CONCAT(DISTINCT mkt) FROM rows GROUP BY y,m,opid,o,d"
    ):
        x = group((y, m, opid, o, d))
        x["source_rows"] = n
        x["utc_unresolved_rows"] = utc
        x["operating_codes"].update(v for v in (ops or "").split(",") if v)
        x["marketing_carriers"].update(v for v in (mkts or "").split(",") if v)
    for y, m, opid, o, d, n in db.execute(
        "SELECT y,m,opid,o,d,COUNT(*) FROM rows WHERE k IS NULL GROUP BY y,m,opid,o,d"
    ):
        group((y, m, opid, o, d))["unresolved_identity_rows"] = n
    for y, m, opid, o, d, cmin, cmax, vmin, vmax, cu, vu, nm in db.execute(
        "SELECT MIN(y),MIN(m),MIN(opid),MIN(o),MIN(d),MIN(c),MAX(c),MIN(v),MAX(v),SUM(c IS NULL),SUM(v IS NULL),COUNT(DISTINCT mkt) FROM rows WHERE k IS NOT NULL GROUP BY k"
    ):
        x = group((y, m, opid, o, d))
        x["reported_schedule_known"] += 1
        x["shared_marketing_flights"] += nm > 1
        conflict = cmin != cmax or vmin != vmax
        x["status_conflict_flights"] += conflict
        if cu or cmin != cmax:
            x["cancellation_unknown_flights"] += 1
            continue
        if cmin == 1:
            x["cancelled_known"] += 1
        else:
            x["non_cancelled_known"] += 1
            if not vu and vmin == vmax:
                x["diverted_known"] += vmin == 1
                x["completed_proxy_known"] += vmin == 0
    db.close()
    for x in result.values():
        x["marketing_carriers"] = sorted(x["marketing_carriers"])
        x["operating_codes"] = sorted(x["operating_codes"])
        x["schedule_count_complete"] = x["unresolved_identity_rows"] == 0
        x["non_cancelled_count_complete"] = (
            x["schedule_count_complete"] and x["cancellation_unknown_flights"] == 0
        )
    return result, count


def t100_projection(path, pairs, covered_months, out, progress):
    rows = []
    groups = {}
    read = 0
    for batch in batches(path, columns=FIELDS, block_size=1024**2):
        for r in batch.to_pylist():
            ordinal = read
            read += 1
            if (
                tuple(sorted(((r["ORIGIN"] or "").strip(), (r["DEST"] or "").strip())))
                not in pairs
            ):
                continue
            year, month, opid = (integer(r[k]) for k in ["YEAR", "MONTH", "AIRLINE_ID"])
            if year != 2026 or month not in covered_months:
                raise ValueError(
                    "T100 period differs from pinned local coverage record"
                )
            item = dict(r, source_row_ordinal=ordinal)
            rows.append(item)
            key = (
                year,
                month,
                opid,
                (r["ORIGIN"] or "").strip(),
                (r["DEST"] or "").strip(),
                (r["CLASS"] or "").strip(),
            )
            x = groups.setdefault(
                key,
                {
                    "rows": 0,
                    "operating_codes": set(),
                    "sums": {k: 0 for k in MEASURES},
                    "unknown": {k: 0 for k in MEASURES},
                    "aircraft_types": set(),
                },
            )
            x["rows"] += 1
            x["operating_codes"].add((r["UNIQUE_CARRIER"] or "").strip())
            x["aircraft_types"].add(r["AIRCRAFT_TYPE"])
            for k in MEASURES:
                value = integer(r[k])
                if value is None:
                    x["unknown"][k] += 1
                else:
                    x["sums"][k] += value
        progress(stage="T100_scoped_projection", rows_read=read, rows_kept=len(rows))
    schema = pa.schema(
        [(k, pa.string()) for k in FIELDS] + [("source_row_ordinal", pa.int64())]
    )
    pq.write_table(
        pa.Table.from_pylist(rows, schema=schema),
        out / "t100_scoped_rows.parquet",
        compression="zstd",
    )
    strata = []
    for key, x in sorted(
        groups.items(), key=lambda kv: tuple("" if v is None else str(v) for v in kv[0])
    ):
        strata.append(
            dict(
                zip(
                    [
                        "year",
                        "month",
                        "operating_airline_id",
                        "origin",
                        "dest",
                        "service_class",
                    ],
                    key,
                ),
                source_rows=x["rows"],
                operating_codes=sorted(x["operating_codes"]),
                aircraft_types=sorted(v for v in x["aircraft_types"] if v),
                **{k: None if x["unknown"][k] else x["sums"][k] for k in MEASURES},
                **{k + "_unknown_rows": x["unknown"][k] for k in MEASURES},
            )
        )
    return strata, read, len(rows)


def compare_counts(on_time, strata, covered_months):
    target = [x for x in strata if x["service_class"] == "F"]
    primary = {
        tuple(
            x[k] for k in ["year", "month", "operating_airline_id", "origin", "dest"]
        ): x
        for x in target
        if x["operating_airline_id"] not in [None, 0]
    }
    valid_on = {k: o for k, o in on_time.items() if k[2] not in [None, 0]}
    cases = [
        (k, primary.get(k), valid_on.get(k)) for k in primary.keys() | valid_on.keys()
    ]
    cases += [(k, None, o) for k, o in on_time.items() if k[2] in [None, 0]]
    cases += [
        (
            tuple(
                x[k]
                for k in ["year", "month", "operating_airline_id", "origin", "dest"]
            ),
            x,
            None,
        )
        for x in target
        if x["operating_airline_id"] in [None, 0]
    ]
    rows = []
    for key, t, o in sorted(
        cases, key=lambda row: tuple("" if v is None else str(v) for v in row[0])
    ):
        coverage = "available" if key[1] in covered_months else "missing_source_month"
        status = (
            "matched"
            if t is not None and o is not None
            else "on_time_only"
            if o is not None
            else "T100_only"
        )
        row = dict(
            zip(["year", "month", "operating_airline_id", "origin", "dest"], key),
            coverage=coverage,
            join_status=status,
            T100_class="F",
            operator_id_unresolved=key[2] in [None, 0],
            T100_source_rows=None if t is None else t["source_rows"],
            on_time_source_rows=None if o is None else o["source_rows"],
            marketing_carriers=[] if o is None else o["marketing_carriers"],
            on_time_operating_codes=[] if o is None else o["operating_codes"],
            T100_operating_codes=[] if t is None else t["operating_codes"],
        )
        for k in MEASURES:
            row["T100_" + k] = None if t is None else t[k]
        for k in [
            "reported_schedule_known",
            "non_cancelled_known",
            "completed_proxy_known",
            "diverted_known",
            "cancelled_known",
            "unresolved_identity_rows",
            "cancellation_unknown_flights",
            "utc_unresolved_rows",
            "shared_marketing_flights",
            "status_conflict_flights",
        ]:
            row["on_time_" + k] = None if o is None else o[k]
        scheduled = (
            o["reported_schedule_known"] if o and o["schedule_count_complete"] else None
        )
        performed = (
            o["non_cancelled_known"]
            if o and o["non_cancelled_count_complete"]
            else None
        )
        ts = None if t is None else t["DEPARTURES_SCHEDULED"]
        tp = None if t is None else t["DEPARTURES_PERFORMED"]
        seats = None if t is None else t["SEATS"]
        row.update(
            on_time_scheduled=scheduled,
            on_time_non_cancelled=performed,
            scheduled_difference=None
            if scheduled is None or ts is None
            else scheduled - ts,
            performed_difference=None
            if performed is None or tp is None
            else performed - tp,
            performed_relative_difference=None
            if performed is None or tp is None or tp <= 0
            else (performed - tp) / tp,
            seats_per_performed_departure_mean=None
            if seats is None or tp is None or tp <= 0
            else seats / tp,
            implied_on_time_seats=None
            if seats is None or tp is None or tp <= 0 or performed is None
            else seats * performed / tp,
            implied_seat_difference=None
            if seats is None or tp is None or tp <= 0 or performed is None
            else seats * (performed - tp) / tp,
            implied_seat_relative_difference=None
            if not seats or tp is None or tp <= 0 or performed is None
            else (performed - tp) / tp,
            native_per_flight_capacity_adopted=False,
            V10_native_seat_check=None,
        )
        rows.append(row)
    return rows


def run_reconciliation(
    t100_path, source_parquet, scope, covered_months, out, progress=lambda **_: None
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "reported_counts.sqlite").exists():
        raise ValueError("new reconciliation output required")
    pairs = {tuple(p) for p in scope["airport_pairs"]}
    counts, source_rows = source_counts(source_parquet, out, progress)
    strata, read, kept = t100_projection(
        t100_path, pairs, covered_months, out, progress
    )
    rows = compare_counts(counts, strata, covered_months)
    return {
        "plan_id": "A.15/V10",
        "status": "completed",
        "source_rows": source_rows,
        "T100_rows_read": read,
        "T100_scoped_rows": kept,
        "coverage_months": sorted(covered_months),
        "class_counts": dict(Counter(x["service_class"] for x in strata)),
        "T100_strata": strata,
        "operator_route_months": rows,
        "join_counts": dict(Counter(x["join_status"] for x in rows)),
        "missing_source_month_keys": sum(
            x["coverage"] == "missing_source_month" for x in rows
        ),
        "matched_performed_comparisons": sum(
            x["performed_difference"] is not None for x in rows
        ),
        "performed_count_mismatches": sum(
            x["performed_difference"] not in (None, 0) for x in rows
        ),
        "no_seat_rescaling_or_allocation": True,
        "native_capacity_adopted": False,
        "full_V10_pass": False,
        "scientific_budget": None,
        "performed_basis": "Reported operating-flight identity; cancelled=0 includes diversions. Completed/no-diversion proxy separate. Missing sides are null, not zero.",
        "seat_scope": "SEATS/performed departures and implied on-time seats are unadopted diagnostics, not observed per-flight capacity, hidden inventory or native-model validation",
    }
