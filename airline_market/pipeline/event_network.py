"""A.15/V10: E3 dated observed-service resources and cancellation series.

Reuses the saved pre-event pair scope. Incumbents use Jan-Apr completed service
only. No DB1C scan, seat imputation, fare fit, legal-bookability or forecast claim.
"""

import hashlib
import json
import sqlite3
import zipfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from airline_market.pipeline.event_scope_v2 import marketing_calendar_date

FIELDS = (
    "FL_DATE",
    "MKT_UNIQUE_CARRIER",
    "OP_UNIQUE_CARRIER",
    "OP_CARRIER_AIRLINE_ID",
    "OP_CARRIER_FL_NUM",
    "MKT_CARRIER_FL_NUM",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "CRS_ELAPSED_TIME",
    "CANCELLED",
    "DIVERTED",
    "DUP",
    "DISTANCE",
)


def select_incumbents(scope):
    if sorted(tuple(p) for p in scope["periods"]) != [(2026, m) for m in range(1, 5)]:
        raise ValueError("saved selection must use all four pre-event months")
    pairs = {tuple(p) for p in scope["airport_pairs"]}
    counts = {p: Counter() for p in pairs}
    for r in scope["route_counts"]:
        p = tuple(r["airports"])
        if p in counts:
            counts[p][r["marketing_carrier"]] += r.get("completed_proxy", 0)
    result = []
    for pair in sorted(pairs):
        ranked = sorted(
            ((k, v) for k, v in counts[pair].items() if k != "NK" and v > 0),
            key=lambda x: (-x[1], x[0]),
        )
        if counts[pair]["NK"] <= 0 or not ranked:
            raise ValueError("saved candidate lacks NK or incumbent service")
        result.append(
            {
                "airports": list(pair),
                "strategic_carriers": ["NK"] + [k for k, _ in ranked[:2]],
                "incumbent_ranking": [
                    {"carrier": k, "completed_service": v, "rank": i + 1}
                    for i, (k, v) in enumerate(ranked)
                ],
                "selection_basis": "Jan-Apr2026 completed marketed-service counts; ties use carrier code",
                "tied_boundary": bool(len(ranked) > 2 and ranked[1][1] == ranked[2][1]),
            }
        )
    return result


def integer(value):
    v = Decimal(str(value))
    if not v.is_finite() or v != v.to_integral_value():
        raise ValueError("not an integer")
    return int(v)


def clock(value):
    v = integer(value)
    if v == 2400:
        return 0, 0, 1
    h, m = divmod(v, 100)
    if not 0 <= h < 24 or not 0 <= m < 60:
        raise ValueError("invalid clock")
    return h, m, 0


def scheduled_times(day, dep, arr, elapsed, origin, dest, zones):
    """Enumerate valid local folds, require exact arrival-minute agreement."""
    if origin not in zones or dest not in zones:
        return {"timing_status": "missing_timezone"}
    try:
        h, m, nextday = clock(dep)
        ah, am, _ = clock(arr)
        duration = integer(elapsed)
        if duration <= 0:
            raise ValueError("nonpositive elapsed")
        naive = datetime.fromisoformat(day).replace(hour=h, minute=m) + timedelta(
            days=nextday
        )
    except (ValueError, TypeError, ArithmeticError):
        return {"timing_status": "invalid_clock_or_elapsed"}
    oz, dz = ZoneInfo(zones[origin]), ZoneInfo(zones[dest])
    candidates = {}
    valid_local = set()
    for fold in [0, 1]:
        utc = naive.replace(tzinfo=oz, fold=fold).astimezone(timezone.utc)
        if utc.astimezone(oz).replace(tzinfo=None) != naive:
            continue
        valid_local.add(utc)
        end = utc + timedelta(minutes=duration)
        local_arr = end.astimezone(dz)
        if (local_arr.hour, local_arr.minute) == (ah, am):
            candidates[utc] = end
    if len(candidates) != 1:
        return {
            "timing_status": "nonexistent_local_departure"
            if not valid_local
            else "ambiguous_local_departure"
            if len(candidates) > 1
            else "arrival_clock_disagreement"
        }
    dep_utc, arr_utc = next(iter(candidates.items()))
    return {
        "timing_status": "ok",
        "departure_utc": int(dep_utc.timestamp()),
        "arrival_utc": int(arr_utc.timestamp()),
        "departure_local": dep_utc.astimezone(oz).isoformat(),
        "arrival_local": arr_utc.astimezone(dz).isoformat(),
    }


def status(value):
    try:
        v = integer(value)
    except (ValueError, TypeError, ArithmeticError):
        return None
    return v if v in [0, 1] else None


def normalized_row(raw, source, row_number, selection, zones):
    row = dict(raw)
    row["source_row_id"] = source["sha256"] + ":" + str(row_number)
    row["source_year"] = source["year"]
    row["source_month"] = source["month"]
    canon = {
        k: (raw.get(k) or "").strip()
        for k in ["ORIGIN", "DEST", "MKT_UNIQUE_CARRIER", "OP_UNIQUE_CARRIER"]
    }
    row.update(
        origin=canon["ORIGIN"],
        dest=canon["DEST"],
        marketing_carrier=canon["MKT_UNIQUE_CARRIER"],
        operating_carrier=canon["OP_UNIQUE_CARRIER"],
    )
    day = None
    if raw["FL_DATE"]:
        d, _ = marketing_calendar_date(raw["FL_DATE"])
        if (d.year, d.month) != (source["year"], source["month"]):
            raise ValueError("wrong source reporting period")
        day = d.isoformat()
    row["service_date"] = day
    row["cancelled_flag"] = status(raw["CANCELLED"])
    row["diverted_flag"] = status(raw["DIVERTED"])
    row["strategic"] = row["marketing_carrier"] in selection["strategic_carriers"]
    times = (
        scheduled_times(
            day,
            raw["CRS_DEP_TIME"],
            raw["CRS_ARR_TIME"],
            raw["CRS_ELAPSED_TIME"],
            row["origin"],
            row["dest"],
            zones,
        )
        if day
        else {"timing_status": "missing_date"}
    )
    row.update(
        {
            k: times.get(k)
            for k in [
                "departure_utc",
                "arrival_utc",
                "departure_local",
                "arrival_local",
            ]
        }
    )
    row.update(timing_status=times["timing_status"], physical_id=None)
    if times["timing_status"] == "ok":
        try:
            opid = integer(raw["OP_CARRIER_AIRLINE_ID"])
            flight = integer(raw["OP_CARRIER_FL_NUM"])
            if opid <= 0 or flight <= 0 or not row["operating_carrier"]:
                raise ValueError("missing operator identity")
        except (ValueError, TypeError, ArithmeticError):
            row["timing_status"] = "missing_operator_identity"
        else:
            key = [
                row["operating_carrier"],
                opid,
                flight,
                row["origin"],
                row["dest"],
                times["departure_utc"],
            ]
            row["physical_id"] = hashlib.sha256(
                json.dumps(key, separators=(",", ":")).encode()
            ).hexdigest()
    return row


def row_schema():
    return pa.schema(
        [(f, pa.string()) for f in FIELDS]
        + [
            (k, t)
            for k, t in [
                ("origin", pa.string()),
                ("dest", pa.string()),
                ("marketing_carrier", pa.string()),
                ("operating_carrier", pa.string()),
                ("source_row_id", pa.string()),
                ("source_year", pa.int64()),
                ("source_month", pa.int64()),
                ("service_date", pa.string()),
                ("cancelled_flag", pa.int64()),
                ("diverted_flag", pa.int64()),
                ("strategic", pa.bool_()),
                ("departure_utc", pa.int64()),
                ("arrival_utc", pa.int64()),
                ("departure_local", pa.string()),
                ("arrival_local", pa.string()),
                ("timing_status", pa.string()),
                ("physical_id", pa.string()),
            ]
        ]
    )


def compile_service(sources, scope, zones, out, progress=lambda **_: None):
    """Stream all8 months, keep every scoped raw row and a single physical resource."""
    if sorted((s["year"], s["month"]) for s in sources) != [
        (2026, m) for m in range(1, 9)
    ]:
        raise ValueError("all eight Jan-Aug2026 marketing sources required")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    incumbents = select_incumbents(scope)
    selected = {tuple(r["airports"]): r for r in incumbents}
    # All timezone identifiers are explicit configuration, checked before output.
    for z in zones.values():
        ZoneInfo(z)
    markets = {a + "|" + b for a, b in scope["directed_markets"]}
    selection_set = pa.array(sorted(markets))
    dbpath = out / "service_index.sqlite"
    assert not dbpath.exists()
    db = sqlite3.connect(dbpath)
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-8192")
    db.execute(
        "CREATE TABLE rows(row_id TEXT PRIMARY KEY,pid TEXT,day TEXT,origin TEXT,dest TEXT,mkt TEXT,op TEXT,mflight TEXT,cancelled INTEGER,diverted INTEGER,dep INTEGER,arr INTEGER,timing TEXT,strategic INTEGER)"
    )
    writer = pq.ParquetWriter(
        out / "source_rows.parquet", row_schema(), compression="zstd"
    )
    files = []
    issues = Counter()
    total = Counter()
    daily = None
    physical = None
    products = None
    try:
        for source in sorted(sources, key=lambda s: s["month"]):
            read = kept = 0
            formats = Counter()
            datecache = {}
            with zipfile.ZipFile(source["path"]) as archive:
                members = [
                    m
                    for m in archive.infolist()
                    if m.filename.lower().endswith(".csv")
                    and Path(m.filename).name.lower() != "documentation.csv"
                ]
                if len(members) != 1:
                    raise ValueError("one marketing CSV required")
                member = members[0]
                with archive.open(member) as stream:
                    reader = pacsv.open_csv(
                        stream,
                        read_options=pacsv.ReadOptions(
                            block_size=1024**2, use_threads=False
                        ),
                        convert_options=pacsv.ConvertOptions(
                            include_columns=FIELDS,
                            column_types={f: pa.string() for f in FIELDS},
                        ),
                    )
                    for batch in reader:
                        for value in pc.unique(batch.column("FL_DATE")).to_pylist():
                            if value is None or value == "":
                                continue
                            if value not in datecache:
                                datecache[value] = marketing_calendar_date(value)
                            d, _fmt = datecache[value]
                            if (d.year, d.month) != (source["year"], source["month"]):
                                raise ValueError("wrong source reporting period")
                        keys = pc.binary_join_element_wise(
                            pc.utf8_trim_whitespace(batch.column("ORIGIN")),
                            pc.utf8_trim_whitespace(batch.column("DEST")),
                            "|",
                        )
                        mask = pc.is_in(keys, value_set=selection_set).to_pylist()
                        raw = batch.filter(pa.array(mask)).to_pylist()
                        ordinals = [read + i for i, yes in enumerate(mask) if yes]
                        read += batch.num_rows
                        normalized = []
                        for r, n in zip(raw, ordinals):
                            pair = tuple(
                                sorted((r["ORIGIN"].strip(), r["DEST"].strip()))
                            )
                            row = normalized_row(r, source, n, selected[pair], zones)
                            normalized.append(row)
                            issues[row["timing_status"]] += 1
                            if r["FL_DATE"]:
                                formats[datecache[r["FL_DATE"]][1]] += 1
                        if normalized:
                            writer.write_table(
                                pa.Table.from_pylist(normalized, schema=row_schema())
                            )
                            db.executemany(
                                "INSERT INTO rows VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                [
                                    (
                                        r["source_row_id"],
                                        r["physical_id"],
                                        r["service_date"],
                                        r["origin"],
                                        r["dest"],
                                        r["marketing_carrier"],
                                        r["operating_carrier"],
                                        r["MKT_CARRIER_FL_NUM"],
                                        r["cancelled_flag"],
                                        r["diverted_flag"],
                                        r["departure_utc"],
                                        r["arrival_utc"],
                                        r["timing_status"],
                                        r["strategic"],
                                    )
                                    for r in normalized
                                ],
                            )
                            db.commit()
                            kept += len(normalized)
                        progress(
                            stage="dated_marketing_stream",
                            month=source["month"],
                            rows_read=read,
                            rows_kept=kept,
                        )
            files.append(
                {
                    "path": source["path"],
                    "sha256": source["sha256"],
                    "year": source["year"],
                    "month": source["month"],
                    "rows_read": read,
                    "rows_kept": kept,
                    "member": member.filename,
                    "date_formats": dict(formats),
                }
            )
            total.update(rows_read=read, rows_kept=kept)
        db.execute("CREATE INDEX physical_rows ON rows(pid)")
        db.execute("CREATE INDEX daily_rows ON rows(day,mkt,op,origin,dest)")
        db.commit()
        ds = pa.schema(
            [
                (k, t)
                for k, t in [
                    ("service_date", pa.string()),
                    ("marketing_carrier", pa.string()),
                    ("operating_carrier", pa.string()),
                    ("origin", pa.string()),
                    ("dest", pa.string()),
                    ("source_rows", pa.int64()),
                    ("cancelled", pa.int64()),
                    ("diverted", pa.int64()),
                    ("completed_proxy", pa.int64()),
                    ("unknown_status", pa.int64()),
                ]
            ]
        )
        daily = pq.ParquetWriter(
            out / "daily_marketing.parquet", ds, compression="zstd"
        )
        daily_count = 0
        cursor = db.execute(
            "SELECT day,mkt,op,origin,dest,COUNT(*),SUM(cancelled=1),SUM(diverted=1),SUM(cancelled=0 AND diverted=0),SUM(cancelled IS NULL OR diverted IS NULL) FROM rows GROUP BY day,mkt,op,origin,dest ORDER BY day,mkt,op,origin,dest"
        )
        while records := cursor.fetchmany(2048):
            table = pa.Table.from_pylist(
                [
                    dict(zip(ds.names, [*r[:6], *[int(x or 0) for x in r[6:]]]))
                    for r in records
                ],
                schema=ds,
            )
            daily.write_table(table)
            daily_count += len(records)
        ps = pa.schema(
            [
                (k, t)
                for k, t in [
                    ("physical_id", pa.string()),
                    ("origin", pa.string()),
                    ("dest", pa.string()),
                    ("departure_utc", pa.int64()),
                    ("arrival_utc", pa.int64()),
                    ("source_rows", pa.int64()),
                    ("marketing_carriers", pa.list_(pa.string())),
                    ("operating_carriers", pa.list_(pa.string())),
                    ("identity_or_status_conflict", pa.bool_()),
                    ("service_status", pa.string()),
                    ("ownership_ambiguous", pa.bool_()),
                ]
            ]
        )
        physical = pq.ParquetWriter(
            out / "physical_resources.parquet", ps, compression="zstd"
        )
        xs = pa.schema(
            [
                (k, t)
                for k, t in [
                    ("product_id", pa.string()),
                    ("physical_id", pa.string()),
                    ("origin", pa.string()),
                    ("dest", pa.string()),
                    ("marketing_carrier", pa.string()),
                    ("marketing_flight_numbers", pa.list_(pa.string())),
                    ("strategic", pa.bool_()),
                    ("shared_resource", pa.bool_()),
                ]
            ]
        )
        products = pq.ParquetWriter(
            out / "nonstop_products.parquet", xs, compression="zstd"
        )
        physical_count = product_count = ambiguous = conflicts = 0
        cursor = db.execute(
            "SELECT pid,MIN(origin),MIN(dest),MIN(dep),MIN(arr),MAX(arr),COUNT(*),GROUP_CONCAT(DISTINCT mkt),GROUP_CONCAT(DISTINCT op),MIN(cancelled),MAX(cancelled),MIN(diverted),MAX(diverted),SUM(cancelled IS NULL OR diverted IS NULL) FROM rows WHERE pid IS NOT NULL GROUP BY pid ORDER BY pid"
        )
        while records := cursor.fetchmany(1024):
            pr = []
            xr = []
            for r in records:
                (
                    pid,
                    o,
                    d,
                    dep,
                    arr,
                    amax,
                    n,
                    mkts,
                    ops,
                    cmin,
                    cmax,
                    dmin,
                    dmax,
                    unknown,
                ) = r
                mkts = sorted(x for x in (mkts or "").split(",") if x)
                ops = sorted(x for x in (ops or "").split(",") if x)
                conflict = bool(arr != amax or cmin != cmax or dmin != dmax)
                state = (
                    "conflicting"
                    if conflict
                    else "unknown"
                    if unknown
                    else "cancelled"
                    if cmin
                    else "diverted"
                    if dmin
                    else "completed_proxy"
                )
                pr.append(
                    {
                        "physical_id": pid,
                        "origin": o,
                        "dest": d,
                        "departure_utc": dep,
                        "arrival_utc": arr if arr == amax else None,
                        "source_rows": n,
                        "marketing_carriers": mkts,
                        "operating_carriers": ops,
                        "identity_or_status_conflict": conflict,
                        "service_status": state,
                        "ownership_ambiguous": len(mkts) > 1,
                    }
                )
                ambiguous += len(mkts) > 1
                conflicts += conflict
                for mkt in mkts:
                    nums = sorted(
                        {
                            x[0]
                            for x in db.execute(
                                "SELECT DISTINCT mflight FROM rows WHERE pid=? AND mkt=?",
                                (pid, mkt),
                            )
                            if x[0]
                        }
                    )
                    xr.append(
                        {
                            "product_id": hashlib.sha256(
                                (pid + "|" + mkt).encode()
                            ).hexdigest(),
                            "physical_id": pid,
                            "origin": o,
                            "dest": d,
                            "marketing_carrier": mkt,
                            "marketing_flight_numbers": nums,
                            "strategic": mkt
                            in selected[tuple(sorted((o, d)))]["strategic_carriers"],
                            "shared_resource": len(mkts) > 1,
                        }
                    )
            physical.write_table(pa.Table.from_pylist(pr, schema=ps))
            products.write_table(pa.Table.from_pylist(xr, schema=xs))
            physical_count += len(pr)
            product_count += len(xr)
            progress(stage="physical_resource_join", physical_resources=physical_count)
        native_counts = db.execute(
            "SELECT COUNT(*),SUM(cancelled=1),SUM(diverted=1),SUM(cancelled=0 AND diverted=0),SUM(pid IS NULL) FROM rows"
        ).fetchone()
        return dict(
            plan_id="A.15/V10",
            status="completed",
            files=files,
            incumbents=incumbents,
            airport_pairs=len(selected),
            directed_markets=len(markets),
            **dict(total),
            timing_status_counts=dict(issues),
            physical_resources=physical_count,
            nonstop_products=product_count,
            daily_groups=daily_count,
            ambiguous_marketing_resources=ambiguous,
            conflicting_resources=conflicts,
            scoped_source_totals=dict(
                zip(
                    [
                        "rows",
                        "cancelled",
                        "diverted",
                        "completed_proxy",
                        "unresolved_physical_identity",
                    ],
                    [int(x or 0) for x in native_counts],
                )
            ),
            source_count_basis="Original marketing rows, not unique operating flights; duplicate flags retained",
            full_V10_pass=False,
            scientific_budget=None,
            scope="Observed scheduled resources/nonstop marketing products only; no seat reconciliation, legal bookability, ex ante forecasts or native model/calibration adoption",
        )
    finally:
        writer.close()
        for sink in [daily, physical, products]:
            if sink is not None:
                sink.close()
        db.close()
