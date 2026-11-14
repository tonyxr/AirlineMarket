"""A.11/A.15/V10: lossless-decimal moments of prepared Market purchases.

Reporting/scheduled months, validity, nonstop and raw window labels stay separate.
Known-support means do not impute missing mass. No calibration target is adopted.
"""

import json
import math
import sqlite3
from decimal import Context, Decimal, Inexact
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

KEYS = (
    "RpYear",
    "RpMonth",
    "SchFlYear",
    "SchFlMonth",
    "Origin",
    "Dest",
    "MktCarrier",
    "MktCarrierAirlineID",
    "PurchaseWindowGroup",
    "_purchase_window_status",
    "_event_partition",
    "Nonstop",
    "V_Yield",
)
VALUES = ("Passengers", "MktAmount", "MktTax", "_sample_expansion")
COUNTS = (
    "rows",
    "passenger_missing_rows",
    "passenger_negative_rows",
    "passenger_zero_rows",
    "expansion_invalid_rows",
    "gross_missing_rows",
    "gross_nonpositive_rows",
    "gross_missing_positive_passenger_rows",
    "tax_missing_rows",
    "tax_negative_rows",
    "tax_missing_positive_passenger_rows",
)
SUMS = (
    "raw_passengers",
    "known_nonnegative_passengers",
    "raw_expanded_passengers",
    "gross_numerator",
    "gross_denominator",
    "gross_expanded_numerator",
    "gross_expanded_denominator",
    "tax_numerator",
    "tax_denominator",
    "tax_expanded_numerator",
    "tax_expanded_denominator",
)
EXACT = Context(prec=100)
EXACT.traps[Inexact] = True
DISPLAY = Context(prec=34)


def number(x):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    d = Decimal(str(x))
    return d if d.is_finite() else None


def empty():
    return [0] * len(COUNTS) + [Decimal(0)] * len(SUMS)


def add(a, b):
    return [x + y for x, y in zip(a[: len(COUNTS)], b[: len(COUNTS)])] + [
        EXACT.add(x, y) for x, y in zip(a[len(COUNTS) :], b[len(COUNTS) :])
    ]


def contribution(p, f, t, e, n):
    p, f, t, e = map(number, (p, f, t, e))
    valid_p = p is not None and p >= 0
    valid_e = e is not None and e > 0
    c = [
        n,
        n if p is None else 0,
        n if p is not None and p < 0 else 0,
        n if p == 0 else 0,
        n if not valid_e else 0,
        n if f is None else 0,
        n if f is not None and f <= 0 else 0,
        n if valid_p and p > 0 and f is None else 0,
        n if t is None else 0,
        n if t is not None and t < 0 else 0,
        n if valid_p and p > 0 and t is None else 0,
    ]
    s = [Decimal(0) for _ in SUMS]
    pn = EXACT.multiply(p, n) if p is not None else Decimal(0)
    if p is not None:
        s[0] = pn
    if valid_p:
        s[1] = pn
    if p is not None and valid_e:
        s[2] = EXACT.multiply(pn, e)
    for value, offset in [(f, 3), (t, 7)]:
        if valid_p and value is not None:
            s[offset] = EXACT.multiply(pn, value)
            s[offset + 1] = pn
            if valid_e:
                s[offset + 2] = EXACT.multiply(s[offset], e)
                s[offset + 3] = EXACT.multiply(pn, e)
    return c + s


def finish(key, stat):
    row = dict(zip(KEYS, key))
    row.update(zip(COUNTS, stat[: len(COUNTS)]))
    row.update({k: str(v) for k, v in zip(SUMS, stat[len(COUNTS) :])})
    invalid_p = row["passenger_missing_rows"] + row["passenger_negative_rows"]
    for name in ["gross", "tax"]:
        for expanded in [False, True]:
            stem = name + ("_expanded" if expanded else "")
            num, den = (
                Decimal(row[stem + "_numerator"]),
                Decimal(row[stem + "_denominator"]),
            )
            value = float(DISPLAY.divide(num, den)) if den > 0 else None
            row[stem + "_known_support_mean"] = value
            complete = not (
                invalid_p
                or row[name + "_missing_positive_passenger_rows"]
                or (expanded and row["expansion_invalid_rows"])
            )
            row[stem + "_complete_mean"] = value if complete else None
    row["complete_observed_passengers"] = (
        float(Decimal(row["raw_passengers"])) if not invalid_p else None
    )
    row["complete_expanded_passengers"] = (
        float(Decimal(row["raw_expanded_passengers"]))
        if not (invalid_p or row["expansion_invalid_rows"])
        else None
    )
    row["calibration_adopted"] = False
    return row


def aggregate(sources, out, progress=lambda **_: None, batch_rows=32768):
    """Same accumulator accepts prepared data or declared simulation tables.

    Arrow collapses identical key/value tuples within a batch, retaining their
    exact multiplicity. Decimal accumulators and SQLite merge every contribution.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dbpath = out / "moment_groups.sqlite"
    if dbpath.exists():
        raise ValueError("new moment output required")
    db = sqlite3.connect(dbpath)
    db.execute("PRAGMA cache_size=-8192")
    db.execute("PRAGMA temp_store=FILE")
    db.create_function("dadd", 2, lambda a, b: str(EXACT.add(Decimal(a), Decimal(b))))
    names = [*COUNTS, *SUMS]
    db.execute(
        "CREATE TABLE groups(k TEXT PRIMARY KEY,"
        + ",".join(n + (" INTEGER" if n in COUNTS else " TEXT") for n in names)
        + ")"
    )
    query = (
        "INSERT INTO groups VALUES("
        + ",".join("?" for _ in range(len(names) + 1))
        + ") ON CONFLICT(k) DO UPDATE SET "
        + ",".join(
            n
            + "="
            + (
                "groups." + n + "+excluded." + n
                if n in COUNTS
                else "dadd(groups." + n + ",excluded." + n + ")"
            )
            for n in names
        )
    )
    months = []
    source_rows = 0
    for source in sources:
        total = empty()
        seen = 0
        for batch in pq.ParquetFile(source["path"]).iter_batches(
            batch_size=batch_rows,
            columns=[*KEYS, *VALUES, "_source_row"],
            use_threads=False,
        ):
            table = pa.Table.from_batches([batch])
            grouped = table.group_by([*KEYS, *VALUES], use_threads=False).aggregate(
                [("_source_row", "count")]
            )
            parts = {}
            for z in grouped.to_pylist():
                key = tuple(z[k] for k in KEYS)
                if (key[0], key[1]) != (source["year"], source["month"]):
                    raise ValueError("reporting period differs from registered source")
                n = z["_source_row_count"]
                stat = contribution(*(z[k] for k in VALUES), n)
                parts[key] = add(parts.get(key, empty()), stat)
                total = add(total, stat)
            db.executemany(
                query,
                [
                    (
                        json.dumps(k, separators=(",", ":"), allow_nan=False),
                        *v[: len(COUNTS)],
                        *(str(d) for d in v[len(COUNTS) :]),
                    )
                    for k, v in parts.items()
                ],
            )
            db.commit()
            seen += batch.num_rows
            source_rows += batch.num_rows
            progress(
                stage="prepared_market_moments",
                reporting_month=source["month"],
                source_rows=source_rows,
            )
        if seen != source["expected_rows"]:
            raise ValueError("prepared row count differs from original audit")
        observed = total[len(COUNTS)]
        if observed != Decimal(str(source["expected_observed_passengers"])):
            raise ValueError(
                "observed passengers differ from saved scoped BTS source total"
            )
        months.append(
            dict(
                reporting_year=source["year"],
                reporting_month=source["month"],
                **finish(
                    (
                        source["year"],
                        source["month"],
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ),
                    total,
                ),
                saved_source_rows=source["expected_rows"],
                saved_scope_observed_passengers=str(
                    source["expected_observed_passengers"]
                ),
                source_total_matches=True,
            )
        )
    # Wide Parquet keeps raw labels/nulls and exact decimal numerator/denominator strings.
    keytypes = [
        pa.int64(),
        pa.int64(),
        pa.int64(),
        pa.int64(),
        pa.string(),
        pa.string(),
        pa.string(),
        pa.int64(),
        pa.string(),
        pa.string(),
        pa.string(),
        pa.int64(),
        pa.int64(),
    ]
    schema = pa.schema(
        list(zip(KEYS, keytypes))
        + [(k, pa.int64()) for k in COUNTS]
        + [(k, pa.string()) for k in SUMS]
        + [
            (name + suffix, pa.float64())
            for name in ["gross", "gross_expanded", "tax", "tax_expanded"]
            for suffix in ["_known_support_mean", "_complete_mean"]
        ]
        + [
            ("complete_observed_passengers", pa.float64()),
            ("complete_expanded_passengers", pa.float64()),
            ("calibration_adopted", pa.bool_()),
        ]
    )
    groups = 0
    pending = []
    overall = empty()
    partitions = {}
    validity = {}
    windows = {}
    with pq.ParquetWriter(
        out / "event_fare_moments.parquet", schema, compression="zstd"
    ) as writer:
        for key, *values in db.execute("SELECT * FROM groups ORDER BY k"):
            stat = [
                *values[: len(COUNTS)],
                *(Decimal(v) for v in values[len(COUNTS) :]),
            ]
            key = json.loads(key)
            row = finish(key, stat)
            pending.append(row)
            groups += 1
            overall = add(overall, stat)
            for target, k in [
                (partitions, key[10]),
                (validity, str(key[12])),
                (windows, key[9]),
            ]:
                target[k] = add(target.get(k, empty()), stat)
            if len(pending) >= 2048:
                writer.write_table(pa.Table.from_pylist(pending, schema=schema))
                pending = []
        if pending:
            writer.write_table(pa.Table.from_pylist(pending, schema=schema))
    db.close()
    assert overall[0] == source_rows

    def totals(table):
        return [
            dict(
                label=k,
                **{
                    name: str(value) if name in SUMS else value
                    for name, value in zip(names, stat)
                },
            )
            for k, stat in sorted(table.items(), key=lambda kv: str(kv[0]))
        ]

    return dict(
        plan_id="A.11/A.15/V10",
        status="completed",
        rows=source_rows,
        groups=groups,
        months=months,
        partition_totals=totals(partitions),
        validity_totals=totals(validity),
        window_status_totals=totals(windows),
        overall=dict(
            zip(COUNTS, overall[: len(COUNTS)]),
            **{k: str(v) for k, v in zip(SUMS, overall[len(COUNTS) :])},
        ),
        passenger_source_totals_match=True,
        scope="Prepared monthly O&D purchases; raw validity/window/Nonstop and report/scheduled month strata; not potential demand, inventory, legal flight or net profit",
        national_fare_aggregate_reproduction=None,
        calibration_adopted=False,
        full_A11_pass=False,
        full_V10_pass=False,
        scientific_budget=None,
    )
