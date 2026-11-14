"""A.15/V10: arithmetic source traces of saved departure-count discrepancies.

A T100 scheduled-minus-performed count is a signed difference, not an observed
cancellation count. No causal reason, changed count basis or capacity is adopted.
"""

import hashlib
import itertools
import json
import sqlite3
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from airline_market.pipeline.event_seats import integer

KEYS = ("year", "month", "operating_airline_id", "origin", "dest")


def _difference(a, b):
    return None if a is None or b is None else a - b


def trace_row(row):
    """Retain a row and decompose its primary difference without zero filling."""
    x = dict(row)
    os, op = row["on_time_scheduled"], row["on_time_non_cancelled"]
    ts, tp = row["T100_DEPARTURES_SCHEDULED"], row["T100_DEPARTURES_PERFORMED"]
    sd, pd = _difference(os, ts), _difference(op, tp)
    for name, expected in [("scheduled_difference", sd), ("performed_difference", pd)]:
        if row[name] != expected:
            raise ValueError("saved count arithmetic differs: " + name)
    ot_shortfall, t100_shortfall = _difference(os, op), _difference(ts, tp)
    shortfall_delta = _difference(ot_shortfall, t100_shortfall)
    residual = (
        None
        if sd is None or pd is None or shortfall_delta is None
        else pd - (sd - shortfall_delta)
    )
    if residual not in [None, 0]:
        raise ValueError("departure decomposition does not close")
    if row["join_status"] != "matched":
        category = row["join_status"]
    elif residual is None:
        category = "matched_missing_count"
    elif sd == 0 and shortfall_delta == 0:
        category = "both_components_agree"
    elif sd == 0:
        category = "schedule_agrees_net_shortfall_differs"
    elif shortfall_delta == 0:
        category = "schedule_differs_net_shortfall_agrees"
    elif pd == 0:
        category = "nonzero_components_offset"
    else:
        category = "both_components_differ"
    complete = row["on_time_completed_proxy_known"]
    diversion = row["on_time_diverted_known"]
    # Complete/proxy is a sensitivity, not a replacement for primary non-cancelled.
    proxy_delta = _difference(complete, tp)
    basis_change = _difference(proxy_delta, pd)
    if op is not None and complete is not None and diversion is not None:
        if complete + diversion != op:
            raise ValueError("non-cancelled does not equal completed plus diverted")
    if (
        basis_change is not None
        and diversion is not None
        and basis_change != -diversion
    ):
        raise ValueError("diversion-basis sensitivity does not close")
    x.update(
        trace_category=category,
        on_time_schedule_minus_non_cancelled=ot_shortfall,
        T100_schedule_minus_performed=t100_shortfall,
        net_shortfall_difference=shortfall_delta,
        decomposition_residual=residual,
        T100_performed_exceeds_scheduled=None
        if t100_shortfall is None
        else t100_shortfall < 0,
        completed_proxy_difference_sensitivity=proxy_delta,
        completed_basis_change_sensitivity=basis_change,
        cancellation_cause_identified=False,
        discrepancy_resolved=False if pd not in [None, 0] else None,
        primary_count_basis_changed=False,
    )
    return x


def source_witnesses(path):
    """Digest each prepared operating group, retaining null statuses and UTC IDs."""
    db = sqlite3.connect("file:" + str(Path(path).resolve()) + "?mode=ro", uri=True)
    db.execute("PRAGMA temp_store=FILE")
    cursor = db.execute(
        "SELECT y,m,opid,o,d,sid,k,c,v,utc_id FROM rows ORDER BY y,m,opid,o,d,sid"
    )
    result = {}
    for key, rows in itertools.groupby(cursor, key=lambda r: r[:5]):
        h = hashlib.sha256()
        counts = Counter()
        known = set()
        for z in rows:
            h.update((json.dumps(z, separators=(",", ":")) + "\n").encode())
            counts["source_rows"] += 1
            counts["cancelled_rows"] += z[7] == 1
            counts["diverted_rows"] += z[8] == 1
            counts["unknown_status_rows"] += z[7] is None or z[8] is None
            counts["UTC_unresolved_rows"] += z[9] is None
            counts["unresolved_reported_identity_rows"] += z[6] is None
            if z[6] is not None:
                known.add(z[6])
        result[key] = dict(
            counts, reported_identities=len(known), source_group_sha256=h.hexdigest()
        )
    db.close()
    return result


def T100_witnesses(path):
    """Use the saved selected projection only; do not read the source archive."""
    groups = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
        for row in batch.to_pylist():
            key = (
                integer(row["YEAR"]),
                integer(row["MONTH"]),
                integer(row["AIRLINE_ID"]),
                row["ORIGIN"].strip(),
                row["DEST"].strip(),
                row["CLASS"].strip(),
            )
            x = groups.setdefault(key, {"source_rows": 0, "digest": hashlib.sha256()})
            x["source_rows"] += 1
            x["digest"].update(
                (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            )
    return {
        k: dict(
            source_rows=x["source_rows"],
            projection_group_sha256=x["digest"].hexdigest(),
        )
        for k, x in groups.items()
    }


def run_trace(parent, prepared_index, T100_projection):
    """All outer rows, all witnesses and signed/null outcomes; no reason imputation."""
    source, t100 = source_witnesses(prepared_index), T100_witnesses(T100_projection)
    traces = []
    for row in parent["operator_route_months"]:
        z = trace_row(row)
        key = tuple(row[k] for k in KEYS)
        o = source.get(key) if row["on_time_source_rows"] is not None else None
        t = t100.get((*key, "F")) if row["T100_source_rows"] is not None else None
        if (None if o is None else o["source_rows"]) != row["on_time_source_rows"]:
            raise ValueError("prepared source group does not match comparison")
        if (None if t is None else t["source_rows"]) != row["T100_source_rows"]:
            raise ValueError("saved T100 group does not match comparison")
        z.update(
            source_group_sha256=None if o is None else o["source_group_sha256"],
            projection_group_sha256=None if t is None else t["projection_group_sha256"],
        )
        traces.append(z)
    compared = [z for z in traces if z["performed_difference"] is not None]
    mismatches = [z for z in compared if z["performed_difference"] != 0]
    all_categories = dict(Counter(z["trace_category"] for z in traces))
    return dict(
        plan_id="A.15/V10",
        status="completed",
        operator_route_months=len(traces),
        trace_categories=all_categories,
        mismatch_categories=dict(Counter(z["trace_category"] for z in mismatches)),
        performed_count_mismatches=len(mismatches),
        performed_difference_signed_sum=sum(
            z["performed_difference"] for z in compared
        ),
        performed_difference_absolute_sum=sum(
            abs(z["performed_difference"]) for z in compared
        ),
        matched_T100_performed_exceeds_scheduled=sum(
            z["T100_performed_exceeds_scheduled"] is True
            for z in traces
            if z["join_status"] == "matched"
        ),
        matched_diversion_basis_changes=sum(
            z["completed_basis_change_sensitivity"] not in [None, 0]
            for z in traces
            if z["join_status"] == "matched"
        ),
        missing_local_month_rows=sum(
            z["coverage"] == "missing_source_month" for z in traces
        ),
        source_rows=sum(z["source_rows"] for z in source.values()),
        T100_projection_rows=sum(z["source_rows"] for z in t100.values()),
        trace_rows=traces,
        source_witnesses=[dict(zip(KEYS, k), **v) for k, v in source.items()],
        T100_witnesses=[
            dict(zip((*KEYS, "service_class"), k), **v) for k, v in t100.items()
        ],
        scope="Arithmetic components and source membership only; T100 signed net shortfall is not observed cancellations or identified cause",
        primary_basis_changed=False,
        discrepancies_resolved=0,
        capacity_adopted=False,
        full_V10_pass=False,
        scientific_budget=None,
    )
