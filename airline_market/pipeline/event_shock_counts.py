"""A.15/B1/V10: source-calendar flight counts from TWO prepared indexes.

A new exact-ID/date join only. Reported flight cancellations do not identify
installed-seat loss, foregone schedules, legal bookability or stranded people.
"""

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from fractions import Fraction
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def rational(value):
    return (
        None
        if value is None
        else {"numerator": str(value.numerator), "denominator": str(value.denominator)}
    )


@dataclass(frozen=True)
class EventCountWindow:
    operating_code: str
    directions: tuple
    start_date: str
    stop_date: str

    def __post_init__(self):
        if not isinstance(self.operating_code, str) or not self.operating_code.strip():
            raise ValueError("explicit source operating code required")
        start, stop = (
            date.fromisoformat(self.start_date),
            date.fromisoformat(self.stop_date),
        )
        if (
            start.isoformat() != self.start_date
            or stop.isoformat() != self.stop_date
            or stop <= start
        ):
            raise ValueError(
                "explicit nonempty ISO source-calendar [start,stop) required"
            )
        directions = tuple(sorted(tuple(p) for p in self.directions))
        if (
            not directions
            or len(set(directions)) != len(directions)
            or any(
                len(p) != 2
                or p[0] == p[1]
                or any(not isinstance(a, str) or not a for a in p)
                for p in directions
            )
        ):
            raise ValueError("distinct explicit directed airport pairs required")
        object.__setattr__(self, "directions", directions)

    def record(self):
        return {
            "operating_code": self.operating_code,
            "directions": [list(p) for p in self.directions],
            "start_date": self.start_date,
            "stop_date": self.stop_date,
            "date_basis": "prepared source service calendar, NOT UTC reveal/observation time",
        }

    def calendar_coordinates(self):
        day, stop = (
            date.fromisoformat(self.start_date),
            date.fromisoformat(self.stop_date),
        )
        while day < stop:
            for origin, dest in self.directions:
                yield day.isoformat(), origin, dest
            day += timedelta(days=1)


def join_prepared_indexes(reported_index, service_index, definition):
    """Read-only exact sid=row_id join; no archive/Parquet/compiler/T100 replay.

    Every source row matching the explicit operating code is accounted for.
    Rows lacking a date join remain unassignable, never nearest-time filled.
    """
    if type(definition) is not EventCountWindow:
        raise TypeError("EventCountWindow required")
    db = sqlite3.connect(Path(reported_index).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        db.execute("PRAGMA query_only=ON")
        db.execute(
            "ATTACH DATABASE ? AS service",
            (Path(service_index).resolve().as_uri() + "?mode=ro",),
        )
        rows, unmatched = [], []
        outside = []
        seen = set()
        pairs = set(definition.directions)
        query = """SELECT r.sid,r.k,r.opid,r.op,r.o,r.d,r.c,r.v,r.utc_id,
                          s.row_id,s.day,s.origin,s.dest,s.op
                   FROM main.rows r LEFT JOIN service.rows s ON r.sid=s.row_id
                   WHERE r.op=? ORDER BY r.sid"""
        for (
            sid,
            key,
            opid,
            op,
            origin,
            dest,
            c,
            v,
            utc_id,
            right_id,
            day,
            so,
            sd,
            sop,
        ) in db.execute(query, (definition.operating_code,)):
            if sid in seen:
                raise ValueError("prepared source IDs are not unique")
            seen.add(sid)
            item = {
                "source_id": sid,
                "reported_key": key,
                "operator_id": opid,
                "operating_code": op,
                "origin": origin,
                "dest": dest,
                "cancelled": c,
                "diverted": v,
                "utc_id": utc_id,
                "service_date": day,
                "join_consistent": right_id is not None
                and (origin, dest, op) == (so, sd, sop),
            }
            if right_id is None or day is None:
                unmatched.append(item)
                continue
            parsed = date.fromisoformat(day)
            if parsed.isoformat() != day:
                raise ValueError("prepared service date is not canonical ISO calendar")
            if (
                definition.start_date <= day < definition.stop_date
                and (origin, dest) in pairs
            ):
                rows.append(item)
            else:
                outside.append(sid)
        return {
            "definition": definition.record(),
            "rows": rows,
            "unassignable_rows": unmatched,
            "source_operator_rows": len(seen),
            "outside_declared_window_or_routes": len(outside),
            "outside_source_ids_sha256": digest(outside),
        }
    finally:
        db.close()


def profile_joined_rows(joined, definition):
    """Count reported operating identities once, retaining unknown/conflicting flags.

    This can review the NEW saved join without reopening old prepared banks.
    No-source calendar cells have NULL rates, never zero flight demand/removal.
    """
    if joined["definition"] != definition.record():
        raise ValueError("joined source scope differs")
    groups = {k: [] for k in definition.calendar_coordinates()}
    source_ids = set()
    key_groups = {}
    for row in joined["rows"]:
        sid = row["source_id"]
        if sid in source_ids:
            raise ValueError("duplicate joined source row")
        source_ids.add(sid)
        key = row["service_date"], row["origin"], row["dest"]
        if key not in groups or row["operating_code"] != definition.operating_code:
            raise ValueError("joined row outside explicit source scope")
        groups[key].append(row)
        if row["reported_key"] is not None:
            key_groups.setdefault(row["reported_key"], set()).add(key)
    if (
        len(source_ids)
        + len(joined["unassignable_rows"])
        + joined["outside_declared_window_or_routes"]
        != joined["source_operator_rows"]
    ):
        raise ValueError("selected operator source partition does not conserve rows")
    unassignable = len(joined["unassignable_rows"])
    result = []
    for (day, origin, dest), members in groups.items():
        flights = {}
        unresolved = inconsistent = 0
        for row in members:
            unresolved += row["reported_key"] is None
            inconsistent += not row["join_consistent"]
            if row["reported_key"] is not None:
                flights.setdefault(row["reported_key"], []).append(row)
        cancelled = non_cancelled = unknown = conflicts = completed = diverted = (
            diversion_unknown
        ) = cross_group = 0
        for key, records in flights.items():
            c, v = {r["cancelled"] for r in records}, {r["diverted"] for r in records}
            cross_group += len(key_groups[key]) > 1
            conflicts += len(c - {None}) > 1 or len(v - {None}) > 1
            if len(c) != 1 or next(iter(c)) not in {0, 1}:
                unknown += 1
                continue
            if next(iter(c)) == 1:
                cancelled += 1
            else:
                non_cancelled += 1
                if len(v) != 1 or next(iter(v)) not in {0, 1}:
                    diversion_unknown += 1
                elif next(iter(v)) == 1:
                    diverted += 1
                else:
                    completed += 1
        reasons = []
        if not members:
            reasons.append("no_reported_source_rows")
        if unresolved:
            reasons.append("unresolved_operating_identity")
        if unknown:
            reasons.append("unknown_or_conflicting_cancellation")
        if inconsistent:
            reasons.append("inconsistent_exact_source_join")
        if cross_group:
            reasons.append("operating_identity_crosses_calendar_groups")
        if unassignable:
            reasons.append("unassignable_operator_rows_may_belong_to_window")
        rate = None if reasons else Fraction(cancelled, len(flights))
        result.append(
            {
                "service_date": day,
                "origin": origin,
                "dest": dest,
                "operating_code": definition.operating_code,
                "operator_ids": sorted(
                    {r["operator_id"] for r in members if r["operator_id"] is not None}
                ),
                "source_rows": len(members),
                "identified_reported_flights": len(flights),
                "known_cancelled": cancelled,
                "known_non_cancelled_including_diversions": non_cancelled,
                "unknown_cancellation": unknown,
                "status_conflict_flights": conflicts,
                "completed_no_diversion_proxy": completed,
                "non_cancelled_diversion": diverted,
                "non_cancelled_unknown_diversion": diversion_unknown,
                "unresolved_identity_rows": unresolved,
                "utc_unresolved_rows": sum(r["utc_id"] is None for r in members),
                "join_inconsistent_rows": inconsistent,
                "cross_group_flights": cross_group,
                "reported_cancellation_fraction": rational(rate),
                "rate_null_reasons": reasons,
                "source_membership_sha256": digest(
                    sorted(r["source_id"] for r in members)
                ),
                "native_installed_seat_loss_share": None,
                "B1_physical_range_check": None,
            }
        )
    return {
        "plan_id": "A.15/B1/V10",
        "definition": definition.record(),
        "rows": result,
        "source_partition": {
            k: joined[k]
            for k in [
                "source_operator_rows",
                "outside_declared_window_or_routes",
                "outside_source_ids_sha256",
            ]
        },
        "joined_rows": len(source_ids),
        "unassignable_operator_rows": unassignable,
        "calendar_cells": len(result),
        "measured_reported_flight_cells": sum(
            r["reported_cancellation_fraction"] is not None for r in result
        ),
        "native_capacity_adopted": False,
        "full_B1_anchoring": False,
        "scientific_budget": None,
        "limitations": [
            "reported source-calendar flight counts, not installed seats or foregone schedules",
            "no-report days are not zero demand, zero removal or evidence of absent scheduled/booked passengers",
            "UTC timing errors retained; source service day is not a UTC reveal boundary",
            "cancellation includes reported cancellations only; non-cancelled retains diversions",
            "no per-flight seats, exposure, legal bookability, stranded identity or post-event fitting",
        ],
    }
