"""A.10/A.5/B0/V9: recurrent unreserved spans on a declared native UTC clock.

Read NEW supplied complete accounts only. Not continuous waiting or delivered CS.
"""

from fractions import Fraction

from airline_market.core.hashing import jsonable, spec_hash
from airline_market.core.snapshot import deep_freeze
from airline_market.policy.dated_reserve import UtcBookingClock, utc
from airline_market.sim.passenger_accounts import OriginalPassengerAccounts, _number


def _hours(start, stop):
    delta = stop - start
    micros = (delta.days * 86400 + delta.seconds) * 1_000_000 + delta.microseconds
    if micros < 0:
        raise ValueError("negative native unreserved span; no clock repair")
    return Fraction(micros, 3_600_000_000)


def unreserved_spells(accounts: OriginalPassengerAccounts, clock: UtcBookingClock):
    """Map actual cancellation -> next native replacement or observed prefix end.

    Every cancelled ticket contributes one non-overlapping span on its ORIGINAL
    person. Native cancellation period c maps to clock[c]; genuine re-entry book
    period b maps to clock[b], as in the frozen dated re-entry adapter. Zero span
    means the SAME declared opportunity instant, not zero real-world waiting.
    """
    if (
        type(accounts) is not OriginalPassengerAccounts
        or type(clock) is not UtcBookingClock
    ):
        raise TypeError(
            "NEW OriginalPassengerAccounts and explicit UtcBookingClock required"
        )
    data = accounts.data
    K = data["completed_native_periods"]
    if type(K) is not int or not 0 <= K < len(clock.boundaries):
        raise ValueError("clock must cover the complete observed native prefix")
    now = clock.at(K)
    roots = set()
    ticket_ids = set()
    coordinates = set()
    people = []
    checked = undated = 0
    for account in data["lineages"]:
        root = account["original_record"]
        coord = tuple(account["potential_coordinate"])
        if type(root) is not int or root < 0 or root in roots or coord in coordinates:
            raise ValueError("distinct original person/root coordinates required")
        roots.add(root)
        coordinates.add(coord)
        tickets = tuple(sorted(account["tickets"], key=lambda t: t["record_id"]))
        if (
            not tickets
            or tickets[0]["record_id"] != root
            or len(coord) != 3
            or coord[0] != tickets[0]["native_period"]
        ):
            raise ValueError(
                "complete original-ticket chain and prospective coordinate required"
            )
        first = tickets[0]["cancellation"]
        first_period = None if first is None else first["native_period"]
        if account["first_stranded_native_period"] != first_period:
            raise ValueError(
                "first cancellation must remain original across replacements"
            )
        deadline_index = account["native_reentry_deadline_exclusive"]
        if deadline_index is not None and (
            type(deadline_index) is not int
            or not 0 <= deadline_index < len(clock.boundaries)
        ):
            raise ValueError("explicit native deadline index outside supplied clock")
        attempts = tuple(account["reentry_attempts"])
        successes = {}
        seen_attempts = set()
        due = None
        local_checked = local_undated = 0
        for row in attempts:
            t = row["period"]
            if (
                row["original_record"] != root
                or type(t) is not int
                or not 0 <= t < K
                or t in seen_attempts
                or first_period is None
            ):
                raise ValueError(
                    "distinct genuine completed re-entry attempts required"
                )
            seen_attempts.add(t)
            if t < first_period or deadline_index is None or not t < deadline_index:
                raise ValueError(
                    "attempt precedes stranding or extends first native deadline"
                )
            names = ("attempt_utc", "first_stranded_utc", "reentry_deadline_utc")
            present = tuple(name in row for name in names)
            if any(present) and not all(present):
                raise ValueError(
                    "all three exposed native UTC timestamps required; no fill"
                )
            if all(present):
                attempt, first_time, expiry = (utc(row[name]) for name in names)
                if (
                    attempt != clock.at(t)
                    or first_time != clock.at(first_period)
                    or expiry <= first_time
                    or not attempt < expiry
                ):
                    raise ValueError(
                        "native attempt/first-stranding UTC disagrees with declared clock/deadline"
                    )
                if due is not None and due != expiry:
                    raise ValueError(
                        "repeated cancellation cannot extend the first UTC deadline"
                    )
                due = expiry
                end = next(
                    (
                        i
                        for i, instant in enumerate(clock.boundaries[:-1])
                        if instant >= due
                    ),
                    len(clock.boundaries) - 1,
                )
                if deadline_index != end:
                    raise ValueError(
                        "native exclusive index and exact UTC expiry disagree"
                    )
                local_checked += 1
            else:
                local_undated += 1
            rid = row["replacement_record"]
            if rid is not None:
                if type(rid) is not int or rid in successes:
                    raise ValueError(
                        "one successful native attempt per replacement ticket required"
                    )
                successes[rid] = row
        active = []
        spells = []
        for j, ticket in enumerate(tickets):
            rid, booked, cancel = (
                ticket["record_id"],
                ticket["native_period"],
                ticket["cancellation"],
            )
            if (
                type(rid) is not int
                or rid < 0
                or rid in ticket_ids
                or type(booked) is not int
                or not 0 <= booked < K
            ):
                raise ValueError(
                    "distinct completed native ticket IDs/book periods required"
                )
            ticket_ids.add(rid)
            if type(ticket["active_reservation"]) is not bool or ticket[
                "active_reservation"
            ] != (cancel is None):
                raise ValueError(
                    "complete disjoint current/cancelled ticket partition required"
                )
            if j:
                previous = tickets[j - 1]["cancellation"]
                if (
                    previous is None
                    or booked < previous["native_period"]
                    or rid not in successes
                    or successes[rid]["period"] != booked
                ):
                    raise ValueError(
                        "replacement requires previous cancellation and same-period native successful attempt"
                    )
            if cancel is None:
                active.append(rid)
                if j != len(tickets) - 1:
                    raise ValueError("active ticket cannot precede another replacement")
                continue
            c = cancel["native_period"]
            if type(c) is not int or not booked <= c <= K:
                raise ValueError(
                    "actual cancellation outside completed native ticket prefix"
                )
            nxt = None if j + 1 == len(tickets) else tickets[j + 1]
            end_period = K if nxt is None else nxt["native_period"]
            start, stop = clock.at(c), clock.at(end_period)
            duration = _hours(start, stop)
            spells.append(
                {
                    "cancelled_ticket": rid,
                    "cancellation_cause": cancel["cause"],
                    "start_native_period": c,
                    "start_utc": start.isoformat(),
                    "stop_native_period": end_period,
                    "stop_utc": stop.isoformat(),
                    "replacement_ticket": None if nxt is None else nxt["record_id"],
                    "status": "open_at_observed_prefix"
                    if nxt is None
                    else "closed_by_actual_native_reentry",
                    "declared_clock_elapsed_hours": _number(duration),
                    "exact_first_reentry_deadline_utc": None
                    if due is None
                    else due.isoformat(),
                    "extends_after_first_reentry_deadline": None
                    if due is None
                    else stop > due,
                    "continuous_waiting_or_delivered_outcome": None,
                }
            )
        if (
            set(successes) != {t["record_id"] for t in tickets[1:]}
            or len(active) > 1
            or account["current_active_record"] != (active[0] if active else None)
        ):
            raise ValueError(
                "full replacement success/current reservation support required"
            )
        total = sum(
            (_hours(utc(s["start_utc"]), utc(s["stop_utc"])) for s in spells),
            Fraction(),
        )
        closed = sum(
            (
                _hours(utc(s["start_utc"]), utc(s["stop_utc"]))
                for s in spells
                if s["replacement_ticket"] is not None
            ),
            Fraction(),
        )
        checked += local_checked
        undated += local_undated
        people.append(
            {
                "original_record": root,
                "potential_coordinate": coord,
                "segment": account["segment"],
                "request_market": account["request_market"],
                "spells": tuple(spells),
                "cancelled_ticket_spells": len(spells),
                "closed_spells": sum(
                    s["replacement_ticket"] is not None for s in spells
                ),
                "open_at_prefix": bool(
                    spells and spells[-1]["replacement_ticket"] is None
                ),
                "observed_unreserved_opportunity_hours": _number(total),
                "completed_spell_opportunity_hours": _number(closed),
                "open_spell_observed_hours": _number(total - closed),
                "native_utc_attempt_rows_checked": local_checked,
                "attempt_rows_without_native_UTC": local_undated,
                "exact_reentry_deadline_utc": None if due is None else due.isoformat(),
                "native_deadline_exclusive_index": deadline_index,
                "current_active_record": account["current_active_record"],
                "source_account": account,
                "full_lifecycle_waiting_hours": None,
                "lifecycle_consumer_surplus": None,
                "delivered_travel": None,
                "followup_after_prefix_unknown": True,
            }
        )
    if len(roots) != data["original_booked_lineages"] or ticket_ids != set(
        range(data["native_tickets_ever"])
    ):
        raise ValueError("every originally booked person/native ticket must remain")
    total = sum(
        (
            Fraction(
                int(p["observed_unreserved_opportunity_hours"]["numerator"]),
                int(p["observed_unreserved_opportunity_hours"]["denominator"]),
            )
            for p in people
        ),
        Fraction(),
    )
    return deep_freeze(
        {
            "plan_id": "A.10/A.5/B0/V9",
            "population": "Originally BOOKED ordinary people in supplied complete native accounts; outside choices retained as context only",
            "completed_native_periods": K,
            "observed_prefix_end_utc": now.isoformat(),
            "clock_UTC_boundaries": tuple(t.isoformat() for t in clock.boundaries),
            "original_booked_people": len(people),
            "outside_ordinary_choices_in_prefix": data[
                "ordinary_outside_choices_in_prefix"
            ],
            "persons": tuple(people),
            "observed_unreserved_opportunity_person_hours": _number(total),
            "mean_observed_hours_per_original_booked_person": None
            if not people
            else _number(total / len(people)),
            "native_utc_attempt_rows_checked": checked,
            "attempt_rows_without_native_UTC": undated,
            "clock_scope": "Supplied explicit opportunity clock; all exposed native attempt timestamps checked, caller declaration where none exposed",
            "input_accounts_current_canonical_sha256": spec_hash(data),
            "input_clock_current_canonical_sha256": spec_hash(
                tuple(t.isoformat() for t in clock.boundaries)
            ),
            "producer_original_capture_or_native_clock_authentication": False,
            "closing_boundary_states_reconstructed": False,
            "clock_interpolation_or_deadline_extension": False,
            "searches_tickets_spells_are_additional_people": False,
            "continuous_waiting_hours": None,
            "delivered_travel": None,
            "lifecycle_consumer_surplus": None,
            "denied_requests": None,
            "misallocation": None,
            "primary_default_empirical_production_adopted": False,
            "scientific_budget": None,
        }
    )


def unreserved_spells_record(result):
    """Owned data/clock/interval export; no saved-run or old-account loader."""
    return jsonable(result)
