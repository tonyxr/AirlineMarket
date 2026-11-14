"""A.10/A.5/B0/V9: read-only original-booked-passenger ticket/fare accounts.

All tickets and re-entry searches join to prospective potential-lineage roots.
Reservations/payments are not delivered travel, lifecycle CS or denied demand.
"""

import math
import struct
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from airline_market.core.hashing import jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.core.types import PeriodPhase
from airline_market.policy.potential_lineage import PotentialLineageMarketEngine


def _number(value):
    if value is None:
        return None
    return {"numerator": str(value.numerator), "denominator": str(value.denominator)}


def _exact(value):
    value = float(value)
    return Fraction.from_float(value) if math.isfinite(value) else None


def _binary(value):
    value = float(value)
    return {
        "binary64_bits": struct.pack(">d", value).hex(),
        "float_hex": value.hex(),
        "repr": repr(value),
        "exact": _number(_exact(value)),
    }


def _encode(value):
    if isinstance(value, Mapping):
        return {k: _encode(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_encode(v) for v in value)
    if isinstance(value, float):
        return _binary(value)
    return value


def _sum(values):
    values = tuple(_exact(x) for x in values)
    return None if any(x is None for x in values) else sum(values, Fraction(0))


def _subtract(a, b):
    return None if a is None or b is None else a - b


@dataclass(frozen=True)
class OriginalPassengerAccounts:
    data: object


def original_passenger_accounts(engine):
    """Join complete current native ledgers without replaying any physical path.

    Population: originally BOOKED ordinary passengers in this native prefix.
    Outside requests are not reconstructed as ticketed people. A successful
    producer/prefix is required; structural checks do not authenticate its source.
    Exact sums refer to serialized binary64 ticket fares, not economic expectations
    or the native rounded accumulation order. Native residuals are never repaired.
    """
    if not isinstance(engine, PotentialLineageMarketEngine):
        raise TypeError(
            "prospective potential-lineage engine required; no old-ticket reconstruction"
        )
    c, w = engine.compiled, engine.world
    if w is None or w.phase not in {PeriodPhase.PRE_DECISION, PeriodPhase.TERMINAL}:
        raise ValueError(
            "completed native prefix required; no partial-posting accounts"
        )
    trace = engine._trace
    if (
        len(trace) != w.t
        or tuple(r["period"] for r in trace) != tuple(range(w.t))
        or engine._opening_period is not None
        or engine._pending_events
        or any(
            getattr(engine, name, None) is not None
            for name in ("_pending_order", "_pending_allocation_pool")
        )
    ):
        raise ValueError(
            "complete successful-prefix structure required; no failed-run salvage"
        )
    active = {}
    for i, cs in enumerate(w.carriers):
        for b in cs.bookings:
            if b.record_id in active or b.carrier != i:
                raise ValueError(
                    "distinct native active tickets/carrier correspondence required"
                )
            active[b.record_id] = b
    cancelled = {}
    for row in engine.stranded:
        b = row.booking
        if (
            b.record_id in cancelled
            or b.record_id in active
            or not b.period <= row.canceled_at <= w.t
        ):
            raise ValueError(
                "one native cancellation per ticket, disjoint from active tickets required"
            )
        cancelled[b.record_id] = row
    tickets = {**active, **{k: r.booking for k, r in cancelled.items()}}
    if set(tickets) != set(range(w.next_record_id)):
        raise ValueError("full active plus cancelled native ticket history required")
    purchases = [b for r in trace for b in r["purchases"]]
    if len(purchases) != len(tickets) or {b.record_id for b in purchases} != set(
        tickets
    ):
        raise ValueError("every ticket must have exactly one completed native purchase")
    for b in purchases:
        if b != tickets[b.record_id]:
            raise ValueError("native purchase and active/cancelled ticket disagree")
    coordinates = engine._physical_coordinates
    replacement = engine._replacement_roots
    roots = set(tickets) - set(replacement)
    if (
        set(coordinates) != roots
        or set(replacement) - set(tickets)
        or set(replacement.values()) - roots
    ):
        raise ValueError(
            "complete original potential coordinates and direct replacement roots required; no fallback"
        )
    if len({x.stream_key for x in coordinates.values()}) != len(roots):
        raise ValueError("distinct original physical request coordinates required")
    by_root = {k: [] for k in roots}
    for k, b in sorted(tickets.items()):
        root = replacement.get(k, k)
        original = tickets[root]
        coord = coordinates[root]
        market = int(c.product_market[b.product])
        if (
            coord.period,
            coord.native_rank,
            coord.segment,
            coord.taste_class,
            coord.market,
        ) != (
            original.period,
            original.arrival_rank,
            original.segment,
            original.taste_class,
            int(c.product_market[original.product]),
        ):
            raise ValueError(
                "original booking and potential-coordinate sidecar disagree"
            )
        if (b.segment, b.taste_class, market) != (
            original.segment,
            original.taste_class,
            coord.market,
        ) or (
            k != root
            and (b.arrival_rank != -1 - root or k <= root or b.period < original.period)
        ):
            raise ValueError(
                "replacement must retain original native lineage/market/class"
            )
        by_root[root].append(b)
    histories = {k: [] for k in roots}
    successes = set()
    attempts = set()
    for row in engine.rebooking_history:
        root, period = row["original_record"], row["period"]
        if (
            root not in roots
            or root not in cancelled
            or not 0 <= period < w.t
            or (root, period) in attempts
        ):
            raise ValueError("distinct completed genuine re-entry attempts required")
        attempts.add((root, period))
        k = row["replacement_record"]
        if k is not None:
            if k in successes or replacement.get(k) != root:
                raise ValueError(
                    "each replacement ticket requires one matching successful attempt"
                )
            b = tickets[k]
            if (b.period, b.product) != (period, row["chosen"]) or struct.pack(
                ">d", b.fare
            ) != struct.pack(">d", row["fare"]):
                raise ValueError(
                    "successful attempt and replacement native ticket disagree"
                )
            successes.add(k)
        elif row["chosen"] is not None or row["fare"] is not None:
            raise ValueError("unsuccessful attempt cannot invent a purchase/fare")
        histories[root].append(row)
    if successes != set(replacement):
        raise ValueError(
            "full successful re-entry history required, no missing replacement"
        )
    rows = []
    for root, chain in sorted(by_root.items()):
        current = [b.record_id for b in chain if b.record_id in active]
        if len(current) > 1:
            raise ValueError(
                "at most one active native reservation per original passenger"
            )
        queued = engine._passengers.get(root)
        if queued is not None and (
            queued.original != tickets[root]
            or queued.active_record != (current[0] if current else None)
            or root not in cancelled
            or queued.first_stranded_at != cancelled[root].canceled_at
        ):
            raise ValueError("re-entry queue and actual lineage ledger disagree")
        gross = _sum(b.fare for b in chain)
        refund = _sum(b.fare for b in chain if b.record_id in cancelled)
        rows.append(
            {
                "original_record": root,
                "potential_coordinate": coordinates[root].stream_key,
                "native_original_rank": coordinates[root].native_rank,
                "segment": c.spec.segments[chain[0].segment].id,
                "request_market": c.spec.markets[coordinates[root].market].id,
                "tickets": tuple(
                    {
                        "record_id": b.record_id,
                        "carrier": c.spec.carriers[b.carrier].code,
                        "product": c.spec.products[b.product].id,
                        "native_period": b.period,
                        "native_arrival_rank": b.arrival_rank,
                        "fare": _binary(b.fare),
                        "active_reservation": b.record_id in active,
                        "cancellation": None
                        if b.record_id not in cancelled
                        else {
                            "native_period": cancelled[b.record_id].canceled_at,
                            "cause": cancelled[b.record_id].cause,
                            "fare_refunded": _binary(b.fare),
                        },
                    }
                    for b in chain
                ),
                "current_active_record": current[0] if current else None,
                "current_reservation_state": "active_original"
                if current == [root]
                else "active_replacement"
                if current
                else "no_active_reservation",
                "reentry_queue_present": queued is not None,
                "first_stranded_native_period": None
                if root not in cancelled
                else cancelled[root].canceled_at,
                "native_reentry_deadline_exclusive": None
                if queued is None
                else queued.deadline,
                "reentry_attempts": _encode(histories[root]),
                "gross_ticket_fares": _number(gross),
                "fare_refunds": _number(refund),
                "net_ticket_fare_paid": _number(_subtract(gross, refund)),
                "carrier_penalties_allocated_to_passenger": None,
                "delivered_service_or_lifecycle_utility": None,
            }
        )
    carriers = []
    for i, state in enumerate(w.carriers):
        gross = _sum(b.fare for b in tickets.values() if b.carrier == i)
        refunds = _sum(
            r.booking.fare for r in cancelled.values() if r.booking.carrier == i
        )
        net = _subtract(gross, refunds)
        penalties = _exact(state.penalties)
        expected_cash = _subtract(net, penalties)
        carriers.append(
            {
                "carrier": c.spec.carriers[i].code,
                "ticket_fares_exact": _number(gross),
                "cancelled_ticket_fares_exact": _number(refunds),
                "active_ticket_fares_exact": _number(
                    _sum(b.fare for b in active.values() if b.carrier == i)
                ),
                "native_accumulators": {
                    k: _binary(getattr(state, k))
                    for k in ("revenue", "refunds", "penalties", "cash")
                },
                "revenue_arithmetic_residual": _number(
                    _subtract(_exact(state.revenue), gross)
                ),
                "refund_arithmetic_residual": _number(
                    _subtract(_exact(state.refunds), refunds)
                ),
                "cash_arithmetic_residual": _number(
                    _subtract(_exact(state.cash), expected_cash)
                ),
                "cancelled_tickets": sum(
                    r.booking.carrier == i for r in cancelled.values()
                ),
                "native_cancelled_counter": state.canceled,
                "cancelled_counter_residual": state.canceled
                - sum(r.booking.carrier == i for r in cancelled.values()),
            }
        )
    return OriginalPassengerAccounts(
        deep_freeze(
            {
                "plan_id": "A.10/A.5/B0/V9",
                "population": "Originally booked ordinary native passengers only; outside choices are not ticketed lineages",
                "completed_native_periods": w.t,
                "original_booked_lineages": len(roots),
                "native_tickets_ever": len(tickets),
                "replacement_tickets_ever": len(replacement),
                "reentry_search_opportunities": len(attempts),
                "searches_are_additional_people": False,
                "ordinary_outside_choices_in_prefix": sum(
                    r["chosen"] is None for r in w.request_log
                ),
                "reservation_state_counts": dict(
                    Counter(r["current_reservation_state"] for r in rows)
                ),
                "lineages": tuple(rows),
                "carriers": tuple(carriers),
                "penalty_allocation_to_passengers_adopted": False,
                "native_arithmetic_repaired_or_tolerance_applied": False,
                "delivered_travel_authenticated": False,
                "producer_or_original_source_time_authenticated": False,
                "stranded_lifecycle_consumer_surplus": None,
                "total_lifecycle_surplus": None,
                "denied_requests": None,
                "misallocation": None,
                "primary_B4_welfare_adopted": False,
                "scientific_budget": None,
            }
        )
    )


def passenger_accounts_record(row: OriginalPassengerAccounts):
    """Owned lossless JSON export; never an actor observation or account repair."""
    if type(row) is not OriginalPassengerAccounts:
        raise TypeError("OriginalPassengerAccounts required")
    return jsonable(row.data)
