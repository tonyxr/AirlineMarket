"""A.10/A.5/B0/V9: NEW Bernoulli-law to native single-boundary ledger map.

Declared unreserved BASE ledgers only. This is not a complete engine, dated
shock/re-entry kernel, delivered-service model or lifecycle welfare evaluator.
"""

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from airline_market.core.hashing import jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.core.types import PeriodPhase
from airline_market.sim.arrivals import Request
from airline_market.sim.bernoulli_person import (
    NativeBernoulliPersonLaw,
    bernoulli_person_record,
)
from airline_market.sim.ledger import book, inventory, resolve, sellable
from airline_market.sim.passenger_accounts import _encode
from airline_market.sim.payoff_logit import _ratio
from airline_market.sim.quotes import QuoteHistory
from airline_market.sim.state import Booking, CarrierState, WorldState
from airline_market.spec.compiled import CompiledMarket


@dataclass(frozen=True)
class DeclaredBaseLedger:
    installed: tuple
    tickets: tuple
    # Complete cash/refunds/penalties/canceled/revenue rows, in native carrier order.
    accounts: tuple
    next_record_id: int

    def __post_init__(self):
        object.__setattr__(self, "installed", tuple(self.installed))
        object.__setattr__(self, "tickets", tuple(self.tickets))
        object.__setattr__(self, "accounts", tuple(tuple(row) for row in self.accounts))


@dataclass(frozen=True)
class BernoulliBoundaryResult:
    expected_counts: object | None
    details: object


def resolve_bernoulli_boundary(
    compiled,
    law,
    ledger,
    *,
    installed_after,
    definition,
    unreserved_base_ledger,
    boundary_fixed_before_current_private_marks,
    branch_limit,
    ticket_limit,
):
    """Map ALL NEW supplied birth/choice branches through native book/resolve.

    Caller fixes one deterministic installed-seat-loss boundary before current
    marks. All existing tickets are original ordinary Bernoulli purchases;
    previous outside/cancelled people are not reconstructed or zero-filled.
    """
    if (
        type(compiled) is not CompiledMarket
        or type(law) is not NativeBernoulliPersonLaw
        or type(ledger) is not DeclaredBaseLedger
    ):
        raise TypeError(
            "native compiled model, NEW Bernoulli law and declared BASE ledger required"
        )
    if (
        unreserved_base_ledger is not True
        or boundary_fixed_before_current_private_marks is not True
        or not isinstance(definition, str)
        or not definition.strip()
    ):
        raise ValueError(
            "explicit unreserved BASE/fixed-before-private boundary definition required"
        )
    if any(type(n) is not int or n < 1 for n in (branch_limit, ticket_limit)):
        raise ValueError(
            "positive ALL-support branch/ticket infrastructure limits required"
        )
    raw = tuple(law.details["raw_joint"])
    if len(raw) > branch_limit or len(ledger.tickets) + 1 > ticket_limit:
        raise RuntimeError(
            "ALL branch/ticket support exceeds limit; no partial/fallback/retry"
        )
    if (
        compiled.spec.demand.mode != "single_request_bernoulli"
        or law.details["spec_hash"] != compiled.spec_hash
        or _encode(jsonable(compiled.spec))
        != _encode(jsonable(law.details["complete_native_model"]))
    ):
        raise ValueError("same complete native Bernoulli model required")
    expected = [(None, None, None, None)] + [
        (m, g, k, j)
        for m in range(len(compiled.spec.markets))
        for g, segment in enumerate(compiled.spec.segments)
        for k in range(len(segment.classes))
        for j in (*compiled.market_products[m], len(compiled.spec.products))
    ]
    if [tuple(row[:4]) for row in raw] != expected or any(
        type(v) is not int for row in raw[1:] for v in row[:4]
    ):
        raise ValueError(
            "complete ordered no-arrival/market/segment/class/choice support required"
        )
    if law.joint is not None:
        supplied = tuple((*row[:-1], _ratio(row[-1])) for row in law.joint)
        if supplied != raw or law.details["needed_underflows"]:
            raise ValueError("supplied public/raw joint or underflow contract differs")
    elif not law.details["needed_underflows"]:
        raise ValueError("unknown input law must retain its underflow diagnostics")
    c, period = compiled, law.details["period"]
    if type(period) is not int or not 0 <= period < c.spec.horizon.n_periods:
        raise ValueError("native request period required")
    prices = tuple(law.details["global_fixed_prices"])
    if len(prices) != len(c.spec.products) or any(
        type(v) is not float or not math.isfinite(v) for v in prices
    ):
        raise ValueError("complete native binary64 finite price context required")
    before, after = ledger.installed, tuple(installed_after)
    if (
        len(before) != len(c.spec.legs)
        or len(after) != len(before)
        or any(
            type(v) is not int or not 0 <= v <= np.iinfo(np.int64).max
            for v in (*before, *after)
        )
        or any(a > b for a, b in zip(after, before))
    ):
        raise ValueError(
            "complete integer installed capacities with nonincreasing boundary required"
        )
    if type(ledger.next_record_id) is not int or ledger.next_record_id < 0:
        raise ValueError("native next record ID required")
    if len(ledger.accounts) != len(c.spec.carriers):
        raise ValueError("ALL native carrier account rows required")
    for account in ledger.accounts:
        if (
            len(account) != 5
            or any(type(account[k]) is not float for k in (0, 1, 2, 4))
            or type(account[3]) is not int
            or account[3] < 0
        ):
            raise ValueError(
                "explicit native binary64 cash/refunds/penalties/revenue and integer canceled required"
            )
    ids, periods = set(), set()
    for b in ledger.tickets:
        if (
            type(b) is not Booking
            or any(
                type(getattr(b, k)) is not int
                for k in (
                    "record_id",
                    "carrier",
                    "product",
                    "period",
                    "arrival_rank",
                    "segment",
                    "taste_class",
                )
            )
            or type(b.fare) is not float
        ):
            raise ValueError("complete immutable native original Booking rows required")
        if (
            b.record_id in ids
            or not 0 <= b.record_id < ledger.next_record_id
            or b.period in periods
            or not 0 <= b.period < period
            or b.arrival_rank != 0
        ):
            raise ValueError(
                "distinct prior-period original Bernoulli tickets only; no re-entry/history inference"
            )
        if (
            not 0 <= b.product < len(c.spec.products)
            or b.carrier != int(c.product_carrier[b.product])
            or not 0 <= b.segment < len(c.spec.segments)
            or not 0 <= b.taste_class < len(c.spec.segments[b.segment].classes)
        ):
            raise ValueError("native ticket carrier/product/type support differs")
        ids.add(b.record_id)
        periods.add(b.period)
    carriers = []
    for i, account in enumerate(ledger.accounts):
        cash, refunds, penalties, canceled, revenue = account
        carriers.append(
            CarrierState(
                np.array([before[l] for l in c.legs_of[i]], dtype=np.int64),
                [b for b in ledger.tickets if b.carrier == i],
                cash,
                refunds,
                penalties,
                canceled,
                np.empty(0),
                np.empty(0),
                revenue=revenue,
            )
        )
    # These ledger primitives do not read pricing memories/quotes/background.
    # Empty adapter fields are NOT declarations of previous fares or a full world.
    base = WorldState(
        period,
        PeriodPhase.LEADERS_POSTED,
        law.details["exogenous_state"],
        carriers,
        [],
        QuoteHistory(),
        next_record_id=ledger.next_record_id,
    )
    free = tuple(
        int(
            inventory(base.carriers[int(c.leg_carrier[l])], c, int(c.leg_carrier[l]))[
                c.legs_of[int(c.leg_carrier[l])].index(l)
            ]
        )
        for l in range(len(before))
    )
    if any(v < 0 for v in free) or free != tuple(law.details["initial_net_free_stock"]):
        raise ValueError(
            "declared unreserved native inventory and law NET stock differ"
        )
    mask = sellable(np.asarray(free, dtype=np.int64), c.A_global)
    for m, g, k, j, target, ratio in raw:
        weight = Fraction(int(ratio["numerator"]), int(ratio["denominator"]))
        predicted = (
            free
            if j is None or j == len(c.spec.products)
            else tuple(x - int(a) for x, a in zip(free, c.A_global[:, j]))
            if mask[j]
            else None
        )
        if (
            weight < 0
            or (predicted is None and weight)
            or target != predicted
            or (target is not None and any(type(v) is not int for v in target))
        ):
            raise ValueError("complete branch feasibility/target/mass contract differs")
    affected = tuple(
        i
        for i in range(len(c.spec.carriers))
        if any(after[l] != before[l] for l in c.legs_of[i])
    )

    names = (
        "new_purchase_mass",
        "new_canceled_mass",
        "new_active_reservation_mass",
        "carried_canceled_expected_tickets",
        "all_canceled_expected_tickets",
    )
    sums = dict.fromkeys(names, Fraction())
    branches = []
    native_book_calls = native_resolve_calls = 0
    for index, (m, g, k, j, target_stock, ratio) in enumerate(raw):
        weight = Fraction(int(ratio["numerator"]), int(ratio["denominator"]))
        if weight < 0:
            raise ValueError("negative input joint mass")
        if target_stock is None:
            if weight:
                raise ValueError("inactive input row has positive mass")
            branches.append(
                {
                    "row": index,
                    "input": raw[index],
                    "executed": False,
                    "reason": "inactive alternative retained",
                    "transition": None,
                }
            )
            continue
        world = base.clone()
        new = None
        if m is not None and j != len(c.spec.products):
            if (
                not 0 <= m < len(c.spec.markets)
                or j not in c.market_products[m]
                or not 0 <= g < len(c.spec.segments)
                or not 0 <= k < len(c.spec.segments[g].classes)
            ):
                raise ValueError("native NEW request choice support differs")
            request = Request(0, m, g, k, 0, 0)
            new = book(
                world,
                int(c.product_carrier[j]),
                j,
                law.details["global_fixed_prices"][j],
                request,
                c,
            )
            native_book_calls += 1
        elif m is None and (g is not None or k is not None or j is not None):
            raise ValueError("no-arrival row must not create a person")
        actual = tuple(
            int(
                inventory(
                    world.carriers[int(c.leg_carrier[l])], c, int(c.leg_carrier[l])
                )[c.legs_of[int(c.leg_carrier[l])].index(l)]
            )
            for l in range(len(before))
        )
        if actual != tuple(target_stock):
            raise ValueError("native booking and supplied branch NET stock differ")
        for i, state in enumerate(world.carriers):
            state.capacity[:] = [after[l] for l in c.legs_of[i]]
        outcomes = tuple((i, resolve(world, i, c)) for i in affected)
        native_resolve_calls += len(outcomes)
        canceled = tuple(b for i, o in outcomes for b in o.canceled)
        kept = tuple(b for state in world.carriers for b in state.bookings)
        all_ids = ids | ({new.record_id} if new else set())
        if (
            {b.record_id for b in canceled} & {b.record_id for b in kept}
            or {b.record_id for b in (*canceled, *kept)} != all_ids
            or any(
                np.any(inventory(state, c, i) < 0)
                for i, state in enumerate(world.carriers)
            )
        ):
            raise RuntimeError(
                "native complete ticket partition/inventory differs; no partial report"
            )
        lost = new is not None and new.record_id in {b.record_id for b in canceled}
        measures = (
            int(new is not None),
            int(lost),
            int(new is not None and not lost),
            sum(b.record_id in ids for b in canceled),
            len(canceled),
        )
        for name, value in zip(names, measures):
            sums[name] += weight * value
        branches.append(
            {
                "row": index,
                "input": raw[index],
                "executed": True,
                "new_ticket": None if new is None else jsonable(new),
                "canceled": tuple(jsonable(b) for b in canceled),
                "kept": tuple(jsonable(b) for b in kept),
                "resolution_outcomes": tuple((i, jsonable(o)) for i, o in outcomes),
                "native_carrier_ledgers": tuple(
                    {
                        "carrier": i,
                        "capacity": tuple(int(v) for v in state.capacity),
                        "cash": state.cash,
                        "refunds": state.refunds,
                        "penalties": state.penalties,
                        "canceled": state.canceled,
                        "revenue": state.revenue,
                    }
                    for i, state in enumerate(world.carriers)
                ),
                "next_record_id": world.next_record_id,
                "native_period_unchanged": world.t,
                "raw_count_contributions": tuple(
                    (name, _ratio(weight * value))
                    for name, value in zip(names, measures)
                ),
            }
        )
    known = law.joint is not None
    return BernoulliBoundaryResult(
        deep_freeze(sums) if known else None,
        deep_freeze(
            {
                "plan_id": "A.10/A.5/B0/V9",
                "definition": definition,
                "complete_native_model": jsonable(c.spec),
                "supplied_new_birth_law": bernoulli_person_record(law),
                "declared_base_ledger": jsonable(ledger),
                "installed_after": after,
                "branches": tuple(branches),
                "raw_weighted_count_sums": tuple(
                    (name, _ratio(value)) for name, value in sums.items()
                ),
                "input_public_law_known": known,
                "native_book_or_resolve_on_owned_ledgers_executed": bool(
                    native_book_calls or native_resolve_calls
                ),
                "native_book_calls": native_book_calls,
                "native_resolve_calls": native_resolve_calls,
                "complete_engine_world_tape_or_native_period_advance_executed": False,
                "pricing_quote_background_adapter_fields_unused_not_source_observations": True,
                "seed_information_current_ledger_boundary_future_law_or_empirical_authentication": False,
                "lifecycle_CS_delivered_waiting_penalty_incidence_or_planner_adopted": False,
                "branch_limit": branch_limit,
                "ticket_limit": ticket_limit,
                "scientific_budget": None,
            }
        ),
    )


def bernoulli_boundary_record(result):
    if type(result) is not BernoulliBoundaryResult:
        raise TypeError("BernoulliBoundaryResult required")
    return _encode(jsonable(result.details))
