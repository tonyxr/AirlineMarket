"""A.8/B0/V9: current-period native ordinary pool and hold-aware capacity.

Inputs are captured after caller-owned native posting and re-entry. This adapter
constructs a static pool, not a booking/period executor or cap-binding detector.
Applicant key rank is the original potential tape slot (gumbel_row), while native
admission rank is retained separately and may change after thinning.
Current tape marks are read only; no new random draws or actor calls are made.
"""

import hashlib
import math
import pickle
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from airline_market.core.types import PeriodPhase
from airline_market.policy.rationing import Applicant, Offer, from_compiled
from airline_market.sim.arrivals import period_requests
from airline_market.sim.choice import sampled_taste, systematic_utilities
from airline_market.sim.ledger import inventory


def native_market_problem(compiled, period, free_capacity, applicants):
    """Validate native request-O&D/offer correspondence, then fixed incidence."""
    people = tuple(applicants)
    for person in people:
        if type(person) is not Applicant:
            raise TypeError("explicit Applicant required")
        market = person.key[2]
        if market >= len(compiled.spec.markets):
            raise ValueError("unknown native request market")
        allowed = {
            compiled.spec.products[j].id for j in compiled.market_products[market]
        }
        if any(o.product not in allowed for o in person.offers):
            raise ValueError("offer product does not match native request O&D")
    return from_compiled(compiled, period, free_capacity, people)


def _token(engine):
    t = engine.world.t
    tape = engine.tape
    current_marks = (
        tape.seed,
        tape.request_uniforms[t].tolist(),
        tape.class_uniform[t].tolist(),
        tape.arrival_marks[t],
        tape.gumbel[t].tolist(),
        tape.mixed_draws[t].tolist(),
    )
    trigger = getattr(engine, "trigger", None)
    mutable = (
        engine.compiled.spec,
        engine.world,
        tuple(
            getattr(engine, name, None)
            for name in (
                "cap_policy",
                "booking_clock",
                "leg_departures",
                "dated_rebooking",
                "dated_menu",
                "dated_reserve",
                "rebooking",
                "seat_reserve",
                "_reserve_eligible",
                "presented_itineraries",
                "_presented",
                "_counter_visits",
            )
        ),
        getattr(engine, "_reserve_ledger", None),
        getattr(engine, "_dated_released", None),
        getattr(engine, "reserve_events", None),
        getattr(engine, "_passengers", None),
        getattr(engine, "_attempts", None),
        getattr(engine, "rebooking_history", None),
        getattr(engine, "_regulator_reports", None),
        None
        if trigger is None
        else (trigger.rule, trigger.benchmark, trigger.active_until, trigger.history),
        current_marks,
    )
    return hashlib.sha256(pickle.dumps(mutable, protocol=4)).hexdigest()


@dataclass(frozen=True)
class NativeOrdinaryPool:
    problem: object
    requests: tuple
    posted_prices: tuple
    active_cap_limits: tuple
    score_definition: str
    state_token: str
    input_seed: int
    binding_cap: object = None
    native_booking_adopted: bool = False


def _active_caps(engine):
    limits = {}
    policy = getattr(engine, "cap_policy", None)
    if policy is not None:
        limits.update(policy.limits(engine.world.t))
    trigger = getattr(engine, "trigger", None)
    if trigger is not None:
        for row in trigger.history:
            if (
                row["period"] == engine.world.t
                and row["active"]
                and row["cap"] is not None
            ):
                p, cap = row["product"], float(row["cap"])
                limits[p] = min(limits.get(p, cap), cap)
    return limits


def native_ordinary_pool(engine, *, posted_prices, allocation_scores, score_definition):
    """Capture all actual ordinary requests for a caller-prepared posting.

    Pass the existing native post_actions return AFTER native re-entry attempts.
    The supplied vector is checked against current prices. Full pricing protocol
    ownership remains with that caller; the adapter never invokes controllers.
    Existing hold synchronization is invoked, normally idempotent after posting.
    Objective scores must explicitly cover every native market-product alternative
    for every actual request, including outside-preferring/stock-blocked people.
    """
    c, world = engine.compiled, engine.world
    if world is None or world.t >= c.spec.horizon.n_periods:
        raise ValueError("initialized nonterminal native opportunity required")
    if (
        world.phase == PeriodPhase.PRE_DECISION
        or world.t not in world.quote_history.records
    ):
        raise ValueError("caller must complete native posting before pool capture")
    if not isinstance(score_definition, str) or not score_definition.strip():
        raise ValueError("explicit allocation-score definition required")
    if any(r["period"] == world.t for r in world.request_log):
        raise ValueError("ordinary arrivals already admitted in this period")
    if getattr(engine, "_reserve_eligible", False):
        raise ValueError("ordinary pool cannot use stranded held-seat override")
    for root, p in getattr(engine, "_passengers", {}).items():
        if (
            p.active_record is None
            and world.t < p.deadline
            and (root, world.t) not in engine._attempts
        ):
            raise ValueError("complete required native re-entry attempts first")
    actual = tuple(float(x) for x in engine._prices_global())
    supplied = tuple(float(x) for x in posted_prices)
    if (
        len(supplied) != len(c.spec.products)
        or supplied != actual
        or any(not math.isfinite(x) for x in actual)
    ):
        raise ValueError("supplied posting differs from current native price vector")
    limits = _active_caps(engine)
    if any(actual[c.product_index[p]] > cap for p, cap in limits.items()):
        raise ValueError("posted price exceeds active native ceiling")
    if hasattr(engine, "_sync_reserve"):
        engine._sync_reserve()
    held = {k: row["held"] for k, row in getattr(engine, "_reserve_ledger", {}).items()}
    free = {
        c.spec.legs[l].id: int(n) - held.get(c.spec.legs[l].id, 0)
        for i, state in enumerate(world.carriers)
        for l, n in zip(c.legs_of[i], inventory(state, c, i))
    }
    if any(n < 0 for n in free.values()):
        raise ValueError("held stock exceeds native remaining capacity")
    requests = tuple(period_requests(c, engine.tape, world.t, world.exo))
    required = {
        ((world.t, r.gumbel_row, r.market), c.spec.products[j].id)
        for r in requests
        for j in c.market_products[r.market]
    }
    if set(allocation_scores) != required:
        raise ValueError(
            "allocation scores must cover exactly the full native request/market-product pool"
        )
    people = []
    for r in requests:
        taste = sampled_taste(
            c, r.segment, r.taste_class, engine.tape.mixed_draws[world.t, r.mixed_row]
        )
        u = systematic_utilities(
            c, r.market, taste, actual, np.ones(len(c.spec.products), dtype=bool)
        )
        columns = (*c.market_products[r.market], len(c.spec.products))
        total = u + engine.tape.gumbel[world.t, r.gumbel_row, list(columns)]
        if not np.isfinite(total).all():
            raise ValueError("finite current-period native choice marks required")
        # Exact differences of native Float64 sums preserve native argmax ordering
        # and ties without cancellation/division rounding. Utility units, not CS.
        outside = Fraction.from_float(float(total[-1]))
        key = (world.t, r.gumbel_row, r.market)
        offers = tuple(
            Offer(
                c.spec.products[j].id,
                Fraction.from_float(float(total[k])) - outside,
                allocation_scores[(key, c.spec.products[j].id)],
            )
            for k, j in enumerate(c.market_products[r.market])
        )
        people.append(Applicant(key, offers))
    problem = native_market_problem(c, world.t, free, people)
    return NativeOrdinaryPool(
        problem,
        requests,
        actual,
        tuple(sorted(limits.items())),
        score_definition,
        _token(engine),
        engine.tape.seed,
    )


def assert_pool_current(engine, pool):
    """Refuse stale price, ledger, proof, current-mark or period snapshots."""
    if type(pool) is not NativeOrdinaryPool:
        raise TypeError("NativeOrdinaryPool required")
    if (
        engine.world is None
        or engine.world.t != pool.problem.period
        or _token(engine) != pool.state_token
    ):
        raise ValueError("native pool is stale; no booking commitment supplied")


def native_pool_record(pool):
    """Analyst JSON snapshot of prepared inputs, never a controller observation."""
    if type(pool) is not NativeOrdinaryPool:
        raise TypeError("NativeOrdinaryPool required")

    def fraction(x):
        return {"numerator": str(x.numerator), "denominator": str(x.denominator)}

    return {
        "plan_id": "A.8/B0/V9",
        "period": pool.problem.period,
        "input_seed": pool.input_seed,
        "state_token": pool.state_token,
        "score_definition": pool.score_definition,
        "posted_prices": pool.posted_prices,
        "active_cap_limits": pool.active_cap_limits,
        "free_capacity": pool.problem.capacities,
        "requests": tuple(
            (r.rank, r.market, r.segment, r.taste_class, r.gumbel_row, r.mixed_row)
            for r in pool.requests
        ),
        "applicants": tuple(
            {
                "key": a.key,
                "offers": tuple(
                    {
                        "product": o.product,
                        "choice_gain_native_utility": fraction(o.choice_value),
                        "allocation_score": fraction(o.allocation_value),
                    }
                    for o in a.offers
                ),
            }
            for a in pool.problem.applicants
        ),
        "binding_cap": None,
        "native_booking_adopted": False,
        "scope": "Prepared current-period native input snapshot; active ceilings are not proof of cap binding; no dynamic welfare/equilibrium value",
    }
