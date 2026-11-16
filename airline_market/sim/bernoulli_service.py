"""A.10/A.5/B0/V9/B4: conditional service on complete finite branch histories.

Separate direct-itinerary service model, not a genuine ordinary-person journal,
received-information law, native departure engine, financial or welfare model.
"""

import itertools
from dataclasses import dataclass, replace
from fractions import Fraction

from airline_market.policy.dated_reserve import utc
from airline_market.sim import source_future_transport as source_reader
from airline_market.sim.bernoulli_future import BernoulliFutureResult
from airline_market.sim.current_itinerary_service import LegOperation
from airline_market.sim.received_information import _text, _units
from airline_market.sim.shared_mark_attempt import _encoded, _owned
from airline_market.spec.compiled import CompiledMarket

RULE = "complete-branch-direct-service-programmes-v1"


def _check(value, message):
    if not value:
        raise ValueError(message)


class BernoulliServiceLimit(RuntimeError):
    """Complete structural input/output support exceeds limit; no partial result."""


@dataclass(frozen=True)
class BranchServiceClock:
    source_close: object
    leg_departures: tuple
    definition: str

    def __post_init__(self):
        object.__setattr__(self, "source_close", utc(self.source_close))
        object.__setattr__(
            self,
            "leg_departures",
            tuple((leg, utc(t)) for leg, t in self.leg_departures),
        )


@dataclass(frozen=True)
class BranchServiceProgramme:
    name: str
    probability: Fraction
    operations: tuple
    boarding_intents: tuple
    boarding_orders: tuple
    definition: str
    source_ref: str

    def __post_init__(self):
        _check(
            all(type(o) is LegOperation for o in self.operations),
            "typed complete leg operations required",
        )
        object.__setattr__(
            self, "operations", tuple(replace(o) for o in self.operations)
        )
        object.__setattr__(self, "boarding_intents", _owned(self.boarding_intents))
        object.__setattr__(self, "boarding_orders", _owned(self.boarding_orders))


@dataclass(frozen=True)
class BranchServiceCase:
    path: str
    branches: tuple
    active: tuple
    programmes: tuple

    def __post_init__(self):
        object.__setattr__(self, "branches", _owned(self.branches))
        object.__setattr__(self, "active", _owned(self.active))
        _check(
            all(type(p) is BranchServiceProgramme for p in self.programmes),
            "typed full service programmes required",
        )
        object.__setattr__(
            self, "programmes", tuple(replace(p) for p in self.programmes)
        )


@dataclass(frozen=True)
class BernoulliServiceResult:
    analyst: object

    def __post_init__(self):
        object.__setattr__(self, "analyst", _owned(self.analyst))


def _people(c, branches, active, start):
    people = []
    ticket = 0
    for offset, (m, g, k, j) in enumerate(branches):
        if m is None:
            continue
        purchase = j != len(c.spec.products)
        current = purchase and active[ticket]
        people.append(
            {
                "person_key": (start + offset, 0, m),
                "segment": g,
                "taste_class": k,
                "original_ticket": ticket if purchase else None,
                "original_product": j if purchase else None,
                "current_ticket": ticket if current else None,
                "current_leg": c.spec.products[j].legs[0] if current else None,
            }
        )
        ticket += int(purchase)
    return tuple(people)


def _prepare(c, clock, people, programme):
    for text in (programme.name, programme.definition, programme.source_ref):
        _text(text)
    legs = tuple(leg.id for leg in c.spec.legs)
    scheduled = dict(clock.leg_departures)
    ops = programme.operations
    _check(
        tuple(o.leg for o in ops) == legs,
        "ALL ordered native-leg operations including unused required",
    )
    tickets = {
        p["current_ticket"]: p for p in people if p["current_ticket"] is not None
    }
    targets = {(rid, p["current_leg"]) for rid, p in tickets.items()}
    intents = {}
    for rid, leg, flag in programme.boarding_intents:
        _check(
            type(rid) is int and type(flag) is bool and (rid, leg) not in intents,
            "distinct explicit Boolean ticket-leg intents required",
        )
        intents[rid, leg] = flag
    _check(
        set(intents) == targets,
        "ALL current ticket-leg intents required without extras",
    )
    orders = {}
    for leg, order in programme.boarding_orders:
        expected = {rid for rid, l in targets if l == leg}
        _check(
            leg in scheduled
            and leg not in orders
            and all(type(x) is int for x in order)
            and len(set(order)) == len(order)
            and set(order) == expected,
            "full priority for EVERY leg including no-shows and empty legs required",
        )
        orders[leg] = order
    _check(tuple(orders) == legs, "ALL ordered leg priorities required")
    events = []
    operations = []
    for o in ops:
        _check(
            type(o.seats) is int and o.seats >= 0,
            "explicit nonnegative integer service seats required",
        )
        if o.status == "operates":
            _check(
                o.departure is not None
                and o.arrival is not None
                and o.cancellation is None
                and o.departure >= scheduled[o.leg]
                and o.arrival > o.departure,
                "complete non-early operating departure and later arrival required",
            )
            events.extend(
                (
                    (o.departure.isoformat(), 2, o.leg, "departure"),
                    (o.arrival.isoformat(), 1, o.leg, "arrival"),
                )
            )
        elif o.status == "cancelled":
            _check(
                o.cancellation is not None
                and o.departure is None
                and o.arrival is None
                and o.cancellation > clock.source_close,
                "strictly post-source cancellation only required",
            )
            events.append((o.cancellation.isoformat(), 0, o.leg, "cancellation"))
        else:
            raise ValueError("explicit operating/cancelled status required")
        operations.append(
            {
                "leg": o.leg,
                "status": o.status,
                "seats": o.seats,
                "departure": None if o.departure is None else o.departure.isoformat(),
                "arrival": None if o.arrival is None else o.arrival.isoformat(),
                "cancellation": None
                if o.cancellation is None
                else o.cancellation.isoformat(),
            }
        )
    return _owned(
        {
            "people": people,
            "operations": operations,
            "intents": tuple((rid, leg, flag) for (rid, leg), flag in intents.items()),
            "orders": tuple(orders.items()),
            "events": tuple(sorted(events)),
            "definition": programme.definition,
            "source_ref": programme.source_ref,
        }
    )


def _report_shape(plan):
    people = plan["people"]
    keys = tuple(p["person_key"] for p in people)
    tickets = tuple(
        p["current_ticket"] for p in people if p["current_ticket"] is not None
    )
    legs = tuple(o["leg"] for o in plan["operations"])
    envelope = {
        "at": "UTC",
        "kind": "event",
        "leg": "leg",
        "all_person_keys": keys,
        "all_target_tickets": tickets,
        "decisions": tuple((rid, "reason") for rid in tickets),
        "physical_seats": 0,
        "boarded_count": 0,
        "unused_departure_seats": 0,
        "airborne": tuple((leg, tickets) for leg in legs),
        "delivered": tickets,
        "outcomes": tuple((rid, True) for rid in tickets),
    }
    return {
        "input_plan": plan,
        "steps": tuple(envelope for _ in plan["events"]),
        "current_service": tuple((p["person_key"], None) for p in people),
        "delivered_current_target_count": 0,
        "new_branch_service_state_machine_executed": True,
        "genuine_panel_journal_or_native_departure_engine_executed": False,
    }


def _execute(plan):
    """Execute on owned branch-service state only; no native ledger mutation."""
    people = plan["people"]
    keys = tuple(p["person_key"] for p in people)
    tickets = {
        p["current_ticket"]: p for p in people if p["current_ticket"] is not None
    }
    ops = {o["leg"]: o for o in plan["operations"]}
    orders = dict(plan["orders"])
    intents = {(rid, leg): flag for rid, leg, flag in plan["intents"]}
    airborne = {}
    delivered = set()
    outcomes = {}
    steps = []
    for at, _, leg, kind in plan["events"]:
        o = ops[leg]
        decisions = []
        boarded = None
        unused = None
        if kind == "cancellation":
            for rid in orders[leg]:
                outcomes[rid] = False
                decisions.append((rid, "operation_cancelled"))
        elif kind == "departure":
            free = o["seats"]
            manifest = []
            for rid in orders[leg]:
                if not intents[rid, leg]:
                    reason = "no_boarding_intent"
                    outcomes[rid] = False
                elif free == 0:
                    reason = "no_seat"
                    outcomes[rid] = False
                else:
                    reason = "boarded_not_delivered"
                    manifest.append(rid)
                    free -= 1
                decisions.append((rid, reason))
            airborne[leg] = tuple(manifest)
            boarded = len(manifest)
            unused = free
            if free + boarded != o["seats"]:
                raise RuntimeError("internal exact seat debit failed")
        else:
            for rid in airborne.pop(leg):
                delivered.add(rid)
                outcomes[rid] = True
                decisions.append((rid, "delivered_at_arrival"))
        steps.append(
            {
                "at": at,
                "kind": kind,
                "leg": leg,
                "all_person_keys": keys,
                "all_target_tickets": orders[leg],
                "decisions": tuple(decisions),
                "physical_seats": o["seats"],
                "boarded_count": boarded,
                "unused_departure_seats": unused,
                "airborne": tuple(sorted(airborne.items())),
                "delivered": tuple(sorted(delivered)),
                "outcomes": tuple(sorted(outcomes.items())),
            }
        )
    if set(outcomes) != set(tickets) or airborne:
        raise RuntimeError("internal complete current-ticket service partition failed")
    return _owned(
        {
            "input_plan": plan,
            "steps": steps,
            "current_service": tuple(
                (
                    p["person_key"],
                    outcomes[p["current_ticket"]]
                    if p["current_ticket"] is not None
                    else None,
                )
                for p in people
            ),
            "delivered_current_target_count": len(delivered),
            "new_branch_service_state_machine_executed": True,
            "genuine_panel_journal_or_native_departure_engine_executed": False,
        }
    )


def _preflight(c, source, clock, programme_names, cases, support_limit, definition):
    _check(
        type(c) is CompiledMarket
        and type(source) is BernoulliFutureResult
        and type(clock) is BranchServiceClock,
        "NEW complete native Bernoulli source and typed service clock required",
    )
    _check(
        type(support_limit) is int and support_limit > 0,
        "positive structural support limit required",
    )
    _check(
        all(seg.mixed is None and seg.classes for seg in c.spec.segments),
        "finite native-class declarations only; mixed tastes unsupported",
    )
    _text(definition)
    _text(clock.definition)
    _check(
        all(len(p.legs) == 1 for p in c.spec.products),
        "direct single-required-leg products only; no connection fallback",
    )
    names = _owned(programme_names)
    _check(
        bool(names) and len(set(names)) == len(names),
        "complete distinct named programmes required",
    )
    for name in names:
        _text(name)
    _check(
        tuple(leg for leg, t in clock.leg_departures)
        == tuple(leg.id for leg in c.spec.legs)
        and all(t > clock.source_close for leg, t in clock.leg_departures),
        "full ordered scheduled departures strictly after declared source close required",
    )
    _check(
        all(type(case) is BranchServiceCase for case in cases),
        "typed full prospective cases required",
    )
    cases = tuple(replace(case) for case in cases)
    record = _owned(_encoded(source))
    base = {
        "plan_id": "A.10/A.5/B0/V9/B4",
        "rule": RULE,
        "complete_native_model": _encoded(c.spec),
        "complete_source": record,
        "clock": {
            "source_close": clock.source_close.isoformat(),
            "leg_departures": tuple(
                (l, t.isoformat()) for l, t in clock.leg_departures
            ),
            "definition": clock.definition,
        },
        "programme_names": names,
        "full_prospective_cases": _encoded(cases),
        "definition": definition,
        "support_limit": support_limit,
    }
    input_units = _units(base)
    if input_units > support_limit:
        raise BernoulliServiceLimit(
            "complete input support exceeds limit before source traversal/transitions"
        )
    profiles, known, total = source_reader._source(c, record)
    d = record["details"]
    paths = d["paths"]
    T = len(paths[0]["opportunities"])
    start = d["start_period"]
    catalogue = ((None, None, None, None),) + tuple(
        (m, g, k, int(j))
        for m in range(len(c.spec.markets))
        for g, s in enumerate(c.spec.segments)
        for k in range(len(s.classes))
        for j in (*c.market_products[m], len(c.spec.products))
    )
    purchase = sum(b[0] is not None and b[3] != len(c.spec.products) for b in catalogue)
    expected = len(paths)
    for _ in range(T):
        expected *= len(catalogue) + purchase
        if expected > support_limit:
            raise BernoulliServiceLimit("ALL syntactic/status support exceeds limit")
    _check(
        len(cases) == expected,
        "ALL prospective syntactic/status cases including unused and zero support required",
    )
    prepared = {}
    index = 0
    plan_units = 0
    for path in paths:
        for branches in itertools.product(catalogue, repeat=T):
            purchases = sum(
                b[0] is not None and b[3] != len(c.spec.products) for b in branches
            )
            for active in itertools.product((False, True), repeat=purchases):
                case = cases[index]
                index += 1
                coordinate = (path["name"], branches, active)
                _check(
                    (case.path, case.branches, case.active) == coordinate
                    and all(type(x) is bool for x in case.active),
                    "complete ordered prospective history/status coordinates required",
                )
                _check(
                    tuple(p.name for p in case.programmes) == names,
                    "ALL same ordered programme names required",
                )
                source_reader._distribution(
                    tuple(p.probability for p in case.programmes)
                )
                people = _people(c, branches, active, start)
                plans = tuple(_prepare(c, clock, people, p) for p in case.programmes)
                plan_units += sum(_units(p) for p in plans)
                if input_units + plan_units > support_limit:
                    raise BernoulliServiceLimit(
                        "complete prepared support exceeds limit before transitions"
                    )
                prepared[coordinate] = (case, people, plans)
    actual = []
    shapes = []
    joint = []
    used = set()
    for hi, (terminal, persons) in enumerate(zip(d["terminals"], profiles)):
        branches = tuple(
            tuple(d["nodes"][ni]["boundary"]["branches"][ri]["input"][:4])
            for ni, ri in terminal["history"]
        )
        purchases = sum(
            b[0] is not None and b[3] != len(c.spec.products) for b in branches
        )
        current = {
            p["current_ticket"] for p in persons if p["current_ticket"] is not None
        }
        active = tuple(rid in current for rid in range(purchases))
        coordinate = (terminal["path"], branches, active)
        case, people, plans = prepared[coordinate]
        used.add(coordinate)
        _check(
            len(people) == len(persons)
            and all(
                (
                    p["person_key"],
                    p["segment"],
                    p["taste_class"],
                    p["original_ticket"],
                    p["current_ticket"],
                )
                == (
                    x["person_key"],
                    x["segment"],
                    x["taste_class"],
                    x["original_ticket"],
                    x["current_ticket"],
                )
                and (
                    p["original_product"] is None
                    if x["original_source_ticket"] is None
                    else p["original_product"] == x["original_source_ticket"]["product"]
                )
                for p, x in zip(people, persons)
            ),
            "full actual source-person/current-ticket binding differs",
        )
        mass = source_reader._q(terminal["raw_mass"])
        for programme, plan in zip(case.programmes, plans):
            common = {
                "history_index": hi,
                "path": terminal["path"],
                "history": terminal["history"],
                "programme": programme.name,
                "source_mass": mass,
                "programme_probability": programme.probability,
                "raw_joint_mass": mass * programme.probability,
            }
            actual.append((common, plan))
            shape = _report_shape(plan)
            shapes.append({**common, "service": shape})
            joint.append({**common, "current_service": shape["current_service"]})
    output = {
        **base,
        "all_original_person_profiles": profiles,
        "all_atoms": tuple(shapes),
        "public_joint": tuple(joint) if known else None,
        "source_public_joint_known": known,
        "raw_source_measure": total,
        "raw_source_residual": total - 1,
        "raw_joint_measure": total,
        "raw_weighted_delivered_current_count_diagnostic": Fraction(),
        "delivered_current_count_integral": Fraction() if known else None,
        "unused_case_coordinates": tuple(k for k in prepared if k not in used),
        "all_support_bound": 0,
        "native_source_probability_book_resolve_actor_tape_RNG_or_old_service_executor_called": False,
        "source_time_probability_reception_empirical_service_truth_authenticated": False,
        "financial_payment_material_utility_CS_denial_misallocation": None,
        "primary_default_empirical_production_planner_adopted": False,
        "scientific_budget": None,
    }
    required = input_units + plan_units + _units(_encoded(output))
    if required > support_limit:
        raise BernoulliServiceLimit(
            "complete proposed output support exceeds limit before transitions"
        )
    output["all_support_bound"] = required
    return output, tuple(actual)


def bernoulli_service_programmes(
    compiled, source, clock, programme_names, cases, *, definition, support_limit
):
    """Execute ALL actual histories after validating ALL prospective declarations.

    Source and future laws remain declarations. Counts cover current-ticket
    targets only; outside/unreserved current-service statuses remain unknown.
    """
    output, actual = _preflight(
        compiled, source, clock, programme_names, cases, support_limit, definition
    )
    atoms = []
    joint = []
    measure = Fraction()
    delivered = Fraction()
    for common, plan in actual:
        report = _execute(plan)
        atom = {**common, "service": report}
        atoms.append(atom)
        joint.append({**common, "current_service": report["current_service"]})
        measure += common["raw_joint_mass"]
        delivered += common["raw_joint_mass"] * report["delivered_current_target_count"]
    _check(
        measure == output["raw_source_measure"],
        "internal complete raw programme mass differs",
    )
    output.update(
        all_atoms=tuple(atoms),
        public_joint=tuple(joint) if output["source_public_joint_known"] else None,
        raw_joint_measure=measure,
        raw_weighted_delivered_current_count_diagnostic=delivered,
        delivered_current_count_integral=delivered
        if output["source_public_joint_known"]
        else None,
    )
    return BernoulliServiceResult(_encoded(output))


def bernoulli_service_record(result):
    _check(
        type(result) is BernoulliServiceResult,
        "typed complete branch-service result required",
    )
    return _encoded(result.analyst)
