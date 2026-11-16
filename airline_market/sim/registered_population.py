"""A.10/A.5/B0/V9/B4: owned prospective population programmes.

Restricted baseline chronological source declarations and exact current-ticket
attachments. Content/order are local declarations, not external preregistration,
received knowledge, native probability, preference or lifecycle CS evidence.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from datetime import datetime
from fractions import Fraction

from airline_market.core.money import round_fare
from airline_market.policy.dated_demand import compile_dated_demand
from airline_market.policy.dated_reserve import UtcBookingClock, utc
from airline_market.sim import conditional_population as population
from airline_market.sim import current_itinerary_service as service
from airline_market.sim import person_material_payoff as material
from airline_market.sim import service_ticket_payments as payments
from airline_market.sim.payoff_logit import ContingentPayoff
from airline_market.sim.received_information import _text, _units
from airline_market.sim.service_observations import ServiceObservationJournal
from airline_market.sim.shared_mark_attempt import _encoded, _owned
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.io import dump_spec, load_spec

RULE = "owned-prospective-population-programme-v1"
ENGINE = "airline_market.sim.ordinary_cohort.OrdinaryCohortMarketEngine"
BASE_CONTROLS = _owned(
    {
        "ordinary_priority": None,
        "trigger_rule": None,
        "mechanical_benchmark": None,
        "dated_rebooking": None,
        "dated_menu": None,
        "dated_reserve": None,
        "cap_policy": {"windows": (), "declared_emergencies": ()},
        "check_invariants": True,
    }
)
_TYPES = tuple(
    getattr(population, n)
    for n in (
        "PhysicalServiceProgramme",
        "ServiceOutcomeTerms",
        "PersonServiceTerms",
        "PopulationPayoffCell",
        "OriginalPersonInformation",
        "ConditionalPopulationProgramme",
        "ConditionalPopulationLaw",
    )
) + (
    service.LegOperation,
    payments.AdditionalTicketTransfer,
    material.DeclaredDecisionInformation,
    ContingentPayoff,
)


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(
        _encoded(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _same(a, b):
    return _canonical(a) == _canonical(b)


def _clone(value):
    """Whitelist typed declarations; never retain caller-owned mutable fields."""
    if type(value) in _TYPES:
        return type(value)(
            **{f.name: _clone(getattr(value, f.name)) for f in fields(value)}
        )
    if is_dataclass(value):
        raise TypeError("unsupported declaration object")
    if isinstance(value, Mapping):
        return _owned({k: _clone(v) for k, v in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_clone(v) for v in value)
    if type(value) is datetime:
        return utc(value)
    if value is None or type(value) in (str, bool, int, float, Fraction):
        return value
    raise TypeError("owned primitive or whitelisted typed declaration required")


@dataclass(frozen=True)
class PrimitiveTapePlan:
    seed: int
    registry_root: int
    namespace: str
    arrival_marks: tuple
    gumbel_fill: float
    outside_default: float
    gumbel_overrides: tuple
    minimum_slots: int
    generated_row_limit: int

    def __post_init__(self):
        object.__setattr__(self, "arrival_marks", _owned(self.arrival_marks))
        object.__setattr__(self, "gumbel_overrides", _owned(self.gumbel_overrides))


@dataclass(frozen=True)
class OriginalPersonPlan:
    key: tuple
    originally_booked: bool
    current_product: str | None
    ticket_token: int | None

    def __post_init__(self):
        object.__setattr__(self, "key", _owned(self.key))


@dataclass(frozen=True)
class NativePopulationSourcePlan:
    complete_spec_json: str
    clock: tuple
    departures: tuple
    postings: tuple
    tape: PrimitiveTapePlan
    people: tuple
    source_definition: str
    observation_scope: str
    context_row_limit: int
    observation_limit: int

    def __post_init__(self):
        object.__setattr__(self, "clock", tuple(utc(t) for t in self.clock))
        object.__setattr__(
            self, "departures", tuple((leg, utc(t)) for leg, t in self.departures)
        )
        object.__setattr__(self, "postings", _owned(self.postings))
        _check(
            type(self.tape) is PrimitiveTapePlan, "typed complete tape recipe required"
        )
        object.__setattr__(self, "tape", replace(self.tape))
        _check(
            all(type(p) is OriginalPersonPlan for p in self.people),
            "typed ALL original support required",
        )
        object.__setattr__(self, "people", tuple(replace(p) for p in self.people))


@dataclass(frozen=True)
class TicketAttachment:
    path: tuple
    person_key: tuple

    def __post_init__(self):
        object.__setattr__(self, "path", _owned(self.path))
        object.__setattr__(self, "person_key", _owned(self.person_key))


@dataclass(frozen=True)
class RegisteredPopulation:
    source: NativePopulationSourcePlan
    programme: population.ConditionalPopulationProgramme
    laws: tuple
    attachments: tuple
    row_limit: int
    digest: str
    rule: str = RULE


@dataclass(frozen=True)
class SuppliedPopulationSource:
    journal: ServiceObservationJournal
    evidence: object
    generated_tape: object
    native_tape: object

    def __post_init__(self):
        _check(
            type(self.journal) is ServiceObservationJournal,
            "NEW supplied journal required",
        )
        object.__setattr__(
            self,
            "journal",
            ServiceObservationJournal(self.journal.context, self.journal.observations),
        )
        for name in ("evidence", "generated_tape", "native_tape"):
            object.__setattr__(self, name, _owned(getattr(self, name)))


@dataclass(frozen=True)
class RegisteredPopulationResult:
    analyst: object

    def __post_init__(self):
        object.__setattr__(self, "analyst", _owned(self.analyst))


def _payload(registration):
    return {
        f.name: getattr(registration, f.name)
        for f in fields(registration)
        if f.name != "digest"
    }


def _digest(registration):
    return hashlib.sha256(_canonical(_payload(registration)).encode()).hexdigest()


def _walk(value, path=()):
    if type(value) in _TYPES:
        for f in fields(value):
            yield from _walk(getattr(value, f.name), (*path, f.name))
    elif isinstance(value, Mapping):
        for k, v in value.items():
            yield from _walk(v, (*path, k))
    elif isinstance(value, (tuple, list)):
        for i, v in enumerate(value):
            yield from _walk(v, (*path, i))
    else:
        yield path, value


def _ticket_paths(programme):
    result = []
    for i, p in enumerate(programme.physical):
        for j, row in enumerate(p.boarding_intents):
            result.append((("physical", i, "boarding_intents", j, 0), row[0]))
        for j, (_, order) in enumerate(p.boarding_orders):
            for k, token in enumerate(order):
                result.append((("physical", i, "boarding_orders", j, 1, k), token))
    for i, c in enumerate(programme.cells):
        for j, person in enumerate(c.persons):
            for k, term in enumerate(person.outcomes):
                for n, t in enumerate(term.transfers):
                    result.append(
                        (
                            (
                                "cells",
                                i,
                                "persons",
                                j,
                                "outcomes",
                                k,
                                "transfers",
                                n,
                                "ticket_record",
                            ),
                            t.ticket_record,
                        )
                    )
    return tuple(result)


def _registration_check(r):
    _check(
        type(r) is RegisteredPopulation and r.rule == RULE,
        "exact registration rule required",
    )
    _check(
        type(r.row_limit) is int and r.row_limit > 0,
        "positive structural limit required",
    )
    _check(
        _units(_encoded(_payload(r))) <= r.row_limit,
        "complete registration exceeds infrastructure limit",
    )
    s = r.source
    _check(type(s) is NativePopulationSourcePlan, "typed source plan required")
    raw = json.loads(s.complete_spec_json)
    c = compile_market(load_spec(raw))
    _check(
        raw == json.loads(dump_spec(c.spec)),
        "ALL native spec fields including defaults required",
    )
    _check(
        c.spec.demand.mode == "poisson",
        "restricted native Poisson source recipe required",
    )
    for text in (s.source_definition, s.observation_scope):
        _text(text)
    _check(
        type(s.context_row_limit) is int
        and s.context_row_limit > 0
        and type(s.observation_limit) is int
        and s.observation_limit > 0,
        "explicit source structural limits required",
    )
    scenario = compile_dated_demand(c, UtcBookingClock(s.clock), s.departures)
    _check(
        len(s.postings) == c.spec.horizon.n_periods
        and all(len(row) == len(c.spec.products) for row in s.postings),
        "ALL period x product postings required",
    )
    for row in s.postings:
        for product, fare in zip(c.spec.products, row):
            _check(
                type(fare) is float
                and product.fare_floor <= fare <= product.fare_cap
                and round_fare(fare) == fare,
                "fixed finite native fare within bounds required",
            )
    keys = tuple(p.key for p in s.people)
    _check(
        len(set(keys)) == len(keys)
        and all(
            population._key(k)
            and k[0] < c.spec.horizon.n_periods
            and k[2] < len(c.spec.markets)
            for k in keys
        ),
        "unique original potential keys required",
    )
    tokens = {}
    for p in s.people:
        _check(
            type(p.originally_booked) is bool,
            "explicit original booking state required",
        )
        if p.current_product is None:
            _check(
                p.ticket_token is None, "outside/unreserved has no current-ticket token"
            )
        else:
            _check(
                p.originally_booked
                and p.current_product in c.product_index
                and type(p.ticket_token) is int
                and p.ticket_token < 0
                and p.ticket_token not in tokens,
                "unique negative current-ticket token and booked owner required",
            )
            _check(
                int(c.product_market[c.product_index[p.current_product]]) == p.key[2],
                "same original O&D required",
            )
            tokens[p.ticket_token] = p.key
    t = s.tape
    _check(
        type(t.seed) is int and type(t.registry_root) is int and t.registry_root >= 0,
        "explicit named tape seed/root required",
    )
    _text(t.namespace)
    _check(
        type(t.minimum_slots) is int
        and t.minimum_slots > 0
        and type(t.generated_row_limit) is int
        and t.generated_row_limit > 0,
        "positive realized tape support limits required",
    )
    _check(
        len(t.arrival_marks) == c.spec.horizon.n_periods,
        "ALL primitive arrival periods required",
    )
    for period in t.arrival_marks:
        previous = 0.0
        for mark in period:
            _check(len(mark) == 5, "complete arrival mark required")
            at, market, segment, thinning, class_u = mark
            _check(
                type(at) is float
                and previous <= at < 1
                and type(market) is int
                and 0 <= market < len(c.spec.markets)
                and type(segment) is int
                and 0 <= segment < len(c.spec.segments)
                and type(thinning) is float
                and 0 <= thinning < 1
                and type(class_u) is float
                and 0 <= class_u < 1,
                "complete sorted native primitive mark support required",
            )
            previous = at
    _check(
        type(t.gumbel_fill) is float
        and type(t.outside_default) is float
        and all(
            float("-inf") < x < float("inf") for x in (t.gumbel_fill, t.outside_default)
        ),
        "finite explicit Gumbel replacements required",
    )
    seen = set()
    for period, slot, product, value in t.gumbel_overrides:
        coordinate = (period, slot, product)
        _check(
            coordinate not in seen
            and all(type(i) is int for i in coordinate)
            and 0 <= period < c.spec.horizon.n_periods
            and 0 <= slot < t.minimum_slots
            and 0 <= product <= len(c.spec.products)
            and type(value) is float
            and float("-inf") < value < float("inf"),
            "unique finite complete Gumbel override required",
        )
        seen.add(coordinate)
    _check(
        all(len(marks) <= t.minimum_slots for marks in t.arrival_marks),
        "declared primitive marks exceed required generated slots",
    )
    _check(
        type(r.programme) is population.ConditionalPopulationProgramme
        and all(type(law) is population.ConditionalPopulationLaw for law in r.laws)
        and bool(r.laws),
        "typed full programme and laws required",
    )
    _clone(r.programme)
    _clone(r.laws)
    _check(
        len({law.name for law in r.laws}) == len(r.laws), "distinct full laws required"
    )
    for law in r.laws:
        _check(
            len(law.weights) == len(r.programme.cells)
            and all(type(w) is Fraction and w >= 0 for w in law.weights)
            and sum(law.weights, Fraction()) == 1,
            "ALL exact joint weights summing one required",
        )
    _check(
        tuple(p.person_key for p in r.programme.information) == keys
        and all(
            tuple(p.person_key for p in cell.persons) == keys
            for cell in r.programme.cells
        ),
        "complete registered original population in every cell required",
    )
    expected = _ticket_paths(r.programme)
    _check(
        all(type(a) is TicketAttachment for a in r.attachments),
        "typed complete ticket attachments required",
    )
    _check(
        tuple((a.path, a.person_key) for a in r.attachments)
        == tuple((path, tokens.get(token)) for path, token in expected)
        and all(token in tokens for _, token in expected),
        "ALL and ONLY legal current-ticket attachment paths/owners required",
    )
    # Tokens are field-specific, never a global value/string replacement.
    allowed = {path for path, _ in expected}
    _check(
        all(
            path in allowed
            for path, value in _walk(r.programme)
            if type(value) is int and value in tokens
        ),
        "ticket token occurs outside permitted ticket field",
    )
    return c, scenario


def register_population(source, programme, laws, attachments, *, row_limit):
    _check(
        type(source) is NativePopulationSourcePlan, "typed full source plan required"
    )
    r = RegisteredPopulation(
        replace(source),
        _clone(programme),
        _clone(laws),
        tuple(replace(a) for a in attachments),
        row_limit,
        "",
    )
    _registration_check(r)
    return replace(r, digest=_digest(r))


def _tapes(plan, compiled, generated, native):
    t = plan.tape
    names = {f.name for f in fields(ExogenousTape)}
    _check(
        set(generated) == names
        and set(native) == names
        and generated["seed"] == t.seed,
        "ALL generated and native tape fields required",
    )
    _check(
        _units(generated) <= t.generated_row_limit,
        "realized generated tape exceeds infrastructure rule; no regeneration",
    )
    T, products, slots = (
        compiled.spec.horizon.n_periods,
        len(compiled.spec.products),
        len(generated["gumbel"][0]),
    )
    _check(
        slots >= t.minimum_slots
        and len(generated["gumbel"]) == T
        and all(
            len(period) == slots and all(len(row) == products + 1 for row in period)
            for period in generated["gumbel"]
        ),
        "complete generated Gumbel support required",
    )
    spec = compiled.spec

    def shape(value, dimensions, kind):
        if dimensions:
            _check(
                isinstance(value, (tuple, list)) and len(value) == dimensions[0],
                "complete generated tape shape required",
            )
            for row in value:
                shape(row, dimensions[1:], kind)
        elif kind == "int":
            _check(
                type(value) is int and value >= 0,
                "nonnegative native integer tape field required",
            )
        else:
            amount = payments._native_amount(value)
            _check(
                amount is not None and (kind != "uniform" or 0 <= amount < 1),
                "finite full native binary64 tape field required",
            )

    for name, dimensions, kind in (
        ("exo_uniform", (T,), "uniform"),
        ("initial_capacity_uniform", (len(spec.legs),), "uniform"),
        ("request_uniforms", (T, 3), "uniform"),
        ("class_uniform", (T, slots), "uniform"),
        ("gumbel", (T, slots, products + 1), "float"),
        ("mixed_draws", (T, slots, len(spec.carriers) + 1), "float"),
        ("capacity_uniforms", (T, len(spec.capacity_process.events)), "uniform"),
        ("background_uniforms", (T, len(spec.background)), "uniform"),
        ("background_initial", (len(spec.background),), "int"),
        (
            "quote_uniforms",
            (
                T + 1,
                len(spec.quotes.requests),
                max([len(q.target_products) for q in spec.quotes.requests] or [1]),
                2,
            ),
            "uniform",
        ),
        ("initial_exo_uniform", (), "uniform"),
        ("arrival_counts", (T, len(spec.markets), len(spec.segments)), "int"),
    ):
        shape(generated[name], dimensions, kind)
    _check(
        len(generated["arrival_marks"]) == T,
        "complete generated arrival periods required",
    )
    for period, marks in enumerate(generated["arrival_marks"]):
        _check(
            len(marks) <= slots
            and len(marks)
            == sum(sum(row) for row in generated["arrival_counts"][period]),
            "full generated count/mark support required",
        )
        previous = (Fraction(-1), -1, -1)
        for mark in marks:
            _check(
                set(mark) == {"time", "market", "segment", "thinning", "class_uniform"},
                "complete generated mark required",
            )
            at, thin, cl = (
                payments._native_amount(mark[n])
                for n in ("time", "thinning", "class_uniform")
            )
            market, segment = mark["market"], mark["segment"]
            _check(
                all(v is not None and 0 <= v < 1 for v in (at, thin, cl))
                and type(market) is int
                and 0 <= market < len(spec.markets)
                and type(segment) is int
                and 0 <= segment < len(spec.segments),
                "valid complete generated mark required",
            )
            coordinate = (at, market, segment)
            _check(coordinate >= previous, "generated mark source order required")
            previous = coordinate
    expected = dict(generated)
    expected["arrival_marks"] = _encoded(
        tuple(
            tuple(ArrivalMark(*mark) for mark in period) for period in t.arrival_marks
        )
    )
    g = [
        [
            [_encoded(t.gumbel_fill) for _ in range(products)]
            + [_encoded(t.outside_default)]
            for _ in range(slots)
        ]
        for _ in range(T)
    ]
    for period, slot, product, value in t.gumbel_overrides:
        g[period][slot][product] = _encoded(value)
    expected["gumbel"] = g
    _check(
        _same(native, expected),
        "native tape differs beyond the complete registered primitive replacement programme",
    )


def _attach(programme, replacements, path=()):
    if path in replacements:
        return replacements[path]
    if type(programme) in _TYPES:
        return type(programme)(
            **{
                f.name: _attach(
                    getattr(programme, f.name), replacements, (*path, f.name)
                )
                for f in fields(programme)
            }
        )
    if isinstance(programme, Mapping):
        return _owned(
            {k: _attach(v, replacements, (*path, k)) for k, v in programme.items()}
        )
    if isinstance(programme, (tuple, list)):
        return tuple(
            _attach(v, replacements, (*path, i)) for i, v in enumerate(programme)
        )
    return programme


def evaluate_registered_population(registration, source, *, row_limit):
    """Validate current full source/bounds, then ONE new conditional evaluation."""
    c, scenario = _registration_check(registration)
    _check(
        registration.digest == _digest(registration),
        "registered content digest changed",
    )
    _check(
        type(source) is SuppliedPopulationSource and not source.journal.observations,
        "NEW complete empty supplied source required",
    )
    _check(
        type(row_limit) is int and row_limit > 0,
        "positive actual structural limit required",
    )
    plan, ctx, ev = registration.source, source.journal.context, source.evidence
    _check(
        _same(ctx["complete_native_model"], _encoded(c.spec))
        and ctx["spec_hash"] == c.spec_hash
        and tuple(ctx["booking_clock"]) == tuple(t.isoformat() for t in plan.clock)
        and tuple(ctx["leg_departures"])
        == tuple((leg, t.isoformat()) for leg, t in plan.departures),
        "complete native source model/clock/departures differ",
    )
    _check(
        ctx["definition"] == plan.source_definition
        and ctx["observation_scope"] == plan.observation_scope
        and ctx["context_row_limit"] == plan.context_row_limit
        and ctx["observation_limit"] == plan.observation_limit,
        "source declaration/limits differ",
    )
    _check(
        set(ev)
        == {
            "engine_class",
            "controls",
            "scenario",
            "controllers",
            "native_period_after",
            "ordinary_periods",
            "trace",
            "metrics",
        },
        "complete supplied source evidence required",
    )
    _check(
        ev["engine_class"] == ENGINE
        and _same(ev["controls"], BASE_CONTROLS)
        and _same(ev["scenario"], _encoded(scenario))
        and ev["native_period_after"] == c.spec.horizon.n_periods,
        "restricted complete baseline chronological source required",
    )
    _check(
        _same(
            ev["controllers"],
            {str(c.carrier_index[a.carrier]): a for a in c.spec.controllers},
        ),
        "complete model controller assignments required",
    )
    _check(
        len(ev["trace"]) == c.spec.horizon.n_periods
        and len(ev["ordinary_periods"]) == c.spec.horizon.n_periods,
        "ALL closed native periods including empty required",
    )
    panel = ctx["complete_ordinary_panel"]["persons"]
    observed_rows = []
    for period, (trace, ordinary, posting) in enumerate(
        zip(ev["trace"], ev["ordinary_periods"], plan.postings)
    ):
        _check(
            trace["period"] == period
            and ordinary["period"] == period
            and _same(trace["prices"], posting),
            "full actual posting/source period differs",
        )
        _check(
            ordinary["log_start"] == len(observed_rows)
            and ordinary["log_stop"] == len(observed_rows) + len(ordinary["rows"]),
            "complete ordinary log support required",
        )
        observed_rows.extend(ordinary["rows"])
    _check(
        len(observed_rows) == len(panel),
        "complete original request population required",
    )
    for row, person in zip(observed_rows, panel):
        _check(
            tuple(row["key"]) == tuple(person["potential_coordinate"])
            and all(
                _same(row[name], person[name])
                for name in (
                    "native_rank",
                    "segment",
                    "taste_class",
                    "request_market",
                    "native_log_index",
                    "original_record",
                )
            )
            and _same(row["native_row"], person["original_ordinary_log"]),
            "same complete ordinary receipt/panel required",
        )
    _tapes(plan, c, source.generated_tape, source.native_tape)
    _check(
        len(plan.people) == len(ctx["persons"]),
        "ALL registered original people required",
    )
    bindings = {}
    for declared, actual in zip(plan.people, ctx["persons"]):
        _check(
            declared.key == actual["person_key"]
            and declared.originally_booked == (actual["original_record"] is not None)
            and declared.current_product == actual["current_product"],
            "original identity/booking/current product differs",
        )
        if declared.ticket_token is not None:
            _check(
                type(actual["current_ticket"]) is int and actual["current_ticket"] >= 0,
                "actual current ticket required",
            )
            bindings[declared.key] = actual["current_ticket"]
        else:
            _check(
                actual["current_ticket"] is None,
                "outside/unreserved cannot gain ticket",
            )
    _check(
        len(set(bindings.values())) == len(bindings),
        "distinct same-person native current tickets required",
    )
    replacements = {a.path: bindings[a.person_key] for a in registration.attachments}
    attached = _attach(registration.programme, replacements)
    _check(
        _same(_attach(registration.programme, replacements), attached),
        "only declared ticket paths may attach",
    )
    parent_bound = population._preflight(
        c, source.journal, attached, registration.laws, row_limit
    )
    required = (
        8 * (_units(_encoded({"registration": registration, "source": source})) + 1)
        + 2 * parent_bound
        + 10000
    )
    _check(
        required <= row_limit,
        "complete actual source/registration/parent output exceeds infrastructure limit",
    )
    result = population.evaluate_conditional_population(
        c, source.journal, attached, registration.laws, row_limit=row_limit
    )
    return RegisteredPopulationResult(
        _encoded(
            {
                "rule": RULE,
                "registration": registration,
                "source": source,
                "ticket_bindings": tuple(bindings.items()),
                "attached_programme": attached,
                "conditional_population": population.conditional_population_record(
                    result
                ),
                "actual_parent_bound": parent_bound,
                "all_support_bound": required,
                "row_limit": row_limit,
                "conditional_population_evaluations": 1,
                "native_source_or_rng_or_actor_executed": False,
                "external_preregistration_native_probability_knowledge_preferences_or_CS_authenticated": False,
                "scientific_budget": None,
            }
        )
    )


def registered_population_record(result):
    _check(type(result) is RegisteredPopulationResult, "typed result required")
    return _encoded(result.analyst)
