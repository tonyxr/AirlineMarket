"""A.10/A.5/B0/V9/B4: owned finite Bernoulli source/programme declarations.

Conditional laws and local ordering are declarations, not native probability,
received knowledge, service truth, preferences or lifecycle utility/CS evidence.
"""

import hashlib
import itertools
import json
import math
from dataclasses import dataclass, replace
from fractions import Fraction

from airline_market.sim import source_future_transport as transport
from airline_market.sim.bernoulli_boundary import DeclaredBaseLedger
from airline_market.sim.bernoulli_future import (
    BernoulliFuturePath,
    BernoulliOpportunity,
    bernoulli_future_expectation,
)
from airline_market.sim.bernoulli_person import MARK_LAW
from airline_market.sim.received_information import _text, _units
from airline_market.sim.shared_mark_attempt import _encoded, _owned
from airline_market.spec import compile_market
from airline_market.spec.compiled import CompiledMarket
from airline_market.spec.io import dump_spec, load_spec

RULE = "owned-prospective-finite-Bernoulli-programmes-v1"


def _check(value, message):
    if not value:
        raise ValueError(message)


def _positive(value):
    _check(
        type(value) is int and value > 0,
        "positive structural infrastructure limit required",
    )


def _names(values):
    _check(
        bool(values) and len(set(values)) == len(values),
        "complete distinct named support required",
    )
    for value in values:
        _text(value)


def _same(a, b):
    return _encoded(a) == _encoded(b)


class BernoulliRegistrationLimit(RuntimeError):
    """Complete named support exceeds the infrastructure limit, no partial output."""


@dataclass(frozen=True)
class BernoulliProgrammeContract:
    signals: tuple
    cells: tuple
    payoff_unit: str
    conditioning_definition: str
    definition: str
    source_ref: str

    def __post_init__(self):
        object.__setattr__(self, "signals", _owned(self.signals))
        object.__setattr__(self, "cells", _owned(self.cells))


@dataclass(frozen=True)
class BernoulliDeclaredEvent:
    name: str
    definition: str
    source_ref: str


@dataclass(frozen=True)
class BernoulliEvaluation:
    name: str
    law: str
    event: str


@dataclass(frozen=True)
class BernoulliHistoryCase:
    path: str
    branches: tuple
    active: tuple
    observations: tuple
    memberships: tuple

    def __post_init__(self):
        for name in ("branches", "active", "memberships"):
            object.__setattr__(self, name, _owned(getattr(self, name)))
        _check(
            all(type(o) is transport.HistoryObservation for o in self.observations),
            "typed complete observation declarations required",
        )
        object.__setattr__(
            self, "observations", tuple(replace(o) for o in self.observations)
        )


@dataclass(frozen=True)
class BernoulliDeclaredLaw:
    name: str
    cases: tuple

    def __post_init__(self):
        _check(
            all(type(c) is BernoulliHistoryCase for c in self.cases),
            "typed complete prospective history cases required",
        )
        object.__setattr__(self, "cases", tuple(replace(c) for c in self.cases))


@dataclass(frozen=True)
class BernoulliSourceRecipe:
    model_json: str
    spec_hash: str
    ledger: DeclaredBaseLedger
    paths: tuple
    start_period: int
    conditioning: str
    mark_law: str
    independent_future_private_marks: bool
    unreserved_base_ledger: bool
    tree_row_limit: int
    conditional_call_limit: int
    ticket_limit: int
    transport_row_limit: int

    def __post_init__(self):
        _check(
            type(self.ledger) is DeclaredBaseLedger,
            "typed empty initial BASE ledger required",
        )
        object.__setattr__(
            self,
            "ledger",
            replace(
                self.ledger,
                installed=_owned(self.ledger.installed),
                tickets=_owned(self.ledger.tickets),
                accounts=_owned(self.ledger.accounts),
            ),
        )
        _check(
            all(type(p) is BernoulliFuturePath for p in self.paths),
            "typed full source paths required",
        )
        paths = []
        for p in self.paths:
            _check(
                all(type(o) is BernoulliOpportunity for o in p.opportunities),
                "typed complete source opportunities required",
            )
            paths.append(
                replace(
                    p,
                    opportunities=tuple(
                        replace(
                            o,
                            prices=_owned(o.prices),
                            installed_after=_owned(o.installed_after),
                        )
                        for o in p.opportunities
                    ),
                )
            )
        object.__setattr__(self, "paths", tuple(paths))


@dataclass(frozen=True)
class RegisteredBernoulliSource:
    recipe: BernoulliSourceRecipe
    contract: BernoulliProgrammeContract
    laws: tuple
    events: tuple
    evaluations: tuple
    support_limit: int
    bound: object
    digest: str
    rule: str = RULE

    def __post_init__(self):
        _check(
            type(self.recipe) is BernoulliSourceRecipe
            and type(self.contract) is BernoulliProgrammeContract,
            "typed owned source/programme contract required",
        )
        object.__setattr__(self, "recipe", replace(self.recipe))
        object.__setattr__(self, "contract", replace(self.contract))
        for name, kind in (
            ("laws", BernoulliDeclaredLaw),
            ("events", BernoulliDeclaredEvent),
            ("evaluations", BernoulliEvaluation),
        ):
            values = getattr(self, name)
            _check(
                all(type(x) is kind for x in values),
                "complete typed " + name + " required",
            )
            object.__setattr__(self, name, tuple(replace(x) for x in values))
        object.__setattr__(self, "bound", _owned(self.bound))


@dataclass(frozen=True)
class RegisteredBernoulliResult:
    analyst: object

    def __post_init__(self):
        object.__setattr__(self, "analyst", _owned(self.analyst))


def _payload(reg):
    return {
        k: getattr(reg, k)
        for k in (
            "recipe",
            "contract",
            "laws",
            "events",
            "evaluations",
            "support_limit",
            "rule",
        )
    }


def _digest(reg):
    return hashlib.sha256(
        json.dumps(
            _encoded(_payload(reg)),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _source(reg):
    r = reg.recipe
    c = compile_market(load_spec(json.loads(r.model_json)))
    _check(
        dump_spec(c.spec) == r.model_json and c.spec_hash == r.spec_hash,
        "same complete canonical source model required",
    )
    _check(
        c.spec.demand.mode == "single_request_bernoulli"
        and all(g.mixed is None and g.classes for g in c.spec.segments),
        "finite native-class Bernoulli source only",
    )
    _check(
        c.spec.resolution.refund == "full_original_fare"
        and not c.spec.resolution.stranded_reentry,
        "no-reentry full-original source refunds required",
    )
    _text(r.conditioning)
    _check(
        r.mark_law == MARK_LAW
        and r.independent_future_private_marks is True
        and r.unreserved_base_ledger is True,
        "explicit conditional independent BASE mark law required",
    )
    for n in (
        r.tree_row_limit,
        r.conditional_call_limit,
        r.ticket_limit,
        r.transport_row_limit,
        reg.support_limit,
    ):
        _positive(n)
    _check(
        not r.ledger.tickets
        and type(r.ledger.next_record_id) is int
        and r.ledger.next_record_id == 0,
        "empty initial tickets only; no earlier person reconstruction",
    )
    _check(
        len(r.ledger.installed) == len(c.spec.legs)
        and all(type(v) is int and 0 <= v <= 2**63 - 1 for v in r.ledger.installed),
        "full installed integer support required",
    )
    _check(
        len(r.ledger.accounts) == len(c.spec.carriers)
        and all(
            len(a) == 5
            and all(type(a[k]) is float for k in (0, 1, 2, 4))
            and type(a[3]) is int
            and a[3] >= 0
            for a in r.ledger.accounts
        ),
        "complete signed/nonfinite carrier monetary context required",
    )
    _names(tuple(p.name for p in r.paths))
    transport._distribution(tuple(p.weight for p in r.paths))
    T = len(r.paths[0].opportunities)
    _check(
        type(r.start_period) is int
        and r.start_period >= 0
        and T > 0
        and r.start_period + T <= c.spec.horizon.n_periods
        and all(len(p.opportunities) == T for p in r.paths),
        "full common consecutive in-horizon source support required",
    )
    for path in r.paths:
        before = r.ledger.installed
        for o in path.opportunities:
            _check(
                type(o.exogenous_state) is int
                and 0 <= o.exogenous_state < len(c.spec.exogenous.states)
                and len(o.prices) == len(c.spec.products)
                and all(type(v) is float and math.isfinite(v) for v in o.prices),
                "full finite Python binary64 fares/exogenous conditioning required",
            )
            _check(
                len(o.installed_after) == len(before)
                and all(
                    type(v) is int and 0 <= v <= old
                    for v, old in zip(o.installed_after, before)
                ),
                "complete nonincreasing installed path required",
            )
            before = o.installed_after
    catalogue = ((None, None, None, None),) + tuple(
        (m, g, k, int(j))
        for m in range(len(c.spec.markets))
        for g, s in enumerate(c.spec.segments)
        for k in range(len(s.classes))
        for j in (*c.market_products[m], len(c.spec.products))
    )
    return c, catalogue, T


def _bound(reg, c, catalogue, T):
    """Full named domain support; each delegated stage also applies its own bound.

    Counts are support slots plus owned metadata nodes, not bytes/RSS or a bound
    on numerical-library scratch space or every serializer implementation node.
    """
    rows = len(catalogue)
    P = len(reg.recipe.paths)
    limit = reg.support_limit
    calls = 0
    frontier = 1
    for _ in range(T):
        calls += P * frontier
        if (
            calls > min(limit, reg.recipe.conditional_call_limit)
            or calls * rows > reg.recipe.tree_row_limit
        ):
            raise BernoulliRegistrationLimit("ALL source-tree support exceeds limit")
        frontier *= rows
    purchase = sum(x[0] is not None and x[3] != len(c.spec.products) for x in catalogue)
    branching = rows + purchase
    contingencies = P
    for _ in range(T):
        contingencies *= branching
        if contingencies > limit:
            raise BernoulliRegistrationLimit(
                "ALL syntactic purchase-status support exceeds limit"
            )
    _check(T <= reg.recipe.ticket_limit, "ALL future ticket support exceeds limit")
    S = len(reg.contract.signals)
    C = len(reg.contract.cells)
    L = len(reg.laws)
    E = len(reg.evaluations)
    model_units = _units(_encoded(c.spec))
    leaves = P * rows**T
    counts = {
        "input_metadata_nodes": _units(_encoded(_payload(reg))),
        "source_conditionals_upper": calls,
        "source_branch_rows_upper": calls * rows,
        "source_terminal_histories_upper": leaves,
        "source_ticket_slots_upper": (calls * rows + leaves) * T,
        "registered_contingencies": contingencies * L,
        "registered_observation_rows": contingencies * L * S,
        "registered_joint_cells": contingencies * L * S * C,
        "registered_person_terms_upper": contingencies * L * S * C * T,
        "evaluation_atoms_upper": E * leaves * S * C,
        "evaluation_person_terms_upper": E * leaves * S * C * T,
        "complete_model_metadata_nodes_upper": model_units * (1 + 2 * calls) * (1 + E),
    }
    total = sum(counts.values())
    if total > limit:
        raise BernoulliRegistrationLimit(
            "complete named source/contingent/evaluation support exceeds limit"
        )
    return _owned(
        {
            **counts,
            "total": total,
            "catalogue": catalogue,
            "source_parent_all_tree_row_call_ticket_limits_before_native_forwards": True,
            "transport_parent_complete_actual_source_input_output_bound_before_each_pure_forward": True,
            "bytes_RSS_scratch_space_scientific_F9_or_scalability_bound": False,
        }
    )


def _people(c, path, branches, active, start):
    people = []
    rid = 0
    for offset, (m, g, k, j) in enumerate(branches):
        if m is None:
            continue
        purchase = j != len(c.spec.products)
        ticket = rid if purchase else None
        people.append(
            {
                "key": (start + offset, 0, m),
                "ticket": ticket,
                "active": active[rid] if purchase else None,
                "fare": Fraction.from_float(path.opportunities[offset].prices[j])
                if purchase
                else None,
                "carrier": c.spec.carriers[int(c.product_carrier[j])].code
                if purchase
                else None,
            }
        )
        rid += int(purchase)
    return people


def _validate(reg):
    _check(
        type(reg) is RegisteredBernoulliSource and reg.rule == RULE,
        "current complete typed registration required",
    )
    c, catalogue, T = _source(reg)
    for field in ("signals", "cells"):
        _names(getattr(reg.contract, field))
    for field in ("payoff_unit", "conditioning_definition", "definition", "source_ref"):
        _text(getattr(reg.contract, field))
    _names(tuple(l.name for l in reg.laws))
    _names(tuple(e.name for e in reg.events))
    _names(tuple(e.name for e in reg.evaluations))
    for e in reg.events:
        _text(e.definition)
        _text(e.source_ref)
    for e in reg.evaluations:
        _check(
            e.law in {l.name for l in reg.laws}
            and e.event in {x.name for x in reg.events},
            "evaluation names one full declared law/event",
        )
    bound = _bound(reg, c, catalogue, T)
    expected_count = bound["registered_contingencies"] // len(reg.laws)
    for law in reg.laws:
        _check(
            len(law.cases) == expected_count,
            "ALL prospective cases including unused/zero support required",
        )
        index = 0
        for path in reg.recipe.paths:
            for branches in itertools.product(catalogue, repeat=T):
                purchases = sum(
                    row[0] is not None and row[3] != len(c.spec.products)
                    for row in branches
                )
                for active in itertools.product((False, True), repeat=purchases):
                    row = law.cases[index]
                    index += 1
                    _check(
                        row.path == path.name
                        and row.branches == branches
                        and row.active == active
                        and all(type(x) is bool for x in row.active),
                        "complete ordered source-history/purchase-status support required",
                    )
                    _check(
                        tuple(n for n, m in row.memberships)
                        == tuple(e.name for e in reg.events)
                        and all(
                            len(m) == len(reg.contract.signals)
                            and all(type(b) is bool for b in m)
                            for n, m in row.memberships
                        ),
                        "ALL history x signal event memberships, no future-cell selector",
                    )
                    _check(
                        tuple(o.signal for o in row.observations)
                        == reg.contract.signals,
                        "ALL observation rows required",
                    )
                    transport._distribution(
                        tuple(o.probability for o in row.observations)
                    )
                    people = _people(c, path, branches, active, reg.recipe.start_period)
                    for obs in row.observations:
                        _check(
                            tuple(cell.name for cell in obs.cells)
                            == reg.contract.cells,
                            "ALL joint future cells required",
                        )
                        transport._distribution(
                            tuple(cell.probability for cell in obs.cells)
                        )
                        for cell in obs.cells:
                            _check(
                                tuple(p.person_key for p in cell.persons)
                                == tuple(p["key"] for p in people),
                                "ALL original people including outside/unreserved in every cell required",
                            )
                            ids = set()
                            for term, person in zip(cell.persons, people):
                                _check(
                                    term.current_service is None
                                    or type(term.current_service) is bool,
                                    "explicit Boolean-or-unknown service declaration required",
                                )
                                _check(
                                    person["active"] is True
                                    or term.current_service is None,
                                    "outside/unreserved current service stays unknown",
                                )
                                _check(
                                    type(term.financial_closed) is bool
                                    and type(term.material_closed) is bool,
                                    "explicit independent financial/material closures required",
                                )
                                for text in (
                                    term.service_source,
                                    term.financial_source,
                                    term.material_source,
                                ):
                                    _text(text)
                                _check(
                                    tuple(k for k, v in term.component_sources)
                                    == transport.COMPONENTS,
                                    "FOUR ordered explicit component sources required",
                                )
                                for k, v in term.component_sources:
                                    _text(v)
                                net = (
                                    person["fare"]
                                    if person["active"] is True
                                    else Fraction()
                                )
                                for a in term.adjustments:
                                    _text(a.event_id)
                                    _text(a.source_ref)
                                    _check(
                                        a.event_id not in ids
                                        and person["ticket"] is not None
                                        and type(a.ticket_record) is int
                                        and a.ticket_record == person["ticket"]
                                        and a.carrier == person["carrier"],
                                        "distinct same original ticket/carrier NET adjustments required",
                                    )
                                    ids.add(a.event_id)
                                    _check(
                                        a.amount is None or type(a.amount) is Fraction,
                                        "exact-or-unknown additional NET payment required",
                                    )
                                    net = (
                                        None
                                        if net is None or a.amount is None
                                        else net + a.amount
                                    )
                                _check(
                                    term.payoff.net_ticket_payment
                                    == (net if term.financial_closed else None),
                                    "closed NET equals source fare/refund baseline plus ALL additional transfers",
                                )
    return c, catalogue, T, bound


def register_bernoulli_source(
    compiled,
    ledger,
    paths,
    contract,
    laws,
    events,
    evaluations,
    *,
    start_period,
    conditioning,
    mark_law,
    independent_future_private_marks,
    unreserved_base_ledger,
    tree_row_limit,
    conditional_call_limit,
    ticket_limit,
    transport_row_limit,
    support_limit,
):
    _check(type(compiled) is CompiledMarket, "complete compiled model required")
    recipe = BernoulliSourceRecipe(
        dump_spec(compiled.spec),
        compiled.spec_hash,
        ledger,
        paths,
        start_period,
        conditioning,
        mark_law,
        independent_future_private_marks,
        unreserved_base_ledger,
        tree_row_limit,
        conditional_call_limit,
        ticket_limit,
        transport_row_limit,
    )
    reg = RegisteredBernoulliSource(
        recipe, contract, laws, events, evaluations, support_limit, {}, ""
    )
    _, _, _, bound = _validate(reg)
    return replace(reg, bound=bound, digest=_digest(reg))


def _match_source(c, source, recipe):
    d = source.details
    expected_paths = tuple(
        {"name": p.name, "weight": p.weight, "opportunities": p.opportunities}
        for p in recipe.paths
    )
    _check(
        _same(d["complete_native_model"], c.spec)
        and _same(d["initial_declared_ledger"], recipe.ledger)
        and _same(d["paths"], expected_paths),
        "actual full native source must equal registered model/ledger/path recipe",
    )
    _check(
        d["start_period"] == recipe.start_period
        and d["conditioning"] == recipe.conditioning
        and d["mark_law"] == recipe.mark_law
        and d["independent_future_private_marks_declared_not_authenticated"] is True,
        "actual source conditional information contract differs",
    )
    for key in ("tree_row_limit", "conditional_call_limit", "ticket_limit"):
        _check(
            d[key] == getattr(recipe, key),
            "actual source infrastructure contract differs",
        )
    transport._source(c, _owned(_encoded(source)))


def execute_registered_bernoulli_source(registration):
    c, catalogue, T, bound = _validate(registration)
    _check(
        _same(bound, registration.bound)
        and _digest(registration) == registration.digest,
        "owned full registration digest/recomputed support bound differs",
    )
    r = registration.recipe
    source = bernoulli_future_expectation(
        c,
        r.ledger,
        r.paths,
        start_period=r.start_period,
        conditioning=r.conditioning,
        mark_law=r.mark_law,
        independent_future_private_marks=r.independent_future_private_marks,
        unreserved_base_ledger=r.unreserved_base_ledger,
        tree_row_limit=r.tree_row_limit,
        conditional_call_limit=r.conditional_call_limit,
        ticket_limit=r.ticket_limit,
    )
    _match_source(c, source, r)
    programmes = {}
    events = {}
    links = []
    nodes = source.details["nodes"]
    for law in registration.laws:
        cases = {(x.path, x.branches, x.active): x for x in law.cases}
        histories = []
        members = {e.name: [] for e in registration.events}
        used = []
        for terminal in source.details["terminals"]:
            branches = tuple(
                tuple(nodes[ni]["boundary"]["branches"][ri]["input"][:4])
                for ni, ri in terminal["history"]
            )
            _check(
                len(branches) == T and all(b in catalogue for b in branches),
                "same complete source coordinate support required",
            )
            purchases = sum(
                b[0] is not None and b[3] != len(c.spec.products) for b in branches
            )
            current = {b["record_id"] for b in terminal["ledger"]["tickets"]}
            active = tuple(i in current for i in range(purchases))
            key = (terminal["path"], branches, active)
            row = cases[key]
            used.append(key)
            histories.append(
                transport.HistoryFuture(row.path, terminal["history"], row.observations)
            )
            for name, membership in row.memberships:
                members[name].append(membership)
        contract = registration.contract
        programmes[law.name] = transport.SourceFutureProgramme(
            contract.signals,
            contract.cells,
            histories,
            contract.payoff_unit,
            contract.conditioning_definition,
            contract.definition,
            contract.source_ref,
        )
        events[law.name] = {
            e.name: transport.SourceSelectionEvent(
                e.name, members[e.name], e.definition, e.source_ref
            )
            for e in registration.events
        }
        used_set = set(used)
        links.append(
            {
                "law": law.name,
                "matched": tuple(used),
                "unused": tuple(k for k in cases if k not in used_set),
                "only_dynamic_attachment": "native node,row history indices; all numerical declarations/events unchanged",
            }
        )
    outputs = []
    for e in registration.evaluations:
        outputs.append(
            {
                "name": e.name,
                "law": e.law,
                "event": e.event,
                "result": transport.condition_source_future(
                    c,
                    source,
                    programmes[e.law],
                    events[e.law][e.event],
                    row_limit=r.transport_row_limit,
                ),
            }
        )
    return RegisteredBernoulliResult(
        _encoded(
            {
                "plan_id": "A.10/A.5/B0/V9/B4",
                "rule": RULE,
                "registration": registration,
                "complete_native_source": source,
                "attachment_links": tuple(links),
                "attached_programmes": programmes,
                "attached_events": events,
                "evaluations": tuple(outputs),
                "recomputed_full_named_support": bound,
                "native_source_compilations": 1,
                "transport_calls": len(outputs),
                "genuine_engine_actor_tape_RNG_service_or_old_saved_loader_called": False,
                "source_time_public_reception_native_probability_preferences_service_truth_authenticated": False,
                "lifecycle_utility_CS_total_denial_misallocation": None,
                "full_primary_default_empirical_production_planner_adopted": False,
                "scientific_budget": None,
            }
        )
    )


def registered_bernoulli_record(value):
    _check(
        type(value) in (RegisteredBernoulliSource, RegisteredBernoulliResult),
        "typed complete registration/result required",
    )
    return _encoded(
        value if type(value) is RegisteredBernoulliSource else value.analyst
    )
