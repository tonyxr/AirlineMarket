"""A.10/A.5/B0/V9/B4: complete supplied finite source/history transport.

Analyst event conditioning of a numerical native kernel measure, with declared
observation/future/material laws. No source producer, actor or lifecycle CS.
"""

from dataclasses import dataclass, replace
from fractions import Fraction

from airline_market.sim import service_ticket_payments as payments
from airline_market.sim.bernoulli_future import BernoulliFutureResult
from airline_market.sim.bernoulli_person import MARK_LAW
from airline_market.sim.payoff_logit import ContingentPayoff
from airline_market.sim.received_information import _text, _units
from airline_market.sim.shared_mark_attempt import _encoded, _owned
from airline_market.spec.compiled import CompiledMarket

RULE = "finite-source-history-joint-future-transport-v1"
COMPONENTS = (
    "service_benefit",
    "net_ticket_payment",
    "waiting_loss",
    "cancellation_loss",
)


class SourceTransportLimit(RuntimeError):
    pass


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _same(a, b):
    return _owned(_encoded(a)) == _owned(_encoded(b))


def _q(value):
    return payments._rational(value)


@dataclass(frozen=True)
class NetTicketAdjustment:
    event_id: str
    ticket_record: int
    carrier: str
    amount: Fraction | None
    source_ref: str


@dataclass(frozen=True)
class PersonFutureMaterial:
    person_key: tuple
    current_service: bool | None
    service_source: str
    adjustments: tuple
    payoff: ContingentPayoff
    component_sources: tuple
    financial_closed: bool
    financial_source: str
    material_closed: bool
    material_source: str

    def __post_init__(self):
        object.__setattr__(self, "person_key", _owned(self.person_key))
        _check(
            all(type(a) is NetTicketAdjustment for a in self.adjustments),
            "typed full ticket adjustments required",
        )
        object.__setattr__(
            self, "adjustments", tuple(replace(a) for a in self.adjustments)
        )
        _check(
            type(self.payoff) is ContingentPayoff,
            "FOUR explicit payoff components required",
        )
        object.__setattr__(self, "payoff", replace(self.payoff))
        object.__setattr__(self, "component_sources", _owned(self.component_sources))


@dataclass(frozen=True)
class JointFutureCell:
    name: str
    probability: Fraction
    persons: tuple

    def __post_init__(self):
        _check(
            all(type(p) is PersonFutureMaterial for p in self.persons),
            "typed ALL-person future terms required",
        )
        object.__setattr__(self, "persons", tuple(replace(p) for p in self.persons))


@dataclass(frozen=True)
class HistoryObservation:
    signal: str
    probability: Fraction
    cells: tuple

    def __post_init__(self):
        _check(
            all(type(c) is JointFutureCell for c in self.cells),
            "typed complete joint cells required",
        )
        object.__setattr__(self, "cells", tuple(replace(c) for c in self.cells))


@dataclass(frozen=True)
class HistoryFuture:
    path: str
    history: tuple
    observations: tuple

    def __post_init__(self):
        object.__setattr__(self, "history", _owned(self.history))
        _check(
            all(type(o) is HistoryObservation for o in self.observations),
            "typed complete observation law required",
        )
        object.__setattr__(
            self, "observations", tuple(replace(o) for o in self.observations)
        )


@dataclass(frozen=True)
class SourceFutureProgramme:
    signals: tuple
    cells: tuple
    histories: tuple
    payoff_unit: str
    conditioning_definition: str
    definition: str
    source_ref: str

    def __post_init__(self):
        for name in ("signals", "cells"):
            object.__setattr__(self, name, _owned(getattr(self, name)))
        _check(
            all(type(h) is HistoryFuture for h in self.histories),
            "typed full original histories required",
        )
        object.__setattr__(self, "histories", tuple(replace(h) for h in self.histories))


@dataclass(frozen=True)
class SourceSelectionEvent:
    name: str
    membership: tuple
    definition: str
    source_ref: str

    def __post_init__(self):
        object.__setattr__(self, "membership", _owned(self.membership))


@dataclass(frozen=True)
class SourceFutureTransport:
    analyst: object
    rule: str = RULE

    def __post_init__(self):
        object.__setattr__(self, "analyst", _owned(self.analyst))


def _free(c, state):
    stock = list(state["installed"])
    for ticket in state["tickets"]:
        for l, a in enumerate(c.A_global[:, ticket["product"]]):
            stock[l] -= int(a)
    _check(
        all(type(x) is int and x >= 0 for x in stock),
        "complete nonnegative NET stock required",
    )
    return tuple(stock)


def _accounts(rows):
    for row in rows:
        _check(
            len(row) == 5 and type(row[3]) is int and row[3] >= 0,
            "full native carrier account/cancellation support required",
        )
        for index in (0, 1, 2, 4):
            # Validate lossless signed/nonfinite payloads, without monetary repair.
            payments._native_amount(row[index])


def _source(c, record):
    """Read the complete NEW finite graph; no native probability/ledger replay."""
    d = record["details"]
    initial = d["initial_declared_ledger"]
    nodes = d["nodes"]
    terminals = d["terminals"]
    _check(
        _same(d["complete_native_model"], c.spec)
        and c.spec.demand.mode == "single_request_bernoulli",
        "same full native Bernoulli model required",
    )
    _check(
        d["mark_law"] == MARK_LAW
        and d["independent_future_private_marks_declared_not_authenticated"] is True,
        "complete declared mark law required",
    )
    _check(
        not initial["tickets"] and initial["next_record_id"] == 0,
        "empty initial ledger only; no earlier people/ticket reconstruction",
    )
    _check(
        len(initial["installed"]) == len(c.spec.legs)
        and all(type(x) is int and x >= 0 for x in initial["installed"]),
        "full installed support required",
    )
    _check(
        len(initial["accounts"]) == len(c.spec.carriers)
        and all(len(a) == 5 for a in initial["accounts"]),
        "complete carrier monetary context required",
    )
    _accounts(initial["accounts"])
    _check(
        c.spec.resolution.refund == "full_original_fare"
        and not c.spec.resolution.stranded_reentry,
        "restricted no-reentry full-original-refund source required",
    )
    paths = d["paths"]
    _check(
        bool(paths) and len({p["name"] for p in paths}) == len(paths),
        "full distinct source paths required",
    )
    weights = tuple(_q(p["weight"]) for p in paths)
    _check(
        all(w >= 0 for w in weights) and sum(weights, Fraction()) == 1,
        "exact complete source path weights required",
    )
    periods = len(paths[0]["opportunities"])
    start = d["start_period"]
    _check(
        type(start) is int
        and 0 <= start
        and periods > 0
        and start + periods <= c.spec.horizon.n_periods
        and all(len(p["opportunities"]) == periods for p in paths),
        "full common in-horizon source support required",
    )
    expected_coordinates = [(None, None, None, None)]
    expected_coordinates.extend(
        (m, g, k, j)
        for m in range(len(c.spec.markets))
        for g, seg in enumerate(c.spec.segments)
        for k in range(len(seg.classes))
        for j in (*c.market_products[m], len(c.spec.products))
    )
    node_index = terminal_index = 0
    unknown = []
    profiles = []
    public = []
    raw_sum = Fraction()
    for path, prior in zip(paths, weights):
        frontier = [((), initial, prior, prior > 0)]
        for depth, op in enumerate(path["opportunities"]):
            _check(
                len(op["prices"]) == len(c.spec.products)
                and all(payments._native_amount(p) is not None for p in op["prices"]),
                "full finite source fares required",
            )
            following = []
            for history, state, mass, needed in frontier:
                _check(node_index < len(nodes), "omitted native source node")
                ni = node_index
                n = nodes[ni]
                node_index += 1
                _check(
                    n["node"] == ni
                    and n["path"] == path["name"]
                    and n["period"] == start + depth
                    and n["history"] == history
                    and _same(n["boundary"]["declared_base_ledger"], state),
                    "complete ordered node/history/ledger chain required",
                )
                _check(
                    _q(n["raw_reaching_mass"]) == mass
                    and n["structurally_needed"] is needed,
                    "raw/structural reaching mass differs",
                )
                b = n["boundary"]
                law = b["supplied_new_birth_law"]
                stock = _free(c, state)
                _check(
                    _same(b["complete_native_model"], c.spec)
                    and _same(law["complete_native_model"], c.spec)
                    and law["spec_hash"] == c.spec_hash,
                    "same complete native model in every source node required",
                )
                _check(
                    law["period"] == start + depth
                    and law["exogenous_state"] == op["exogenous_state"]
                    and _same(law["global_fixed_prices"], op["prices"])
                    and law["initial_net_free_stock"] == stock,
                    "source conditioning inputs differ",
                )
                mask = tuple(
                    all(x >= int(a) for x, a in zip(stock, c.A_global[:, j]))
                    for j in range(len(c.spec.products))
                )
                _check(
                    law["global_net_sellable"] == mask
                    and b["installed_after"] == op["installed_after"]
                    and all(
                        type(x) is int and 0 <= x <= old
                        for x, old in zip(op["installed_after"], state["installed"])
                    )
                    and len(op["installed_after"]) == len(stock),
                    "full source menu/installed boundary differs",
                )
                underflows = {tuple(x) for x in law["needed_underflows"]}
                _check(
                    b["input_public_law_known"] is (not bool(underflows)),
                    "upstream known/underflow flags differ",
                )
                if needed and underflows:
                    unknown.append(ni)
                rows = b["branches"]
                raw = law["raw_joint"]
                _check(
                    len(rows) == len(raw) == len(expected_coordinates),
                    "ALL native source branch support required",
                )
                for ri, (branch, row, coordinate) in enumerate(
                    zip(rows, raw, expected_coordinates)
                ):
                    _check(
                        branch["row"] == ri
                        and branch["input"] == row
                        and tuple(row[:4]) == coordinate
                        and len(row) == 6,
                        "ordered complete branch coordinates required",
                    )
                    m, g, k, j = coordinate
                    prob = _q(row[-1])
                    _check(prob >= 0, "negative source mass")
                    active = m is None or j == len(c.spec.products) or mask[j]
                    target = (
                        None
                        if not active
                        else stock
                        if m is None or j == len(c.spec.products)
                        else tuple(x - int(a) for x, a in zip(stock, c.A_global[:, j]))
                    )
                    _check(
                        row[4] == target and branch["executed"] is active,
                        "complete source feasibility/physical debit differs",
                    )
                    if not active:
                        _check(prob == 0, "inactive source row has mass")
                        continue
                    new = branch["new_ticket"]
                    purchase = m is not None and j != len(c.spec.products)
                    _check(
                        (new is not None) == purchase,
                        "source purchase/outside/no-arrival differs",
                    )
                    if purchase:
                        _check(
                            new["record_id"] == state["next_record_id"]
                            and new["product"] == j
                            and new["carrier"] == int(c.product_carrier[j])
                            and new["period"] == start + depth
                            and new["arrival_rank"] == 0
                            and new["segment"] == g
                            and new["taste_class"] == k
                            and _same(new["fare"], op["prices"][j]),
                            "same actual NEW source ticket required",
                        )
                    all_tickets = {t["record_id"]: t for t in state["tickets"]}
                    if new is not None:
                        all_tickets[new["record_id"]] = new
                    kept = {t["record_id"]: t for t in branch["kept"]}
                    lost = {t["record_id"]: t for t in branch["canceled"]}
                    _check(
                        len(kept) == len(branch["kept"])
                        and len(lost) == len(branch["canceled"])
                        and not kept.keys() & lost.keys()
                        and kept.keys() | lost.keys() == all_tickets.keys()
                        and all(
                            _same(t, all_tickets[rid])
                            for rid, t in (*kept.items(), *lost.items())
                        ),
                        "complete immutable kept/cancelled ticket partition required",
                    )
                    _check(
                        branch["next_record_id"]
                        == state["next_record_id"] + int(purchase)
                        and branch["native_period_unchanged"] == start + depth,
                        "source native IDs/period differ",
                    )
                    carriers = branch["native_carrier_ledgers"]
                    _check(
                        len(carriers) == len(c.spec.carriers)
                        and tuple(x["carrier"] for x in carriers)
                        == tuple(range(len(carriers))),
                        "complete ordered carrier context required",
                    )
                    next_state = {
                        "installed": op["installed_after"],
                        "tickets": branch["kept"],
                        "accounts": tuple(
                            (
                                x["cash"],
                                x["refunds"],
                                x["penalties"],
                                x["canceled"],
                                x["revenue"],
                            )
                            for x in carriers
                        ),
                        "next_record_id": branch["next_record_id"],
                    }
                    _accounts(next_state["accounts"])
                    _free(c, next_state)
                    next_history = (*history, (ni, ri))
                    next_mass = mass * prob
                    next_needed = needed and (prob > 0 or coordinate in underflows)
                    if depth + 1 < periods:
                        following.append(
                            (next_history, next_state, next_mass, next_needed)
                        )
                        continue
                    _check(
                        terminal_index < len(terminals),
                        "omitted complete terminal history",
                    )
                    t = terminals[terminal_index]
                    terminal_index += 1
                    _check(
                        t["path"] == path["name"]
                        and t["history"] == next_history
                        and _same(t["ledger"], next_state)
                        and _q(t["raw_mass"]) == next_mass
                        and t["structurally_needed"] is next_needed,
                        "complete ordered terminal state/factors required",
                    )
                    people = []
                    canceled = set()
                    created = []
                    for native_node, branch_row in next_history:
                        nb = nodes[native_node]["boundary"]["branches"][branch_row]
                        canceled.update(x["record_id"] for x in nb["canceled"])
                        if nb["input"][0] is None:
                            continue
                        key = (nodes[native_node]["period"], 0, nb["input"][0])
                        ticket = nb["new_ticket"]
                        rid = None if ticket is None else ticket["record_id"]
                        fare = (
                            None
                            if ticket is None
                            else payments._native_amount(ticket["fare"])
                        )
                        people.append(
                            {
                                "person_key": key,
                                "segment": nb["input"][1],
                                "taste_class": nb["input"][2],
                                "original_ticket": rid,
                                "carrier": None
                                if ticket is None
                                else c.spec.carriers[ticket["carrier"]].code,
                                "original_fare": fare,
                                "original_source_ticket": ticket,
                            }
                        )
                        if rid is not None:
                            created.append(rid)
                    current = set(kept)
                    _check(
                        current | canceled == set(created) and not current & canceled,
                        "full original ticket lifecycle partition required",
                    )
                    for person in people:
                        rid = person["original_ticket"]
                        person.update(
                            current_ticket=rid if rid in current else None,
                            native_source_refund=person["original_fare"]
                            if rid in canceled
                            else Fraction(),
                            native_net_ticket_payment=Fraction()
                            if rid is None or rid in canceled
                            else person["original_fare"],
                        )
                    arrivals = len(people)
                    purchases = len(created)
                    counts = (
                        ("new_original_arrivals", arrivals),
                        ("new_outside_choices", arrivals - purchases),
                        ("new_purchases", purchases),
                        ("all_cancellations", len(canceled)),
                        ("terminal_active_tickets", len(current)),
                    )
                    # Native count labels are retained; compare their full values.
                    _check(
                        t["counts"] == counts
                        and t["new_active_ticket_ids"]
                        == tuple(x["record_id"] for x in branch["kept"]),
                        "full terminal original-person/ticket counts differ",
                    )
                    profiles.append(tuple(people))
                    public.append((path["name"], next_history, next_mass))
                    raw_sum += next_mass
            frontier = following
    _check(
        node_index == len(nodes) and terminal_index == len(terminals),
        "unmatched extra source nodes/terminals",
    )
    _check(
        tuple(unknown) == d["needed_unknown_nodes"]
        and (record["joint"] is None) == bool(unknown),
        "whole-source undefined contract differs",
    )
    if not unknown:
        _check(
            _same(record["joint"], public),
            "complete public/raw terminal measure differs",
        )
    _check(
        _q(d["exact_raw_mass"]) == raw_sum
        and _q(d["exact_raw_mass_residual"]) == raw_sum - 1,
        "raw source mass/residual differs",
    )
    return tuple(profiles), not bool(unknown), raw_sum


def _names(values):
    _check(
        bool(values) and len(set(values)) == len(values),
        "complete distinct named support required",
    )
    for v in values:
        _text(v)


def _distribution(values):
    _check(
        all(type(v) is Fraction and v >= 0 for v in values)
        and sum(values, Fraction()) == 1,
        "ALL exact nonnegative weights summing one required",
    )


def _preflight(c, source, programme, event, row_limit):
    _check(
        type(c) is CompiledMarket and type(source) is BernoulliFutureResult,
        "NEW complete compiled/Bernoulli source required",
    )
    _check(
        type(programme) is SourceFutureProgramme
        and type(event) is SourceSelectionEvent,
        "typed full programme/event required",
    )
    _check(
        type(row_limit) is int and row_limit > 0, "positive structural limit required"
    )
    record = _owned(_encoded(source))
    required = (
        32
        * (
            _units(
                _encoded(
                    {
                        "source": record,
                        "programme": programme,
                        "event": event,
                        "rule": RULE,
                        "row_limit": row_limit,
                    }
                )
            )
            + 1
        )
        * (len(programme.signals) * len(programme.cells) + 1)
        + 10000
    )
    if required > row_limit:
        raise SourceTransportLimit(
            "complete input/proposed output exceeds structural infrastructure limit"
        )
    for text in (
        programme.payoff_unit,
        programme.conditioning_definition,
        programme.definition,
        programme.source_ref,
        event.name,
        event.definition,
        event.source_ref,
    ):
        _text(text)
    _names(programme.signals)
    _names(programme.cells)
    profiles, known, total = _source(c, record)
    terminals = record["details"]["terminals"]
    _check(
        len(programme.histories) == len(profiles) == len(event.membership),
        "ALL source histories/event membership including zero support required",
    )
    for hi, (future, terminal, people, membership) in enumerate(
        zip(programme.histories, terminals, profiles, event.membership)
    ):
        _check(
            future.path == terminal["path"] and future.history == terminal["history"],
            "same complete source history order required",
        )
        _check(
            len(membership) == len(programme.signals)
            and all(type(x) is bool for x in membership),
            "complete history x signal boolean event, no future-cell selector",
        )
        _check(
            tuple(o.signal for o in future.observations) == programme.signals,
            "ALL observation rows required",
        )
        _distribution(tuple(o.probability for o in future.observations))
        for observation in future.observations:
            _check(
                tuple(cell.name for cell in observation.cells) == programme.cells,
                "ALL joint future cells required",
            )
            _distribution(tuple(cell.probability for cell in observation.cells))
            for cell in observation.cells:
                _check(
                    tuple(p.person_key for p in cell.persons)
                    == tuple(p["person_key"] for p in people),
                    "ALL original people including outside/unreserved in EVERY cell required",
                )
                ids = set()
                for term, person in zip(cell.persons, people):
                    _check(
                        term.current_service is None
                        or type(term.current_service) is bool,
                        "explicit boolean-or-unknown declared service required",
                    )
                    _check(
                        person["current_ticket"] is not None
                        or term.current_service is None,
                        "outside/unreserved current service stays unknown",
                    )
                    _check(
                        type(term.financial_closed) is bool
                        and type(term.material_closed) is bool,
                        "explicit independent closures required",
                    )
                    for text in (
                        term.service_source,
                        term.financial_source,
                        term.material_source,
                    ):
                        _text(text)
                    _check(
                        tuple(k for k, v in term.component_sources) == COMPONENTS,
                        "FOUR complete ordered component sources required",
                    )
                    for k, v in term.component_sources:
                        _text(v)
                    net = person["native_net_ticket_payment"]
                    for adjustment in term.adjustments:
                        _text(adjustment.event_id)
                        _text(adjustment.source_ref)
                        _check(
                            adjustment.event_id not in ids
                            and person["original_ticket"] is not None
                            and type(adjustment.ticket_record) is int
                            and adjustment.ticket_record == person["original_ticket"]
                            and adjustment.carrier == person["carrier"],
                            "distinct same original ticket/carrier adjustments only",
                        )
                        ids.add(adjustment.event_id)
                        _check(
                            adjustment.amount is None
                            or type(adjustment.amount) is Fraction,
                            "explicit exact-or-unknown signed NET adjustment required",
                        )
                        net = (
                            None
                            if net is None or adjustment.amount is None
                            else net + adjustment.amount
                        )
                    expected = net if term.financial_closed else None
                    _check(
                        term.payoff.net_ticket_payment == expected,
                        "explicit NET component must match declared-closed native baseline plus adjustments",
                    )
    return record, profiles, known, total, required


def condition_source_future(compiled, source, programme, event, *, row_limit):
    record, profiles, known, total, required = _preflight(
        compiled, source, programme, event, row_limit
    )
    atoms = []
    raw_evidence = Fraction()
    for hi, (history, terminal, people) in enumerate(
        zip(programme.histories, record["details"]["terminals"], profiles)
    ):
        mass = _q(terminal["raw_mass"])
        for si, observation in enumerate(history.observations):
            selected = event.membership[hi][si]
            if selected:
                raw_evidence += mass * observation.probability
            for ci, cell in enumerate(observation.cells):
                terms = []
                for term, person in zip(cell.persons, people):
                    values = tuple(getattr(term.payoff, k) for k in COMPONENTS)
                    material = (
                        None
                        if not term.financial_closed
                        or not term.material_closed
                        or None in values
                        else values[0] - values[1] - values[2] - values[3]
                    )
                    terms.append(
                        {
                            "person_key": person["person_key"],
                            "original_ticket": person["original_ticket"],
                            "current_ticket": person["current_ticket"],
                            "native_net_ticket_payment": person[
                                "native_net_ticket_payment"
                            ],
                            "declared_current_service": term.current_service,
                            "material": material,
                            "payment": term.payoff.net_ticket_payment
                            if term.financial_closed
                            else None,
                        }
                    )

                def total_value(name, terms=terms):
                    values = tuple(t[name] for t in terms)
                    return None if None in values else sum(values, Fraction())

                joint = mass * observation.probability * cell.probability
                atoms.append(
                    {
                        "history_index": hi,
                        "signal_index": si,
                        "cell_index": ci,
                        "path": history.path,
                        "history": history.history,
                        "signal": observation.signal,
                        "cell": cell.name,
                        "source_mass": mass,
                        "observation_factor": observation.probability,
                        "future_factor": cell.probability,
                        "raw_joint_mass": joint,
                        "event_member": selected,
                        "persons": tuple(terms),
                        "population_material": total_value("material"),
                        "population_payment": total_value("payment"),
                    }
                )
    defined = known and raw_evidence > 0
    selected = (
        tuple(
            {**a, "conditional_mass": a["raw_joint_mass"] / raw_evidence}
            for a in atoms
            if a["event_member"]
        )
        if defined
        else None
    )

    def expectation(field):
        if selected is None:
            return None
        contributions = tuple(
            Fraction()
            if a["conditional_mass"] == 0
            else None
            if a[field] is None
            else a["conditional_mass"] * a[field]
            for a in selected
        )
        return {
            "mean": None if None in contributions else sum(contributions, Fraction()),
            "contributions": contributions,
            "unknown_mass": sum(
                (a["conditional_mass"] for a in selected if a[field] is None),
                Fraction(),
            ),
        }

    cells = (
        tuple(
            sum(
                (a["conditional_mass"] for a in selected if a["cell_index"] == i),
                Fraction(),
            )
            for i in range(len(programme.cells))
        )
        if defined
        else None
    )
    return SourceFutureTransport(
        _encoded(
            {
                "plan_id": "A.10/A.5/B0/V9/B4",
                "rule": RULE,
                "complete_source": record,
                "programme": programme,
                "event": event,
                "all_original_person_profiles": profiles,
                "all_atoms": tuple(atoms),
                "source_public_joint_known": known,
                "raw_total_source_measure": total,
                "raw_total_source_residual": total - 1,
                "raw_event_measure_diagnostic_only": raw_evidence,
                "event_measure": raw_evidence if known else None,
                "conditional_atoms": selected,
                "joint_future_cells": cells,
                "population_material": expectation("population_material"),
                "population_payment": expectation("population_payment"),
                "all_support_bound": required,
                "row_limit": row_limit,
                "native_source_probability_ledger_actor_RNG_or_service_executor_called": False,
                "source_time_public_reception_native_probability_preferences_service_truth_authenticated": False,
                "lifecycle_utility_CS_total_denial_misallocation": None,
                "primary_default_empirical_production_planner_adopted": False,
                "scientific_budget": None,
            }
        )
    )


def source_future_transport_record(result):
    _check(
        type(result) is SourceFutureTransport and result.rule == RULE,
        "complete typed source transport required",
    )
    return _encoded(result.analyst)
