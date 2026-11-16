"""A.10/A.5/B0/V9/B4: owned prospective finite branch financial programmes.

Local registration/order is not external preregistration, native probability,
received knowledge, service/payment truth, preference or lifecycle CS adoption.
"""

import hashlib
import itertools
import json
from dataclasses import dataclass, fields, replace
from datetime import datetime
from fractions import Fraction

from airline_market.policy.dated_reserve import utc
from airline_market.sim import bernoulli_service as service
from airline_market.sim import branch_financial as finance
from airline_market.sim import registered_bernoulli as source_contract
from airline_market.sim.bernoulli_future import bernoulli_future_expectation
from airline_market.sim.received_information import _text, _units
from airline_market.sim.shared_mark_attempt import _encoded, _owned

RULE = "owned-finite-branch-service-financial-registration-v1"


def _check(value, message):
    if not value:
        raise ValueError(message)


def _sequence(value):
    _check(
        type(value) in (tuple, list),
        "explicit tuple/list support required; no opaque iterator",
    )
    return tuple(value)


def _named(values):
    values = _owned(_sequence(values))
    _check(
        values and len(set(values)) == len(values),
        "complete distinct ordered names required",
    )
    for name in values:
        _text(name)
    return values


class BranchRegistrationLimit(RuntimeError):
    """Complete named source/service/financial support exceeds infrastructure limit."""


@dataclass(frozen=True)
class BranchFinancialContingency:
    path: str
    branches: tuple
    active: tuple
    service_programme: str
    current_service: tuple
    settlements: tuple
    law_weights: tuple
    event_flags: tuple

    def __post_init__(self):
        for name in (
            "branches",
            "active",
            "current_service",
            "law_weights",
            "event_flags",
        ):
            object.__setattr__(self, name, _owned(_sequence(getattr(self, name))))
        values = _sequence(self.settlements)
        _check(
            all(type(x) is finance.BranchSettlement for x in values),
            "typed complete branch settlements required",
        )
        object.__setattr__(self, "settlements", tuple(replace(x) for x in values))


@dataclass(frozen=True)
class RegisteredBranchFinancial:
    recipe: source_contract.BernoulliSourceRecipe
    clock: service.BranchServiceClock
    service_names: tuple
    service_cases: tuple
    financial_names: tuple
    law_definitions: tuple
    event_definitions: tuple
    contingencies: tuple
    financial_asof: object
    unit: str
    definition: str
    service_definition: str
    support_limit: int
    bound: object
    digest: str
    rule: str = RULE

    def __post_init__(self):
        _check(
            type(self.recipe) is source_contract.BernoulliSourceRecipe
            and type(self.clock) is service.BranchServiceClock,
            "typed complete source recipe and branch clock required",
        )
        object.__setattr__(self, "recipe", replace(self.recipe))
        object.__setattr__(self, "clock", replace(self.clock))
        for name in (
            "service_names",
            "financial_names",
            "law_definitions",
            "event_definitions",
        ):
            object.__setattr__(self, name, _owned(_sequence(getattr(self, name))))
        for name, kind in (
            ("service_cases", service.BranchServiceCase),
            ("contingencies", BranchFinancialContingency),
        ):
            values = _sequence(getattr(self, name))
            _check(
                all(type(x) is kind for x in values),
                "complete typed " + name + " required",
            )
            object.__setattr__(self, name, tuple(replace(x) for x in values))
        object.__setattr__(self, "financial_asof", utc(self.financial_asof))
        object.__setattr__(self, "bound", _owned(self.bound))


@dataclass(frozen=True)
class RegisteredBranchFinancialResult:
    analyst: object

    def __post_init__(self):
        object.__setattr__(self, "analyst", _owned(self.analyst))


def _payload(reg):
    return {
        f.name: getattr(reg, f.name)
        for f in fields(reg)
        if f.name not in ("bound", "digest")
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


def _coordinate(row):
    return (
        row.path,
        row.branches,
        row.active,
        row.service_programme,
        row.current_service,
    )


def _people(c, path, branches, active, start):
    people, rid = [], 0
    for offset, (market, segment, taste_class, j) in enumerate(branches):
        if market is None:
            continue
        purchase = j != len(c.spec.products)
        fare = (
            Fraction.from_float(path.opportunities[offset].prices[j])
            if purchase
            else None
        )
        kept = purchase and active[rid]
        people.append(
            {
                "person_key": (start + offset, 0, market),
                "segment": segment,
                "taste_class": taste_class,
                "original_ticket": rid if purchase else None,
                "current_ticket": rid if kept else None,
                "carrier": c.spec.products[j].carrier if purchase else None,
                "original_fare": fare,
                "native_source_refund": fare if purchase and not kept else Fraction(),
                "native_net_ticket_payment": fare if kept else Fraction(),
            }
        )
        rid += int(purchase)
    return tuple(people)


def _payment_shape(people, settlement):
    return {
        "service_atom": 0,
        "programme": settlement.name,
        "all_original_people": tuple(
            {
                "person_key": p["person_key"],
                "original_ticket": p["original_ticket"],
                "current_ticket": p["current_ticket"],
                "native_fare": p["original_fare"],
                "native_refund_once": p["native_source_refund"],
                "native_net": p["native_net_ticket_payment"],
                "additional_signed_net_terms": tuple(
                    Fraction()
                    for e in settlement.transfers
                    if e.person_key == p["person_key"]
                ),
                "recorded_net": Fraction(),
                "financial_closed": True,
                "final_net": Fraction(),
            }
            for p in people
        ),
        "population_recorded_net": Fraction(),
        "population_final_net": Fraction(),
    }


def _source_bound(reg, c, catalogue, T):
    rows, paths = len(catalogue), len(reg.recipe.paths)
    frontier, calls = 1, 0
    for _ in range(T):
        calls += paths * frontier
        if (
            calls > min(reg.support_limit, reg.recipe.conditional_call_limit)
            or calls * rows > reg.recipe.tree_row_limit
        ):
            raise BranchRegistrationLimit(
                "ALL source-tree row/call support exceeds limits before forwards"
            )
        frontier *= rows
    _check(T <= reg.recipe.ticket_limit, "ALL future ticket support exceeds limit")
    purchase = sum(x[0] is not None and x[3] != len(c.spec.products) for x in catalogue)
    cases = paths * (rows + purchase) ** T
    if cases > reg.support_limit:
        raise BranchRegistrationLimit("ALL syntactic/status support exceeds limit")
    leaves = paths * rows**T
    return {
        "source_conditionals_upper": calls,
        "source_branch_rows_upper": calls * rows,
        "source_terminal_histories_upper": leaves,
        "source_ticket_slots_upper": (calls * rows + leaves) * T,
        "source_leg_stock_slots_upper": (calls * rows + leaves) * len(c.spec.legs),
        "source_carrier_account_slots_upper": (calls * rows + leaves)
        * len(c.spec.carriers)
        * 5,
        "source_kernel_product_columns_upper": calls
        * len(c.spec.markets)
        * sum(len(g.classes) for g in c.spec.segments)
        * (len(c.spec.products) + 1),
        "source_repeated_model_metadata_nodes_upper": _units(_encoded(c.spec))
        * (1 + 2 * calls),
        "full_source_status_cases": cases,
    }


def _validate(reg, *, verify):
    _check(
        type(reg) is RegisteredBranchFinancial and reg.rule == RULE,
        "current complete typed registration required",
    )
    _check(
        type(reg.support_limit) is int and reg.support_limit > 0,
        "positive complete named support limit required",
    )
    if verify:
        _check(
            reg.digest == _digest(reg),
            "full registration digest differs; no attachments or numerical edits",
        )
    input_units = _units(_encoded(_payload(reg)))
    if input_units > reg.support_limit:
        raise BranchRegistrationLimit(
            "complete owned input exceeds limit before source traversal/forwards"
        )
    # Frozen PURE canonical model/empty BASE/source programme validator only.
    c, catalogue, T = source_contract._source(reg)
    _check(
        all(len(p.legs) == 1 for p in c.spec.products),
        "direct single-required-leg products only",
    )
    service_names, financial_names = (
        _named(reg.service_names),
        _named(reg.financial_names),
    )
    law_names = _named(tuple(x[0] for x in reg.law_definitions))
    event_names = _named(tuple(x[0] for x in reg.event_definitions))
    for definitions in (reg.law_definitions, reg.event_definitions):
        for row in definitions:
            _check(len(row) == 2, "every named law/event definition required")
            _text(row[1])
    for text in (
        reg.unit,
        reg.definition,
        reg.service_definition,
        reg.clock.definition,
    ):
        _text(text)
    legs = tuple(x.id for x in c.spec.legs)
    _check(
        tuple(x[0] for x in reg.clock.leg_departures) == legs
        and all(t > reg.clock.source_close for _, t in reg.clock.leg_departures),
        "full ordered future departures after declared source close required",
    )
    _check(
        reg.financial_asof >= reg.clock.source_close,
        "financial as-of before source close",
    )
    source_bound = _source_bound(reg, c, catalogue, T)
    _check(
        len(reg.service_cases) == source_bound["full_source_status_cases"],
        "ALL prospective source-status cases including unused required",
    )
    case_index = contingency_index = service_units = payment_units = person_slots = (
        transfer_slots
    ) = 0
    prepared = {}
    for path in reg.recipe.paths:
        for branches in itertools.product(catalogue, repeat=T):
            purchases = sum(
                b[0] is not None and b[3] != len(c.spec.products) for b in branches
            )
            for active in itertools.product((False, True), repeat=purchases):
                case = reg.service_cases[case_index]
                case_index += 1
                _check(
                    finance._same(
                        (case.path, case.branches, case.active),
                        (path.name, branches, active),
                    )
                    and all(type(x) is bool for x in case.active),
                    "full ordered syntactic/status coordinate required",
                )
                _check(
                    tuple(x.name for x in case.programmes) == service_names,
                    "ALL same ordered service programmes required",
                )
                source_contract.transport._distribution(
                    tuple(x.probability for x in case.programmes)
                )
                people = _people(c, path, branches, active, reg.recipe.start_period)
                current = tuple(
                    p["current_ticket"]
                    for p in people
                    if p["current_ticket"] is not None
                )
                for programme in case.programmes:
                    plan = service._prepare(
                        c,
                        reg.clock,
                        service._people(c, branches, active, reg.recipe.start_period),
                        programme,
                    )
                    _check(
                        all(
                            reg.financial_asof >= datetime.fromisoformat(at)
                            for at, _, _, _ in plan["events"]
                        ),
                        "as-of covers ALL service programmes including unused",
                    )
                    service_units += _units(_encoded(plan)) + _units(
                        _encoded(service._report_shape(plan))
                    )
                    for outcome in itertools.product(
                        (False, True), repeat=len(current)
                    ):
                        _check(
                            contingency_index < len(reg.contingencies),
                            "omitted hypothetical financial context",
                        )
                        row = reg.contingencies[contingency_index]
                        contingency_index += 1
                        status = dict(zip(current, outcome))
                        statuses = tuple(
                            (p["person_key"], status.get(p["current_ticket"]))
                            for p in people
                        )
                        coordinate = (
                            path.name,
                            branches,
                            active,
                            programme.name,
                            statuses,
                        )
                        _check(
                            finance._same(_coordinate(row), coordinate)
                            and all(type(b) is bool for b in row.active)
                            and all(
                                v is None or type(v) is bool
                                for _, v in row.current_service
                            ),
                            "ALL ordered Boolean/NULL financial outcome contexts required, no missing unused support",
                        )
                        _check(
                            tuple(x.name for x in row.settlements) == financial_names,
                            "ALL same ordered settlements in every context required",
                        )
                        _check(
                            len(row.law_weights) == len(law_names),
                            "ALL conditional financial laws required",
                        )
                        for weights in row.law_weights:
                            _check(
                                len(weights) == len(financial_names),
                                "ALL settlement columns required",
                            )
                            source_contract.transport._distribution(weights)
                        _check(
                            len(row.event_flags) == len(event_names)
                            and all(type(x) is bool for x in row.event_flags),
                            "ALL explicit Boolean analyst events required",
                        )
                        for settlement in row.settlements:
                            finance._prepare(
                                settlement,
                                people,
                                reg.clock.source_close,
                                reg.financial_asof,
                            )
                            payment_units += _units(
                                _encoded(_payment_shape(people, settlement))
                            )
                            person_slots += len(people)
                            transfer_slots += len(settlement.transfers)
                        prepared[coordinate] = (row, people, contingency_index - 1)
    _check(
        contingency_index == len(reg.contingencies),
        "extra hypothetical financial contexts refused",
    )
    financial_contexts = contingency_index
    law_rows = financial_contexts * len(financial_names) * len(law_names)
    law_shape = {
        "law": "law",
        "service_atom": 0,
        "programme": "programme",
        "report_index": 0,
        "source_service_mass": Fraction(),
        "financial_probability": Fraction(),
        "raw_joint_mass": Fraction(),
    }
    summary_shape = {
        "law": "law",
        "event": "event",
        "raw_event_evidence": Fraction(),
        "positive_unknown_mass": Fraction(),
        "known_payment_numerator_diagnostic": Fraction(),
        "net_payment_integral": Fraction(),
        "raw_measure_conditional_payment_ratio": Fraction(),
    }
    counts = {
        **{k: 2 * v for k, v in source_bound.items() if k.startswith("source_")},
        "owned_and_repeated_input_metadata_nodes_upper": 3 * input_units,
        "full_source_status_cases": source_bound["full_source_status_cases"],
        "full_financial_contexts": financial_contexts,
        "all_prepared_service_and_witness_shape_nodes": 2 * service_units,
        "all_financial_report_shape_nodes": payment_units,
        "all_financial_person_slots": person_slots,
        "all_additional_transfer_slots": transfer_slots,
        "all_raw_and_public_law_row_nodes_upper": 2
        * law_rows
        * _units(_encoded(law_shape)),
        "all_law_event_accumulation_terms_upper": law_rows * len(event_names),
        "all_summary_shape_nodes": len(law_names)
        * len(event_names)
        * _units(_encoded(summary_shape)),
        "actual_attachment_slots_upper": source_bound["source_terminal_histories_upper"]
        * len(service_names)
        * 3,
    }
    total = sum(counts.values()) + _units(_encoded({**counts, "total": 0}))
    if total > reg.support_limit:
        raise BranchRegistrationLimit(
            "complete named input/source/service/financial/output support exceeds limit before forwards"
        )
    bound = _owned(
        {
            **counts,
            "total": total,
            "complete_named_domain_shapes_before_any_source_service_financial_forward": True,
            "source_graph_copy_count": 2,
            "input_registration_and_parent_context_copy_count": 3,
            "source_parent_row_call_ticket_limits_also_apply": True,
            "parents_complete_actual_output_bounds_before_each_delegated_forward": True,
            "every_serializer_node_bytes_RSS_scratch_F9_scientific_or_scalability_bound": False,
        }
    )
    if verify:
        _check(
            finance._same(bound, reg.bound),
            "stored complete named bound differs from recomputed support",
        )
    return c, prepared, bound


def register_branch_financial(
    recipe,
    clock,
    service_names,
    service_cases,
    financial_names,
    law_definitions,
    event_definitions,
    contingencies,
    *,
    financial_asof,
    unit,
    definition,
    service_definition,
    support_limit,
):
    """Own ALL prospective programmes before source; not external preregistration."""
    reg = RegisteredBranchFinancial(
        recipe,
        clock,
        service_names,
        service_cases,
        financial_names,
        law_definitions,
        event_definitions,
        contingencies,
        financial_asof,
        unit,
        definition,
        service_definition,
        support_limit,
        {},
        "",
    )
    _, _, bound = _validate(reg, verify=False)
    return replace(reg, bound=bound, digest=_digest(reg))


def _actual_recipe(c, source, recipe):
    d = source.details
    _check(
        finance._same(d["complete_native_model"], c.spec)
        and finance._same(d["initial_declared_ledger"], recipe.ledger)
        and finance._same(
            d["paths"],
            tuple(
                {"name": p.name, "weight": p.weight, "opportunities": p.opportunities}
                for p in recipe.paths
            ),
        ),
        "entire actual source model/ledger/programme differs",
    )
    for name in (
        "start_period",
        "conditioning",
        "mark_law",
        "tree_row_limit",
        "conditional_call_limit",
        "ticket_limit",
    ):
        _check(
            finance._same(d[name], getattr(recipe, name)),
            "actual source backend/information contract differs: " + name,
        )
    _check(
        d["independent_future_private_marks_declared_not_authenticated"] is True,
        "same declared source mark independence required",
    )


def execute_registered_branch_financial(registration):
    """One NEW source/service/financial route after revalidating owned complete input."""
    c, prepared, bound = _validate(registration, verify=True)
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
    _actual_recipe(c, source, r)
    service_result = service.bernoulli_service_programmes(
        c,
        source,
        registration.clock,
        registration.service_names,
        registration.service_cases,
        definition=registration.service_definition,
        support_limit=registration.support_limit,
    )
    record = service.bernoulli_service_record(service_result)
    selected, cells, attachments = [], [], []
    for i, atom in enumerate(record["all_atoms"]):
        terminal = source.details["terminals"][atom["history_index"]]
        branches = tuple(
            tuple(source.details["nodes"][ni]["boundary"]["branches"][ri]["input"][:4])
            for ni, ri in terminal["history"]
        )
        actual_people = record["all_original_person_profiles"][atom["history_index"]]
        kept = {
            p["current_ticket"]
            for p in actual_people
            if p["current_ticket"] is not None
        }
        purchases = sum(
            b[0] is not None and b[3] != len(c.spec.products) for b in branches
        )
        active = tuple(rid in kept for rid in range(purchases))
        coordinate = (
            terminal["path"],
            branches,
            active,
            atom["programme"],
            atom["service"]["current_service"],
        )
        # Parent exports retain primitive tuples; no source-dependent numerical fill.
        coordinate = _owned(coordinate)
        _check(
            coordinate in prepared,
            "actual full service outcome absent from registration",
        )
        row, declared_people, index = prepared[coordinate]
        _check(
            len(actual_people) == len(declared_people)
            and all(
                all(finance._same(actual[k], v) for k, v in declared.items())
                for actual, declared in zip(actual_people, declared_people)
            ),
            "actual original-person ticket/fare/refund baseline differs",
        )
        selected.append(row)
        cells.append(finance.BranchFinancialCell(i, row.settlements))
        attachments.append(
            {
                "source_history_index": atom["history_index"],
                "service_atom_index": i,
                "registered_context_index": index,
            }
        )
    laws = tuple(
        finance.BranchFinancialLaw(
            name, tuple(row.law_weights[j] for row in selected), definition
        )
        for j, (name, definition) in enumerate(registration.law_definitions)
    )
    events = tuple(
        finance.BranchFinancialEvent(
            name, tuple(row.event_flags[j] for row in selected), definition
        )
        for j, (name, definition) in enumerate(registration.event_definitions)
    )
    financial = finance.branch_financial_laws(
        c,
        source,
        registration.clock,
        registration.service_cases,
        service_result,
        registration.financial_names,
        tuple(cells),
        laws,
        events,
        financial_asof=registration.financial_asof,
        unit=registration.unit,
        definition=registration.definition,
        support_limit=registration.support_limit,
    )
    return RegisteredBranchFinancialResult(
        _encoded(
            {
                "plan_id": "A.10/A.5/B0/V9/B4",
                "rule": RULE,
                "registration": registration,
                "attachments": tuple(attachments),
                "financial": finance.branch_financial_record(financial),
                "recomputed_named_support": bound,
                "one_source_one_service_one_financial_route": True,
                "native_business_algorithms_copied_or_changed": False,
                "genuine_panel_journal_or_native_departure_engine_executed": False,
                "source_time_preregistration_native_probability_knowledge_service_payment_truth_preferences_authenticated": False,
                "material_utility_CS_denial_misallocation_penalty_incidence": None,
                "full_primary_default_empirical_production_planner_adopted": False,
                "scientific_budget": None,
            }
        )
    )


def registered_branch_financial_record(value):
    _check(
        type(value) in (RegisteredBranchFinancial, RegisteredBranchFinancialResult),
        "typed complete registration or result required",
    )
    return _encoded(
        value if type(value) is RegisteredBranchFinancial else value.analyst
    )
