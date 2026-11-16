"""A.10/A.5/B0/V9/B4: owned registered branch material/utility candidates.

Declared terminal accounts under native myopic choices, not lifecycle CS,
received knowledge, preference identification or an outside intervention law.
"""

import hashlib
import json
from dataclasses import dataclass, fields, replace
from fractions import Fraction

from airline_market.policy.dated_reserve import utc
from airline_market.sim import registered_branch_financial as financial
from airline_market.sim.received_information import _text, _units
from airline_market.sim.shared_mark_attempt import _encoded, _owned

RULE = "owned-registered-branch-utility-v1"
REFERENCE_SCOPE = "same-selected-path-declared-reference-not-outside-intervention"
FIELDS = (
    "net_payment",
    "material",
    "reference_material",
    "decision_noise_excluded",
    "experienced",
    "normalized_noise_excluded",
    "normalized_experienced",
    "additional_utility",
)


def _check(ok, message):
    if not ok:
        raise ValueError(message)


def _sequence(value):
    _check(
        type(value) in (tuple, list), "explicit tuple/list support, no opaque iterator"
    )
    return tuple(value)


class BranchUtilityLimit(RuntimeError):
    """Complete extra named input/output support exceeds infrastructure limit."""


@dataclass(frozen=True)
class BranchUtilityAccount:
    components: tuple
    component_sources: tuple
    additional_utility: Fraction | None
    utility_source: str
    financial_closed: bool
    material_closed: bool
    utility_closed: bool
    closure_sources: tuple
    recorded_at: object
    definition: str
    origin_and_incidence: str

    def __post_init__(self):
        for name in ("components", "component_sources", "closure_sources"):
            object.__setattr__(self, name, _owned(_sequence(getattr(self, name))))
        object.__setattr__(self, "recorded_at", utc(self.recorded_at))


@dataclass(frozen=True)
class BranchPersonUtility:
    person_key: tuple
    account: BranchUtilityAccount
    outside: BranchUtilityAccount
    outside_name: str
    material_utility_nonoverlap: str

    def __post_init__(self):
        object.__setattr__(self, "person_key", _owned(_sequence(self.person_key)))
        _check(
            len(self.person_key) == 3
            and all(type(x) is int and x >= 0 for x in self.person_key),
            "full nonnegative integer original-person key, no Boolean alias",
        )
        for name in ("account", "outside"):
            _check(
                type(getattr(self, name)) is BranchUtilityAccount,
                "typed full actual/reference account",
            )
            object.__setattr__(self, name, replace(getattr(self, name)))


@dataclass(frozen=True)
class BranchUtilityProgramme:
    name: str
    people: tuple

    def __post_init__(self):
        people = _sequence(self.people)
        _check(
            all(type(p) is BranchPersonUtility for p in people),
            "ALL typed original people",
        )
        object.__setattr__(self, "people", tuple(replace(p) for p in people))


@dataclass(frozen=True)
class BranchUtilityContingency:
    financial_context_index: int
    programmes: tuple
    joint_laws: tuple

    def __post_init__(self):
        rows = tuple(_sequence(row) for row in _sequence(self.programmes))
        _check(
            all(type(p) is BranchUtilityProgramme for row in rows for p in row),
            "typed full financial-by-utility programme matrix",
        )
        object.__setattr__(
            self, "programmes", tuple(tuple(replace(p) for p in row) for row in rows)
        )
        object.__setattr__(self, "joint_laws", _owned(_sequence(self.joint_laws)))


@dataclass(frozen=True)
class RegisteredBranchUtility:
    financial: financial.RegisteredBranchFinancial
    programme_names: tuple
    law_definitions: tuple
    contingencies: tuple
    alpha: Fraction
    utility_unit: str
    definition: str
    reference_scope: str
    support_limit: int
    bound: object
    digest: str
    rule: str = RULE

    def __post_init__(self):
        _check(
            type(self.financial) is financial.RegisteredBranchFinancial,
            "typed complete financial registration",
        )
        object.__setattr__(self, "financial", replace(self.financial))
        for name in ("programme_names", "law_definitions"):
            object.__setattr__(self, name, _owned(_sequence(getattr(self, name))))
        rows = _sequence(self.contingencies)
        _check(
            all(type(r) is BranchUtilityContingency for r in rows),
            "typed complete prospective utility contexts",
        )
        object.__setattr__(self, "contingencies", tuple(replace(r) for r in rows))
        object.__setattr__(self, "bound", _owned(self.bound))


@dataclass(frozen=True)
class RegisteredBranchUtilityResult:
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
            _encoded(_payload(reg)), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _account(account, asof):
    _check(type(account) is BranchUtilityAccount, "typed account required")
    _check(
        len(account.components) == 4
        and len(account.component_sources) == 4
        and len(account.closure_sources) == 3,
        "FOUR complete components/sources and THREE closure sources required",
    )
    _check(
        all(
            x is None or type(x) is Fraction
            for x in (*account.components, account.additional_utility)
        ),
        "explicit rational-or-unknown components; no scalar coercion",
    )
    _check(
        all(
            type(x) is bool
            for x in (
                account.financial_closed,
                account.material_closed,
                account.utility_closed,
            )
        ),
        "independent Boolean financial/material/utility closure",
    )
    _check(
        account.financial_closed or account.components[1] is None,
        "open financial tail requires unknown complete payment",
    )
    _check(
        account.recorded_at == asof,
        "ALL accounts share explicit financial common-as-of",
    )
    for text in (
        *account.component_sources,
        *account.closure_sources,
        account.utility_source,
        account.definition,
        account.origin_and_incidence,
    ):
        _text(text)


def _person_shape(key):
    return {"person_key": key, **{field: Fraction() for field in FIELDS}}


def _row_shape():
    return {
        "law": "law",
        "service_atom": 0,
        "financial_index": 0,
        "utility_index": 0,
        "report_index": 0,
        "source_service_mass": Fraction(),
        "conditional_joint": Fraction(),
        "raw_joint_mass": Fraction(),
    }


def _summary_shape(law, event):
    return {
        "law": law,
        "event": event,
        "raw_evidence": Fraction(),
        "fields": {
            field: {
                "positive_unknown_mass": Fraction(),
                "known_numerator_diagnostic": Fraction(),
                "integral": Fraction(),
                "raw_measure_conditional_ratio": Fraction(),
            }
            for field in FIELDS
        },
    }


def _validate(reg, *, verify):
    _check(
        type(reg) is RegisteredBranchUtility and reg.rule == RULE,
        "current complete typed utility registration",
    )
    _check(
        type(reg.support_limit) is int and reg.support_limit > 0,
        "positive named support limit",
    )
    if verify:
        _check(
            reg.digest == _digest(reg), "complete utility registration digest differs"
        )
    input_units = _units(_encoded(_payload(reg)))
    if input_units > reg.support_limit:
        raise BranchUtilityLimit(
            "full owned input exceeds limit before source traversal/forwards"
        )
    _, prepared, parent_bound = financial._validate(reg.financial, verify=True)
    names = financial._named(reg.programme_names)
    _check(
        all(type(row) is tuple and len(row) == 2 for row in reg.law_definitions),
        "explicit complete law name-definition pairs required",
    )
    law_names = financial._named(tuple(row[0] for row in reg.law_definitions))
    for row in reg.law_definitions:
        _check(len(row) == 2, "every named joint-law definition required")
        _text(row[1])
    _check(
        type(reg.alpha) is Fraction and reg.alpha > 0,
        "explicit positive rational utility-per-material conversion",
    )
    for text in (reg.utility_unit, reg.definition):
        _text(text)
    _check(
        reg.utility_unit != reg.financial.unit
        and reg.reference_scope == REFERENCE_SCOPE,
        "distinct utility unit and same-selected-path reference scope",
    )
    _check(
        len(reg.contingencies) == len(reg.financial.contingencies),
        "ALL prospective contexts including unused required",
    )
    parent_rows = sorted(prepared.values(), key=lambda x: x[2])
    reports = people_slots = shape_units = 0
    for index, (context, (parent, people, parent_index)) in enumerate(
        zip(reg.contingencies, parent_rows)
    ):
        _check(
            type(context.financial_context_index) is int
            and context.financial_context_index == index == parent_index,
            "ALL exact ordered context indices, no Boolean alias",
        )
        _check(
            len(context.programmes) == len(parent.settlements)
            and len(context.joint_laws) == len(law_names),
            "complete financial programme and joint-law axes",
        )
        for matrix in context.joint_laws:
            _check(
                type(matrix) is tuple
                and len(matrix) == len(parent.settlements)
                and all(
                    type(row) is tuple and len(row) == len(names) for row in matrix
                ),
                "ALL financial-by-utility joint-law cells",
            )
            probs = tuple(p for row in matrix for p in row)
            _check(
                all(type(p) is Fraction and p >= 0 for p in probs)
                and sum(probs, Fraction()) == 1,
                "full exact joint distribution sums one without normalization",
            )
        for programmes, settlement in zip(context.programmes, parent.settlements):
            _check(
                tuple(p.name for p in programmes) == names,
                "ALL same ordered utility names in every financial context",
            )
            # Frozen PURE prospective payment arithmetic on complete declared profiles.
            payments = financial.finance._payments(settlement, people)[
                "all_original_people"
            ]
            for programme in programmes:
                _check(
                    tuple(p.person_key for p in programme.people)
                    == tuple(p["person_key"] for p in people),
                    "ALL and ONLY original people including outside in source order",
                )
                for p, payment in zip(programme.people, payments):
                    _account(p.account, reg.financial.financial_asof)
                    _account(p.outside, reg.financial.financial_asof)
                    _text(p.outside_name)
                    _text(p.material_utility_nonoverlap)
                    _check(
                        p.account.components[1] == payment["final_net"]
                        and p.account.financial_closed == payment["financial_closed"],
                        "explicit P/financial closure must match registered complete financial account",
                    )
                shape = {
                    "service_atom": 0,
                    "financial_context_index": index,
                    "financial_index": 0,
                    "utility_index": 0,
                    "all_original_people": tuple(
                        _person_shape(p.person_key) for p in programme.people
                    ),
                    "population": {field: Fraction() for field in FIELDS},
                }
                shape_units += _units(_encoded(shape))
                reports += 1
                people_slots += len(programme.people)
    rows = reports * len(law_names)
    events = tuple(row[0] for row in reg.financial.event_definitions)
    counts = {
        "parent_financial_named_bound": parent_bound["total"],
        "owned_and_repeated_extra_input_nodes_upper": 3 * input_units,
        "all_prospective_report_shape_nodes_upper": shape_units,
        "all_original_person_slots_upper": people_slots,
        "all_raw_and_public_joint_row_nodes_upper": 2
        * rows
        * _units(_encoded(_row_shape())),
        "all_law_event_field_accumulation_terms_upper": rows
        * len(events)
        * len(FIELDS),
        "all_summary_nodes_upper": sum(
            _units(_encoded(_summary_shape(law, event)))
            for law in law_names
            for event in events
        ),
        "all_report_count_upper": reports,
    }
    bound = {
        **counts,
        "total": sum(counts.values()) + _units(_encoded({**counts, "total": 0})),
        "complete_named_extra_input_output_before_parent_forward": True,
        "every_serializer_node_bytes_RSS_scratch_F9_scientific_scalability_bound": False,
    }
    if bound["total"] > reg.support_limit:
        raise BranchUtilityLimit(
            "ALL named source/input/output/law/event/person support exceeds limit before forwards"
        )
    if verify:
        _check(
            financial.finance._same(reg.bound, bound),
            "stored bound differs from full recomputed support",
        )
    return _owned(bound)


def register_branch_utility(
    financial_registration,
    programme_names,
    law_definitions,
    contingencies,
    *,
    alpha,
    utility_unit,
    definition,
    reference_scope,
    support_limit,
):
    """Own complete additional programmes before source; not external preregistration."""
    reg = RegisteredBranchUtility(
        financial_registration,
        programme_names,
        law_definitions,
        contingencies,
        alpha,
        utility_unit,
        definition,
        reference_scope,
        support_limit,
        {},
        "",
    )
    bound = _validate(reg, verify=False)
    return replace(reg, bound=bound, digest=_digest(reg))


def _sum(values):
    values = tuple(values)
    return None if any(x is None for x in values) else sum(values, Fraction())


def _material(account):
    b, p, w, c = account.components
    if (
        not account.financial_closed
        or not account.material_closed
        or any(x is None for x in (b, p, w, c))
    ):
        return None
    return b - p - w - c


def _values(person, alpha):
    a, r = person.account, person.outside
    z, z0 = _material(a), _material(r)
    az, az0 = None if z is None else alpha * z, None if z0 is None else alpha * z0
    e, e0 = (
        a.additional_utility if a.utility_closed else None,
        r.additional_utility if r.utility_closed else None,
    )
    experienced = None if az is None or e is None else az + e
    reference = None if az0 is None or e0 is None else az0 + e0
    return {
        "person_key": person.person_key,
        "net_payment": a.components[1],
        "material": z,
        "reference_material": z0,
        "decision_noise_excluded": az,
        "experienced": experienced,
        "normalized_noise_excluded": None if az is None or az0 is None else az - az0,
        "normalized_experienced": None
        if experienced is None or reference is None
        else experienced - reference,
        "additional_utility": e,
    }


def _fraction(value):
    _check(
        type(value) is dict and set(value) == {"numerator", "denominator"},
        "complete supplied exact mass envelope",
    )
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def execute_registered_branch_utility(registration):
    """One frozen registered financial route, then explicit candidate accounts only."""
    bound = _validate(registration, verify=True)
    result = financial.registered_branch_financial_record(
        financial.execute_registered_branch_financial(registration.financial)
    )
    _check(
        financial.finance._same(result["registration"], registration.financial),
        "entire returned financial registration differs",
    )
    parent = result["financial"]
    first_law = parent["laws"][0]["name"]
    masses = {
        row["service_atom"]: _fraction(row["source_service_mass"])
        for row in parent["raw_law_rows"]
        if row["law"] == first_law
    }
    reports, rows = [], []
    for attachment in result["attachments"]:
        atom, index = (
            attachment["service_atom_index"],
            attachment["registered_context_index"],
        )
        context = registration.contingencies[index]
        for fi, programmes in enumerate(context.programmes):
            payment = parent["payment_reports"][atom * len(context.programmes) + fi]
            for mi, programme in enumerate(programmes):
                _check(
                    tuple(p.person_key for p in programme.people)
                    == tuple(
                        tuple(p["person_key"]) for p in payment["all_original_people"]
                    ),
                    "actual ALL original-person support differs",
                )
                _check(
                    all(
                        financial.finance._same(
                            p.account.components[1], actual["final_net"]
                        )
                        and p.account.financial_closed == actual["financial_closed"]
                        for p, actual in zip(
                            programme.people, payment["all_original_people"]
                        )
                    ),
                    "actual complete financial payment differs from prospective account",
                )
                people = tuple(_values(p, registration.alpha) for p in programme.people)
                report_index = len(reports)
                reports.append(
                    {
                        "service_atom": atom,
                        "financial_context_index": index,
                        "financial_index": fi,
                        "utility_index": mi,
                        "all_original_people": people,
                        "population": {
                            field: _sum(p[field] for p in people) for field in FIELDS
                        },
                    }
                )
                for li, (law, _) in enumerate(registration.law_definitions):
                    probability = context.joint_laws[li][fi][mi]
                    rows.append(
                        {
                            "law": law,
                            "service_atom": atom,
                            "financial_index": fi,
                            "utility_index": mi,
                            "report_index": report_index,
                            "source_service_mass": masses[atom],
                            "conditional_joint": probability,
                            "raw_joint_mass": masses[atom] * probability,
                        }
                    )
    summaries = []
    for law, _ in registration.law_definitions:
        for ei, (event, _) in enumerate(registration.financial.event_definitions):
            selected = tuple(
                row
                for row in rows
                if row["law"] == law
                and registration.financial.contingencies[
                    reports[row["report_index"]]["financial_context_index"]
                ].event_flags[ei]
            )
            evidence = sum((r["raw_joint_mass"] for r in selected), Fraction())
            values = {}
            for field in FIELDS:
                unknown = sum(
                    (
                        r["raw_joint_mass"]
                        for r in selected
                        if reports[r["report_index"]]["population"][field] is None
                    ),
                    Fraction(),
                )
                numerator = sum(
                    (
                        r["raw_joint_mass"]
                        * reports[r["report_index"]]["population"][field]
                        for r in selected
                        if reports[r["report_index"]]["population"][field] is not None
                    ),
                    Fraction(),
                )
                value = (
                    numerator
                    if parent["source_public_joint_known"] and unknown == 0
                    else None
                )
                values[field] = {
                    "positive_unknown_mass": unknown,
                    "known_numerator_diagnostic": numerator,
                    "integral": value,
                    "raw_measure_conditional_ratio": None
                    if value is None or evidence == 0
                    else value / evidence,
                }
            summaries.append(
                {"law": law, "event": event, "raw_evidence": evidence, "fields": values}
            )
    return RegisteredBranchUtilityResult(
        _encoded(
            {
                "plan_id": "A.10/A.5/B0/V9/B4",
                "rule": RULE,
                "registration": registration,
                "complete_financial_result": result,
                "reports": tuple(reports),
                "raw_joint_rows": tuple(rows),
                "public_joint_rows": tuple(rows)
                if parent["source_public_joint_known"]
                else None,
                "summaries": tuple(summaries),
                "recomputed_named_support": bound,
                "one_registered_financial_route_delegated": True,
                "native_choices_or_private_marks_reinterpreted": False,
                "source_time_preregistration_native_probability_knowledge_service_payment_truth_preferences_authenticated": False,
                "lifecycle_CS_total_denial_misallocation_penalty_incidence": None,
                "full_primary_default_empirical_production_planner_adopted": False,
                "scientific_budget": None,
            }
        )
    )


def registered_branch_utility_record(value):
    _check(
        type(value) in (RegisteredBranchUtility, RegisteredBranchUtilityResult),
        "complete typed utility registration/result",
    )
    return _encoded(value if type(value) is RegisteredBranchUtility else value.analyst)
