"""A.10/A.5/A.14/B0/V9/B4: explicit realized candidate utility accounts.

Pure supplied material inputs. No native shock interpretation, decision law,
probability expectation, causal outside intervention or lifecycle CS adoption.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from airline_market.core.snapshot import deep_freeze
from airline_market.policy.dated_reserve import utc
from airline_market.sim.passenger_accounts import _encode, _number
from airline_market.sim.payoff_logit import ContingentPayoff
from airline_market.sim.person_material_payoff import (
    COMPONENTS,
    SCOPE,
    SHOCK_RULE,
    PersonMaterialPayoffReport,
    person_material_payoff_record,
)
from airline_market.sim.service_ticket_payments import _rational

RULE = "declared-realized-person-utility-incidence-v1"
REFERENCE_SCOPE = "same-realized-path-declared-reference-not-outside-intervention"


@dataclass(frozen=True)
class DeclaredOutsideReference:
    name: str
    payoff: ContingentPayoff
    component_sources: tuple
    additional_utility: Fraction | None
    additional_utility_source: str
    no_further_financial_components: bool
    no_further_material_components: bool
    no_further_utility_components: bool
    followup_source: str
    definition: str

    def __post_init__(self):
        object.__setattr__(
            self, "component_sources", tuple(tuple(v) for v in self.component_sources)
        )


@dataclass(frozen=True)
class PersonUtilityIncidence:
    person_key: tuple
    additional_utility: Fraction | None
    additional_utility_source: str
    no_further_utility_components: bool
    followup_source: str
    outside: DeclaredOutsideReference
    recorded_at: object
    material_additional_nonoverlap: str
    utility_origin_and_incidence: str

    def __post_init__(self):
        object.__setattr__(self, "person_key", tuple(self.person_key))
        object.__setattr__(self, "recorded_at", utc(self.recorded_at))


@dataclass(frozen=True)
class PersonUtilityReport:
    data: object

    def __post_init__(self):
        object.__setattr__(self, "data", deep_freeze(self.data))


def _text(x):
    return type(x) is str and bool(x.strip())


def _sum(xs):
    xs = tuple(xs)
    return None if any(x is None for x in xs) else sum(xs, Fraction(0))


def _minus(a, b):
    return None if a is None or b is None else a - b


def _scale(alpha, z):
    return None if z is None else alpha * z


def _plus(a, b):
    return None if a is None or b is None else a + b


def _count(x):
    if isinstance(x, Mapping):
        return 1 + sum(_count(k) + _count(v) for k, v in x.items())
    if isinstance(x, (tuple, list)):
        return 1 + sum(_count(v) for v in x)
    if x is None or type(x) in (str, int, bool, float):
        return 1
    raise TypeError("owned primitive complete context required")


def _sources(rows):
    sources = dict(rows)
    if (
        len(sources) != len(rows)
        or set(sources) != set(COMPONENTS)
        or any(not _text(v) for v in sources.values())
    ):
        raise ValueError(
            "explicit sources for ALL FOUR components including unknowns required"
        )
    return tuple((k, sources[k]) for k in COMPONENTS)


def _material_source(material):
    if type(material) is not PersonMaterialPayoffReport:
        raise TypeError("NEW complete typed original-person material report required")
    s = material.data
    if s["scope"] != SCOPE or s["shock_rule"] != SHOCK_RULE:
        raise ValueError(
            "material account must explicitly exclude private-choice shocks"
        )
    payment = s["complete_payment_input"]
    if s["payoff_unit"] != payment["native_ticket_unit"]:
        raise ValueError("common material/payment unit differs")
    at = utc(payment["as_of"])
    people = tuple(s["all_original_persons"])
    pay_people = tuple(payment["all_original_persons"])
    originals = tuple(payment["complete_service_input"]["journal_context"]["persons"])
    keys = tuple(p["person_key"] for p in people)
    if (
        len(set(keys)) != len(keys)
        or keys != tuple(p["person_key"] for p in pay_people)
        or keys != tuple(p["person_key"] for p in originals)
    ):
        raise ValueError("ALL original people in complete source order required")
    values, closed_payments = [], []
    for p, pay, original in zip(people, pay_people, originals):
        key = p["person_key"]
        if len(key) != 3 or any(type(v) is not int or v < 0 for v in key):
            raise ValueError("original potential key required")
        for name in ("original_record", "current_ticket"):
            if p[name] != pay[name] or pay[name] != original[name]:
                raise ValueError("complete original/current ticket linkage differs")
        if (
            p["booking_state"] != pay["booking_state"]
            or utc(p["components_recorded_at"]) != at
            or utc(pay["observed_through"]) != at
        ):
            raise ValueError("complete source identity/common-as-of differs")
        for name in (
            "no_further_material_components_declared",
            "no_further_ticket_transfers_declared",
        ):
            if type(p[name]) is not bool:
                raise ValueError("explicit material and financial closures required")
        if (
            p["no_further_ticket_transfers_declared"]
            != pay["no_further_ticket_transfers_declared"]
            or type(pay["no_further_ticket_transfers_declared"]) is not bool
        ):
            raise ValueError("financial closure differs")
        observed = _rational(pay["observed_net_ticket_payment"])
        closed = _rational(pay["declared_closed_total_ticket_payment"])
        if closed != (
            observed if pay["no_further_ticket_transfers_declared"] else None
        ):
            raise ValueError("financial tail/closed payment arithmetic differs")
        closed_payments.append(closed)
        raw = p["four_explicit_components"]
        if set(raw) != {*COMPONENTS, "net_value"}:
            raise ValueError("full FOUR-component material context required")
        payoff = ContingentPayoff(*(_rational(raw[k]) for k in COMPONENTS))
        if (
            payoff.net_ticket_payment != closed
            or raw["net_value"] != _number(payoff.net_value())
            or p["declared_component_arithmetic_diagnostic"]
            != _number(payoff.net_value())
        ):
            raise ValueError("saved material/payment arithmetic differs")
        _sources(tuple(p["component_sources"]))
        if (
            not _text(p["followup_source_ref"])
            or not p["declared_decision_information"]
        ):
            raise ValueError(
                "full supplied material follow-up/information context required"
            )
        final = (
            payoff.net_value()
            if p["no_further_material_components_declared"]
            and p["no_further_ticket_transfers_declared"]
            else None
        )
        if p["declared_closed_material_payoff"] != _number(final):
            raise ValueError("saved closed material value differs")
        values.append(final)
    if s["declared_closed_population_material_payoff"] != _number(
        _sum(values)
    ) or payment["declared_closed_population_total_ticket_payment"] != _number(
        _sum(closed_payments)
    ):
        raise ValueError("full saved population arithmetic differs")
    return s, people, at, tuple(values)


def reconcile_person_utility(
    material,
    declarations,
    *,
    alpha,
    material_unit,
    utility_unit,
    reference_scope,
    definition,
    row_limit,
):
    """Realized declared accounts on ALL original people, no probability law.

    Supplied source values are structurally checked without upstream producer
    replay. No typed container/hash authenticates its source or utility meaning.
    """
    if type(alpha) is not Fraction or alpha <= 0:
        raise ValueError(
            "explicit positive rational utility/material conversion required"
        )
    if (
        not all(_text(v) for v in (material_unit, utility_unit, definition))
        or material_unit == utility_unit
    ):
        raise ValueError("explicit distinct material and utility units required")
    if reference_scope != REFERENCE_SCOPE:
        raise ValueError(
            "explicit same-path reference scope; no causal outside law inferred"
        )
    if type(row_limit) is not int or row_limit < 1:
        raise ValueError("positive complete structural infrastructure limit required")
    s, people, at, material_values = _material_source(material)
    if material_unit != s["payoff_unit"]:
        raise ValueError(
            "material unit must exactly match supplied material/payment source"
        )
    declarations = tuple(declarations)
    if any(type(d) is not PersonUtilityIncidence for d in declarations):
        raise TypeError("typed utility declaration for EVERY original person required")
    by_key = {d.person_key: d for d in declarations}
    if len(by_key) != len(declarations) or set(by_key) != {
        p["person_key"] for p in people
    }:
        raise ValueError(
            "one explicit account per ALL original people including outside/unreserved required"
        )
    declaration_records = []
    for d in declarations:
        o = d.outside
        if (
            type(o) is not DeclaredOutsideReference
            or type(o.payoff) is not ContingentPayoff
        ):
            raise TypeError("FULL explicit outside reference payoff required")
        if d.recorded_at != at or any(
            type(v) is not bool
            for v in (
                d.no_further_utility_components,
                o.no_further_financial_components,
                o.no_further_material_components,
                o.no_further_utility_components,
            )
        ):
            raise ValueError(
                "common-as-of and explicit independent utility/reference tail closures required"
            )
        if any(
            not _text(v)
            for v in (
                d.additional_utility_source,
                d.followup_source,
                d.material_additional_nonoverlap,
                d.utility_origin_and_incidence,
                o.name,
                o.additional_utility_source,
                o.followup_source,
                o.definition,
            )
        ):
            raise ValueError(
                "explicit sources/non-overlap/origin/incidence/reference definitions required"
            )
        if any(
            v is not None and type(v) is not Fraction
            for v in (d.additional_utility, o.additional_utility)
        ):
            raise TypeError("explicit rational-or-unknown additional utility required")
        declaration_records.append(
            {
                "person_key": d.person_key,
                "recorded_at": d.recorded_at.isoformat(),
                "additional_utility": _number(d.additional_utility),
                "additional_utility_source": d.additional_utility_source,
                "no_further_utility_components": d.no_further_utility_components,
                "followup_source": d.followup_source,
                "material_additional_nonoverlap": d.material_additional_nonoverlap,
                "utility_origin_and_incidence": d.utility_origin_and_incidence,
                "outside": {
                    "name": o.name,
                    "four_components": o.payoff.record(),
                    "component_sources": _sources(o.component_sources),
                    "additional_utility": _number(o.additional_utility),
                    "additional_utility_source": o.additional_utility_source,
                    "no_further_financial_components": o.no_further_financial_components,
                    "no_further_material_components": o.no_further_material_components,
                    "no_further_utility_components": o.no_further_utility_components,
                    "followup_source": o.followup_source,
                    "definition": o.definition,
                },
            }
        )
    source_record = person_material_payoff_record(material)
    context = {
        "rule": RULE,
        "plan_id": "A.10/A.5/A.14/B0/V9/B4",
        "definition": definition,
        "alpha": _number(alpha),
        "material_unit": material_unit,
        "utility_unit": utility_unit,
        "reference_scope": reference_scope,
        "complete_material_input": source_record,
        "declarations": declaration_records,
        "row_limit": row_limit,
    }
    # Conservative complete proposed-output bound: context twice plus fixed
    # per-person result envelopes/labels, checked BEFORE utility forwards.
    bound = 2 * _count(context) + 512 * len(people) + 512
    if bound > row_limit:
        raise RuntimeError(
            "full source/declaration/proposed utility output exceeds structural limit; no partial/fallback/retry"
        )
    records = {tuple(d["person_key"]): d for d in declaration_records}
    output = []
    names = (
        "decision_noise_excluded",
        "outside_decision_noise_excluded",
        "normalized_decision_noise_excluded",
        "experienced_candidate",
        "outside_experienced_candidate",
        "normalized_experienced_candidate",
    )
    totals = {k: [] for k in names}
    for p, z in zip(people, material_values):
        d = by_key[p["person_key"]]
        o = d.outside
        z0 = (
            o.payoff.net_value()
            if o.no_further_financial_components and o.no_further_material_components
            else None
        )
        v = _scale(alpha, z)
        v0 = _scale(alpha, z0)
        experienced = (
            _plus(v, d.additional_utility) if d.no_further_utility_components else None
        )
        outside_experienced = (
            _plus(v0, o.additional_utility) if o.no_further_utility_components else None
        )
        values = (
            v,
            v0,
            _minus(v, v0),
            experienced,
            outside_experienced,
            _minus(experienced, outside_experienced),
        )
        row = {
            "person_key": p["person_key"],
            "original_record": p["original_record"],
            "current_ticket": p["current_ticket"],
            "booking_state": p["booking_state"],
            "declaration": records[p["person_key"]],
            "declared_closed_material_payoff": _number(z),
            "declared_closed_reference_material_payoff": _number(z0),
            "known_component_experience_diagnostic": _number(
                _plus(
                    _scale(
                        alpha, _rational(p["declared_component_arithmetic_diagnostic"])
                    ),
                    d.additional_utility,
                )
            ),
            "lifecycle_consumer_surplus_or_adopted_experienced_utility": None,
        }
        for name, value in zip(names, values):
            row[name] = _number(value)
            totals[name].append(value)
        output.append(row)
    result = {
        **context,
        "all_original_persons": output,
        "declared_population_sums": {k: _number(_sum(v)) for k, v in totals.items()},
        "all_support_structural_bound": bound,
        "source_or_nonoverlap_tail_closure_utility_meaning_authenticated": False,
        "decision_information_or_native_actor_observation_extended": False,
        "native_choice_shocks_automatically_included": False,
        "probability_expectation_or_causal_outside_comparison": None,
        "native_lifecycle_CS_total_denial_misallocation": None,
        "primary_default_empirical_production_planner_adopted": False,
        "scientific_budget": None,
    }
    if _count(result) > bound:
        raise RuntimeError(
            "complete result exceeds preflight structural bound; no positive partial result"
        )
    return PersonUtilityReport(result)


def person_utility_record(report):
    if type(report) is not PersonUtilityReport:
        raise TypeError("typed realized candidate utility report required")
    return _encode(report.data)
