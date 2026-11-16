"""A.10/A.5/B0/V9: declared original-person material payoff and information.

Exact supplied component accounting, not experienced utility, behavior or CS.
"""

from dataclasses import dataclass
from fractions import Fraction

from airline_market.core.snapshot import deep_freeze
from airline_market.policy.dated_reserve import utc
from airline_market.sim.passenger_accounts import _encode, _number
from airline_market.sim.payoff_logit import ContingentPayoff
from airline_market.sim.service_ticket_payments import (
    ServiceTicketPayments,
    _rational,
    service_ticket_payment_record,
)

COMPONENTS = (
    "service_benefit",
    "net_ticket_payment",
    "waiting_loss",
    "cancellation_loss",
)
SCOPE = "declared-realized-material-payoff"
SHOCK_RULE = "private-choice-shocks-excluded-from-this-material-account"


@dataclass(frozen=True)
class DeclaredDecisionInformation:
    person_key: tuple
    decision_id: str
    decision_at: object
    service_event_ids: tuple
    transfer_event_ids: tuple
    definition: str
    source_ref: str

    def __post_init__(self):
        object.__setattr__(self, "person_key", tuple(self.person_key))
        object.__setattr__(self, "decision_at", utc(self.decision_at))
        object.__setattr__(self, "service_event_ids", tuple(self.service_event_ids))
        object.__setattr__(self, "transfer_event_ids", tuple(self.transfer_event_ids))


@dataclass(frozen=True)
class OriginalPersonMaterialPayoff:
    person_key: tuple
    payoff: ContingentPayoff
    component_sources: tuple
    components_recorded_at: object
    no_further_material_components: bool
    followup_source_ref: str
    decision_information: tuple

    def __post_init__(self):
        object.__setattr__(self, "person_key", tuple(self.person_key))
        object.__setattr__(
            self, "component_sources", tuple(tuple(v) for v in self.component_sources)
        )
        object.__setattr__(
            self, "components_recorded_at", utc(self.components_recorded_at)
        )
        object.__setattr__(
            self, "decision_information", tuple(self.decision_information)
        )


@dataclass(frozen=True)
class PersonMaterialPayoffReport:
    data: object

    def __post_init__(self):
        object.__setattr__(self, "data", deep_freeze(self.data))


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _key(value):
    return len(value) == 3 and all(type(v) is int and v >= 0 for v in value)


def _sum(values):
    values = tuple(values)
    return None if any(v is None for v in values) else sum(values, Fraction(0))


def reconcile_person_material_payoff(
    payments,
    declarations,
    *,
    payoff_unit,
    definition,
    scope,
    shock_rule,
    row_limit,
):
    """One explicit four-component account per ALL original ordinary people.

    Payoffs are ex post declarations, never supplied as decision information.
    Named own-person disclosures must be recorded STRICTLY before the declared
    decision instant. Equal timestamps have no inferred within-instant order.
    """
    if type(payments) is not ServiceTicketPayments:
        raise TypeError("NEW typed service ticket-payment reconciliation required")
    if not _text(payoff_unit) or not _text(definition):
        raise ValueError("explicit common payoff unit and definition required")
    if scope != SCOPE or shock_rule != SHOCK_RULE:
        raise ValueError(
            "explicit material-only scope/private-choice-shock exclusion required"
        )
    if type(row_limit) is not int or row_limit < 1:
        raise ValueError("positive complete-support infrastructure limit required")
    source = payments.data
    if payoff_unit != source["native_ticket_unit"]:
        raise ValueError(
            "payoff components must use the SAME explicit native ticket unit"
        )
    as_of = utc(source["as_of"])
    people = tuple(source["all_original_persons"])
    by_key = {p["person_key"]: p for p in people}
    ctx = source["complete_service_input"]["journal_context"]
    original = tuple(ctx["persons"])
    keys = tuple(p["person_key"] for p in original)
    if (
        len(by_key) != len(people)
        or tuple(p["person_key"] for p in people) != keys
        or len(set(keys)) != len(keys)
        or any(not _key(k) for k in keys)
    ):
        raise ValueError(
            "ALL original people in complete supplied source order required"
        )
    observed, closed = [], []
    for p, original_person in zip(people, original):
        if (
            p["original_record"] != original_person["original_record"]
            or p["current_ticket"] != original_person["current_ticket"]
            or utc(p["observed_through"]) != as_of
            or type(p["no_further_ticket_transfers_declared"]) is not bool
        ):
            raise ValueError(
                "complete matching original-root/payment follow-up support required"
            )
        o = _rational(p["observed_net_ticket_payment"])
        c = _rational(p["declared_closed_total_ticket_payment"])
        if c != (o if p["no_further_ticket_transfers_declared"] else None):
            raise ValueError("source financial closure and per-person total disagree")
        observed.append(o)
        closed.append(c)
    if _sum(observed) != _rational(
        source["observed_population_net_ticket_payment"]
    ) or _sum(closed) != _rational(
        source["declared_closed_population_total_ticket_payment"]
    ):
        raise ValueError("complete saved population payment arithmetic differs")
    declarations = tuple(declarations)
    if any(type(d) is not OriginalPersonMaterialPayoff for d in declarations):
        raise TypeError(
            "typed explicit declarations for EVERY original person required"
        )
    declared = {d.person_key: d for d in declarations}
    if len(declared) != len(declarations) or set(declared) != set(keys):
        raise ValueError(
            "one explicit payoff per ALL original people including outside/unreserved required"
        )
    rows = (
        source["all_support_rows"]
        + len(declarations)
        + sum(
            len(d.component_sources)
            + len(d.decision_information)
            + sum(
                len(i.service_event_ids) + len(i.transfer_event_ids)
                for i in d.decision_information
                if type(i) is DeclaredDecisionInformation
            )
            for d in declarations
        )
    )
    if rows > row_limit:
        raise RuntimeError(
            "ALL source/component/information rows exceed limit; no partial/fallback/retry"
        )
    service_rows = tuple(source["complete_service_input"]["observations"])
    transfer_rows = tuple(source["additional_transfers"])
    service_events = {r["event_id"]: r for r in service_rows}
    transfer_events = {r["event_id"]: r for r in transfer_rows}
    if len(service_events) != len(service_rows) or len(transfer_events) != len(
        transfer_rows
    ):
        raise ValueError("complete distinct source service/transfer event IDs required")
    output, closed_payoffs = [], []
    for key in keys:
        p, d = by_key[key], declared[key]
        if (
            not _key(d.person_key)
            or type(d.payoff) is not ContingentPayoff
            or d.components_recorded_at != as_of
            or type(d.no_further_material_components) is not bool
            or not _text(d.followup_source_ref)
        ):
            raise ValueError(
                "typed full payoff/common-as-of/explicit material-tail declaration required"
            )
        sources = dict(d.component_sources)
        if (
            len(sources) != len(d.component_sources)
            or set(sources) != set(COMPONENTS)
            or any(not _text(v) for v in sources.values())
        ):
            raise ValueError(
                "explicit source for EACH of the FOUR components including unknowns required"
            )
        payment = _rational(p["declared_closed_total_ticket_payment"])
        if d.payoff.net_ticket_payment != payment:
            raise ValueError(
                "declared NET payment must match closed source total EXACTLY; no prefix/rounding/zero fill"
            )
        infos = tuple(d.decision_information)
        if not infos or any(type(i) is not DeclaredDecisionInformation for i in infos):
            raise ValueError(
                "at least one explicit typed information declaration per person required"
            )
        decision_ids, previous_time, views = set(), None, []
        for info in infos:
            if (
                not _key(info.person_key)
                or info.person_key != key
                or not _text(info.decision_id)
                or info.decision_id in decision_ids
                or not _text(info.definition)
                or not _text(info.source_ref)
                or info.decision_at < utc(ctx["booking_clock"][key[0]])
                or info.decision_at > as_of
                or (previous_time is not None and info.decision_at < previous_time)
            ):
                raise ValueError(
                    "distinct own-person chronological declared decision information required"
                )
            visible = {}
            for label, ids, bank in (
                ("service_events", info.service_event_ids, service_events),
                ("additional_transfers", info.transfer_event_ids, transfer_events),
            ):
                if len(set(ids)) != len(ids) or any(
                    event_id not in bank for event_id in ids
                ):
                    raise ValueError(
                        "distinct existing explicit disclosure IDs required"
                    )
                records = tuple(bank[event_id] for event_id in ids)
                if any(
                    tuple(r["person_key"]) != key
                    or utc(r["recorded_at"]) >= info.decision_at
                    for r in records
                ):
                    raise ValueError(
                        "only SAME person's records STRICTLY recorded before decision; no future/tie/other-person disclosure"
                    )
                visible[label] = _encode(records)
            views.append(
                {
                    "person_key": key,
                    "decision_id": info.decision_id,
                    "decision_at": info.decision_at.isoformat(),
                    "definition": info.definition,
                    "source_ref": info.source_ref,
                    **visible,
                    "complete_information_or_subjective_belief_authenticated": False,
                    "ex_post_payoff_components_or_future_program_included": False,
                    "native_actor_or_passenger_observation_extended": False,
                }
            )
            decision_ids.add(info.decision_id)
            previous_time = info.decision_at
        diagnostic = d.payoff.net_value()
        final = (
            diagnostic
            if d.no_further_material_components
            and p["no_further_ticket_transfers_declared"]
            else None
        )
        closed_payoffs.append(final)
        output.append(
            {
                "person_key": key,
                "original_record": p["original_record"],
                "current_ticket": p["current_ticket"],
                "booking_state": p["booking_state"],
                "four_explicit_components": d.payoff.record(),
                "component_sources": tuple((k, sources[k]) for k in COMPONENTS),
                "components_recorded_at": as_of.isoformat(),
                "declared_component_arithmetic_diagnostic": _number(diagnostic),
                "no_further_material_components_declared": d.no_further_material_components,
                "no_further_ticket_transfers_declared": p[
                    "no_further_ticket_transfers_declared"
                ],
                "followup_source_ref": d.followup_source_ref,
                "declared_closed_material_payoff": _number(final),
                "declared_decision_information": tuple(views),
                "lifecycle_consumer_surplus_or_experienced_utility": None,
            }
        )
    return PersonMaterialPayoffReport(
        {
            "plan_id": "A.10/A.5/B0/V9",
            "definition": definition,
            "payoff_unit": payoff_unit,
            "scope": scope,
            "shock_rule": shock_rule,
            "complete_payment_input": service_ticket_payment_record(payments),
            "all_original_persons": tuple(output),
            "declared_closed_population_material_payoff": _number(_sum(closed_payoffs)),
            "all_support_rows": rows,
            "row_limit": row_limit,
            "disclosure_timing_rule": "recorded_at STRICTLY before declared decision_at; no within-instant inference",
            "payoff_source_tail_closure_or_complete_information_authenticated": False,
            "decision_views_are_analyst_only": True,
            "native_actor_utility_account_service_payment_or_default_changed": False,
            "population_expectation_consumer_surplus_total_surplus_denial_misallocation": None,
            "primary_default_empirical_production_planner_adopted": False,
            "scientific_budget": None,
        }
    )


def person_material_payoff_record(result):
    if type(result) is not PersonMaterialPayoffReport:
        raise TypeError("typed original-person material payoff report required")
    return _encode(result.data)
