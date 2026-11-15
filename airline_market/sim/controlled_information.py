"""A.10/A.5/B0/V9: exact declared controlled histories on ONE original person.

Forward Bayes recursion for a fixed known type and history-only action program.
No native/physical probability, preference, actual knowledge or utility adoption.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from itertools import product

from airline_market.core.hashing import jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.policy.dated_reserve import utc
from airline_market.sim.passenger_accounts import _encode
from airline_market.sim.received_information import (
    ReceivedDecisionReceipt,
    ReceivedInformationJournal,
    _key,
    _text,
    _units,
    _value,
    received_decision_view,
    received_information_record,
)

RULE = "complete-declared-controlled-history-v1"
ASSUMPTION = "actions-independent-of-hidden-state-given-full-history-and-known-type"


def _names(values):
    values = tuple(values)
    if not values or len(set(values)) != len(values):
        raise ValueError("nonempty distinct full named support required")
    for value in values:
        _text(value)
    return values


def _probabilities(values):
    if any(type(p) is not Fraction or p < 0 for p in values):
        raise TypeError("explicit nonnegative Fraction probabilities required")
    if sum(values, Fraction()) != 1:
        raise ValueError("complete declared probabilities must sum EXACTLY one")


@dataclass(frozen=True)
class ControlledStage:
    actions: tuple
    observations: tuple
    # Canonical rows (state, action, next_state, observation, probability).
    kernel: tuple
    # Canonical rows (full earlier (action, observation) history, probabilities).
    program: tuple

    def __post_init__(self):
        for name in ("actions", "observations", "kernel", "program"):
            object.__setattr__(self, name, deep_freeze(tuple(getattr(self, name))))


@dataclass(frozen=True)
class ControlledInformationModel:
    person_key: tuple
    known_type: str
    type_definition: str
    states: tuple
    initial: tuple
    stages: tuple
    conditioning: str
    law_source: str
    action_assumption: str

    def __post_init__(self):
        object.__setattr__(self, "person_key", _key(self.person_key))
        for name in ("states", "initial", "stages"):
            object.__setattr__(self, name, tuple(getattr(self, name)))


@dataclass(frozen=True)
class ControlledInformationResult:
    model: ControlledInformationModel
    cells: tuple
    transitions: tuple
    support_units: int
    row_limit: int
    rule: str = RULE

    def __post_init__(self):
        object.__setattr__(self, "cells", deep_freeze(self.cells))
        object.__setattr__(self, "transitions", deep_freeze(self.transitions))


class ControlledInformationLimit(RuntimeError):
    """ALL supplied and generated structural support exceeds infrastructure limit."""


def _validate(model, row_limit):
    if type(model) is not ControlledInformationModel:
        raise TypeError("typed complete controlled-information model required")
    if type(row_limit) is not int or row_limit < 1:
        raise ValueError("positive complete-support infrastructure limit required")
    for name in ("known_type", "type_definition", "conditioning", "law_source"):
        _text(getattr(model, name))
    if model.action_assumption != ASSUMPTION:
        raise ValueError(
            "explicit history-only action independence declaration required"
        )
    states = _names(model.states)
    if len(model.initial) != len(states):
        raise ValueError("complete conditional initial law required")
    _probabilities(model.initial)
    if not model.stages:
        raise ValueError("at least one complete controlled stage required")
    histories = ((),)
    # Exact output tree size is counted BEFORE ANY probabilistic forwards.
    units = _units(
        {
            "model": jsonable(model),
            "cells": (),
            "transitions": (),
            "support_units": 0,
            "row_limit": row_limit,
            "rule": RULE,
        }
    )
    n = len(states)

    def cell_units(t, history):
        pairs = tuple((s, Fraction()) for s in states)
        return _units(
            {
                "stage": t,
                "history": history,
                "evidence": Fraction(),
                "state_joint": pairs,
                "posterior": pairs,
            }
        )

    units += cell_units(0, ())
    for stage in model.stages:
        if type(stage) is not ControlledStage:
            raise TypeError("typed controlled stage required")
        actions, observations = _names(stage.actions), _names(stage.observations)
        expected = tuple(product(states, actions, states, observations))
        if tuple(tuple(r[:4]) for r in stage.kernel) != expected or any(
            len(r) != 5 for r in stage.kernel
        ):
            raise ValueError(
                "complete canonical joint transition/observation support required"
            )
        for state in states:
            for action in actions:
                _probabilities(
                    tuple(r[4] for r in stage.kernel if r[:2] == (state, action))
                )
        if tuple(r[0] for r in stage.program) != histories or any(
            len(r) != 2 or len(r[1]) != len(actions) for r in stage.program
        ):
            raise ValueError(
                "complete FULL-history action program required, including off-path"
            )
        for _, probabilities in stage.program:
            _probabilities(probabilities)
        # History coordinates, factors, joint mass and all next-state components.
        for history in histories:
            template = {
                "stage": 0,
                "history": history,
                "state": "s",
                "action": "a",
                "next_state": "s",
                "observation": "o",
                "old_joint": Fraction(),
                "action_probability": Fraction(),
                "kernel_probability": Fraction(),
                "joint_contribution": Fraction(),
            }
            units += n * len(actions) * n * len(observations) * _units(template)
        histories = tuple(
            h + ((a, o),) for h in histories for a in actions for o in observations
        )
        units += sum(cell_units(0, h) for h in histories)
    if units > row_limit:
        raise ControlledInformationLimit(
            "ALL zero/off-path tree support exceeds limit; no forwards"
        )
    return units


def compile_controlled_information(model, *, row_limit):
    """Bayes Eq: mass' = sum_s mass(s,h) pi(a|h,type) K(s',o|s,a).

    K is one COMPLETE joint law, allowing dependence of transition and signal.
    Bayes divides by exactly positive history evidence; zero evidence stays NULL.
    """
    units = _validate(model, row_limit)
    states = model.states
    current = {(): model.initial}
    cells, transitions = [], []
    for t in range(len(model.stages) + 1):
        for history, weights in current.items():
            evidence = sum(weights, Fraction())
            cells.append(
                {
                    "stage": t,
                    "history": history,
                    "evidence": evidence,
                    "state_joint": tuple(zip(states, weights)),
                    "posterior": None
                    if evidence == 0
                    else tuple((s, w / evidence) for s, w in zip(states, weights)),
                }
            )
        if t == len(model.stages):
            break
        stage = model.stages[t]
        kernels = {tuple(r[:4]): r[4] for r in stage.kernel}
        programs = dict(stage.program)
        following = {}
        for history, weights in current.items():
            for ai, action in enumerate(stage.actions):
                action_p = programs[history][ai]
                for observation in stage.observations:
                    new_history = history + ((action, observation),)
                    totals = []
                    for next_state in states:
                        contributions = []
                        for state, old_mass in zip(states, weights):
                            kernel = kernels[state, action, next_state, observation]
                            mass = old_mass * action_p * kernel
                            transitions.append(
                                {
                                    "stage": t,
                                    "history": history,
                                    "state": state,
                                    "action": action,
                                    "next_state": next_state,
                                    "observation": observation,
                                    "old_joint": old_mass,
                                    "action_probability": action_p,
                                    "kernel_probability": kernel,
                                    "joint_contribution": mass,
                                }
                            )
                            contributions.append(mass)
                        totals.append(sum(contributions, Fraction()))
                    following[new_history] = tuple(totals)
        current = following
    return ControlledInformationResult(
        model, tuple(cells), tuple(transitions), units, row_limit
    )


@dataclass(frozen=True)
class DeclaredOwnAction:
    person_key: tuple
    decision_id: str
    action: str
    taken_at: object
    recorded_at: object
    source_ref: str

    def __post_init__(self):
        object.__setattr__(self, "person_key", _key(self.person_key))
        for name in ("decision_id", "action", "source_ref"):
            _text(getattr(self, name))
        for name in ("taken_at", "recorded_at"):
            object.__setattr__(self, name, utc(getattr(self, name)))
        if self.recorded_at < self.taken_at:
            raise ValueError("own action recording cannot precede action")


@dataclass(frozen=True)
class KnownPersonType:
    person_key: tuple
    value: str
    definition: str
    known_at: object
    source_ref: str

    def __post_init__(self):
        object.__setattr__(self, "person_key", _key(self.person_key))
        for name in ("value", "definition", "source_ref"):
            _text(getattr(self, name))
        object.__setattr__(self, "known_at", utc(self.known_at))


@dataclass(frozen=True)
class ControlledReceiptBinding:
    analyst_context: object
    permitted_history: tuple
    evidence: Fraction
    posterior: object
    decision_view: object

    def __post_init__(self):
        for name in (
            "analyst_context",
            "permitted_history",
            "posterior",
            "decision_view",
        ):
            object.__setattr__(self, name, deep_freeze(getattr(self, name)))


def bind_controlled_receipts(
    result, versions, own_actions, known_type, *, signal_field, row_limit
):
    """Bind ALL consecutive NEW decision receipts to explicit own actions/signals.

    Restricted bridge: exactly one text signal per earlier action, no unmodelled
    own disclosures before first decision. Every supplied own record is consumed.
    Known type/actions are declarations, NOT inferred from native class/outcomes.
    """
    if type(result) is not ControlledInformationResult or result.rule != RULE:
        raise TypeError("compiled controlled-information result required")
    _validate(result.model, result.row_limit)
    if type(row_limit) is not int or row_limit < 1:
        raise ValueError("positive complete-support infrastructure limit required")
    model = result.model
    versions, own_actions = tuple(versions), tuple(own_actions)
    if type(known_type) is not KnownPersonType or (
        known_type.person_key != model.person_key
        or known_type.value != model.known_type
        or known_type.definition != model.type_definition
    ):
        raise ValueError(
            "same original person and explicit persistent known type required"
        )
    _text(signal_field)
    if (
        not 1 <= len(versions) <= len(model.stages) + 1
        or len(own_actions) != len(versions) - 1
    ):
        raise ValueError("ALL consecutive decisions and earlier own actions required")
    source, previous_disclosures, previous_decision = None, (), None
    previous_asof, contract = None, None
    source_records = []
    history = ()
    ids = set()
    receipt_views = []
    for i, pair in enumerate(versions):
        if len(pair) != 2:
            raise TypeError("each version requires journal and its decision receipt")
        journal, receipt = pair
        if (
            type(journal) is not ReceivedInformationJournal
            or type(receipt) is not ReceivedDecisionReceipt
        ):
            raise TypeError("typed NEW journal and decision receipt required")
        full = received_information_record(
            journal
        )  # Pure full supplied-source validator/export.
        view = received_decision_view(receipt)
        current_source = full["analyst_context"]
        if source is None:
            source = current_source
        if current_source != source or model.person_key not in journal.person_keys:
            raise ValueError(
                "fixed identical full original-person source prefix required"
            )
        current_contract = (
            full["fields"],
            full["definition"],
            full["unobserved_definition"],
            full["source_closed_at"],
            full["rule"],
            full["original_person_keys"],
        )
        if contract is None:
            contract = current_contract
        if current_contract != contract:
            raise ValueError(
                "identical full disclosure schema/definitions/version contract required"
            )
        if full["disclosures"][: len(previous_disclosures)] != previous_disclosures:
            raise ValueError("immutable global disclosure prefix versions required")
        if previous_asof is not None and (
            journal.as_of < previous_asof
            or any(
                d.received_at <= previous_asof
                for d in journal.disclosures[len(previous_disclosures) :]
            )
        ):
            raise ValueError("no global version backfill or decreasing as-of permitted")
        previous_disclosures = full["disclosures"]
        previous_asof = journal.as_of
        source_records.append(full)
        decision = utc(view["decision_at"])
        if (
            receipt.analyst_person_key != model.person_key
            or receipt.analyst_decision_id in ids
            or receipt.analyst_journal_asof != journal.as_of.isoformat()
            or decision <= journal.as_of
            or (previous_decision is not None and decision <= previous_decision)
        ):
            raise ValueError(
                "same person/distinct ordered decisions/exact earlier version required"
            )
        _text(receipt.analyst_decision_id)
        ids.add(receipt.analyst_decision_id)
        if known_type.known_at >= decision:
            raise ValueError(
                "persistent known type must be known strictly before decision"
            )
        own = tuple(d for d in journal.disclosures if d.person_key == model.person_key)
        expected_view = {
            "scope": "declared-received-history-only",
            "decision_at": decision.isoformat(),
            "definition": journal.definition,
            "unobserved_definition": journal.unobserved_definition,
            "schema": jsonable(journal.fields),
            "received": tuple(
                {
                    "person_message": k,
                    "field": d.field,
                    "value": _value(d.value),
                    "declared_received_at": d.received_at.isoformat(),
                }
                for k, d in enumerate(own)
            ),
        }
        if view != _encode(jsonable(expected_view)) or len(own) != i:
            raise ValueError(
                "ALL supplied own messages and complete exact receipt required"
            )
        if any(d.field != signal_field or type(d.value) is not str for d in own):
            raise ValueError(
                "every own message must be an explicit modelled text signal"
            )
        if i:
            action = own_actions[i - 1]
            earlier_receipt = versions[i - 1][1]
            message = own[-1]
            if type(action) is not DeclaredOwnAction or (
                action.person_key != model.person_key
                or action.decision_id != earlier_receipt.analyst_decision_id
                or action.taken_at != previous_decision
                or action.recorded_at >= decision
                or message.occurred_at < action.taken_at
                or message.received_at <= previous_decision
            ):
                raise ValueError(
                    "complete causal earlier own action and subsequent signal required"
                )
            stage = model.stages[i - 1]
            if (
                action.action not in stage.actions
                or message.value not in stage.observations
            ):
                raise ValueError(
                    "action/signal outside complete declared controlled support"
                )
            history += ((action.action, message.value),)
        previous_decision = decision
        receipt_views.append(view)
    matches = [
        c
        for c in result.cells
        if c["stage"] == len(own_actions) and c["history"] == history
    ]
    if len(matches) != 1:
        raise ValueError("complete compiled history cell required")
    cell = matches[0]
    narrow = {
        "scope": "declared-full-action-signal-history-and-known-type",
        "known_type": model.known_type,
        "type_definition": model.type_definition,
        "history": history,
        "received_decision_view": receipt_views[-1],
    }
    context = {
        "model": jsonable(model),
        "compiled_cells": jsonable(result.cells),
        "compiled_transitions": jsonable(result.transitions),
        "compiled_support_units": result.support_units,
        "compiled_row_limit": result.row_limit,
        "rule": result.rule,
        "sources": tuple(source_records),
        "receipts": jsonable(tuple(r for _, r in versions)),
        "own_actions": _export(own_actions),
        "known_type": _export(known_type),
        "signal_field": signal_field,
        "row_limit": row_limit,
    }
    output = {
        "analyst_context": context,
        "permitted_history": history,
        "evidence": cell["evidence"],
        "posterior": cell["posterior"],
        "decision_view": narrow,
    }
    if _units(output) > row_limit:
        raise ControlledInformationLimit(
            "ALL source/model/history/receipt/output support exceeds limit"
        )
    return ControlledReceiptBinding(
        context, history, cell["evidence"], cell["posterior"], narrow
    )


def _export(value):
    value = jsonable(value)
    if isinstance(value, Mapping):
        return {k: _export(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_export(v) for v in value)
    if type(value) is Fraction:
        return {
            "numerator": str(value.numerator),
            "denominator": str(value.denominator),
        }
    if isinstance(value, datetime):
        return utc(value).isoformat()
    return _encode(value)


def controlled_information_record(value):
    """Owned exact ANALYST export, no native producer/solver or Float64 forwards."""
    if type(value) not in (ControlledInformationResult, ControlledReceiptBinding):
        raise TypeError("controlled result or receipt binding required")
    return {
        "plan_id": "A.10/A.5/B0/V9",
        "rule": RULE,
        "result": _export(value),
        "native_probability_actual_knowledge_or_type_authenticated": False,
        "native_actor_choices_or_default_changed": False,
        "utility_CS_service_or_payoff": None,
        "scientific_budget": None,
    }
