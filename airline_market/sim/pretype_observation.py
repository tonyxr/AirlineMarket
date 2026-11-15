"""A.10/A.5/A.14/B0/V9: bind a declared law BEFORE a category receipt.

Restricted pure association of supplied complete native inputs, not knowledge,
native probability, preference, utility or lifecycle authentication.
"""

from dataclasses import dataclass

from airline_market.core.hashing import canonical_json
from airline_market.core.snapshot import deep_freeze
from airline_market.sim.causal_information import (
    CausalInformationJournal,
    DecisionInput,
    DecisionResult,
    MessageReceived,
    MessageRecorded,
    MessageSource,
    OwnAction,
    _require,
    causal_decision_view,
)
from airline_market.sim.latent_type_attempt import (
    _validate,
    compile_type_observation_attempt,
)
from airline_market.sim.native_causal_information import NativeCausalInput
from airline_market.sim.received_information import _text, _units
from airline_market.sim.shared_mark_attempt import (
    _complete_source,
    _encoded,
    _owned,
    _visible_menu,
)

RULE = "pre-category-complete-native-source-binding-v1"
SCOPE = "pre-observation-full-joint"


class PretypeObservationLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class PretypeObservation:
    delivery_id: str
    field: str
    category: str
    prior_scope: str
    definition: str
    source_ref: str


@dataclass(frozen=True)
class PretypeBinding:
    analyst: object
    view: object
    row_limit: int
    rule: str = RULE

    def __post_init__(self):
        object.__setattr__(self, "analyst", deep_freeze(self.analyst))
        object.__setattr__(self, "view", deep_freeze(self.view))


def _same(a, b):
    return canonical_json(_encoded(a)) == canonical_json(_encoded(b))


def pretype_history_view(journal, key):
    """ALL earlier own rows in this supplied complete prefix; no slicing."""
    if type(journal) is not CausalInformationJournal:
        raise TypeError("typed saved pre-category journal required")
    state = _require(journal)
    if state["pending"] is not None or key not in state["people"]:
        raise ValueError(
            "complete prefix and already registered original person required"
        )
    registration = state["people"][key].payload
    history = []
    for event in journal.events:
        if event.person_key != key:
            continue
        p, row = event.payload, None
        if type(p) is OwnAction:
            row = {"kind": p.kind, "detail": p.detail}
        elif type(p) is MessageReceived:
            message = state["sources"][p.delivery_id][0].payload
            row = {
                "kind": "declared_received",
                "field": message.field,
                "value": message.value,
            }
        elif type(p) is DecisionInput:
            row = {"kind": "earlier_input", "stage": p.stage}
        elif type(p) is DecisionResult:
            row = {
                "kind": "earlier_choice",
                "outside": p.outside,
                "product_id": p.product_id,
            }
        if row is not None:
            history.append({"person_event": len(history), **row})
    return deep_freeze(
        _encoded(
            {
                "scope": "pre-category-own-history-projection-not-authenticated-knowledge",
                "known_type": {
                    "value": registration.known_type,
                    "definition": registration.definition,
                },
                "schema": journal.fields,
                "definition": journal.definition,
                "unobserved_definition": journal.unobserved_definition,
                "earlier_person_history": history,
            }
        )
    )


def _source_inputs(source, own, interline, model, menus):
    if (
        source["failed"]
        or source["pending"] is not None
        or source["native_parent_steps_completed"] != len(source["periods"])
    ):
        raise ValueError("complete source only; no pending salvage")
    _complete_source(source, own)
    a = own.receipt.source_journal
    event = a.events[-1]
    if type(event.payload) is not DecisionInput:
        raise TypeError("actual decision input prefix required")
    period = a.clock.index(event.position.anchor_utc)
    if period >= len(source["periods"]):
        raise ValueError("source lacks complete selected period")
    selected = [
        r
        for r in source["periods"][period]["inputs"]
        if r["analyst_root"] == own.analyst_root and r["kind"] != "ordinary"
    ]
    actual = [own] if interline is None else [own, interline]
    if not _same(actual, selected):
        raise ValueError("ALL and ONLY reached same-root inputs required")
    full = source["journal"]["analyst_context"]["events"]
    views = []
    for row, menu in zip(actual, menus):
        view = causal_decision_view(row.receipt)
        journal = row.receipt.source_journal
        e = journal.events[-1]
        if (row.analyst_original_key, row.analyst_root) != (
            own.analyst_original_key,
            own.analyst_root,
        ) or e.person_key != own.analyst_original_key:
            raise ValueError("same original source key/root required")
        if not _same(journal.events, full[: len(journal.events)]):
            raise ValueError("exact complete global source prefix required")
        if (
            e.payload.stage != row.kind
            or e.payload.interline_enabled != model.interline_enabled
        ):
            raise ValueError("same stage/interline contract required")
        offers = tuple(
            {"product_id": x.product_id, "fare": x.fare, "offered": x.offered}
            for x in menu
        )
        if not _same(offers, row.computation_input["offers"]):
            raise ValueError("complete actual computation menu differs")
        parent = source["parent_source"]["periods"][period]["inputs"][
            row.parent_input_index
        ]
        if not _same(row.computation_input, parent["view"]) or (
            tuple(parent["analyst_original_key"]),
            parent["analyst_root"],
        ) != (row.analyst_original_key, row.analyst_root):
            raise ValueError("actual parent stage association differs")
        views.append(view)
    return views


def bind_pretype_observation(
    model,
    baseline,
    own_input,
    interline_input,
    source_record,
    observation,
    *,
    row_limit,
):
    """Validate complete pre-category support, then compile this NEW law ONCE.

    A baseline must match a supplied complete native period journal. The own
    input extends it by exactly source/record/receipt/input. Reached interline
    extends own input by exactly own-outside result/input. General intervening
    histories require a separate model; no record selection or backfill.
    """
    shape = _validate(model)
    if type(own_input) is not NativeCausalInput or own_input.kind != "own":
        raise TypeError("typed actual own input required")
    if interline_input is not None and (
        type(interline_input) is not NativeCausalInput
        or interline_input.kind != "interline"
    ):
        raise TypeError("typed actual interline input or None required")
    if type(observation) is not PretypeObservation:
        raise TypeError("explicit typed observation declaration required")
    for name in observation.__dataclass_fields__:
        _text(getattr(observation, name))
    if (
        observation.prior_scope != SCOPE
        or observation.category not in model.observations
    ):
        raise ValueError(
            "explicit pre-observation scope and supported category required"
        )
    if type(row_limit) is not int or row_limit <= 0:
        raise TypeError("positive structural row limit required")
    source = _owned(source_record)
    bundle = {
        "source": source,
        "baseline": baseline,
        "own": own_input,
        "interline": interline_input,
        "model": model,
        "observation": observation,
        "rule": RULE,
        "row_limit": row_limit,
    }
    n, k, g, o, s, p = shape
    latent = n * k * g
    cells = o * (1 + p + s + s * p)
    # Conservative complete compiler output plus owned analyst/view copies.
    required = (
        8 * _units(_encoded(bundle))
        + 512
        * (latent * p + latent * n * s + o * s * latent * p + cells * (1 + latent))
        + 4000
    )
    if required > row_limit:
        raise PretypeObservationLimit(
            "ALL source/input/output support exceeds structural limit; no forward or partial result"
        )
    key = own_input.analyst_original_key
    before = pretype_history_view(baseline, key)
    if not _same(before, model.conditioning_view):
        raise ValueError("law must condition on complete BEFORE-category history")
    b = _encoded(baseline)
    if not any(_same(b, period["journal"]) for period in source["periods"]):
        raise ValueError("separately supplied complete native period baseline required")
    a = own_input.receipt.source_journal
    if not _same(
        {k: v for k, v in b.items() if k != "events"},
        {k: v for k, v in _encoded(a).items() if k != "events"},
    ) or not _same(baseline.events, a.events[: len(baseline.events)]):
        raise ValueError("same fixed contract and complete global baseline required")
    extra = a.events[len(baseline.events) :]
    if (
        len(extra) != 4
        or tuple(type(e.payload) for e in extra)
        != (MessageSource, MessageRecorded, MessageReceived, DecisionInput)
        or any(e.person_key != key for e in extra)
    ):
        raise ValueError(
            "exact same-person SOURCE/RECORD/RECEIPT/INPUT extension required"
        )
    message = extra[0].payload
    if (message.delivery_id, message.field, message.value, message.source_ref) != (
        observation.delivery_id,
        observation.field,
        observation.category,
        observation.source_ref,
    ) or any(e.payload.delivery_id != observation.delivery_id for e in extra[1:3]):
        raise ValueError("same complete declared category message required")
    obs, sig = model.observations.index(observation.category), model.signals.index(None)
    menus = (model.own_menus[obs], model.interline_menus[obs][sig])
    views = _source_inputs(source, own_input, interline_input, model, menus)
    prior_history = before["earlier_person_history"]
    received = {
        "person_event": len(prior_history),
        "kind": "declared_received",
        "field": observation.field,
        "value": observation.category,
    }
    if not _same(views[0]["earlier_person_history"], (*prior_history, received)):
        raise ValueError("complete earlier own history plus category required")
    if interline_input is not None:
        later = interline_input.receipt.source_journal
        tail = later.events[len(a.events) :]
        if (
            not _same(a.events, later.events[: len(a.events)])
            or len(tail) != 2
            or tuple(type(e.payload) for e in tail) != (DecisionResult, DecisionInput)
            or any(e.person_key != key for e in tail)
        ):
            raise ValueError(
                "exact own-outside RESULT/interline INPUT extension required"
            )
        result, current = tail[0].payload, tail[1].payload
        original = a.events[-1]
        if (
            result.decision_id != original.payload.decision_id
            or result.outside is not True
            or result.product_id is not None
            or current.attempt_id != original.payload.attempt_id
            or any(e.position.anchor_utc != original.position.anchor_utc for e in tail)
        ):
            raise ValueError("same reached attempt/anchor/own outside required")
    compiled = compile_type_observation_attempt(model, row_limit=row_limit)
    cells = {tuple(c["history"]): c for c in compiled.report["history_cells"]}
    initial = cells["initial", obs, None, None]
    reached = None if interline_input is None else cells["reach", obs, sig, None]

    def predictions(stage, evidence, signal):
        if not evidence:
            return None
        return tuple(
            cells[stage, obs, signal, action]["evidence"] / evidence
            for action in range(p)
        )

    view = {
        "scope": "declared-visibility-not-authenticated-knowledge",
        "before_category": before,
        "own": views[0],
        "own_declared_visible_menu": _visible_menu(model, menus[0]),
        "interline": None if interline_input is None else views[1],
        "interline_declared_visible_menu": None
        if interline_input is None
        else _visible_menu(model, menus[1]),
    }
    analyst = {
        **bundle,
        "compiled": compiled,
        "selected_category_cell": initial,
        "selected_reach_cell": reached,
        "own_action_probabilities": predictions("own", initial["evidence"], None),
        "interline_action_probabilities": None
        if reached is None
        else predictions("interline", reached["evidence"], sig),
        "structural_bound": required,
        "source_binding_executed": True,
        "native_probability_actual_knowledge_type_preferences_or_CS_authenticated": False,
        "lifecycle_utility_CS_service_payoff": None,
        "scientific_budget": None,
    }
    return PretypeBinding(analyst, view, row_limit)


def pretype_binding_record(result):
    if type(result) is not PretypeBinding or result.rule != RULE:
        raise TypeError("typed pre-category binding required")
    return _encoded(result)


def pretype_binding_view(result):
    if type(result) is not PretypeBinding or result.rule != RULE:
        raise TypeError("typed pre-category binding required")
    return _encoded(result.view)
