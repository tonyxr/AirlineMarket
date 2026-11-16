"""A.10/A.5/A.14/B0/V9/B4: complete native context/compatible-label relation.

Pure supplied-source validation and descriptive measurability, not native G,
received knowledge, probabilities, preferences or lifecycle CS authentication.
"""

import copy
from dataclasses import dataclass
from fractions import Fraction

from airline_market.core.snapshot import deep_freeze
from airline_market.sim.booking_loss_binding import _people, _same
from airline_market.sim.causal_information import (
    DecisionInput,
    OwnAction,
    _require,
    causal_decision_view,
)
from airline_market.sim.decision_consistency import _digest, _nodes
from airline_market.sim.native_causal_information import (
    NativeCausalInput,
    NativeCausalPeriod,
)
from airline_market.sim.received_information import _key, _text
from airline_market.sim.shared_mark_attempt import _complete_source, _encoded, _owned

RULE = "complete-native-context-compatible-labels-v1"
WEIGHT_SCOPE = "declared-annotations-not-an-authenticated-probability-law"


@dataclass(frozen=True)
class CompatibleLabel:
    name: str
    definition: str
    weight_annotation: Fraction | None


@dataclass(frozen=True)
class ContextAction:
    name: str
    product_id: str | None
    outside: bool


@dataclass(frozen=True)
class LabelAction:
    label: str
    action: str | None


@dataclass(frozen=True)
class CompatibilityPlan:
    labels: tuple[CompatibleLabel, ...]
    actions: tuple[ContextAction, ...]
    policy: tuple[LabelAction, ...]
    definition: str
    source_ref: str
    observation_definition: str
    weight_scope: str = WEIGHT_SCOPE


@dataclass(frozen=True)
class ContextCompatibility:
    source_record: object
    periods: tuple
    selected_input: NativeCausalInput
    plan: CompatibilityPlan
    earlier_view: object
    common_action: str | None
    input_digest: str
    output_digest: str
    required_nodes: int
    support_limit: int
    realized_G: None = None
    probability_law: None = None
    native_G_authenticated: bool = False
    native_knowledge_preferences_CS_adopted: bool = False
    rule: str = RULE

    def __post_init__(self):
        object.__setattr__(self, "source_record", deep_freeze(self.source_record))
        object.__setattr__(self, "earlier_view", deep_freeze(self.earlier_view))
        object.__setattr__(self, "periods", tuple(self.periods))

    def export(self):
        return _encoded(self)


@dataclass(frozen=True)
class ContextActionComparison:
    relation: ContextCompatibility
    observed_action: str
    agrees: bool | None
    agreement_probability: None
    required_nodes: int
    support_limit: int
    rule: str = RULE

    def export(self):
        return _encoded(self)


def _sequence(value):
    if type(value) not in (tuple, list):
        raise TypeError("explicit finite list/tuple required")
    return tuple(value)


def _limit(limit, needed):
    if type(limit) is not int or limit <= 0:
        raise TypeError("positive integer complete structural limit required")
    if needed > limit:
        raise ValueError("ALL source/input/proposed output exceeds structural limit")


def _count(value):
    return _nodes(_encoded(value))


def _plan(value):
    if type(value) is not CompatibilityPlan or value.weight_scope != WEIGHT_SCOPE:
        raise TypeError("explicit current CompatibilityPlan/weight scope required")
    labels, actions, rows = [], [], []
    for label in _sequence(value.labels):
        if type(label) is not CompatibleLabel:
            raise TypeError("CompatibleLabel required")
        weight = label.weight_annotation
        if weight is not None and (type(weight) is not Fraction or weight < 0):
            raise TypeError(
                "explicit nonnegative Fraction-or-unknown weight annotation"
            )
        labels.append(
            CompatibleLabel(_text(label.name), _text(label.definition), weight)
        )
    if not labels or len({l.name for l in labels}) != len(labels):
        raise ValueError(
            "complete distinct compatible labels including zero/unknown required"
        )
    for action in _sequence(value.actions):
        if type(action) is not ContextAction or type(action.outside) is not bool:
            raise TypeError("ContextAction with exact Boolean outside required")
        if action.outside != (action.product_id is None):
            raise ValueError(
                "outside has NULL product; physical action has explicit product"
            )
        actions.append(
            ContextAction(
                _text(action.name),
                None if action.outside else _text(action.product_id),
                action.outside,
            )
        )
    if not actions or len({a.name for a in actions}) != len(actions):
        raise ValueError("complete distinct model actions required")
    if sum(a.outside for a in actions) != 1:
        raise ValueError("ONE explicit outside action required")
    products = [a.product_id for a in actions if not a.outside]
    if len(products) != len(set(products)):
        raise ValueError("one-to-one physical action mapping required")
    for row in _sequence(value.policy):
        if type(row) is not LabelAction:
            raise TypeError("LabelAction required")
        if row.action is not None and _text(row.action) not in {
            a.name for a in actions
        }:
            raise ValueError("policy action outside full map")
        rows.append(LabelAction(_text(row.label), row.action))
    names = [r.label for r in rows]
    if len(names) != len(labels) or set(names) != {l.name for l in labels}:
        raise ValueError("one policy row for EVERY compatible label required")
    by_name = {r.label: r for r in rows}
    return CompatibilityPlan(
        tuple(labels),
        tuple(actions),
        tuple(by_name[l.name] for l in labels),
        _text(value.definition),
        _text(value.source_ref),
        _text(value.observation_definition),
        WEIGHT_SCOPE,
    )


# PURE validator adaptation below: original range and exact change are recorded.
def _source_any(source, periods, selected):
    if (
        source["failed"]
        or source["pending"] is not None
        or source["native_parent_steps_completed"] != len(periods)
        or not periods
    ):
        raise ValueError("complete nonempty successful native prefix required")
    _complete_source(source, selected)
    if not _same(source["periods"], periods):
        raise ValueError("ALL supplied typed native periods and inputs required")
    closed = periods[-1].journal
    if not _same(source["journal"]["analyst_context"], closed):
        raise ValueError(
            "exact native closing journal required; no post-close tail filtering"
        )
    state = _require(closed)
    if (
        state["pending"] is not None
        or source["journal"]["pending_declared_decision"] is not None
    ):
        raise ValueError("complete actual input/result support required")
    decisions = []
    for i, period in enumerate(periods):
        if period.period != i or period.native_world_t != i + 1:
            raise ValueError("consecutive full native closures required")
        if _require(period.journal)["pending"] is not None:
            raise ValueError("no incomplete period journal")
        for event in period.journal.events:
            closing = event.position.anchor_utc == closed.clock[i + 1]
            if event.position.anchor_utc > closed.clock[i + 1] or (
                closing
                and not (
                    type(event.payload) is OwnAction
                    and event.payload.kind == "cancellation"
                )
            ):
                raise ValueError(
                    "earlier native close cannot include future registration/input/result/message actions"
                )
        for row in period.inputs:
            if type(row) is not NativeCausalInput:
                raise TypeError("all typed actual input rows required")
            causal_decision_view(row.receipt)
            parent = source["parent_source"]
            parent_menu = (
                parent["source_context"]["ordinary_information"][i]["rows"][
                    row.parent_input_index
                ]["menu_view"]
                if row.kind == "ordinary"
                else parent["periods"][i]["inputs"][row.parent_input_index]["view"]
            )
            if not _same(row.computation_input, parent_menu):
                raise ValueError("same actual parent computation menu required")
            prefix = row.receipt.source_journal.events
            event = prefix[-1]
            if (
                type(event.payload) is not DecisionInput
                or event.payload.stage != row.kind
                or event.person_key != row.analyst_original_key
                or event.position.anchor_utc != closed.clock[i]
                or not _same(prefix, period.journal.events[: len(prefix)])
            ):
                raise ValueError(
                    "same exact immutable input-ending global prefix required"
                )
            if event.payload.decision_id not in state["results"]:
                raise ValueError(
                    "every actual input requires its complete native result"
                )
            decisions.append(event.payload.decision_id)
    if len(set(decisions)) != len(decisions) or set(decisions) != set(
        state["decisions"]
    ):
        raise ValueError("ALL and ONLY native decision inputs required")
    if selected.analyst_root is not None or not any(
        _same(selected, row) for p in periods for row in p.inputs
    ):
        raise ValueError(
            "original ordinary input from supplied complete prefix required"
        )
    return closed, state


def _inputs(source_record, periods, selected):
    periods = _sequence(periods)
    if not periods or any(type(p) is not NativeCausalPeriod for p in periods):
        raise TypeError("ALL nonempty complete typed native periods required")
    if type(selected) is not NativeCausalInput or selected.kind != "ordinary":
        raise TypeError(
            "typed selected ordinary input required; sequential protocol unsupported"
        )
    for p in periods:
        if type(p.period) is not int or type(p.native_world_t) is not int:
            raise TypeError("exact integer period/closure stamps required")
        for row in _sequence(p.inputs):
            if type(row) is not NativeCausalInput:
                raise TypeError("complete typed input support required")
            _key(row.analyst_original_key)
            if type(row.parent_input_index) is not int or row.parent_input_index < 0:
                raise TypeError("exact nonnegative parent input index required")
            if row.kind == "ordinary":
                if row.analyst_root is not None:
                    raise ValueError(
                        "original ordinary input has NULL root applicability"
                    )
            elif (
                row.kind not in ("own", "interline")
                or type(row.analyst_root) is not int
                or row.analyst_root < 0
            ):
                raise ValueError("explicit reached stage and native root required")
    owned_periods = tuple(
        NativeCausalPeriod(
            p.period,
            p.native_world_t,
            tuple(copy.deepcopy(row) for row in p.inputs),
            copy.deepcopy(p.journal),
        )
        for p in periods
    )
    source = _owned(source_record)
    if (
        type(source["failed"]) is not bool
        or type(source["native_decision_input_rows_captured"]) is not bool
        or type(source["native_parent_steps_completed"]) is not int
    ):
        raise TypeError("exact complete-source Boolean/integer stamps required")
    if (
        type(source["parent_source"]["native_reentry_stage_inputs_captured"])
        is not bool
    ):
        raise TypeError("exact parent input-capture Boolean required")
    return source, owned_periods, copy.deepcopy(selected)


def _validated_source(source, periods, selected):
    closed, state = _source_any(source, periods, selected)
    context = source["parent_source"]["source_context"]
    _people(context, state)
    menu = selected.computation_input
    if (
        type(menu["outside_available"]) is not bool
        or menu["outside_available"] is not True
    ):
        raise ValueError("explicit native outside availability required")
    for offer in menu["offers"]:
        if type(offer["offered"]) is not bool:
            raise TypeError("exact Boolean native offered flags required")
    return closed, state


def _mapping(plan, menu):
    products = tuple(o["product_id"] for o in menu["offers"])
    if {a.product_id for a in plan.actions if not a.outside} != set(products):
        raise ValueError(
            "complete SAME-market physical product map required, no omission"
        )
    if not all(o["offered"] for o in menu["offers"]):
        raise ValueError(
            "restricted ordinary association requires ALL mapped products offered; no drop/fill"
        )


def _classify(plan):
    actions = tuple(row.action for row in plan.policy)
    return actions[0] if None not in actions and len(set(actions)) == 1 else None


def _shape(source, periods, selected, plan):
    # Full typed receipt is larger than the actual earlier-only view.
    return ContextCompatibility(
        source,
        periods,
        selected,
        plan,
        _encoded(selected.receipt),
        None,
        "digest",
        "digest",
        0,
        0,
    )


def _relation_required(source, periods, selected, plan):
    return _count(
        {
            "inputs": (source, periods, selected, plan),
            "output": _shape(source, periods, selected, plan),
        }
    )


def _input_digest(source, periods, selected, plan):
    return _digest(_encoded((source, periods, selected, plan, RULE)))


def bind_context_compatibility(
    source_record, periods, selected_input, plan, *, support_limit
):
    """Own complete source/context/plan; classify labels without reading result."""
    source, periods, selected = _inputs(source_record, periods, selected_input)
    plan = _plan(plan)
    needed = _relation_required(source, periods, selected, plan)
    _limit(support_limit, needed)
    _validated_source(source, periods, selected)
    _mapping(plan, selected.computation_input)
    # This forward gets ONLY the earlier receipt, not completed source/result/menu.
    earlier = causal_decision_view(selected.receipt)
    common = _classify(plan)
    return ContextCompatibility(
        source,
        periods,
        selected,
        plan,
        earlier,
        common,
        _input_digest(source, periods, selected, plan),
        _digest(_encoded((common, earlier, RULE))),
        needed,
        support_limit,
    )


def _validated_relation(value):
    if type(value) is not ContextCompatibility or value.rule != RULE:
        raise TypeError("current owned ContextCompatibility required")
    source, periods, selected = _inputs(
        value.source_record, value.periods, value.selected_input
    )
    plan = _plan(value.plan)
    needed = _relation_required(source, periods, selected, plan)
    if type(value.required_nodes) is not int or value.required_nodes != needed:
        raise ValueError("recomputed full structural support mismatch")
    _limit(value.support_limit, needed)
    _, state = _validated_source(source, periods, selected)
    _mapping(plan, selected.computation_input)
    if not _same(value.earlier_view, causal_decision_view(selected.receipt)):
        raise ValueError(
            "ONLY exact earlier own receipt view allowed; no analyst/result/menu leak"
        )
    if (
        value.realized_G is not None
        or value.probability_law is not None
        or value.native_G_authenticated is not False
        or value.native_knowledge_preferences_CS_adopted is not False
    ):
        raise ValueError("unprovided G/law/knowledge/preferences stay NULL/false")
    actions = tuple(r.action for r in plan.policy)
    if value.common_action is None:
        if None not in actions and len(set(actions)) == 1:
            raise ValueError("stored unresolved relation inconsistent with full policy")
    elif any(a != value.common_action for a in actions):
        raise ValueError("stored common action must match EVERY policy row")
    if value.input_digest != _input_digest(
        source, periods, selected, plan
    ) or value.output_digest != _digest(
        _encoded((value.common_action, value.earlier_view, RULE))
    ):
        raise ValueError("full input/earlier-view/common-action digest mismatch")
    return plan, state


def context_compatibility_view(relation):
    """Detached earlier H ONLY; no analyst keys, G labels, menu or current result."""
    _validated_relation(relation)
    return _encoded(relation.earlier_view)


def compare_context_action(relation, *, support_limit):
    """Separate analyst comparison of fixed descriptive action; no label solve."""
    plan, state = _validated_relation(relation)
    shape = ContextActionComparison(relation, "action", None, None, 0, support_limit)
    needed = _count({"input": relation, "output": shape})
    _limit(support_limit, needed)
    decision = relation.selected_input.receipt.source_journal.events[
        -1
    ].payload.decision_id
    outcome = state["results"][decision].payload
    mapped = tuple(
        a
        for a in plan.actions
        if a.product_id == outcome.product_id and a.outside == outcome.outside
    )
    if len(mapped) != 1:
        raise ValueError("observed native action differs from complete physical map")
    agrees = (
        None
        if relation.common_action is None
        else relation.common_action == mapped[0].name
    )
    return ContextActionComparison(
        relation, mapped[0].name, agrees, None, needed, support_limit
    )
