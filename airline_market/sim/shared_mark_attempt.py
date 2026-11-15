"""A.10/A.5/A.14/B0/V9: exact finite DECLARED same-mark re-entry attempt.

No native probability, preference, received-knowledge or lifecycle adoption.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

from airline_market.core.hashing import canonical_json, jsonable
from airline_market.core.snapshot import deep_freeze
from airline_market.sim.causal_information import causal_decision_view
from airline_market.sim.native_causal_information import NativeCausalInput
from airline_market.sim.received_information import _text, _units
from airline_market.sim.reentry_stage_information import _stage_encode

RULE = "finite-declared-shared-mark-attempt-v1"


class SharedMarkLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class AttemptOffer:
    product_id: str
    fare: float
    offered: bool
    disclose_fare: bool
    disclose_offer: bool


@dataclass(frozen=True)
class SharedMarkAttempt:
    physical_states: tuple
    private_marks: tuple
    known_type: str
    conditioning_view: object
    conditioning_definition: str
    own_menu: tuple
    interline_menu: tuple
    product_catalogue_disclosed: bool
    interline_enabled: bool
    signals: tuple
    signal_field: str
    initial_joint: tuple
    own_program: tuple
    next_state_signal: tuple
    interline_program: tuple
    definition: str
    source_ref: str

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _owned(getattr(self, name)))


@dataclass(frozen=True)
class SharedMarkResult:
    model: SharedMarkAttempt
    report: object
    row_limit: int
    rule: str = RULE

    def __post_init__(self):
        object.__setattr__(self, "report", deep_freeze(self.report))


def _owned(value):
    if type(value) is AttemptOffer:
        return value
    if isinstance(value, Mapping):
        return deep_freeze({k: _owned(v) for k, v in value.items()})
    if type(value) in (tuple, list):
        return tuple(_owned(v) for v in value)
    if value is None or type(value) in (str, bool, int, float, Fraction):
        return value
    raise TypeError(
        "owned primitive containers required; opaque mutable arrays unsupported"
    )


def _encoded(value):
    return _stage_encode(jsonable(value))


def _names(values, name):
    if type(values) is not tuple or not values or len(set(values)) != len(values):
        raise ValueError("distinct complete " + name + " required")
    for value in values:
        _text(value)


def _array(value, shape):
    if shape:
        if type(value) is not tuple or len(value) != shape[0]:
            raise ValueError("complete exact declared probability array required")
        for row in value:
            _array(row, shape[1:])
    elif type(value) is not Fraction or value < 0:
        raise TypeError("nonnegative exact Fraction required; no float conversion")


def _validate(model):
    if type(model) is not SharedMarkAttempt:
        raise TypeError("complete typed finite attempt required")
    for text in (
        model.known_type,
        model.conditioning_definition,
        model.signal_field,
        model.definition,
        model.source_ref,
    ):
        _text(text)
    _names(model.physical_states, "physical states")
    _names(model.private_marks, "private marks")
    for b in (model.interline_enabled, model.product_catalogue_disclosed):
        if type(b) is not bool:
            raise TypeError("explicit boolean contracts required")
    if (
        type(model.signals) is not tuple
        or not model.signals
        or len(set(model.signals)) != len(model.signals)
    ):
        raise ValueError("complete distinct received signal support required")
    for signal in model.signals:
        if signal is not None:
            _text(signal)
    if None not in model.signals:
        raise ValueError("explicit absence-of-intervening-record signal required")
    if len(model.own_menu) != len(model.interline_menu):
        raise ValueError("same full physical-product catalogue required")
    for menu in (model.own_menu, model.interline_menu):
        names = []
        for offer in menu:
            if type(offer) is not AttemptOffer or type(offer.fare) is not float:
                raise TypeError("typed offer with actual Python binary64 fare required")
            _text(offer.product_id)
            if any(
                type(b) is not bool
                for b in (offer.offered, offer.disclose_fare, offer.disclose_offer)
            ):
                raise TypeError("explicit offer and visibility booleans required")
            if not model.product_catalogue_disclosed and (
                offer.disclose_fare or offer.disclose_offer
            ):
                raise ValueError(
                    "no fare/offer disclosure without declared catalogue visibility"
                )
            names.append(offer.product_id)
        if len(set(names)) != len(names):
            raise ValueError("distinct full product support required")
    if tuple(x.product_id for x in model.own_menu) != tuple(
        x.product_id for x in model.interline_menu
    ):
        raise ValueError("same ordered full products required")
    n, g, o, p = (
        len(model.physical_states),
        len(model.private_marks),
        len(model.signals),
        len(model.own_menu) + 1,
    )
    _array(model.initial_joint, (n, g))
    _array(model.own_program, (g, p))
    _array(model.next_state_signal, (n, g, n, o))
    _array(model.interline_program, (o, g, p))
    if sum(sum(row) for row in model.initial_joint) != 1:
        raise ValueError("full conditional initial joint must sum exactly one")
    for row in model.own_program:
        if sum(row) != 1:
            raise ValueError("complete own action distribution must sum exactly one")
        if any(q and not offer.offered for q, offer in zip(row, model.own_menu)):
            raise ValueError("positive own action on an unoffered product")
    for physical in model.next_state_signal:
        for mark in physical:
            if sum(sum(row) for row in mark) != 1:
                raise ValueError(
                    "full joint next-state/signal kernel must sum exactly one"
                )
    for signal in model.interline_program:
        for row in signal:
            if sum(row) != 1:
                raise ValueError("complete interline program must sum exactly one")
            if any(
                q and not offer.offered for q, offer in zip(row, model.interline_menu)
            ):
                raise ValueError("positive interline action on an unoffered product")
    return n, g, o, p


def _limit(model, shape, row_limit, extra=None):
    if type(row_limit) is not int or row_limit < 1:
        raise ValueError("positive full structural limit required")
    n, g, o, p = shape
    # 128 exceeds each row's encoded keys/scalars/Fraction envelopes. The full
    # model is retained twice and the complete source/output extra separately.
    upper = (
        2 * _units(_encoded(model))
        + 128 * (n * g * p + n * g * n * o + o * n * g + o * n * g * p)
        + 1000
    )
    if upper + 2 * _units(_encoded(extra)) > row_limit:
        raise SharedMarkLimit(
            "ALL model/zero support/conservative output/source exceeds limit; no partial result"
        )


def _compile(model, shape, row_limit):
    n, g, o, p = shape
    outside = p - 1
    own = []
    transitions = []
    second = []
    cells = []
    joint = [[[Fraction() for _ in range(g)] for _ in range(n)] for _ in range(o)]
    for s in range(n):
        for mark in range(g):
            for action in range(p):
                mass = model.initial_joint[s][mark] * model.own_program[mark][action]
                own.append(
                    {
                        "physical": s,
                        "mark": mark,
                        "action": action,
                        "initial": model.initial_joint[s][mark],
                        "action_probability": model.own_program[mark][action],
                        "mass": mass,
                        "interline_reach": action == outside
                        and model.interline_enabled,
                    }
                )
            reach = model.initial_joint[s][mark] * model.own_program[mark][outside]
            for nxt in range(n):
                for signal in range(o):
                    q = model.next_state_signal[s][mark][nxt][signal]
                    mass = reach * q if model.interline_enabled else Fraction()
                    transitions.append(
                        {
                            "physical": s,
                            "mark": mark,
                            "next_physical": nxt,
                            "signal": signal,
                            "own_outside_joint": reach,
                            "joint_kernel": q,
                            "enabled": model.interline_enabled,
                            "mass": mass,
                        }
                    )
                    joint[signal][nxt][mark] += mass
    for signal in range(o):
        evidence = sum(sum(row) for row in joint[signal])
        components = []
        for s in range(n):
            for mark in range(g):
                mass = joint[signal][s][mark]
                components.append(
                    {
                        "physical": s,
                        "mark": mark,
                        "joint_mass": mass,
                        "posterior": None if not evidence else mass / evidence,
                    }
                )
                for action in range(p):
                    q = model.interline_program[signal][mark][action]
                    second.append(
                        {
                            "signal": signal,
                            "physical": s,
                            "mark": mark,
                            "action": action,
                            "reaching_joint_mass": mass,
                            "action_probability": q,
                            "mass": mass * q,
                        }
                    )
        cells.append(
            {
                "signal": model.signals[signal],
                "evidence": evidence,
                "components": components,
            }
        )
    terminal = sum(
        (r["mass"] for r in own if not r["interline_reach"]), Fraction()
    ) + sum((r["mass"] for r in second), Fraction())
    if terminal != 1:
        raise ArithmeticError("complete terminal mass differs; no normalization")
    report = {
        "plan_id": "A.10/A.5/A.14/B0/V9",
        "rule": RULE,
        "input_digest": hashlib.sha256(
            canonical_json(_encoded(model)).encode()
        ).hexdigest(),
        "own_rows": own,
        "transition_signal_rows": transitions,
        "interline_cells": cells,
        "interline_action_rows": second,
        "terminal_mass": terminal,
        "shared_private_mark_carried": True,
        "native_probability_received_knowledge_preferences_or_utility_authenticated": False,
        "lifecycle_utility_CS_service_payoff": None,
        "scientific_budget": None,
    }
    return SharedMarkResult(model, report, row_limit)


def compile_shared_mark_attempt(model, *, row_limit):
    shape = _validate(model)
    _limit(model, shape, row_limit)
    return _compile(model, shape, row_limit)


def shared_mark_record(result):
    if type(result) is not SharedMarkResult or result.rule != RULE:
        raise TypeError("typed finite shared-mark result required")
    return _encoded(result)


def _visible_menu(model, menu):
    if not model.product_catalogue_disclosed:
        return ()
    return tuple(
        {
            "product_id": x.product_id,
            "fare": x.fare if x.disclose_fare else None,
            "offered": x.offered if x.disclose_offer else None,
        }
        for x in menu
    )


def _complete_source(source, own_input):
    """Check ALL supplied period/core/parent support; no producer invocation."""
    periods = source["periods"]
    parent = source["parent_source"]
    context = parent["source_context"]
    if (
        source["rule"] != "prospective-native-causal-input-source-v1"
        or parent["rule"] != "same-call-final-native-reentry-stage-inputs-v1"
    ):
        raise ValueError("new complete native causal/stage source rules required")
    if parent["pending_inputs"] is not None or parent["pending_results"] is not None:
        raise ValueError("incomplete parent stage capture; no positive binding")
    groups = [
        parent["periods"],
        context["ordinary_information"],
        context["own_information"],
        context["proof_information"],
    ]
    if any(len(rows) != len(periods) for rows in groups):
        raise ValueError("ALL complete parent period support required")
    global_journal = source["journal"]["analyst_context"]
    fixed = {k: v for k, v in global_journal.items() if k != "events"}
    original = own_input.receipt.source_journal
    own_fixed = {k: v for k, v in _encoded(original).items() if k != "events"}
    if canonical_json(fixed) != canonical_json(own_fixed):
        raise ValueError(
            "full native model/clock/schema/definitions/source contract differs"
        )
    if context["spec_hash"] != original.analyst_context["spec_hash"] or canonical_json(
        context["native_spec"]
    ) != canonical_json(original.analyst_context["native_spec"]):
        raise ValueError("native model/spec source association differs")
    events = global_journal["events"]
    for i, period in enumerate(periods):
        if (
            period["period"] != i
            or period["native_world_t"] != i + 1
            or any(rows[i]["period"] != i for rows in groups)
        ):
            raise ValueError(
                "full consecutive native source closure boundaries required"
            )
        journal = period["journal"]
        if canonical_json(
            {k: v for k, v in journal.items() if k != "events"}
        ) != canonical_json(fixed) or canonical_json(
            journal["events"]
        ) != canonical_json(events[: len(journal["events"])]):
            raise ValueError(
                "complete immutable global period journal prefixes required"
            )
        stages = parent["periods"][i]["inputs"]
        ordinary = context["ordinary_information"][i]["rows"]
        expected = [
            (
                row["view"]["stage"],
                j,
                tuple(row["analyst_original_key"]),
                row["analyst_root"],
            )
            for j, row in enumerate(stages)
        ]
        expected += [
            ("ordinary", j, tuple(row["analyst_person"]["key"]), None)
            for j, row in enumerate(ordinary)
        ]
        actual = [
            (
                row["kind"],
                row["parent_input_index"],
                tuple(row["analyst_original_key"]),
                row["analyst_root"],
            )
            for row in period["inputs"]
        ]
        if actual != expected:
            raise ValueError(
                "ALL original ordinary/outside and reached-stage source support required"
            )
        anchor = global_journal["clock"][i]
        original_keys = [
            tuple(event["person_key"])
            for event in events
            if event["position"]["anchor_utc"] == anchor
            and event["payload"].get("stage") == "ordinary"
        ]
        if original_keys != [tuple(row["analyst_person"]["key"]) for row in ordinary]:
            raise ValueError(
                "all actual registered ordinary input keys required; no person omission"
            )
    if source["native_decision_input_rows_captured"] != any(
        period["inputs"] for period in periods
    ) or parent["native_reentry_stage_inputs_captured"] != any(
        period["inputs"] for period in parent["periods"]
    ):
        raise ValueError("actual complete input-row flags differ")


def bind_shared_mark_attempt(
    model, own_input, interline_input, source_record, *, row_limit
):
    """NEW supplied source/model only. Compile ONCE after structural checks.

    Fixed menus and one optional text receipt between the two reached stages.
    The initial law is explicitly conditional on the entire own-input view.
    """
    shape = _validate(model)
    if type(own_input) is not NativeCausalInput or own_input.kind != "own":
        raise TypeError("actual supplied own-stage input required")
    if interline_input is not None and (
        type(interline_input) is not NativeCausalInput
        or interline_input.kind != "interline"
    ):
        raise TypeError("actual supplied interline-stage input required")
    source = _owned(source_record)
    _limit(
        model,
        shape,
        row_limit,
        {"source": source, "own": own_input, "interline": interline_input},
    )
    if (
        source["failed"]
        or source["pending"] is not None
        or source["native_parent_steps_completed"] != len(source["periods"])
    ):
        raise ValueError("full complete source periods required; no pending salvage")
    _complete_source(source, own_input)
    own_view = causal_decision_view(own_input.receipt)
    if canonical_json(own_view) != canonical_json(_encoded(model.conditioning_view)):
        raise ValueError(
            "initial law must condition on this complete earlier own-input view"
        )
    if own_view["known_type"]["value"] != model.known_type:
        raise ValueError("same explicit persistent known label required")
    if tuple(own_input.analyst_original_key) != tuple(
        own_input.receipt.source_journal.events[-1].person_key
    ):
        raise ValueError("same original registered source key required")
    period = own_input.receipt.source_journal.clock.index(
        own_input.receipt.source_journal.events[-1].position.anchor_utc
    )
    if period >= len(source["periods"]):
        raise ValueError("supplied source lacks this completed native period")
    selected = [
        r
        for r in source["periods"][period]["inputs"]
        if r["analyst_root"] == own_input.analyst_root and r["kind"] != "ordinary"
    ]
    actual = [own_input] if interline_input is None else [own_input, interline_input]
    if canonical_json(_encoded(actual)) != canonical_json(selected):
        raise ValueError(
            "ALL and ONLY actually reached same-root stage inputs required"
        )
    full_events = source["journal"]["analyst_context"]["events"]
    for row, menu in zip(actual, (model.own_menu, model.interline_menu)):
        event = row.receipt.source_journal.events[-1]
        if (row.analyst_root, row.analyst_original_key) != (
            own_input.analyst_root,
            own_input.analyst_original_key,
        ):
            raise ValueError("same original person/root required")
        encoded_events = _encoded(row.receipt.source_journal.events)
        if canonical_json(encoded_events) != canonical_json(
            full_events[: len(encoded_events)]
        ):
            raise ValueError(
                "exact global source prefix required; no later filtering/backfill"
            )
        if (
            event.payload.stage != row.kind
            or event.payload.interline_enabled != model.interline_enabled
        ):
            raise ValueError("actual native stage/interline contract differs")
        offers = tuple(
            {"product_id": x.product_id, "fare": x.fare, "offered": x.offered}
            for x in menu
        )
        if canonical_json(_encoded(offers)) != canonical_json(
            _encoded(row.computation_input["offers"])
        ):
            raise ValueError(
                "complete native computation menu differs; no feasibility/fare fill"
            )
        parent = source["parent_source"]["periods"][period]["inputs"][
            row.parent_input_index
        ]
        if canonical_json(_encoded(row.computation_input)) != canonical_json(
            parent["view"]
        ):
            raise ValueError("source final-stage input view differs")
        if (
            tuple(parent["analyst_original_key"]) != row.analyst_original_key
            or parent["analyst_root"] != row.analyst_root
        ):
            raise ValueError("native parent person linkage differs")
    signal = None
    second_view = None
    if interline_input is not None:
        second_view = causal_decision_view(interline_input.receipt)
        a, b = own_input.receipt.source_journal, interline_input.receipt.source_journal
        if (
            b.events[-1].payload.attempt_id != a.events[-1].payload.attempt_id
            or b.events[-1].position.anchor_utc != a.events[-1].position.anchor_utc
        ):
            raise ValueError("same attempt/anchor/shared-vector source required")
        before = own_view["earlier_person_history"]
        after = second_view["earlier_person_history"]
        required = (
            {"person_event": len(before), "kind": "earlier_input", "stage": "own"},
            {
                "person_event": len(before) + 1,
                "kind": "earlier_choice",
                "outside": True,
                "product_id": None,
            },
        )
        if canonical_json(after[: len(before) + 2]) != canonical_json(
            [*before, *required]
        ):
            raise ValueError(
                "complete earlier own history and SAME own-outside event required"
            )
        rest = after[len(before) + 2 :]
        if len(rest) > 1:
            raise ValueError(
                "this bridge supports exactly zero or one intervening own received text record; no omission"
            )
        if rest:
            row = rest[0]
            if (
                set(row) != {"person_event", "kind", "field", "value"}
                or row["person_event"] != len(before) + 2
                or row["kind"] != "declared_received"
                or row["field"] != model.signal_field
                or type(row["value"]) is not str
            ):
                raise ValueError("complete declared intervening text signal required")
            signal = row["value"]
        if signal not in model.signals:
            raise ValueError("intervening receipt outside full declared signal support")
    view = {
        "scope": "declared-menu-visibility-not-authenticated-passenger-knowledge",
        "own": own_view,
        "own_declared_visible_menu": _visible_menu(model, model.own_menu),
        "interline": second_view,
        "interline_declared_visible_menu": None
        if second_view is None
        else _visible_menu(model, model.interline_menu),
    }
    _limit(
        model,
        shape,
        row_limit,
        {
            "source": source,
            "own": own_input,
            "interline": interline_input,
            "complete_view": view,
        },
    )
    result = _compile(model, shape, row_limit)
    cell = (
        None
        if interline_input is None
        else result.report["interline_cells"][model.signals.index(signal)]
    )
    output = {
        "plan_id": "A.10/A.5/A.14/B0/V9",
        "rule": RULE,
        "view": view,
        "analyst": {
            "source": source,
            "model_result": result,
            "reached_signal": signal,
            "interline_cell": cell,
        },
        "actual_external_knowledge_type_law_utility_or_primary_adopted": False,
        "scientific_budget": None,
    }
    return deep_freeze(output)
