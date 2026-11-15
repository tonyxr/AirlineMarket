"""A.10/A.5/A.14/B0/V9: paired material from SAME original native history.

Pure restricted association; declarations do not authenticate native probability,
received knowledge, preferences, service/payment truth or lifecycle CS.
"""

from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import product

from airline_market.sim.booking_loss_law import BookingLossLaw
from airline_market.sim.full_path_transport import (
    FullPathKernel,
    bind_full_path_transport,
    full_path_transport_record,
    transported_joint,
)
from airline_market.sim.native_causal_information import NativeCausalInput
from airline_market.sim.paired_material import (
    SCOPE,
    SHOCK_SCOPE,
    DeclaredMaterialLaw,
    FixedMaterialPolicy,
    MaterialSupport,
    PairedMaterialLimit,
    PairedMaterialModel,
    _sequence,
    evaluate_paired_material,
    paired_material_record,
)
from airline_market.sim.shared_mark_attempt import (
    _array,
    _encoded,
    _names,
    _owned,
    _visible_menu,
)
from airline_market.sim.terminal_material import TerminalMaterialAccount, _accounts
from airline_market.sim.transported_stage_signal import (
    DecisionInput,
    DecisionResult,
    MessageReceived,
    MessageRecorded,
    MessageSource,
    StageSignalLimit,
    StageSignalObservation,
    StageSignalProgramme,
    _check,
    _complete_source,
    _later_source,
    _people,
    _prepare,
    _programme,
    _require,
    _same,
    _source_inputs,
    _text,
    _units,
    causal_decision_view,
    pretype_history_view,
)

RULE = "paired-original-history-material-binding-v1"


@dataclass(frozen=True)
class OriginalHistoryMaterialLaw:
    name: str
    original: BookingLossLaw
    transport: FullPathKernel
    transition_signal: tuple
    terminal_probabilities: tuple
    terminal_accounts: tuple
    definition: str
    source_ref: str

    def __post_init__(self):
        if (
            type(self.original) is not BookingLossLaw
            or type(self.transport) is not FullPathKernel
        ):
            raise TypeError(
                "complete typed original law and full-path transport required"
            )
        for n in (
            "name",
            "transition_signal",
            "terminal_probabilities",
            "terminal_accounts",
            "definition",
            "source_ref",
        ):
            object.__setattr__(
                self,
                n,
                _accounts(getattr(self, n))
                if n == "terminal_accounts"
                else _owned(getattr(self, n)),
            )


@dataclass(frozen=True)
class PairedHistoryProgramme:
    support: MaterialSupport
    policy: FixedMaterialPolicy
    source_programme: StageSignalProgramme
    absent_signal: str
    definition: str
    source_ref: str


@dataclass(frozen=True)
class PairedHistoryBinding:
    analyst: object
    view: object
    row_limit: int
    rule: str = RULE

    def __post_init__(self):
        object.__setattr__(self, "analyst", _owned(self.analyst))
        object.__setattr__(self, "view", _owned(self.view))


def _preflight_templates(model, row_limit):
    if (
        type(model) is not PairedMaterialModel
        or type(model.support) is not MaterialSupport
        or type(model.policy) is not FixedMaterialPolicy
    ):
        raise TypeError(
            "complete typed support, ONE fixed policy and TWO laws required"
        )
    if type(row_limit) is not int or row_limit <= 0:
        raise ValueError("positive integer structural limit required")
    s = model.support
    for name in (
        "original_person_label",
        "market",
        "original_carrier",
        "payoff_unit",
        "conditioning_definition",
        "observation_definition",
        "private_mark_interpretation",
    ):
        _text(getattr(s, name))
    if s.conditioning_scope != SCOPE or s.choice_shock_scope != SHOCK_SCOPE:
        raise ValueError("explicit declared conditioning/material/shock scope required")
    for name in (
        "physical_states",
        "latent_types",
        "private_marks",
        "observations",
        "signals",
        "products",
        "outcomes",
    ):
        _names(getattr(s, name), name)
    (n, k, g, o, h, p, d) = (
        len(getattr(s, name))
        for name in (
            "physical_states",
            "latent_types",
            "private_marks",
            "observations",
            "signals",
            "products",
            "outcomes",
        )
    )
    a = p + 1
    if len(s.type_observation) != k or any(
        x not in s.observations for x in s.type_observation
    ):
        raise ValueError("complete deterministic O=m(K) required")
    if len(s.product_carriers) != p or type(s.interline_enabled) is not bool:
        raise ValueError("complete carriers and explicit interline enable required")
    for carrier in s.product_carriers:
        _text(carrier)

    def boolean(x):
        if type(x) is not bool:
            raise TypeError("explicit boolean fixed menu required")

    _sequence(s.own_menus, (o, p), boolean)
    _sequence(s.interline_menus, (o, h, p), boolean)
    for obs, j in product(range(o), range(p)):
        if s.own_menus[obs][j] and s.product_carriers[j] != s.original_carrier:
            raise ValueError("own menu must use declared original carrier")
        for signal in range(h):
            if s.interline_menus[obs][signal][j] and (
                not s.interline_enabled or s.product_carriers[j] == s.original_carrier
            ):
                raise ValueError(
                    "interline menu must match enabled other-carrier declaration"
                )
    policy = model.policy
    _text(policy.definition)
    _text(policy.source_ref)
    complete = True

    def policy_row(row, mask):
        nonlocal complete
        if type(row) is not tuple or len(row) != a:
            raise ValueError("complete policy row including outside required")
        if all(x is None for x in row):
            complete = False
            return
        if (
            any(type(x) is not Fraction or x < 0 for x in row)
            or sum(row, Fraction()) != 1
        ):
            raise ValueError(
                "unit exact policy OR WHOLE explicitly missing row required"
            )
        if any(row[j] and (not mask[j]) for j in range(p)):
            raise ValueError("positive action on unavailable product refused")

    _sequence(policy.own_program, (o, g, a), lambda x: None)
    _sequence(policy.interline_program, (o, h, g, a), lambda x: None)
    for obs, mark in product(range(o), range(g)):
        policy_row(policy.own_program[obs][mark], s.own_menus[obs])
        for signal in range(h):
            policy_row(
                policy.interline_program[obs][signal][mark],
                s.interline_menus[obs][signal],
            )
    if (
        type(model.forecast) is not DeclaredMaterialLaw
        or type(model.reference) is not DeclaredMaterialLaw
    ):
        raise TypeError("TWO complete typed laws required")
    if model.forecast.name == model.reference.name:
        raise ValueError(
            "distinct law names required; role does not authenticate truth"
        )
    for law in (model.forecast, model.reference):
        for text in (law.name, law.definition, law.source_ref):
            _text(text)
        _array(law.transition_signal, (n, k, g, n, h))
        for y, kind, mark in product(range(n), range(k), range(g)):
            if (
                sum(
                    (
                        law.transition_signal[y][kind][mark][yn][signal]
                        for (yn, signal) in product(range(n), range(h))
                    ),
                    Fraction(),
                )
                != 1
            ):
                raise ValueError(
                    "EVERY joint transition/signal row must sum EXACTLY one"
                )
        _array(law.terminal_probabilities, (2, n, k, g, a, d))

        def account(value):
            if (
                type(value) is not TerminalMaterialAccount
                or value.unit != s.payoff_unit
            ):
                raise ValueError(
                    "EVERY outcome needs explicit FOUR-component account/common unit"
                )

        _sequence(law.terminal_accounts, (2, n, k, g, a, d), account)
        for stage, y, kind, mark, action in product(
            range(2), range(n), range(k), range(g), range(a)
        ):
            if (
                sum(
                    law.terminal_probabilities[stage][y][kind][mark][action], Fraction()
                )
                != 1
            ):
                raise ValueError(
                    "EVERY terminal row including unused support must sum EXACTLY one"
                )
    rows = n * k * g * a * d * (1 + n * h)
    cells = o * (1 + a + h + h * a)
    required = (
        16 * _units(_encoded(model))
        + 2 * (512 * rows + 128 * cells * n * k * g)
        + 20000
    )
    if required > row_limit:
        raise PairedMaterialLimit(
            "ALL input/zero/unknown/proposed output support exceeds structural limit"
        )
    return ((n, k, g, o, h, a, d), required, complete)


def _validate_source_pair(
    law,
    kernel,
    ordinary_input,
    early_periods,
    early_source,
    association,
    baseline,
    own,
    interline,
    later_periods,
    later_source,
    programme,
    observation,
    *,
    row_limit,
):
    """ONE new transport then at most ONE new finite attempt compilation.

    Restricted to a reached enabled interline input with one explicit category
    and one subsequent explicit signal. All original/later context is retained.
    No source is executed here; inputs are supplied complete typed prefixes.
    """
    shape = _programme(programme)
    (n, k, g, o, s, p) = shape
    if type(observation) is not StageSignalObservation:
        raise TypeError("typed complete category/signal declaration required")
    for name in observation.__dataclass_fields__:
        _text(getattr(observation, name))
    if observation.category_delivery_id == observation.signal_delivery_id:
        raise ValueError("distinct category/signal delivery IDs required")
    if type(own) is not NativeCausalInput or type(interline) is not NativeCausalInput:
        raise TypeError("typed reached own and interline inputs required")
    if not programme.interline_enabled:
        raise ValueError(
            "this received-signal scope requires reached enabled interline"
        )
    (early_periods, later_periods) = (tuple(early_periods), tuple(later_periods))
    (early, full) = (_owned(early_source), _owned(later_source))
    inputs = _encoded(
        {
            "law": law,
            "kernel": kernel,
            "ordinary": ordinary_input,
            "early_periods": early_periods,
            "early_source": early,
            "association": association,
            "baseline": baseline,
            "own": own,
            "interline": interline,
            "later_periods": later_periods,
            "later_source": full,
            "programme": programme,
            "observation": observation,
        }
    )
    upstream_bound = _prepare(law, kernel, row_limit, inputs)
    latent = n * k * g
    cells = o * (1 + p + s + s * p)
    required = (
        8 * upstream_bound
        + 16 * _units(inputs)
        + 1024
        * (latent * p + latent * n * s + o * s * latent * p + cells * (1 + latent))
        + 20000
    )
    if required > row_limit:
        raise StageSignalLimit(
            "ALL original/later/template/proposed output support exceeds limit"
        )
    if (
        programme.physical_states != kernel.future_states
        or programme.latent_types != law.latent_types
        or programme.private_marks != law.attempt_marks
    ):
        raise ValueError("same ordered transported Y/K/G axes required")
    if (
        observation.category not in programme.observations
        or observation.signal not in programme.signals
    ):
        raise ValueError(
            "received category and signal must be in complete declared support"
        )
    (closed, early_state) = _later_source(early, early_periods, ordinary_input)
    _check(
        _same(closed, baseline),
        "separately saved exact original closing baseline required",
    )
    context = early["parent_source"]["source_context"]
    originals = _people(context, early_state)
    (key, root) = (own.analyst_original_key, own.analyst_root)
    _check(
        str(root) in originals and tuple(originals[str(root)][0]["key"]) == key,
        "same actual original booked root/key required",
    )
    _check(
        key == ordinary_input.analyst_original_key
        and (key, root) == (interline.analyst_original_key, interline.analyst_root),
        "same selected original person through all inputs",
    )
    (_, final_state) = _later_source(full, later_periods, ordinary_input)
    _people(full["parent_source"]["source_context"], final_state)
    _check(
        _same(early["periods"], full["periods"][: len(early_periods)]),
        "all earlier native periods/global versions must persist",
    )
    _check(
        _same(early["journal"]["analyst_context"], baseline),
        "exact original closing prefix is the separately saved baseline",
    )
    _check(
        _same(early["programme"], full["programme"]),
        "full future declared programme cannot change",
    )
    before_context = early["parent_source"]["source_context"]
    after_context = full["parent_source"]["source_context"]
    static = (
        "native_spec",
        "spec_hash",
        "scenario",
        "dated_rebooking",
        "dated_menu",
        "dated_reserve",
        "cap_policy",
        "trigger_rule",
        "mechanical_benchmark",
    )
    _check(
        all(_same(before_context[n], after_context[n]) for n in static),
        "complete fixed model/scenario/policy context differs",
    )
    _check(
        not full["failed"]
        and full["pending"] is None
        and (full["native_parent_steps_completed"] == len(full["periods"])),
        "complete later source required",
    )
    _complete_source(full, own)
    state = _require(baseline)
    _check(state["pending"] is None, "complete baseline only")
    global_events = full["journal"]["analyst_context"]["events"]
    for row in (own, interline):
        causal_decision_view(row.receipt)
        _check(
            _same(
                row.receipt.source_journal.events,
                global_events[: len(row.receipt.source_journal.events)],
            ),
            "exact input-ending full global prefix only",
        )
    a = own.receipt.source_journal
    later = interline.receipt.source_journal
    _check(
        _same(baseline.events, a.events[: len(baseline.events)]),
        "no omitted original global history",
    )
    _check(
        _same(
            {n: v for (n, v) in _encoded(baseline).items() if n != "events"},
            {n: v for (n, v) in _encoded(a).items() if n != "events"},
        ),
        "same fixed journal contract required",
    )
    extra = a.events[len(baseline.events) :]
    tail = later.events[len(a.events) :]
    _check(
        len(extra) == 4
        and tuple(type(e.payload) for e in extra)
        == (MessageSource, MessageRecorded, MessageReceived, DecisionInput),
        "complete own category SOURCE/RECORD/RECEIPT/INPUT required",
    )
    _check(
        len(tail) == 5
        and tuple(type(e.payload) for e in tail)
        == (
            DecisionResult,
            MessageSource,
            MessageRecorded,
            MessageReceived,
            DecisionInput,
        ),
        "complete own-result/signal SOURCE/RECORD/RECEIPT/interline INPUT required",
    )
    _check(
        _same(a.events, later.events[: len(a.events)]), "same own input prefix required"
    )
    _check(
        all(e.person_key == key for e in (*extra, *tail)),
        "same person at every declared delivery/action/input",
    )
    (category, signal) = (extra[0].payload, tail[1].payload)
    _check(
        (category.delivery_id, category.field, category.value, category.source_ref)
        == (
            observation.category_delivery_id,
            observation.category_field,
            observation.category,
            observation.category_source_ref,
        ),
        "complete category declaration differs",
    )
    _check(
        (signal.delivery_id, signal.field, signal.value, signal.source_ref)
        == (
            observation.signal_delivery_id,
            observation.signal_field,
            observation.signal,
            observation.signal_source_ref,
        ),
        "complete signal declaration differs",
    )
    _check(
        all(e.payload.delivery_id == category.delivery_id for e in extra[1:3])
        and all(e.payload.delivery_id == signal.delivery_id for e in tail[2:4]),
        "same explicit delivery IDs required",
    )
    (first, last) = (a.events[-1], later.events[-1])
    result = tail[0].payload
    _check(
        result.decision_id == first.payload.decision_id
        and result.outside is True
        and (result.product_id is None),
        "only genuine own outside reaches interline",
    )
    _check(
        first.payload.stage == "own"
        and last.payload.stage == "interline"
        and (first.payload.attempt_id == last.payload.attempt_id)
        and first.payload.interline_enabled
        and last.payload.interline_enabled,
        "same reached enabled native attempt",
    )
    _check(
        all(
            e.position.anchor_utc == first.position.anchor_utc for e in (*extra, *tail)
        ),
        "same compiled anchor; no physical retiming",
    )
    period = later.clock.index(last.position.anchor_utc)
    actual = [
        r
        for r in full["periods"][period]["inputs"]
        if r["analyst_root"] == root and r["kind"] != "ordinary"
    ]
    _check(
        _same(actual, (own, interline)),
        "ALL and ONLY reached same-root inputs required",
    )
    for row in (own, interline):
        native = full["parent_source"]["periods"][period]["inputs"][
            row.parent_input_index
        ]
        _check(
            _same(native["view"], row.computation_input),
            "same actual final stage computation menu required",
        )
    before = pretype_history_view(baseline, key)
    _check(
        _same(programme.conditioning_view, before),
        "initial law must be BEFORE category, not category posterior",
    )
    obs = programme.observations.index(observation.category)
    sig = programme.signals.index(observation.signal)
    menus = (programme.own_menus[obs], programme.interline_menus[obs][sig])
    views = _source_inputs(full, own, interline, programme, menus)
    return {
        "before": before,
        "views": views,
        "key": key,
        "root": root,
        "original_product": originals[str(root)][2],
        "structural_bound": required,
    }


def _future(law, joint):
    return DeclaredMaterialLaw(
        law.name,
        joint,
        law.transition_signal,
        law.terminal_probabilities,
        law.terminal_accounts,
        law.definition,
        law.source_ref,
    )


def bind_paired_history(
    programme,
    forecast,
    reference,
    ordinary_input,
    early_periods,
    early_source,
    association,
    baseline,
    own,
    interline,
    later_periods,
    later_source,
    observation,
    *,
    row_limit,
):
    """TWO original-history associations, then ONE pair only if BOTH defined.

    Restricted to a complete fixed policy and actual reached enabled interline,
    one declared category and subsequent signal. Missing policy rows are refused
    before forwards on this bridge; general paired-material behavior stays separate.
    """
    if type(programme) is not PairedHistoryProgramme or any(
        type(law) is not OriginalHistoryMaterialLaw for law in (forecast, reference)
    ):
        raise TypeError(
            "complete typed common programme and TWO original/future laws required"
        )
    s, policy, p = programme.support, programme.policy, programme.source_programme
    if type(s) is not MaterialSupport or type(policy) is not FixedMaterialPolicy:
        raise TypeError("typed complete support and fixed policy required")
    for name in ("absent_signal", "definition", "source_ref"):
        _text(getattr(programme, name))
    if type(p) is not StageSignalProgramme or programme.absent_signal not in s.signals:
        raise TypeError("typed complete source programme and named absence required")
    expected_signals = tuple(
        None if x == programme.absent_signal else x for x in s.signals
    )
    _check(
        (
            p.physical_states,
            p.latent_types,
            p.private_marks,
            p.observations,
            p.type_observation,
            p.signals,
        )
        == (
            s.physical_states,
            s.latent_types,
            s.private_marks,
            s.observations,
            s.type_observation,
            expected_signals,
        ),
        "SAME complete named axes/category map/explicit absence correspondence required",
    )
    _check(
        p.own_program == policy.own_program
        and p.interline_program == policy.interline_program
        and p.interline_enabled == s.interline_enabled,
        "ONE unchanged full policy and interline contract required",
    )
    _check(
        p.next_state_signal == forecast.transition_signal,
        "source template uses actual forecast joint transition, no unused substitute",
    )
    _check(
        tuple(tuple(x.offered for x in row) for row in p.own_menus) == s.own_menus
        and tuple(
            tuple(tuple(x.offered for x in row) for row in obs)
            for obs in p.interline_menus
        )
        == s.interline_menus,
        "ALL fixed boolean source/material menus must agree",
    )
    for row in (*p.own_menus, *(row for obs in p.interline_menus for row in obs)):
        _check(
            tuple(x.product_id for x in row) == s.products,
            "SAME full ordered physical products required",
        )
    for law in (forecast, reference):
        _check(
            law.original.original_person_label == s.original_person_label,
            "SAME declared original person label required",
        )
        _check(
            (
                law.transport.future_states,
                law.original.latent_types,
                law.original.attempt_marks,
            )
            == (s.physical_states, s.latent_types, s.private_marks),
            "SAME ordered transported Y/K/G axes required",
        )
    # None is an intentionally absent, UNEVALUATED preflight field, never a prior.
    templates = PairedMaterialModel(
        s, policy, _future(forecast, None), _future(reference, None)
    )
    _, future_bound, complete = _preflight_templates(templates, row_limit)
    if not complete:
        raise ValueError(
            "this restricted native bridge requires a complete explicit fixed policy"
        )
    early_periods, later_periods = tuple(early_periods), tuple(later_periods)
    early, full = _owned(early_source), _owned(later_source)
    inputs = _encoded(
        {
            "programme": programme,
            "forecast": forecast,
            "reference": reference,
            "ordinary_input": ordinary_input,
            "early_periods": early_periods,
            "early_source": early,
            "association": association,
            "baseline": baseline,
            "own": own,
            "interline": interline,
            "later_periods": later_periods,
            "later_source": full,
            "observation": observation,
        }
    )
    upper = sum(
        _prepare(law.original, law.transport, row_limit, inputs)
        for law in (forecast, reference)
    )
    required = 16 * _units(inputs) + 8 * (upper + future_bound) + 40000
    if required > row_limit:
        raise PairedMaterialLimit(
            "ALL complete source/templates/TWO transports/pair output exceed structural limit"
        )
    checks = tuple(
        _validate_source_pair(
            law.original,
            law.transport,
            ordinary_input,
            early_periods,
            early,
            association,
            baseline,
            own,
            interline,
            later_periods,
            full,
            replace(p, next_state_signal=law.transition_signal),
            observation,
            row_limit=row_limit,
        )
        for law in (forecast, reference)
    )
    _check(
        _same(
            {k: v for k, v in checks[0].items() if k != "structural_bound"},
            {k: v for k, v in checks[1].items() if k != "structural_bound"},
        ),
        "both laws must associate SAME original and later source/history",
    )
    context = full["parent_source"]["source_context"]
    native_products = {row["id"]: row for row in context["native_spec"]["products"]}
    _check(
        s.market == own.computation_input["market_id"]
        and tuple(native_products[j]["carrier"] for j in s.products)
        == s.product_carriers
        and native_products[checks[0]["original_product"]]["carrier"]
        == s.original_carrier,
        "SAME actual native market/product carriers/original carrier required",
    )
    transports = tuple(
        bind_full_path_transport(
            law.original,
            law.transport,
            ordinary_input,
            early_periods,
            early,
            association,
            row_limit=row_limit,
        )
        for law in (forecast, reference)
    )
    for result in transports:
        parent = result.analyst["parent"]["analyst"]
        _check(
            (tuple(parent["actual_original_key"]), parent["actual_root"])
            == (checks[0]["key"], checks[0]["root"]),
            "SAME actual original person through each association",
        )
    joints = tuple(transported_joint(t) for t in transports)
    paired = None
    if all(joint is not None for joint in joints):
        result = evaluate_paired_material(
            PairedMaterialModel(
                s, policy, _future(forecast, joints[0]), _future(reference, joints[1])
            ),
            selected_observation=observation.category,
            selected_signal=observation.signal,
            row_limit=row_limit,
        )
        paired = paired_material_record(result)
    obs = s.observations.index(observation.category)
    sig = s.signals.index(observation.signal)
    view = {
        "scope": "declared-source-association-not-authenticated-knowledge",
        "before_category": checks[0]["before"],
        "own": checks[0]["views"][0],
        "interline": checks[0]["views"][1],
        "own_declared_visible_menu": _visible_menu(p, p.own_menus[obs]),
        "interline_declared_visible_menu": _visible_menu(
            p, p.interline_menus[obs][sig]
        ),
    }
    return PairedHistoryBinding(
        {
            "plan_id": "A.10/A.5/A.14/B0/V9",
            "inputs": inputs,
            "transports": tuple(full_path_transport_record(t) for t in transports),
            "precategory_joints": joints,
            "original_history_evidence": tuple(
                t.analyst["evidence"] for t in transports
            ),
            "paired": paired,
            "original_conditioning_calls": 2,
            "paired_evaluator_called": paired is not None,
            "structural_bound": required,
            "native_probability_knowledge_preferences_service_payment_authenticated": False,
            "lifecycle_utility_CS": None,
            "scientific_budget": None,
        },
        view,
        row_limit,
    )


def paired_history_record(result):
    if type(result) is not PairedHistoryBinding or result.rule != RULE:
        raise TypeError("complete typed paired original-history binding required")
    return _encoded(result)


def paired_history_view(result):
    if type(result) is not PairedHistoryBinding or result.rule != RULE:
        raise TypeError("complete typed paired original-history binding required")
    return _encoded(result.view)
