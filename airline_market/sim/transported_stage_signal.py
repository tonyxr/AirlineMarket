"""A.10/A.5/A.14/B0/V9: conditioned paths to received stage signal.

Pure declared-law association. Source order and typed records authenticate
neither native probability nor passenger knowledge, preferences or welfare.
"""

from dataclasses import dataclass

from airline_market.sim.booking_loss_binding import _people
from airline_market.sim.causal_information import (
    DecisionInput,
    DecisionResult,
    MessageReceived,
    MessageRecorded,
    MessageSource,
    OwnAction,
    _require,
    causal_decision_view,
)
from airline_market.sim.full_path_transport import (
    _prepare,
    bind_full_path_transport,
    full_path_transport_record,
    transported_joint,
)
from airline_market.sim.latent_type_attempt import (
    TypeObservationAttempt,
    compile_type_observation_attempt,
    type_observation_record,
)
from airline_market.sim.native_causal_information import NativeCausalInput
from airline_market.sim.pretype_observation import (
    _same,
    _source_inputs,
    pretype_history_view,
)
from airline_market.sim.received_information import _text, _units
from airline_market.sim.shared_mark_attempt import (
    AttemptOffer,
    _array,
    _complete_source,
    _encoded,
    _names,
    _owned,
    _visible_menu,
)

RULE = "transported-joint-received-stage-signal-binding-v1"


class StageSignalLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class StageSignalObservation:
    category_delivery_id: str
    category_field: str
    category: str
    category_source_ref: str
    signal_delivery_id: str
    signal_field: str
    signal: str
    signal_source_ref: str
    definition: str


@dataclass(frozen=True)
class StageSignalProgramme:
    physical_states: tuple
    latent_types: tuple
    private_marks: tuple
    observations: tuple
    type_observation: tuple
    signals: tuple
    own_menus: tuple
    interline_menus: tuple
    product_catalogue_disclosed: bool
    interline_enabled: bool
    own_program: tuple
    next_state_signal: tuple
    interline_program: tuple
    conditioning_view: object
    conditioning_definition: str
    observation_definition: str
    private_mark_interpretation: str
    definition: str
    source_ref: str

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _owned(getattr(self, name)))


@dataclass(frozen=True)
class TransportedStageSignal:
    analyst: object
    view: object
    row_limit: int
    rule: str = RULE

    def __post_init__(self):
        object.__setattr__(self, "analyst", _owned(self.analyst))
        object.__setattr__(self, "view", _owned(self.view))


def _programme(model):
    if type(model) is not StageSignalProgramme:
        raise TypeError("complete typed latent-type attempt required")
    for name in ("physical_states", "latent_types", "private_marks", "observations"):
        _names(getattr(model, name), name)
    for name in (
        "conditioning_definition",
        "observation_definition",
        "private_mark_interpretation",
        "definition",
        "source_ref",
    ):
        _text(getattr(model, name))
    n, k, g, o = (
        len(getattr(model, name))
        for name in (
            "physical_states",
            "latent_types",
            "private_marks",
            "observations",
        )
    )
    if len(model.type_observation) != k or any(
        x not in model.observations for x in model.type_observation
    ):
        raise ValueError("complete deterministic observation map for ALL latent types")
    if (
        not model.signals
        or len(set(model.signals)) != len(model.signals)
        or None not in model.signals
    ):
        raise ValueError("complete distinct signals including explicit absence None")
    for signal in model.signals:
        if signal is not None:
            _text(signal)
    for b in (model.product_catalogue_disclosed, model.interline_enabled):
        if type(b) is not bool:
            raise TypeError("explicit catalogue/interline booleans required")
    s = len(model.signals)
    if (
        len(model.own_menus) != o
        or len(model.interline_menus) != o
        or any(len(x) != s for x in model.interline_menus)
    ):
        raise ValueError("complete observation and signal menus required")
    menus = list(model.own_menus) + [m for row in model.interline_menus for m in row]
    catalogue = None
    for menu in menus:
        names = []
        for offer in menu:
            if type(offer) is not AttemptOffer or type(offer.fare) is not float:
                raise TypeError(
                    "typed full offer catalogue with binary64 fare metadata required"
                )
            _text(offer.product_id)
            if any(
                type(b) is not bool
                for b in (offer.offered, offer.disclose_fare, offer.disclose_offer)
            ):
                raise TypeError("explicit offered and visibility booleans required")
            if not model.product_catalogue_disclosed and (
                offer.disclose_fare or offer.disclose_offer
            ):
                raise ValueError("visibility requires declared catalogue disclosure")
            names.append(offer.product_id)
        if len(set(names)) != len(names):
            raise ValueError("distinct full products required")
        if catalogue is None:
            catalogue = tuple(names)
        elif tuple(names) != catalogue:
            raise ValueError(
                "same full ordered catalogue for ALL observation/signal menus"
            )
    p = len(catalogue) + 1
    _array(model.own_program, (o, g, p))
    _array(model.next_state_signal, (n, k, g, n, s))
    _array(model.interline_program, (o, s, g, p))
    for physical in model.next_state_signal:
        for kind in physical:
            for mark in kind:
                if sum(q for state in mark for q in state) != 1:
                    raise ValueError(
                        "every complete joint next-state/signal row must sum exactly one"
                    )
    for obs in range(o):
        programs = [(model.own_program[obs], model.own_menus[obs])]
        programs.extend(
            (model.interline_program[obs][sig], model.interline_menus[obs][sig])
            for sig in range(s)
        )
        for program, menu in programs:
            for row in program:
                if sum(row) != 1:
                    raise ValueError(
                        "every action row including unreachable support must sum exactly one"
                    )
                if any(q and not offer.offered for q, offer in zip(row, menu)):
                    raise ValueError(
                        "positive action on unoffered product, including off-path rows"
                    )
    return n, k, g, o, s, p


def _later_source(source, periods, selected):
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
    return closed, state


def _check(value, message):
    if not value:
        raise ValueError(message)


def bind_transported_stage_signal(
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
    n, k, g, o, s, p = shape
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
    early_periods, later_periods = tuple(early_periods), tuple(later_periods)
    early, full = _owned(early_source), _owned(later_source)
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
    closed, early_state = _later_source(early, early_periods, ordinary_input)
    _check(
        _same(closed, baseline),
        "separately saved exact original closing baseline required",
    )
    context = early["parent_source"]["source_context"]
    originals = _people(context, early_state)
    key, root = own.analyst_original_key, own.analyst_root
    _check(
        str(root) in originals and tuple(originals[str(root)][0]["key"]) == key,
        "same actual original booked root/key required",
    )
    _check(
        key == ordinary_input.analyst_original_key
        and (key, root) == (interline.analyst_original_key, interline.analyst_root),
        "same selected original person through all inputs",
    )
    _, final_state = _later_source(full, later_periods, ordinary_input)
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
        and full["native_parent_steps_completed"] == len(full["periods"]),
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
            {n: v for n, v in _encoded(baseline).items() if n != "events"},
            {n: v for n, v in _encoded(a).items() if n != "events"},
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
    category, signal = extra[0].payload, tail[1].payload
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
    first, last = a.events[-1], later.events[-1]
    result = tail[0].payload
    _check(
        result.decision_id == first.payload.decision_id
        and result.outside is True
        and result.product_id is None,
        "only genuine own outside reaches interline",
    )
    _check(
        first.payload.stage == "own"
        and last.payload.stage == "interline"
        and first.payload.attempt_id == last.payload.attempt_id
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
    # All source, timing, programme and support checks precede BOTH forwards.
    transported = bind_full_path_transport(
        law,
        kernel,
        ordinary_input,
        early_periods,
        early,
        association,
        row_limit=row_limit,
    )
    parent = transported.analyst["parent"]["analyst"]
    _check(
        (tuple(parent["actual_original_key"]), parent["actual_root"]) == (key, root),
        "transport association differs from reached original person",
    )
    joint = transported_joint(transported)
    compiled = initial = reached = own_prediction = interline_prediction = None
    if joint is not None:
        values = {
            name: getattr(programme, name) for name in programme.__dataclass_fields__
        }
        model = TypeObservationAttempt(initial_joint=joint, **values)
        result = compile_type_observation_attempt(model, row_limit=row_limit)
        compiled = type_observation_record(result)
        history = {tuple(c["history"]): c for c in result.report["history_cells"]}
        initial = history["initial", obs, None, None]
        reached = history["reach", obs, sig, None]

        def predict(stage, evidence, signal):
            return (
                None
                if not evidence
                else tuple(
                    history[stage, obs, signal, action]["evidence"] / evidence
                    for action in range(p)
                )
            )

        own_prediction = predict("own", initial["evidence"], None)
        interline_prediction = predict("interline", reached["evidence"], sig)
    view = {
        "scope": "declared-visibility-not-authenticated-knowledge",
        "before_category": before,
        "own": views[0],
        "interline": views[1],
        "own_declared_visible_menu": _visible_menu(programme, menus[0]),
        "interline_declared_visible_menu": _visible_menu(programme, menus[1]),
    }
    analyst = {
        "plan_id": "A.10/A.5/A.14/B0/V9",
        "inputs": inputs,
        "transport": full_path_transport_record(transported),
        "compiled": compiled,
        "transported_initial_joint": joint,
        "selected_category_cell": initial,
        "selected_signal_reach_cell": reached,
        "own_action_probabilities": own_prediction,
        "interline_action_probabilities": interline_prediction,
        "upstream_evidence": transported.analyst["evidence"],
        "structural_bound": required,
        "source_binding_executed": True,
        "downstream_compiler_called": joint is not None,
        "native_probability_knowledge_preferences_authenticated": False,
        "lifecycle_utility_CS_service_payoff": None,
        "scientific_budget": None,
    }
    return TransportedStageSignal(analyst, view, row_limit)


def transported_stage_signal_record(result):
    """Owned lossless export, no source or compiler invocation."""
    if type(result) is not TransportedStageSignal or result.rule != RULE:
        raise TypeError("complete typed transported stage-signal binding required")
    return _encoded(result)


def transported_stage_signal_view(result):
    """Narrow permitted-history/declaration projection, never analyst context."""
    if type(result) is not TransportedStageSignal or result.rule != RULE:
        raise TypeError("complete typed transported stage-signal binding required")
    return _encoded(result.view)
