"""A.10/A.5/A.14/B0/V9/B4: owned role laws and earlier native context.

Finite declared conditioning and fixed-policy utility, not native probability,
private-vector awareness, preference identification or lifecycle CS adoption.
"""

from dataclasses import dataclass, replace
from fractions import Fraction

from airline_market.sim import action_kernel_utility as utility
from airline_market.sim import context_compatibility as context
from airline_market.sim.booking_loss_binding import _same
from airline_market.sim.decision_consistency import DecisionAtom, _text
from airline_market.sim.shared_mark_attempt import _encoded, _owned

RULE = "owned-role-law-earlier-context-v1"
STATIC = (
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
CAUSAL = (
    "rule",
    "spec_hash",
    "fields",
    "known_types",
    "deliveries",
    "definition",
    "unobserved_definition",
    "row_limit",
)
MANIFEST_ROLES = ("framework", "scenario", "controller", "capture")


@dataclass(frozen=True)
class SourceManifest:
    role: str
    definition: str
    sha256: str


@dataclass(frozen=True)
class SourceProgramme:
    context: object
    causal: object
    postings: tuple
    manifests: tuple[SourceManifest, ...]
    definition: str
    source_ref: str

    def __post_init__(self):
        object.__setattr__(self, "context", _owned(self.context))
        object.__setattr__(self, "causal", _owned(self.causal))


@dataclass(frozen=True)
class RegisteredRoleContext:
    raw: utility.RegisteredActionUtility
    selected_history: str
    person_key: tuple[int, int, int]
    actions: tuple[context.ContextAction, ...]
    programme: SourceProgramme
    observation_definition: str
    digest: str
    required_nodes: int
    support_limit: int
    rule: str = RULE

    def export(self):
        return _encoded(self)


@dataclass(frozen=True)
class RoleContextResult:
    registration: RegisteredRoleContext
    context: context.ContextCompatibility
    actual_postings: tuple
    actual_manifests: tuple[SourceManifest, ...]
    subjective_evidence: Fraction
    reference_evidence: Fraction
    subjective_conditional: tuple[DecisionAtom, ...] | None
    reference_conditional: tuple[DecisionAtom, ...] | None
    declared_evaluation: utility.ActionUtilityResult | None
    digest: str
    required_nodes: int
    support_limit: int
    native_G: None = None
    native_law_knowledge_preferences_or_CS_authenticated: bool = False
    rule: str = RULE

    def export(self):
        return _encoded(self)


def _hash(value):
    return utility._digest(_encoded(value))


def _manifests(value):
    rows = tuple(value)
    if len(rows) != 4 or any(type(r) is not SourceManifest for r in rows):
        raise TypeError(
            "complete typed framework/scenario/controller/capture manifests"
        )
    if len({r.role for r in rows}) != 4 or {r.role for r in rows} != set(
        MANIFEST_ROLES
    ):
        raise ValueError("one manifest for EVERY declared source role")
    for row in rows:
        _text(row.definition)
        if (
            type(row.sha256) is not str
            or len(row.sha256) != 64
            or any(c not in "0123456789abcdef" for c in row.sha256)
        ):
            raise ValueError("explicit lowercase SHA256 declaration required")
    by = {r.role: r for r in rows}
    return tuple(by[k] for k in MANIFEST_ROLES)


def _postings(value, programme):
    spec = programme.context["native_spec"]
    products = tuple(p["id"] for p in spec["products"])
    periods = spec["horizon"]["n_periods"]
    if type(periods) is not int or periods < 1:
        raise ValueError("full positive native horizon required")
    rows = utility._sequence(value)
    if len(rows) != periods:
        raise ValueError("ALL postings, including empty native periods, required")
    result = []
    for row in rows:
        pairs = utility._sequence(row)
        if len(pairs) != len(products) or any(
            type(p) not in (tuple, list) or len(p) != 2 for p in pairs
        ):
            raise ValueError("full physical-product posting vector required")
        by = dict(pairs)
        if len(by) != len(products) or set(by) != set(products):
            raise ValueError("every physical product exactly once")
        if any(type(v) is not float for v in by.values()):
            raise TypeError("explicit Python binary64 fare values; no coercion")
        result.append(tuple((k, by[k]) for k in products))
    return tuple(result)


def _programme(value):
    if type(value) is not SourceProgramme:
        raise TypeError("complete SourceProgramme required")
    if set(value.context) != set(STATIC) or set(value.causal) != set(CAUSAL):
        raise ValueError("complete static parent and causal programme schemas required")
    if (
        value.context["spec_hash"] != value.causal["spec_hash"]
        or value.context["spec_hash"]
        != value.context["scenario"]["compiled_spec_sha256"]
    ):
        raise ValueError("same declared full native model/scenario/programme")
    _text(value.definition)
    _text(value.source_ref)
    return replace(
        value,
        postings=_postings(value.postings, value),
        manifests=_manifests(value.manifests),
    )


def _registration_payload(reg):
    return (
        reg.raw,
        reg.selected_history,
        reg.person_key,
        reg.actions,
        reg.programme,
        reg.observation_definition,
        RULE,
    )


def register_role_context(
    problem,
    subjective,
    reference,
    descriptive,
    *,
    selected_history,
    person_key,
    actions,
    programme,
    observation_definition,
    criterion,
    support_limit,
):
    """Own BOTH complete raw laws/accounts/source recipe before any forwards."""
    raw = utility.register_action_utility(
        problem,
        subjective,
        reference,
        descriptive,
        criterion=criterion,
        support_limit=support_limit,
    )
    if selected_history not in raw.problem.histories:
        raise ValueError("explicit declared earlier-event label required")
    person_key = context._key(person_key)
    programme = _programme(programme)
    plan = context._plan(
        context.CompatibilityPlan(
            tuple(
                context.CompatibleLabel(g.name, g.definition, None)
                for g in raw.problem.vectors
            ),
            tuple(actions),
            tuple(
                context.LabelAction(
                    g.name,
                    next(
                        r.action
                        for r in raw.descriptive.actions
                        if (r.history, r.vector) == (selected_history, g.name)
                    ),
                )
                for g in raw.problem.vectors
            ),
            raw.descriptive.definition,
            programme.source_ref,
            _text(observation_definition),
        )
    )
    if (
        tuple(a.name for a in plan.actions) != raw.problem.actions
        or next(a.name for a in plan.actions if a.outside) != raw.problem.outside_action
    ):
        raise ValueError("same complete ordered role-law and physical/outside actions")
    reg = RegisteredRoleContext(
        raw,
        selected_history,
        person_key,
        plan.actions,
        programme,
        observation_definition,
        "digest",
        0,
        support_limit,
    )
    needed = context._count({"inputs": _registration_payload(reg), "output": reg})
    context._limit(support_limit, needed)
    return replace(reg, digest=_hash(_registration_payload(reg)), required_nodes=needed)


def _validated_registration(reg):
    if type(reg) is not RegisteredRoleContext or reg.rule != RULE:
        raise TypeError("current typed complete registration required")
    value = register_role_context(
        reg.raw.problem,
        reg.raw.subjective,
        reg.raw.reference,
        reg.raw.descriptive,
        selected_history=reg.selected_history,
        person_key=reg.person_key,
        actions=reg.actions,
        programme=reg.programme,
        observation_definition=reg.observation_definition,
        criterion=reg.raw.criterion,
        support_limit=reg.support_limit,
    )
    utility._validated(reg.raw)
    if (
        type(reg.required_nodes) is not int
        or reg.required_nodes != value.required_nodes
        or reg.digest != value.digest
    ):
        raise ValueError("recomputed complete registration rule/digest/bound mismatch")
    return value


def _plan(reg):
    return context.CompatibilityPlan(
        tuple(
            context.CompatibleLabel(g.name, g.definition, None)
            for g in reg.raw.problem.vectors
        ),
        reg.actions,
        tuple(
            context.LabelAction(
                g.name,
                next(
                    a.action
                    for a in reg.raw.descriptive.actions
                    if (a.history, a.vector) == (reg.selected_history, g.name)
                ),
            )
            for g in reg.raw.problem.vectors
        ),
        reg.raw.descriptive.definition,
        reg.programme.source_ref,
        reg.observation_definition,
    )


def _evidence(reg, law):
    return sum(
        (r.probability for r in law.prior if r.history == reg.selected_history),
        Fraction(0),
    )


def _condition(reg, law, evidence):
    # Earlier H is applied ONCE here. Backend conditions this joint on G within
    # H; it does not multiply a second history likelihood or reset hidden X.
    return (
        None
        if evidence == 0
        else tuple(
            replace(
                r,
                probability=r.probability / evidence
                if r.history == reg.selected_history
                else Fraction(0),
            )
            for r in law.prior
        )
    )


def _required(reg, source, periods, selected, postings, manifests, plan):
    ctx = context._shape(source, periods, selected, plan)
    shape = RoleContextResult(
        reg,
        ctx,
        postings,
        manifests,
        Fraction(0),
        Fraction(0),
        reg.raw.subjective.prior,
        reg.raw.reference.prior,
        utility._shape(reg.raw),
        "digest",
        0,
        0,
    )
    return context._count(
        {
            "inputs": (reg, source, periods, selected, postings, manifests),
            "output": shape,
        }
    )


def _source_agreement(reg, source, periods, selected, postings, manifests):
    if selected.analyst_original_key != reg.person_key:
        raise ValueError("exact registered original person/input key required")
    programme = reg.programme
    if not _same(source["programme"], programme.causal):
        raise ValueError("full causal programme differs from registration")
    actual = source["parent_source"]["source_context"]
    if not _same({k: actual[k] for k in STATIC}, programme.context):
        raise ValueError("full native model/scenario/policy contracts differ")
    if len(periods) != len(programme.postings):
        raise ValueError("full registered horizon including empty periods required")
    if not _same(postings, programme.postings) or manifests != programme.manifests:
        raise ValueError(
            "ALL supplied actual postings/manifests must match registration"
        )
    for period in periods:
        prices = dict(programme.postings[period.period])
        for row in period.inputs:
            if row.kind == "ordinary":
                for offer in row.computation_input["offers"]:
                    if not _same(offer["fare"], _encoded(prices[offer["product_id"]])):
                        raise ValueError(
                            "actual ordinary fare differs from fixed posting programme"
                        )


def _result_payload(value):
    return (
        value.registration,
        value.context,
        value.actual_postings,
        value.actual_manifests,
        value.subjective_evidence,
        value.reference_evidence,
        value.subjective_conditional,
        value.reference_conditional,
        value.declared_evaluation,
        RULE,
    )


def attach_role_context(
    registration,
    source_record,
    periods,
    selected_input,
    *,
    actual_postings,
    actual_manifests,
    support_limit,
):
    """Attach ONLY complete actual source/context; do not create numerical laws."""
    reg = _validated_registration(registration)
    source, periods, selected = context._inputs(source_record, periods, selected_input)
    postings, manifests = (
        _postings(actual_postings, reg.programme),
        _manifests(actual_manifests),
    )
    plan = _plan(reg)
    needed = _required(reg, source, periods, selected, postings, manifests, plan)
    context._limit(support_limit, needed)
    _source_agreement(reg, source, periods, selected, postings, manifests)
    ctx = context.bind_context_compatibility(
        source, periods, selected, plan, support_limit=support_limit
    )
    q, r = reg.raw.subjective, reg.raw.reference
    qe, re = _evidence(reg, q), _evidence(reg, r)
    qc, rc = _condition(reg, q, qe), _condition(reg, r, re)
    evaluation = None
    if qc is not None and rc is not None:
        conditional = utility.register_action_utility(
            reg.raw.problem,
            replace(q, prior=qc),
            replace(r, prior=rc),
            reg.raw.descriptive,
            criterion=reg.raw.criterion,
            support_limit=support_limit,
        )
        evaluation = utility.execute_action_utility(conditional)
    result = RoleContextResult(
        reg,
        ctx,
        postings,
        manifests,
        qe,
        re,
        qc,
        rc,
        evaluation,
        "digest",
        needed,
        support_limit,
    )
    return replace(result, digest=_hash(_result_payload(result)))


def _validated_result(value):
    if type(value) is not RoleContextResult or value.rule != RULE:
        raise TypeError("current typed attached result required")
    reg = _validated_registration(value.registration)
    context._validated_relation(value.context)
    if not _same(value.context.plan, _plan(reg)):
        raise ValueError("same complete descriptive plan/compatible labels required")
    source, periods, selected = (
        value.context.source_record,
        value.context.periods,
        value.context.selected_input,
    )
    postings, manifests = (
        _postings(value.actual_postings, reg.programme),
        _manifests(value.actual_manifests),
    )
    needed = _required(reg, source, periods, selected, postings, manifests, _plan(reg))
    if type(value.required_nodes) is not int or value.required_nodes != needed:
        raise ValueError("recomputed complete attachment support mismatch")
    context._limit(value.support_limit, needed)
    _source_agreement(reg, source, periods, selected, postings, manifests)
    if (
        value.native_G is not None
        or value.native_law_knowledge_preferences_or_CS_authenticated is not False
    ):
        raise ValueError("native G/law/awareness/preferences/CS stay NULL/false")
    if value.digest != _hash(_result_payload(value)):
        raise ValueError("full stored source/law/result digest mismatch")
    return value


def earlier_role_context_view(value):
    """ONLY detached exact earlier H, excluding laws/G/menu/current result."""
    value = _validated_result(value)
    return context.context_compatibility_view(value.context)
