"""A.10/A.5/A.14/B0/V9: declared conditional terminal material accounts.

Immutable numerical registration and ONE fresh source/law association. These
are not authenticated service, payment, knowledge, probability or lifecycle CS.
"""

import hashlib
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import product

from airline_market.core.hashing import canonical_json
from airline_market.sim.booking_loss_binding import choice_boundary_conditioning_context
from airline_market.sim.full_path_transport import _fraction, _prepare
from airline_market.sim.payoff_logit import ContingentPayoff
from airline_market.sim.pretype_observation import _same, pretype_history_view
from airline_market.sim.received_information import _text, _units
from airline_market.sim.shared_mark_attempt import _array, _encoded, _names, _owned
from airline_market.sim.transported_stage_signal import (
    StageSignalProgramme,
    _programme,
    bind_transported_stage_signal,
    transported_stage_signal_record,
)

RULE = "registered-declared-terminal-material-v1"
SCOPE = "declared-conditional-terminal-material"
SHOCK_SCOPE = "private-choice-shocks-excluded"
SOURCE_FIELDS = (
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
PROGRAMME_FIELDS = (
    "rule",
    "spec_hash",
    "fields",
    "known_types",
    "deliveries",
    "definition",
    "unobserved_definition",
    "row_limit",
)


class TerminalMaterialLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class TerminalMaterialAccount:
    payoff: ContingentPayoff
    component_sources: tuple
    unit: str
    financial_tail_closed: bool
    material_tail_closed: bool
    definition: str

    def __post_init__(self):
        if type(self.payoff) is not ContingentPayoff:
            raise TypeError("ALL FOUR explicit typed payoff components required")
        if type(self.component_sources) not in (tuple, list):
            raise TypeError(
                "explicit ordered sequence of FOUR component sources required"
            )
        object.__setattr__(self, "component_sources", _owned(self.component_sources))
        if len(self.component_sources) != 4:
            raise ValueError("source for EACH of the FOUR components required")
        for value in (*self.component_sources, self.unit, self.definition):
            _text(value)
        if (
            not self.financial_tail_closed
            and self.payoff.net_ticket_payment is not None
        ):
            raise ValueError("open financial tail requires unknown NET ticket payment")
        if any(
            type(x) is not bool
            for x in (self.financial_tail_closed, self.material_tail_closed)
        ):
            raise TypeError("explicit financial AND material tail closures required")


def _accounts(value):
    if type(value) is TerminalMaterialAccount:
        return value
    if type(value) not in (tuple, list):
        raise TypeError("complete owned tuple of typed terminal accounts required")
    return tuple(_accounts(x) for x in value)


@dataclass(frozen=True)
class TerminalMaterialLaw:
    name: str
    physical_states: tuple
    latent_types: tuple
    private_marks: tuple
    action_products: tuple
    outcomes: tuple
    probabilities: tuple
    accounts: tuple
    payoff_unit: str
    definition: str
    source_ref: str
    scope: str
    choice_shock_scope: str

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            if name == "accounts":
                value = _accounts(self.accounts)
            else:
                value = _owned(getattr(self, name))
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class MaterialRegistration:
    original_law: object
    transport_kernel: object
    programme: StageSignalProgramme
    terminal_laws: tuple
    source_context: object
    source_programme: object
    definition: str
    source_ref: str
    row_limit: int
    digest: str
    rule: str = RULE

    def __post_init__(self):
        object.__setattr__(self, "terminal_laws", tuple(self.terminal_laws))
        object.__setattr__(
            self, "source_context", _owned(_encoded(self.source_context))
        )
        object.__setattr__(
            self, "source_programme", _owned(_encoded(self.source_programme))
        )


@dataclass(frozen=True)
class TerminalMaterialResult:
    analyst: object
    view: object
    row_limit: int
    rule: str = RULE

    def __post_init__(self):
        object.__setattr__(self, "analyst", _owned(self.analyst))
        object.__setattr__(self, "view", _owned(self.view))


def _payload(registration):
    return {
        name: getattr(registration, name)
        for name in registration.__dataclass_fields__
        if name != "digest"
    }


def _digest(registration):
    return hashlib.sha256(
        canonical_json(_encoded(_payload(registration))).encode()
    ).hexdigest()


def _terminal(law, original, kernel, programme, shape):
    if type(law) is not TerminalMaterialLaw:
        raise TypeError("complete typed terminal law required")
    for name in ("name", "payoff_unit", "definition", "source_ref"):
        _text(getattr(law, name))
    if law.scope != SCOPE or law.choice_shock_scope != SHOCK_SCOPE:
        raise ValueError(
            "explicit declared material/private-shocks-excluded scope required"
        )
    if (
        law.physical_states != kernel.future_states
        or law.latent_types != original.latent_types
        or law.private_marks != original.attempt_marks
    ):
        raise ValueError("SAME complete named Y/K/G axes required; no reset/redraw")
    actions = tuple(x.product_id for x in programme.own_menus[0]) + (None,)
    if law.action_products != actions or actions != kernel.action_products:
        raise ValueError(
            "ALL physical products/outside in SAME ordered catalogue required"
        )
    _names(law.outcomes, "terminal outcomes")
    n, k, g, _, _, p = shape
    d = len(law.outcomes)
    _array(law.probabilities, (n, k, g, p, d))

    def accounts(values, dimensions):
        if not dimensions:
            if (
                type(values) is not TerminalMaterialAccount
                or values.unit != law.payoff_unit
            ):
                raise ValueError(
                    "every outcome needs typed FOUR-component account in common unit"
                )
            return
        if type(values) is not tuple or len(values) != dimensions[0]:
            raise ValueError(
                "ALL outcomes/accounts including unused zero support required"
            )
        for value in values:
            accounts(value, dimensions[1:])

    accounts(law.accounts, (n, k, g, p, d))
    for y, kind, mark, action in product(range(n), range(k), range(g), range(p)):
        if sum(law.probabilities[y][kind][mark][action], Fraction()) != 1:
            raise ValueError("EVERY full terminal kernel row must sum EXACTLY one")
    return d


def _validate_registration(registration):
    if type(registration) is not MaterialRegistration or registration.rule != RULE:
        raise TypeError("complete typed numerical material registration required")
    _text(registration.definition)
    _text(registration.source_ref)
    shape = _programme(registration.programme)
    upstream = _prepare(
        registration.original_law, registration.transport_kernel, registration.row_limit
    )
    if set(registration.source_context) != set(SOURCE_FIELDS) or set(
        registration.source_programme
    ) != set(PROGRAMME_FIELDS):
        raise ValueError(
            "complete expected static native context AND full delivery programme required"
        )
    if (
        registration.source_context["spec_hash"]
        != registration.source_programme["spec_hash"]
    ):
        raise ValueError("same declared source spec hash required")
    if (
        registration.programme.physical_states
        != registration.transport_kernel.future_states
        or registration.programme.latent_types != registration.original_law.latent_types
        or registration.programme.private_marks
        != registration.original_law.attempt_marks
    ):
        raise ValueError("same registered original/transport/later named axes required")
    if any(type(x) is not TerminalMaterialLaw for x in registration.terminal_laws):
        raise TypeError("complete typed terminal laws required")
    if not registration.terminal_laws or len(
        {x.name for x in registration.terminal_laws}
    ) != len(registration.terminal_laws):
        raise ValueError("complete distinct named terminal cases required")
    outcomes = sum(
        _terminal(
            x,
            registration.original_law,
            registration.transport_kernel,
            registration.programme,
            shape,
        )
        for x in registration.terminal_laws
    )
    if len({x.payoff_unit for x in registration.terminal_laws}) != 1:
        raise ValueError("same explicit unit across ALL terminal cases required")
    n, k, g, o, s, p = shape
    future_cells = o * (1 + p + s + s * p)
    required = (
        8 * upstream
        + 16 * _units(_encoded(_payload(registration)))
        + 1024
        * (
            n * k * g * (p + n * s + o * s * p + future_cells)
            + o * s * n * k * g * p * outcomes
        )
        + 20000
    )
    if required > registration.row_limit:
        raise TerminalMaterialLimit(
            "ALL registration/source/template/proposed support exceeds structural limit"
        )
    return required


def register_terminal_material(
    original_law,
    transport_kernel,
    programme,
    terminal_laws,
    source_context,
    source_programme,
    *,
    definition,
    source_ref,
    row_limit,
):
    """Own a complete numerical declaration. Caller saves it before a new source.

    No filesystem/time/native/RNG/probability forward. A digest does not prove
    preregistration, external time or truth; actual source views attach later.
    """
    registration = MaterialRegistration(
        original_law,
        transport_kernel,
        programme,
        tuple(terminal_laws),
        source_context,
        source_programme,
        definition,
        source_ref,
        row_limit,
        "",
    )
    _validate_registration(registration)
    return replace(registration, digest=_digest(registration))


def _registered(registration):
    required = _validate_registration(registration)
    if registration.digest != _digest(registration):
        raise ValueError("registered complete numerical/source-plan bytes differ")
    return required


def _projection(law, binding, observation):
    compiled = binding.analyst["compiled"]
    if compiled is None:
        return {
            "name": law.name,
            "rows": (),
            "raw_all_history_terminal_mass": None,
            "lifecycle_utility_CS": None,
            "scientific_budget": None,
            "selected_evidence": None,
            "known_selected_numerator_diagnostic_only": None,
            "selected_unknown_mass": None,
            "conditional_unknown_probability": None,
            "conditional_material_expectation": None,
            "scope": "upstream joint NULL; full terminal template retained in registration; no unit-prior fill",
        }
    model = compiled["model"]
    obs = tuple(model["observations"]).index(observation.category)
    sig = tuple(model["signals"]).index(observation.signal)
    evidence = binding.analyst["selected_signal_reach_cell"]["evidence"]
    rows = []
    known = Fraction()
    unknown = Fraction()
    selected_total = Fraction()
    raw_total = Fraction()
    for row in compiled["report"]["interline_rows"]:
        o, s, y, k, g, a = (
            row[x]
            for x in ("observation", "signal", "physical", "type", "mark", "action")
        )
        mass = _fraction(row["mass"])
        selected = (o, s) == (obs, sig)
        for d, q in enumerate(law.probabilities[y][k][g][a]):
            account = law.accounts[y][k][g][a][d]
            material = account.payoff.net_value()
            joint = mass * q
            current = joint if selected else Fraction()
            missing = (
                material is None
                or not account.financial_tail_closed
                or not account.material_tail_closed
            )
            contribution = (
                Fraction() if not current else None if missing else current * material
            )
            raw_total += joint
            selected_total += current
            if current and missing:
                unknown += current
            if contribution is not None:
                known += contribution
            rows.append(
                {
                    "interline_coordinate": (o, s, y, k, g, a),
                    "outcome": d,
                    "selected_history": selected,
                    "interline_joint_mass": mass,
                    "terminal_probability": q,
                    "raw_terminal_joint": joint,
                    "selected_terminal_joint": current,
                    "account": _encoded(account),
                    "known_material": material,
                    "selected_contribution": contribution,
                }
            )
    if selected_total != evidence:
        raise ArithmeticError(
            "complete terminal selected mass differs; no normalization repair"
        )
    return {
        "name": law.name,
        "rows": rows,
        "raw_all_history_terminal_mass": raw_total,
        "selected_evidence": evidence,
        "known_selected_numerator_diagnostic_only": known,
        "selected_unknown_mass": unknown,
        "conditional_unknown_probability": unknown / evidence if evidence else None,
        "conditional_material_expectation": None
        if not evidence or unknown
        else known / evidence,
        "scope": SCOPE,
        "lifecycle_utility_CS": None,
        "scientific_budget": None,
    }


def bind_terminal_material(
    registration,
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
):
    """Attach ONLY actual pre-choice/pre-category views, then one fresh binding.

    Every registered numerical/menu/source-plan value stays fixed. All supplied
    earlier/later native context remains analyst-only, with no source execution.
    """
    declared_bound = _registered(registration)
    early_periods, later_periods = tuple(early_periods), tuple(later_periods)
    inputs = _encoded(
        {
            "ordinary": ordinary_input,
            "early_periods": early_periods,
            "early_source": early_source,
            "association": association,
            "baseline": baseline,
            "own": own,
            "interline": interline,
            "later_periods": later_periods,
            "later_source": later_source,
            "observation": observation,
        }
    )
    required = 8 * declared_bound + 32 * _units(inputs) + 20000
    if required > registration.row_limit:
        raise TerminalMaterialLimit(
            "ALL supplied original/later source and proposed exports exceed structural limit"
        )
    for source in (early_source, later_source):
        context = source["parent_source"]["source_context"]
        if not _same(
            {name: context[name] for name in SOURCE_FIELDS}, registration.source_context
        ):
            raise ValueError(
                "actual complete static source plan differs from registration"
            )
        if not _same(source["programme"], registration.source_programme):
            raise ValueError(
                "actual FULL declared delivery programme differs from registration"
            )
    original = replace(
        registration.original_law,
        conditioning_context=choice_boundary_conditioning_context(ordinary_input),
    )
    programme = replace(
        registration.programme,
        conditioning_view=pretype_history_view(baseline, own.analyst_original_key),
    )
    binding = bind_transported_stage_signal(
        original,
        registration.transport_kernel,
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
        row_limit=registration.row_limit,
    )
    projections = tuple(
        _projection(law, binding, observation) for law in registration.terminal_laws
    )
    analyst = {
        "plan_id": "A.10/A.5/A.14/B0/V9",
        "registration": _encoded(registration),
        "stage_binding": transported_stage_signal_record(binding),
        "terminal_projections": projections,
        "structural_bound": required,
        "actual_context_fields_attached_later": (
            "original_law.conditioning_context",
            "programme.conditioning_view",
        ),
        "registration_preceded_source_or_external_time_authenticated": False,
        "native_probability_knowledge_service_payment_truth_or_preferences_authenticated": False,
        "lifecycle_utility_CS_service_truth": None,
        "scientific_budget": None,
    }
    return TerminalMaterialResult(analyst, binding.view, registration.row_limit)


def material_registration_record(registration):
    """Owned lossless declaration export; no producer/forward/time authority."""
    _registered(registration)
    return _encoded(registration)


def terminal_material_record(result):
    if type(result) is not TerminalMaterialResult or result.rule != RULE:
        raise TypeError("complete typed terminal material result required")
    return _encoded(result)


def terminal_material_view(result):
    if type(result) is not TerminalMaterialResult or result.rule != RULE:
        raise TypeError("complete typed terminal material result required")
    return _encoded(result.view)
