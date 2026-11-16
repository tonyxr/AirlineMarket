"""A.10/A.5/A.14/B0/V9/B4: owned paired utility registration and source binding.

Local content identity and supplied-source conformance are not external
preregistration, native probability, received knowledge, preferences or CS.
"""

import copy
import hashlib
import math
from dataclasses import dataclass, replace
from fractions import Fraction

from airline_market.core.hashing import canonical_json
from airline_market.policy.regulator import NO_CAP
from airline_market.sim import full_path_utility as utility
from airline_market.sim import paired_history_binding as parent
from airline_market.sim.booking_loss_binding import (
    ChoiceBoundaryAssociation,
    choice_boundary_conditioning_context,
)
from airline_market.sim.full_path_transport import _prepare
from airline_market.sim.paired_material import (
    FixedMaterialPolicy,
    MaterialSupport,
    PairedMaterialModel,
)
from airline_market.sim.pretype_observation import _same, pretype_history_view
from airline_market.sim.received_information import _text, _units
from airline_market.sim.shared_mark_attempt import _encoded, _owned
from airline_market.sim.terminal_material import PROGRAMME_FIELDS, SOURCE_FIELDS
from airline_market.sim.transported_stage_signal import (
    StageSignalObservation,
    StageSignalProgramme,
    _programme,
)

RULE = "registered-paired-full-path-utility-v1"
CONTEXT_PATHS = (
    "forecast.original.conditioning_context",
    "reference.original.conditioning_context",
    "programme.source_programme.conditioning_view",
)


class RegisteredUtilityLimit(RuntimeError):
    pass


def _check(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class UtilityIncidence:
    alpha: Fraction
    material_unit: str
    utility_unit: str
    material_additional_nonoverlap: str
    utility_origin_and_incidence: str
    definition: str
    reference_scope: str


@dataclass(frozen=True)
class FixedPosting:
    period: int
    product_ids: tuple
    prices: tuple

    def __post_init__(self):
        object.__setattr__(self, "product_ids", _owned(self.product_ids))
        object.__setattr__(self, "prices", _owned(self.prices))


@dataclass(frozen=True)
class PairedUtilityRegistration:
    programme: parent.PairedHistoryProgramme
    forecast: parent.OriginalHistoryMaterialLaw
    reference: parent.OriginalHistoryMaterialLaw
    forecast_utility: utility.FullPathUtilityLaw
    reference_utility: utility.FullPathUtilityLaw
    incidence: UtilityIncidence
    source_context: object
    source_programme: object
    fixed_postings: tuple
    association: ChoiceBoundaryAssociation
    observation: StageSignalObservation
    definition: str
    source_ref: str
    row_limit: int
    digest: str
    rule: str = RULE

    def __post_init__(self):
        for name in ("source_context", "source_programme"):
            object.__setattr__(self, name, _owned(_encoded(getattr(self, name))))
        object.__setattr__(self, "fixed_postings", tuple(self.fixed_postings))
        # Typed parent declarations own their containers. Copy the outer graph
        # too, so no caller container or subsequently replaced object is retained.
        for name in (
            "programme",
            "forecast",
            "reference",
            "forecast_utility",
            "reference_utility",
            "incidence",
            "association",
            "observation",
        ):
            object.__setattr__(self, name, copy.deepcopy(getattr(self, name)))


@dataclass(frozen=True)
class RegisteredPathUtility:
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


def _limit(required, limit):
    if required > limit:
        raise RegisteredUtilityLimit(
            "complete structural support exceeds infrastructure limit"
        )


def _validate(registration):
    r = registration
    if type(r) is not PairedUtilityRegistration or r.rule != RULE:
        raise TypeError("typed registration with current rule required")
    if type(r.row_limit) is not int or r.row_limit <= 0:
        raise ValueError("positive structural infrastructure limit required")
    for value in (r.definition, r.source_ref):
        _text(value)
    base = _units(_encoded(_payload(r)))
    _limit(16 * base + 50000, r.row_limit)
    _check(
        set(r.source_context) == set(SOURCE_FIELDS),
        "complete nine-field static source contract required",
    )
    _check(
        set(r.source_programme) == set(PROGRAMME_FIELDS),
        "complete eight-field delivery programme required",
    )
    _check(
        r.source_context["spec_hash"] == r.source_programme["spec_hash"],
        "same native spec hash required",
    )
    _check(
        _same(r.source_context["cap_policy"], NO_CAP),
        "restricted literal NO_CAP source contract required",
    )
    if type(r.incidence) is not UtilityIncidence:
        raise TypeError("explicit typed utility incidence required")
    opts = r.incidence
    if type(opts.alpha) is not Fraction or opts.alpha <= 0:
        raise ValueError("explicit positive rational alpha required")
    for name in opts.__dataclass_fields__:
        if name != "alpha":
            _text(getattr(opts, name))
    _check(
        opts.material_unit != opts.utility_unit
        and opts.reference_scope == utility.REFERENCE_SCOPE,
        "distinct units and same-path reference scope required",
    )
    programme, forecast, reference = r.programme, r.forecast, r.reference
    shape, future_bound = _templates(programme, forecast, reference, r.row_limit)
    _programme(programme.source_programme)
    s = programme.support
    _check(s.payoff_unit == opts.material_unit, "same explicit material unit required")
    if (
        type(r.association) is not ChoiceBoundaryAssociation
        or type(r.observation) is not StageSignalObservation
    ):
        raise TypeError("typed fixed original association and observation required")
    for obj in (r.association, r.observation):
        for name in obj.__dataclass_fields__:
            _text(getattr(obj, name))
    _check(
        r.association.original_person_label == s.original_person_label,
        "same original person label required",
    )
    _check(
        r.observation.category in s.observations and r.observation.signal in s.signals,
        "registered category and signal support required",
    )
    n, k, g, _o, h, a, d = shape
    upstream = sum(
        _prepare(law.original, law.transport, r.row_limit)
        for law in (forecast, reference)
    )
    for material, law in zip(
        (forecast, reference), (r.forecast_utility, r.reference_utility)
    ):
        if type(law) is not utility.FullPathUtilityLaw:
            raise TypeError("TWO complete typed utility laws required")
        for value in (law.name, law.definition, law.source_ref):
            _text(value)
        _check(
            law.name == material.name, "ordered material and utility law names required"
        )
        _check(
            (
                law.physical_states,
                law.latent_types,
                law.private_marks,
                law.action_products,
                law.outcomes,
            )
            == (
                s.physical_states,
                s.latent_types,
                s.private_marks,
                (*s.products, None),
                s.outcomes,
            ),
            "complete same ordered utility axes required",
        )

        def leaf(account):
            _check(
                type(account) is utility.TerminalUtilityAccount
                and account.outside_material.unit == opts.material_unit,
                "all typed outside accounts and units including zero rows required",
            )

        utility._shape(law.accounts, (2, n, k, g, a, d), leaf)
    spec = r.source_context["native_spec"]
    products = tuple(p["id"] for p in spec["products"])
    horizon = spec["horizon"]["n_periods"]
    _check(
        len(r.fixed_postings) == horizon,
        "ALL native periods including empty postings required",
    )
    for t, row in enumerate(r.fixed_postings):
        _check(
            type(row) is FixedPosting and type(row.period) is int and row.period == t,
            "complete consecutive typed posting periods required",
        )
        _check(
            row.product_ids == products and len(row.prices) == len(products),
            "full global product posting order required",
        )
        _check(
            all(type(x) is float and math.isfinite(x) for x in row.prices),
            "explicit finite binary64 fixed prices required",
        )
    rows = n * k * g * a * d * (1 + n * h)
    required = 32 * base + 8 * (upstream + future_bound) + 8192 * rows + 50000
    _limit(required, r.row_limit)
    return required


def register_paired_utility(
    *,
    programme,
    forecast,
    reference,
    forecast_utility,
    reference_utility,
    incidence,
    source_context,
    source_programme,
    fixed_postings,
    association,
    observation,
    definition,
    source_ref,
    row_limit,
):
    """Own and validate complete declarations; no source or probability forward."""
    r = PairedUtilityRegistration(
        programme,
        forecast,
        reference,
        forecast_utility,
        reference_utility,
        incidence,
        source_context,
        source_programme,
        fixed_postings,
        association,
        observation,
        definition,
        source_ref,
        row_limit,
        "",
    )
    _validate(r)
    return replace(r, digest=_digest(r))


def bind_registered_path_utility(
    registration,
    *,
    ordinary_input,
    early_periods,
    early_source,
    baseline,
    own,
    interline,
    later_periods,
    later_source,
    source_trace,
):
    """Check unchanged content and both full source contracts before ONE binding."""
    r = registration
    if type(r) is not PairedUtilityRegistration or r.rule != RULE:
        raise TypeError("typed registration with current rule required")
    _check(r.digest == _digest(r), "unchanged saved registration content required")
    required = _validate(r)  # Never trust externally computed bound metadata.
    early_periods, later_periods = tuple(early_periods), tuple(later_periods)
    inputs = _encoded(
        {
            "ordinary_input": ordinary_input,
            "early_periods": early_periods,
            "early_source": early_source,
            "baseline": baseline,
            "own": own,
            "interline": interline,
            "later_periods": later_periods,
            "later_source": later_source,
            "source_trace": source_trace,
        }
    )
    bound = 8 * required + 128 * _units(inputs) + 50000
    _limit(bound, r.row_limit)
    for source in (early_source, later_source):
        context = source["parent_source"]["source_context"]
        _check(
            _same({key: context[key] for key in SOURCE_FIELDS}, r.source_context),
            "actual supported static source contract differs",
        )
        _check(
            _same(source["programme"], r.source_programme),
            "actual entire delivery programme differs",
        )
    _check(
        early_periods and _same(early_periods[-1].journal, baseline),
        "actual separately saved original GLOBAL closing baseline required",
    )
    products = r.fixed_postings[0].product_ids
    trace = tuple(
        {"period": row["period"], "product_ids": products, "prices": row["prices"]}
        for row in source_trace
    )
    _check(
        _same(trace, r.fixed_postings),
        "ALL actual postings including empty periods must agree",
    )
    context = choice_boundary_conditioning_context(ordinary_input)
    programme = replace(
        r.programme,
        source_programme=replace(
            r.programme.source_programme,
            conditioning_view=pretype_history_view(
                baseline, ordinary_input.analyst_original_key
            ),
        ),
    )
    laws = tuple(
        replace(law, original=replace(law.original, conditioning_context=context))
        for law in (r.forecast, r.reference)
    )
    # Restore exactly these three paths and compare ALL numerical templates.
    restored = (
        replace(
            programme,
            source_programme=replace(
                programme.source_programme,
                conditioning_view=r.programme.source_programme.conditioning_view,
            ),
        ),
        *(
            replace(
                law,
                original=replace(
                    law.original,
                    conditioning_context=original.original.conditioning_context,
                ),
            )
            for law, original in zip(laws, (r.forecast, r.reference))
        ),
    )
    _check(
        _same(restored, (r.programme, r.forecast, r.reference)),
        "only three context paths may attach",
    )
    binding = parent.bind_paired_history(
        programme,
        *laws,
        ordinary_input,
        early_periods,
        early_source,
        r.association,
        baseline,
        own,
        interline,
        later_periods,
        later_source,
        r.observation,
        row_limit=r.row_limit,
    )
    options = {
        name: getattr(r.incidence, name) for name in r.incidence.__dataclass_fields__
    }
    result = utility.evaluate_full_path_utility(
        binding,
        r.forecast_utility,
        r.reference_utility,
        **options,
        row_limit=r.row_limit,
    )
    return RegisteredPathUtility(
        {
            "plan_id": "A.10/A.5/A.14/B0/V9/B4",
            "registration": _encoded(r),
            "source_inputs": inputs,
            "attached_context_paths": CONTEXT_PATHS,
            "paired_binding": parent.paired_history_record(binding),
            "utility_result": utility.full_path_utility_record(result),
            "structural_bound": bound,
            "paired_binding_calls": 1,
            "utility_consumer_calls": 1,
            "native_source_or_probability_authenticated": False,
            "external_preregistration_time_authenticated": False,
            "knowledge_preferences_authenticated": False,
            "lifecycle_CS": None,
            "scientific_budget": None,
        },
        result.view,
        r.row_limit,
    )


def registered_path_utility_record(result):
    if type(result) is not RegisteredPathUtility or result.rule != RULE:
        raise TypeError("typed complete registered utility result required")
    return _encoded(result)


def registered_path_utility_view(result):
    if type(result) is not RegisteredPathUtility or result.rule != RULE:
        raise TypeError("typed complete registered utility result required")
    return _encoded(result.view)


def _templates(programme, forecast, reference, row_limit):
    if type(programme) is not parent.PairedHistoryProgramme or any(
        type(law) is not parent.OriginalHistoryMaterialLaw
        for law in (forecast, reference)
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
        s, policy, parent._future(forecast, None), parent._future(reference, None)
    )
    _, future_bound, complete = parent._preflight_templates(templates, row_limit)
    if not complete:
        raise ValueError(
            "this restricted native bridge requires a complete explicit fixed policy"
        )
    return _, future_bound
