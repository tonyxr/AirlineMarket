"""A.10/A.5/A.14/B0/V9: NEW registered terminal-material synthetic references."""

import copy
import pickle
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.dated_demand import compile_dated_demand
from airline_market.policy.dated_rescue import DatedRebookingPolicy
from airline_market.policy.dated_reserve import UtcBookingClock
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.policy.regulator import NO_CAP
from airline_market.sim import terminal_material as subject
from airline_market.sim.booking_loss_binding import ChoiceBoundaryAssociation
from airline_market.sim.booking_loss_law import BookingLossLaw, OriginalOffer
from airline_market.sim.full_path_transport import FullPathKernel
from airline_market.sim.native_causal_information import (
    RULE as NATIVE_RULE,
)
from airline_market.sim.native_causal_information import (
    TIMING,
    DeclaredInputDelivery,
    DeclaredKnownType,
    NativeCausalInformationMarketEngine,
    native_causal_information_record,
)
from airline_market.sim.payoff_logit import ContingentPayoff
from airline_market.sim.received_information import DisclosureField, _units
from airline_market.sim.shared_mark_attempt import AttemptOffer, _encoded
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.sim.transported_stage_signal import (
    StageSignalObservation,
    StageSignalProgramme,
)
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

LIMIT = 500_000_000
UNIT = "declared-fare-unit102974"


def check(value, message):
    assert value, message


def nested(shape, fn, prefix=()):
    return (
        fn(*prefix)
        if not shape
        else tuple(nested(shape[1:], fn, prefix + (i,)) for i in range(shape[0]))
    )


def fresh_source(register):
    spec = screen_spec(
        {
            "T": 3,
            "actions": 2,
            "n_spokes": 1,
            "capacity": 20,
            "capacity_support": [3, 4],
        }
    )
    compiled = compile_market(
        replace(
            spec,
            demand=replace(
                spec.demand,
                mode="poisson",
                arrival_probability=None,
                market_weights=None,
                intensity={
                    r.id: PeriodSchedule.parse({"constant": 2.0}) for r in spec.markets
                },
            ),
        )
    )
    start = datetime(2027, 6, 22, tzinfo=timezone.utc)
    clock = UtcBookingClock(tuple(start + timedelta(hours=12 * i) for i in range(4)))
    departures = tuple(
        (leg.id, start + timedelta(hours=84 if leg.id == "B_OH0" else 120))
        for leg in compiled.spec.legs
    )
    scenario = replace(
        compile_dated_demand(compiled, clock, departures),
        physical_shocks=(
            B1DatedShock(
                "new_signal_exit102974", "rival_exit", "B_OH0", 3, rival_carrier="B"
            ),
        ),
    )
    target = compiled.product_index["B_OH0"]
    market = int(compiled.product_market[target])
    key = (0, 0, market)
    source_kwargs = {
        "dated_rebooking": DatedRebookingPolicy(72, 20, 100, interline=True),
        "causal_fields": (
            DisclosureField(
                "category",
                "text",
                "NEW declared category, not native class/awareness",
                "text",
            ),
            DisclosureField(
                "stage_signal",
                "text",
                "NEW declared conditional signal, not authenticated physical/public report",
                "text",
            ),
        ),
        "causal_known_types": tuple(
            DeclaredKnownType(
                (0, i, market),
                None,
                "no supplied label; not ignorance or sampled class",
                "unit102974",
            )
            for i in range(2)
        ),
        "causal_deliveries": (
            DeclaredInputDelivery(
                key,
                1,
                "own",
                "category102974",
                "category",
                "view-a",
                "declared category102974",
                TIMING,
            ),
            DeclaredInputDelivery(
                key,
                1,
                "interline",
                "signal102974",
                "stage_signal",
                "notice-high",
                "declared signal102974",
                TIMING,
            ),
        ),
        "causal_definition": "NEW transported joint and received stage signal102974",
        "causal_unobserved_definition": "caller schedule not authenticated reception, knowledge, physical law or preference",
        "causal_row_limit": 2000000,
    }
    registered = register(compiled, scenario, clock, departures, source_kwargs)
    tape = ExogenousTape.generate(
        compiled, 31, registry=StreamRegistry(102974, "unit/V9_terminal_material")
    )
    check(tape.gumbel.shape[1] >= 2, "fresh generated support insufficient")
    arrivals = tuple(
        ArrivalMark(
            time=(i + 1) / 3, market=market, segment=0, thinning=0, class_uniform=0.55
        )
        for i in range(2)
    )
    g = tape.gumbel.copy()
    g[:] = -1000000000.0
    g[:, :, -1] = 0
    g[:, :, target] = 1000000000.0
    g[:, 1, -1] = 2000000000.0
    g.setflags(write=False)
    tape = replace(tape, arrival_marks=(arrivals, (), ()), gumbel=g)
    engine = NativeCausalInformationMarketEngine(scenario, **source_kwargs)
    engine.reset(tape=tape)
    actions = {
        c.code: PriceAction(
            {j: 310 if i == 0 else 460 for j in compiled.products_of[i]}
        )
        for (i, c) in enumerate(compiled.spec.carriers)
    }
    engine.step(actions)
    ordinary = engine.causal_periods()[0].inputs[0]
    early_periods = engine.causal_periods()
    early = native_causal_information_record(engine)
    baseline = early_periods[0].journal
    for _ in range(2):
        engine.step(actions)
    rows = engine.causal_periods()[1].inputs
    own = next(r for r in rows if r.kind == "own")
    interline = next(r for r in rows if r.kind == "interline")
    full = native_causal_information_record(engine)
    return (
        engine,
        ordinary,
        early_periods,
        early,
        baseline,
        own,
        interline,
        full,
        engine.causal_periods(),
        registered,
    )


def register(compiled, scenario, clock, departures, kwargs):
    del clock, departures
    market = int(compiled.product_market[compiled.product_index["B_OH0"]])
    products = tuple(
        p
        for j, p in enumerate(compiled.spec.products)
        if int(compiled.product_market[j]) == market
    )
    offers = tuple(
        OriginalOffer(p.id, 310.0 if p.carrier == "A" else 460.0, True)
        for p in products
    )
    a = len(offers) + 1
    ib = next(i for i, p in enumerate(offers) if p.product_id == "B_OH0")
    ia = next(i for i, p in enumerate(offers) if p.product_id.startswith("A_"))
    pk = (F(5, 11), F(6, 11), F())
    pz = (F(1, 3), F(2, 5), F(1, 2))
    ug = (
        ((F(1, 3), F(1, 6)), (F(1, 6), F(1, 3))),
        ((F(1, 5), F(2, 5)), (F(3, 10), F(1, 10))),
        ((F(1, 4), F(1, 4)), (F(1, 4), F(1, 4))),
    )

    def boundary(z, k, u, g, j, x, c):
        q = F(2 + z + u + g, 9) if j < a - 1 else F()
        x1 = F(2 + k + c, 6)
        return (q if c else 1 - q) * (x1 if x else 1 - x1)

    law = BookingLossLaw(
        "material-person102974",
        ("z0", "z1"),
        ("x0", "x1"),
        ("k0", "k1", "zero-k"),
        ("u0", "u1"),
        ("g0", "g1"),
        offers,
        nested(
            (2, 3, 2, 2),
            lambda z, k, u, g: pk[k] * (pz[k] if z else 1 - pz[k]) * ug[k][u][g],
        ),
        nested(
            (3, 2, a),
            lambda k, u, j: (
                F(3 + k + u, 10)
                if j == ib
                else F(1, 9)
                if j == ia
                else 1 - F(3 + k + u, 10) - F(1, 9)
            ),
        ),
        nested((2, 3, 2, 2, a, 2, 2), boundary),
        {"source_context_pending": "native pre-choice view not yet produced"},
        "before-original-choice-and-boundary",
        "NEW numeric declaration before source; actual history later",
        "distinct U",
        "SAME G",
        "joint X/cancellation",
        "NEW material law102974",
        "declared102974",
    )
    kernel = FullPathKernel(
        law.initial_states,
        law.boundary_states,
        law.latent_types,
        law.ordinary_marks,
        law.attempt_marks,
        tuple(p.product_id for p in offers) + (None,),
        ("y0", "y1"),
        nested(
            (2, 3, 2, 2, a, 2, 2, 2),
            lambda z, k, u, g, j, x, c, y: (
                F(2 + z + u + g + x, 8) if y else 1 - F(2 + z + u + g + x, 8)
            ),
        ),
        "after-original-choice-boundary-before-later-observation",
        "full path102974",
        "declared102974",
    )
    own = tuple(AttemptOffer(p.product_id, p.fare, False, False, False) for p in offers)
    other = tuple(
        AttemptOffer(p.product_id, p.fare, p.product_id.startswith("A_"), False, False)
        for p in offers
    )

    def signal(y, k, g, yn, s):
        next1 = F(2 + y + k, 6)
        high = F(2 + y + g + yn, 7)
        return (next1 if yn else 1 - next1) * (
            F() if s == 0 else high if s == 1 else 1 - high
        )

    def action(o, s, g, j):
        q = ((F(1, 2), F(1, 2)), (F(3, 7), F(4, 5)), (F(2, 3), F(1, 4)))[s][g]
        return q if j == ia else 1 - q if j == a - 1 else F()

    programme = StageSignalProgramme(
        kernel.future_states,
        law.latent_types,
        law.attempt_marks,
        ("view-a", "view-b", "unused"),
        ("view-a", "view-b", "view-a"),
        (None, "notice-high", "notice-low"),
        (own,) * 3,
        ((other,) * 3,) * 3,
        False,
        True,
        nested((3, 2, a), lambda o, g, j: F(j == a - 1)),
        nested((2, 3, 2, 2, 3), signal),
        nested((3, 3, 2, a), action),
        {"source_context_pending": "native closing view not yet produced"},
        "joint BEFORE category",
        "declared category not native class",
        "SAME G",
        "NEW material programme102974",
        "declared102974",
    )

    def account(y, k, g, j, d):
        if d == 3:
            values = (None,) * 4
        elif j == a - 1:
            values = (F(12), F(), F(2), F(-1))
        else:
            values = (
                F(600 + 45 * k) if d == 0 else F(),
                F(offers[j].fare) * (F(1, 2) if d == 1 else F(1)),
                F(9, 4) + y + F(g, 2),
                F(18 + k) if d == 1 else F(),
            )
        return subject.TerminalMaterialAccount(
            ContingentPayoff(*values),
            tuple(
                "explicit " + n + " source102974"
                for n in ("benefit", "NET payment", "waiting", "cancellation")
            ),
            UNIT,
            d != 3,
            d != 3,
            "NEW declared material only; no actual settlement/service truth",
        )

    laws = []
    for name, rho in (("closed", F()), ("unknown", F(1, 9)), ("tiny", F(1, 10**950))):

        def probability(y, k, g, j, d, rho=rho):
            if d == 3:
                return rho
            if j == a - 1:
                return 1 - rho if d == 2 else F()
            served = F(3 + y + k + g, 8)
            return (1 - rho) * (served if d == 0 else 1 - served if d == 1 else F())

        laws.append(
            subject.TerminalMaterialLaw(
                name,
                kernel.future_states,
                law.latent_types,
                law.attempt_marks,
                kernel.action_products,
                (
                    "declared-served",
                    "declared-not-served",
                    "outside-followup",
                    "unreported",
                ),
                nested((2, 3, 2, a, 4), probability),
                nested((2, 3, 2, a, 4), account),
                UNIT,
                "NEW complete terminal law102974",
                "declared102974",
                subject.SCOPE,
                subject.SHOCK_SCOPE,
            )
        )
    context = {
        "native_spec": compiled.spec,
        "spec_hash": compiled.spec_hash,
        "scenario": scenario.record(),
        "dated_rebooking": kwargs["dated_rebooking"],
        "dated_menu": None,
        "dated_reserve": None,
        "cap_policy": NO_CAP,
        "trigger_rule": None,
        "mechanical_benchmark": None,
    }
    source_programme = {
        "rule": NATIVE_RULE,
        "spec_hash": compiled.spec_hash,
        "fields": kwargs["causal_fields"],
        "known_types": kwargs["causal_known_types"],
        "deliveries": kwargs["causal_deliveries"],
        "definition": kwargs["causal_definition"],
        "unobserved_definition": kwargs["causal_unobserved_definition"],
        "row_limit": kwargs["causal_row_limit"],
    }
    return subject.register_terminal_material(
        law,
        kernel,
        programme,
        tuple(laws),
        context,
        source_programme,
        definition="NEW full numerical/material declaration before source creation; no external-time truth",
        source_ref="unit102974",
        row_limit=LIMIT,
    )


@pytest.fixture(scope="module")
def bundle():
    return fresh_source(register)


def args(bundle, registration=None):
    (
        engine,
        ordinary,
        early_periods,
        early,
        baseline,
        own,
        interline,
        full,
        periods,
        registered,
    ) = bundle
    del engine
    return {
        "registration": registered if registration is None else registration,
        "ordinary_input": ordinary,
        "early_periods": early_periods,
        "early_source": early,
        "association": ChoiceBoundaryAssociation(
            registered.original_law.original_person_label,
            "NEW same root102974",
            "unit102974",
        ),
        "baseline": baseline,
        "own": own,
        "interline": interline,
        "later_periods": periods,
        "later_source": full,
        "observation": StageSignalObservation(
            "category102974",
            "category",
            "view-a",
            "declared category102974",
            "signal102974",
            "stage_signal",
            "notice-high",
            "declared signal102974",
            "explicit NEW receipt declaration",
        ),
    }


def updated(registration, **changes):
    r = replace(registration, **changes, digest="")
    return subject.register_terminal_material(
        r.original_law,
        r.transport_kernel,
        r.programme,
        r.terminal_laws,
        r.source_context,
        r.source_programme,
        definition=r.definition,
        source_ref=r.source_ref,
        row_limit=r.row_limit,
    )


def independent(registration):
    original = registration.original_law
    kernel = registration.transport_kernel
    pr = registration.programme
    selected = next(
        i for i, p in enumerate(original.products) if p.product_id == "B_OH0"
    )
    raw = {coord: F() for coord in product(range(2), range(3), range(2))}
    e = F()
    for z, k, u, g, x, y in product(
        range(2), range(3), range(2), range(2), range(2), range(2)
    ):
        mass = (
            original.initial_joint[z][k][u][g]
            * original.ordinary_likelihood[k][u][selected]
            * original.boundary_kernel[z][k][u][g][selected][x][1]
        )
        raw[y, k, g] += mass * kernel.probabilities[z][k][u][g][selected][x][1][y]
    e = sum(raw.values(), F())
    out = []
    if not e:
        return tuple((None, None) for _ in registration.terminal_laws)
    for law in registration.terminal_laws:
        evidence = F()
        known = F()
        unknown = F()
        for y, k, g, yn, a, d in product(
            range(2), range(3), range(2), range(2), range(3), range(4)
        ):
            if pr.type_observation[k] != "view-a":
                continue
            mass = (
                raw[y, k, g]
                / e
                * pr.own_program[0][g][-1]
                * pr.next_state_signal[y][k][g][yn][1]
                * pr.interline_program[0][1][g][a]
                * law.probabilities[yn][k][g][a][d]
            )
            account = law.accounts[yn][k][g][a][d]
            values = account.payoff.components()
            evidence += mass
            if (
                any(x is None for x in values)
                or not account.financial_tail_closed
                or not account.material_tail_closed
            ):
                unknown += mass
            else:
                known += mass * (values[0] - values[1] - values[2] - values[3])
        out.append(
            (
                None if not evidence or unknown else known / evidence,
                None if not evidence else unknown / evidence,
            )
        )
    return tuple(out)


def test_full_original_future_terminal_reference_and_identity(bundle):
    reg = bundle[-1]
    before = pickle.dumps(bundle[0].snapshot())
    rng = np.random.get_state()
    digest = reg.digest
    result = subject.bind_terminal_material(**args(bundle))
    expected = independent(reg)
    for row, (value, unknown) in zip(result.analyst["terminal_projections"], expected):
        assert row["conditional_material_expectation"] == value
        assert row["conditional_unknown_probability"] == unknown
        assert len(row["rows"]) == 1296
        assert any(
            r["known_material"] is not None and r["known_material"] < 0
            for r in row["rows"]
        )
        assert any(
            r["selected_terminal_joint"] == 0 and r["known_material"] is None
            for r in row["rows"]
        )
    assert before == pickle.dumps(bundle[0].snapshot()) and reg.digest == digest
    now = np.random.get_state()
    assert rng[0] == now[0] and np.array_equal(rng[1], now[1]) and rng[2:] == now[2:]
    assert (
        _units(subject.terminal_material_record(result))
        < result.analyst["structural_bound"]
    )
    assert not result.analyst[
        "registration_preceded_source_or_external_time_authenticated"
    ]


def test_zero_upstream_retains_template_no_downstream_joint_fill(bundle, monkeypatch):
    reg = bundle[-1]
    law = reg.original_law

    def kernel(z, k, u, g, a, x, c):
        q = F(2 + k + c, 6)
        return (q if x else 1 - q) if not c else F()

    new = updated(
        reg,
        original_law=replace(
            law, boundary_kernel=nested((2, 3, 2, 2, 3, 2, 2), kernel)
        ),
    )
    import airline_market.sim.transported_stage_signal as parent

    def forbidden(*a, **kw):
        raise AssertionError("no downstream compiler on zero upstream")

    monkeypatch.setattr(parent, "compile_type_observation_attempt", forbidden)
    result = subject.bind_terminal_material(**args(bundle, new))
    for row in result.analyst["terminal_projections"]:
        assert row["conditional_material_expectation"] is None and row["rows"] == ()
    assert len(result.analyst["registration"]["terminal_laws"]) == 3


@pytest.mark.parametrize("which", ["category", "signal"])
def test_zero_received_history_keeps_all_rows_and_null_conditional(bundle, which):
    reg = bundle[-1]
    pr = reg.programme
    if which == "category":
        pr = replace(pr, type_observation=("view-b",) * 3)
    else:
        pr = replace(
            pr,
            next_state_signal=nested(
                (2, 3, 2, 2, 3), lambda y, k, g, yn, s: F(yn == y and s == 2)
            ),
        )
    result = subject.bind_terminal_material(**args(bundle, updated(reg, programme=pr)))
    for row in result.analyst["terminal_projections"]:
        assert (
            row["selected_evidence"] == 0
            and row["conditional_material_expectation"] is None
        )
        assert (
            len(row["rows"]) == 1296 and row["conditional_unknown_probability"] is None
        )


def test_exact_tiny_unknown_and_zero_unknown_not_float_filtered(bundle):
    result = subject.bind_terminal_material(**args(bundle))
    closed, unknown, tiny = result.analyst["terminal_projections"]
    assert (
        closed["conditional_material_expectation"] is not None
        and closed["selected_unknown_mass"] == 0
    )
    assert (
        unknown["conditional_unknown_probability"] == F(1, 9)
        and unknown["conditional_material_expectation"] is None
    )
    assert (
        tiny["conditional_unknown_probability"] == F(1, 10**950) > 0
        and float(tiny["conditional_unknown_probability"]) == 0
    )
    assert tiny["conditional_material_expectation"] is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("physical_states", ("wrong", "y1")),
        ("latent_types", ("wrong", "k1", "zero-k")),
        ("private_marks", ("wrong", "g1")),
        ("action_products", ("wrong", None)),
        ("probabilities", ()),
        ("accounts", ()),
        ("outcomes", ()),
        ("payoff_unit", "wrong"),
        ("scope", "wrong"),
        ("choice_shock_scope", "wrong"),
        ("definition", ""),
    ],
)
def test_complete_terminal_laws_refused_before_probability_forward(
    bundle, monkeypatch, field, value
):
    reg = bundle[-1]

    def forbidden(*a, **kw):
        raise AssertionError("invalid terminal support reached probability forward")

    monkeypatch.setattr(subject, "bind_transported_stage_signal", forbidden)
    with pytest.raises((TypeError, ValueError)):
        updated(reg, terminal_laws=(replace(reg.terminal_laws[0], **{field: value}),))


@pytest.mark.parametrize(
    "field,value",
    [
        ("component_sources", ("only-one",)),
        ("unit", ""),
        ("financial_tail_closed", 1),
        ("material_tail_closed", None),
        ("definition", ""),
    ],
)
def test_explicit_account_contract(field, value, bundle):
    account = bundle[-1].terminal_laws[0].accounts[0][0][0][0][0]
    with pytest.raises((TypeError, ValueError)):
        replace(account, **{field: value})


@pytest.mark.parametrize(
    "case", ["digest", "programme", "policy", "clock", "source_spec"]
)
def test_registered_full_source_plan_mismatch_before_forward(bundle, monkeypatch, case):
    kw = args(bundle)
    reg = kw["registration"]
    if case == "digest":
        kw["registration"] = replace(reg, digest="not-the-registered-bytes")
    else:
        source = copy.deepcopy(_encoded(kw["later_source"]))
        if case == "programme":
            source["programme"]["definition"] = "changed future programme"
        elif case == "policy":
            source["parent_source"]["source_context"]["dated_rebooking"] = {}
        elif case == "clock":
            source["parent_source"]["source_context"]["scenario"]["clock_utc"] = ()
        else:
            source["parent_source"]["source_context"]["spec_hash"] = "changed"
        kw["later_source"] = source

    def forbidden(*a, **kw):
        raise AssertionError("source plan check must precede probability forwards")

    monkeypatch.setattr(subject, "bind_transported_stage_signal", forbidden)
    with pytest.raises(ValueError):
        subject.bind_terminal_material(**kw)


def test_all_support_limits_before_forward_and_no_input_mutation(bundle, monkeypatch):
    reg = bundle[-1]
    before = subject.material_registration_record(reg)

    def forbidden(*a, **kw):
        raise AssertionError("limit precedes probability forward")

    monkeypatch.setattr(subject, "bind_transported_stage_signal", forbidden)
    with pytest.raises(RuntimeError):
        updated(reg, row_limit=1)
    assert subject.material_registration_record(reg) == before


def test_one_fresh_binding_all_cases_and_private_owned_exports(bundle, monkeypatch):
    calls = []
    original = subject.bind_transported_stage_signal

    def count(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    monkeypatch.setattr(subject, "bind_transported_stage_signal", count)
    reg = bundle[-1]
    result = subject.bind_terminal_material(**args(bundle))
    assert calls == [1]
    view = subject.terminal_material_view(result)
    for word in (
        "terminal_laws",
        "conditional_material_expectation",
        "registration",
        "initial_joint",
        "analyst_root",
    ):
        assert word not in str(view)
    exported = subject.material_registration_record(reg)
    exported["source_context"] = {}
    assert reg.source_context
    with pytest.raises(TypeError):
        reg.source_context["spec_hash"] = "changed"
    output = subject.terminal_material_record(result)
    output["analyst"] = {}
    assert result.analyst["registration"]


def test_projection_failure_propagates_without_input_or_native_mutation(
    bundle, monkeypatch
):
    before = pickle.dumps(bundle[0].snapshot())
    reg = bundle[-1]
    digest = reg.digest

    def failure(*a, **kw):
        raise RuntimeError("NEW terminal projection injected failure")

    monkeypatch.setattr(subject, "_projection", failure)
    with pytest.raises(RuntimeError, match="NEW terminal projection"):
        subject.bind_terminal_material(**args(bundle))
    assert reg.digest == digest and before == pickle.dumps(bundle[0].snapshot())


def test_open_financial_tail_cannot_label_known_prefix_as_net_payment(bundle):
    account = bundle[-1].terminal_laws[0].accounts[0][0][0][0][0]
    with pytest.raises(ValueError, match="open financial tail"):
        replace(account, financial_tail_closed=False)


def test_open_material_tail_keeps_complete_expectation_null(bundle):
    reg = bundle[-1]
    laws = []
    for law in reg.terminal_laws:
        accounts = nested(
            (2, 3, 2, 3, 4),
            lambda y, k, g, a, d, law=law: replace(
                law.accounts[y][k][g][a][d], material_tail_closed=False
            ),
        )
        laws.append(replace(law, accounts=accounts))
    result = subject.bind_terminal_material(
        **args(bundle, updated(reg, terminal_laws=tuple(laws)))
    )
    for row in result.analyst["terminal_projections"]:
        assert row["conditional_material_expectation"] is None
        assert row["conditional_unknown_probability"] == 1
        assert any(r["known_material"] is not None for r in row["rows"])


@pytest.mark.parametrize("bad", [F(), F(-1)])
def test_invalid_unused_zero_class_terminal_kernel_before_forward(
    bundle, bad, monkeypatch
):
    reg = bundle[-1]
    law = reg.terminal_laws[0]
    invalid = nested(
        (2, 3, 2, 3, 4),
        lambda y, k, g, a, d: (
            bad if (y, k, g, a) == (1, 2, 1, 1) else law.probabilities[y][k][g][a][d]
        ),
    )

    def forbidden(*a, **kw):
        raise AssertionError(
            "unused zero-support validation before probability forward"
        )

    monkeypatch.setattr(subject, "bind_transported_stage_signal", forbidden)
    with pytest.raises((TypeError, ValueError)):
        updated(reg, terminal_laws=(replace(law, probabilities=invalid),))


@pytest.mark.parametrize("sources", [{"a": 1, "b": 2, "c": 3, "d": 4}, "abcd"])
def test_new_component_sources_require_explicit_ordered_sequence(sources):
    with pytest.raises(TypeError, match="ordered sequence"):
        subject.TerminalMaterialAccount(
            ContingentPayoff(F(1), F(), F(), F()),
            sources,
            UNIT,
            True,
            True,
            "NEW explicit source order",
        )


def test_new_ordered_component_source_list_is_owned():
    values = [
        "benefit source",
        "payment source",
        "waiting source",
        "cancellation source",
    ]
    account = subject.TerminalMaterialAccount(
        ContingentPayoff(F(1), F(), F(), F()),
        values,
        UNIT,
        True,
        True,
        "NEW ordered ownership",
    )
    values[0] = "changed"
    assert account.component_sources == (
        "benefit source",
        "payment source",
        "waiting source",
        "cancellation source",
    )
