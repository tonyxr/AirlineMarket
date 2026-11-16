"""A.10/A.5/A.14/B0/V9/B4: NEW August10/112974 genuine units only."""

import json
import pickle
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F

import numpy as np
import pytest

from airline_market.core.hashing import canonical_json
from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.dated_demand import compile_dated_demand
from airline_market.policy.dated_rescue import DatedRebookingPolicy
from airline_market.policy.dated_reserve import UtcBookingClock
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.policy.regulator import NO_CAP
from airline_market.sim import full_path_utility as utility
from airline_market.sim import paired_history_binding as parent
from airline_market.sim import paired_material as pm
from airline_market.sim import registered_path_utility as subject
from airline_market.sim import terminal_material as tm
from airline_market.sim.booking_loss_binding import (
    ChoiceBoundaryAssociation,
)
from airline_market.sim.booking_loss_law import BookingLossLaw, OriginalOffer
from airline_market.sim.full_path_transport import FullPathKernel
from airline_market.sim.native_causal_information import RULE as NATIVE_RULE
from airline_market.sim.native_causal_information import (
    TIMING,
    DeclaredInputDelivery,
    DeclaredKnownType,
    NativeCausalInformationMarketEngine,
    native_causal_information_record,
)
from airline_market.sim.payoff_logit import ContingentPayoff
from airline_market.sim.pretype_observation import _same
from airline_market.sim.received_information import DisclosureField
from airline_market.sim.shared_mark_attempt import AttemptOffer, _encoded
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.sim.transported_stage_signal import (
    StageSignalObservation,
    StageSignalProgramme,
)
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

LIMIT = 2_000_000_000
UNIT = "declared-fare-unit112974"
PLAN = "A.10/A.5/A.14/B0/V9/B4/R3/R4"
EVENTS = []
CALLS = {}


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
    start = datetime(2027, 8, 10, tzinfo=timezone.utc)
    clock = UtcBookingClock(tuple(start + timedelta(hours=12 * i) for i in range(4)))
    departures = tuple(
        (leg.id, start + timedelta(hours=84 if leg.id == "B_OH0" else 120))
        for leg in compiled.spec.legs
    )
    scenario = replace(
        compile_dated_demand(compiled, clock, departures),
        physical_shocks=(
            B1DatedShock(
                "new_signal_exit112974", "rival_exit", "B_OH0", 3, rival_carrier="B"
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
                "unit112974",
            )
            for i in range(2)
        ),
        "causal_deliveries": (
            DeclaredInputDelivery(
                key,
                1,
                "own",
                "category112974",
                "category",
                "view-a",
                "declared category112974",
                TIMING,
            ),
            DeclaredInputDelivery(
                key,
                1,
                "interline",
                "signal112974",
                "stage_signal",
                "notice-high",
                "declared signal112974",
                TIMING,
            ),
        ),
        "causal_definition": "NEW transported joint and received stage signal112974",
        "causal_unobserved_definition": "caller schedule not authenticated reception, knowledge, physical law or preference",
        "causal_row_limit": 2000000,
        "cap_policy": NO_CAP,
    }
    registered = register(compiled, scenario, clock, departures, source_kwargs)
    EVENTS.append("before_tape")
    tape = ExogenousTape.generate(
        compiled, 31, registry=StreamRegistry(112974, "unit/V9_registered_path_utility")
    )
    assert_check(tape.gumbel.shape[1] >= 2, "fresh generated support insufficient")
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
            {j: 250 if i == 0 else 375 for j in compiled.products_of[i]}
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


def register_numeric_base(compiled, scenario, clock, departures, kwargs):
    del clock, departures
    market = int(compiled.product_market[compiled.product_index["B_OH0"]])
    products = tuple(
        p
        for (j, p) in enumerate(compiled.spec.products)
        if int(compiled.product_market[j]) == market
    )
    offers = tuple(
        OriginalOffer(p.id, 250.0 if p.carrier == "A" else 375.0, True)
        for p in products
    )
    a = len(offers) + 1
    ib = next(i for (i, p) in enumerate(offers) if p.product_id == "B_OH0")
    ia = next(i for (i, p) in enumerate(offers) if p.product_id.startswith("A_"))
    pk = (F(4, 11), F(7, 11), F())
    pz = (F(2, 5), F(3, 7), F(1, 2))
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
        "material-person112974",
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
        "NEW material law112974",
        "declared112974",
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
        "full path112974",
        "declared112974",
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
        "NEW material programme112974",
        "declared112974",
    )
    reference = replace(
        law,
        initial_joint=nested(
            (2, 3, 2, 2),
            lambda z, k, u, g: (
                (F(5, 12), F(7, 12), F())[k]
                * (
                    (F(1, 4), F(2, 3), F(1, 2))[k]
                    if z
                    else 1 - (F(1, 4), F(2, 3), F(1, 2))[k]
                )
                * ((F(2, 5), F(1, 10)), (F(1, 5), F(3, 10)))[u][g]
            ),
        ),
        ordinary_likelihood=nested(
            (3, 2, a),
            lambda k, u, j: (
                F(2 + k + u, 9)
                if j == ib
                else F(1, 8)
                if j == ia
                else 1 - F(2 + k + u, 9) - F(1, 8)
            ),
        ),
        boundary_kernel=nested(
            (2, 3, 2, 2, a, 2, 2),
            lambda z, k, u, g, j, x, c: (
                (
                    (F(3 + z + u + g, 11) if c else 1 - F(3 + z + u + g, 11))
                    if j < a - 1
                    else F(not c)
                )
                * (F(1 + k + c, 5) if x else 1 - F(1 + k + c, 5))
            ),
        ),
        definition="NEW separate reference original law112974",
        source_ref="declared-reference112974",
    )
    reference_kernel = replace(
        kernel,
        probabilities=nested(
            (2, 3, 2, 2, a, 2, 2, 2),
            lambda z, k, u, g, j, x, c, y: (
                F(1 + z + 2 * u + g + x, 8) if y else 1 - F(1 + z + 2 * u + g + x, 8)
            ),
        ),
        definition="NEW reference full path112974",
    )
    support = pm.MaterialSupport(
        law.original_person_label,
        compiled.spec.markets[market].id,
        "B",
        kernel.future_states,
        law.latent_types,
        law.attempt_marks,
        programme.observations,
        programme.type_observation,
        tuple("no-signal" if s is None else s for s in programme.signals),
        tuple(p.product_id for p in offers),
        tuple(p.carrier for p in products),
        tuple(tuple(p.offered for p in row) for row in programme.own_menus),
        tuple(
            tuple(tuple(p.offered for p in row) for row in obs)
            for obs in programme.interline_menus
        ),
        True,
        ("declared-served", "declared-not-served", "unknown"),
        UNIT,
        pm.SCOPE,
        "original history then category/signal ONCE; not authenticated sufficient history",
        programme.observation_definition,
        programme.private_mark_interpretation,
        pm.SHOCK_SCOPE,
    )
    policy = pm.FixedMaterialPolicy(
        programme.own_program,
        programme.interline_program,
        "ONE explicit fixed policy112974; not optimized native preferences",
        "declared-policy112974",
    )

    def future(role, rho):
        transition = (
            programme.next_state_signal
            if role == 0
            else nested(
                (2, 3, 2, 2, 3),
                lambda y, k, g, yn, s: (
                    (F(1 + y + g, 4) if yn else 1 - F(1 + y + g, 4))
                    * (
                        F()
                        if s == 0
                        else F(1 + k + g + yn, 7)
                        if s == 1
                        else 1 - F(1 + k + g + yn, 7)
                    )
                ),
            )
        )

        def probability(stage, y, k, g, j, d):
            if d == 2:
                return rho
            served = F(2 + y + k + g + role, 8)
            return (1 - rho) * (served if d == 0 else 1 - served)

        def account(stage, y, k, g, j, d):
            components = (
                (None,) * 4
                if d == 2
                else (
                    F(650 + 40 * k + 15 * stage) if d == 0 else F(15),
                    F() if j == a - 1 else F(offers[j].fare) * (F(1, 3) if d else F(1)),
                    F(7 + 2 * y + g, 3),
                    F(13 + k) if d else F(-2),
                )
            )
            return tm.TerminalMaterialAccount(
                ContingentPayoff(*components),
                tuple(
                    "explicit declared " + n + "112974" for n in ("B", "P", "W", "C")
                ),
                UNIT,
                d != 2,
                d != 2,
                "NEW declared material; not actual service/payment",
            )

        return {
            "transition_signal": transition,
            "terminal_probabilities": nested((2, 2, 3, 2, a, 3), probability),
            "terminal_accounts": nested((2, 2, 3, 2, a, 3), account),
            "definition": "NEW full future declaration112974",
            "source_ref": "declared112974",
        }

    data = {
        "original_laws": (law, reference),
        "kernels": (kernel, reference_kernel),
        "programme": programme,
        "observation": StageSignalObservation(
            "category112974",
            "category",
            "view-a",
            "declared category112974",
            "signal112974",
            "stage_signal",
            "notice-high",
            "declared signal112974",
            "NEW source-association declaration112974",
        ),
        "support": support,
        "policy": policy,
        "future": (future(0, F()), future(1, F())),
        "source_plan": {
            "spec": compiled.spec,
            "scenario": scenario.record(),
            "kwargs": kwargs,
        },
        "registration_scope": "local before NEW tape/engine, not external preregistration/time/native truth",
    }
    return data


def register(compiled, scenario, clock, departures, kwargs):
    data = register_numeric_base(compiled, scenario, clock, departures, kwargs)
    for future in data["future"]:

        def known(stage, y, k, g, action, outcome, future=future):
            account = future["terminal_accounts"][stage][y][k][g][action][outcome]
            if outcome == 2:
                return replace(
                    account,
                    payoff=ContingentPayoff(F(-9), F(-3), F(4, 7), F(2)),
                    financial_tail_closed=True,
                    material_tail_closed=True,
                )
            return account

        future["terminal_accounts"] = nested((2, 2, 3, 2, 3, 3), known)
    data["utilities"] = tuple(
        utility_law(data["support"], name, role)
        for role, name in enumerate(("forecast", "reference"))
    )
    data["plan"] = make_plan(compiled, scenario, kwargs, data)
    plan = data["plan"]
    incidence = subject.UtilityIncidence(
        **{k: v for k, v in options().items() if k != "row_limit"}
    )
    arguments = {
        k: plan[k]
        for k in (
            "programme",
            "forecast",
            "reference",
            "source_context",
            "source_programme",
            "association",
            "observation",
            "definition",
            "source_ref",
        )
    }
    arguments.update(
        forecast_utility=data["utilities"][0],
        reference_utility=data["utilities"][1],
        incidence=incidence,
        fixed_postings=tuple(
            subject.FixedPosting(**p) for p in plan["fixed_regular_postings"]
        ),
        row_limit=LIMIT,
    )
    data["registration_arguments"] = arguments
    data["registration"] = subject.register_paired_utility(**arguments)
    zero_original = replace(
        arguments["reference"].original,
        ordinary_likelihood=nested(
            (3, 2, 3), lambda k, u, a: F(1, 3) if a == 0 else F() if a == 1 else F(2, 3)
        ),
    )
    data["zero_registration"] = subject.register_paired_utility(
        **dict(
            arguments, reference=replace(arguments["reference"], original=zero_original)
        )
    )
    tiny = F(1, 10**1200)
    ref = arguments["reference"]
    tiny_prob = nested(
        (2, 2, 3, 2, 3, 3),
        lambda stage, y, k, g, a, d: (
            tiny
            if d == 2
            else (1 - tiny) * ref.terminal_probabilities[stage][y][k][g][a][d]
        ),
    )
    law = arguments["reference_utility"]
    unknown = nested(
        (2, 2, 3, 2, 3, 3),
        lambda stage, y, k, g, a, d: (
            replace(law.accounts[stage][y][k][g][a][d], additional_utility=None)
            if d == 2
            else law.accounts[stage][y][k][g][a][d]
        ),
    )
    data["tiny_registration"] = subject.register_paired_utility(
        **dict(
            arguments,
            reference=replace(ref, terminal_probabilities=tiny_prob),
            reference_utility=replace(law, accounts=unknown),
        )
    )
    EVENTS.append("registered")
    return data


def utility_law(support, name, role):
    def leaf(stage, y, k, g, action, outcome):
        outside = tm.TerminalMaterialAccount(
            ContingentPayoff(
                F(21 + 5 * k + y), F(-4 if outcome else 2), F(1 + g, 7), F(3 - y)
            ),
            tuple("explicit112974:" + n for n in ("B0", "P0", "W0", "C0")),
            UNIT,
            True,
            True,
            "declared full same-path outside material112974",
        )
        return utility.TerminalUtilityAccount(
            F((-1) ** (action + stage + g) * (5 + y + k + g), 13) + F(role, 17),
            "declared E112974; not decoded native Gumbel",
            True,
            outside,
            F(3 + y + k - 2 * g, 11) + F(role, 19),
            "declared E0-112974",
            True,
            "explicit all-component followup112974",
        )

    return utility.FullPathUtilityLaw(
        name,
        support.physical_states,
        support.latent_types,
        support.private_marks,
        (*support.products, None),
        support.outcomes,
        nested((2, 2, 3, 2, 3, 3), leaf),
        "complete terminal utility112974",
        "unit-declared112974",
    )


def options():
    return {
        "alpha": F(1, 97),
        "material_unit": UNIT,
        "utility_unit": "declared-stage-utility112974",
        "material_additional_nonoverlap": "FOUR material components exclude all E112974",
        "utility_origin_and_incidence": "explicit terminal incidence, no native interpretation112974",
        "definition": "NEW declared fixed-policy full-path expectation112974",
        "reference_scope": utility.REFERENCE_SCOPE,
        "row_limit": LIMIT,
    }


def make_plan(compiled, scenario, kwargs, data):
    programme = parent.PairedHistoryProgramme(
        data["support"],
        data["policy"],
        data["programme"],
        "no-signal",
        "NEW prospectively registered pair112974",
        "declared112974",
    )
    laws = tuple(
        parent.OriginalHistoryMaterialLaw(
            name, data["original_laws"][i], data["kernels"][i], **data["future"][i]
        )
        for i, name in enumerate(("forecast", "reference"))
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
    causal = {
        "rule": NATIVE_RULE,
        "spec_hash": compiled.spec_hash,
        "fields": kwargs["causal_fields"],
        "known_types": kwargs["causal_known_types"],
        "deliveries": kwargs["causal_deliveries"],
        "definition": kwargs["causal_definition"],
        "unobserved_definition": kwargs["causal_unobserved_definition"],
        "row_limit": kwargs["causal_row_limit"],
    }
    postings = tuple(
        {
            "period": t,
            "product_ids": tuple(p.id for p in compiled.spec.products),
            "prices": tuple(
                250.0 if p.carrier == "A" else 375.0 for p in compiled.spec.products
            ),
        }
        for t in range(compiled.spec.horizon.n_periods)
    )
    return {
        "rule": "prospective-paired-utility-design-reference-v1",
        "plan_id": PLAN,
        "programme": programme,
        "forecast": laws[0],
        "reference": laws[1],
        "utilities": data["utilities"],
        "utility_options": options(),
        "source_context": context,
        "source_programme": causal,
        "fixed_regular_postings": postings,
        "association": ChoiceBoundaryAssociation(
            data["original_laws"][0].original_person_label,
            "SAME actual original history112974",
            "declared112974",
        ),
        "observation": data["observation"],
        "structural_limit": LIMIT,
        "definition": "Complete selected saved numerical/source plan; local order not external preregistration",
        "source_ref": "new-standalone112974",
    }


def assert_check(condition, message):
    assert condition, message


@pytest.fixture(scope="module")
def source():
    bundle = fresh_source(register)
    engine = bundle[0]
    packet = {
        "ordinary_input": bundle[1],
        "early_periods": bundle[2],
        "early_source": bundle[3],
        "baseline": bundle[4],
        "own": bundle[5],
        "interline": bundle[6],
        "later_source": bundle[7],
        "later_periods": bundle[8],
        "source_trace": engine.trace(),
    }
    return bundle[9], packet, engine


@pytest.fixture(scope="module")
def result(source):
    data, packet, engine = source
    snapshot = pickle.dumps(engine.snapshot(), protocol=5)
    rng = pickle.dumps(np.random.get_state(), protocol=5)
    mp = pytest.MonkeyPatch()
    for owner, name in (
        (parent, "bind_full_path_transport"),
        (parent, "evaluate_paired_material"),
        (utility, "evaluate_full_path_utility"),
    ):
        original = getattr(owner, name)

        def observed(*args, _original=original, _name=name, **kwargs):
            CALLS[_name] = CALLS.get(_name, 0) + 1
            return _original(*args, **kwargs)

        mp.setattr(owner, name, observed)
    try:
        value = subject.bind_registered_path_utility(data["registration"], **packet)
    finally:
        mp.undo()
    assert pickle.dumps(engine.snapshot(), protocol=5) == snapshot
    assert pickle.dumps(np.random.get_state(), protocol=5) == rng
    return value


def mutable(value):
    return json.loads(json.dumps(_encoded(value)))


def test_registration_precedes_source_and_owns_payload(source):
    data, _, _ = source
    assert EVENTS == ["registered", "before_tape"]
    r = data["registration"]
    assert len(r.digest) == 64 and r.rule == subject.RULE
    assert not hasattr(r, "structural_bound")
    context = mutable(r.source_context)
    context["scenario"] = "changed"
    assert r.source_context["scenario"] != "changed"
    assert all(type(p) is subject.FixedPosting for p in r.fixed_postings)


def test_full_binding_and_three_paths(result):
    a = utility._decode(result.analyst)
    assert a["attached_context_paths"] == subject.CONTEXT_PATHS
    assert a["paired_binding_calls"] == a["utility_consumer_calls"] == 1
    assert a["paired_binding"]["analyst"]["original_conditioning_calls"] == 2
    assert all(
        x > 0 for x in a["paired_binding"]["analyst"]["original_history_evidence"]
    )
    assert a["utility_result"]["analyst"]["expectations"] is not None
    assert a["native_source_or_probability_authenticated"] is False
    assert a["lifecycle_CS"] is None


def test_view_is_exact_parent_and_has_no_future_tables(result):
    assert _same(result.view, result.analyst["utility_result"]["view"])
    assert _same(result.view, result.analyst["paired_binding"]["view"])
    text = canonical_json(subject.registered_path_utility_view(result))
    assert "terminal_accounts" not in text and "initial_joint" not in text
    assert "registration" not in text and "utility_origin" not in text


def test_result_exports_are_owned(result):
    record = subject.registered_path_utility_record(result)
    record["analyst"]["registration"]["definition"] = "altered"
    assert result.analyst["registration"]["definition"] != "altered"


@pytest.mark.parametrize(
    "field,value", [("rule", "old-rule"), ("digest", "0" * 64), ("row_limit", 1)]
)
def test_changed_registration_refused_before_forwards(
    source, monkeypatch, field, value
):
    data, packet, _ = source
    monkeypatch.setattr(
        parent, "bind_paired_history", lambda *a, **k: pytest.fail("unexpected forward")
    )
    r = replace(data["registration"], **{field: value})
    with pytest.raises((TypeError, ValueError, subject.RegisteredUtilityLimit)):
        subject.bind_registered_path_utility(r, **packet)


@pytest.mark.parametrize(
    "kind",
    [
        "missing_static",
        "missing_programme",
        "alpha",
        "units",
        "scope",
        "policy",
        "utility_shape",
        "zero_unit",
        "posting_missing",
        "posting_order",
        "posting_integer",
        "posting_nan",
        "limit",
        "association",
        "observation",
    ],
)
def test_complete_registration_refusals(source, monkeypatch, kind):
    data, _, _ = source
    args = dict(data["registration_arguments"])
    monkeypatch.setattr(
        parent, "bind_paired_history", lambda *a, **k: pytest.fail("unexpected forward")
    )
    if kind == "missing_static":
        args["source_context"] = dict(args["source_context"])
        args["source_context"].pop("scenario")
    elif kind == "missing_programme":
        args["source_programme"] = dict(args["source_programme"])
        args["source_programme"].pop("deliveries")
    elif kind in ("alpha", "units", "scope"):
        changes = {
            "alpha": {"alpha": F()},
            "units": {"utility_unit": UNIT},
            "scope": {"reference_scope": "outside intervention"},
        }
        args["incidence"] = replace(args["incidence"], **changes[kind])
    elif kind == "policy":
        p = args["programme"]
        args["programme"] = replace(p, policy=replace(p.policy, own_program=()))
    elif kind == "utility_shape":
        args["forecast_utility"] = replace(args["forecast_utility"], accounts=())
    elif kind == "zero_unit":
        law = args["reference_utility"]

        def bad(stage, y, k, g, a, d):
            acc = law.accounts[stage][y][k][g][a][d]
            return (
                replace(
                    acc, outside_material=replace(acc.outside_material, unit="wrong")
                )
                if k == 2
                else acc
            )

        args["reference_utility"] = replace(
            law, accounts=nested((2, 2, 3, 2, 3, 3), bad)
        )
    elif kind.startswith("posting_"):
        rows = list(args["fixed_postings"])
        if kind == "posting_missing":
            rows.pop()
        elif kind == "posting_order":
            rows.reverse()
        else:
            prices = list(rows[2].prices)
            prices[0] = 250 if kind == "posting_integer" else float("nan")
            rows[2] = replace(rows[2], prices=prices)
        args["fixed_postings"] = rows
    elif kind == "limit":
        args["row_limit"] = 1
    elif kind == "association":
        args["association"] = replace(
            args["association"], original_person_label="other"
        )
    elif kind == "observation":
        args["observation"] = replace(args["observation"], category="missing")
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        subject.register_paired_utility(**args)


@pytest.mark.parametrize(
    "kind",
    [
        "early_static",
        "later_static",
        "early_programme",
        "future_delivery",
        "empty_posting",
        "missing_posting",
        "reordered_posting",
        "later_baseline",
    ],
)
def test_source_refusals_before_probability_forwards(source, monkeypatch, kind):
    data, packet, _ = source
    packet = dict(packet)
    monkeypatch.setattr(
        parent, "bind_paired_history", lambda *a, **k: pytest.fail("unexpected forward")
    )
    if kind in ("early_static", "later_static"):
        key = "early_source" if kind == "early_static" else "later_source"
        packet[key] = mutable(packet[key])
        packet[key]["parent_source"]["source_context"]["dated_menu"] = "different"
    elif kind in ("early_programme", "future_delivery"):
        key = "early_source" if kind == "early_programme" else "later_source"
        packet[key] = mutable(packet[key])
        packet[key]["programme"]["definition"] = "different"
    elif kind in ("empty_posting", "missing_posting", "reordered_posting"):
        trace = mutable(packet["source_trace"])
        if kind == "empty_posting":
            trace[2]["prices"][0] = _encoded(251.0)
        elif kind == "missing_posting":
            trace.pop()
        else:
            trace.reverse()
        packet["source_trace"] = trace
    else:
        packet["baseline"] = packet["later_periods"][-1].journal
    with pytest.raises((TypeError, ValueError)):
        subject.bind_registered_path_utility(data["registration"], **packet)


def test_recomputed_limit_has_no_trusted_wrapper(source, monkeypatch):
    data, packet, _ = source
    r = data["registration"]
    with pytest.raises(TypeError):
        replace(r, structural_bound=0)
    # Even a self-consistent newly selected low-limit digest cannot bypass bound.
    small = replace(r, row_limit=1)
    small = replace(small, digest=subject._digest(small))
    monkeypatch.setattr(
        parent, "bind_paired_history", lambda *a, **k: pytest.fail("unexpected forward")
    )
    with pytest.raises(subject.RegisteredUtilityLimit):
        subject.bind_registered_path_utility(small, **packet)


def test_pure_failure_keeps_native_state_and_inputs(source, monkeypatch):
    data, packet, engine = source
    before = pickle.dumps(engine.snapshot(), protocol=5)
    inputs = canonical_json(_encoded(packet))

    def fail(*_args, **_kwargs):
        raise RuntimeError("NEW injected pure binding failure")

    monkeypatch.setattr(parent, "bind_paired_history", fail)
    with pytest.raises(RuntimeError, match="NEW injected"):
        subject.bind_registered_path_utility(data["registration"], **packet)
    assert pickle.dumps(engine.snapshot(), protocol=5) == before
    assert canonical_json(_encoded(packet)) == inputs


def test_zero_original_evidence_retains_null_without_pair(source, monkeypatch):
    data, packet, _ = source

    def forbidden(*_a, **_k):
        pytest.fail("undefined original evidence must not create pair")

    monkeypatch.setattr(parent, "evaluate_paired_material", forbidden)
    result = subject.bind_registered_path_utility(data["zero_registration"], **packet)
    a = utility._decode(result.analyst)
    assert a["paired_binding"]["analyst"]["original_history_evidence"][1] == 0
    assert a["paired_binding"]["analyst"]["paired"] is None
    assert a["utility_result"]["analyst"]["expectations"] is None
    assert a["registration"]["reference_utility"] is not None


def test_tiny_positive_unknown_utility_is_complete_null(source):
    data, packet, _ = source
    result = subject.bind_registered_path_utility(data["tiny_registration"], **packet)
    fields = utility._decode(result.analyst)["utility_result"]["analyst"][
        "expectations"
    ]["reference"]["summaries"]["root"]["fields"]
    assert fields["material"]["mean"] is not None
    assert fields["experienced_candidate"]["mean"] is None
    assert 0 < fields["experienced_candidate"]["unknown_mass"] < F(1, 10**1100)


def test_bound_recomputed_before_each_bind_and_no_extra_source_calls(
    source, monkeypatch
):
    data, packet, engine = source
    before = pickle.dumps(engine.snapshot(), protocol=5)
    calls = []
    original = subject._validate

    def checked(registration):
        calls.append(registration.digest)
        return original(registration)

    def stop(*_a, **_k):
        raise RuntimeError("after preflight")

    monkeypatch.setattr(subject, "_validate", checked)
    monkeypatch.setattr(parent, "bind_paired_history", stop)
    with pytest.raises(RuntimeError, match="after preflight"):
        subject.bind_registered_path_utility(data["registration"], **packet)
    assert calls == [data["registration"].digest]
    assert pickle.dumps(engine.snapshot(), protocol=5) == before


def test_actual_delegation_counts_and_no_native_or_global_rng_mutation(result):
    assert result is not None
    assert CALLS == {
        "bind_full_path_transport": 2,
        "evaluate_paired_material": 1,
        "evaluate_full_path_utility": 1,
    }
