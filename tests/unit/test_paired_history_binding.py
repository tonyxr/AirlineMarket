"""A.10/A.5/A.14/B0/V9: NEW paired original-history synthetic support."""

import copy
import pickle
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F
from itertools import product

import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.dated_demand import compile_dated_demand
from airline_market.policy.dated_rescue import DatedRebookingPolicy
from airline_market.policy.dated_reserve import UtcBookingClock
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.sim import paired_history_binding as subject
from airline_market.sim import paired_material as pm
from airline_market.sim import terminal_material as tm
from airline_market.sim.booking_loss_binding import (
    ChoiceBoundaryAssociation,
    choice_boundary_conditioning_context,
)
from airline_market.sim.booking_loss_law import BookingLossLaw, OriginalOffer
from airline_market.sim.full_path_transport import FullPathKernel
from airline_market.sim.native_causal_information import (
    TIMING,
    DeclaredInputDelivery,
    DeclaredKnownType,
    NativeCausalInformationMarketEngine,
    native_causal_information_record,
)
from airline_market.sim.payoff_logit import ContingentPayoff
from airline_market.sim.pretype_observation import pretype_history_view
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
UNIT = "declared-fare-unit106974"


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
    start = datetime(2027, 7, 6, tzinfo=timezone.utc)
    clock = UtcBookingClock(tuple(start + timedelta(hours=12 * i) for i in range(4)))
    departures = tuple(
        (leg.id, start + timedelta(hours=84 if leg.id == "B_OH0" else 120))
        for leg in compiled.spec.legs
    )
    scenario = replace(
        compile_dated_demand(compiled, clock, departures),
        physical_shocks=(
            B1DatedShock(
                "new_signal_exit106974", "rival_exit", "B_OH0", 3, rival_carrier="B"
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
                "unit106974",
            )
            for i in range(2)
        ),
        "causal_deliveries": (
            DeclaredInputDelivery(
                key,
                1,
                "own",
                "category106974",
                "category",
                "view-a",
                "declared category106974",
                TIMING,
            ),
            DeclaredInputDelivery(
                key,
                1,
                "interline",
                "signal106974",
                "stage_signal",
                "notice-high",
                "declared signal106974",
                TIMING,
            ),
        ),
        "causal_definition": "NEW transported joint and received stage signal106974",
        "causal_unobserved_definition": "caller schedule not authenticated reception, knowledge, physical law or preference",
        "causal_row_limit": 2000000,
    }
    registered = register(compiled, scenario, clock, departures, source_kwargs)
    tape = ExogenousTape.generate(
        compiled, 31, registry=StreamRegistry(106974, "unit/V9_paired_history_binding")
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
            {j: 340 if i == 0 else 490 for j in compiled.products_of[i]}
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
        for (j, p) in enumerate(compiled.spec.products)
        if int(compiled.product_market[j]) == market
    )
    offers = tuple(
        OriginalOffer(p.id, 340.0 if p.carrier == "A" else 490.0, True)
        for p in products
    )
    a = len(offers) + 1
    ib = next(i for (i, p) in enumerate(offers) if p.product_id == "B_OH0")
    ia = next(i for (i, p) in enumerate(offers) if p.product_id.startswith("A_"))
    pk = (F(3, 8), F(5, 8), F())
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
        "material-person106974",
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
        "NEW material law106974",
        "declared106974",
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
        "full path106974",
        "declared106974",
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
        "NEW material programme106974",
        "declared106974",
    )
    reference = replace(
        law,
        initial_joint=nested(
            (2, 3, 2, 2),
            lambda z, k, u, g: (
                (F(4, 9), F(5, 9), F())[k]
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
        definition="NEW separate reference original law106974",
        source_ref="declared-reference106974",
    )
    reference_kernel = replace(
        kernel,
        probabilities=nested(
            (2, 3, 2, 2, a, 2, 2, 2),
            lambda z, k, u, g, j, x, c, y: (
                F(1 + z + 2 * u + g + x, 8) if y else 1 - F(1 + z + 2 * u + g + x, 8)
            ),
        ),
        definition="NEW reference full path106974",
    )
    zero = replace(
        reference,
        ordinary_likelihood=nested(
            (3, 2, a),
            lambda k, u, j: F() if j == ib else F(1, 8) if j == ia else F(7, 8),
        ),
        definition="NEW zero original purchase evidence law106974",
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
        "ONE explicit fixed policy106974; not optimized native preferences",
        "declared-policy106974",
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
                    "explicit declared " + n + "106974" for n in ("B", "P", "W", "C")
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
            "definition": "NEW full future declaration106974",
            "source_ref": "declared106974",
        }

    data = {
        "original_laws": (law, reference, zero),
        "kernels": (kernel, reference_kernel, reference_kernel),
        "programme": programme,
        "observation": StageSignalObservation(
            "category106974",
            "category",
            "view-a",
            "declared category106974",
            "signal106974",
            "stage_signal",
            "notice-high",
            "declared signal106974",
            "NEW source-association declaration106974",
        ),
        "support": support,
        "policy": policy,
        "future": (future(0, F()), future(1, F()), future(1, F(1, 10**1100))),
        "source_plan": {
            "spec": compiled.spec,
            "scenario": scenario.record(),
            "kwargs": kwargs,
        },
        "registration_scope": "local before NEW tape/engine, not external preregistration/time/native truth",
    }
    return data


@pytest.fixture(scope="module")
def bundle():
    return fresh_source(register)


def arguments(bundle):
    (
        _engine,
        ordinary,
        early_periods,
        early,
        baseline,
        own,
        interline,
        full,
        periods,
        data,
    ) = bundle
    source_programme = replace(
        data["programme"],
        conditioning_view=pretype_history_view(baseline, ordinary.analyst_original_key),
    )
    programme = subject.PairedHistoryProgramme(
        data["support"],
        data["policy"],
        source_programme,
        "no-signal",
        "NEW typed full-history bridge106974",
        "declared106974",
    )
    originals = tuple(
        replace(
            law, conditioning_context=choice_boundary_conditioning_context(ordinary)
        )
        for law in data["original_laws"]
    )

    def law(i, future, name):
        return subject.OriginalHistoryMaterialLaw(
            name, originals[i], data["kernels"][i], **future
        )

    return {
        "programme": programme,
        "forecast": law(0, data["future"][0], "forecast"),
        "reference": law(1, data["future"][1], "reference"),
        "ordinary_input": ordinary,
        "early_periods": early_periods,
        "early_source": early,
        "association": ChoiceBoundaryAssociation(
            originals[0].original_person_label,
            "SAME actual history106974",
            "declared106974",
        ),
        "baseline": baseline,
        "own": own,
        "interline": interline,
        "later_periods": periods,
        "later_source": full,
        "observation": data["observation"],
        "row_limit": LIMIT,
    }


def test_independent_full_joint_reference_and_fixed_policy(bundle):
    args = arguments(bundle)
    before = pickle.dumps(bundle[0].snapshot(), protocol=5)
    result = subject.bind_paired_history(**args)
    assert pickle.dumps(bundle[0].snapshot(), protocol=5) == before
    model = result.analyst["paired"]["analyst"]["model"]
    for role, i in (("forecast", 0), ("reference", 1)):
        law = args[role]
        origin = law.original
        mass = {q: F() for q in product(range(2), range(3), range(2))}
        evidence = F()
        action = next(
            j for j, p in enumerate(origin.products) if p.product_id == "B_OH0"
        )
        for z, k, u, g, x, y in product(
            range(2), range(3), range(2), range(2), range(2), range(2)
        ):
            joint = (
                origin.initial_joint[z][k][u][g]
                * origin.ordinary_likelihood[k][u][action]
                * origin.boundary_kernel[z][k][u][g][action][x][1]
                * law.transport.probabilities[z][k][u][g][action][x][1][y]
            )
            mass[y, k, g] += joint
            evidence += joint
        expected = tuple(
            tuple(tuple(mass[y, k, g] / evidence for g in range(2)) for k in range(3))
            for y in range(2)
        )
        assert result.analyst["precategory_joints"][i] == expected
        assert model[role]["initial_joint"] == _encoded(expected)
        assert result.analyst["original_history_evidence"][i] == evidence
        selected_evidence = F()
        numerator = F()
        root = F()
        for y, k, g, yn, s, a, d in product(
            range(2), range(3), range(2), range(2), range(3), range(3), range(3)
        ):
            o = args["programme"].support.observations.index(
                args["programme"].support.type_observation[k]
            )
            policy = args["programme"].policy
            path = (
                expected[y][k][g]
                * policy.own_program[o][g][-1]
                * law.transition_signal[y][k][g][yn][s]
                * policy.interline_program[o][s][g][a]
                * law.terminal_probabilities[1][yn][k][g][a][d]
            )
            account = law.terminal_accounts[1][yn][k][g][a][d]
            zv = account.payoff.net_value()
            if path:
                root += path * zv
            if (o, s) == (0, 1):
                selected_evidence += path
                if path:
                    numerator += path * zv
        observed = result.analyst["paired"]["analyst"][role]
        assert observed["root_material"] == _encoded(root)
        assert observed["selected_evidence"] == _encoded(selected_evidence)
        assert observed["selected_material"] == _encoded(numerator / selected_evidence)
    assert (
        result.analyst["precategory_joints"][0]
        != result.analyst["precategory_joints"][1]
    )
    assert result.analyst["original_conditioning_calls"] == 2


def test_only_two_original_forwards_and_one_pair(bundle, monkeypatch):
    args = arguments(bundle)
    calls = []
    parents = subject.bind_full_path_transport
    paired = subject.evaluate_paired_material

    def parent(*a, **k):
        calls.append("original")
        return parents(*a, **k)

    def pair(*a, **k):
        calls.append("pair")
        return paired(*a, **k)

    monkeypatch.setattr(subject, "bind_full_path_transport", parent)
    monkeypatch.setattr(subject, "evaluate_paired_material", pair)
    subject.bind_paired_history(**args)
    assert calls == ["original", "original", "pair"]


def test_zero_original_reference_never_fills_or_evaluates_pair(bundle, monkeypatch):
    args = arguments(bundle)
    data = bundle[-1]
    origin = replace(
        data["original_laws"][2],
        conditioning_context=args["reference"].original.conditioning_context,
    )
    args["reference"] = replace(args["reference"], original=origin)

    def forbidden(*a, **k):
        raise AssertionError("pair must not evaluate")

    monkeypatch.setattr(subject, "evaluate_paired_material", forbidden)
    result = subject.bind_paired_history(**args)
    assert result.analyst["precategory_joints"][0] is not None
    assert result.analyst["precategory_joints"][1] is result.analyst["paired"] is None
    assert (
        result.analyst["original_history_evidence"][1] == 0
        and not result.analyst["paired_evaluator_called"]
    )


def test_tiny_unknown_and_owned_source_export(bundle):
    args = arguments(bundle)
    future = bundle[-1]["future"][2]
    args["reference"] = replace(args["reference"], **future)
    result = subject.bind_paired_history(**args)
    record = subject.paired_history_record(result)
    paired = record["analyst"]["paired"]["analyst"]
    assert (
        paired["reference"]["root_material"]
        is paired["reference"]["selected_material"]
        is None
    )
    assert paired["forecast"]["root_material"] is not None
    record["analyst"]["inputs"].clear()
    assert subject.paired_history_record(result)["analyst"]["inputs"]
    view = subject.paired_history_view(result)
    text = str(view)
    for forbidden in (
        "precategory_joints",
        "terminal_accounts",
        "private_marks",
        "actual_root",
        "scientific_budget",
    ):
        assert forbidden not in text
    assert (
        result.analyst["inputs"]["early_source"]["periods"]
        and len(result.analyst["inputs"]["later_source"]["periods"]) == 3
    )


@pytest.mark.parametrize(
    "kind",
    (
        "category",
        "signal",
        "baseline",
        "later_programme",
        "market",
        "carrier",
        "axes",
        "policy",
        "future_row",
        "future_zero_account_unit",
        "original_unused",
        "absence",
        "missing_policy",
        "source_menu",
        "postcategory",
        "limit",
    ),
)
def test_all_invalid_support_before_probability_forward(bundle, monkeypatch, kind):
    args = arguments(bundle)
    p = args["programme"]
    q = args["reference"]
    if kind == "category":
        args["observation"] = replace(
            args["observation"], category_delivery_id="missing"
        )
    elif kind == "signal":
        args["observation"] = replace(args["observation"], signal_delivery_id="missing")
    elif kind == "baseline":
        args["baseline"] = args["own"].receipt.source_journal
    elif kind == "later_programme":
        full = copy.deepcopy(args["later_source"])
        full["programme"]["definition"] = "changed"
        args["later_source"] = full
    elif kind == "market":
        args["programme"] = replace(p, support=replace(p.support, market="other"))
    elif kind == "carrier":
        args["programme"] = replace(p, support=replace(p.support, original_carrier="Z"))
    elif kind == "axes":
        args["reference"] = replace(
            q,
            transport=replace(q.transport, future_states=("different0", "different1")),
        )
    elif kind == "policy":
        args["programme"] = replace(
            p,
            policy=replace(
                p.policy,
                definition="changed",
                own_program=(),
            ),
        )
    elif kind == "future_row":
        rows = copy.deepcopy(q.terminal_probabilities)
        rows = list(rows)
        rows[0] = rows[1][:-1]
        args["reference"] = replace(q, terminal_probabilities=rows)
    elif kind == "future_zero_account_unit":

        def account(stage, y, k, g, a, d):
            x = q.terminal_accounts[stage][y][k][g][a][d]
            return replace(x, unit="wrong") if k == 2 else x

        args["reference"] = replace(
            q, terminal_accounts=nested((2, 2, 3, 2, 3, 3), account)
        )
    elif kind == "original_unused":
        args["reference"] = replace(q, transport=replace(q.transport, probabilities=()))
    elif kind == "absence":
        args["programme"] = replace(p, absent_signal="not-in-support")
    elif kind == "missing_policy":
        own = list(p.policy.own_program)
        own[2] = tuple((None,) * 3 for _ in range(2))
        policy = replace(p.policy, own_program=tuple(own))
        args["programme"] = replace(
            p,
            policy=policy,
            source_programme=replace(
                p.source_programme, own_program=policy.own_program
            ),
        )
    elif kind == "source_menu":
        args["programme"] = replace(
            p, source_programme=replace(p.source_programme, own_menus=())
        )
    elif kind == "postcategory":
        args["programme"] = replace(
            p,
            source_programme=replace(
                p.source_programme, conditioning_view={"postcategory": True}
            ),
        )
    elif kind == "limit":
        args["row_limit"] = 1

    def forbidden(*a, **k):
        raise AssertionError("probability forward before validation")

    monkeypatch.setattr(subject, "bind_full_path_transport", forbidden)
    monkeypatch.setattr(subject, "evaluate_paired_material", forbidden)
    with pytest.raises((ValueError, TypeError, subject.PairedMaterialLimit)):
        subject.bind_paired_history(**args)


def test_second_original_failure_no_partial_or_input_mutation(bundle, monkeypatch):
    args = arguments(bundle)
    before = pickle.dumps(args, protocol=5)
    parent = subject.bind_full_path_transport
    count = 0

    def fail(*a, **k):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("new injected second-law failure")
        return parent(*a, **k)

    monkeypatch.setattr(subject, "bind_full_path_transport", fail)
    with pytest.raises(RuntimeError, match="second-law"):
        subject.bind_paired_history(**args)
    assert count == 2 and pickle.dumps(args, protocol=5) == before


def test_tiny_positive_original_evidence_not_rounded_to_zero(bundle):
    args = arguments(bundle)
    ref = args["reference"]
    origin = ref.original
    tiny = F(1, 10**1003)
    b = next(i for i, p in enumerate(origin.products) if p.product_id == "B_OH0")
    likelihood = nested(
        (3, 2, 3),
        lambda k, u, a: tiny if a == b else F(1, 8) if a == 0 else F(7, 8) - tiny,
    )
    args["reference"] = replace(
        ref, original=replace(origin, ordinary_likelihood=likelihood)
    )
    result = subject.bind_paired_history(**args)
    assert 0 < result.analyst["original_history_evidence"][1] < F(1, 10**1000)
    assert (
        result.analyst["precategory_joints"][1] is not None
        and result.analyst["paired"] is not None
    )


def test_source_only_adapter_exact_original_AST_prefix():
    import ast
    from pathlib import Path

    from airline_market.sim import transported_stage_signal as frozen

    original = next(
        n
        for n in ast.parse(Path(frozen.__file__).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "bind_transported_stage_signal"
    )
    cut = next(
        i
        for i, n in enumerate(original.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "transported" for t in n.targets)
    )
    new = next(
        n
        for n in ast.parse(Path(subject.__file__).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_validate_source_pair"
    )
    assert ast.dump(ast.Module(body=original.body[:cut], type_ignores=[])) == ast.dump(
        ast.Module(body=new.body[:-1], type_ignores=[])
    )


def test_future_only_preflight_omits_only_absent_initial_joint_checks():
    import ast
    from pathlib import Path

    original = next(
        n
        for n in ast.parse(Path(pm.__file__).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_validate"
    )
    original.name = "_preflight_templates"
    removed = 0
    for node in ast.walk(original):
        if (
            isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "law"
        ):
            keep = []
            for n in node.body:
                if (
                    isinstance(n, ast.Expr)
                    and ast.unparse(n).startswith("_array(law.initial_joint,")
                ) or (
                    isinstance(n, ast.If)
                    and "complete initial joint must sum EXACTLY one" in ast.unparse(n)
                ):
                    removed += 1
                else:
                    keep.append(n)
            node.body = keep
    new = next(
        n
        for n in ast.parse(Path(subject.__file__).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "_preflight_templates"
    )
    assert removed == 2 and ast.dump(original) == ast.dump(new)
