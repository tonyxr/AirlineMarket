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
from airline_market.sim import full_path_utility as subject
from airline_market.sim import paired_history_binding as parent
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
from airline_market.sim.shared_mark_attempt import AttemptOffer
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.sim.transported_stage_signal import (
    StageSignalObservation,
    StageSignalProgramme,
)
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

LIMIT = 2_000_000_000
UNIT = "declared-fare-unit110974"


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
    start = datetime(2027, 7, 27, tzinfo=timezone.utc)
    clock = UtcBookingClock(tuple(start + timedelta(hours=12 * i) for i in range(4)))
    departures = tuple(
        (leg.id, start + timedelta(hours=84 if leg.id == "B_OH0" else 120))
        for leg in compiled.spec.legs
    )
    scenario = replace(
        compile_dated_demand(compiled, clock, departures),
        physical_shocks=(
            B1DatedShock(
                "new_signal_exit110974", "rival_exit", "B_OH0", 3, rival_carrier="B"
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
                "unit110974",
            )
            for i in range(2)
        ),
        "causal_deliveries": (
            DeclaredInputDelivery(
                key,
                1,
                "own",
                "category110974",
                "category",
                "view-a",
                "declared category110974",
                TIMING,
            ),
            DeclaredInputDelivery(
                key,
                1,
                "interline",
                "signal110974",
                "stage_signal",
                "notice-high",
                "declared signal110974",
                TIMING,
            ),
        ),
        "causal_definition": "NEW transported joint and received stage signal110974",
        "causal_unobserved_definition": "caller schedule not authenticated reception, knowledge, physical law or preference",
        "causal_row_limit": 2000000,
    }
    registered = register(compiled, scenario, clock, departures, source_kwargs)
    tape = ExogenousTape.generate(
        compiled, 31, registry=StreamRegistry(110974, "unit/V9_full_path_utility")
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
            {j: 225 if i == 0 else 345 for j in compiled.products_of[i]}
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
        OriginalOffer(p.id, 225.0 if p.carrier == "A" else 345.0, True)
        for p in products
    )
    a = len(offers) + 1
    ib = next(i for (i, p) in enumerate(offers) if p.product_id == "B_OH0")
    ia = next(i for (i, p) in enumerate(offers) if p.product_id.startswith("A_"))
    pk = (F(2, 7), F(5, 7), F())
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
        "material-person110974",
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
        "NEW material law110974",
        "declared110974",
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
        "full path110974",
        "declared110974",
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
        "NEW material programme110974",
        "declared110974",
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
        definition="NEW separate reference original law110974",
        source_ref="declared-reference110974",
    )
    reference_kernel = replace(
        kernel,
        probabilities=nested(
            (2, 3, 2, 2, a, 2, 2, 2),
            lambda z, k, u, g, j, x, c, y: (
                F(1 + z + 2 * u + g + x, 8) if y else 1 - F(1 + z + 2 * u + g + x, 8)
            ),
        ),
        definition="NEW reference full path110974",
    )
    zero = replace(
        reference,
        ordinary_likelihood=nested(
            (3, 2, a),
            lambda k, u, j: F() if j == ib else F(1, 8) if j == ia else F(7, 8),
        ),
        definition="NEW zero original purchase evidence law110974",
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
        "ONE explicit fixed policy110974; not optimized native preferences",
        "declared-policy110974",
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
                    "explicit declared " + n + "110974" for n in ("B", "P", "W", "C")
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
            "definition": "NEW full future declaration110974",
            "source_ref": "declared110974",
        }

    data = {
        "original_laws": (law, reference, zero),
        "kernels": (kernel, reference_kernel, reference_kernel),
        "programme": programme,
        "observation": StageSignalObservation(
            "category110974",
            "category",
            "view-a",
            "declared category110974",
            "signal110974",
            "stage_signal",
            "notice-high",
            "declared signal110974",
            "NEW source-association declaration110974",
        ),
        "support": support,
        "policy": policy,
        "future": (future(0, F()), future(1, F()), future(1, F(1, 10**1400))),
        "source_plan": {
            "spec": compiled.spec,
            "scenario": scenario.record(),
            "kwargs": kwargs,
        },
        "registration_scope": "local before NEW tape/engine, not external preregistration/time/native truth",
    }
    return data


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
    programme = parent.PairedHistoryProgramme(
        data["support"],
        data["policy"],
        source_programme,
        "no-signal",
        "NEW typed full-history bridge110974",
        "declared110974",
    )
    originals = tuple(
        replace(
            law, conditioning_context=choice_boundary_conditioning_context(ordinary)
        )
        for law in data["original_laws"]
    )

    def law(i, future, name):
        return parent.OriginalHistoryMaterialLaw(
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
            "SAME actual history110974",
            "declared110974",
        ),
        "baseline": baseline,
        "own": own,
        "interline": interline,
        "later_periods": periods,
        "later_source": full,
        "observation": data["observation"],
        "row_limit": LIMIT,
    }


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
    return data


def utility_law(support, name, role):
    def leaf(stage, y, k, g, action, outcome):
        outside = tm.TerminalMaterialAccount(
            ContingentPayoff(
                F(21 + 5 * k + y), F(-4 if outcome else 2), F(1 + g, 7), F(3 - y)
            ),
            tuple("explicit110974:" + n for n in ("B0", "P0", "W0", "C0")),
            UNIT,
            True,
            True,
            "declared full same-path outside material110974",
        )
        return subject.TerminalUtilityAccount(
            F((-1) ** (action + stage + g) * (5 + y + k + g), 13) + F(role, 17),
            "declared E110974; not decoded native Gumbel",
            True,
            outside,
            F(3 + y + k - 2 * g, 11) + F(role, 19),
            "declared E0-110974",
            True,
            "explicit all-component followup110974",
        )

    return subject.FullPathUtilityLaw(
        name,
        support.physical_states,
        support.latent_types,
        support.private_marks,
        (*support.products, None),
        support.outcomes,
        nested((2, 2, 3, 2, 3, 3), leaf),
        "complete terminal utility110974",
        "unit-declared110974",
    )


@pytest.fixture(scope="module")
def new_bundle():
    return fresh_source(register)


@pytest.fixture(scope="module")
def new_parent(new_bundle):
    return parent.bind_paired_history(**arguments(new_bundle))


def options():
    return {
        "alpha": F(1, 83),
        "material_unit": UNIT,
        "utility_unit": "declared-stage-utility110974",
        "material_additional_nonoverlap": "FOUR material components exclude all E110974",
        "utility_origin_and_incidence": "explicit terminal incidence, no native interpretation110974",
        "definition": "NEW declared fixed-policy full-path expectation110974",
        "reference_scope": subject.REFERENCE_SCOPE,
        "row_limit": LIMIT,
    }


def evaluate(new_parent, new_bundle, **updates):
    return subject.evaluate_full_path_utility(
        new_parent, *new_bundle[-1]["utilities"], **(options() | updates)
    )


def transform(law, fn):
    return replace(
        law,
        accounts=nested(
            (2, 2, 3, 2, 3, 3),
            lambda *c: fn(law.accounts[c[0]][c[1]][c[2]][c[3]][c[4]][c[5]], c),
        ),
    )


def test_full_exact_reference_scope_and_native_identity(new_parent, new_bundle):
    engine = new_bundle[0]
    before = pickle.dumps(engine.snapshot(), protocol=5)
    report = evaluate(new_parent, new_bundle)
    assert pickle.dumps(engine.snapshot(), protocol=5) == before
    assert subject.full_path_utility_view(report) == parent.paired_history_view(
        new_parent
    )
    assert report.analyst["parent_producer_or_probability_forward_called"] is False
    assert report.analyst["lifecycle_CS"] is None
    for i, role in enumerate(("forecast", "reference")):
        out = report.analyst["expectations"][role]
        saved = subject._decode(new_parent.analyst["paired"])["analyst"][role]
        assert len(out["rows"]) == len(saved["rows"]) == 756
        for scope in ("root", "selected"):
            selected = [
                r
                for r in saved["rows"]
                if scope == "root"
                or r["stage"] == "interline"
                and r["coordinate"][3] == 0
                and r["coordinate"][5] == 1
            ]
            evidence = sum((r["mass"] for r in selected), F())
            material = F()
            outside_material = F()
            experience = F()
            outside_experience = F()
            for row in selected:
                c = row["coordinate"]
                stage = int(row["stage"] == "interline")
                account = new_bundle[-1]["utilities"][i].accounts[stage][c[4]][c[1]][
                    c[2]
                ][c[6]][c[7]]
                z = row["material_if_closed"]
                z0 = account.outside_material.payoff.net_value()
                material += row["mass"] * z
                outside_material += row["mass"] * z0
                experience += row["mass"] * (z / F(83) + account.additional_utility)
                outside_experience += row["mass"] * (
                    z0 / F(83) + account.outside_additional_utility
                )
            fields = out["summaries"][scope]["fields"]
            assert fields["material"]["mean"] == material / evidence
            assert fields["outside_material"]["mean"] == outside_material / evidence
            assert fields["experienced_candidate"]["mean"] == experience / evidence
            assert (
                fields["normalized_candidate"]["mean"]
                == (experience - outside_experience) / evidence
            )
            assert out["summaries"][scope]["evidence"] == evidence
        assert (
            out["summaries"]["root"]["fields"]["material"]["mean"]
            == saved["root_material"]
        )
        assert (
            out["summaries"]["selected"]["fields"]["material"]["mean"]
            == saved["selected_material"]
        )
    assert report.analyst["source"]["analyst"]["inputs"]["later_source"]


def test_parent_producers_never_called(new_parent, new_bundle, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("parent producer replay")

    monkeypatch.setattr(parent, "bind_full_path_transport", forbidden)
    monkeypatch.setattr(parent, "evaluate_paired_material", forbidden)
    assert evaluate(new_parent, new_bundle).analyst["expectations"] is not None


@pytest.mark.parametrize("tiny", [False, True])
def test_positive_unknown_utility_is_complete_null(new_parent, new_bundle, tiny):
    args = arguments(new_bundle)
    if tiny:
        args["reference"] = parent.OriginalHistoryMaterialLaw(
            "reference",
            args["reference"].original,
            args["reference"].transport,
            **new_bundle[-1]["future"][2],
        )
        supplied = parent.bind_paired_history(**args)
        target = 2
    else:
        supplied = new_parent
        target = 0
    f, r = new_bundle[-1]["utilities"]
    r = transform(
        r,
        lambda acc, c: (
            replace(acc, additional_utility=None) if c[-1] == target else acc
        ),
    )
    result = subject.evaluate_full_path_utility(supplied, f, r, **options())
    for scope in ("root", "selected"):
        fields = result.analyst["expectations"]["reference"]["summaries"][scope][
            "fields"
        ]
        assert fields["material"]["mean"] is not None
        assert fields["experienced_candidate"]["mean"] is None
        assert fields["normalized_candidate"]["mean"] is None
        assert fields["experienced_candidate"]["unknown_mass"] > 0
        if tiny:
            assert float(fields["experienced_candidate"]["unknown_mass"]) == 0


def test_zero_unknown_rows_retained_with_algebraic_zero(new_parent, new_bundle):
    f, r = new_bundle[-1]["utilities"]
    r = transform(
        r,
        lambda acc, c: (
            replace(acc, additional_utility=None, outside_additional_utility=None)
            if c[2] == 2
            else acc
        ),
    )
    result = subject.evaluate_full_path_utility(new_parent, f, r, **options())
    base = evaluate(new_parent, new_bundle)
    assert (
        result.analyst["expectations"]["reference"]["summaries"]
        == base.analyst["expectations"]["reference"]["summaries"]
    )
    rows = [
        x
        for x in result.analyst["expectations"]["reference"]["rows"]
        if x["coordinate"][1] == 2
    ]
    assert rows and all(
        x["values"]["normalized_candidate"] is None
        and x["contributions"]["normalized_candidate"] == 0
        for x in rows
    )


def test_common_terminal_translation(new_parent, new_bundle):
    f, r = new_bundle[-1]["utilities"]
    shifted = tuple(
        transform(
            law,
            lambda acc, c: replace(
                acc,
                additional_utility=acc.additional_utility + F(17, 8),
                outside_additional_utility=acc.outside_additional_utility + F(17, 8),
            ),
        )
        for law in (f, r)
    )
    base = evaluate(new_parent, new_bundle)
    changed = subject.evaluate_full_path_utility(new_parent, *shifted, **options())
    for role, scope in product(("forecast", "reference"), ("root", "selected")):
        a = base.analyst["expectations"][role]["summaries"][scope]["fields"]
        b = changed.analyst["expectations"][role]["summaries"][scope]["fields"]
        assert b["normalized_candidate"]["mean"] == a["normalized_candidate"]["mean"]
        assert b["experienced_candidate"]["mean"] - a["experienced_candidate"][
            "mean"
        ] == F(17, 8)


@pytest.mark.parametrize(
    "field", ["utility_tail_closed", "outside_utility_tail_closed"]
)
def test_open_tail_is_not_final_utility(new_parent, new_bundle, field):
    f, r = new_bundle[-1]["utilities"]
    r = transform(r, lambda acc, c: replace(acc, **{field: False}))
    result = subject.evaluate_full_path_utility(new_parent, f, r, **options())
    for scope in ("root", "selected"):
        a = result.analyst["expectations"]["reference"]["summaries"][scope]["fields"]
        assert a["normalized_candidate"]["mean"] is None
        assert a["material"]["mean"] is not None


def test_upstream_zero_stays_null_complete_declarations_retained(new_bundle):
    args = arguments(new_bundle)
    args["reference"] = replace(
        args["reference"],
        original=replace(
            new_bundle[-1]["original_laws"][2],
            conditioning_context=args["reference"].original.conditioning_context,
        ),
    )
    result = parent.bind_paired_history(**args)
    assert result.analyst["paired"] is None
    out = evaluate(result, new_bundle)
    assert out.analyst["expectations"] is None
    assert len(out.analyst["utility_declarations"]) == 2
    assert out.analyst["source"]["analyst"]["paired_evaluator_called"] is False


@pytest.mark.parametrize(
    "update",
    [
        {"alpha": 0},
        {"alpha": 1.0},
        {"alpha": F(-1)},
        {"material_unit": "dollars"},
        {"utility_unit": UNIT},
        {"material_additional_nonoverlap": ""},
        {"utility_origin_and_incidence": ""},
        {"reference_scope": "outside-intervention"},
        {"row_limit": True},
    ],
)
def test_explicit_definition_refusals(new_parent, new_bundle, update):
    with pytest.raises((TypeError, ValueError)):
        evaluate(new_parent, new_bundle, **update)


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "other-law"),
        ("physical_states", ("y1", "y0")),
        ("accounts", ()),
        ("source_ref", ""),
    ],
)
def test_full_named_declaration_refusals(new_parent, new_bundle, field, value):
    f, r = new_bundle[-1]["utilities"]
    with pytest.raises((TypeError, ValueError)):
        subject.evaluate_full_path_utility(
            new_parent, f, replace(r, **{field: value}), **options()
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("additional_utility", 0.0),
        ("utility_tail_closed", 1),
        ("additional_utility_source", ""),
        ("outside_utility_tail_closed", None),
    ],
)
def test_no_coercion_or_inferred_closure(new_bundle, field, value):
    account = new_bundle[-1]["utilities"][0].accounts[0][0][0][0][0][0]
    with pytest.raises((TypeError, ValueError)):
        replace(account, **{field: value})


def test_bound_precedes_utility_arithmetic(new_parent, new_bundle, monkeypatch):
    def forbidden(*args):
        raise AssertionError("arithmetic before structural bound")

    monkeypatch.setattr(subject, "_saved_rows", forbidden)
    with pytest.raises(subject.FullPathUtilityLimit):
        evaluate(new_parent, new_bundle, row_limit=1)


@pytest.mark.parametrize(
    "fault", ["mass", "factor", "account", "selected", "evidence", "joint"]
)
def test_saved_source_consistency_refusals(new_parent, new_bundle, fault):
    data = copy.deepcopy(subject._decode(subject._encoded(new_parent.analyst)))
    p = data["paired"]["analyst"]
    row = p["reference"]["rows"][0]
    if fault == "mass":
        row["mass"] += F(1)
    elif fault == "factor":
        row["factors"] = (F(1), *row["factors"][1:])
    elif fault == "account":
        row["material_if_closed"] = F(99)
    elif fault == "selected":
        p["reference"]["selected_evidence"] += F(1)
    elif fault == "evidence":
        data["original_history_evidence"] = (F(1), data["original_history_evidence"][1])
    else:
        data["precategory_joints"] = (None, data["precategory_joints"][1])
    bad = replace(new_parent, analyst=data)
    with pytest.raises(ValueError):
        evaluate(bad, new_bundle)


def test_ownership_exports_and_pure_failure(new_parent, new_bundle, monkeypatch):
    before = subject._encoded(new_parent)
    result = evaluate(new_parent, new_bundle)
    record = subject.full_path_utility_record(result)
    record["analyst"]["utility_declarations"][0]["name"] = "changed export only"
    assert len(result.analyst["utility_declarations"]) == 2
    with pytest.raises(TypeError):
        result.analyst["alpha"] = F(2)

    def fail(*a):
        raise RuntimeError("NEW injected projection failure")

    monkeypatch.setattr(subject, "_material", fail)
    with pytest.raises(RuntimeError, match="NEW injected"):
        evaluate(new_parent, new_bundle)
    assert subject._encoded(new_parent) == before


def test_zero_selected_evidence_has_no_conditional_fill(new_bundle):
    args = arguments(new_bundle)
    r = args["reference"]
    transition = nested(
        (2, 3, 2, 2, 3), lambda y, k, g, yn, s: F(s == 2) * (F(2, 5) if yn else F(3, 5))
    )
    args["reference"] = replace(r, transition_signal=transition)
    saved = parent.bind_paired_history(**args)
    out = evaluate(saved, new_bundle).analyst["expectations"]["reference"]
    assert (
        out["summaries"]["root"]["fields"]["experienced_candidate"]["mean"] is not None
    )
    assert out["summaries"]["selected"]["evidence"] == 0
    assert all(
        x["mean"] is None for x in out["summaries"]["selected"]["fields"].values()
    )
    assert any(r["selected"] for r in out["rows"])


@pytest.mark.parametrize("which", ["financial", "material"])
def test_outside_independent_open_material_accounts(new_parent, new_bundle, which):
    f, r = new_bundle[-1]["utilities"]

    def change(acc, coordinate):
        old = acc.outside_material
        if which == "financial":
            old = replace(
                old,
                payoff=replace(old.payoff, net_ticket_payment=None),
                financial_tail_closed=False,
            )
        else:
            old = replace(old, material_tail_closed=False)
        return replace(acc, outside_material=old)

    changed = subject.evaluate_full_path_utility(
        new_parent, f, transform(r, change), **options()
    )
    for scope in ("root", "selected"):
        fields = changed.analyst["expectations"]["reference"]["summaries"][scope][
            "fields"
        ]
        assert fields["material"]["mean"] is not None
        assert fields["outside_material"]["mean"] is None
        assert fields["normalized_candidate"]["mean"] is None


def test_exact_huge_signed_utility_no_float_conversion(new_parent, new_bundle):
    f, r = new_bundle[-1]["utilities"]
    huge = F(-(10**700), 37)
    changed = subject.evaluate_full_path_utility(
        new_parent,
        f,
        transform(r, lambda acc, c: replace(acc, additional_utility=huge)),
        **options(),
    )
    for scope in ("root", "selected"):
        fields = changed.analyst["expectations"]["reference"]["summaries"][scope][
            "fields"
        ]
        assert (
            fields["experienced_candidate"]["mean"]
            == fields["noise_excluded"]["mean"] + huge
        )


def test_complete_zero_row_outside_unit_is_checked(new_parent, new_bundle):
    f, r = new_bundle[-1]["utilities"]
    r = transform(
        r,
        lambda acc, c: (
            replace(
                acc,
                outside_material=replace(
                    acc.outside_material, unit="unsupported other unit"
                ),
            )
            if c[2] == 2
            else acc
        ),
    )
    with pytest.raises(ValueError, match="outside material unit"):
        subject.evaluate_full_path_utility(new_parent, f, r, **options())
