"""A.10/A.5/A.14/B0/V9: NEW typed stage-signal synthetic references."""

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
from airline_market.sim import transported_stage_signal as subject
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
from airline_market.sim.pretype_observation import _same, pretype_history_view
from airline_market.sim.received_information import DisclosureField, _units
from airline_market.sim.shared_mark_attempt import AttemptOffer, _encoded
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

LIMIT = 500_000_000


def check(value, message):
    assert value, message


def nested(shape, fn, prefix=()):
    return (
        fn(*prefix)
        if not shape
        else tuple(nested(shape[1:], fn, prefix + (i,)) for i in range(shape[0]))
    )


def source():
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
    start = datetime(2027, 6, 8, tzinfo=timezone.utc)
    clock = UtcBookingClock(tuple(start + timedelta(hours=12 * i) for i in range(4)))
    departures = tuple(
        (leg.id, start + timedelta(hours=84 if leg.id == "B_OH0" else 120))
        for leg in compiled.spec.legs
    )
    scenario = replace(
        compile_dated_demand(compiled, clock, departures),
        physical_shocks=(
            B1DatedShock(
                "new_signal_exit100974", "rival_exit", "B_OH0", 3, rival_carrier="B"
            ),
        ),
    )
    tape = ExogenousTape.generate(
        compiled,
        31,
        registry=StreamRegistry(100974, "unit/V9_transported_stage_signal"),
    )
    target = compiled.product_index["B_OH0"]
    market = int(compiled.product_market[target])
    key = (0, 0, market)
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
    engine = NativeCausalInformationMarketEngine(
        scenario,
        dated_rebooking=DatedRebookingPolicy(72, 20, 100, interline=True),
        causal_fields=(
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
        causal_known_types=tuple(
            DeclaredKnownType(
                (0, i, market),
                None,
                "no supplied label; not ignorance or sampled class",
                "unit100974",
            )
            for i in range(2)
        ),
        causal_deliveries=(
            DeclaredInputDelivery(
                key,
                1,
                "own",
                "category100974",
                "category",
                "copper",
                "declared category100974",
                TIMING,
            ),
            DeclaredInputDelivery(
                key,
                1,
                "interline",
                "signal100974",
                "stage_signal",
                "clear",
                "declared signal100974",
                TIMING,
            ),
        ),
        causal_definition="NEW transported joint and received stage signal100974",
        causal_unobserved_definition="caller schedule not authenticated reception, knowledge, physical law or preference",
    )
    engine.reset(tape=tape)
    actions = {
        c.code: PriceAction(
            {j: 280 if i == 0 else 410 for j in compiled.products_of[i]}
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
    )


def parent_law(bundle, scale):
    row = bundle[1]
    offers = tuple(
        OriginalOffer(
            r["product_id"], float.fromhex(r["fare"]["float_hex"]), r["offered"]
        )
        for r in row.computation_input["offers"]
    )
    a = len(offers) + 1
    target = next(i for (i, r) in enumerate(offers) if r.product_id == "B_OH0")
    other = next(i for (i, r) in enumerate(offers) if r.product_id.startswith("A_"))
    pk = (F(3, 8), F(5, 8), F())
    pz = (F(1, 4), F(2, 3), F(1, 2))
    ug = (
        ((F(1, 8), F(3, 8)), (F(1, 4), F(1, 4))),
        ((F(1, 2), F()), (F(1, 6), F(1, 3))),
        ((F(1, 4), F(1, 4)), (F(1, 4), F(1, 4))),
    )
    prior = nested(
        (2, 3, 2, 2),
        lambda z, k, u, g: pk[k] * (pz[k] if z else 1 - pz[k]) * ug[k][u][g],
    )
    choices = nested(
        (3, 2, a),
        lambda k, u, j: (
            F(2 + k + u, 7)
            if j == target
            else F(1, 7)
            if j == other
            else 1 - F(3 + k + u, 7)
        ),
    )

    def boundary(z, k, u, g, j, x, c):
        q = scale * F(2 + z + u + g, 9) if j < a - 1 else F()
        x1 = F(2 + k + c, 6)
        return (q if c else 1 - q) * (x1 if x else 1 - x1)

    return BookingLossLaw(
        "new-transport-signal-person100974",
        ("z0", "z1"),
        ("x0", "x1"),
        ("k0", "k1", "zero-k"),
        ("u0", "u1"),
        ("g0", "g1"),
        offers,
        prior,
        choices,
        nested((2, 3, 2, 2, a, 2, 2), boundary),
        choice_boundary_conditioning_context(row),
        "before-original-choice-and-boundary",
        "NEW complete supplied I0; actual original action/cancel conditions once",
        "distinct ordinary U declaration",
        "persistent later shared G declaration",
        "joint boundary X/cancellation declaration",
        "NEW retrospective finite law100974",
        "declared100974",
    )


def transport_kernel(law):
    return FullPathKernel(
        law.initial_states,
        law.boundary_states,
        law.latent_types,
        law.ordinary_marks,
        law.attempt_marks,
        tuple(p.product_id for p in law.products) + (None,),
        ("y0", "y1"),
        nested(
            (2, 3, 2, 2, len(law.products) + 1, 2, 2, 2),
            lambda z, k, u, g, a, x, c, y: (
                F(3 + z + 2 * u + g + x, 10) if y else 1 - F(3 + z + 2 * u + g + x, 10)
            ),
        ),
        "after-original-choice-boundary-before-later-observation",
        "NEW full-path R, before category and stage signal",
        "declared100974",
    )


def template(bundle):
    (_, _, _, _, baseline, own, interline, _, _) = bundle

    def menu(row):
        return tuple(
            AttemptOffer(r["product_id"], r["fare"], r["offered"], False, False)
            for r in row.computation_input["offers"]
        )

    (own_menu, other_menu) = (menu(own), menu(interline))
    p = len(own_menu) + 1
    target = next(
        i
        for (i, r) in enumerate(other_menu)
        if r.offered and r.product_id.startswith("A_")
    )
    check(not any(r.offered for r in own_menu), "NEW native own menu is not empty")

    def signal(y, k, g, yn, s):
        clear = F(1 + 2 * y + k + g, 8)
        return F() if yn != y or s == 0 else clear if s == 1 else 1 - clear

    def action(o, s, g, a):
        buy = ((F(1, 2), F(1, 2)), (F(2, 3), F(1, 4)), (F(1, 5), F(2, 3)))[s][g]
        return buy if a == target else 1 - buy if a == p - 1 else F()

    return {
        "physical_states": ("y0", "y1"),
        "latent_types": ("k0", "k1", "zero-k"),
        "private_marks": ("g0", "g1"),
        "observations": ("copper", "silver", "unmapped"),
        "type_observation": ("copper", "silver", "copper"),
        "signals": (None, "clear", "fog"),
        "own_menus": (own_menu,) * 3,
        "interline_menus": ((other_menu,) * 3,) * 3,
        "product_catalogue_disclosed": False,
        "interline_enabled": True,
        "own_program": nested((3, 2, p), lambda o, g, a: F(a == p - 1)),
        "next_state_signal": nested((2, 3, 2, 2, 3), signal),
        "interline_program": nested((3, 3, 2, p), action),
        "conditioning_view": pretype_history_view(baseline, own.analyst_original_key),
        "conditioning_definition": "Transported full joint AFTER original choice/cancel ONCE, BEFORE category/source signal; complete supplied earlier history",
        "observation_definition": "Explicit coarse m(K), not actual native private class or awareness",
        "private_mark_interpretation": "SAME declared G through own outside and interline signal/action, not fresh native marks",
        "definition": "NEW retrospectively declared conditional physical signal and randomized programme",
        "source_ref": "declared100974",
    }


@pytest.fixture(scope="module")
def bundle():
    return source()


def arguments(bundle, scale=F(1)):
    _, ordinary, early_periods, early, baseline, own, interline, full, periods = bundle
    law = parent_law(bundle, scale)
    return {
        "law": law,
        "kernel": transport_kernel(law),
        "ordinary_input": ordinary,
        "early_periods": early_periods,
        "early_source": early,
        "association": ChoiceBoundaryAssociation(
            law.original_person_label, "NEW original root association", "unit100974"
        ),
        "baseline": baseline,
        "own": own,
        "interline": interline,
        "later_periods": periods,
        "later_source": full,
        "programme": subject.StageSignalProgramme(**template(bundle)),
        "observation": subject.StageSignalObservation(
            "category100974",
            "category",
            "copper",
            "declared category100974",
            "signal100974",
            "stage_signal",
            "clear",
            "declared signal100974",
            "NEW category then conditional signal, not authenticated knowledge",
        ),
        "row_limit": LIMIT,
    }


def reference(args):
    law, kernel, pr = (args[n] for n in ("law", "kernel", "programme"))
    a = next(i for i, p in enumerate(law.products) if p.product_id == "B_OH0")
    raw = [[[F() for _ in range(2)] for _ in range(3)] for _ in range(2)]
    for z, k, u, g, x, y in product(
        range(2), range(3), range(2), range(2), range(2), range(2)
    ):
        raw[y][k][g] += (
            law.initial_joint[z][k][u][g]
            * law.ordinary_likelihood[k][u][a]
            * law.boundary_kernel[z][k][u][g][a][x][1]
            * kernel.probabilities[z][k][u][g][a][x][1][y]
        )
    evidence = sum(q for y in raw for k in y for q in k)
    masses = [F()] * (len(law.products) + 1)
    selected = F()
    for y, k, g, yn in product(range(2), range(3), range(2), range(2)):
        if pr.type_observation[k] != "copper":
            continue
        q = (
            raw[y][k][g]
            * pr.own_program[0][g][-1]
            * pr.next_state_signal[y][k][g][yn][1]
        )
        selected += q
        for action, p in enumerate(pr.interline_program[0][1][g]):
            masses[action] += q * p
    prediction = None if not selected else tuple(q / selected for q in masses)
    return evidence, prediction, raw


def test_full_path_signal_reference_and_source_identity(bundle):
    args = arguments(bundle)
    snapshot = pickle.dumps(bundle[0].snapshot())
    rng = np.random.get_state()
    result = subject.bind_transported_stage_signal(**args)
    evidence, prediction, raw = reference(args)
    assert result.analyst["upstream_evidence"] == evidence
    assert result.analyst["interline_action_probabilities"] == prediction
    expected = tuple(tuple(tuple(q / evidence for q in k) for k in y) for y in raw)
    assert result.analyst["transported_initial_joint"] == expected
    assert snapshot == pickle.dumps(bundle[0].snapshot())
    now = np.random.get_state()
    assert rng[0] == now[0] and np.array_equal(rng[1], now[1]) and rng[2:] == now[2:]
    assert (
        _units(subject.transported_stage_signal_record(result))
        < result.analyst["structural_bound"]
    )
    assert result.analyst["lifecycle_utility_CS_service_payoff"] is None


@pytest.mark.parametrize("scale", [F(1, 10**850), F()])
def test_tiny_and_zero_upstream_no_unit_prior(bundle, scale, monkeypatch):
    args = arguments(bundle, scale)
    calls = []
    original = subject.compile_type_observation_attempt

    def count(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    monkeypatch.setattr(subject, "compile_type_observation_attempt", count)
    result = subject.bind_transported_stage_signal(**args)
    evidence, prediction, _ = reference(args)
    assert result.analyst["upstream_evidence"] == evidence
    assert result.analyst["interline_action_probabilities"] == prediction
    assert len(calls) == int(bool(scale))
    if not scale:
        assert (
            result.analyst["compiled"] is None
            and result.analyst["transported_initial_joint"] is None
        )
        assert result.analyst["inputs"]["programme"] is not None


@pytest.mark.parametrize("which", ["category", "signal"])
def test_zero_category_or_signal_evidence_keeps_complete_support(bundle, which):
    args = arguments(bundle)
    pr = args["programme"]
    if which == "category":
        pr = replace(pr, type_observation=("silver",) * 3)
    else:
        pr = replace(
            pr,
            next_state_signal=nested(
                (2, 3, 2, 2, 3), lambda y, k, g, yn, s: F(yn == y and s == 2)
            ),
        )
    args["programme"] = pr
    result = subject.bind_transported_stage_signal(**args)
    assert result.analyst["compiled"] is not None
    assert result.analyst["interline_action_probabilities"] is None
    assert result.analyst["selected_signal_reach_cell"]["evidence"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("physical_states", ("other", "y1")),
        ("latent_types", ("new", "k1", "zero-k")),
        ("private_marks", ("new", "g1")),
        ("interline_enabled", False),
        ("conditioning_view", {}),
        ("type_observation", ("copper",)),
        ("signals", ("clear", "fog")),
        ("own_program", ()),
        ("next_state_signal", ()),
        ("interline_program", ()),
        ("own_menus", ()),
        ("definition", ""),
    ],
)
def test_invalid_complete_programmes_refused_before_forwards(
    bundle, monkeypatch, field, value
):
    args = arguments(bundle)
    args["programme"] = replace(args["programme"], **{field: value})

    def forbidden(*a, **kw):
        raise AssertionError("probability forward before complete validation")

    monkeypatch.setattr(subject, "bind_full_path_transport", forbidden)
    monkeypatch.setattr(subject, "compile_type_observation_attempt", forbidden)
    with pytest.raises((TypeError, ValueError)):
        subject.bind_transported_stage_signal(**args)


@pytest.mark.parametrize(
    "case",
    [
        "programme",
        "policy",
        "periods",
        "root",
        "menu",
        "baseline",
        "category",
        "signal",
        "pending",
    ],
)
def test_full_source_receipt_refusals_before_forwards(bundle, case, monkeypatch):
    args = arguments(bundle)
    full = copy.deepcopy(_encoded(args["later_source"]))
    if case == "programme":
        full["programme"]["definition"] = "changed future caller programme"
    elif case == "policy":
        full["parent_source"]["source_context"]["dated_rebooking"] = {}
    elif case == "periods":
        args["later_periods"] = args["later_periods"][:-1]
    elif case == "root":
        args["interline"] = replace(args["interline"], analyst_root=999999)
    elif case == "menu":
        args["interline"] = replace(
            args["interline"],
            computation_input={**args["interline"].computation_input, "offers": ()},
        )
    elif case == "baseline":
        args["baseline"] = args["later_periods"][-1].journal
    elif case in ("category", "signal"):
        args["observation"] = replace(args["observation"], **{case: "different"})
    elif case == "pending":
        full["pending"] = {"unfinished": True}
    args["later_source"] = full

    def forbidden(*a, **kw):
        raise AssertionError("forward before full source check")

    monkeypatch.setattr(subject, "bind_full_path_transport", forbidden)
    with pytest.raises((TypeError, ValueError, KeyError)):
        subject.bind_transported_stage_signal(**args)


def test_limit_before_any_forward_and_owned_failure(bundle, monkeypatch):
    args = arguments(bundle)
    before = _encoded(args)
    args["row_limit"] = 1

    def forbidden(*a, **kw):
        raise AssertionError("limit must precede forwards")

    monkeypatch.setattr(subject, "bind_full_path_transport", forbidden)
    with pytest.raises(RuntimeError):
        subject.bind_transported_stage_signal(**args)
    args["row_limit"] = LIMIT
    assert _same(args, before)


def test_owned_exports_privacy_and_one_forward_each(bundle, monkeypatch):
    args = arguments(bundle)
    calls = []
    for name in ("bind_full_path_transport", "compile_type_observation_attempt"):
        original = getattr(subject, name)

        def wrapper(*a, _original=original, _name=name, **kw):
            calls.append(_name)
            return _original(*a, **kw)

        monkeypatch.setattr(subject, name, wrapper)
    result = subject.bind_transported_stage_signal(**args)
    assert calls == ["bind_full_path_transport", "compile_type_observation_attempt"]
    view = subject.transported_stage_signal_view(result)
    text = str(view)
    for word in (
        "analyst_root",
        "posterior",
        "initial_joint",
        "current_result",
        "actual_class",
    ):
        assert word not in text
    record = subject.transported_stage_signal_record(result)
    record["analyst"] = {}
    assert result.analyst["inputs"]
    with pytest.raises(TypeError):
        result.analyst["inputs"]["law"] = {}
    assert subject.transported_stage_signal_view(result) == view


def test_new_unused_zero_upstream_kernel_invalid_before_forward(bundle, monkeypatch):
    args = arguments(bundle, F())
    pr = args["programme"]
    invalid = nested(
        (2, 3, 2, 2, 3),
        lambda y, k, g, yn, s: (
            F() if (y, k, g) == (1, 2, 1) else pr.next_state_signal[y][k][g][yn][s]
        ),
    )
    args["programme"] = replace(pr, next_state_signal=invalid)

    def forbidden(*a, **kw):
        raise AssertionError("invalid unused template before upstream forward")

    monkeypatch.setattr(subject, "bind_full_path_transport", forbidden)
    with pytest.raises(ValueError, match="every complete joint"):
        subject.bind_transported_stage_signal(**args)


def test_new_forward_failure_no_input_mutation_or_partial_result(bundle, monkeypatch):
    args = arguments(bundle)
    before = _encoded(args)

    def fail(*a, **kw):
        raise RuntimeError("NEW injected finite backend failure")

    monkeypatch.setattr(subject, "compile_type_observation_attempt", fail)
    with pytest.raises(RuntimeError, match="NEW injected"):
        subject.bind_transported_stage_signal(**args)
    assert _same(args, before)


def test_new_all_source_export_support_and_structural_bound(bundle):
    args = arguments(bundle)
    result = subject.bind_transported_stage_signal(**args)
    record = subject.transported_stage_signal_record(result)
    assert len(result.analyst["inputs"]["later_periods"]) == 3
    assert len(result.analyst["inputs"]["early_periods"]) == 1
    assert len(args["baseline"].events) == 7
    assert len(args["own"].receipt.source_journal.events) == 11
    assert len(args["interline"].receipt.source_journal.events) == 16
    assert _same(result.analyst["inputs"]["later_source"], args["later_source"])
    assert len(result.analyst["compiled"]["report"]["history_cells"]) == 48
    assert _units(record) < result.analyst["structural_bound"]
    assert len(result.analyst["inputs"]["early_source"]["programme"]["deliveries"]) == 2
