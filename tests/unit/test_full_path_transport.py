"""A.10/A.5/A.14/B0/V9: NEW exact transport and genuine-prefix units."""

import copy
import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.dated_demand import compile_dated_demand
from airline_market.policy.dated_rescue import DatedRebookingPolicy
from airline_market.policy.dated_reserve import UtcBookingClock
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.sim.booking_loss_binding import (
    ChoiceBoundaryAssociation,
    choice_boundary_conditioning_context,
)
from airline_market.sim.booking_loss_law import BookingLossLaw, OriginalOffer
from airline_market.sim.native_causal_information import (
    DeclaredKnownType,
    NativeCausalInformationMarketEngine,
    native_causal_information_record,
)
from airline_market.sim.received_information import DisclosureField, _units
from airline_market.sim.shared_mark_attempt import _encoded
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

if os.environ.get("FULL_PATH_TRANSPORT_SOURCE_DIR"):
    module_spec = importlib.util.spec_from_file_location(
        "airline_market.sim.full_path_transport",
        Path(os.environ["FULL_PATH_TRANSPORT_SOURCE_DIR"]) / "full_path_transport.py",
    )
    loaded = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = loaded
    module_spec.loader.exec_module(loaded)
from airline_market.sim import full_path_transport as m

LIMIT = 10**9  # ALL structural units, not bytes/RSS/scientific budget.


def nested(shape, fn, prefix=()):
    return (
        fn(*prefix)
        if not shape
        else tuple(nested(shape[1:], fn, prefix + (i,)) for i in range(shape[0]))
    )


def law(scale=F(1), offers=None, context=None):
    # NEW literals; logical label is not a native class/awareness claim.
    offers = offers or (OriginalOffer("new-P", -0.0, True),)
    a = len(offers) + 1
    target = next((i for i, p in enumerate(offers) if p.product_id.startswith("B_")), 0)
    prior = nested(
        (2, 2, 2, 2), lambda z, k, u, g: F(1 + 8 * z + 4 * k + 2 * u + g, 136)
    )
    choices = nested(
        (2, 2, a),
        lambda k, u, j: (
            F(2 + k + u, 6)
            if j == target
            else 1 - F(2 + k + u, 6)
            if j == a - 1
            else F()
        ),
    )

    def boundary(z, k, u, g, j, x, c):
        cancel = scale * F(1 + z + u + g, 6) if j < a - 1 else F()
        x1 = F(1 + k + c, 4)
        return (cancel if c else 1 - cancel) * (x1 if x else 1 - x1)

    return BookingLossLaw(
        "new-full-path-person98974",
        ("Z0", "Z1"),
        ("X0", "X1"),
        ("K0", "K1"),
        ("U0", "U1"),
        ("G0", "G1"),
        offers,
        prior,
        choices,
        nested((2, 2, 2, 2, a, 2, 2), boundary),
        {"earlier": "NEW fixed I0", "unknown": None, "signed": -0.0}
        if context is None
        else context,
        "before-original-choice-and-boundary",
        "explicit NEW full varying history",
        "U distinct ordinary mark",
        "G persistent within later attempt",
        "joint X and cancellation law",
        "NEW declared finite model",
        "unit98974",
    )


def kernel(parent):
    def row(z, k, u, g, a, x, c, y):
        p0, p1 = F(1 + u, 6), F(1 + z + g, 6)
        return (p0, p1, 1 - p0 - p1)[y]

    return m.FullPathKernel(
        parent.initial_states,
        parent.boundary_states,
        parent.latent_types,
        parent.ordinary_marks,
        parent.attempt_marks,
        tuple(p.product_id for p in parent.products) + (None,),
        ("Y0", "Y1", "Y2"),
        nested((2, 2, 2, 2, len(parent.products) + 1, 2, 2, 3), row),
        m.SCOPE,
        "R depends on full original Z/U, not just X/K/G",
        "new-unit98974",
    )


def run(parent=None, future=None, selected="new-P", cancelled=True, limit=LIMIT):
    parent = law() if parent is None else parent
    future = kernel(parent) if future is None else future
    return m.compile_full_path_transport(
        parent, future, selected, cancelled=cancelled, row_limit=limit
    )


def reference(parent, future, selected, cancel):
    # Independent conditional expectation: accumulate unconditioned complete
    # paths into Y/K/G first, divide once by selected event evidence.
    a = (
        tuple(p.product_id for p in parent.products).index(selected)
        if selected
        else len(parent.products)
    )
    masses = nested((3, 2, 2), lambda *_: F())
    out = [[[masses[y][k][g] for g in range(2)] for k in range(2)] for y in range(3)]
    evidence = F()
    for z, k, u, g, x in product(range(2), repeat=5):
        w = (
            parent.initial_joint[z][k][u][g]
            * parent.ordinary_likelihood[k][u][a]
            * parent.boundary_kernel[z][k][u][g][a][x][int(cancel)]
        )
        evidence += w
        for y in range(3):
            out[y][k][g] += w * future.probabilities[z][k][u][g][a][x][int(cancel)][y]
    return evidence, None if not evidence else tuple(
        tuple(tuple(v / evidence for v in row) for row in state) for state in out
    )


def test_full_independent_path_reference_and_KG_dependence():
    p = law()
    f = kernel(p)
    r = run(p, f)
    e, j = reference(p, f, "new-P", True)
    assert r.analyst["evidence"] == e and m.transported_joint(r) == j
    assert sum(v for state in j for row in state for v in row) == 1
    assert j[0][0][0] * j[1][1][1] != j[0][1][1] * j[1][0][0]
    assert len(r.analyst["terms"]) == 2**7 * 3
    assert len(r.analyst["components"]) == 12
    assert any(
        not t["selected_history"] and t["conditional_probability"] > 0
        for t in r.analyst["terms"]
    )


def test_omitted_U_redraw_is_different_model():
    p = law()
    r = run(p)
    e = r.analyst["evidence"]
    selected_u = sum(
        t["conditional_mass"]
        for t in r.analyst["terms"]
        if t["parent_coordinate"][2] == 1
    )
    prior_u = sum(
        p.initial_joint[z][k][1][g] for z, k, g in product(range(2), repeat=3)
    )
    assert e > 0 and selected_u != prior_u
    actual_y0 = sum(row[g] for row in m.transported_joint(r)[0] for g in range(2))
    assert actual_y0 == (1 + selected_u) / 6
    assert actual_y0 != (1 + prior_u) / 6


@pytest.mark.parametrize("scale", [F(1, 10**750), F()])
def test_tiny_and_zero_no_downstream_choice(scale):
    p = law(scale)
    r = run(p)
    assert r.analyst["evidence"] == reference(p, kernel(p), "new-P", True)[0]
    assert r.analyst["downstream_choice_compiler_called"] is False
    assert (m.transported_joint(r) is None) == (scale == 0)
    assert all(
        (t["conditional_mass"] is None) == (scale == 0) for t in r.analyst["terms"]
    )
    if scale:
        assert (
            float(r.analyst["evidence"]) == 0
            and sum(v for st in m.transported_joint(r) for row in st for v in row) == 1
        )


@pytest.mark.parametrize(
    "selected,cancel", [("new-P", False), (None, False), (None, True)]
)
def test_all_action_and_outside_histories(selected, cancel):
    p = law()
    r = run(p, selected=selected, cancelled=cancel)
    assert (r.analyst["evidence"], m.transported_joint(r)) == reference(
        p, kernel(p), selected, cancel
    )
    assert len(r.analyst["terms"]) == 384


def test_zero_prior_and_zero_kernel_support_retained():
    p = law()
    p = replace(
        p,
        initial_joint=nested(
            (2, 2, 2, 2), lambda z, k, u, g: F(1, 8) if k == 0 else F()
        ),
    )
    f = kernel(p)
    f = replace(
        f,
        probabilities=nested(
            (2, 2, 2, 2, 2, 2, 2, 3), lambda z, k, u, g, a, x, c, y: F(y == 0)
        ),
    )
    r = run(p, f)
    assert len(r.analyst["terms"]) == 384
    assert all(
        t["selected_joint_mass"] == 0
        for t in r.analyst["terms"]
        if t["parent_coordinate"][1] == 1
    )
    assert all(
        v == 0 for state in m.transported_joint(r)[1:] for row in state for v in row
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("initial_states", ("Z1", "Z0")),
        ("boundary_states", ("X1", "X0")),
        ("latent_types", ("wrong", "K1")),
        ("ordinary_marks", ("G0", "G1")),
        ("attempt_marks", ("U0", "U1")),
        ("action_products", (None, "new-P")),
        ("future_states", ("dup", "dup")),
        ("conditioning_scope", "after-category"),
        ("definition", ""),
    ],
)
def test_full_named_axis_scope_refusals_before_forward(monkeypatch, field, value):
    p = law()
    f = replace(kernel(p), **{field: value})
    monkeypatch.setattr(
        m,
        "compile_booking_loss_law",
        lambda *a, **k: pytest.fail("forward before validation"),
    )
    with pytest.raises((ValueError, TypeError)):
        run(p, f)


@pytest.mark.parametrize(
    "mode", ["missing", "negative", "nonunit", "float", "unused-outside"]
)
def test_every_kernel_row_even_unused_refused_before_forward(monkeypatch, mode):
    p = law()
    f = kernel(p)
    v = copy.deepcopy(_encoded(f.probabilities))

    # Build the full tuple using literal typed Fraction changes at an UNUSED
    # outside/uncancelled history, so zero selected mass cannot excuse invalid R.
    def alter(z, k, u, g, a, x, c, y):
        original = f.probabilities[z][k][u][g][a][x][c][y]
        if (z, k, u, g, a, x, c) == (1, 1, 1, 1, 1, 1, 0) and y == 0:
            return (
                -F(1)
                if mode == "negative"
                else 0.5
                if mode == "float"
                else original + F(1)
            )
        return original

    v = nested((2, 2, 2, 2, 2, 2, 2, 3), alter)
    if mode == "missing":
        v = v[:-1]
    f = replace(f, probabilities=v)
    monkeypatch.setattr(
        m,
        "compile_booking_loss_law",
        lambda *a, **k: pytest.fail("invalid full support forwarded"),
    )
    with pytest.raises((ValueError, TypeError)):
        run(p, f)


def test_structural_bound_counts_full_output_and_fails_before_backend(monkeypatch):
    r = run()
    bound = r.analyst["structural_bound"]
    assert _units(m.full_path_transport_record(r)) <= bound
    monkeypatch.setattr(
        m, "compile_booking_loss_law", lambda *a, **k: pytest.fail("limit forwarded")
    )
    with pytest.raises(m.FullPathTransportLimit):
        run(limit=bound - 1)


@pytest.mark.parametrize(
    "selected,cancel,limit",
    [
        (4, True, LIMIT),
        ("missing", True, LIMIT),
        ("new-P", 1, LIMIT),
        ("new-P", True, True),
    ],
)
def test_invalid_selection_or_limit(selected, cancel, limit):
    with pytest.raises((ValueError, TypeError)):
        run(selected=selected, cancelled=cancel, limit=limit)


def test_owned_immutable_lossless_and_reader_no_forward(monkeypatch):
    p = law()
    r = run(p)
    before = pickle.dumps(r)
    out = m.full_path_transport_record(r)
    out["analyst"]["terms"] = ()
    assert pickle.dumps(r) == before
    assert (
        m.full_path_transport_record(r)["analyst"]["parent"]["law"]["products"][0][
            "fare"
        ]["float_hex"]
        == "-0x0.0p+0"
    )
    json.dumps(m.full_path_transport_record(r), allow_nan=False)
    monkeypatch.setattr(
        m, "compile_booking_loss_law", lambda *a, **k: pytest.fail("reader replay")
    )
    assert m.transported_joint(r) is r.analyst["conditional_joint"]
    assert m.full_path_transport_record(r)
    with pytest.raises((TypeError, AttributeError)):
        r.analyst["terms"].clear()
    with pytest.raises(TypeError):
        m.full_path_transport_record(object())


def native_source(loss=True, selected_period=0, outside=False):
    # Locally adapted constructor text only; no prior fixture/helper invocation.
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
    start = datetime(2027, 5, 25, tzinfo=timezone.utc)
    clock = UtcBookingClock(tuple(start + timedelta(hours=12 * i) for i in range(4)))
    departures = tuple(
        (
            leg.id,
            start
            + timedelta(
                hours=72 + 12 * (selected_period + 1) if leg.id == "B_OH0" else 120
            ),
        )
        for leg in compiled.spec.legs
    )
    scenario = replace(
        compile_dated_demand(compiled, clock, departures),
        physical_shocks=(
            B1DatedShock(
                "new_transport98974", "rival_exit", "B_OH0", 3, rival_carrier="B"
            ),
        )
        if loss
        else (),
    )
    tape = ExogenousTape.generate(
        compiled, 31, registry=StreamRegistry(98974, "unit/V9_full_path_transport")
    )
    target = compiled.product_index["B_OH0"]
    market = int(compiled.product_market[target])
    arrivals = tuple(
        ArrivalMark(
            time=(i + 1) / 3, market=market, segment=0, thinning=0, class_uniform=0.3
        )
        for i in range(2)
    )
    marks = [(), (), ()]
    marks[selected_period] = arrivals
    g = tape.gumbel.copy()
    g[:] = -1e9
    g[:, :, -1] = 0
    g[:, :, target] = 1e9
    g[:, 1, -1] = 2e9
    if outside:
        g[:, 0, -1] = 2e9
    g.setflags(write=False)
    tape = replace(tape, arrival_marks=tuple(marks), gumbel=g)
    engine = NativeCausalInformationMarketEngine(
        scenario,
        dated_rebooking=DatedRebookingPolicy(72, 20, 100, interline=True),
        causal_fields=(
            DisclosureField("notice", "text", "NEW unused declaration", "text"),
        ),
        causal_known_types=tuple(
            DeclaredKnownType(
                (selected_period, i, market),
                None,
                "no supplied label; not ignorance/class",
                "unit98974",
            )
            for i in range(2)
        ),
        causal_deliveries=(),
        causal_definition="NEW typed transport source98974",
        causal_unobserved_definition="not native law/knowledge",
    )
    engine.reset(tape=tape)
    actions = {
        c.code: PriceAction(
            {j: 230 if i == 0 else 360 for j in compiled.products_of[i]}
        )
        for i, c in enumerate(compiled.spec.carriers)
    }
    for _ in range(selected_period + 1):
        engine.step(actions)
    return (
        engine,
        engine.causal_periods()[-1].inputs[0],
        engine.causal_periods(),
        native_causal_information_record(engine),
    )


@pytest.fixture(scope="module")
def native():
    return native_source()


def native_law(source, scale=F(1)):
    offers = tuple(
        OriginalOffer(
            r["product_id"], float.fromhex(r["fare"]["float_hex"]), r["offered"]
        )
        for r in source[1].computation_input["offers"]
    )
    return law(scale, offers, choice_boundary_conditioning_context(source[1]))


def bind(source, scale=F(1), future=None, limit=LIMIT):
    p = native_law(source, scale)
    return m.bind_full_path_transport(
        p,
        kernel(p) if future is None else future,
        source[1],
        source[2],
        source[3],
        ChoiceBoundaryAssociation(
            p.original_person_label,
            "explicit logical association, not sampled class",
            "unit98974",
        ),
        row_limit=limit,
    )


def test_genuine_closed_source_link_all_people_and_identity(native, monkeypatch):
    snapshot = pickle.dumps(native[0].snapshot())
    rng = pickle.dumps(np.random.get_state())
    calls = []
    original = m.bind_choice_boundary

    def spy(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    monkeypatch.setattr(m, "bind_choice_boundary", spy)
    r = bind(native)
    assert calls == [1] and r.analyst["source_prefix_binding_executed"] is True
    parent = r.analyst["parent"]["analyst"]
    assert parent["cancelled"] and parent["closed_native_periods"] == 1
    assert len(parent["inputs"]["source"]["journal"]["analyst_context"]["events"]) == 7
    assert parent["actual_product"] == "B_OH0"
    assert (r.analyst["evidence"], m.transported_joint(r)) == reference(
        native_law(native), kernel(native_law(native)), "B_OH0", True
    )
    assert (
        pickle.dumps(native[0].snapshot()) == snapshot
        and pickle.dumps(np.random.get_state()) == rng
    )


def test_native_zero_keeps_full_source_and_null(native):
    r = bind(native, F())
    assert m.transported_joint(r) is None and r.analyst["evidence"] == 0
    assert r.analyst["parent"]["analyst"]["cancelled"] is True
    assert r.analyst["downstream_choice_compiler_called"] is False


def test_native_earlier_empty_and_outside():
    source = native_source(selected_period=1, outside=True)
    r = bind(source)
    assert r.analyst["parent"]["analyst"]["closed_native_periods"] == 2
    assert r.analyst["parent"]["analyst"]["actual_product"] is None
    assert r.analyst["parent"]["analyst"]["actual_root"] is None
    assert m.transported_joint(r) is not None


def test_invalid_kernel_before_source_or_parent_forward(native, monkeypatch):
    p = native_law(native)
    f = replace(kernel(p), attempt_marks=("bad", "axis"))
    monkeypatch.setattr(
        m,
        "bind_choice_boundary",
        lambda *a, **k: pytest.fail("invalid kernel forwarded"),
    )
    with pytest.raises(ValueError):
        bind(native, future=f)


def test_complete_source_refusals_and_injected_failure(native, monkeypatch):
    bad = copy.deepcopy(native[3])
    bad["failed"] = True
    snapshot = pickle.dumps(native[0].snapshot())
    with pytest.raises(ValueError):
        bind((*native[:3], bad))
    monkeypatch.setattr(
        m,
        "bind_choice_boundary",
        lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("new injected pure failure")
        ),
    )
    with pytest.raises(RuntimeError, match="injected"):
        bind(native)
    assert pickle.dumps(native[0].snapshot()) == snapshot


def test_native_full_export_bound_before_binding(native, monkeypatch):
    r = bind(native)
    assert _units(m.full_path_transport_record(r)) <= r.analyst["structural_bound"]
    monkeypatch.setattr(
        m,
        "bind_choice_boundary",
        lambda *a, **k: pytest.fail("full source limit forwarded"),
    )
    with pytest.raises(m.FullPathTransportLimit):
        bind(native, limit=r.analyst["structural_bound"] - 1)
