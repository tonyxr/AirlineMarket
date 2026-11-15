"""A.10/A.5/A.14/B0/V9: NEW genuine source/typed association units."""

import copy
import importlib.util
import os
import pickle
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F
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
from airline_market.sim.booking_loss_law import SCOPE, BookingLossLaw, OriginalOffer
from airline_market.sim.native_causal_information import (
    DeclaredKnownType,
    NativeCausalInformationMarketEngine,
    native_causal_information_record,
)
from airline_market.sim.received_information import DisclosureField
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

if os.environ.get("BOOKING_LOSS_BINDING_SOURCE_DIR"):
    spec = importlib.util.spec_from_file_location(
        "airline_market.sim.booking_loss_binding",
        Path(os.environ["BOOKING_LOSS_BINDING_SOURCE_DIR"]) / "booking_loss_binding.py",
    )
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
from airline_market.sim import booking_loss_binding as m

LIMIT = 10**9  # Structural support units, not bytes/RSS/scientific budget.


def source(*, loss=True, selected_period=0, outside=False, connection=False):
    # NEW constructor text adaptation only; no old fixture/helper execution.
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
    start = datetime(2027, 5, 1, tzinfo=timezone.utc)
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
                "new_binding_exit96974", "rival_exit", "B_OH0", 3, rival_carrier="B"
            ),
        )
        if loss
        else (),
    )
    tape = ExogenousTape.generate(
        compiled, 31, registry=StreamRegistry(96974, "unit/V9_booking_loss_binding")
    )
    target = compiled.product_index["B_OD0_0" if connection else "B_OH0"]
    market = int(compiled.product_market[target])
    arrivals = tuple(
        ArrivalMark(
            time=(i + 1) / 3, market=market, segment=0, thinning=0, class_uniform=0.4
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
            DisclosureField("message", "text", "NEW explicit unused schema", "text"),
        ),
        causal_known_types=tuple(
            DeclaredKnownType(
                (selected_period, i, market),
                None,
                "no supplied label; not ignorance or native class",
                "new96974",
            )
            for i in range(2)
        ),
        causal_deliveries=(),
        causal_definition="NEW source bridge units96974",
        causal_unobserved_definition="not native probability or complete knowledge",
    )
    engine.reset(tape=tape)
    actions = {
        c.code: PriceAction(
            {j: 220 if i == 0 else 340 for j in compiled.products_of[i]}
        )
        for i, c in enumerate(compiled.spec.carriers)
    }
    for _ in range(selected_period + 1):
        engine.step(actions)
    periods = engine.causal_periods()
    row = periods[-1].inputs[0]
    return engine, row, periods, native_causal_information_record(engine)


@pytest.fixture(scope="module")
def cancelled():
    return source()


@pytest.fixture(scope="module")
def unchanged():
    return source(loss=False)


def law(native, scale=F(1)):
    offers = tuple(
        OriginalOffer(
            r["product_id"], float.fromhex(r["fare"]["float_hex"]), r["offered"]
        )
        for r in native[1].computation_input["offers"]
    )
    target = next(i for i, r in enumerate(offers) if r.product_id.startswith("B_"))
    ug = (
        ((F(1, 2), F()), (F(1, 4), F(1, 4))),
        ((F(1, 4), F(1, 4)), (F(1, 4), F(1, 4))),
    )
    prior = (
        tuple(
            tuple(
                tuple((F(1, 4) if k == 0 else F(3, 4)) * ug[k][u][g] for g in range(2))
                for u in range(2)
            )
            for k in range(2)
        ),
    )
    a = len(offers) + 1
    choices = tuple(
        tuple(
            tuple(
                (F(2, 3) if k == 0 else F(1, 3))
                if j == target
                else (F(1, 6) if k == 0 else F(1, 3))
                for j in range(a)
            )
            for _ in range(2)
        )
        for k in range(2)
    )

    def boundary(k, u, j):
        c = scale * (F(1, 5) if k == 0 else F(3, 5)) if j < a - 1 else F()
        return tuple(
            tuple(
                (c if cancel else 1 - c)
                * (F(1 + u + cancel, 4) if x else 1 - F(1 + u + cancel, 4))
                for cancel in range(2)
            )
            for x in range(2)
        )

    kernel = (
        tuple(
            tuple(
                tuple(tuple(boundary(k, u, j) for j in range(a)) for _ in range(2))
                for u in range(2)
            )
            for k in range(2)
        ),
    )
    return BookingLossLaw(
        "new-person96974",
        ("Z",),
        ("X0", "X1"),
        ("K0", "K1"),
        ("U0", "U1"),
        ("G0", "G1"),
        offers,
        prior,
        choices,
        kernel,
        m.choice_boundary_conditioning_context(native[1]),
        SCOPE,
        "before current choice/cancellation, not already conditioned",
        "declared U, not actual ordinary class/marks",
        "declared persistent G distinct from U",
        "joint latent X/cancellation declaration, not native law",
        "NEW law96974",
        "unit declaration",
    )


def bind(native, model=None, **kwargs):
    return m.bind_choice_boundary(
        law(native) if model is None else model,
        native[1],
        native[2],
        native[3],
        m.ChoiceBoundaryAssociation(
            "new-person96974",
            "explicit logical label to supplied original key",
            "new declaration",
        ),
        row_limit=kwargs.get("row_limit", LIMIT),
    )


def test_exact_selection_and_single_forward(cancelled, monkeypatch):
    calls = []
    original = m.compile_booking_loss_law

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(m, "compile_booking_loss_law", spy)
    out = bind(cancelled)
    selected = out.analyst["selected_history"]
    assert calls == [1] and selected["evidence"] == F(11, 60)
    assert sum(c["posterior"] for c in selected["components"] if c["type"] == 1) == F(
        9, 11
    )
    assert sum(
        c["posterior"] for c in selected["components"] if c["attempt_mark"] == 0
    ) == F(6, 11)
    assert out.analyst["cancelled"] and out.analyst["closed_native_periods"] == 1
    assert (
        out.analyst["inputs"]["periods"][0]["journal"]["events"][-1]["position"][
            "anchor_utc"
        ]["aware_utc"]
        == cancelled[0].demand_scenario.clock.at(1).isoformat()
    )


@pytest.mark.parametrize("scale", [F(1, 10**600), F()])
def test_tiny_and_zero_actual_event_retained(cancelled, scale):
    out = bind(cancelled, law(cancelled, scale))
    h = out.analyst["selected_history"]
    assert h["evidence"] == F(11, 60) * scale
    assert (h["conditional_joint"] is None) == (scale == 0)
    assert len(h["components"]) == 8


def test_uncancelled_actual_source(unchanged):
    out = bind(unchanged)
    assert not out.analyst["cancelled"] and out.analyst["actual_root"] is not None
    assert out.analyst["selected_history"]["evidence"] == F(7, 30)


def test_outside_and_all_original_support():
    native = source(outside=True)
    out = bind(native)
    assert out.analyst["actual_root"] is out.analyst["actual_product"] is None
    assert not out.analyst["cancelled"]
    assert (
        len(
            out.analyst["inputs"]["source"]["parent_source"]["source_context"][
                "ordinary_information"
            ][0]["rows"]
        )
        == 2
    )


def test_multiple_complete_periods_and_closing_n():
    native = source(selected_period=1)
    out = bind(native)
    assert out.analyst["closed_native_periods"] == 2
    assert out.analyst["inputs"]["periods"][0]["inputs"] == ()
    assert out.analyst["cancelled"]


def test_native_connection_purchase_loss():
    native = source(connection=True)
    out = bind(native)
    assert out.analyst["actual_product"] == "B_OD0_0" and out.analyst["cancelled"]


def test_prechoice_privacy_reader_and_owned_inputs(cancelled, monkeypatch):
    before = pickle.dumps(cancelled[0].snapshot())
    rng = np.random.get_state()
    out = bind(cancelled)
    monkeypatch.setattr(
        m, "compile_booking_loss_law", lambda *a, **k: pytest.fail("reader forward")
    )
    view = m.choice_boundary_prechoice_view(out)
    raw = m.choice_boundary_record(out)
    assert view["known_type"]["value"] is None and view["earlier_person_history"] == ()
    text = str(view)
    for field in (
        "actual_root",
        "taste_class",
        "selected_history",
        "native_spec",
        "actual_original_key",
    ):
        assert field not in text
    raw["analyst"]["inputs"]["source"]["failed"] = True
    assert not m.choice_boundary_record(out)["analyst"]["inputs"]["source"]["failed"]
    assert pickle.dumps(cancelled[0].snapshot()) == before
    now = np.random.get_state()
    assert all(np.array_equal(a, b) for a, b in zip(rng, now))


@pytest.mark.parametrize(
    "damage",
    [
        "failed",
        "pending",
        "outside_omitted",
        "root_partition",
        "cancellation_omitted",
        "cancellation_period",
        "fare",
        "offered",
        "context",
        "typed_prefix",
        "postclose_tail",
        "label",
        "last_period",
        "native_row_market",
        "first_stage",
    ],
)
def test_full_source_refusals_before_forward(cancelled, damage, monkeypatch):
    native = list(cancelled)
    native[3] = copy.deepcopy(native[3])
    model = law(cancelled)
    if damage in ("failed", "pending"):
        native[3][damage] = True if damage == "failed" else []
    elif damage == "outside_omitted":
        period = native[3]["parent_source"]["source_context"]["ordinary_information"][0]
        period["rows"] = period["rows"][:-1]
    elif damage == "root_partition":
        native[3]["parent_source"]["source_context"]["own_tickets"] = {}
    elif damage == "cancellation_omitted":
        histories = native[3]["parent_source"]["source_context"]["final_own_histories"]
        histories["0"] = histories["0"][:-1]
    elif damage == "cancellation_period":
        native[3]["parent_source"]["source_context"]["final_own_histories"]["0"][1][
            "native_period"
        ] = 0
    elif damage == "fare":
        offers = list(model.products)
        offers[1] = replace(offers[1], fare=111.0)
        model = replace(model, products=tuple(offers))
    elif damage == "offered":
        row = copy.deepcopy(native[1])
        menu = dict(row.computation_input)
        offers = [dict(o) for o in menu["offers"]]
        offers[1]["offered"] = False
        menu["offers"] = offers
        native[1] = replace(row, computation_input=menu)
    elif damage == "context":
        model = replace(model, conditioning_context={"already_conditioned": True})
    elif damage == "typed_prefix":
        native[1] = replace(
            native[1],
            receipt=replace(native[1].receipt, source_journal=native[2][-1].journal),
        )
    elif damage == "postclose_tail":
        journal = native[3]["journal"]["analyst_context"]
        journal["events"] = journal["events"][:-1]
    elif damage == "native_row_market":
        native[3]["parent_source"]["source_context"]["ordinary_information"][0]["rows"][
            0
        ]["analyst_person"]["native_row"]["market"] = 1
    elif damage == "first_stage":
        native[3]["parent_source"]["source_context"]["final_own_histories"]["0"][0][
            "stage"
        ] = "own"
    elif damage == "label":
        model = replace(model, original_person_label="other")
    else:
        native[2] = ()
    monkeypatch.setattr(
        m,
        "compile_booking_loss_law",
        lambda *a, **k: pytest.fail("invalid input forwarded"),
    )
    with pytest.raises((ValueError, TypeError, KeyError)):
        bind(native, model)


def test_limit_before_any_backend_forward(cancelled, monkeypatch):
    monkeypatch.setattr(
        m, "compile_booking_loss_law", lambda *a, **k: pytest.fail("limit forwarded")
    )
    with pytest.raises(RuntimeError):
        bind(cancelled, row_limit=1)


def test_complete_output_fits_conservative_structural_bound(cancelled):
    out = bind(cancelled)
    assert m._units(m.choice_boundary_record(out)) <= out.analyst["structural_bound"]


def test_backend_failure_preserves_all_inputs(cancelled, monkeypatch):
    before = pickle.dumps(cancelled[0].snapshot())
    source_before = copy.deepcopy(cancelled[3])

    def fail(*args, **kwargs):
        raise RuntimeError("NEW injected backend error")

    monkeypatch.setattr(m, "compile_booking_loss_law", fail)
    with pytest.raises(RuntimeError, match="injected"):
        bind(cancelled)
    assert (
        source_before == cancelled[3]
        and pickle.dumps(cancelled[0].snapshot()) == before
    )


def test_owned_export_retains_full_native_binary64_context(cancelled):
    raw = m.choice_boundary_record(bind(cancelled))
    assert (
        raw["analyst"]["law_result"]["law"]["products"][1]["fare"]["float_hex"]
        == float(340).hex()
    )
    assert not raw["analyst"]["native_probability_or_knowledge_authenticated"]
    assert raw["analyst"]["lifecycle_utility_CS_service_payoff"] is None


@pytest.mark.parametrize("bad", [None, {}, "wrong"])
def test_reader_requires_new_typed_binding(bad):
    with pytest.raises(TypeError):
        m.choice_boundary_record(bad)


def test_earlier_close_refuses_future_input_prefix(monkeypatch):
    native = list(source(selected_period=1))
    native[3] = copy.deepcopy(native[3])
    periods = list(native[2])
    periods[0] = replace(periods[0], journal=periods[-1].journal)
    native[2] = tuple(periods)
    from airline_market.sim.shared_mark_attempt import _encoded

    native[3]["periods"] = _encoded(native[2])
    monkeypatch.setattr(
        m,
        "compile_booking_loss_law",
        lambda *a, **k: pytest.fail("overlong close forwarded"),
    )
    with pytest.raises(ValueError, match="future"):
        bind(native)
