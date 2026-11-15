"""A.10/A.5/A.14/B0/V9: NEW finite shared-mark and native-prefix units."""

import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest

from airline_market.core.hashing import jsonable
from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.dated_demand import compile_dated_demand
from airline_market.policy.dated_rescue import DatedRebookingPolicy
from airline_market.policy.dated_reserve import UtcBookingClock
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.sim.causal_information import causal_decision_view
from airline_market.sim.native_causal_information import (
    TIMING,
    DeclaredInputDelivery,
    DeclaredKnownType,
    NativeCausalInformationMarketEngine,
    native_causal_information_record,
)
from airline_market.sim.received_information import DisclosureField, _units
from airline_market.sim.reentry_stage_information import _stage_encode
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

if os.environ.get("SHARED_MARK_SOURCE_DIR"):
    p = Path(os.environ["SHARED_MARK_SOURCE_DIR"]) / "shared_mark_attempt.py"
    spec = importlib.util.spec_from_file_location(
        "airline_market.sim.shared_mark_attempt", p
    )
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
from airline_market.sim import shared_mark_attempt as m


def model():
    own = (
        m.AttemptOffer("own", -0.0, True, False, False),
        m.AttemptOffer("other", -12.5, False, False, False),
    )
    inter = (replace(own[0], offered=False), replace(own[1], offered=True))
    kernel = ((F(1, 3), F(1, 6), F()), (F(1, 6), F(1, 3), F()))
    other = ((F(1, 8), F(1, 8), F()), (F(1, 4), F(1, 2), F()))
    zero = ((F(1), F(), F()), (F(), F(), F()))
    return m.SharedMarkAttempt(
        ("good", "bad"),
        ("a", "b", "zero"),
        "declared-fixed",
        {"new": "conditional source declaration"},
        "explicit prior conditioning",
        own,
        inter,
        False,
        True,
        (None, "note", "unreachable"),
        "notice",
        ((F(1, 5), F(2, 5), F()), (F(1, 10), F(3, 10), F())),
        ((F(2, 5), F(), F(3, 5)), (F(1, 5), F(), F(4, 5)), (F(1), F(), F())),
        ((kernel, other, zero), (other, kernel, zero)),
        (
            ((F(), F(3, 4), F(1, 4)), (F(), F(1, 4), F(3, 4)), (F(), F(), F(1))),
            ((F(), F(1, 5), F(4, 5)), (F(), F(4, 5), F(1, 5)), (F(), F(), F(1))),
            ((F(), F(), F(1)),) * 3,
        ),
        "finite programme, no native probability",
        "new-unit",
    )


def compile_new(value=None, limit=2000000):
    return m.compile_shared_mark_attempt(
        model() if value is None else value, row_limit=limit
    )


def test_new_complete_path_reference_and_mass_same_mark():
    value = model()
    result = compile_new(value)
    r = result.report
    n, g, o, p = 2, 3, 3, 3
    expected = []
    for signal in range(o):
        for nxt in range(n):
            for mark in range(g):
                for action in range(p):
                    mass = sum(
                        value.initial_joint[s][mark]
                        * value.own_program[mark][-1]
                        * value.next_state_signal[s][mark][nxt][signal]
                        * value.interline_program[signal][mark][action]
                        for s in range(n)
                    )
                    expected.append(mass)
    assert [x["mass"] for x in r["interline_action_rows"]] == expected
    assert (
        len(r["own_rows"]) == n * g * p
        and len(r["transition_signal_rows"]) == n * g * n * o
    )
    assert len(r["interline_action_rows"]) == o * n * g * p
    assert r["terminal_mass"] == 1
    for cell in r["interline_cells"]:
        if cell["evidence"]:
            assert sum(c["posterior"] for c in cell["components"]) == 1
            assert all(
                c["posterior"] * cell["evidence"] == c["joint_mass"]
                for c in cell["components"]
            )
        else:
            assert all(c["posterior"] is None for c in cell["components"])
    assert not next(c for c in r["interline_cells"] if c["signal"] == "unreachable")[
        "evidence"
    ]


def test_new_tiny_positive_joint_retained_without_float_conversion():
    tiny = F(1, 10**700)
    value = replace(model(), initial_joint=((tiny, 1 - tiny, F()), (F(), F(), F())))
    r = compile_new(value).report
    positives = [
        c["posterior"]
        for cell in r["interline_cells"]
        for c in cell["components"]
        if c["mark"] == 0 and c["posterior"]
    ]
    assert positives and all(x > 0 and float(x) == 0 for x in positives)
    assert any(row["mass"] > 0 and float(row["mass"]) == 0 for row in r["own_rows"])


def test_new_disabled_interline_keeps_all_programmes_but_no_executed_mass():
    r = compile_new(replace(model(), interline_enabled=False)).report
    assert r["transition_signal_rows"] and r["interline_action_rows"]
    assert all(not row["mass"] for row in r["transition_signal_rows"])
    assert all(
        c["posterior"] is None
        for cell in r["interline_cells"]
        for c in cell["components"]
    )
    assert r["terminal_mass"] == 1


def test_new_owned_inputs_exports_and_signed_binary64():
    value = model()
    before = pickle.dumps(value)
    rng = pickle.dumps(np.random.get_state())
    result = compile_new(value)
    record = m.shared_mark_record(result)
    text = json.dumps(record, allow_nan=False)
    assert "-0x0.0p+0" in text
    record["model"].clear()
    assert m.shared_mark_record(result)["model"]
    assert pickle.dumps(value) == before and pickle.dumps(np.random.get_state()) == rng
    with pytest.raises(TypeError):
        result.report["new"] = 0


@pytest.mark.parametrize(
    "change",
    [
        {"initial_joint": ((F(1), F(), F()),)},
        {"initial_joint": ((F(1), F(), F()), (F(1), F(), F()))},
        {"own_program": (((F(1), F(), F()),) * 3,) * 2},
        {"own_program": ((F(), F(1), F()),) * 3},
        {"signals": ("note", "unreachable")},
        {"known_type": None},
        {"interline_enabled": 1},
        {"initial_joint": ((0.2, F(2, 5), F()), (F(1, 10), F(3, 10), F()))},
    ],
)
def test_new_complete_law_type_support_and_feasibility_refusals(change):
    with pytest.raises((ValueError, TypeError)):
        compile_new(replace(model(), **change))


def test_new_full_limit_before_any_forward_and_zero_mark_count(monkeypatch):
    value = model()
    seen = []

    def fail(*args):
        seen.append(True)
        raise AssertionError("forward must not run")

    monkeypatch.setattr(m, "_compile", fail)
    with pytest.raises(m.SharedMarkLimit):
        compile_new(value, limit=1)
    assert not seen


def test_new_structural_bound_covers_full_export():
    result = compile_new()
    actual = _units(m.shared_mark_record(result))
    n, g, o, p = m._validate(result.model)
    upper = (
        2 * _units(m._encoded(result.model))
        + 64 * (n * g * p + n * g * n * o + o * n * g + o * n * g * p)
        + 1000
    )
    assert actual < upper


def fresh_source(*, notice=True, enabled=True, extra=False):
    # NEW constructor adapted locally, no old fixture helper or source replay.
    s = screen_spec(
        {
            "T": 3,
            "actions": 2,
            "n_spokes": 1,
            "capacity": 20,
            "capacity_support": [3, 4],
        }
    )
    c = compile_market(
        replace(
            s,
            demand=replace(
                s.demand,
                mode="poisson",
                arrival_probability=None,
                market_weights=None,
                intensity={
                    x.id: PeriodSchedule.parse({"constant": 2.0}) for x in s.markets
                },
            ),
        )
    )
    start = datetime(2027, 2, 1, tzinfo=timezone.utc)
    clock = UtcBookingClock(tuple(start + timedelta(hours=12 * t) for t in range(4)))
    departures = tuple(
        (x.id, start + timedelta(hours=84 if x.id == "B_OH0" else 108))
        for x in c.spec.legs
    )
    scenario = replace(
        compile_dated_demand(c, clock, departures),
        physical_shocks=(
            B1DatedShock(
                "new_attempt_exit", "rival_exit", "B_OH0", 3, rival_carrier="B"
            ),
        ),
    )
    target = c.product_index["B_OH0"]
    market = int(c.product_market[target])
    keys = ((0, 0, market), (0, 1, market))
    deliveries = (
        tuple(
            DeclaredInputDelivery(
                keys[0],
                1,
                "interline",
                f"update-{i}",
                "notice",
                "update",
                "new-declared-source",
                TIMING,
            )
            for i in range(2 if extra else 1)
        )
        if notice
        else ()
    )
    e = NativeCausalInformationMarketEngine(
        scenario,
        causal_fields=(DisclosureField("notice", "text", "declared update", "text"),),
        causal_known_types=tuple(
            DeclaredKnownType(
                k,
                "declared-fixed" if i == 0 else None,
                "explicit unit label",
                "new-type",
            )
            for i, k in enumerate(keys)
        ),
        causal_deliveries=deliveries,
        causal_definition="NEW attempt source",
        causal_unobserved_definition="no automatic knowledge",
        dated_rebooking=DatedRebookingPolicy(72, 20, 100, interline=enabled),
    )
    tape = ExogenousTape.generate(
        c,
        31,
        registry=StreamRegistry(
            88974, f"unit/V9_shared_mark_attempt/{notice}/{enabled}/{extra}"
        ),
    )
    assert tape.gumbel.shape[1] >= 2
    arr = (ArrivalMark(0.25, market, 0, 0, 0), ArrivalMark(0.75, market, 0, 0, 0))
    g = tape.gumbel.copy()
    g[:] = -1e9
    g[:, :, -1] = 0
    g[:, :, target] = 1e9
    g[:, 1, -1] = 2e9
    g.setflags(write=False)
    tape = replace(tape, arrival_marks=(arr, (), ()), gumbel=g)
    e.reset(tape=tape)
    actions = {
        x.code: PriceAction({j: 200 if i == 0 else 300 for j in c.products_of[i]})
        for i, x in enumerate(c.spec.carriers)
    }
    for t in range(3):
        if t == 1:
            for person in e._passengers.values():
                e.present_cancelled_itinerary(person.original, airport_counter=True)
        e.step(actions)
    rows = e.causal_periods()[1].inputs
    own = rows[0]
    inter = rows[1] if enabled else None
    ownmenu = tuple(
        m.AttemptOffer(x["product_id"], x["fare"], x["offered"], False, False)
        for x in own.computation_input["offers"]
    )
    intermenu = tuple(
        m.AttemptOffer(x["product_id"], x["fare"], x["offered"], False, False)
        for x in (
            inter.computation_input["offers"]
            if inter
            else own.computation_input["offers"]
        )
    )
    count = len(ownmenu) + 1
    outside = tuple([F()] * (count - 1) + [F(1)])
    buy = list(outside)
    if inter:
        j = next(i for i, x in enumerate(intermenu) if x.offered)
        buy[j] = F(1)
        buy[-1] = F()
    kernel = (((F(1, 3), F(2, 3), F()), (F(), F(), F())),) * 3
    # Each physical/mark gets a full joint next-state/signal kernel, same mark carried.
    value = m.SharedMarkAttempt(
        ("good", "bad"),
        ("a", "b", "zero"),
        "declared-fixed",
        causal_decision_view(own.receipt),
        "explicit full pre-attempt conditional joint",
        ownmenu,
        intermenu,
        False,
        enabled,
        (None, "update", "unreachable"),
        "notice",
        ((F(3, 10), F(1, 10), F()), (F(1, 10), F(1, 2), F())),
        (outside,) * 3,
        (kernel, kernel),
        ((tuple(buy), outside, outside),) * 3,
        "declared programme; not inferred native law",
        "new-native-unit",
    )
    return e, value, own, inter, native_causal_information_record(e)


@pytest.mark.parametrize(
    "notice,enabled", [(True, True), (False, True), (False, False)]
)
def test_new_genuine_native_prefix_binding_privacy_and_no_producer_replay(
    notice, enabled
):
    e, value, own, inter, source = fresh_source(notice=notice, enabled=enabled)
    snapshot = pickle.dumps(e.snapshot())
    rng = pickle.dumps(np.random.get_state())
    bound = m.bind_shared_mark_attempt(value, own, inter, source, row_limit=4000000)
    assert not bound["view"]["own_declared_visible_menu"]
    assert (
        bound["view"]["interline"] is None
        if not enabled
        else bound["view"]["interline"]["current_stage"] == "interline"
    )
    assert (
        bound["analyst"]["interline_cell"] is None
        if not enabled
        else bound["analyst"]["interline_cell"]["evidence"] > 0
    )
    text = json.dumps(_stage_encode(jsonable(bound["view"])))
    for private in [
        "physical_states",
        "private_marks",
        "posterior",
        "analyst_root",
        "gumbel",
        "scientific_budget",
    ]:
        assert private not in text
    assert (
        pickle.dumps(e.snapshot()) == snapshot
        and pickle.dumps(np.random.get_state()) == rng
    )
    assert (
        len(
            bound["analyst"]["source"]["parent_source"]["source_context"][
                "ordinary_information"
            ][0]["rows"]
        )
        == 2
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "menu",
        "missing_stage",
        "global_prefix",
        "parent_view",
        "conditioning",
        "known_type",
    ],
)
def test_new_full_native_source_binding_refusals(mutation):
    _e, value, own, inter, source = fresh_source()
    if mutation == "menu":
        value = replace(
            value, own_menu=(replace(value.own_menu[0], fare=99.0), *value.own_menu[1:])
        )
    elif mutation == "missing_stage":
        inter = None
    elif mutation == "global_prefix":
        source["journal"]["analyst_context"]["events"] = source["journal"][
            "analyst_context"
        ]["events"][1:]
    elif mutation == "parent_view":
        source["parent_source"]["periods"][1]["inputs"][0]["view"]["stage"] = "changed"
    elif mutation == "conditioning":
        value = replace(value, conditioning_view={"omitted": "all earlier history"})
    elif mutation == "known_type":
        value = replace(value, known_type="changed")
    with pytest.raises((ValueError, TypeError)):
        m.bind_shared_mark_attempt(value, own, inter, source, row_limit=4000000)


def test_new_extra_own_received_record_never_filtered():
    _e, value, own, inter, source = fresh_source(extra=True)
    with pytest.raises(ValueError, match="no omission"):
        m.bind_shared_mark_attempt(value, own, inter, source, row_limit=4000000)


def test_new_declared_visibility_explicit_and_source_menu_preserved():
    _e, value, own, inter, source = fresh_source()
    value = replace(
        value,
        product_catalogue_disclosed=True,
        own_menu=tuple(
            replace(x, disclose_offer=True, disclose_fare=False) for x in value.own_menu
        ),
    )
    bound = m.bind_shared_mark_attempt(value, own, inter, source, row_limit=4000000)
    assert all(x["fare"] is None for x in bound["view"]["own_declared_visible_menu"])
    assert all(
        x["offered"] is False for x in bound["view"]["own_declared_visible_menu"]
    )
    assert (
        bound["analyst"]["model_result"].model.own_menu[0].fare
        == own.computation_input["offers"][0]["fare"]
    )


@pytest.mark.parametrize(
    "mutation", ["outside_person", "closing_boundary", "core_contract", "row_flag"]
)
def test_new_complete_parent_source_refusals(mutation):
    _e, value, own, inter, source = fresh_source()
    if mutation == "outside_person":
        source["parent_source"]["source_context"]["ordinary_information"][0]["rows"] = (
            source["parent_source"]["source_context"]["ordinary_information"][0][
                "rows"
            ][:-1]
        )
    elif mutation == "closing_boundary":
        source["periods"][0]["native_world_t"] = 7
    elif mutation == "core_contract":
        source["journal"]["analyst_context"]["definition"] = "changed"
    elif mutation == "row_flag":
        source["native_decision_input_rows_captured"] = False
    with pytest.raises(ValueError):
        m.bind_shared_mark_attempt(value, own, inter, source, row_limit=4000000)


def test_new_binding_full_proposed_output_limit_precedes_forward(monkeypatch):
    _e, value, own, inter, source = fresh_source()
    original = m._limit
    seen = []

    def counted(model, shape, row_limit, extra=None):
        if isinstance(extra, dict) and "complete_view" in extra:
            raise m.SharedMarkLimit("new full proposed binding bound")
        return original(model, shape, row_limit, extra)

    def fail(*args):
        seen.append(True)
        raise AssertionError("no forward permitted")

    monkeypatch.setattr(m, "_limit", counted)
    monkeypatch.setattr(m, "_compile", fail)
    with pytest.raises(m.SharedMarkLimit):
        m.bind_shared_mark_attempt(value, own, inter, source, row_limit=4000000)
    assert not seen


def test_new_actual_zero_likelihood_signal_keeps_null_belief_no_prior_fill():
    _e, value, own, inter, source = fresh_source()
    deterministic = ((F(1), F(), F()), (F(), F(), F()))
    value = replace(value, next_state_signal=((deterministic,) * 3,) * 2)
    bound = m.bind_shared_mark_attempt(value, own, inter, source, row_limit=4000000)
    cell = bound["analyst"]["interline_cell"]
    assert bound["analyst"]["reached_signal"] == "update"
    assert cell["evidence"] == 0 and all(
        c["posterior"] is None for c in cell["components"]
    )
    assert bound["analyst"]["source"]["native_parent_steps_completed"] == 3


@pytest.mark.parametrize("field", ["pending_inputs", "pending_results"])
def test_new_parent_pending_cannot_be_hidden_by_complete_top_level(field):
    _e, value, own, inter, source = fresh_source()
    source["parent_source"][field] = ({"incomplete": "retained"},)
    with pytest.raises(ValueError, match="incomplete parent stage capture"):
        m.bind_shared_mark_attempt(value, own, inter, source, row_limit=4000000)


@pytest.mark.parametrize("field", ["conditioning_view", "own_menu"])
def test_new_opaque_array_model_container_refused(field):
    value = model()
    data = (
        np.array(value.own_menu, dtype=object)
        if field == "own_menu"
        else {"opaque": np.array([1])}
    )
    with pytest.raises(TypeError, match="owned primitive"):
        replace(value, **{field: data})


def test_new_opaque_array_source_context_refused_without_mutation():
    _e, value, own, inter, source = fresh_source()
    source["extra_opaque_context"] = np.array([1])
    before = pickle.dumps(source)
    with pytest.raises(TypeError, match="owned primitive"):
        m.bind_shared_mark_attempt(value, own, inter, source, row_limit=4000000)
    assert pickle.dumps(source) == before


def test_new_nested_primitive_conditioning_context_owned():
    context = {"full": ["earlier", {"received": [-0.0, F(1, 7), None]}]}
    value = replace(model(), conditioning_view=context)
    before = pickle.dumps(value)
    context["full"][1]["received"].append("later")
    result = compile_new(value)
    assert pickle.dumps(value) == before
    assert len(result.model.conditioning_view["full"][1]["received"]) == 3
