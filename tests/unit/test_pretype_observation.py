"""A.10/A.5/A.14/B0/V9: NEW supplied-source/pre-category binding units."""

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
from airline_market.sim.causal_information import causal_decision_view
from airline_market.sim.latent_type_attempt import TypeObservationAttempt
from airline_market.sim.native_causal_information import (
    TIMING,
    DeclaredInputDelivery,
    DeclaredKnownType,
    NativeCausalInformationMarketEngine,
    native_causal_information_record,
)
from airline_market.sim.received_information import DisclosureField
from airline_market.sim.shared_mark_attempt import AttemptOffer
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

if os.environ.get("PRETYPE_SOURCE_DIR"):
    spec = importlib.util.spec_from_file_location(
        "airline_market.sim.pretype_observation",
        Path(os.environ["PRETYPE_SOURCE_DIR"]) / "pretype_observation.py",
    )
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
from airline_market.sim import pretype_observation as binding

PretypeObservation = binding.PretypeObservation
PretypeObservationLimit = binding.PretypeObservationLimit
bind_pretype_observation = binding.bind_pretype_observation
pretype_binding_record = binding.pretype_binding_record
pretype_binding_view = binding.pretype_binding_view
pretype_history_view = binding.pretype_history_view

LIMIT = 10**9  # Structural unit scope, not bytes/RSS/scientific budget.


def make_native(interline_enabled=True):
    # NEW text-adapted constructor; no old fixture/helper/bank invocation.
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
                    m.id: PeriodSchedule.parse({"constant": 2.0}) for m in spec.markets
                },
            ),
        )
    )
    start = datetime(2027, 3, 8, tzinfo=timezone.utc)
    clock = UtcBookingClock(tuple(start + timedelta(hours=12 * i) for i in range(4)))
    departures = tuple(
        (leg.id, start + timedelta(hours=84 if leg.id == "B_OH0" else 108))
        for leg in compiled.spec.legs
    )
    scenario = replace(
        compile_dated_demand(compiled, clock, departures),
        physical_shocks=(
            B1DatedShock(
                "pretype_observation_exit", "rival_exit", "B_OH0", 3, rival_carrier="B"
            ),
        ),
    )
    tape = ExogenousTape.generate(
        compiled, 31, registry=StreamRegistry(93974, "unit/V9_pretype_observation")
    )
    target = compiled.product_index["B_OH0"]
    market = int(compiled.product_market[target])
    arrivals = tuple(
        ArrivalMark(
            time=(i + 1) / 3, market=market, segment=0, thinning=0, class_uniform=0.4
        )
        for i in range(2)
    )
    gumbel = tape.gumbel.copy()
    gumbel[:] = -1e9
    gumbel[:, :, -1] = 0
    gumbel[:, :, target] = 1e9
    gumbel[:, 1, -1] = 2e9
    gumbel.setflags(write=False)
    tape = replace(tape, arrival_marks=(arrivals, (), ()), gumbel=gumbel)
    key = (0, 0, market)
    engine = NativeCausalInformationMarketEngine(
        scenario,
        dated_rebooking=DatedRebookingPolicy(72, 20, 100, interline=interline_enabled),
        causal_fields=(
            DisclosureField(
                "category", "text", "explicit NEW declared category", "text"
            ),
        ),
        causal_known_types=tuple(
            DeclaredKnownType(
                (0, slot, market),
                None,
                "unsupplied label, not ignorance",
                "new-source-93974",
            )
            for slot in range(2)
        ),
        causal_deliveries=(
            DeclaredInputDelivery(
                key,
                1,
                "own",
                "category-93974",
                "category",
                "red",
                "declared-category-93974",
                TIMING,
            ),
        ),
        causal_definition="NEW complete pre-category binding fixture",
        causal_unobserved_definition="no authenticated knowledge/type/probability",
    )
    engine.reset(tape=tape)
    actions = {
        carrier.code: PriceAction(
            {j: 220 if i == 0 else 320 for j in compiled.products_of[i]}
        )
        for i, carrier in enumerate(compiled.spec.carriers)
    }
    engine.step(actions)
    baseline = engine._causal_journal
    engine.step(actions)
    engine.step(actions)
    rows = tuple(
        row
        for period in engine._causal_periods
        for row in period.inputs
        if row.kind != "ordinary"
    )
    own = next(
        row
        for row in rows
        if row.kind == "own"
        and row.receipt.source_journal.events[-1].position.anchor_utc == clock.at(1)
    )
    interline = next(
        (
            row
            for row in rows
            if row.kind == "interline"
            and row.receipt.source_journal.events[-1].position.anchor_utc == clock.at(1)
        ),
        None,
    )
    return engine, baseline, own, interline, native_causal_information_record(engine)


@pytest.fixture(scope="module")
def native():
    return make_native()


@pytest.fixture(scope="module")
def disabled_native():
    return make_native(False)


def model_for(native, q=F(2, 5), *, visible=False):
    _, baseline, own, interline, _ = native
    menus = tuple(
        tuple(
            AttemptOffer(x["product_id"], x["fare"], x["offered"], visible, visible)
            for x in row.computation_input["offers"]
        )
        for row in (own, own if interline is None else interline)
    )
    own_menu, other_menu = menus
    p = len(own_menu) + 1
    target = next((i for i, x in enumerate(other_menu) if x.offered), None)
    own_program = tuple(
        tuple(tuple(F(action == p - 1) for action in range(p)) for _ in range(2))
        for _ in range(3)
    )
    interline_program = tuple(
        tuple(
            tuple(
                tuple(
                    (F(1, 4) if g == 0 else F(3, 4))
                    if a == target
                    else (
                        (F(3, 4) if g == 0 else F(1, 4)) if target is not None else F(1)
                    )
                    if a == p - 1
                    else F()
                    for a in range(p)
                )
                for g in range(2)
            )
            for _ in range(2)
        )
        for _ in range(3)
    )
    type_weights = (q / 3, 2 * q / 3, 1 - q)
    mark_weights = ((F(1, 4), F(3, 4)), (F(3, 4), F(1, 4)), (F(1, 2), F(1, 2)))
    initial = tuple(
        tuple(
            tuple(px * type_weights[k] * mark_weights[k][g] for g in range(2))
            for k in range(3)
        )
        for px in (F(2, 5), F(3, 5))
    )
    kernel = tuple(
        tuple(
            tuple(
                tuple(
                    tuple(F(nxt == 1 - x and sig == 0) for sig in range(2))
                    for nxt in range(2)
                )
                for _ in range(2)
            )
            for _ in range(3)
        )
        for x in range(2)
    )
    return TypeObservationAttempt(
        physical_states=("intact", "disrupted"),
        latent_types=("tau0", "tau1", "tau2"),
        private_marks=("g0", "g1"),
        observations=("red", "grey", "unused"),
        type_observation=("red", "red", "grey"),
        signals=(None, "unreached"),
        own_menus=(own_menu,) * 3,
        interline_menus=((other_menu, other_menu),) * 3,
        product_catalogue_disclosed=visible,
        interline_enabled=interline is not None,
        initial_joint=initial,
        own_program=own_program,
        next_state_signal=kernel,
        interline_program=interline_program,
        conditioning_view=pretype_history_view(baseline, own.analyst_original_key),
        conditioning_definition="explicit full joint conditional BEFORE category, not current-input mixture",
        observation_definition="declared deterministic latent category, not actual private class",
        private_mark_interpretation="finite declared SAME mark, not authenticated native Gumbels",
        definition="NEW native-associated declared law 93974",
        source_ref="constructed-law-93974",
    )


def observation(**changes):
    return replace(
        PretypeObservation(
            "category-93974",
            "category",
            "red",
            "pre-observation-full-joint",
            "explicit category mapping declaration",
            "declared-category-93974",
        ),
        **changes,
    )


def bind(native, model=None, **changes):
    _, baseline, own, interline, source = native
    args = {
        "model": model_for(native) if model is None else model,
        "baseline": baseline,
        "own_input": own,
        "interline_input": interline,
        "source_record": source,
        "observation": observation(),
        "row_limit": LIMIT,
    }
    args.update(changes)
    return bind_pretype_observation(**args)


@pytest.mark.parametrize("q", [F(2, 5), F(1, 10**600), F()])
def test_joint_selection_complete_zero_tiny_and_native_identity(native, q):
    engine, baseline, own, later, _ = native
    snapshot = pickle.dumps(engine.snapshot())
    rng = pickle.dumps(np.random.get_state())
    result = bind(native, model_for(native, q))
    assert result.analyst["selected_category_cell"]["evidence"] == q
    probabilities = result.analyst["interline_action_probabilities"]
    target = next(
        i for i, x in enumerate(later.computation_input["offers"]) if x["offered"]
    )
    assert probabilities is None if not q else probabilities[target] == F(11, 24)
    assert (
        all(
            c["posterior"] is None
            for c in result.analyst["selected_category_cell"]["components"]
        )
        if not q
        else sum(
            c["posterior"]
            for c in result.analyst["selected_category_cell"]["components"]
        )
        == 1
    )
    assert (
        len(baseline.events) == 7
        and len(own.receipt.source_journal.events) == 11
        and len(later.receipt.source_journal.events) == 13
    )
    assert result.analyst["compiled"].report["terminal_mass"] == 1
    assert result.analyst["compiled"].report["source_receipt_bridge_executed"] is False
    assert result.analyst["source_binding_executed"] is True
    assert pickle.dumps(engine.snapshot()) == snapshot
    assert pickle.dumps(np.random.get_state()) == rng


def test_exact_full_path_reference_and_no_current_result_conditioning(native):
    result = bind(native)
    model = result.analyst["model"]
    selected = F()
    for x in range(2):
        for k in range(3):
            for g in range(2):
                if model.type_observation[k] == "red":
                    selected += model.initial_joint[x][k][g] * (
                        F(1, 4) if g == 0 else F(3, 4)
                    )
    assert selected / F(2, 5) == F(11, 24)
    # The pre-choice result integrates all actions, not the actually chosen one.
    assert len([p for p in result.analyst["interline_action_probabilities"] if p]) == 2
    cells = result.analyst["compiled"].report["history_cells"]
    assert any(c["evidence"] == 0 for c in cells)
    assert all(
        c["posterior"] is None
        for cell in cells
        if not cell["evidence"]
        for c in cell["components"]
    )


def test_visible_view_privacy_and_owned_full_source(native):
    model = model_for(native, visible=True)
    source = copy.deepcopy(native[-1])
    result = bind(native, model, source_record=source)
    source["periods"] = ()
    exported = pretype_binding_record(result)
    exported["analyst"] = {}
    view = pretype_binding_view(result)
    text = str(view)
    for excluded in (
        "latent_types",
        "posterior",
        "evidence",
        "analyst_root",
        "person_key",
        "source_ref",
        "native_spec",
        "current_result",
    ):
        assert excluded not in text
    assert view["own_declared_visible_menu"]
    assert view["before_category"]["known_type"]["value"] is None
    assert len(result.analyst["source"]["periods"]) == 3
    assert pretype_binding_view(bind(native))["own_declared_visible_menu"] == ()


@pytest.mark.parametrize(
    "field,value",
    [
        ("delivery_id", "wrong"),
        ("field", "wrong"),
        ("category", "grey"),
        ("prior_scope", "current-own-input"),
        ("source_ref", "wrong"),
    ],
)
def test_observation_contract_refusals(native, field, value):
    with pytest.raises(ValueError):
        bind(native, observation=observation(**{field: value}))


def test_current_input_cannot_be_prior_or_baseline(native):
    own = native[2]
    with pytest.raises(ValueError, match="BEFORE"):
        bind(
            native,
            replace(
                model_for(native), conditioning_view=causal_decision_view(own.receipt)
            ),
        )
    with pytest.raises(ValueError):
        bind(native, baseline=own.receipt.source_journal)
    with pytest.raises(ValueError):
        bind(native, baseline=replace(native[1], events=native[1].events[:-1]))


@pytest.mark.parametrize(
    "change",
    ["pending", "failed", "period", "outside", "stage", "journal", "model", "parent"],
)
def test_complete_global_source_and_all_people_refusals(native, change):
    source = copy.deepcopy(native[-1])
    if change == "pending":
        source["pending"] = {}
    elif change == "failed":
        source["failed"] = True
    elif change == "period":
        source["periods"] = source["periods"][:-1]
    elif change == "outside":
        source["periods"][0]["inputs"] = source["periods"][0]["inputs"][:-1]
    elif change == "stage":
        source["periods"][1]["inputs"] = source["periods"][1]["inputs"][:-1]
    elif change == "journal":
        source["journal"]["analyst_context"]["events"] = source["journal"][
            "analyst_context"
        ]["events"][:-1]
    elif change == "model":
        source["parent_source"]["source_context"]["spec_hash"] = "wrong"
    else:
        source["parent_source"]["pending_inputs"] = ()
    with pytest.raises((ValueError, KeyError, IndexError)):
        bind(native, source_record=source)


def test_reached_interline_cannot_be_omitted_or_relinked(native):
    with pytest.raises(ValueError, match="ALL and ONLY"):
        bind(native, interline_input=None)
    with pytest.raises(ValueError):
        bind(native, interline_input=replace(native[3], analyst_root="wrong"))
    with pytest.raises(ValueError):
        bind(
            native,
            own_input=replace(
                native[2],
                analyst_original_key=(0, 1, native[2].analyst_original_key[2]),
            ),
        )


def test_exact_native_menu_and_interline_contract(native):
    model = model_for(native)
    first = model.own_menus[0]
    changed = (replace(first[0], fare=first[0].fare + 1), *first[1:])
    with pytest.raises(ValueError, match="menu"):
        bind(native, replace(model, own_menus=(changed, *model.own_menus[1:])))
    with pytest.raises(ValueError, match="interline"):
        bind(native, replace(model, interline_enabled=False))


def test_limit_and_refusals_precede_compiler(native, monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise AssertionError("compiler must not run on refused inputs")

    monkeypatch.setattr(binding, "compile_type_observation_attempt", fail)
    with pytest.raises(PretypeObservationLimit):
        bind(native, row_limit=1)
    with pytest.raises(ValueError):
        bind(native, observation=observation(prior_scope="wrong"))
    assert calls == []


def test_one_delegation_and_owned_backend_failure(native, monkeypatch):
    original = binding.compile_type_observation_attempt
    calls = []

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(binding, "compile_type_observation_attempt", spy)
    result = bind(native)
    assert calls == [1]
    required = result.analyst["structural_bound"]
    with pytest.raises(PretypeObservationLimit):
        bind(native, row_limit=required - 1)
    source = pickle.dumps(native[-1])

    def fail(*args, **kwargs):
        raise RuntimeError("NEW injected pure failure")

    monkeypatch.setattr(binding, "compile_type_observation_attempt", fail)
    with pytest.raises(RuntimeError, match="injected"):
        bind(native)
    assert pickle.dumps(native[-1]) == source


@pytest.mark.parametrize("limit", [True, 0, -1, 1.0])
def test_typed_limits(native, limit):
    with pytest.raises(TypeError):
        bind(native, row_limit=limit)


def test_opaque_source_and_unregistered_baseline_refused(native):
    with pytest.raises(TypeError):
        bind(native, source_record={"opaque": np.zeros(1)})
    with pytest.raises(ValueError):
        pretype_history_view(native[1], (0, 99, 0))
    with pytest.raises(TypeError):
        bind(native, observation={})


def test_disabled_interline_retains_full_source_and_null_reach(disabled_native):
    assert disabled_native[3] is None
    result = bind(disabled_native)
    assert result.analyst["selected_reach_cell"] is None
    assert result.analyst["interline_action_probabilities"] is None
    assert pretype_binding_view(result)["interline"] is None
    assert result.analyst["compiled"].report["terminal_mass"] == 1
    assert all(
        row["mass"] == 0 for row in result.analyst["compiled"].report["interline_rows"]
    )


def test_later_baseline_and_changed_global_definition_are_refused(native):
    with pytest.raises(ValueError):
        bind(native, baseline=native[0]._causal_journal)
    with pytest.raises(ValueError):
        bind(native, baseline=replace(native[1], definition="different full contract"))


def test_same_attempt_and_message_types_cannot_be_substituted(native):
    own, later = native[2:4]
    journal = later.receipt.source_journal
    event = journal.events[-1]
    changed = replace(event, payload=replace(event.payload, attempt_id="other-attempt"))
    receipt = replace(
        later.receipt,
        source_journal=replace(journal, events=(*journal.events[:-1], changed)),
    )
    with pytest.raises(ValueError):
        bind(native, interline_input=replace(later, receipt=receipt))
    first = own.receipt.source_journal
    event = first.events[-4]
    changed = replace(event, payload=replace(event.payload, value="grey"))
    events = (*first.events[:-4], changed, *first.events[-3:])
    with pytest.raises(ValueError):
        bind(
            native,
            own_input=replace(
                own,
                receipt=replace(
                    own.receipt, source_journal=replace(first, events=events)
                ),
            ),
        )
