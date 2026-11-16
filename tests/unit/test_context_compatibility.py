"""A.10/A.5/A.14/B0/V9/B4: NEW134974 pure relation, NEW native units only."""

import copy
import pickle
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.dated_demand import compile_dated_demand
from airline_market.policy.dated_rescue import DatedRebookingPolicy
from airline_market.policy.dated_reserve import UtcBookingClock
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.sim import context_compatibility as m
from airline_market.sim.causal_information import causal_decision_view
from airline_market.sim.native_causal_information import (
    TIMING,
    DeclaredInputDelivery,
    DeclaredKnownType,
    NativeCausalInformationMarketEngine,
    native_causal_information_record,
)
from airline_market.sim.received_information import DisclosureField
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

LIMIT = 10**8


def new_source(*, count=3, depleted=False, late=False):
    # Locally adapted constructor only; NEW June7/namespace135974 declarations.
    spec = screen_spec(
        {
            "T": 3,
            "actions": 2,
            "n_spokes": 1,
            "capacity": 2 if depleted else 20,
            "capacity_support": [1, 2] if depleted else [3, 4],
        }
    )
    c = compile_market(
        replace(
            spec,
            demand=replace(
                spec.demand,
                mode="poisson",
                arrival_probability=None,
                market_weights=None,
                intensity={
                    x.id: PeriodSchedule.parse({"constant": 2.0}) for x in spec.markets
                },
            ),
        )
    )
    t0 = datetime(2027, 6, 7, tzinfo=timezone.utc)
    clock = UtcBookingClock(tuple(t0 + timedelta(hours=12 * t) for t in range(4)))
    departures = tuple(
        (l.id, t0 + timedelta(hours=(96 if late else 84) if l.id == "B_OH0" else 108))
        for l in c.spec.legs
    )
    scenario = compile_dated_demand(c, clock, departures)
    scenario = replace(
        scenario,
        physical_shocks=(
            B1DatedShock("NEW134974-exit", "rival_exit", "B_OH0", 3, rival_carrier="B"),
        ),
    )
    period = 1 if late else 0
    market = int(c.product_market[c.product_index["B_OH0"]])
    keys = tuple((period, i, market) for i in range(count))
    types = tuple(
        DeclaredKnownType(
            k, None, "explicit unknown declaration, not ignorance", "NEW134974-type"
        )
        for k in keys
    )
    deliveries = tuple(
        DeclaredInputDelivery(
            k,
            t,
            stage,
            f"NEW134974-{i}-{t}-{stage}",
            "advice",
            "declared notice without price display",
            "NEW134974-source",
            TIMING,
        )
        for i, k in enumerate(keys)
        for t in range(period, 3)
        for stage in ("ordinary", "own", "interline")
        if stage != "ordinary" or t == period
    )
    e = NativeCausalInformationMarketEngine(
        scenario,
        causal_fields=(DisclosureField("advice", "text", "declared notice", "text"),),
        causal_known_types=types,
        causal_deliveries=deliveries,
        causal_definition="NEW134974 source order only",
        causal_unobserved_definition="No G/native law/awareness inference",
        causal_row_limit=2000000,
        dated_rebooking=DatedRebookingPolicy(72, 20, 100, interline=True),
    )
    tape = ExogenousTape.generate(
        c,
        31,
        registry=StreamRegistry(
            135974,
            (
                "unit/V9_context_compatibility/depletion-v2/"
                if depleted
                else "unit/V9_context_compatibility/"
            )
            + str((count, depleted, late)),
        ),
    )
    marks = tuple(
        ArrivalMark((i + 1) / (count + 1), market, 0, 0, 0) for i in range(count)
    )
    arrivals = [(), (), ()]
    arrivals[period] = marks
    g = tape.gumbel.copy()
    g[:] = -1e9
    g[:, :, -1] = 0
    g[:, :, c.product_index["B_OH0"]] = 1e9
    if count:
        g[:, count - 1, -1] = 2e9
    g.setflags(write=False)
    tape = replace(tape, arrival_marks=tuple(arrivals), gumbel=g)
    e.reset(tape=tape)
    actions = {
        carrier.code: PriceAction({j: 220 if i == 0 else 320 for j in c.products_of[i]})
        for i, carrier in enumerate(c.spec.carriers)
    }
    for _ in range(3):
        e.step(actions)
    periods = e.causal_periods()
    ordinary = tuple(row for p in periods for row in p.inputs if row.kind == "ordinary")
    return e, native_causal_information_record(e), periods, ordinary


@pytest.fixture(scope="module")
def fresh():
    return new_source()


def plan(*, mode="invariant", weights=(None, F(0), F(1, 10**2600))):
    names = ("left", "zero-annotation", "tiny-annotation")
    actions = (
        ("B_OH0",) * 3
        if mode == "invariant"
        else ("A_OH0", "B_OH0", "B_OH0")
        if mode == "dependent"
        else ("B_OH0", None, "B_OH0")
    )
    return m.CompatibilityPlan(
        tuple(
            m.CompatibleLabel(n, "explicit compatibility; no realized native vector", w)
            for n, w in zip(names, weights)
        ),
        (
            m.ContextAction("A_OH0", "A_OH0", False),
            m.ContextAction("B_OH0", "B_OH0", False),
            m.ContextAction("outside", None, True),
        ),
        tuple(m.LabelAction(n, a) for n, a in zip(names, actions)),
        "NEW134974 supplied descriptive policy",
        "declared plan source",
        "No inferred observation law or mark awareness",
    )


def bind(fresh, *, selected=0, p=None, limit=LIMIT):
    _, source, periods, ordinary = fresh
    return m.bind_context_compatibility(
        source, periods, ordinary[selected], p or plan(), support_limit=limit
    )


def test_new_genuine_earlier_input_and_all_completed_empty_periods(fresh):
    result = bind(fresh)
    assert (
        result.selected_input.receipt.source_journal.events[-1].payload.stage
        == "ordinary"
    )
    assert len(result.periods) == 3 and not result.periods[-1].inputs
    assert result.common_action == "B_OH0"
    assert m.context_compatibility_view(result) == causal_decision_view(
        fresh[3][0].receipt
    )
    assert (
        result.realized_G is None
        and result.probability_law is None
        and not result.native_G_authenticated
    )
    assert len(result.plan.labels) == 3 and result.plan.labels[1].weight_annotation == 0
    assert result.plan.labels[2].weight_annotation == F(1, 10**2600)


def test_identical_H_menu_different_results_and_separate_comparison(fresh):
    a, b = bind(fresh, selected=1), bind(fresh, selected=2)
    assert a.earlier_view == b.earlier_view
    assert a.selected_input.computation_input == b.selected_input.computation_input
    first = m.compare_context_action(a, support_limit=LIMIT)
    second = m.compare_context_action(b, support_limit=LIMIT)
    assert first.agrees is True and second.agrees is False
    assert (first.observed_action, second.observed_action) == ("B_OH0", "outside")
    assert first.agreement_probability is second.agreement_probability is None


@pytest.mark.parametrize("mode", ["dependent", "missing"])
def test_all_compatible_labels_even_zero_or_unknown_keep_unresolved(fresh, mode):
    r = bind(fresh, p=plan(mode=mode))
    assert r.common_action is None
    assert m.compare_context_action(r, support_limit=LIMIT).agrees is None
    assert len(r.plan.policy) == 3


def test_no_weight_normalization_or_uniform_fill(fresh):
    p = plan(weights=(F(4), None, F(0)))
    r = bind(fresh, p=p)
    assert tuple(x.weight_annotation for x in r.plan.labels) == (F(4), None, F(0))
    assert r.common_action == "B_OH0" and r.probability_law is None


def test_forward_view_excludes_every_analyst_and_model_axis(fresh):
    r = bind(fresh)
    view = m.context_compatibility_view(r)
    assert set(view) == set(causal_decision_view(fresh[3][0].receipt))
    text = repr(view)
    for term in (
        "analyst_original_key",
        "parent_input_index",
        "compatible_labels",
        "weight_annotation",
        "common_action",
        "source_record",
        "computation_input",
        "native_G_authenticated",
        "binary64_bits",
    ):
        assert term not in text
    assert "current_stage" in view and view["known_type"]["value"] is None


def test_full_input_output_bounds_before_classify(fresh, monkeypatch):
    r = bind(fresh)
    assert (
        m._count(
            {
                "inputs": (r.source_record, r.periods, r.selected_input, r.plan),
                "output": r,
            }
        )
        <= r.required_nodes
    )
    assert bind(fresh, limit=r.required_nodes).required_nodes == r.required_nodes

    def forbidden(*args):
        pytest.fail("classify must follow full support check")

    monkeypatch.setattr(m, "_classify", forbidden)
    with pytest.raises(ValueError, match="exceeds"):
        bind(fresh, limit=r.required_nodes - 1)


def test_comparator_never_reclassifies_policy(fresh, monkeypatch):
    r = bind(fresh)

    def forbidden(*args):
        pytest.fail("separate comparison must not classify labels again")

    monkeypatch.setattr(m, "_classify", forbidden)
    assert m.compare_context_action(r, support_limit=LIMIT).agrees is True
    expected = m.compare_context_action(r, support_limit=LIMIT).required_nodes
    with pytest.raises(ValueError, match="exceeds"):
        m.compare_context_action(r, support_limit=expected - 1)


def test_owned_sources_detached_exports_native_and_global_rng_unchanged(fresh):
    e, source, periods, rows = fresh
    before = pickle.dumps(e.snapshot())
    rng = pickle.dumps(np.random.get_state())
    mutable = copy.deepcopy(source)
    p = plan()
    label_list = list(p.labels)
    r = m.bind_context_compatibility(
        mutable,
        list(periods),
        rows[0],
        replace(p, labels=label_list),
        support_limit=LIMIT,
    )
    mutable["periods"] = ()
    label_list.clear()
    assert (
        len(r.periods) == 3
        and len(r.plan.labels) == 3
        and len(r.source_record["periods"]) == 3
    )
    exported = r.export()
    exported["source_record"]["periods"] = ()
    view = m.context_compatibility_view(r)
    view["earlier_person_history"] = ()
    assert (
        len(r.source_record["periods"]) == 3
        and r.earlier_view["earlier_person_history"]
    )
    with pytest.raises(FrozenInstanceError):
        r.common_action = "outside"
    assert (
        pickle.dumps(e.snapshot()) == before
        and pickle.dumps(np.random.get_state()) == rng
    )


def test_unexpected_pure_failure_preserves_inputs(fresh, monkeypatch):
    before = pickle.dumps(fresh[1:])

    def crash(*args):
        raise RuntimeError("NEW134974 injected pure failure")

    monkeypatch.setattr(m, "_classify", crash)
    with pytest.raises(RuntimeError, match="injected"):
        bind(fresh)
    assert pickle.dumps(fresh[1:]) == before


def test_new_late_original_input_with_earlier_empty_and_boundary_cancellation():
    late = new_source(late=True)
    r = bind(late)
    assert not r.periods[0].inputs and r.selected_input.analyst_original_key[0] == 1
    assert any(
        type(e.payload).__name__ == "OwnAction" and e.payload.kind == "cancellation"
        for e in r.periods[1].journal.events
    )


def test_new_depleted_menu_refused_without_dropping_product():
    depleted = new_source(depleted=True)
    assert any(not o["offered"] for o in depleted[3][-1].computation_input["offers"])
    with pytest.raises(ValueError, match="ALL mapped"):
        bind(depleted, selected=-1)


def test_new_empty_source_has_no_invented_individual():
    empty = new_source(count=0)
    assert empty[3] == () and all(not p.inputs for p in empty[2])
    with pytest.raises(TypeError):
        m.bind_context_compatibility(
            empty[1], empty[2], None, plan(), support_limit=LIMIT
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("rule", "old"),
        ("required_nodes", True),
        ("required_nodes", 1),
        ("input_digest", "bad"),
        ("output_digest", "bad"),
        ("common_action", "outside"),
        ("realized_G", "guessed"),
        ("probability_law", F(1)),
        ("native_G_authenticated", True),
        ("native_knowledge_preferences_CS_adopted", True),
    ],
)
def test_stored_integrity_no_false_positive_binding(fresh, field, value):
    with pytest.raises((ValueError, TypeError)):
        m.context_compatibility_view(replace(bind(fresh), **{field: value}))


def test_forged_earlier_result_leak_even_with_recomputed_digest_refused(fresh):
    r = bind(fresh)
    leak = m._encoded(r.earlier_view)
    leak["current_result"] = "B_OH0"
    forged = replace(
        r,
        earlier_view=leak,
        output_digest=m._digest(m._encoded((r.common_action, leak, m.RULE))),
    )
    with pytest.raises(ValueError, match="ONLY exact earlier"):
        m.context_compatibility_view(forged)


@pytest.mark.parametrize(
    "bad",
    [
        "missing_label",
        "duplicate_label",
        "negative_weight",
        "float_weight",
        "unknown_policy",
        "missing_policy",
        "duplicate_policy",
        "missing_outside",
        "bool_outside",
        "duplicate_product",
        "missing_product",
        "foreign_product",
        "unknown_weight_scope",
    ],
)
def test_complete_plan_refusals(fresh, bad):
    p = plan()
    if bad == "missing_label":
        p = replace(p, labels=p.labels[:-1])
    elif bad == "duplicate_label":
        p = replace(p, labels=p.labels + (p.labels[0],))
    elif bad == "negative_weight":
        p = replace(
            p, labels=(replace(p.labels[0], weight_annotation=F(-1)),) + p.labels[1:]
        )
    elif bad == "float_weight":
        p = replace(
            p, labels=(replace(p.labels[0], weight_annotation=0.1),) + p.labels[1:]
        )
    elif bad == "unknown_policy":
        p = replace(p, policy=(replace(p.policy[0], action="hidden-X"),) + p.policy[1:])
    elif bad == "missing_policy":
        p = replace(p, policy=p.policy[:-1])
    elif bad == "duplicate_policy":
        p = replace(p, policy=p.policy + (p.policy[0],))
    elif bad == "missing_outside":
        p = replace(p, actions=p.actions[:-1])
    elif bad == "bool_outside":
        p = replace(p, actions=(replace(p.actions[0], outside=1),) + p.actions[1:])
    elif bad == "duplicate_product":
        p = replace(
            p, actions=(replace(p.actions[0], product_id="B_OH0"),) + p.actions[1:]
        )
    elif bad == "missing_product":
        p = replace(p, actions=p.actions[1:])
    elif bad == "foreign_product":
        p = replace(
            p, actions=(replace(p.actions[0], product_id="A_HD0"),) + p.actions[1:]
        )
    elif bad == "unknown_weight_scope":
        p = replace(p, weight_scope="iid uniform prior")
    with pytest.raises((ValueError, TypeError)):
        bind(fresh, p=p)


@pytest.mark.parametrize(
    "bad",
    [
        "missing_period",
        "incomplete",
        "wrong_key",
        "later_receipt",
        "changed_menu",
        "bool_index",
        "missing_person",
        "future_event_in_close",
        "unsupported_stage",
    ],
)
def test_complete_source_and_exact_prefix_refusals(fresh, bad):
    _, raw, periods, rows = fresh
    source = copy.deepcopy(raw)
    selected = rows[0]
    if bad == "missing_period":
        periods = periods[:-1]
    elif bad == "incomplete":
        source["failed"] = True
    elif bad == "wrong_key":
        selected = replace(selected, analyst_original_key=(0, 99, 0))
    elif bad == "later_receipt":
        selected = replace(
            selected,
            receipt=replace(selected.receipt, source_journal=periods[-1].journal),
        )
    elif bad == "changed_menu":
        menu = m._encoded(selected.computation_input)
        menu["outside_available"] = False
        selected = replace(selected, computation_input=menu)
    elif bad == "bool_index":
        selected = replace(selected, parent_input_index=True)
        periods = (
            replace(periods[0], inputs=(selected,) + periods[0].inputs[1:]),
        ) + periods[1:]
    elif bad == "missing_person":
        source["parent_source"]["source_context"]["ordinary_information"][0]["rows"] = (
            source["parent_source"]["source_context"]["ordinary_information"][0][
                "rows"
            ][:-1]
        )
    elif bad == "future_event_in_close":
        periods = (replace(periods[0], journal=periods[1].journal),) + periods[1:]
    elif bad == "unsupported_stage":
        selected = next(r for p in periods for r in p.inputs if r.kind == "own")
    with pytest.raises((ValueError, TypeError, KeyError)):
        m.bind_context_compatibility(
            source, periods, selected, plan(), support_limit=LIMIT
        )


def test_new_earlier_input_extension_preserves_frozen_last_period_scope(fresh):
    from airline_market.sim.booking_loss_binding import _source

    with pytest.raises(ValueError, match="LAST closed period"):
        _source(fresh[1], fresh[2], fresh[3][0])
    assert (
        bind(fresh).selected_input.analyst_original_key
        == fresh[3][0].analyst_original_key
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("failed", 0),
        ("native_parent_steps_completed", True),
        ("native_decision_input_rows_captured", 1),
    ],
)
def test_exact_source_stamps_refuse_bool_integer_alias(fresh, field, value):
    source = copy.deepcopy(fresh[1])
    source[field] = value
    with pytest.raises(TypeError, match="exact complete-source"):
        m.bind_context_compatibility(
            source, fresh[2], fresh[3][0], plan(), support_limit=LIMIT
        )
