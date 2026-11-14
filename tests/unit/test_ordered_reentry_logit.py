"""A.10/A.5/B0/V9: NEW analytic ordered-choice references, not lifecycle CS."""

import importlib.util
import json
import math
import os
import pickle
import sys
from dataclasses import replace

import numpy as np
import pytest
from scipy.integrate import quad

from airline_market.pipeline.screen_generators import screen_spec
from airline_market.sim.choice import sampled_choice, systematic_utilities
from airline_market.spec import compile_market

if os.environ.get("ORDERED_REENTRY_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_ordered_reentry", os.environ["ORDERED_REENTRY_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.sim import ordered_reentry_logit as module


def problem(
    own=(("own", 0.0),),
    other=(("other", 0.0),),
    *,
    interline=True,
    alpha=1.0,
    outside=0.0,
):
    return module.OrderedReentryLogit(
        own,
        other,
        "outside",
        outside,
        alpha,
        interline,
        "NEW one-attempt unit fixed taste",
    )


def test_shared_outside_priority_is_not_pooled_menu_or_independent_stage_logit():
    result = module.ordered_reentry_expectation(problem())
    p = dict(result["choice_probabilities"])
    assert p == pytest.approx({"own": 0.5, "other": 1 / 6, "outside": 1 / 3})
    assert p["other"] != 0.25 and p["outside"] != 0.25
    assert result["expected_selected_minus_same_outside_utility"] == pytest.approx(
        math.log(2) + 0.5 * math.log(1.5)
    )
    assert result["same_full_menu_expected_max_gain_utility"] == pytest.approx(
        math.log(3)
    )
    assert result["raw_pooled_menu_minus_ordered_gain_utility"] > 0
    assert (
        result["native_lifecycle_consumer_surplus"]
        is result["delivered_travel"]
        is None
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_multiple_products_exact_race_probabilities_and_declared_alpha(enabled):
    r = module.ordered_reentry_expectation(
        problem(
            (("a", math.log(2)), ("b", math.log(3))),
            (("c", math.log(5)),),
            outside=math.log(7),
            alpha=0.25,
            interline=enabled,
        )
    )
    p = dict(r["choice_probabilities"])
    assert p["a"] == pytest.approx(2 / 12) and p["b"] == pytest.approx(3 / 12)
    assert p["c"] == pytest.approx(35 / (12 * 17) if enabled else 0)
    assert p["outside"] == pytest.approx(7 / 17 if enabled else 7 / 12)
    assert r["expected_gain_divided_by_declared_alpha"] == pytest.approx(
        4 * r["expected_selected_minus_same_outside_utility"]
    )
    assert [k for k, v in r["other_utilities"]] == ["c"]


@pytest.mark.parametrize(
    "own,other",
    [
        ((), (("b", 0.0),)),
        ((("a", 0.0),), ()),
        ((), ()),
        ((("a", -math.inf),), (("b", -math.inf),)),
    ],
)
def test_empty_and_explicit_unavailable_groups_keep_every_row(own, other):
    r = module.ordered_reentry_expectation(problem(own, other))
    assert len(r["choice_probabilities"]) == len(own) + len(other) + 1
    assert r["raw_probability_mass"] == pytest.approx(1)
    expected = math.log(2) if any(math.isfinite(v) for k, v in (*own, *other)) else 0
    assert r["expected_selected_minus_same_outside_utility"] == pytest.approx(expected)
    assert r["raw_pooled_menu_minus_ordered_gain_utility"] == pytest.approx(0)


@pytest.mark.parametrize(
    "wa,wo,w0", [(0.3, 0.7, 1.0), (2.0, 3.0, 7.0), (5.0, 0.2, 0.5)]
)
def test_independent_exponential_race_and_tail_integrals(wa, wo, w0):
    r = module.ordered_reentry_expectation(
        problem((("a", math.log(wa)),), (("b", math.log(wo)),), outside=math.log(w0))
    )
    # Native own failure shares the SAME outside race time with second stage.
    interline = quad(
        lambda x: w0 * math.exp(-(w0 + wa) * x) * (-math.expm1(-wo * x)),
        0,
        np.inf,
        epsabs=1e-12,
    )[0]
    outside = quad(
        lambda x: w0 * math.exp(-(w0 + wa + wo) * x), 0, np.inf, epsabs=1e-12
    )[0]
    # Integrate positive chosen-minus-outside utility tail, rather than log identity.
    gain = quad(
        lambda z: (
            wa * math.exp(-z) / (w0 + wa * math.exp(-z))
            + w0 * wo * math.exp(-z) / ((w0 + wa) * (w0 + wa + wo * math.exp(-z)))
        ),
        0,
        np.inf,
        epsabs=1e-12,
    )[0]
    p = dict(r["choice_probabilities"])
    assert p["b"] == pytest.approx(interline, abs=1e-12) and p[
        "outside"
    ] == pytest.approx(outside, abs=1e-12)
    assert r["expected_selected_minus_same_outside_utility"] == pytest.approx(
        gain, abs=1e-12
    )


def test_common_shift_canonical_order_inputs_global_rng_and_immutable_export():
    q = problem((("b", 2.0), ("a", -3.0)), (("d", -0.5), ("c", 1.0)))
    before = pickle.dumps((q, np.random.get_state()))
    r = module.ordered_reentry_expectation(q)
    shifted = replace(
        q,
        own=tuple((k, v + 1000) for k, v in q.own),
        other=tuple((k, v + 1000) for k, v in q.other),
        outside_utility=1000.0,
    )
    s = module.ordered_reentry_expectation(shifted)
    assert (
        r["choice_probabilities"] == s["choice_probabilities"]
        and r["expected_selected_minus_same_outside_utility"]
        == s["expected_selected_minus_same_outside_utility"]
    )
    assert pickle.dumps((q, np.random.get_state())) == before
    exported = module.ordered_reentry_record(r)
    json.dumps(exported, allow_nan=False)
    exported["own_utilities"].clear()
    assert len(r["own_utilities"]) == 2
    with pytest.raises(TypeError):
        r["raw_mass_residual"] = 0


def test_underflow_support_and_rare_stage_gain_not_multiplied_after_probability_underflow():
    r = module.ordered_reentry_expectation(
        problem((("a", 1000.0), ("rare", -1000.0)), (("b", 1e300),))
    )
    assert r["first_stage_outside_probability"] == 0
    assert r["interline_increment_expected_gain_utility"] > 0
    assert math.log(r["interline_increment_expected_gain_utility"]) == pytest.approx(
        math.log(1e300) - 1000
    )
    assert (
        dict(r["choice_probabilities"])["rare"] == 0
        and len(r["choice_probabilities"]) == 4
    )
    assert not r["underflow_or_inactive_zero_is_true_probability_or_variance_claim"]


@pytest.mark.parametrize(
    "change",
    [
        "duplicate",
        "outside collision",
        "unknown",
        "nan",
        "positive infinity",
        "zero alpha",
        "infinite alpha",
        "unknown outside",
        "no definition",
        "implicit interline",
        "boolean utility",
    ],
)
def test_incomplete_or_invalid_declarations_fail_without_fill(change):
    kw = {
        "own": (("a", 0.0),),
        "other": (("b", 0.0),),
        "outside_name": "outside",
        "outside_utility": 0.0,
        "alpha": 1.0,
        "interline": True,
        "definition": "NEW unit",
    }
    if change == "duplicate":
        kw["other"] = (("a", 0.0),)
    elif change == "outside collision":
        kw["own"] = (("outside", 0.0),)
    elif change == "unknown":
        kw["own"] = (("a", None),)
    elif change == "nan":
        kw["other"] = (("b", math.nan),)
    elif change == "positive infinity":
        kw["other"] = (("b", math.inf),)
    elif change == "zero alpha":
        kw["alpha"] = 0.0
    elif change == "infinite alpha":
        kw["alpha"] = math.inf
    elif change == "unknown outside":
        kw["outside_utility"] = None
    elif change == "no definition":
        kw["definition"] = " "
    elif change == "implicit interline":
        kw["interline"] = 1
    elif change == "boolean utility":
        kw["own"] = (("a", True),)
    with pytest.raises((ValueError, TypeError)):
        module.OrderedReentryLogit(**kw)


def test_nonrepresentable_centering_fails_without_partial_positive_result():
    q = problem((("a", 1e308),), (), outside=-1e308)
    with pytest.raises(FloatingPointError):
        module.ordered_reentry_expectation(q)


def native_input():
    c = compile_market(
        screen_spec({"T": 3, "actions": 2, "n_spokes": 1, "capacity": 3})
    )
    taste = c.spec.segments[0].classes[0]
    prices = np.array([100.0 + i for i in range(len(c.spec.products))])
    sellable = np.ones(len(prices), dtype=bool)
    return c, taste, prices, sellable


def test_native_current_adjusted_taste_full_support_one_utility_forward_no_rng(
    monkeypatch,
):
    c, taste, prices, sellable = native_input()
    market = "OH0"
    original = "B"
    market_products = c.market_products[c.market_index[market]]
    sellable[market_products[0]] = False
    before = pickle.dumps((c, taste, prices, sellable, np.random.get_state()))
    original_forward = module.systematic_utilities
    calls = []

    def forward(*args):
        calls.append(1)
        return original_forward(*args)

    monkeypatch.setattr(module, "systematic_utilities", forward)
    r = module.native_ordered_reentry_expectation(
        c,
        market,
        taste,
        prices=prices,
        sellable=sellable,
        original_carrier=original,
        interline=True,
        definition="NEW current supplied adjusted native taste",
    )
    assert (
        len(calls) == 1
        and pickle.dumps((c, taste, prices, sellable, np.random.get_state())) == before
    )
    assert len(r["choice_probabilities"]) == len(market_products) + 1
    assert r["native_projection"]["spec_hash"] == c.spec_hash
    assert len(r["native_projection"]["complete_global_prices"]) == len(c.spec.products)
    assert (
        c.spec.products[market_products[0]].id in r["structural_unavailable_products"]
    )
    assert not r["native_projection"][
        "stock_holds_taste_or_public_source_authenticated"
    ]


def test_native_same_marks_own_first_can_choose_own_despite_better_interline():
    c, taste, prices, sellable = native_input()
    m = c.market_index["OH0"]
    carrier = c.carrier_index["B"]
    products = c.market_products[m]
    own = next(i for i, j in enumerate(products) if c.product_carrier[j] == carrier)
    other = next(i for i, j in enumerate(products) if c.product_carrier[j] != carrier)

    def native_two_stages(g):
        for own_stage in (True, False):
            mask = sellable.copy()
            for j in products:
                mask[j] &= (c.product_carrier[j] == carrier) == own_stage
            u = systematic_utilities(c, m, taste, prices, mask)
            selected = sampled_choice(u, g)
            if selected < len(products):
                return selected
        return len(products)

    g = np.zeros(len(products) + 1)
    g[own] = 1e4
    g[other] = 1e6
    assert native_two_stages(g) == own
    g[own] = -1e6
    assert native_two_stages(g) == other
    g[other] = -1e6
    assert native_two_stages(g) == len(products)


@pytest.mark.parametrize(
    "change",
    [
        "short prices",
        "short mask",
        "integer mask",
        "missing market",
        "missing carrier",
        "nonfinite available utility",
    ],
)
def test_native_projection_requires_complete_context_and_finite_available_utilities(
    change,
):
    c, taste, prices, sellable = native_input()
    kwargs = {
        "prices": prices,
        "sellable": sellable,
        "original_carrier": "B",
        "interline": True,
        "definition": "NEW unit",
    }
    market = "OH0"
    if change == "short prices":
        kwargs["prices"] = prices[:-1]
    elif change == "short mask":
        kwargs["sellable"] = sellable[:-1]
    elif change == "integer mask":
        kwargs["sellable"] = [1] * len(sellable)
    elif change == "missing market":
        market = "absent"
    elif change == "missing carrier":
        kwargs["original_carrier"] = "absent"
    elif change == "nonfinite available utility":
        taste = replace(taste, market_value={"*": math.inf})
    with pytest.raises((ValueError, FloatingPointError)):
        module.native_ordered_reentry_expectation(c, market, taste, **kwargs)


@pytest.mark.parametrize(
    "case", ["own despite better interline", "interline", "outside"]
)
def test_new_base_native_reentry_booking_witness_uses_actual_adjusted_taste(
    case, monkeypatch
):
    from airline_market.core.rng import StreamRegistry
    from airline_market.core.types import PriceAction
    from airline_market.sim.arrivals import Request
    from airline_market.sim.rebooking import RebookingMarketEngine, RebookingPolicy
    from airline_market.sim.tape import ExogenousTape
    from airline_market.spec.model import CapacityEventSpec, CapacityProcessSpec

    base = screen_spec({"T": 3, "actions": 2, "n_spokes": 1, "capacity": 3})
    # NEW BASE native-event fixture: genuine resolve/refund then recovery leaves
    # own seats for the NEXT attempt. Not a dated physical-window acceptance.
    events = tuple(
        CapacityEventSpec(
            trigger={"hazard": {"probability": 1.0}},
            effects=(("B_OH0", delta),),
            once=True,
        )
        for delta in (-3, 3)
    )
    c = compile_market(
        replace(base, capacity_process=CapacityProcessSpec(events=events))
    )
    e = RebookingMarketEngine(c, rebooking=RebookingPolicy(2, 2.0, 0.0, True))
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(60974, "unit/V9_ordered_reentry_logit/" + case)
    )
    e.reset(tape=tape)
    actions = {
        carrier.code: PriceAction({j: 100.0 for j in c.products_of[i]})
        for i, carrier in enumerate(c.spec.carriers)
    }
    b = c.product_index["B_OH0"]
    a = c.product_index["A_OH0"]
    market = c.market_index["OH0"]
    # Allowed NEW BASE unit request/choice hook, never dated/lineage forced hook.
    e.step(actions, _requests=(Request(0, market, 0, 0, 0, 0),), _choices=(b,))
    assert (
        e.stranded
        and e._passengers
        and not e.world.carriers[c.carrier_index["B"]].bookings
    )
    old = e._marks
    captured = []

    def marks(passenger, period):
        taste, g = old(passenger, period)
        g = np.full(len(c.spec.products) + 1, -1e6)
        g[-1] = 0
        if case == "own despite better interline":
            g[b] = 1e4
            g[a] = 1e6
        elif case == "interline":
            g[a] = 1e6
        before = pickle.dumps((e.world, np.random.get_state()))
        result = module.native_ordered_reentry_expectation(
            c,
            "OH0",
            taste,
            prices=[100.0] * len(c.spec.products),
            sellable=e._sellable_global(),
            original_carrier="B",
            interline=True,
            definition="NEW actual current adjusted native taste, BASE witness",
        )
        assert pickle.dumps((e.world, np.random.get_state())) == before
        assert taste.alpha == pytest.approx(c.spec.segments[0].classes[0].alpha / 2)
        captured.append(result)
        return taste, g

    monkeypatch.setattr(e, "_marks", marks)
    e.step(actions, _requests=(), _choices=())
    row = e.rebooking_history[-1]
    assert row["chosen"] == (
        b
        if case == "own despite better interline"
        else a
        if case == "interline"
        else None
    )
    assert row["stage"] == (
        "own"
        if case == "own despite better interline"
        else "interline"
        if case == "interline"
        else None
    )
    assert len(captured) == 1 and e.conservation()["original_booked_passengers"] == 1
