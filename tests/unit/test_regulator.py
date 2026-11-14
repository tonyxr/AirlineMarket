"""A.8/B0/V9: reference windows and native cap execution on new local fixtures."""

import json
import pickle
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.core.types import CoefficientAction, PriceAction
from airline_market.policy.regulator import (
    CapPolicy,
    CapWindow,
    ReferenceQuote,
    RegulatedMarketEngine,
    reference_medians,
)
from airline_market.sim.arrivals import Request
from airline_market.sim.engine import MarketEngine
from airline_market.sim.tape import ExogenousTape
from airline_market.spec import compile_market, load_spec


def fixture_model(cancel=False):
    # Reuse only a physical ledger scaffold, with distinct T4/C6/MNL inputs.
    source = json.loads(Path("config/benchmarks/V4_exponential_k1.json").read_text())[
        "physical_spec"
    ]
    source.update(
        name="unit_V9_cap_native_T4_C6",
        description="Native MNL cap fixture; not the continuous-demand benchmark",
    )
    source["horizon"]["n_periods"] = 4
    source["legs"][0]["capacity"]["support"] = [6]
    source["demand"]["arrival_probability"]["values"] = [1.0]
    source["demand"]["arrival_probability_cap"] = 1.0
    source["segments"][0]["classes"][0]["market_value"]["OD"] = 5
    source["segments"][0]["classes"][0]["alpha"] = 0.03
    if cancel:
        source["capacity_process"]["events"] = [
            {
                "trigger": {"hazard": {"probability": 1}},
                "effects": [["A_0", -6]],
                "once": True,
            }
        ]
    return compile_market(load_spec(source))


def new_tape(c, key=0):
    return ExogenousTape.generate(
        c, key, registry=StreamRegistry(8191, f"unit/V9_caps/{key}")
    )


def test_reference_window_exact_keys_missingness_and_no_future_fare_access():
    reveal = datetime(2026, 5, 2, tzinfo=timezone.utc)
    key = ("ORD", "MIA", "economy")

    class UnreadableFare:
        def __str__(self):
            raise AssertionError("post-reveal fare was inspected")

    rows = [
        ReferenceQuote(key, reveal - timedelta(days=28), "99.99"),
        ReferenceQuote(key, reveal - timedelta(days=1), "100.01"),
        ReferenceQuote(key, reveal - timedelta(days=2), None),
        ReferenceQuote(key, reveal - timedelta(days=2), float("nan")),
        ReferenceQuote(key, reveal - timedelta(days=2), -2),
        ReferenceQuote(("MIA", "ORD", "economy"), reveal - timedelta(days=1), 180),
        ReferenceQuote(key, reveal, UnreadableFare()),
        ReferenceQuote(key, reveal - timedelta(days=29), UnreadableFare()),
    ]
    result = reference_medians(rows, reveal)
    group = next(r for r in result["groups"] if r["key"] == list(key))
    assert Decimal(group["median_fare_decimal"]) == 100
    assert group["usable_quotes"] == 2 and group["observed_days"] == 3
    assert (
        group["missing_quotes"]
        == group["nonfinite_quotes"]
        == group["nonpositive_quotes"]
        == 1
    )
    assert result["out_of_window_quotes"] == 2
    assert (
        next(r for r in result["groups"] if r["key"][0] == "MIA")["median_fare_decimal"]
        == "180"
    )


def test_no_usable_reference_stays_missing_and_naive_clock_is_rejected():
    reveal = datetime(2026, 5, 2, tzinfo=timezone.utc)
    result = reference_medians(
        [ReferenceQuote(("O", "D", "coach"), reveal - timedelta(days=1), None)], reveal
    )
    assert result["groups"][0]["median_fare_decimal"] is None
    with pytest.raises(ValueError, match="timezone-aware"):
        reference_medians([], reveal.replace(tzinfo=None))


def test_cap_decimal_floor_and_public_emergency_windows():
    window = CapWindow(0, 4, (("SEAT", "99.99"),), requires_emergency=True)
    policy = CapPolicy((window,), ((1, 3),))
    assert policy.limits(0) == policy.limits(3) == {}
    assert policy.limits(1) == policy.limits(2) == {"SEAT": 129.98}
    assert Decimal(str(policy.limits(1)["SEAT"])) <= Decimal("99.99") * Decimal("1.30")
    with pytest.raises(FrozenInstanceError):
        window.start = 1
    assert CapPolicy((window,)).limits(1) == {}


def test_overlapping_caps_take_minimum_and_copy_mutable_inputs():
    refs = [["SEAT", 80]]
    window = CapWindow(0, 4, refs, multiplier=1.25)
    refs[0][1] = 900
    policy = CapPolicy([window, CapWindow(2, 3, (("SEAT", 70),), 1)])
    assert policy.limits(1) == {"SEAT": 100}
    assert policy.limits(2) == {"SEAT": 70}
    assert policy.limits(3) == {"SEAT": 100}
    assert policy.limits(4) == {}


@pytest.mark.parametrize(
    "window",
    [
        (-1, 2, (("SEAT", 1),)),
        (2, 2, (("SEAT", 1),)),
        (0, 1, (("SEAT", 0),)),
        (0, 1, (("SEAT", "NaN"),)),
        (0, 1, (("SEAT", 1), ("SEAT", 2))),
    ],
)
def test_invalid_window_and_reference_cannot_silently_change_scope(window):
    with pytest.raises(ValueError):
        CapWindow(*window)


@pytest.mark.parametrize("kind", ["none", "ceiling", "undeclared"])
def test_no_effect_cap_is_byte_identical_to_native_profiles(kind):
    c = fixture_model()
    policy = {
        "none": CapPolicy(),
        "ceiling": CapPolicy((CapWindow(0, 4, (("SEAT", 500),), 1),)),
        "undeclared": CapPolicy(
            (CapWindow(0, 4, (("SEAT", 20),), requires_emergency=True),)
        ),
    }[kind]
    for key in range(8):
        tape = new_tape(c, key)
        base, regulated = MarketEngine(c), RegulatedMarketEngine(c, cap_policy=policy)
        for engine in (base, regulated):
            engine.reset(key, tape=tape)
            while not engine.terminal():
                engine.step({"A": CoefficientAction((0,))})
        assert base.tape is regulated.tape
        assert pickle.dumps(
            (base.world, base.trace(), base.metrics()), protocol=4
        ) == pickle.dumps(
            (regulated.world, regulated.trace(), regulated.metrics()), protocol=4
        )


def test_native_bookings_use_active_cap_only_inside_window():
    c = fixture_model()
    engine = RegulatedMarketEngine(
        c, cap_policy=CapPolicy((CapWindow(1, 3, (("SEAT", 20),)),))
    )
    engine.reset(tape=new_tape(c))
    prices = []
    for t in range(4):
        # Constructed request/choice fixture isolates native posting and accounting.
        result = engine.step(
            {"A": CoefficientAction((0,))},
            _requests=[Request(0, 0, 0, 0, 0, 0)],
            _choices=[0],
        )
        prices.append(result.purchases[0].fare)
    assert prices == [100, 26, 26, 100]
    assert engine.world.carriers[0].cash == sum(prices)
    assert engine.metrics()["carriers"]["A"]["revenue"] == sum(prices)
    assert [x["prices"][0] for x in engine.trace()] == prices


def test_capped_original_fare_is_refunded_by_unmodified_native_resolution():
    c = fixture_model(cancel=True)
    engine = RegulatedMarketEngine(
        c, cap_policy=CapPolicy((CapWindow(0, 1, (("SEAT", 20),)),))
    )
    engine.reset(tape=new_tape(c))
    result = engine.step(
        {"A": PriceAction({0: 200})},
        _requests=[Request(0, 0, 0, 0, 0, 0)],
        _choices=[0],
    )
    state = engine.world.carriers[0]
    assert result.purchases[0].fare == 26 and state.revenue == state.refunds == 26
    assert state.cash == 0 and state.canceled == 1 and not state.bookings


def test_branch_and_restore_retain_cap_with_independent_mutable_world():
    c = fixture_model()
    engine = RegulatedMarketEngine(
        c, cap_policy=CapPolicy((CapWindow(0, 4, (("SEAT", 20),)),))
    )
    engine.reset(tape=new_tape(c))
    engine.step({"A": CoefficientAction((0,))}, _requests=[], _choices=[])
    branch = engine.branch()
    saved = engine.snapshot()
    assert (
        isinstance(branch, RegulatedMarketEngine)
        and branch.cap_policy is engine.cap_policy
        and branch.tape is engine.tape
    )
    assert branch.world.carriers[0].capacity is not engine.world.carriers[0].capacity
    outcome = branch.step(
        {"A": PriceAction({0: 200})},
        _requests=[Request(0, 0, 0, 0, 0, 0)],
        _choices=[0],
    )
    assert outcome.purchases[0].fare == 26 and not engine.world.carriers[0].bookings
    engine.step(
        {"A": PriceAction({0: 200})},
        _requests=[Request(0, 0, 0, 0, 0, 0)],
        _choices=[0],
    )
    engine.restore(saved)
    assert not engine.world.carriers[0].bookings and engine.cap_policy.limits(
        engine.world.t
    ) == {"SEAT": 26}


def test_unknown_product_target_is_rejected_before_running():
    with pytest.raises(ValueError, match="unknown product"):
        RegulatedMarketEngine(
            fixture_model(), cap_policy=CapPolicy((CapWindow(0, 1, (("TYPO", 20),)),))
        )


def test_actor_forward_once_and_next_observation_uses_executed_fare():
    class Spy:
        def reset(self, context, rng):
            self.observations = []

        def act(self, observation):
            assert observation.privileged is None
            self.observations.append(observation)
            return CoefficientAction((0,))

        def memory(self):
            return ()

        def observe_outcome(self, outcome):
            assert set(outcome) == {"purchases", "cash_change"}

    spy = Spy()
    engine = RegulatedMarketEngine(
        fixture_model(),
        {"A": spy},
        cap_policy=CapPolicy((CapWindow(0, 4, (("SEAT", 20),)),)),
    )
    engine.reset(tape=new_tape(engine.compiled))
    while not engine.terminal():
        engine.step(_requests=[], _choices=[])
    assert len(spy.observations) == 4
    assert tuple(spy.observations[0].own_previous_prices) == (100,)
    assert all(tuple(obs.own_previous_prices) == (26,) for obs in spy.observations[1:])


def test_fractional_cap_is_enforced_in_native_booking():
    engine = RegulatedMarketEngine(
        fixture_model(),
        cap_policy=CapPolicy((CapWindow(0, 1, (("SEAT", "99.99"),)),)),
    )
    engine.reset(tape=new_tape(engine.compiled))
    result = engine.step(
        {"A": PriceAction({0: 200})},
        _requests=[Request(0, 0, 0, 0, 0, 0)],
        _choices=[0],
    )
    assert result.purchases[0].fare == 129.98
    assert Decimal(str(result.purchases[0].fare)) <= Decimal("129.987")


def test_multicarrier_target_uses_global_product_identity_only():
    from airline_market.pipeline.screen_generators import screen_spec

    c = compile_market(
        screen_spec(
            {
                "T": 3,
                "actions": 2,
                "n_spokes": 1,
                "capacity": 4,
                "capacity_support": [2, 3],
                "price_scale": 1.23,
                "outside_utility_shift": 0.14,
            }
        )
    )
    target = c.products_of[1][-1]
    product = c.spec.products[target].id
    policy = CapPolicy((CapWindow(0, 3, ((product, 5),)),))
    base, regulated = MarketEngine(c), RegulatedMarketEngine(c, cap_policy=policy)
    tape = new_tape(c, 9)
    action = {
        carrier.code: PriceAction({j: 150.0 for j in c.products_of[i]})
        for i, carrier in enumerate(c.spec.carriers)
    }
    base.reset(tape=tape)
    regulated.reset(tape=tape)
    old_prices = base.post_actions(action)
    prices = regulated.post_actions(action)
    assert prices[target] <= 6.5
    assert all(prices[j] == old_prices[j] for j in range(len(prices)) if j != target)


def test_native_quote_dependent_repricing_is_preserved_under_a_rival_cap():
    from airline_market.pipeline.screen_generators import screen_spec

    c = compile_market(
        screen_spec(
            {
                "T": 3,
                "actions": 2,
                "n_spokes": 1,
                "capacity": 4,
                "capacity_support": [2, 3],
                "price_scale": 1.23,
                "outside_utility_shift": 0.14,
            }
        )
    )
    target = c.products_of[1][-1]
    product = c.spec.products[target].id
    policy = CapPolicy((CapWindow(0, 3, ((product, 5),)),))
    tape = new_tape(c, 9)
    action = {"A": CoefficientAction(c.spec.pricing[0].theta_grid[-1])}
    base, expected = MarketEngine(c), MarketEngine(c)
    regulated = RegulatedMarketEngine(c, cap_policy=policy)
    for engine in (base, expected, regulated):
        engine.reset(tape=tape)
    original = base.post_actions(action)
    rival_prices = {
        j: min(original[j], 6.5) if j == target else original[j]
        for j in c.products_of[1]
    }
    native_feedback = expected.post_actions({**action, "B": PriceAction(rival_prices)})
    actual = regulated.post_actions(action)
    assert actual.tolist() == native_feedback.tolist()
    assert any(actual[j] != original[j] for j in c.products_of[0])
