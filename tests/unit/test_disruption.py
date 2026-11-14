"""A.5/B0/V9: exit, native refunds, conservation and branch ownership."""

import pickle
from dataclasses import FrozenInstanceError, replace

import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.core.types import CoefficientAction, PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.regulator import CapPolicy, CapWindow
from airline_market.sim.arrivals import Request
from airline_market.sim.disruption import DisruptedMarketEngine, RivalExit
from airline_market.sim.engine import MarketEngine
from airline_market.sim.tape import ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.model import CapacityEventSpec, CapacityProcessSpec


def fixture_model(native_event=None):
    spec = screen_spec(
        {
            "T": 4,
            "actions": 2,
            "n_spokes": 1,
            "capacity": 5,
            "capacity_support": [3, 4],
            "price_scale": 1.29,
            "outside_utility_shift": 0.11,
            "arrival_probability": 0.74,
        }
    )
    if native_event:
        legs = [leg.id for leg in spec.legs if leg.carrier == "B"]
        delta = 4 if native_event == "recover" else -20
        spec = replace(
            spec,
            capacity_process=CapacityProcessSpec(
                events=(
                    CapacityEventSpec(
                        trigger={"hazard": {"probability": 1}},
                        effects=tuple((leg, delta) for leg in legs),
                        once=native_event != "recover",
                    ),
                )
            ),
        )
    return compile_market(spec)


def tape(c, key=0):
    return ExogenousTape.generate(
        c, key, registry=StreamRegistry(19417, f"unit/V9_exit/{key}")
    )


def actions(c):
    return {
        carrier.code: PriceAction({j: 100 + 20 * i for j in c.products_of[i]})
        for i, carrier in enumerate(c.spec.carriers)
    }


def buy(engine, carrier):
    c = engine.compiled
    j = max(
        c.products_of[c.carrier_index[carrier]],
        key=lambda j: len(c.spec.products[j].legs),
    )
    request = Request(0, int(c.product_market[j]), 0, 0, 0, 0)
    return engine.step(actions(c), _requests=[request], _choices=[j])


def assert_conserved(engine):
    result = engine.conservation()
    assert result["active_plus_stranded"] == result["ever_booked_tickets"]
    assert len({s.booking.record_id for s in engine.stranded}) == len(engine.stranded)
    for state in engine.world.carriers:
        assert state.cash == pytest.approx(
            state.revenue - state.refunds - state.penalties
        )


@pytest.mark.parametrize("event", [None, "cancel"])
def test_no_exit_overlay_is_byte_identical_to_native(event):
    c = fixture_model(event)
    for key in range(8):
        shared = tape(c, key)
        base, overlay = MarketEngine(c), DisruptedMarketEngine(c)
        for engine in (base, overlay):
            engine.reset(key, tape=shared)
            while not engine.terminal():
                engine.step(
                    {
                        "A": CoefficientAction(
                            c.spec.pricing[0].theta_grid[engine.world.t % 2]
                        )
                    }
                )
        assert pickle.dumps(
            (base.world, base.trace(), base.metrics()), protocol=4
        ) == pickle.dumps(
            (overlay.world, overlay.trace(), overlay.metrics()), protocol=4
        )
        assert_conserved(overlay)


def test_exit_strands_each_shared_leg_ticket_once_and_refunds_native_fare():
    c = fixture_model()
    engine = DisruptedMarketEngine(c, exits=(RivalExit("B", 2),))
    engine.reset(tape=tape(c))
    buy(engine, "B")
    result = buy(engine, "B")
    state = engine.world.carriers[1]
    assert engine.world.t == 2 and not any(state.capacity) and not state.bookings
    assert len(engine.stranded) == state.canceled == 2
    assert all(
        len(c.spec.products[s.booking.product].legs) == 2 for s in engine.stranded
    )
    assert state.revenue == state.refunds == 240
    assert state.penalties == 2 * c.spec.resolution.penalty_per_cancellation
    assert state.cash == -state.penalties
    assert [s.canceled_at for s in engine.stranded] == [2, 2]
    assert {s.cause for s in engine.stranded} == {"rival_exit"}
    assert (
        result.cash_changes[1] == -120 - 2 * c.spec.resolution.penalty_per_cancellation
    )
    assert (
        sum(bool(r.get("disruption") == "rival_exit") for r in result.resolutions) == 1
    )
    for _ in range(2):
        buy(engine, "A")
        assert not any(state.capacity) and len(engine.stranded) == 2
        assert_conserved(engine)


def test_future_capacity_events_cannot_restore_an_exited_carrier():
    c = fixture_model("recover")
    engine = DisruptedMarketEngine(c, exits=(RivalExit("B", 1),))
    engine.reset(tape=tape(c))
    buy(engine, "B")
    for _ in range(3):
        result = buy(engine, "A")
        assert not any(engine.world.carriers[1].capacity)
        assert any(row.get("recovery_blocked") for row in result.resolutions)
    assert len(engine.stranded) == 1 and engine.world.carriers[1].refunds == 120
    assert_conserved(engine)


def test_native_event_cancellations_are_retained_without_changing_native_cash():
    c = fixture_model("cancel")
    engine = DisruptedMarketEngine(c)
    engine.reset(tape=tape(c))
    result = buy(engine, "B")
    assert (
        len(engine.stranded) == 1
        and engine.stranded[0].cause == "native_capacity_event"
    )
    assert engine.stranded[0].canceled_at == 1
    assert result.cash_changes[1] == -c.spec.resolution.penalty_per_cancellation
    assert_conserved(engine)


def test_reveal_zero_is_applied_before_first_observation_and_is_idempotent():
    c = fixture_model()
    engine = DisruptedMarketEngine(c, exits=(RivalExit("B", 0),))
    engine.reset(tape=tape(c))
    assert not any(engine.world.carriers[1].capacity)
    engine.begin_period(actions(c))
    engine.begin_period(actions(c))
    result = buy(engine, "A")
    assert (
        len([r for r in result.resolutions if r.get("disruption") == "rival_exit"]) == 1
    )
    assert not engine.stranded and engine.world.carriers[1].refunds == 0
    assert_conserved(engine)


def test_snapshot_branch_before_reveal_owns_ledger_and_refunds_independently():
    c = fixture_model()
    engine = DisruptedMarketEngine(c, exits=(RivalExit("B", 2),))
    engine.reset(tape=tape(c))
    buy(engine, "B")
    saved = engine.snapshot()
    branch = engine.branch()
    assert isinstance(branch, DisruptedMarketEngine) and branch.tape is engine.tape
    assert branch.world.carriers[1].capacity is not engine.world.carriers[1].capacity
    buy(branch, "B")
    assert (
        len(branch.stranded) == 2
        and not engine.stranded
        and len(engine.world.carriers[1].bookings) == 1
    )
    buy(engine, "B")
    expected = pickle.dumps(
        (engine.world, engine.stranded, engine.metrics()), protocol=4
    )
    assert expected == pickle.dumps(
        (branch.world, branch.stranded, branch.metrics()), protocol=4
    )
    engine.restore(saved)
    assert not engine.stranded and engine.world.carriers[1].refunds == 0
    buy(engine, "B")
    assert (
        pickle.dumps((engine.world, engine.stranded, engine.metrics()), protocol=4)
        == expected
    )


def test_snapshot_after_initial_reveal_retains_pending_event_once():
    c = fixture_model()
    engine = DisruptedMarketEngine(c, exits=(RivalExit("B", 0),))
    engine.reset(tape=tape(c))
    branch = engine.branch()
    a, b = buy(engine, "A"), buy(branch, "A")
    assert a.resolutions == b.resolutions and len(a.resolutions) == 1
    assert engine.conservation() == branch.conservation()


def test_cap_composition_refunds_the_executed_capped_fare():
    c = fixture_model()
    refs = tuple((c.spec.products[j].id, 20) for j in c.products_of[1])
    engine = DisruptedMarketEngine(
        c, exits=(RivalExit("B", 1),), cap_policy=CapPolicy((CapWindow(0, 4, refs),))
    )
    engine.reset(tape=tape(c))
    buy(engine, "B")
    assert engine.stranded[0].booking.fare == 26
    assert engine.world.carriers[1].revenue == engine.world.carriers[1].refunds == 26
    assert_conserved(engine)


def test_exit_refunds_reach_one_native_outcome_callback_without_hidden_actor_input():
    class Spy:
        def reset(self, context, rng):
            self.products = context.compiled.products_of[context.carrier]
            self.calls, self.outcomes = 0, []

        def act(self, observation):
            assert observation.privileged is None
            self.calls += 1
            return PriceAction({j: 120 for j in self.products})

        def observe_outcome(self, outcome):
            self.outcomes.append(outcome)

        def memory(self):
            return ()

    c = fixture_model()
    spy = Spy()
    engine = DisruptedMarketEngine(c, {"B": spy}, exits=(RivalExit("B", 1),))
    engine.reset(tape=tape(c))
    j = max(c.products_of[1], key=lambda j: len(c.spec.products[j].legs))
    engine.step(
        {"A": CoefficientAction((0, 0, 0))},
        _requests=[Request(0, int(c.product_market[j]), 0, 0, 0, 0)],
        _choices=[j],
    )
    assert spy.calls == len(spy.outcomes) == 1
    assert spy.outcomes[0]["cash_change"] == -c.spec.resolution.penalty_per_cancellation
    assert engine.world.carriers[1].refunds == 120


@pytest.mark.parametrize("plan", [RivalExit("MISSING", 1), RivalExit("B", 4)])
def test_invalid_target_or_reveal_is_rejected(plan):
    with pytest.raises(ValueError, match="outside"):
        DisruptedMarketEngine(fixture_model(), exits=(plan,))


def test_exit_plan_is_immutable():
    with pytest.raises(FrozenInstanceError):
        RivalExit("B", 1).reveal_period = 0
