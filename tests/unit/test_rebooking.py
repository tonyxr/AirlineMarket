"""A.5/B0/V9: re-entry identity, passenger lineages, window, money and CRN."""

import pickle
from dataclasses import replace

import numpy as np
import pytest
from airline_market.sim.rebooking import RebookingMarketEngine, RebookingPolicy

from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.regulator import CapPolicy, CapWindow
from airline_market.sim.arrivals import Request
from airline_market.sim.disruption import DisruptedMarketEngine, RivalExit
from airline_market.sim.engine import MarketEngine
from airline_market.sim.tape import ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.model import CapacityEventSpec, CapacityProcessSpec


def model(recover=False, twice=False):
    spec = screen_spec(
        {
            "T": 5,
            "actions": 2,
            "n_spokes": 1,
            "capacity": 5,
            "capacity_support": [3, 4],
            "arrival_probability": 0.7,
        }
    )
    events = []
    if recover:
        for delta in [-20, 5]:
            events.append(
                CapacityEventSpec(
                    trigger={"hazard": {"probability": 0.5}},
                    effects=tuple((l.id, delta) for l in spec.legs if l.carrier == "B"),
                    once=True,
                )
            )
    if twice:
        events.append(
            CapacityEventSpec(
                trigger={"hazard": {"probability": 0.5}},
                effects=tuple((l.id, -20) for l in spec.legs if l.carrier == "A"),
                once=True,
            )
        )
    if events:
        spec = replace(spec, capacity_process=CapacityProcessSpec(events=tuple(events)))
    return compile_market(spec)


def tape(c, key=0, recover=False, twice=False):
    shared = ExogenousTape.generate(
        c, key, registry=StreamRegistry(19781, f"unit/V9_rebooking/{key}")
    )
    if recover or twice:
        u = np.ones_like(shared.capacity_uniforms)
        if recover:
            u[0, :2] = 0
        if twice:
            u[1, -1] = 0
        u.flags.writeable = False
        shared = replace(shared, capacity_uniforms=u)
    return shared


def prices(c):
    return {
        carrier.code: PriceAction({j: 100 if i == 0 else 300 for j in c.products_of[i]})
        for i, carrier in enumerate(c.spec.carriers)
    }


def buy_b(e):
    c = e.compiled
    j = max(c.products_of[1], key=lambda j: len(c.spec.products[j].legs))
    return e.step(
        prices(c),
        _requests=[Request(0, int(c.product_market[j]), 0, 0, 0, 0)],
        _choices=[j],
    )


def empty(e):
    return e.step(prices(e.compiled), _requests=[], _choices=[])


def policy(window=2, interline=True):
    return RebookingPolicy(window, 2, 100, interline)


def conserved(e):
    r = e.conservation()
    assert r["original_booked_passengers"] == r["active_plus_pending_plus_unserved"]
    assert r["active_passengers"] == r["unique_active_passengers"]
    for s in e.world.carriers:
        assert s.cash == pytest.approx(s.revenue - s.refunds - s.penalties)
    return r


@pytest.mark.parametrize("overlay", [False, True])
def test_no_shock_has_identical_world_trace_metrics_and_no_extra_random_calls(
    overlay, monkeypatch
):
    c = model()
    shared = tape(c)
    native = MarketEngine(c)
    engine = RebookingMarketEngine(c, rebooking=policy() if overlay else None)

    def forbidden(*args):
        raise AssertionError("no stranded passenger may draw re-entry randomness")

    monkeypatch.setattr(engine, "_marks", forbidden)
    for e in [native, engine]:
        e.reset(tape=shared)
        while not e.terminal():
            e.step(prices(c))
    assert pickle.dumps(
        (native.world, native.trace(), native.metrics()), protocol=4
    ) == pickle.dumps((engine.world, engine.trace(), engine.metrics()), protocol=4)
    assert not engine.rebooking_history


def test_interline_rebooks_one_connected_passenger_once_through_native_ledger():
    c = model()
    e = RebookingMarketEngine(c, exits=(RivalExit("B", 1),), rebooking=policy())
    e.reset(tape=tape(c))
    buy_b(e)
    assert len(e.stranded) == 1 and e.world.carriers[1].refunds == 300
    outcome = empty(e)
    assert len(outcome.purchases) == 1
    row = e.rebooking_history[0]
    assert row["stage"] == "interline" and row["fare"] == 100
    assert row["alpha"] == c.spec.segments[0].classes[0].alpha / 2
    assert len(c.spec.products[row["chosen"]].legs) == 2
    r = conserved(e)
    assert r["original_booked_passengers"] == r["active_rebooked_passengers"] == 1
    assert r["native_ticket_count"] == 2
    empty(e)
    assert len(e.rebooking_history) == 1
    assert conserved(e)["active_passengers"] == 1


def test_own_carrier_priority_precedes_cheaper_interline_product_after_recovery():
    c = model(recover=True)
    e = RebookingMarketEngine(c, rebooking=policy())
    e.reset(tape=tape(c, recover=True))
    buy_b(e)
    assert len(e.stranded) == 1
    empty(e)
    row = e.rebooking_history[-1]
    assert row["stage"] == "own" and c.product_carrier[row["chosen"]] == 1
    assert row["fare"] == 300  # cheaper A=100 does not jump the own-carrier tier
    assert conserved(e)["active_rebooked_passengers"] == 1


def test_disabled_interline_expires_without_rebooking_or_extending_deadline():
    c = model()
    e = RebookingMarketEngine(c, exits=(RivalExit("B", 1),), rebooking=policy(1, False))
    e.reset(tape=tape(c))
    buy_b(e)
    assert conserved(e)["pending_stranded"] == 1
    empty(e)
    assert e.rebooking_history[0]["chosen"] is None
    assert conserved(e)["unserved_stranded"] == 1
    empty(e)
    assert len(e.rebooking_history) == 1
    assert e.world.next_record_id == 1


def test_replacement_canceled_at_same_transition_keeps_one_passenger_and_original_deadline():
    c = model(twice=True)
    e = RebookingMarketEngine(c, exits=(RivalExit("B", 1),), rebooking=policy(2))
    e.reset(tape=tape(c, twice=True))
    buy_b(e)
    empty(e)
    assert len(e.stranded) == 2  # two tickets, ONE physical passenger
    r = conserved(e)
    assert r["original_booked_passengers"] == r["pending_stranded"] == 1
    assert r["replacement_tickets_ever"] == 1
    assert e._passengers[0].deadline == 3
    empty(e)
    assert conserved(e)["unserved_stranded"] == 1
    assert e._passengers[0].deadline == 3
    assert e.world.carriers[0].revenue == e.world.carriers[0].refunds == 100


def test_snapshot_branch_and_restore_reproduce_named_random_choices_and_lineages():
    c = model()
    e = RebookingMarketEngine(c, exits=(RivalExit("B", 1),), rebooking=policy())
    e.reset(tape=tape(c))
    buy_b(e)
    saved = e.snapshot()
    branch = e.branch()
    assert branch.tape is e.tape and branch._passengers is not e._passengers
    empty(branch)
    assert e.world.next_record_id == 1
    empty(e)
    expected = pickle.dumps(
        (e.world, e.rebooking_history, e.conservation()), protocol=4
    )
    assert expected == pickle.dumps(
        (branch.world, branch.rebooking_history, branch.conservation()), protocol=4
    )
    e.restore(saved)
    empty(e)
    assert expected == pickle.dumps(
        (e.world, e.rebooking_history, e.conservation()), protocol=4
    )
    branch.rebooking_history[0]["fare"] = -100
    assert e.rebooking_history[0]["fare"] == 100


def test_crn_marks_use_original_request_coordinates_instead_of_endogenous_ticket_id():
    c = model()
    e = RebookingMarketEngine(c, exits=(RivalExit("B", 1),), rebooking=policy())
    e.reset(tape=tape(c))
    buy_b(e)
    p = e._passengers[0]
    q = replace(p, original=replace(p.original, record_id=916))
    a, ga = e._marks(p, 1)
    b, gb = e._marks(q, 1)
    assert a == b and np.array_equal(ga, gb)
    assert not np.array_equal(ga, e._marks(p, 2)[1])


def test_cap_applies_to_replacement_sale_and_full_money_identity_survives():
    c = model()
    refs = tuple((c.spec.products[j].id, 20) for j in c.products_of[0])
    e = RebookingMarketEngine(
        c,
        exits=(RivalExit("B", 1),),
        rebooking=policy(),
        cap_policy=CapPolicy((CapWindow(0, 5, refs),)),
    )
    e.reset(tape=tape(c))
    buy_b(e)
    empty(e)
    assert e.rebooking_history[0]["fare"] == 26
    assert conserved(e)["original_booked_passengers"] == 1


def test_rebooking_in_one_outcome_callback_and_actor_sees_no_queue_or_truth():
    class Spy:
        def reset(self, context, rng):
            self.products = context.compiled.products_of[context.carrier]
            self.calls = 0
            self.outcomes = []

        def act(self, obs):
            assert obs.privileged is None and not hasattr(obs, "stranded")
            self.calls += 1
            return PriceAction({j: 100 for j in self.products})

        def observe_outcome(self, outcome):
            self.outcomes.append(outcome)

        def memory(self):
            return ()

    c = model()
    spy = Spy()
    e = RebookingMarketEngine(
        c, {"A": spy}, exits=(RivalExit("B", 1),), rebooking=policy()
    )
    e.reset(tape=tape(c))
    j = max(c.products_of[1], key=lambda j: len(c.spec.products[j].legs))
    b_only = {"B": prices(c)["B"]}
    e.step(
        b_only,
        _requests=[Request(0, int(c.product_market[j]), 0, 0, 0, 0)],
        _choices=[j],
    )
    e.step(b_only, _requests=[], _choices=[])
    assert spy.calls == len(spy.outcomes) == 2
    assert len(spy.outcomes[-1]["purchases"]) == 1
    assert spy.outcomes[-1]["cash_change"] == 100
    conserved(e)


def test_default_no_policy_preserves_frozen_exit_executor():
    c = model()
    a = DisruptedMarketEngine(c, exits=(RivalExit("B", 1),))
    b = RebookingMarketEngine(c, exits=a.exits)
    for e in [a, b]:
        e.reset(tape=tape(c))
        buy_b(e)
        empty(e)
    assert pickle.dumps((a.world, a.trace()), protocol=4) == pickle.dumps(
        (b.world, b.trace()), protocol=4
    )


@pytest.mark.parametrize("args", [(0, 2), (1, 1), (1, float("nan")), (1, 2, -1)])
def test_invalid_high_wtp_or_window_is_rejected(args):
    with pytest.raises(ValueError):
        RebookingPolicy(*args)


def test_two_stranded_passengers_share_one_seat_without_creating_extra_passengers():
    c = model()
    e = RebookingMarketEngine(c, exits=(RivalExit("B", 1),), rebooking=policy())
    shared = tape(c)
    before = pickle.dumps(shared, protocol=4)
    e.reset(tape=shared)
    e.world.carriers[0].capacity[:] = 1  # explicit unit-fixture capacity
    j = max(c.products_of[1], key=lambda j: len(c.spec.products[j].legs))
    requests = [Request(k, int(c.product_market[j]), 0, 0, 0, 0) for k in range(2)]
    e.step(prices(c), _requests=requests, _choices=[j, j])
    outcome = empty(e)
    assert len(outcome.purchases) == 1
    r = conserved(e)
    assert r["original_booked_passengers"] == 2
    assert r["active_rebooked_passengers"] == r["pending_stranded"] == 1
    assert before == pickle.dumps(shared, protocol=4)
