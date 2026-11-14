"""A.14/B0/V9: trigger boundaries, public-only benchmark contract and native money."""

import hashlib
import importlib.util
import os
import pickle
import sys
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.regulator import CapPolicy, CapWindow
from airline_market.policy.reserve import ReserveMarketEngine
from airline_market.sim.arrivals import Request
from airline_market.sim.disruption import RivalExit
from airline_market.sim.tape import ExogenousTape
from airline_market.spec import compile_market

if os.environ.get("TRIGGER_SOURCE"):
    s = importlib.util.spec_from_file_location(
        "staged_trigger", os.environ["TRIGGER_SOURCE"]
    )
    module = importlib.util.module_from_spec(s)
    sys.modules[s.name] = module
    s.loader.exec_module(module)
else:
    from airline_market.policy import trigger as module


def report(
    t=1,
    before=20,
    after=10,
    remaining=9,
    emergency=False,
    leg="leg",
    refs=(("p", 100),),
):
    return module.PublicTriggerReport(
        t, ((leg, before),), ((leg, after),), ((leg, remaining),), refs, emergency
    )


def table(reports, mechanical=None):
    return module.PublicMechanicalTable(
        tuple(
            module.MechanicalFareRow(
                r.period,
                r.remaining_seats,
                r.reference_fares,
                mechanical if mechanical is not None else (("p", 100),),
            )
            for r in reports
        ),
        hashlib.sha256(b"unit pre-shock demand").hexdigest(),
        hashlib.sha256(b"unit frozen rival").hexdigest(),
    )


def rule(k=".3", tau=".3", start=1, stop=5, window=2):
    return module.TriggerRule(start, stop, k, tau, window)


@pytest.mark.parametrize(
    "loss,posted,fired",
    [(8, 200, False), (7, 130, False), (7, "130.0001", True), (7, 200, True)],
)
def test_capacity_inclusive_fare_strict_threshold(loss, posted, fired):
    r = report(before=10, after=loss, remaining=loss)
    t = module.DesignedTrigger(rule(), table([r]))
    result = t.evaluate(r, (("p", posted),))
    assert bool(result) == fired and t.history[-1]["fired_now"] == fired
    if fired:
        assert result == {"p": 130}


def test_emergency_is_or_condition_and_infinite_tau_never_fires():
    r = report(after=20, remaining=20, emergency=True)
    t = module.DesignedTrigger(rule(k=".99"), table([r]))
    assert t.evaluate(r, (("p", 200),)) == {"p": 130}
    disabled = module.DesignedTrigger(rule(tau="Infinity"), table([r]))
    assert (
        disabled.evaluate(r, (("p", 100000),)) == {}
        and not disabled.history
        and not disabled.active_until
    )


def test_zero_kappa_zero_tau_caps_at_mechanical_benchmark_without_raising_cheaper_fare():
    r = report(after=20, remaining=20)
    t = module.DesignedTrigger(rule(k=0, tau=0), table([r]))
    assert t.evaluate(r, (("p", 100),)) == {}
    assert t.evaluate(r, (("p", 200),)) == {"p": 100}
    assert t.evaluate(r, (("p", 0),)) == {
        "p": 100
    }  # cap is an upper bound, not executed price


def test_inventory_depletion_alone_is_not_capacity_shock():
    r = report(after=20, remaining=0)
    t = module.DesignedTrigger(rule(k=".01"), table([r]))
    assert r.loss_fraction == 0 and t.evaluate(r, (("p", 200),)) == {}


def test_window_expiry_no_extension_and_rearm_use_current_public_benchmark():
    rs = [report(t) for t in [0, 1, 2, 3, 4, 5]]
    t = module.DesignedTrigger(rule(), table(rs))
    assert t.evaluate(rs[0], (("p", 200),)) == {}
    assert t.evaluate(rs[1], (("p", 200),)) == {"p": 130} and t.active_until["p"] == 3
    assert t.evaluate(rs[2], (("p", 300),)) == {"p": 130} and t.active_until["p"] == 3
    assert t.evaluate(rs[3], (("p", 100),)) == {}
    assert t.evaluate(rs[4], (("p", 200),)) == {"p": 130} and t.active_until["p"] == 5
    assert t.evaluate(rs[5], (("p", 300),)) == {}
    with pytest.raises(ValueError, match="backwards"):
        t.evaluate(rs[2], (("p", 200),))


def test_active_window_updates_benchmark_from_fresh_public_remaining_seats():
    r1 = report(1)
    r2 = report(2, remaining=5)
    tb = module.PublicMechanicalTable(
        (
            module.MechanicalFareRow(
                1, r1.remaining_seats, r1.reference_fares, (("p", 100),)
            ),
            module.MechanicalFareRow(
                2, r2.remaining_seats, r2.reference_fares, (("p", 110),)
            ),
        ),
        "a" * 64,
        "b" * 64,
    )
    t = module.DesignedTrigger(rule(tau=0), tb)
    assert t.evaluate(r1, (("p", 200),)) == {"p": 100}
    assert t.evaluate(r2, (("p", 200),)) == {"p": 110}


def test_public_table_has_exact_key_no_hidden_or_nearest_state_fallback():
    r = report()
    tb = table([r])
    assert tb.fares(r) == {"p": 100}
    with pytest.raises(KeyError):
        tb.fares(replace(r, remaining_seats=(("leg", 8),)))
    with pytest.raises(TypeError):
        tb.fares({"period": 1, "hidden_stock": 9})
    with pytest.raises((AttributeError, TypeError, FrozenInstanceError)):
        r.hidden_stock = 9

    class Trap:
        def __getattr__(self, k):
            raise AssertionError("hidden field read: " + k)

    with pytest.raises(TypeError):
        tb.fares(Trap())


@pytest.mark.parametrize(
    "args",
    [
        (1, 1, 0.3, 0.3, 1),
        (1, 5, -1, 0.3, 1),
        (1, 5, 2, 0.3, 1),
        (1, 5, 0.3, -0.1, 1),
        (1, 5, 0.3, "NaN", 1),
        (1, 5, 0.3, 0.3, 0),
    ],
)
def test_invalid_rules_fail(args):
    with pytest.raises(ValueError):
        module.TriggerRule(*args)


def test_invalid_reports_duplicate_tables_and_missing_provenance_fail():
    with pytest.raises(ValueError):
        report(after=1, remaining=2)
    with pytest.raises(ValueError):
        report(before=True)
    r = report()
    row = module.MechanicalFareRow(
        1, r.remaining_seats, r.reference_fares, (("p", 100),)
    )
    with pytest.raises(ValueError):
        module.PublicMechanicalTable((row, row), "a" * 64, "b" * 64)
    with pytest.raises(ValueError):
        module.PublicMechanicalTable((row,), "missing", "b" * 64)


def fixture(tau=0, k=0, controllers=None, **kwargs):
    c = compile_market(
        screen_spec(
            {
                "T": 5,
                "actions": 2,
                "n_spokes": 1,
                "capacity": 20,
                "capacity_support": [2, 3],
            }
        )
    )
    products = tuple((p.id, 100) for p in c.spec.products)
    leg = c.spec.legs[0].id
    reports = [report(t, leg=leg, refs=products) for t in range(5)]
    benchmark = table(reports, products)
    e = module.TriggerMarketEngine(
        c,
        controllers,
        trigger_rule=rule(k, tau),
        mechanical_benchmark=benchmark,
        **kwargs,
    )
    e.reset(
        tape=ExogenousTape.generate(
            c, 24, registry=StreamRegistry(21875, "unit/V9_trigger")
        )
    )
    return e, reports


def actions(e, fare=400):
    c = e.compiled
    return {
        x.code: PriceAction({j: fare for j in c.products_of[i]})
        for i, x in enumerate(c.spec.carriers)
    }


def empty(e):
    return e.step(actions(e), _requests=[])


def advance_one(e, reports):
    empty(e)
    e.publish_regulator_report(reports[1])


def test_native_posting_quotes_bookings_and_refund_use_capped_price():
    e, rs = fixture(exits=(RivalExit("B", 2),))
    advance_one(e, rs)
    c = e.compiled
    j = c.products_of[1][0]
    r = e.step(
        actions(e),
        _requests=[Request(0, int(c.product_market[j]), 0, 0, 0, 0)],
        _choices=[j],
    )
    assert (
        r.purchases[0].fare == 100
        and e.world.carriers[1].revenue == e.world.carriers[1].refunds == 100
    )
    assert e.world.carriers[1].cash == -e.world.carriers[1].penalties == -25
    assert all(p <= 100 for p in e.trace()[-1]["prices"])


def test_decimal_floor_never_exceeds_unrounded_threshold():
    e, rs = fixture(tau=".234567")
    advance_one(e, rs)
    c = e.compiled
    j = c.products_of[0][0]
    p = e.step(
        actions(e),
        _requests=[Request(0, int(c.product_market[j]), 0, 0, 0, 0)],
        _choices=[j],
    ).purchases[0]
    assert p.fare == 123.45 and Decimal(str(p.fare)) <= Decimal("123.4567")


def test_existing_cap_can_remain_cheaper_than_trigger():
    e, rs = fixture()
    products = tuple((p.id, 50) for p in e.compiled.spec.products)
    e.cap_policy = CapPolicy((CapWindow(1, 4, products, 1),))
    advance_one(e, rs)
    empty(e)
    assert all(p == 50 for p in e.trace()[-1]["prices"])


def test_no_rule_and_infinite_tau_have_native_byte_identity_without_reports():
    template, _ = fixture()
    for rule_arg, tb in [
        (None, None),
        (rule(tau="Infinity"), template.trigger.benchmark),
    ]:
        e = module.TriggerMarketEngine(
            template.compiled, trigger_rule=rule_arg, mechanical_benchmark=tb
        )
        base = ReserveMarketEngine(template.compiled)
        for x in [e, base]:
            x.reset(tape=template.tape)
            for _ in range(3):
                empty(x)
        assert pickle.dumps((e.world, e.trace(), e.metrics())) == pickle.dumps(
            (base.world, base.trace(), base.metrics())
        )
        assert not e.trigger or not e.trigger.history


def test_missing_report_is_not_reconstructed_from_hidden_world():
    e, rs = fixture()
    empty(e)
    before = tuple(s.posted_prices.copy() for s in e.world.carriers)
    with pytest.raises(ValueError, match="fresh public"):
        empty(e)
    assert all(
        np.array_equal(s.posted_prices, v) for s, v in zip(e.world.carriers, before)
    )
    e.publish_regulator_report(rs[1])
    empty(e)
    with pytest.raises(ValueError, match="current"):
        e.publish_regulator_report(rs[1])


def test_same_public_report_caps_identically_despite_different_private_rival_stock():
    first, rs = fixture()
    second, _ = fixture()
    for e in [first, second]:
        advance_one(e, rs)
    second.world.carriers[1].capacity[:] = 0
    for e in [first, second]:
        empty(e)
    assert first.trace()[-1]["prices"] == second.trace()[-1]["prices"]
    assert first.trigger.history == second.trigger.history


def test_snapshot_branch_owns_trigger_history_reports_and_native_state():
    e, rs = fixture()
    advance_one(e, rs)
    empty(e)
    e.publish_regulator_report(rs[2])
    before = e.snapshot()
    b = e.branch()
    assert (
        b.trigger.benchmark is e.trigger.benchmark
        and b.trigger.active_until is not e.trigger.active_until
    )
    empty(b)
    assert len(b.trigger.history) > len(e.trigger.history)
    e.restore(before)
    empty(e)
    assert pickle.dumps((e.world, e.trigger.history)) == pickle.dumps(
        (b.world, b.trigger.history)
    )
    e.reset(tape=e.tape)
    assert (
        not e.trigger.history
        and not e.trigger.active_until
        and not e._regulator_reports
    )


def test_report_cannot_change_after_publication():
    e, rs = fixture()
    advance_one(e, rs)
    with pytest.raises(ValueError, match="immutable"):
        e.publish_regulator_report(replace(rs[1], declared_emergency=True))


def test_one_actor_forward_no_hidden_report_and_native_callback():
    class Spy:
        def reset(self, ctx, rng):
            self.products = ctx.compiled.products_of[ctx.carrier]
            self.calls = 0
            self.outcomes = []

        def act(self, obs):
            assert obs.privileged is None and not hasattr(obs, "regulator_reports")
            self.calls += 1
            return PriceAction({j: 400 for j in self.products})

        def observe_outcome(self, x):
            self.outcomes.append(x)

        def memory(self):
            return ()

    spy = Spy()
    e, rs = fixture(controllers={"A": spy})
    c = e.compiled
    e.step({"B": actions(e)["B"]}, _requests=[])
    e.publish_regulator_report(rs[1])
    j = c.products_of[0][0]
    e.step(
        {"B": actions(e)["B"]},
        _requests=[Request(0, int(c.product_market[j]), 0, 0, 0, 0)],
        _choices=[j],
    )
    assert (
        spy.calls == len(spy.outcomes) == 2 and spy.outcomes[-1]["cash_change"] == 100
    )


def test_public_input_columns_are_fixed_and_only_known_native_ids_are_bound():
    r = report()
    one = module.MechanicalFareRow(
        1, r.remaining_seats, r.reference_fares, (("p", 100),)
    )
    two = module.MechanicalFareRow(2, (("other", 9),), r.reference_fares, (("p", 100),))
    with pytest.raises(ValueError, match="scope"):
        module.PublicMechanicalTable((one, two), "a" * 64, "b" * 64)
    e, _ = fixture()
    bad = table([report()], tuple((p.id, 100) for p in e.compiled.spec.products))
    with pytest.raises(ValueError, match="unknown public"):
        module.TriggerMarketEngine(
            e.compiled, trigger_rule=rule(), mechanical_benchmark=bad
        )


def test_canonical_public_keys_ignore_serialization_order():
    r = module.PublicTriggerReport(
        1,
        (("b", 20), ("a", 20)),
        (("a", 10), ("b", 10)),
        (("b", 9), ("a", 8)),
        (("y", 110), ("x", 100)),
    )
    row = module.MechanicalFareRow(
        1, (("a", 8), ("b", 9)), (("x", 100), ("y", 110)), (("p", 100),)
    )
    assert module.PublicMechanicalTable((row,), "a" * 64, "b" * 64).fares(r) == {
        "p": 100
    }


def test_active_window_is_retained_through_public_capacity_recovery():
    first = report(1)
    second = report(2, after=20, remaining=20)
    t = module.DesignedTrigger(rule(), table([first, second]))
    assert t.evaluate(first, (("p", 200),)) == {"p": 130}
    assert t.evaluate(second, (("p", 200),)) == {"p": 130}
    assert t.history[-1]["capacity_or_declaration"] is False and t.history[-1]["active"]


def test_trigger_composes_with_exit_proof_rescue_reserved_stock_and_native_cash():
    from airline_market.policy.rescue import RescueFareMenu
    from airline_market.policy.reserve import SeatReserve
    from airline_market.sim.rebooking import RebookingPolicy

    template, rs = fixture()
    c = template.compiled
    menu = RescueFareMenu(
        1, 5, "same_route", route_fares=tuple((m.id, 99) for m in c.spec.markets)
    )
    e = module.TriggerMarketEngine(
        c,
        trigger_rule=template.trigger.rule,
        mechanical_benchmark=template.trigger.benchmark,
        seat_reserve=SeatReserve(
            1, 3, ".10", tuple(l.id for l in c.spec.legs if l.carrier == "A")
        ),
        rescue_menu=menu,
        rebooking=RebookingPolicy(3, 2, 100, True),
        exits=(RivalExit("B", 1),),
    )
    e.reset(tape=template.tape)
    j = max(c.products_of[1], key=lambda j: len(c.spec.products[j].legs))
    e.step(
        actions(e),
        _requests=[Request(0, int(c.product_market[j]), 0, 0, 0, 0)],
        _choices=[j],
    )
    e.present_cancelled_itinerary(e.stranded[0].booking)
    e.publish_regulator_report(rs[1])
    empty(e)
    assert (
        e.rebooking_history[-1]["fare"] == 99
        and len(e.rebooking_history[-1]["held_seats_used"]) == 2
    )
    assert all(p <= 100 for p in e.trace()[-1]["prices"])
    assert e.world.carriers[0].cash == 99 and e.world.carriers[1].cash == -25
    for row in e.reserve_balance():
        assert (
            row["initial"]
            == row["held"] + row["used"] + row["released"] + row["capacity_lost"]
        )
