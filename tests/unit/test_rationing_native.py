"""A.8/B0/V9: native current-mark/O&D/free-capacity pool, no booking adoption."""

import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import pytest

from airline_market.agents.fixed import FixedCoefficientController
from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.rationing import Applicant, Offer, priority_allocation
from airline_market.sim.arrivals import period_requests
from airline_market.sim.engine import MarketEngine
from airline_market.sim.ledger import inventory
from airline_market.sim.tape import ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

if os.environ.get("NATIVE_POOL_SOURCE"):
    s = importlib.util.spec_from_file_location(
        "staged_native_pool", os.environ["NATIVE_POOL_SOURCE"]
    )
    module = importlib.util.module_from_spec(s)
    sys.modules[s.name] = module
    s.loader.exec_module(module)
else:
    from airline_market.policy import rationing_native as module


def fixture():
    base = screen_spec(
        {
            "T": 5,
            "actions": 2,
            "n_spokes": 1,
            "capacity": 20,
            "capacity_support": [3, 4],
        }
    )
    demand = replace(
        base.demand,
        mode="poisson",
        arrival_probability=None,
        market_weights=None,
        intensity={m.id: PeriodSchedule.parse({"constant": 2.0}) for m in base.markets},
    )
    c = compile_market(replace(base, demand=demand))
    e = MarketEngine(c)
    e.reset(
        tape=ExogenousTape.generate(
            c, 31, registry=StreamRegistry(27974, "unit/V9_native_pool")
        )
    )
    return e


def actions(e):
    return {
        x.code: PriceAction(
            {j: 200 if i == 0 else 300 for j in e.compiled.products_of[i]}
        )
        for i, x in enumerate(e.compiled.spec.carriers)
    }


def score_map(e):
    c = e.compiled
    t = e.world.t
    return {
        ((t, r.gumbel_row, r.market), c.spec.products[j].id): "1.2500000000000000000001"
        for r in period_requests(c, e.tape, t, e.world.exo)
        for j in c.market_products[r.market]
    }


def capture(e, prices, scores=None):
    return module.native_ordinary_pool(
        e,
        posted_prices=prices,
        allocation_scores=score_map(e) if scores is None else scores,
        score_definition="unit declared score, not empirical welfare",
    )


def test_full_native_poisson_pool_and_O_and_D_correspondence():
    e = fixture()
    prices = e.post_actions(actions(e))
    before = pickle.dumps(e.snapshot())
    pool = capture(e, prices)
    actual = tuple(period_requests(e.compiled, e.tape, 0, e.world.exo))
    assert (
        pool.requests == actual
        and len(actual) > 1
        and len(pool.problem.applicants) == len(actual)
    )
    for a, r in zip(pool.problem.applicants, actual):
        assert a.key == (0, r.gumbel_row, r.market)
        assert tuple(o.product for o in a.offers) == tuple(
            e.compiled.spec.products[j].id for j in e.compiled.market_products[r.market]
        )
    assert (
        pickle.dumps(e.snapshot()) == before
        and pool.binding_cap is None
        and not pool.native_booking_adopted
    )
    module.assert_pool_current(e, pool)


def test_no_lottery_witness_matches_native_actual_mark_sequential_choices():
    e = fixture()
    for state in e.world.carriers:
        state.capacity[:] = 2
    prices = e.post_actions(actions(e))
    pool = capture(e, prices)
    out = priority_allocation(pool.problem, binding_cap=False)
    native = e.branch()
    native.step(actions(native))
    rows = native.world.request_log
    expected = {
        (0, r.gumbel_row, r.market): None
        if row["chosen"] is None
        else e.compiled.spec.products[row["chosen"]].id
        for r, row in zip(pool.requests, rows)
    }
    assert len(rows) == len(pool.requests) and dict(out["assignments"]) == expected
    assert all(x["remaining"] >= 0 for x in out["capacity"])
    module.assert_pool_current(e, pool)


def test_low_level_native_bridge_refuses_cross_market_and_unknown_O_and_D():
    e = fixture()
    c = e.compiled
    free = {l.id: 2 for l in c.spec.legs}
    j = next(j for j in range(len(c.spec.products)) if int(c.product_market[j]) != 0)
    with pytest.raises(ValueError, match="O&D"):
        module.native_market_problem(
            c, 0, free, (Applicant((0, 0, 0), (Offer(c.spec.products[j].id, 1, 1),)),)
        )
    with pytest.raises(ValueError, match="unknown native request"):
        module.native_market_problem(c, 0, free, (Applicant((0, 0, 999), ()),))


def test_missing_extra_or_cross_market_scores_are_not_zero_filled():
    e = fixture()
    p = e.post_actions(actions(e))
    scores = score_map(e)
    assert scores
    missing = scores.copy()
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="exactly the full"):
        capture(e, p, missing)
    extra = scores.copy()
    extra[((0, 999, 999), "not-a-product")] = 0
    with pytest.raises(ValueError, match="exactly the full"):
        capture(e, p, extra)


def test_active_native_hold_capacity_is_subtracted_after_reentry():
    e, rs = dated_fixture()
    b = dated.strand(e)
    e.present_cancelled_itinerary(b)
    dated.publish(e, rs)
    prices = e.post_actions(dated.actions(e))
    e._reenter(prices)
    before = pickle.dumps(e.snapshot())
    pool = capture(e, prices)
    for i, state in enumerate(e.world.carriers):
        for l, n in zip(e.compiled.legs_of[i], inventory(state, e.compiled, i)):
            leg = e.compiled.spec.legs[l].id
            assert dict(pool.problem.capacities)[leg] == int(n) - e._reserve_ledger.get(
                leg, {}
            ).get("held", 0)
    assert e.rebooking_history and all(v == 80 for _, v in pool.active_cap_limits)
    assert pool.binding_cap is None and pickle.dumps(e.snapshot()) == before
    module.assert_pool_current(e, pool)


def test_required_reentry_is_not_bypassed_and_held_override_not_inherited():
    e, rs = dated_fixture()
    dated.strand(e)
    dated.publish(e, rs)
    prices = e.post_actions(dated.actions(e))
    with pytest.raises(ValueError, match="re-entry attempts first"):
        capture(e, prices)
    e._reenter(prices)
    e._reserve_eligible = True
    with pytest.raises(ValueError, match="held-seat override"):
        capture(e, prices)


def test_native_posting_and_exact_current_prices_are_caller_protocol():
    e = fixture()
    with pytest.raises(ValueError, match="posting before"):
        capture(e, e._prices_global())
    p = e.post_actions(actions(e))
    bad = p.copy()
    bad[0] += 1
    with pytest.raises(ValueError, match="price vector"):
        capture(e, bad)
    with pytest.raises(ValueError, match="score definition"):
        module.native_ordinary_pool(
            e, posted_prices=p, allocation_scores=score_map(e), score_definition=""
        )


def test_active_cap_ceiling_violation_is_refused_without_binding_inference():
    e, rs = dated_fixture()
    dated.strand(e)
    dated.publish(e, rs)
    p = e.post_actions(dated.actions(e))
    e._reenter(p)
    e.world.carriers[0].posted_prices[0] = 1000
    with pytest.raises(ValueError, match="active native ceiling"):
        capture(e, e._prices_global())


@pytest.mark.parametrize(
    "change", ["price", "stock", "current_mark", "clock", "policy", "proof"]
)
def test_snapshot_refuses_stale_inputs(change):
    e = fixture()
    p = e.post_actions(actions(e))
    pool = capture(e, p)
    if change == "price":
        e.world.carriers[0].posted_prices[0] += 1
    elif change == "stock":
        e.world.carriers[0].capacity[0] -= 1
    elif change == "current_mark":
        g = e.tape.gumbel.copy()
        g[0, 0, 0] += 1
        e.tape = replace(e.tape, gumbel=g)
    elif change == "clock":
        e.world.t += 1
    elif change == "policy":
        e.cap_policy = "changed public contract"
    else:
        e._presented = {999}
    with pytest.raises(ValueError, match="stale"):
        module.assert_pool_current(e, pool)


def test_no_actor_RNG_future_marks_or_booking_calls(monkeypatch):
    e = fixture()
    p = e.post_actions(actions(e))
    t = e.world.t
    original = e.tape

    class CurrentOnly:
        def __init__(self, data):
            self.data = data

        def __getitem__(self, key):
            assert (key[0] if isinstance(key, tuple) else key) == t, (
                "future tape row read"
            )
            return self.data[key]

    e.tape = SimpleNamespace(
        seed=original.seed,
        request_uniforms=CurrentOnly(original.request_uniforms),
        class_uniform=CurrentOnly(original.class_uniform),
        arrival_marks=CurrentOnly(original.arrival_marks),
        gumbel=CurrentOnly(original.gumbel),
        mixed_draws=CurrentOnly(original.mixed_draws),
    )

    def fail(*a, **kw):
        raise AssertionError("unexpected actor/RNG/booking/simulation call")

    monkeypatch.setattr(StreamRegistry, "generator", fail)
    for ctrl in e.controllers.values():
        monkeypatch.setattr(ctrl, "act", fail)
    monkeypatch.setattr(e, "post_actions", fail)
    monkeypatch.setattr(e, "step", fail)
    pool = capture(e, p)
    module.assert_pool_current(e, pool)
    assert pool.requests


def test_exact_json_input_record_preserves_utility_and_objective_scores():
    e = fixture()
    p = e.post_actions(actions(e))
    pool = capture(e, p)
    record = json.loads(json.dumps(module.native_pool_record(pool)))
    assert record["binding_cap"] is None and not record["native_booking_adopted"]
    for person, saved in zip(pool.problem.applicants, record["applicants"]):
        for offer, row in zip(person.offers, saved["offers"]):
            for field, value in [
                ("choice_gain_native_utility", offer.choice_value),
                ("allocation_score", offer.allocation_value),
            ]:
                assert (
                    Fraction(
                        int(row[field]["numerator"]), int(row[field]["denominator"])
                    )
                    == value
                )


def test_outside_preference_and_zero_stock_do_not_drop_applicants():
    e = fixture()
    for state in e.world.carriers:
        state.capacity[:] = 0
    p = e.post_actions(actions(e))
    pool = capture(e, p)
    out = priority_allocation(pool.problem, binding_cap=False)
    assert len(pool.problem.applicants) == len(pool.requests) > 0 and out[
        "unassigned_requests"
    ] == len(pool.requests)
    assert all(o for a in pool.problem.applicants for o in a.offers)


def test_empty_native_current_arrivals_are_known_empty_pool_only():
    e = fixture()
    tape = e.tape
    e.tape = replace(tape, arrival_marks=((),) + tape.arrival_marks[1:])
    p = e.post_actions(actions(e))
    pool = capture(e, p)
    assert pool.requests == pool.problem.applicants == () and pool.binding_cap is None


def test_post_admission_prefix_cannot_be_reinterpreted_as_full_batch():
    e = fixture()
    p = e.post_actions(actions(e))
    e.world.request_log.append({"period": 0})
    with pytest.raises(ValueError, match="already admitted"):
        capture(e, p)


def test_prepared_native_actor_calls_are_not_repeated_by_capture(monkeypatch):
    e = fixture()
    e.controllers = {
        i: FixedCoefficientController(tuple(0 for _ in e.rules[i].spec.theta_bounds))
        for i in e.controllers
    }
    calls = []
    for i, ctrl in e.controllers.items():
        act = ctrl.act

        def spy(obs, i=i, act=act):
            assert obs.privileged is None
            calls.append(i)
            return act(obs)

        monkeypatch.setattr(ctrl, "act", spy)
    prices = e.post_actions()
    before = tuple(calls)
    capture(e, prices)
    assert sorted(calls) == list(range(len(e.controllers))) and tuple(calls) == before


s = importlib.util.spec_from_file_location(
    "native_pool_dated_fixture", "tests/unit/test_triggered_dated.py"
)
dated = importlib.util.module_from_spec(s)
s.loader.exec_module(dated)


def dated_fixture():
    return dated.fixture()


def test_thinned_native_admission_rank_keeps_original_potential_slot():
    e = fixture()
    rows = e.tape.arrival_marks[0]
    assert len(rows) > 1
    changed = (replace(rows[0], thinning=2.0),) + rows[1:]
    e.tape = replace(e.tape, arrival_marks=(changed,) + e.tape.arrival_marks[1:])
    prices = e.post_actions(actions(e))
    pool = capture(e, prices)
    assert pool.requests[0].rank == 0 and pool.requests[0].gumbel_row == 1
    assert pool.problem.applicants[0].key[1] == 1
    assert all(
        a.key == (0, r.gumbel_row, r.market)
        for a, r in zip(pool.problem.applicants, pool.requests)
    )
