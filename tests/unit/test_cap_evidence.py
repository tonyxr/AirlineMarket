"""A.8/B0/V9: same-call proposals/clipping, full postings and native identities."""

import importlib.util
import json
import os
import pickle
import sys
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from airline_market.agents.fixed import FixedCoefficientController
from airline_market.core.types import PriceAction
from airline_market.policy.rationing_native import native_ordinary_pool
from airline_market.policy.regulator import CapPolicy, CapWindow
from airline_market.sim.arrivals import period_requests
from tests.unit import test_triggered_dated as fixture_lib

if os.environ.get("CAP_EVIDENCE_SOURCE"):
    spec = importlib.util.spec_from_file_location(
        "staged_cap_evidence", os.environ["CAP_EVIDENCE_SOURCE"]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
else:
    from airline_market.policy import cap_evidence as module


def fixture(**kwargs):
    old, reports = fixture_lib.fixture(**kwargs)
    e = module.CapEvidenceMarketEngine(
        old.compiled,
        trigger_rule=old.trigger.rule if old.trigger else None,
        mechanical_benchmark=old.trigger.benchmark if old.trigger else None,
        booking_clock=old.booking_clock,
        leg_departures=old.leg_departures,
        dated_rebooking=old.dated_rebooking,
        dated_menu=old.dated_menu,
        dated_reserve=old.dated_reserve,
        exits=old.exits,
        cap_policy=old.cap_policy,
    )
    e.reset(tape=old.tape)
    return e, reports, old


def policy(fare, *, start=0, stop=5, emergency=False, declarations=()):
    e, _, _ = fixture(trigger=False)
    refs = tuple((p.id, fare) for p in e.compiled.spec.products)
    return CapPolicy(
        (
            CapWindow(
                start=start,
                stop=stop,
                reference_fares=refs,
                multiplier=1,
                requires_emergency=emergency,
            ),
        ),
        declarations,
    )


def acts(e, fare=400):
    return {
        x.code: PriceAction({j: fare for j in e.compiled.products_of[i]})
        for i, x in enumerate(e.compiled.spec.carriers)
    }


def pool(e, prices):
    c = e.compiled
    t = e.world.t
    scores = {
        ((t, r.gumbel_row, r.market), c.spec.products[j].id): 1
        for r in period_requests(c, e.tape, t, e.world.exo)
        for j in c.market_products[r.market]
    }
    return native_ordinary_pool(
        e,
        posted_prices=prices,
        allocation_scores=scores,
        score_definition="unit declared score",
    )


@pytest.mark.parametrize(
    "fare,expected,binding", [(400, 100, True), (100, 100, False), (60, 60, False)]
)
def test_static_active_is_distinct_from_strictly_binding(fare, expected, binding):
    e, _, _ = fixture(trigger=False, cap_policy=policy(100))
    p = e.post_actions(acts(e, fare))
    r = e.cap_postings()[-1]
    assert all(
        x.native_proposal == fare and x.static_ceiling == 100 and x.executed == expected
        for x in r.products
    )
    assert r.any_price_constraint_binding is binding
    assert all(
        x.static_clipped is binding and not x.trigger_clipped for x in r.products
    )
    assert e.pool_cap_evidence(pool(e, p)) == r


def test_exact_raw_proposal_is_captured_before_both_native_clip_stages():
    e, reports, _ = fixture(start=0, cap_policy=policy(100))
    fixture_lib.publish(e, reports)
    e.post_actions(acts(e, 400))
    rows = e.cap_postings()[-1].products
    assert all(
        (x.native_proposal, x.after_static, x.executed) == (400, 100, 80)
        and x.static_clipped
        and x.trigger_clipped
        for x in rows
    )
    assert all(float(x["posted_fare"]) == 100 for x in e.trigger.history)


def test_stronger_static_cap_can_prevent_native_trigger_activation():
    e, reports, _ = fixture(start=0, cap_policy=policy(50))
    fixture_lib.publish(e, reports)
    e.post_actions(acts(e, 400))
    assert all(
        x.after_static == x.executed == 50 and x.trigger_ceiling is None
        for x in e.cap_postings()[-1].products
    )
    assert all(not x["fired_now"] for x in e.trigger.history)


def test_repricing_active_trigger_below_ceiling_is_not_binding():
    e, reports, _ = fixture(start=0)
    fixture_lib.publish(e, reports)
    e.step(acts(e, 400), _requests=[])
    fixture_lib.publish(e, reports)
    e.post_actions(acts(e, 60))
    r = e.cap_postings()[-1]
    assert r.period == 1 and not r.any_price_constraint_binding
    assert all(
        x.trigger_ceiling == 80 and x.native_proposal == x.executed == 60
        for x in r.products
    )


def test_window_and_declaration_are_native_inputs_not_inferred_from_clipping():
    e, _, _ = fixture(trigger=False, cap_policy=policy(100, emergency=True))
    e.post_actions(acts(e))
    assert all(x.static_ceiling is None for x in e.cap_postings()[-1].products)
    e, _, _ = fixture(
        trigger=False,
        cap_policy=policy(100, start=1, emergency=True, declarations=((1, 2),)),
    )
    e.step(acts(e), _requests=[])
    assert not e.cap_postings()[-1].any_price_constraint_binding
    e.post_actions(acts(e))
    assert e.cap_postings()[-1].any_price_constraint_binding
    e.step(acts(e), _requests=[])
    e.post_actions(acts(e))
    assert not e.cap_postings()[-1].any_price_constraint_binding


def test_native_controller_and_price_rule_are_each_called_once_per_actual_posting():
    e, _, _ = fixture(trigger=False, cap_policy=policy(100))
    counts = {}
    rule_counts = {}
    for i in e.controllers:
        ctrl = FixedCoefficientController((0,) * len(e.rules[i].spec.theta_bounds))
        original = ctrl.act

        def act(obs, i=i, original=original):
            counts[i] = counts.get(i, 0) + 1
            return original(obs)

        ctrl.act = act
        e.controllers[i] = ctrl
        original_price = e.rules[i].prices

        def prices(theta, obs, i=i, original_price=original_price):
            rule_counts[i] = rule_counts.get(i, 0) + 1
            return original_price(theta, obs)

        e.rules[i].prices = prices
    e.post_actions()
    assert counts == rule_counts == {0: 1, 1: 1}
    assert all(x.native_proposal >= x.executed for x in e.cap_postings()[-1].products)


def test_observed_leader_proposal_is_retained_without_repeating_leader_actor():
    e, _, _ = fixture(trigger=False, cap_policy=policy(100))
    e.begin_period(acts(e, 450))
    leaders = set(e.world.posted_leaders)
    assert leaders
    e.post_actions(acts(e, 60))
    rows = {x.product: x for x in e.cap_postings()[-1].products}
    for i in e.controllers:
        for j in e.compiled.products_of[i]:
            assert rows[e.compiled.spec.products[j].id].native_proposal == (
                450 if i in leaders else 60
            )


def test_failed_posting_retains_prior_receipt_but_does_not_certify_current_pool():
    e, _, _ = fixture(trigger=False, cap_policy=policy(100))
    e.post_actions(acts(e))
    old = e.cap_postings()
    with pytest.raises(KeyError):
        e.post_actions({})
    assert e.cap_postings() == old
    p = pool(e, e._prices_global())
    with pytest.raises(ValueError, match="completed native posting"):
        e.pool_cap_evidence(p)


def test_missing_public_report_has_no_completed_receipt_or_native_proposal():
    e, _, _ = fixture(start=0)
    with pytest.raises(ValueError, match="public regulator report"):
        e.post_actions(acts(e))
    assert e.cap_postings() == () and e._cap_action_rows == {}


def test_current_pool_join_refuses_changed_stock_and_old_period():
    e, _, _ = fixture(trigger=False, cap_policy=policy(100))
    p = pool(e, e.post_actions(acts(e)))
    e.world.carriers[0].capacity[0] -= 1
    with pytest.raises(ValueError, match="stale"):
        e.pool_cap_evidence(p)


def test_no_policy_native_world_trace_and_profit_identity_on_actual_arrivals():
    e, _, old = fixture(trigger=False)
    for _ in range(5):
        e.step(acts(e))
        old.step(acts(old))
    assert pickle.dumps(e.world) == pickle.dumps(old.world)
    assert pickle.dumps(e.trace()) == pickle.dumps(old.trace())
    assert e.metrics() == old.metrics()
    assert not any(r.any_price_constraint_binding for r in e.cap_postings())


def test_trigger_dated_rescue_and_holds_remain_same_native_world():
    e, reports, old = fixture(start=1, cap_policy=policy(120))
    original = fixture_lib.strand(e)
    fixture_lib.strand(old)
    fixture_lib.publish(e, reports)
    fixture_lib.publish(old, reports)
    e.step(acts(e))
    old.step(acts(old))
    assert pickle.dumps(e.world) == pickle.dumps(old.world)
    assert e.reserve_balance() == old.reserve_balance()
    assert e.conservation() == old.conservation()
    assert (
        e._passengers[original.record_id].active_record
        == old._passengers[original.record_id].active_record
    )
    assert e.cap_postings()[-1].any_price_constraint_binding


def test_nested_snapshot_branch_reset_keep_independent_evidence_and_contracts():
    e, reports, _ = fixture(start=0)
    fixture_lib.publish(e, reports)
    e.post_actions(acts(e))
    snap = e.snapshot()
    other = e.branch()
    assert type(other) is module.CapEvidenceMarketEngine
    other.post_actions(acts(other, 60))
    assert len(other.cap_postings()) == 2 and len(e.cap_postings()) == 1
    other.restore(snap)
    assert other.cap_postings() == e.cap_postings()
    e.reset(tape=e.tape)
    assert e.cap_postings() == () and not e._cap_posting_complete
    with pytest.raises(TypeError):
        e.restore(snap.native)


def test_frozen_product_posting_records_are_lossless_analyst_json_only():
    e, _, _ = fixture(trigger=False, cap_policy=policy("100.123"))
    e.post_actions(acts(e, 400.123456789))
    r = e.cap_postings()[-1]
    x = module.posting_record(r)
    loaded = json.loads(json.dumps(x))
    assert all(
        y["native_proposal"] == 400.123456789 and y["executed"] == 100.12
        for y in loaded["products"]
    )
    assert not loaded["excess_demand_measured"] and not loaded["lottery_group_adopted"]
    with pytest.raises(FrozenInstanceError):
        r.products[0].executed = 12
    with pytest.raises(TypeError):
        module.posting_record(SimpleNamespace())


def test_snapshot_refuses_different_public_static_contract_before_restore():
    e, _, _ = fixture(trigger=False, cap_policy=policy(100))
    e.post_actions(acts(e))
    other, _, _ = fixture(trigger=False, cap_policy=policy(50))
    before = pickle.dumps(other.world)
    with pytest.raises(ValueError, match="static cap contract"):
        other.restore(e.snapshot())
    assert pickle.dumps(other.world) == before


def test_prior_receipt_cannot_certify_an_incomplete_direct_action_revision():
    e, _, _ = fixture(trigger=False, cap_policy=policy(100))
    e.post_actions(acts(e))
    i = 0
    obs = e.observe(e.compiled.spec.carriers[i].code, e.controller_arm(i))
    e._action_prices(i, acts(e, 500)[e.compiled.spec.carriers[i].code], obs)
    p = pool(e, e._prices_global())
    with pytest.raises(ValueError, match="completed native posting"):
        e.pool_cap_evidence(p)
    assert len(e.cap_postings()) == 1


def test_full_binding_policy_run_has_identical_native_trace_cash_and_quota():
    e, reports, old = fixture(start=0, cap_policy=policy(110))
    for _ in range(5):
        fixture_lib.publish(e, reports)
        fixture_lib.publish(old, reports)
        e.step(acts(e))
        old.step(acts(old))
    assert pickle.dumps(e.world) == pickle.dumps(old.world)
    assert pickle.dumps(e.trace()) == pickle.dumps(old.trace())
    assert e.metrics() == old.metrics() and e.reserve_balance() == old.reserve_balance()
    assert all(row.any_price_constraint_binding for row in e.cap_postings())
