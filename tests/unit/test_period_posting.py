"""A.10/A.5/A.14/B0/V9/B4: NEW138974 / NEW139974 genuine source units."""

import importlib.util
import os
import pickle
import sys
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from airline_market.agents.base import StatelessController
from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.dated_demand import compile_dated_demand
from airline_market.policy.dated_rescue import DatedRebookingPolicy
from airline_market.policy.dated_reserve import UtcBookingClock
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.policy.regulator import CapPolicy, CapWindow
from airline_market.policy.trigger import (
    MechanicalFareRow,
    PublicMechanicalTable,
    PublicTriggerReport,
    TriggerRule,
)
from airline_market.sim.native_causal_information import (
    TIMING,
    DeclaredInputDelivery,
    DeclaredKnownType,
    NativeCausalInformationMarketEngine,
)
from airline_market.sim.received_information import DisclosureField
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

if "PERIOD_POSTING_SOURCE" in os.environ:
    _spec = importlib.util.spec_from_file_location(
        "new_period_posting", os.environ["PERIOD_POSTING_SOURCE"]
    )
    m = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = m
    _spec.loader.exec_module(m)
else:
    from airline_market.sim import period_posting as m


def setup(
    *,
    empty=False,
    trigger=False,
    cap=True,
    engine_type=None,
    limit=16,
    controller_type=None,
):
    # Entirely NEW source declarations and named unit family, no old helper call.
    s = screen_spec(
        {
            "T": 4,
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
    t0 = datetime(2027, 7, 5, tzinfo=timezone.utc)
    clock = UtcBookingClock(tuple(t0 + timedelta(hours=12 * t) for t in range(5)))
    dep = tuple(
        (l.id, t0 + timedelta(hours=84 if l.id == "B_OH0" else 108))
        for l in c.spec.legs
    )
    scenario = replace(
        compile_dated_demand(c, clock, dep),
        physical_shocks=(
            B1DatedShock("NEW138974-exit", "rival_exit", "B_OH0", 3, rival_carrier="B"),
        ),
    )
    market = int(c.product_market[c.product_index["B_OH0"]])
    keys = tuple((0, i, market) for i in range(2))
    kwargs = {
        "causal_fields": (
            DisclosureField("notice", "text", "declared notice", "text"),
        ),
        "causal_known_types": tuple(
            DeclaredKnownType(k, None, "unknown not ignorance", "NEW138974")
            for k in keys
        ),
        "causal_deliveries": tuple(
            DeclaredInputDelivery(
                k,
                t,
                stage,
                f"new-{i}-{t}-{stage}",
                "notice",
                "declaration",
                "NEW138974-source",
                TIMING,
            )
            for i, k in enumerate(keys)
            for t in range(4)
            for stage in ("ordinary", "own", "interline")
            if t == 0 or stage != "ordinary"
        ),
        "causal_definition": "NEW138974 compiled order not real reception",
        "causal_unobserved_definition": "G and preference awareness unprovided",
        "dated_rebooking": DatedRebookingPolicy(12, 20, 100, interline=True),
    }
    if cap:
        kwargs["cap_policy"] = CapPolicy(
            (
                CapWindow(
                    0,
                    2,
                    tuple((p.id, "100") for p in c.spec.products),
                    requires_emergency=True,
                ),
            ),
            ((0, 2),),
        )
    reports = []
    if trigger:
        refs = tuple((p.id, 100) for p in c.spec.products)
        leg = c.spec.legs[0].id
        reports = tuple(
            PublicTriggerReport(t, ((leg, 20),), ((leg, 10),), ((leg, 9),), refs)
            for t in range(4)
        )
        kwargs.update(
            trigger_rule=TriggerRule(0, 4, 0, 0, 2),
            mechanical_benchmark=PublicMechanicalTable(
                tuple(
                    MechanicalFareRow(
                        r.period,
                        r.remaining_seats,
                        r.reference_fares,
                        tuple((p.id, 80) for p in c.spec.products),
                    )
                    for r in reports
                ),
                "a" * 64,
                "b" * 64,
            ),
        )
    if controller_type is not None:
        kwargs["controllers"] = {
            c.carrier_index[c.spec.timing.leaders[0]]: controller_type()
        }
    # Fix complete declarations before one actual tape generation.
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(139974, "unit/V9_period_posting/July5")
    )
    g = tape.gumbel.copy()
    g[:] = -1e9
    g[:, :, -1] = 0
    g[:, :, c.product_index["B_OH0"]] = 1e9
    g[:, 1, -1] = 2e9
    g.setflags(write=False)
    tape = replace(
        tape,
        arrival_marks=(
            (ArrivalMark(0.2, market, 0, 0, 0), ArrivalMark(0.7, market, 0, 0, 0))
            if not empty
            else (),
            (),
            (),
            (),
        ),
        gumbel=g,
    )
    cls = engine_type or m.PeriodPostingMarketEngine
    engine = cls(scenario, posting_call_limit=limit, **kwargs)
    control = NativeCausalInformationMarketEngine(scenario, **kwargs)
    for e in (engine, control):
        e.reset(tape=tape)
    return engine, control, reports


def actions(e, fare=250):
    return {
        x.code: PriceAction(
            {j: float(fare + i * 50) for j in e.compiled.products_of[i]}
        )
        for i, x in enumerate(e.compiled.spec.carriers)
    }


def run(e, reports=(), n=4):
    for _ in range(n):
        if reports:
            e.publish_regulator_report_at(
                e.booking_clock.at(e.world.t), reports[e.world.t]
            )
        e.step(actions(e, 250 + 5 * e.world.t))
    return e.posting_source()


def test_actual_full_parent_trace_metrics_and_rng_identity():
    e, c, reports = setup()
    before = np.random.get_state()
    source = run(e, reports)
    for _ in range(4):
        c.step(actions(c, 250 + 5 * c.world.t))
    assert pickle.dumps(e.world) == pickle.dumps(c.world)
    assert m._same(e.trace(), c.trace()) and m._same(e.metrics(), c.metrics())
    assert m._same(source.body["parent_source"], c._payload())
    after = np.random.get_state()
    assert (
        before[0] == after[0]
        and np.array_equal(before[1], after[1])
        and before[2:] == after[2:]
    )
    assert (
        source.body[
            "native_probability_received_knowledge_preferences_or_CS_authenticated"
        ]
        is False
    )


def test_all_empty_periods_distinct_fares_full_products():
    e, _, _ = setup(empty=True, cap=False)
    source = run(e)
    rows = source.body["periods"]
    assert len(rows) == 4 and all(not r["input_links"] for r in rows)
    assert not m._same(
        rows[2]["postings"][-1]["prices"], rows[3]["postings"][-1]["prices"]
    )
    assert all(
        len(r["postings"][-1]["prices"]) == len(e.compiled.spec.products) for r in rows
    )


def test_outside_cancel_reentry_full_earlier_global_links():
    e, _, _ = setup()
    source = run(e)
    rows = source.body["periods"]
    assert [x["kind"] for x in rows[0]["input_links"]] == ["ordinary", "ordinary"]
    assert [x["kind"] for x in rows[1]["input_links"]] == ["own", "interline"]
    assert all(not r["input_links"] for r in rows[2:])
    inp = source.body["parent_source"]["periods"][1]["inputs"][1]
    assert (
        inp["receipt"]["source_journal"]["events"][-1]["payload"]["stage"]
        == "interline"
    )
    assert (
        inp["computation_input"]["earlier_within_attempt_results"][0]["outside"] is True
    )


def test_active_static_and_trigger_same_native_call():
    e, c, reports = setup(empty=True, trigger=True)
    source = run(e, reports)
    for t in range(4):
        c.publish_regulator_report_at(c.booking_clock.at(t), reports[t])
        c.step(actions(c, 250 + 5 * t))
    assert m._same(e.trace(), c.trace())
    row = source.body["periods"][0]["postings"][0]["native_cap_receipt"]["products"][0]
    assert row["static_clipped"] and row["trigger_clipped"]
    assert row["executed"]["repr"] == "80.0"


def test_earlier_leader_and_multiple_revisions_retained():
    e, c, _ = setup(empty=True, cap=False)
    for x in (e, c):
        x.begin_period(actions(x, 450))
        assert x.world.posted_leaders
        x.post_actions(actions(x, 60))
        x.post_actions(actions(x, 70))
    with pytest.raises(ValueError, match="pending"):
        e.posting_source()
    e.step(actions(e, 90))
    c.step(actions(c, 90))
    source = e.posting_source()
    posts = source.body["periods"][0]["postings"]
    assert len(posts) == 3 and [p["native_cap_receipt"]["revision"] for p in posts] == [
        1,
        2,
        3,
    ]
    assert m._same(e.trace(), c.trace())
    leader = e.compiled.carrier_index[e.compiled.spec.timing.leaders[0]]
    j = e.compiled.products_of[leader][0]
    assert posts[-1]["native_cap_receipt"]["products"][j]["native_proposal"][
        "repr"
    ] == str(float(450 + leader * 50))


def test_snapshot_branch_reset_owned_and_contract_binding():
    e, _, _ = setup()
    e.step(actions(e))
    s = e.snapshot()
    branch = e.branch()
    assert m._same(e.posting_source().export(), branch.posting_source().export())
    branch.step(actions(branch))
    assert len(e.posting_source().body["periods"]) == 1
    e.restore(s)
    e.step(actions(e))
    assert m._same(e.trace()[-1:], branch.trace())
    assert m._same(e.posting_source().export(), branch.posting_source().export())
    e.reset(tape=e.tape)
    assert e.posting_source().body["periods"] == ()
    other, _, _ = setup(limit=17)
    with pytest.raises(ValueError, match="contract|differ"):
        other.restore(s)
    with pytest.raises(TypeError):
        e.restore(s.native)


def test_detached_export_and_pure_reader_no_native_forward(monkeypatch):
    e, _, _ = setup(empty=True)
    s = run(e)

    def refuse(*a, **k):
        raise AssertionError("no native forward from reader")

    monkeypatch.setattr(NativeCausalInformationMarketEngine, "step", refuse)
    monkeypatch.setattr(NativeCausalInformationMarketEngine, "_input", refuse)
    value = s.export()
    value["periods"] = ()
    assert len(s.export()["periods"]) == 4
    with pytest.raises(TypeError):
        s.body["rule"] = "changed"
    with pytest.raises(FrozenInstanceError):
        s.rule = "changed"


@pytest.mark.parametrize(
    "change",
    ["price", "link", "period", "prefix", "flag", "revision", "programme", "bound"],
)
def test_reader_rejects_full_stored_inconsistencies(change):
    e, _, _ = setup()
    s = run(e)
    b = s.export()
    if change == "price":
        b["periods"][0]["postings"][0]["prices"] = ()
    if change == "link":
        b["periods"][0]["input_links"] = ()
    if change == "period":
        b["periods"] = b["periods"][:-1]
    if change == "prefix":
        b["periods"][1]["postings"][0]["earlier_global_event_count"] += 1
    if change == "flag":
        b["native_probability_received_knowledge_preferences_or_CS_authenticated"] = (
            True
        )
    if change == "revision":
        b["periods"][0]["postings"][0]["native_cap_receipt"]["revision"] = 4
    if change == "programme":
        b["parent_source"]["programme"]["definition"] = "other"
    if change == "bound":
        b["required_support"] -= 1
    altered = m.PeriodPostingSource(b, m._digest(b))
    with pytest.raises((ValueError, m.PeriodPostingLimit)):
        altered.export()
    with pytest.raises(ValueError):
        replace(s, digest="0" * 64).export()


@pytest.mark.parametrize(
    "stage", ["prepost", "post", "preinput", "postinput", "postclose"]
)
def test_failure_preserves_native_pending_and_blocks_queries(stage):
    class Fail(m.PeriodPostingMarketEngine):
        def post_actions(self, actions=None):
            if stage == "prepost":
                raise RuntimeError("injected prepost")
            return super().post_actions(actions)

        def _capture_posting(self, prices, cap):
            super()._capture_posting(prices, cap)
            if stage == "post":
                raise RuntimeError("injected post")

        def _input(self, *args):
            if stage == "preinput":
                raise RuntimeError("injected preinput")
            result = super()._input(*args)
            if stage == "postinput":
                raise RuntimeError("injected postinput")
            return result

        def _finish_posting_period(self, period):
            if stage == "postclose":
                raise RuntimeError("injected postclose")
            return super()._finish_posting_period(period)

    e, _, _ = setup(engine_type=Fail)
    with pytest.raises(RuntimeError, match="injected"):
        e.step(actions(e))
    snap = e.snapshot()
    assert snap.failed and not snap.periods
    if stage != "prepost":
        assert e.cap_postings()
    if stage == "postclose":
        assert e.world.t == 1 and snap.pending
    for f in (e.posting_source, lambda: e.step(actions(e))):
        with pytest.raises(ValueError):
            f()
    other = e.branch()
    assert other.snapshot().failed
    with pytest.raises(ValueError):
        other.posting_source()


def test_native_failure_retains_prior_complete_receipts():
    e, _, _ = setup()
    e.step(actions(e))
    prior = e.posting_source()
    with pytest.raises(KeyError):
        e.step({})
    assert e.snapshot().failed and len(e.snapshot().periods) == 1
    assert m._same(prior.body["periods"], e.snapshot().periods)
    with pytest.raises(ValueError):
        e.posting_source()


def test_limits_before_native_forwards(monkeypatch):
    e, _, _ = setup(empty=True, limit=1)
    e.step(actions(e))
    prior = e.snapshot()
    with pytest.raises(m.PeriodPostingLimit):
        e.step(actions(e))
    assert e.world.t == 1 and e.snapshot().calls == prior.calls

    def refuse(*a, **k):
        raise AssertionError("native reset must not run")

    monkeypatch.setattr(NativeCausalInformationMarketEngine, "reset", refuse)
    e.posting_support_limit = 1
    with pytest.raises(m.PeriodPostingLimit):
        e.reset(tape=e.tape)


def test_forced_hooks_terminal_and_unknown_types_refused():
    e, _, _ = setup(empty=True)
    with pytest.raises(ValueError):
        e.step(actions(e), _requests=[])
    with pytest.raises(ValueError):
        e.step(actions(e), _choices=[])
    run(e)
    with pytest.raises(ValueError):
        e.step(actions(e))
    with pytest.raises(TypeError):
        m.period_posting_record({})


def test_zero_closed_prefix_and_pending_snapshot_are_separate():
    e, _, _ = setup(empty=True)
    zero = e.posting_source()
    assert zero.export()["periods"] == ()
    e.post_actions(actions(e))
    snap = e.snapshot()
    assert snap.pending and not snap.periods and snap.calls == 1
    other = e.branch()
    with pytest.raises(ValueError, match="pending"):
        other.posting_source()
    other.step(actions(other))
    assert len(other.posting_source().body["periods"][0]["postings"]) == 2
    assert e.world.t == 0


def test_lossless_fare_envelope_refusal_even_with_updated_digest():
    e, _, _ = setup(empty=True)
    s = run(e, n=1)
    body = s.export()
    body["periods"][0]["postings"][0]["prices"][0]["repr"] = "invented"
    altered = m.PeriodPostingSource(body, m._digest(body))
    with pytest.raises(ValueError):
        altered.export()


def test_standalone_failed_leader_opening_has_no_continuation():
    class Broken(StatelessController):
        def act(self, obs):
            raise KeyError("NEW138974 declared failing leader actor")

    e, _, _ = setup(empty=True, controller_type=Broken)
    with pytest.raises(KeyError):
        e.begin_period()
    assert e.snapshot().failed
    for call in (
        e.posting_source,
        lambda: e.step(actions(e)),
        lambda: e.begin_period(actions(e)),
    ):
        with pytest.raises(ValueError):
            call()


def test_combined_inflight_snapshot_is_explicitly_unsupported():
    class Check(m.PeriodPostingMarketEngine):
        def _capture_posting(self, prices, cap):
            with pytest.raises(ValueError, match="boundary"):
                self.snapshot()
            return super()._capture_posting(prices, cap)

    e, _, _ = setup(empty=True, engine_type=Check)
    e.step(actions(e))
    assert len(e.posting_source().body["periods"]) == 1


def test_successful_leader_opening_is_pending_even_without_complete_post():
    e, _, _ = setup(empty=True)
    assert e.posting_source().body["periods"] == ()
    e.begin_period(actions(e))
    with pytest.raises(ValueError, match="pending"):
        e.posting_source()
    snap = e.snapshot()
    assert snap.pending and not snap.pending["postings"] and not snap.periods
    other = e.branch()
    with pytest.raises(ValueError, match="pending"):
        other.posting_source()
    other.step(actions(other))
    assert len(other.posting_source().body["periods"]) == 1
    assert e.world.t == 0
