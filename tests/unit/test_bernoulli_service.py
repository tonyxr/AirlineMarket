"""A.10/A.5/B0/V9/B4: NEW122974 branch/service declarations, no old fixture."""

import itertools
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F

import numpy as np
import pytest

from airline_market.pipeline.screen_generators import screen_spec
from airline_market.sim import bernoulli_service as bs
from airline_market.sim.bernoulli_boundary import DeclaredBaseLedger
from airline_market.sim.bernoulli_future import (
    BernoulliFuturePath,
    BernoulliOpportunity,
    bernoulli_future_expectation,
)
from airline_market.sim.bernoulli_person import MARK_LAW
from airline_market.sim.current_itinerary_service import LegOperation
from airline_market.sim.shared_mark_attempt import _encoded
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

CLOSE = datetime(2027, 9, 21, 0, 3, tzinfo=timezone.utc)
NAMES = ("two", "one", "zero", "cancelled", "no_show")


def q(x):
    return None if x is None else F(int(x["numerator"]), int(x["denominator"]))


def model(
    T=2, arrival=0.3125, *, alternatives=False, zero_class=False, connection=False
):
    s = screen_spec(
        {"T": T, "actions": 2, "n_spokes": 1, "capacity": 2, "capacity_support": [2]}
    )
    p = next(p for p in s.products if p.id == ("A_OD0_0" if connection else "A_OH0"))
    m = next(m for m in s.markets if m.id == p.market)
    k = replace(
        s.segments[0].classes[0],
        weight=1.0,
        alpha=1 / 128,
        carrier_pref={"A": 1.0, "B": 1.0},
        market_value={"*": 0.0},
        connection_penalty=0.0,
        product_shift={},
        time_penalty_per_hour=0.0,
        schedule_penalty_per_hour=0.0,
        ideal_dep_hour=None,
    )
    products = (replace(p, fare_floor=1.0, reference_fare=128.0, fare_cap=500.0),)
    if alternatives:
        products = (*products, replace(products[0], id=products[0].id + "_alternative"))
    classes = (k, replace(k, weight=0.0)) if zero_class else (k,)
    return compile_market(
        replace(
            s,
            products=products,
            markets=(m,),
            segments=(replace(s.segments[0], classes=classes, mixed=None),),
            demand=replace(
                s.demand,
                mode="single_request_bernoulli",
                arrival_probability=PeriodSchedule((0,), (arrival,)),
                arrival_probability_cap=1.0,
                market_weights=PeriodSchedule((0,), ((1.0,),)),
                segment_weights=PeriodSchedule((0,), ((1.0,),)),
                holiday_multiplier=1.0,
                exogenous_multipliers={},
                intensity=None,
            ),
            quotes=replace(s.quotes, requests=()),
            information=tuple(replace(a, requests=()) for a in s.information),
        )
    )


def source(c, *, underflow=False, start=0, periods=None):
    length = c.spec.horizon.n_periods - start if periods is None else periods
    full = tuple(2 for _ in c.spec.legs)
    ledger = DeclaredBaseLedger(
        full, (), tuple((0.0, 0.0, 0.0, 0, 0.0) for _ in c.spec.carriers), 0
    )
    op = BernoulliOpportunity(
        0, tuple((1e6 if underflow else 128.0) for _ in c.spec.products), full
    )
    low = list(full)
    for leg in c.spec.products[0].legs:
        low[c.leg_index[leg]] = 1
    lower = replace(op, installed_after=tuple(low))
    if length == 2 and not underflow:
        paths = (
            BernoulliFuturePath("never", F(1, 3), (op, op)),
            BernoulliFuturePath("late", F(1, 3), (op, lower)),
            BernoulliFuturePath("early", F(1, 3), (lower, lower)),
        )
    else:
        paths = (BernoulliFuturePath("fixed", F(1), tuple(op for _ in range(length))),)
    value = bernoulli_future_expectation(
        c,
        ledger,
        paths,
        start_period=start,
        conditioning="NEW122974 fixed conditional source, not seed/physical law authentication",
        mark_law=MARK_LAW,
        independent_future_private_marks=True,
        unreserved_base_ledger=True,
        tree_row_limit=100000,
        conditional_call_limit=10000,
        ticket_limit=length,
    )
    return value, paths


def declarations(c, paths, *, start=0, close=CLOSE):
    clock = bs.BranchServiceClock(
        close,
        tuple((leg.id, close + timedelta(hours=2)) for leg in c.spec.legs),
        "explicit source-close/scheduled UTC declaration122974",
    )
    cat = ((None, None, None, None),) + tuple(
        (m, g, k, int(j))
        for m in range(len(c.spec.markets))
        for g, s in enumerate(c.spec.segments)
        for k in range(len(s.classes))
        for j in (*c.market_products[m], len(c.spec.products))
    )
    cases = []
    for path in paths:
        for branches in itertools.product(cat, repeat=len(path.opportunities)):
            count = sum(
                b[0] is not None and b[3] != len(c.spec.products) for b in branches
            )
            for active in itertools.product((False, True), repeat=count):
                targets = []
                rid = 0
                for b in branches:
                    if b[0] is not None and b[3] != len(c.spec.products):
                        if active[rid]:
                            targets.extend(
                                (rid, leg) for leg in c.spec.products[b[3]].legs
                            )
                        rid += 1
                programmes = []
                for name, prob in zip(NAMES, (F(1, 2), F(1, 4), F(), F(1, 4), F())):
                    ops = []
                    orders = []
                    for leg in c.spec.legs:
                        cancelled = name == "cancelled"
                        seats = {
                            "two": 2,
                            "one": 1,
                            "zero": 0,
                            "cancelled": 2,
                            "no_show": 2,
                        }[name]
                        ops.append(
                            LegOperation(
                                leg.id,
                                "cancelled" if cancelled else "operates",
                                seats,
                                None if cancelled else close + timedelta(hours=2),
                                None if cancelled else close + timedelta(hours=3),
                                close + timedelta(hours=1) if cancelled else None,
                            )
                        )
                        orders.append(
                            (leg.id, tuple(r for r, l in targets if l == leg.id))
                        )
                    programmes.append(
                        bs.BranchServiceProgramme(
                            name,
                            prob,
                            ops,
                            tuple((r, l, name != "no_show") for r, l in targets),
                            orders,
                            "NEW direct service programme122974",
                            "constructed full-leg source122974",
                        )
                    )
                cases.append(
                    bs.BranchServiceCase(path.name, branches, active, programmes)
                )
    return clock, tuple(cases)


def run(c, s, clock, cases, **kwargs):
    return bs.bernoulli_service_record(
        bs.bernoulli_service_programmes(
            c,
            s,
            clock,
            NAMES,
            cases,
            definition="NEW complete branch service122974",
            support_limit=kwargs.pop("support_limit", 10**9),
            **kwargs,
        )
    )


@pytest.fixture(scope="module")
def base():
    c = model()
    s, paths = source(c)
    clock, cases = declarations(c, paths)
    return c, s, clock, cases


def test_new_exact_source_future_mass_and_count_reference(base):
    _c, s, _clock, _cases = base
    r = run(*base)
    assert (
        q(r["raw_joint_measure"]) == q(r["raw_source_measure"]) == 1
        and q(r["raw_source_residual"]) == 0
    )
    terminals = s.details["terminals"]
    expected = sum(
        (
            q(t["raw_mass"])
            * (
                F(len(t["ledger"]["tickets"]), 2)
                + F(int(bool(t["ledger"]["tickets"])), 4)
            )
            for t in terminals
        ),
        F(),
    )
    assert q(r["delivered_current_count_integral"]) == expected
    assert (
        len(r["all_atoms"]) == len(terminals) * 5
        and len(r["unused_case_coordinates"]) == 22
    )
    assert (
        r["financial_payment_material_utility_CS_denial_misallocation"] is None
        and not r["primary_default_empirical_production_planner_adopted"]
    )


def test_new_boarding_is_not_delivery_and_exact_shared_seat_debit(base):
    r = run(*base)
    atom = next(
        x
        for x in r["all_atoms"]
        if x["programme"] == "one"
        and len(
            [
                p
                for p in x["service"]["input_plan"]["people"]
                if p["current_ticket"] is not None
            ]
        )
        == 2
    )
    service = atom["service"]
    dep = next(x for x in service["steps"] if x["boarded_count"] == 1)
    assert (
        dep["unused_departure_seats"] == 0
        and not dep["delivered"]
        and sum(len(ids) for leg, ids in dep["airborne"]) == 1
    )
    assert sorted(v for k, v in service["current_service"]) == [False, True]
    assert (
        service["delivered_current_target_count"] == 1
        and service["steps"][-1]["delivered"]
    )


def test_new_operating_zero_cancel_and_no_show_remain_distinct(base):
    r = run(*base)
    reasons = {}
    for name in ("zero", "cancelled", "no_show"):
        a = next(
            x
            for x in r["all_atoms"]
            if x["programme"] == name
            and any(
                p["current_ticket"] is not None
                for p in x["service"]["input_plan"]["people"]
            )
        )
        reasons[name] = {
            reason for st in a["service"]["steps"] for rid, reason in st["decisions"]
        }
        assert a["service"]["delivered_current_target_count"] == 0
    assert reasons == {
        "zero": {"no_seat"},
        "cancelled": {"operation_cancelled"},
        "no_show": {"no_boarding_intent"},
    }


def test_new_all_original_outside_and_source_cancelled_keep_null(base):
    r = run(*base)
    outside = cancelled = 0
    for a in r["all_atoms"]:
        people = a["service"]["input_plan"]["people"]
        status = {tuple(k): v for k, v in a["service"]["current_service"]}
        for p in people:
            if p["current_ticket"] is None:
                assert status[tuple(p["person_key"])] is None
                outside += p["original_ticket"] is None
                cancelled += p["original_ticket"] is not None
        assert all(
            st["all_person_keys"] == tuple(p["person_key"] for p in people)
            for st in a["service"]["steps"]
        )
    assert outside and cancelled


def test_new_unused_leg_and_unused_cases_validated_before_execution(base, monkeypatch):
    c, s, clock, cases = base
    bad = list(cases)
    case = bad[0]
    p = case.programmes[0]
    bad[0] = replace(
        case,
        programmes=(replace(p, operations=p.operations[:-1]), *case.programmes[1:]),
    )
    monkeypatch.setattr(
        bs, "_execute", lambda *a: pytest.fail("transition before full validation")
    )
    with pytest.raises(ValueError, match="ALL ordered"):
        run(c, s, clock, bad)


@pytest.mark.parametrize(
    "change",
    (
        "missing_case",
        "reorder",
        "wrong_active",
        "missing_programme",
        "nonunit",
        "float_probability",
        "bad_clock",
    ),
)
def test_new_complete_prospective_contract_refusals(base, change, monkeypatch):
    c, s, clock, cases = base
    bad = list(cases)
    if change == "missing_case":
        bad.pop()
    elif change == "reorder":
        bad[0], bad[1] = bad[1], bad[0]
    elif change == "wrong_active":
        bad[0] = replace(bad[0], active=(True,))
    elif change == "missing_programme":
        bad[0] = replace(bad[0], programmes=bad[0].programmes[:-1])
    elif change == "nonunit":
        bad[0] = replace(
            bad[0],
            programmes=(
                replace(bad[0].programmes[0], probability=F(2)),
                *bad[0].programmes[1:],
            ),
        )
    elif change == "float_probability":
        bad[0] = replace(
            bad[0],
            programmes=(
                replace(bad[0].programmes[0], probability=0.5),
                *bad[0].programmes[1:],
            ),
        )
    else:
        clock = replace(clock, leg_departures=clock.leg_departures[:-1])
    monkeypatch.setattr(
        bs, "_execute", lambda *a: pytest.fail("invalid input executed")
    )
    with pytest.raises((ValueError, TypeError)):
        run(c, s, clock, bad)


@pytest.mark.parametrize(
    "change",
    (
        "intent",
        "priority",
        "seats",
        "early_departure",
        "nonlater_arrival",
        "early_cancel",
    ),
)
def test_new_actual_and_zero_programme_support_refusals(base, change, monkeypatch):
    c, s, clock, cases = base
    bad = list(cases)
    i = next(i for i, x in enumerate(cases) if x.active == (True, True))
    case = bad[i]
    p = case.programmes[2]
    if change == "intent":
        p = replace(p, boarding_intents=p.boarding_intents[:-1])
    elif change == "priority":
        p = replace(
            p,
            boarding_orders=((p.boarding_orders[0][0], (999,)), *p.boarding_orders[1:]),
        )
    elif change == "seats":
        p = replace(
            p, operations=(replace(p.operations[0], seats=-1), *p.operations[1:])
        )
    elif change == "early_departure":
        p = replace(
            p, operations=(replace(p.operations[0], departure=CLOSE), *p.operations[1:])
        )
    elif change == "nonlater_arrival":
        p = replace(
            p,
            operations=(
                replace(p.operations[0], arrival=p.operations[0].departure),
                *p.operations[1:],
            ),
        )
    else:
        p = replace(
            p,
            operations=(
                LegOperation(p.operations[0].leg, "cancelled", 0, None, None, CLOSE),
                *p.operations[1:],
            ),
        )
    progs = list(case.programmes)
    progs[2] = p
    bad[i] = replace(case, programmes=progs)
    monkeypatch.setattr(
        bs, "_execute", lambda *a: pytest.fail("invalid input executed")
    )
    with pytest.raises(ValueError):
        run(c, s, clock, bad)


def test_new_full_output_bound_before_any_transition(base, monkeypatch):
    r = run(*base)
    bound = r["all_support_bound"]
    assert bs._units(r) <= bound
    monkeypatch.setattr(
        bs, "_execute", lambda *a: pytest.fail("limit executed transition")
    )
    with pytest.raises(bs.BernoulliServiceLimit):
        run(*base, support_limit=bound - 1)
    with pytest.raises(bs.BernoulliServiceLimit):
        run(*base, support_limit=1)


def test_new_tiny_positive_future_weight_survives(base):
    c, s, clock, cases = base
    tiny = F(1, 10**900)
    changed = []
    for case in cases:
        p = list(case.programmes)
        p[0] = replace(p[0], probability=tiny)
        p[1] = replace(p[1], probability=F())
        p[3] = replace(p[3], probability=1 - tiny)
        changed.append(replace(case, programmes=p))
    r = run(c, s, clock, changed)
    assert q(r["delivered_current_count_integral"]) > 0
    assert any(
        q(x["raw_joint_mass"]) > 0 and float(q(x["raw_joint_mass"])) == 0
        for x in r["all_atoms"]
        if x["programme"] == "two"
    )


def test_new_needed_source_underflow_keeps_all_public_results_null():
    c = model()
    s, paths = source(c, underflow=True)
    clock, cases = declarations(c, paths)
    r = run(c, s, clock, cases)
    assert (
        r["public_joint"] is r["delivered_current_count_integral"] is None
        and not r["source_public_joint_known"]
        and r["all_atoms"]
    )


def test_new_multiple_direct_products_share_physical_leg_and_zero_classes():
    c = model(alternatives=True, zero_class=True)
    s, paths = source(c, periods=1)
    clock, cases = declarations(c, paths)
    r = run(c, s, clock, cases)
    assert len(r["all_original_person_profiles"]) > 0 and any(
        x["taste_class"] == 1 for row in r["all_original_person_profiles"] for x in row
    )
    assert q(r["raw_joint_measure"]) == q(r["raw_source_measure"])
    assert (
        all(len(p.legs) == 1 for p in c.spec.products)
        and c.spec.products[0].legs == c.spec.products[1].legs
    )


def test_new_three_period_empty_support_and_shifted_start():
    c = model(3, arrival=0.0)
    s, paths = source(c)
    clock, cases = declarations(c, paths)
    r = run(c, s, clock, cases)
    assert q(r["delivered_current_count_integral"]) == 0 and any(
        not a["service"]["input_plan"]["people"] and q(a["raw_joint_mass"]) > 0
        for a in r["all_atoms"]
    )
    c = model(3)
    s, paths = source(c, start=1, periods=2)
    clock, cases = declarations(c, paths, start=1)
    r = run(c, s, clock, cases)
    assert all(
        k[0] >= 1 for a in r["all_atoms"] for k, v in a["service"]["current_service"]
    )


def test_new_connection_product_refused_before_source_traversal(monkeypatch):
    c = model(connection=True)
    s, paths = source(c, periods=1)
    clock, cases = declarations(c, paths)
    monkeypatch.setattr(
        bs.source_reader,
        "_source",
        lambda *a: pytest.fail("unsupported source traversed"),
    )
    with pytest.raises(ValueError, match="single-required-leg"):
        run(c, s, clock, cases)


def test_new_full_source_mismatch_refused(base, monkeypatch):
    c, s, clock, cases = base
    d = dict(s.details)
    d["start_period"] = 1
    bad = replace(s, details=d)
    monkeypatch.setattr(bs, "_execute", lambda *a: pytest.fail("bad source executed"))
    with pytest.raises((ValueError, IndexError)):
        run(c, bad, clock, cases)


def test_new_owned_inputs_exports_and_global_rng_unchanged(base):
    before = _encoded(base[1])
    state = np.random.get_state()
    r = run(*base)
    assert _encoded(base[1]) == before and all(
        np.array_equal(x, y) for x, y in zip(state, np.random.get_state())
    )
    typed = bs.bernoulli_service_programmes(
        *base[:3],
        NAMES,
        list(base[3]),
        definition="NEW owned122974",
        support_limit=10**9,
    )
    exported = bs.bernoulli_service_record(typed)
    exported["all_atoms"] = ()
    assert bs.bernoulli_service_record(typed)["all_atoms"]
    assert r["all_atoms"]


def test_new_runtime_uses_no_native_probability_or_service_producer(base, monkeypatch):
    import airline_market.sim.bernoulli_future as future
    import airline_market.sim.current_itinerary_service as service

    def fail(*a, **k):
        pytest.fail("native/old producer called")

    monkeypatch.setattr(future, "bernoulli_future_expectation", fail)
    monkeypatch.setattr(future, "native_bernoulli_person_law", fail)
    monkeypatch.setattr(future, "resolve_bernoulli_boundary", fail)
    monkeypatch.setattr(service, "execute_current_itinerary_service", fail)
    assert run(*base)["all_atoms"]


def test_new_mid_execution_failure_no_retry_or_partial_result(base, monkeypatch):
    before = _encoded(base[1])
    calls = []
    original = bs._execute

    def fail(plan):
        calls.append(plan)
        if len(calls) == 2:
            raise RuntimeError("injected NEW transition failure122974")
        return original(plan)

    monkeypatch.setattr(bs, "_execute", fail)
    with pytest.raises(RuntimeError, match="injected"):
        run(*base)
    assert len(calls) == 2 and _encoded(base[1]) == before


def test_new_explicit_priority_reversal_changes_person_not_count(base):
    c, s, clock, cases = base
    changed = tuple(
        replace(
            case,
            programmes=tuple(
                replace(
                    p,
                    boarding_orders=tuple(
                        (leg, tuple(reversed(order)))
                        for leg, order in p.boarding_orders
                    ),
                )
                for p in case.programmes
            ),
        )
        for case in cases
    )
    r = run(c, s, clock, changed)
    rows = [
        a
        for a in r["all_atoms"]
        if a["programme"] == "one"
        and len(
            [
                p
                for p in a["service"]["input_plan"]["people"]
                if p["current_ticket"] is not None
            ]
        )
        == 2
    ]
    assert rows and all(
        tuple(v for k, v in a["service"]["current_service"]) == (False, True)
        for a in rows
    )
    assert all(a["service"]["delivered_current_target_count"] == 1 for a in rows)
    assert _encoded(s) == _encoded(base[1])


def test_new_naive_clock_refused_without_source_or_transition(base, monkeypatch):
    monkeypatch.setattr(bs, "_execute", lambda *a: pytest.fail("naive clock executed"))
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(base[2], source_close=CLOSE.replace(tzinfo=None))
    other_zone = timezone(timedelta(hours=7))
    equivalent = replace(
        base[2],
        source_close=CLOSE.astimezone(other_zone),
        leg_departures=tuple(
            (leg, at.astimezone(other_zone)) for leg, at in base[2].leg_departures
        ),
    )
    assert equivalent == base[2]


def test_new_mixed_declaration_refused_before_pure_source_reader(base, monkeypatch):
    c, s, clock, cases = base
    declared = replace(
        c,
        spec=replace(
            c.spec,
            segments=(
                replace(
                    c.spec.segments[0], mixed={"unsupported-declaration122974": True}
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        bs.source_reader,
        "_source",
        lambda *a: pytest.fail("mixed declaration reached source reader"),
    )
    with pytest.raises(ValueError, match="finite native-class"):
        run(declared, s, clock, cases)


def test_new_finite_nonfinite_cash_source_retained_without_financial_fill():
    c = model(arrival=0.0625, alternatives=True)
    full = tuple(2 for _ in c.spec.legs)
    accounts = tuple(
        (float("nan") if i == 0 else 0.0, 0.0, 0.0, 0, 0.0)
        for i in range(len(c.spec.carriers))
    )
    ledger = DeclaredBaseLedger(full, (), accounts, 0)
    paths = (
        BernoulliFuturePath(
            "new-finite-cash122974",
            F(1),
            (BernoulliOpportunity(0, tuple(128.0 for _ in c.spec.products), full),),
        ),
    )
    s = bernoulli_future_expectation(
        c,
        ledger,
        paths,
        start_period=0,
        conditioning="NEW finite cash declaration122974",
        mark_law=MARK_LAW,
        independent_future_private_marks=True,
        unreserved_base_ledger=True,
        tree_row_limit=1000,
        conditional_call_limit=1000,
        ticket_limit=1,
    )
    clock, cases = declarations(c, paths)
    r = run(c, s, clock, cases)
    cash = r["complete_source"]["details"]["initial_declared_ledger"]["accounts"][0][0]
    assert cash["repr"] == "nan" and cash["exact"] is None
    assert (
        r["source_public_joint_known"]
        and r["delivered_current_count_integral"] is not None
    )
    assert r["financial_payment_material_utility_CS_denial_misallocation"] is None
