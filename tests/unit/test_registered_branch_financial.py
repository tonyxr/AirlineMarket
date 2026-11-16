"""A.10/A.5/B0/V9/B4: NEW126974 prospective factory units, no old helpers."""

import copy
import itertools
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F

import numpy as np
import pytest

from airline_market.pipeline.screen_generators import screen_spec
from airline_market.sim import bernoulli_service as bs
from airline_market.sim import branch_financial as bf
from airline_market.sim import registered_branch_financial as rf
from airline_market.sim.bernoulli_boundary import DeclaredBaseLedger
from airline_market.sim.bernoulli_future import (
    BernoulliFuturePath,
    BernoulliOpportunity,
)
from airline_market.sim.bernoulli_person import MARK_LAW
from airline_market.sim.current_itinerary_service import LegOperation
from airline_market.sim.registered_bernoulli import BernoulliSourceRecipe
from airline_market.spec import compile_market
from airline_market.spec.io import dump_spec
from airline_market.spec.schedule import PeriodSchedule

CLOSE = datetime(2027, 10, 26, 0, 2, tzinfo=timezone.utc)
ASOF = CLOSE + timedelta(days=2)
PNAMES = ("board126974", "cancel126974", "zero126974")
FNAMES = ("standard126974", "credit126974", "open126974", "unknown126974")
EPS = F(1, 10**1100)


def q(value):
    return (
        None if value is None else F(int(value["numerator"]), int(value["denominator"]))
    )


def fixture(
    *,
    T=1,
    arrival=0.125,
    underflow=False,
    nan_cash=False,
    alternatives=False,
    multi=False,
):
    s = screen_spec(
        {"T": 2, "actions": 2, "n_spokes": 1, "capacity": 2, "capacity_support": [2]}
    )
    p = next(x for x in s.products if x.id == "A_OH0")
    market = next(x for x in s.markets if x.id == p.market)
    taste = replace(
        s.segments[0].classes[0],
        weight=1.0,
        alpha=1 / 64,
        carrier_pref={"A": 1.0, "B": 1.0},
        market_value={"*": 0.0},
        connection_penalty=0.0,
        product_shift={},
        time_penalty_per_hour=0.0,
        schedule_penalty_per_hour=0.0,
        ideal_dep_hour=None,
    )
    products = (replace(p, fare_floor=1.0, reference_fare=64.0, fare_cap=500.0),)
    if alternatives:
        products += (replace(products[0], id="A_OH0_new126974"),)
    markets = (market,)
    classes = (taste,)
    if multi:
        markets += (replace(market, id=market.id + "_new126974"),)
        products += (
            replace(products[0], id="A_OH0_multi126974", market=markets[1].id),
        )
        classes += (replace(taste, weight=0.0),)
    c = compile_market(
        replace(
            s,
            products=products,
            markets=markets,
            segments=(replace(s.segments[0], classes=classes, mixed=None),),
            demand=replace(
                s.demand,
                mode="single_request_bernoulli",
                arrival_probability=PeriodSchedule((0,), (arrival,)),
                arrival_probability_cap=1.0,
                market_weights=PeriodSchedule(
                    (0,), (tuple(1.0 if i == 0 else 0.0 for i in range(len(markets))),)
                ),
                segment_weights=PeriodSchedule((0,), ((1.0,),)),
                holiday_multiplier=1.0,
                exogenous_multipliers={},
                intensity=None,
            ),
            quotes=replace(s.quotes, requests=()),
            information=tuple(replace(x, requests=()) for x in s.information),
        )
    )
    full = tuple(2 for _ in c.spec.legs)
    ledger = DeclaredBaseLedger(
        full,
        (),
        tuple(
            (float("nan") if nan_cash and i == 0 else 0.0, 0.0, 0.0, 0, 0.0)
            for i, _ in enumerate(c.spec.carriers)
        ),
        0,
    )
    op = BernoulliOpportunity(
        0, tuple(1e6 if underflow else 64.0 for _ in products), full
    )
    paths = (BernoulliFuturePath("keep126974", F(1), tuple(op for _ in range(T))),)
    if T == 2:
        low = list(full)
        low[c.leg_index[p.legs[0]]] = 1
        paths = (
            BernoulliFuturePath("keep126974", F(3, 4), (op, op)),
            BernoulliFuturePath(
                "loss126974", F(1, 4), (op, replace(op, installed_after=tuple(low)))
            ),
        )
    recipe = BernoulliSourceRecipe(
        dump_spec(c.spec),
        c.spec_hash,
        ledger,
        paths,
        0,
        "NEW126974 explicit finite BASE uniform53/ideal MNL declaration, not native seed/physical probability or received knowledge",
        MARK_LAW,
        True,
        True,
        100000,
        10000,
        T,
        10**8,
    )
    clock = bs.BranchServiceClock(
        CLOSE,
        tuple((x.id, CLOSE + timedelta(hours=2)) for x in c.spec.legs),
        "NEW126974 explicit UTC projection",
    )
    catalogue = ((None, None, None, None),) + tuple(
        (m, 0, k, j)
        for m in range(len(markets))
        for k in range(len(classes))
        for j in (*c.market_products[m], len(products))
    )
    cases, contingencies = [], []
    for path in paths:
        for branches in itertools.product(catalogue, repeat=T):
            purchases = sum(
                b[0] is not None and b[3] != len(products) for b in branches
            )
            for active in itertools.product((False, True), repeat=purchases):
                people = []
                rid = 0
                for t, b in enumerate(branches):
                    if b[0] is None:
                        continue
                    bought = b[3] != len(products)
                    current = bought and active[rid]
                    people.append(
                        {
                            "key": (t, 0, b[0]),
                            "ticket": rid if bought else None,
                            "current": rid if current else None,
                            "fare": path.opportunities[t].prices[b[3]]
                            if bought
                            else None,
                            "carrier": c.spec.products[b[3]].carrier
                            if bought
                            else None,
                        }
                    )
                    rid += int(bought)
                targets = tuple(
                    (x["current"], c.spec.products[branches[x["key"][0]][3]].legs[0])
                    for x in people
                    if x["current"] is not None
                )
                programmes = []
                for j, (name, weight) in enumerate(
                    zip(PNAMES, (F(2, 3), F(1, 3), F()))
                ):
                    cancelled = j == 1
                    programmes.append(
                        bs.BranchServiceProgramme(
                            name,
                            weight,
                            tuple(
                                LegOperation(
                                    l.id,
                                    "cancelled" if cancelled else "operates",
                                    0 if j == 2 else 2,
                                    None if cancelled else CLOSE + timedelta(hours=2),
                                    None if cancelled else CLOSE + timedelta(hours=3),
                                    CLOSE + timedelta(hours=1) if cancelled else None,
                                )
                                for l in c.spec.legs
                            ),
                            tuple((rid, leg, True) for rid, leg in targets),
                            tuple(
                                (
                                    l.id,
                                    tuple(rid for rid, leg in targets if leg == l.id),
                                )
                                for l in c.spec.legs
                            ),
                            "NEW126974 full service support",
                            "NEW126974 declared before source",
                        )
                    )
                    for values in itertools.product((False, True), repeat=len(targets)):
                        status = dict(zip((rid for rid, _ in targets), values))
                        current_service = tuple(
                            (x["key"], status.get(x["current"])) for x in people
                        )
                        settlements = []
                        for name in FNAMES:
                            transfers = []
                            for x in people:
                                if x["ticket"] is None:
                                    continue
                                specifications = []
                                if (
                                    x["current"] is not None
                                    and status[x["current"]] is False
                                ):
                                    specifications.append(
                                        (
                                            "additional_refund",
                                            x["fare"],
                                            6,
                                            "Explicit registered non-delivery refund",
                                        )
                                    )
                                if name == FNAMES[1]:
                                    specifications.extend(
                                        (
                                            (
                                                "additional_payment",
                                                -4.0,
                                                7,
                                                "Separately additional cancelled-or-active-ticket credit",
                                            ),
                                            (
                                                "additional_refund",
                                                -0.0,
                                                8,
                                                "Signed-zero declaration",
                                            ),
                                        )
                                    )
                                if name == FNAMES[3]:
                                    specifications.append(
                                        (
                                            "additional_payment",
                                            None,
                                            9,
                                            "Explicit unknown settlement",
                                        )
                                    )
                                for k, (kind, amount, hour, definition) in enumerate(
                                    specifications
                                ):
                                    transfers.append(
                                        bf.BranchTicketTransfer(
                                            f"{name}-{x['ticket']}-{k}",
                                            x["key"],
                                            x["ticket"],
                                            x["carrier"],
                                            kind,
                                            amount,
                                            CLOSE + timedelta(hours=hour),
                                            CLOSE + timedelta(hours=hour, minutes=1),
                                            "NEW126974 registered settlement",
                                            definition,
                                        )
                                    )
                            transfers.sort(key=lambda x: (x.recorded, x.event_id))
                            settlements.append(
                                bf.BranchSettlement(
                                    name,
                                    transfers,
                                    tuple(
                                        bf.BranchFinancialFollowup(
                                            x["key"],
                                            ASOF,
                                            name != FNAMES[2],
                                            "NEW126974 explicit financial tail",
                                        )
                                        for x in people
                                    ),
                                    "NEW126974 full financial programme",
                                )
                            )
                        contingencies.append(
                            rf.BranchFinancialContingency(
                                path.name,
                                branches,
                                active,
                                service_programme=PNAMES[j],
                                current_service=current_service,
                                settlements=settlements,
                                law_weights=(
                                    (F(1), F(), F(), F()),
                                    (F(), F(1), F(), F()),
                                    (1 - EPS, F(), EPS, F()),
                                    (1 - EPS, F(), F(), EPS),
                                ),
                                event_flags=(
                                    True,
                                    all(b[0] is not None for b in branches),
                                    False,
                                ),
                            )
                        )
                cases.append(
                    bs.BranchServiceCase(path.name, branches, active, programmes)
                )
    return {
        "recipe": recipe,
        "clock": clock,
        "service_names": PNAMES,
        "service_cases": cases,
        "financial_names": FNAMES,
        "law_definitions": tuple(
            (name, "NEW126974 explicit same full contingent law")
            for name in ("standard", "credit", "tiny_open", "tiny_unknown")
        ),
        "event_definitions": (
            ("all", "NEW126974 all histories"),
            ("all_arrivals", "NEW126974 each declared opportunity arrived"),
            ("zero", "NEW126974 empty analyst event"),
        ),
        "contingencies": contingencies,
        "financial_asof": ASOF,
        "unit": "declared-native-ticket-unit-126974",
        "definition": "NEW126974 full registered finance",
        "service_definition": "NEW126974 full registered service",
        "support_limit": 10**8,
    }


def register(**kwargs):
    return rf.register_branch_financial(**fixture(**kwargs))


def result(reg):
    return rf.registered_branch_financial_record(
        rf.execute_registered_branch_financial(reg)
    )


def test_new_one_opportunity_closed_forms_zero_event_and_parent_inputs():
    reg = register()
    record = result(reg)
    f = record["financial"]
    summaries = {(x["law"], x["event"]): x for x in f["summaries"]}
    assert q(summaries["standard", "all"]["net_payment_integral"]) == F(8, 3)
    assert q(summaries["credit", "all"]["net_payment_integral"]) == F(29, 12)
    assert q(summaries["standard", "all_arrivals"]["raw_event_evidence"]) == F(1, 8)
    assert q(
        summaries["standard", "all_arrivals"]["raw_measure_conditional_payment_ratio"]
    ) == F(64, 3)
    assert (
        summaries["standard", "zero"]["raw_measure_conditional_payment_ratio"] is None
    )
    assert q(summaries["standard", "zero"]["net_payment_integral"]) == 0
    assert record["registration"] == rf.registered_branch_financial_record(reg)
    assert record["material_utility_CS_denial_misallocation_penalty_incidence"] is None
    assert record["full_primary_default_empirical_production_planner_adopted"] is False


def test_new_only_three_actual_indices_attach_to_saved_numerics():
    reg = register()
    rec = result(reg)
    f = rec["financial"]
    for link, cell in zip(rec["attachments"], f["cells"]):
        assert set(link) == {
            "source_history_index",
            "service_atom_index",
            "registered_context_index",
        }
        row = reg.contingencies[link["registered_context_index"]]
        assert cell["programmes"] == rf._encoded(row.settlements)
        assert cell["service_atom"] == link["service_atom_index"]
    assert len(reg.contingencies) == 15 and reg.bound["full_financial_contexts"] == 15
    assert reg.bound["full_source_status_cases"] == 4


def test_new_two_opportunity_native_refund_once_and_cancelled_ticket_credit():
    rec = result(register(T=2))
    f = rec["financial"]
    summary = {(x["law"], x["event"]): x for x in f["summaries"]}
    assert q(summary["standard", "all"]["net_payment_integral"]) == F(127, 24)
    assert q(summary["credit", "all"]["net_payment_integral"]) == F(115, 24)
    cancelled = [
        p
        for report in f["payment_reports"]
        if report["programme"] == FNAMES[1]
        for p in report["all_original_people"]
        if p["original_ticket"] is not None and p["current_ticket"] is None
    ]
    assert cancelled and all(
        q(p["native_refund_once"]) == 64
        and q(p["native_net"]) == 0
        and q(p["final_net"]) == -4
        for p in cancelled
    )


def test_new_positive_tiny_unknown_and_open_outside_stay_null():
    f = result(register())["financial"]
    for row in f["summaries"]:
        if row["law"].startswith("tiny") and row["event"] != "zero":
            assert (
                row["net_payment_integral"]
                is row["raw_measure_conditional_payment_ratio"]
                is None
            )
            assert q(row["positive_unknown_mass"]) > 0
    outside = [
        p
        for r in f["payment_reports"]
        if r["programme"] == FNAMES[2]
        for p in r["all_original_people"]
        if p["original_ticket"] is None
    ]
    assert outside and all(p["final_net"] is None for p in outside)


def test_new_zero_arrival_keeps_unknown_reports_with_algebraic_zero():
    f = result(register(arrival=0.0))["financial"]
    assert all(q(x["net_payment_integral"]) == 0 for x in f["summaries"])
    assert any(
        p["final_net"] is None
        for r in f["payment_reports"]
        for p in r["all_original_people"]
    )


def test_new_whole_upstream_underflow_keeps_all_public_results_null():
    f = result(register(underflow=True))["financial"]
    assert f["public_law_rows"] is None and f["raw_law_rows"]
    assert all(
        x["net_payment_integral"] is None
        and x["raw_measure_conditional_payment_ratio"] is None
        for x in f["summaries"]
    )


def test_new_shared_leg_alternatives_and_nan_carrier_context_lossless():
    reg = register(alternatives=True, nan_cash=True)
    before = rf.registered_branch_financial_record(reg)
    assert before["recipe"]["ledger"]["accounts"][0][0]["exact"] is None
    rec = result(reg)
    assert rec["financial"]["complete_native_source"]["details"][
        "exact_raw_mass_residual"
    ]
    amounts = [
        e["amount"]
        for row in before["contingencies"]
        for p in row["settlements"]
        for e in p["transfers"]
        if e["kind"] == "additional_refund"
    ]
    assert any(x["binary64_bits"] == "8000000000000000" for x in amounts)


def test_new_registration_is_pure_and_owns_mutable_inputs(monkeypatch):
    inputs = fixture()
    saved = copy.deepcopy(inputs)
    state = np.random.get_state()
    monkeypatch.setattr(
        rf,
        "bernoulli_future_expectation",
        lambda *a, **k: pytest.fail("source before execution"),
    )
    reg = rf.register_branch_financial(**inputs)
    record = rf.registered_branch_financial_record(reg)
    inputs["service_cases"].clear()
    inputs["contingencies"].clear()
    assert rf.registered_branch_financial_record(reg) == record
    assert saved["contingencies"] and np.array_equal(state[1], np.random.get_state()[1])
    with pytest.raises(TypeError):
        reg.bound["total"] = 0
    record["contingencies"] = ()
    assert rf.registered_branch_financial_record(reg)["contingencies"]


@pytest.mark.parametrize(
    "change",
    (
        "omit_case",
        "omit_unused",
        "omit_settlement",
        "omit_outside",
        "wrong_law",
        "wrong_event",
        "bad_outcome",
        "bad_branch",
        "late_asof",
        "bad_source",
    ),
)
def test_new_complete_prospective_refusals_before_source(change, monkeypatch):
    x = fixture()
    rows = x["contingencies"]
    monkeypatch.setattr(
        rf,
        "bernoulli_future_expectation",
        lambda *a, **k: pytest.fail("invalid registration forwarded"),
    )
    if change == "omit_case":
        x["service_cases"].pop()
    elif change == "omit_unused":
        rows.pop()
    elif change == "omit_settlement":
        rows[0] = replace(rows[0], settlements=rows[0].settlements[:-1])
    elif change == "omit_outside":
        i = next(i for i, r in enumerate(rows) if r.settlements[0].followup)
        s = replace(rows[i].settlements[0], followup=())
        rows[i] = replace(rows[i], settlements=(s, *rows[i].settlements[1:]))
    elif change == "wrong_law":
        rows[0] = replace(
            rows[0], law_weights=((F(1, 2), F(), F(), F()), *rows[0].law_weights[1:])
        )
    elif change == "wrong_event":
        rows[0] = replace(rows[0], event_flags=(True,))
    elif change == "bad_outcome":
        i = next(
            i
            for i, r in enumerate(rows)
            if r.current_service and r.current_service[0][1] is not None
        )
        rows[i] = replace(
            rows[i], current_service=((rows[i].current_service[0][0], 1),)
        )
    elif change == "bad_branch":
        case = x["service_cases"][1]
        b = case.branches[0]
        x["service_cases"][1] = replace(case, branches=((0.0, *b[1:]),))
    elif change == "late_asof":
        x["financial_asof"] = CLOSE
    elif change == "bad_source":
        x["recipe"] = replace(x["recipe"], independent_future_private_marks=False)
    with pytest.raises((ValueError, TypeError, rf.BranchRegistrationLimit)):
        rf.register_branch_financial(**x)


@pytest.mark.parametrize("field", ("service_cases", "contingencies", "law_definitions"))
def test_new_opaque_registration_iterators_refused(field):
    x = fixture()
    x[field] = iter(x[field])
    with pytest.raises(ValueError, match="tuple/list"):
        rf.register_branch_financial(**x)


def test_new_bound_recomputed_exact_threshold_before_all_forwards(monkeypatch):
    x = fixture()
    reg = rf.register_branch_financial(**x)
    bound = reg.bound["total"]
    monkeypatch.setattr(
        rf,
        "bernoulli_future_expectation",
        lambda *a, **k: pytest.fail("limit forwarded"),
    )
    with pytest.raises(rf.BranchRegistrationLimit):
        rf.register_branch_financial(**{**x, "support_limit": bound - 1})
    exact = rf.register_branch_financial(**{**x, "support_limit": bound})
    assert exact.bound["total"] == bound
    bad = replace(reg, bound={**reg.bound, "total": bound - 1})
    with pytest.raises(ValueError, match="recomputed"):
        rf.execute_registered_branch_financial(bad)


def test_new_digest_change_and_rule_refused_before_source(monkeypatch):
    reg = register()
    monkeypatch.setattr(
        rf,
        "bernoulli_future_expectation",
        lambda *a, **k: pytest.fail("tampered forwarded"),
    )
    for bad in (replace(reg, unit=reg.unit + "changed"), replace(reg, rule="old")):
        with pytest.raises(ValueError):
            rf.execute_registered_branch_financial(bad)


@pytest.mark.parametrize("phase", ("source", "service", "financial"))
def test_new_each_stage_failure_once_no_retry_partial_or_input_mutation(
    phase, monkeypatch
):
    reg = register()
    before = rf.registered_branch_financial_record(reg)
    calls = []

    def failure(*a, **k):
        calls.append(phase)
        raise RuntimeError("NEW126974 injected")

    if phase == "source":
        monkeypatch.setattr(rf, "bernoulli_future_expectation", failure)
    elif phase == "service":
        monkeypatch.setattr(rf.service, "bernoulli_service_programmes", failure)
    else:
        monkeypatch.setattr(rf.finance, "branch_financial_laws", failure)
    with pytest.raises(RuntimeError, match="NEW126974 injected"):
        rf.execute_registered_branch_financial(reg)
    assert calls == [phase] and rf.registered_branch_financial_record(reg) == before


def test_new_actual_source_mismatch_stops_before_service(monkeypatch):
    reg = register()
    original = rf.bernoulli_future_expectation
    calls = []

    def wrong(*a, **k):
        source = original(*a, **k)
        return replace(
            source,
            details={
                **source.details,
                "conditioning": "different NEW126974 declaration",
            },
        )

    monkeypatch.setattr(rf, "bernoulli_future_expectation", wrong)
    monkeypatch.setattr(
        rf.service,
        "bernoulli_service_programmes",
        lambda *a, **k: calls.append("service"),
    )
    with pytest.raises(ValueError, match="information"):
        rf.execute_registered_branch_financial(reg)
    assert not calls


def test_new_multiple_markets_zero_class_and_bound_kernel_support():
    reg = register(multi=True)
    rec = result(reg)
    source = rec["financial"]["complete_native_source"]
    kernels = source["details"]["nodes"][0]["boundary"]["supplied_new_birth_law"][
        "kernels"
    ]
    assert len(kernels) == 4
    observed_columns = sum(len(x["probabilities"]) for x in kernels)
    assert observed_columns == 8
    assert reg.bound["source_kernel_product_columns_upper"] == 24
    assert reg.bound["source_kernel_product_columns_upper"] >= 2 * observed_columns
    assert reg.bound["full_source_status_cases"] == 13
    assert reg.bound["full_financial_contexts"] == 51
    assert any(row.branches[0][0] == 1 for row in reg.contingencies)
    assert any(row.branches[0][2] == 1 for row in reg.contingencies)
    summaries = {(x["law"], x["event"]): x for x in rec["financial"]["summaries"]}
    assert q(summaries["standard", "all"]["net_payment_integral"]) == F(8, 3)
