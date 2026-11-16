"""A.10/A.5/B0/V9/B4: NEW128974 typed branch utility units, no old helpers."""

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
from airline_market.sim import registered_branch_utility as ru
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

CLOSE = datetime(2027, 11, 9, 0, 2, tzinfo=timezone.utc)
ASOF = CLOSE + timedelta(days=2)
PNAMES = ("board128974", "cancel128974", "zero128974")
FNAMES = ("standard128974", "credit128974", "open128974", "unknown128974")
MNAMES = ("base128974", "open_material128974", "unknown_E128974", "unknown_B128974")
EPS = F(1, 10**1800)


def q(x):
    return None if x is None else F(int(x["numerator"]), int(x["denominator"]))


def fixture(
    *,
    T=1,
    arrival=0.5,
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
        alpha=1 / 56,
        carrier_pref={"A": 1.0, "B": 1.0},
        market_value={"*": 0.0},
        connection_penalty=0.0,
        product_shift={},
        time_penalty_per_hour=0.0,
        schedule_penalty_per_hour=0.0,
        ideal_dep_hour=None,
    )
    products = (replace(p, fare_floor=1.0, reference_fare=56.0, fare_cap=500.0),)
    if alternatives:
        products += (replace(products[0], id="A_OH0_new128974"),)
    markets = (market,)
    classes = (taste,)
    if multi:
        markets += (replace(market, id=market.id + "_new128974"),)
        products += (
            replace(products[0], id="A_OH0_multi128974", market=markets[1].id),
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
        0, tuple(1e6 if underflow else 56.0 for _ in products), full
    )
    paths = (BernoulliFuturePath("keep128974", F(1), tuple(op for _ in range(T))),)
    if T == 2:
        low = list(full)
        low[c.leg_index[p.legs[0]]] = 1
        paths = (
            BernoulliFuturePath("keep128974", F(2, 5), (op, op)),
            BernoulliFuturePath(
                "loss128974", F(3, 5), (op, replace(op, installed_after=tuple(low)))
            ),
        )
    recipe = BernoulliSourceRecipe(
        dump_spec(c.spec),
        c.spec_hash,
        ledger,
        paths,
        0,
        "NEW128974 explicit finite BASE uniform53/ideal MNL declaration, not native seed/physical probability or received knowledge",
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
        "NEW128974 explicit UTC projection",
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
                            "NEW128974 full service support",
                            "NEW128974 declared before source",
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
                                                -8.0,
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
                                            "NEW128974 registered settlement",
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
                                            "NEW128974 explicit financial tail",
                                        )
                                        for x in people
                                    ),
                                    "NEW128974 full financial programme",
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
            (name, "NEW128974 explicit same full contingent law")
            for name in ("standard", "credit", "tiny_open", "tiny_unknown")
        ),
        "event_definitions": (
            ("all", "NEW128974 all histories"),
            ("all_arrivals", "NEW128974 each declared opportunity arrived"),
            ("zero", "NEW128974 empty analyst event"),
        ),
        "contingencies": contingencies,
        "financial_asof": ASOF,
        "unit": "declared-native-ticket-unit-128974",
        "definition": "NEW128974 full registered finance",
        "service_definition": "NEW128974 full registered service",
        "support_limit": 10**8,
    }


def arguments(**kwargs):
    parent = rf.register_branch_financial(**fixture(**kwargs))
    _, prepared, _ = rf._validate(parent, verify=True)
    contexts = []
    laws = (
        "standard",
        "credit",
        "financial_open",
        "material_open",
        "private_unknown",
        "benefit_unknown",
    )
    for row, people, index in sorted(prepared.values(), key=lambda x: x[2]):
        delivered = dict(row.current_service)
        matrix = []
        for settlement in row.settlements:
            payments = bf._payments(settlement, people)["all_original_people"]
            programmes = []
            for mi, name in enumerate(MNAMES):
                persons = []
                for person, payment in zip(people, payments):
                    bought = person["original_ticket"] is not None
                    served = delivered[person["person_key"]] is True
                    account = ru.BranchUtilityAccount(
                        (
                            None
                            if mi == 3
                            else F(126 if served else 0 if bought else 10),
                            payment["final_net"],
                            F(3 if bought else 2),
                            F(5 if bought and not served else 0 if bought else 1),
                        ),
                        (
                            "explicit B excluding E",
                            "registered NET payment once",
                            "explicit waiting",
                            "explicit failure cost",
                        ),
                        None if mi == 2 else F(8 if served else -2 if bought else 4),
                        "explicit terminal utility, no native private mark decoding",
                        payment["financial_closed"],
                        mi != 1,
                        True,
                        (
                            "actual declared financial follow-up",
                            "independent material follow-up",
                            "independent utility follow-up",
                        ),
                        ASOF,
                        "NEW128974 complete actual candidate account",
                        "declared terminal incidence; not native preferences",
                    )
                    outside = ru.BranchUtilityAccount(
                        (F(12), F(2), F(1), F(2)),
                        ("explicit B0", "explicit P0", "explicit W0", "explicit C0"),
                        F(1),
                        "explicit E0",
                        True,
                        True,
                        True,
                        (
                            "financial reference tail",
                            "material reference tail",
                            "utility reference tail",
                        ),
                        ASOF,
                        "full same-path reference",
                        "terminal reference, not outside intervention",
                    )
                    persons.append(
                        ru.BranchPersonUtility(
                            person["person_key"],
                            account,
                            outside,
                            "reference128974",
                            "B excludes ALL E; explicit non-overlap declaration",
                        )
                    )
                programmes.append(ru.BranchUtilityProgramme(name, persons))
            matrix.append(programmes)
        joint = []
        for name in laws:
            weights = [[F() for _ in MNAMES] for _ in FNAMES]
            weights[0][0] = F(1)
            if name == "credit":
                weights[0][0] = F()
                weights[1][0] = F(1)
            elif name != "standard":
                weights[0][0] -= EPS
                fi, mi = {
                    "financial_open": (2, 0),
                    "material_open": (0, 1),
                    "private_unknown": (0, 2),
                    "benefit_unknown": (0, 3),
                }[name]
                weights[fi][mi] = EPS
            joint.append(weights)
        contexts.append(ru.BranchUtilityContingency(index, matrix, joint))
    return {
        "financial_registration": parent,
        "programme_names": MNAMES,
        "law_definitions": tuple(
            (
                name,
                "NEW128974 complete conditional financial-by-material joint declaration; not independent marginals/native preferences",
            )
            for name in laws
        ),
        "contingencies": contexts,
        "alpha": F(1, 7),
        "utility_unit": "candidate-utility128974",
        "definition": "NEW128974 owned explicit accounts; original source/prices/private marks not reinterpreted",
        "reference_scope": ru.REFERENCE_SCOPE,
        "support_limit": 10**8,
    }


def register(**kwargs):
    return ru.register_branch_utility(**arguments(**kwargs))


def run(reg):
    return ru.registered_branch_utility_record(
        ru.execute_registered_branch_utility(reg)
    )


def summary(result, law, event="all"):
    return next(
        r for r in result["summaries"] if r["law"] == law and r["event"] == event
    )


def test_new_one_period_closed_forms_and_independent_tiny_unknowns():
    r = run(register())
    s = summary(r, "standard")["fields"]
    credit = summary(r, "credit")["fields"]
    assert q(s["net_payment"]["integral"]) == F(28, 3)
    assert q(s["material"]["integral"]) == F(49, 4)
    assert q(s["normalized_noise_excluded"]["integral"]) == F(5, 4)
    assert q(s["normalized_experienced"]["integral"]) == F(35, 12)
    assert q(credit["net_payment"]["integral"]) == F(22, 3)
    assert q(credit["material"]["integral"]) == F(57, 4)
    assert q(credit["normalized_experienced"]["integral"]) == F(269, 84)
    for law, field in (
        ("financial_open", "net_payment"),
        ("material_open", "material"),
        ("private_unknown", "normalized_experienced"),
        ("benefit_unknown", "material"),
    ):
        row = summary(r, law)["fields"][field]
        assert (
            q(row["positive_unknown_mass"]) == F(1, 2) * EPS and row["integral"] is None
        )
    assert q(summary(r, "material_open")["fields"]["net_payment"]["integral"]) == F(
        28, 3
    )
    assert q(summary(r, "private_unknown")["fields"]["material"]["integral"]) == F(
        49, 4
    )
    assert q(
        summary(r, "private_unknown")["fields"]["normalized_noise_excluded"]["integral"]
    ) == F(5, 4)
    assert all(
        v["raw_measure_conditional_ratio"] is None
        for v in summary(r, "standard", "zero")["fields"].values()
    )
    assert r["lifecycle_CS_total_denial_misallocation_penalty_incidence"] is None


def test_new_native_later_cancellation_refund_once_and_credit():
    r = run(register(T=2))
    parent = r["complete_financial_result"]
    reports = parent["financial"]["payment_reports"]
    cancelled = [
        p
        for row in reports
        if row["programme"] == FNAMES[1]
        for p in row["all_original_people"]
        if p["original_ticket"] is not None and p["current_ticket"] is None
    ]
    assert cancelled and all(
        q(p["native_refund_once"]) == F(56)
        and q(p["native_net"]) == 0
        and q(p["final_net"]) == -8
        for p in cancelled
    )
    assert len(parent["attachments"]) == 54
    assert r["recomputed_named_support"]["all_report_count_upper"] >= len(r["reports"])
    assert len(r["raw_joint_rows"]) == len(r["reports"]) * 6


def test_new_outside_nonzero_reference_and_empty_population_retained():
    r = run(register(arrival=0.0))
    assert len(r["reports"]) > 0
    assert any(
        p["original_ticket"] is None
        for row in r["complete_financial_result"]["financial"]["payment_reports"]
        for p in row["all_original_people"]
    )
    assert q(summary(r, "standard")["fields"]["material"]["integral"]) == 0
    assert (
        q(summary(r, "private_unknown")["fields"]["normalized_experienced"]["integral"])
        == 0
    )
    assert r["public_joint_rows"] is not None


def test_new_global_upstream_underflow_nulls_all_public_integrals():
    r = run(register(underflow=True))
    assert r["public_joint_rows"] is None and r["raw_joint_rows"]
    assert all(
        v["integral"] is None and v["raw_measure_conditional_ratio"] is None
        for s in r["summaries"]
        for v in s["fields"].values()
    )


def test_new_signed_huge_tiny_and_common_terminal_translation():
    reg = register()
    person = next(
        p
        for c in reg.contingencies
        for row in c.programmes
        for m in row
        for p in m.people
    )
    p = replace(
        person,
        account=replace(
            person.account,
            components=(F(-3), F(-8), F(-2), F(1)),
            additional_utility=F(1, 10**1900),
        ),
        outside=replace(
            person.outside,
            components=(F(2), F(1), F(3), F(-1)),
            additional_utility=F(-4),
        ),
    )
    before = ru._values(p, F(1, 7))
    shift = F(10**2000)
    translated = replace(
        p,
        account=replace(
            p.account, additional_utility=p.account.additional_utility + shift
        ),
        outside=replace(
            p.outside, additional_utility=p.outside.additional_utility + shift
        ),
    )
    after = ru._values(translated, F(1, 7))
    assert before["normalized_experienced"] == after["normalized_experienced"]
    assert (
        after["experienced"] - before["experienced"] == shift
        and before["material"] == 6
    )


def test_new_unknown_minus_unknown_and_independent_reference_closure():
    reg = register()
    p = next(
        p
        for c in reg.contingencies
        for row in c.programmes
        for m in row
        for p in m.people
    )
    unknown = replace(
        p,
        account=replace(p.account, additional_utility=None),
        outside=replace(p.outside, additional_utility=None),
    )
    assert ru._values(unknown, F(1, 7))["normalized_experienced"] is None
    open_ref = replace(p, outside=replace(p.outside, material_closed=False))
    values = ru._values(open_ref, F(1, 7))
    assert (
        values["material"] is not None
        and values["reference_material"] is None
        and values["normalized_noise_excluded"] is None
    )


def test_new_owned_registration_exports_parent_context_and_global_rng():
    args = arguments()
    reg = ru.register_branch_utility(**args)
    before = ru.registered_branch_utility_record(reg)
    rng = np.random.get_state()
    args["contingencies"].clear()
    export = ru.registered_branch_utility_record(reg)
    export["contingencies"][0]["programmes"][0][0]["people"] = ()
    assert bf._same(before, ru.registered_branch_utility_record(reg))
    result = run(reg)
    result["reports"] = ()
    assert bf._same(before, ru.registered_branch_utility_record(reg))
    after = np.random.get_state()
    assert (
        rng[0] == after[0] and np.array_equal(rng[1], after[1]) and rng[2:] == after[2:]
    )


def test_new_complete_output_limit_threshold_and_parent_never_called(monkeypatch):
    args = arguments()
    reg = ru.register_branch_utility(**args)
    calls = []
    monkeypatch.setattr(
        rf, "execute_registered_branch_financial", lambda *_: calls.append(1)
    )
    low = {**args, "support_limit": reg.bound["total"] - 1}
    with pytest.raises(ru.BranchUtilityLimit):
        ru.register_branch_utility(**low)
    exact = ru.register_branch_utility(**{**args, "support_limit": reg.bound["total"]})
    assert exact.bound["total"] == reg.bound["total"] and not calls


def test_new_stored_digest_bound_rule_and_frozen_parent_revalidated(monkeypatch):
    reg = register()
    calls = []
    monkeypatch.setattr(
        rf, "execute_registered_branch_financial", lambda *_: calls.append(1)
    )
    for bad in (
        replace(reg, digest="wrong"),
        replace(reg, bound={}),
        replace(reg, rule="old"),
        replace(reg, financial=replace(reg.financial, digest="wrong")),
    ):
        with pytest.raises(ValueError):
            ru.execute_registered_branch_utility(bad)
    assert not calls


def change_first(args, edit):
    contexts = list(args["contingencies"])
    i = next(i for i, c in enumerate(contexts) if c.programmes[0][0].people)
    matrix = [list(row) for row in contexts[i].programmes]
    programme = matrix[0][0]
    people = list(programme.people)
    people[0] = edit(people[0])
    matrix[0][0] = replace(programme, people=people)
    contexts[i] = replace(contexts[i], programmes=matrix)
    return {**args, "contingencies": contexts}


@pytest.mark.parametrize(
    "field,value",
    [
        ("alpha", F()),
        ("alpha", 1.0),
        ("utility_unit", "declared-native-ticket-unit-128974"),
        ("reference_scope", "outside-intervention"),
        ("contingencies", ()),
        ("law_definitions", ()),
    ],
)
def test_new_global_invalid_contract_before_source(monkeypatch, field, value):
    args = arguments()
    calls = []
    monkeypatch.setattr(
        rf, "execute_registered_branch_financial", lambda *_: calls.append(1)
    )
    with pytest.raises((ValueError, TypeError)):
        ru.register_branch_utility(**{**args, field: value})
    assert not calls


@pytest.mark.parametrize(
    "edit",
    [
        lambda p: replace(
            p, account=replace(p.account, components=(F(1), F(999), F(), F()))
        ),
        lambda p: replace(p, account=replace(p.account, components=(F(1), None, F()))),
        lambda p: replace(
            p, account=replace(p.account, components=(1.0, None, F(), F()))
        ),
        lambda p: replace(p, account=replace(p.account, component_sources=("B",))),
        lambda p: replace(p, account=replace(p.account, material_closed=1)),
        lambda p: replace(
            p, account=replace(p.account, recorded_at=ASOF + timedelta(seconds=1))
        ),
        lambda p: replace(
            p,
            outside=replace(
                p.outside, components=(F(1), F(2), F(), F()), financial_closed=False
            ),
        ),
        lambda p: replace(p, material_utility_nonoverlap=""),
    ],
)
def test_new_person_account_refusals_include_unused_contexts(monkeypatch, edit):
    args = change_first(arguments(), edit)
    calls = []
    monkeypatch.setattr(
        rf, "execute_registered_branch_financial", lambda *_: calls.append(1)
    )
    with pytest.raises((ValueError, TypeError)):
        ru.register_branch_utility(**args)
    assert not calls


def test_new_omitted_people_matrix_law_and_boolean_alias_refused():
    args = arguments()
    contexts = list(args["contingencies"])
    i = next(i for i, c in enumerate(contexts) if c.programmes[0][0].people)
    c = contexts[i]
    missing = [list(row) for row in c.programmes]
    missing[0][0] = replace(missing[0][0], people=())
    variants = (
        replace(c, programmes=missing),
        replace(c, joint_laws=()),
        replace(c, financial_context_index=True),
    )
    for bad in variants:
        rows = contexts.copy()
        rows[i] = bad
        with pytest.raises(ValueError):
            ru.register_branch_utility(**{**args, "contingencies": rows})
    p = c.programmes[0][0].people[0]
    with pytest.raises(ValueError):
        replace(p, person_key=(False, 0, 0))
    with pytest.raises(ValueError):
        replace(c, programmes=iter(c.programmes))


def test_new_nonproduct_joint_law_retained_not_product_marginals():
    args = arguments()
    contexts = []
    for c in args["contingencies"]:
        laws = list(c.joint_laws)
        matrix = [[F() for _ in MNAMES] for _ in FNAMES]
        matrix[0][0] = matrix[1][2] = F(1, 2)
        laws[0] = matrix
        contexts.append(replace(c, joint_laws=laws))
    r = run(ru.register_branch_utility(**{**args, "contingencies": contexts}))
    rows = [x for x in r["raw_joint_rows"] if x["law"] == "standard"]
    assert all(
        q(x["conditional_joint"])
        == (
            F(1, 2)
            if (x["financial_index"], x["utility_index"]) in ((0, 0), (1, 2))
            else 0
        )
        for x in rows
    )
    assert (
        summary(r, "standard")["fields"]["normalized_experienced"]["integral"] is None
    )


def test_new_multimarket_zero_class_full_support_and_nonfinite_context():
    r = run(register(multi=True, nan_cash=True))
    parent = r["complete_financial_result"]
    assert (
        parent["financial"]["complete_native_source"]["details"][
            "initial_declared_ledger"
        ]["accounts"][0][0]["repr"]
        == "nan"
    )
    assert r["recomputed_named_support"]["all_report_count_upper"] >= len(r["reports"])
    assert q(summary(r, "standard")["fields"]["material"]["integral"]) == F(49, 4)


def test_new_parent_failure_once_and_no_input_mutation(monkeypatch):
    reg = register()
    before = ru.registered_branch_utility_record(reg)
    calls = []

    def fail(*_):
        calls.append(1)
        raise RuntimeError("new parent failure128974")

    monkeypatch.setattr(rf, "execute_registered_branch_financial", fail)
    with pytest.raises(RuntimeError, match="parent failure"):
        ru.execute_registered_branch_utility(reg)
    assert calls == [1] and bf._same(before, ru.registered_branch_utility_record(reg))


def test_new_post_parent_projection_failure_no_partial_retry(monkeypatch):
    reg = register()
    before = ru.registered_branch_utility_record(reg)
    original = rf.execute_registered_branch_financial
    calls = []

    def execute(value):
        calls.append(1)
        return original(value)

    def fail(*_):
        raise RuntimeError("new candidate projection failure128974")

    monkeypatch.setattr(rf, "execute_registered_branch_financial", execute)
    monkeypatch.setattr(ru, "_values", fail)
    with pytest.raises(RuntimeError, match="candidate projection"):
        ru.execute_registered_branch_utility(reg)
    assert calls == [1] and bf._same(before, ru.registered_branch_utility_record(reg))


def test_new_pair_schema_refuses_string_and_incomplete_pairs_before_source(monkeypatch):
    args = arguments()
    calls = []
    monkeypatch.setattr(
        rf, "execute_registered_branch_financial", lambda *_: calls.append(1)
    )
    for pairs in (("AB",), ((),), (("name",),), (("name", "definition", "extra"),)):
        with pytest.raises(ValueError, match="pairs"):
            ru.register_branch_utility(**{**args, "law_definitions": pairs})
    assert not calls


@pytest.mark.parametrize("value", [F(-1), F(2), 0.5])
def test_new_joint_matrix_probabilities_refused_in_unused_context(monkeypatch, value):
    args = arguments()
    contexts = list(args["contingencies"])
    c = contexts[-1]
    joint = [[list(row) for row in matrix] for matrix in c.joint_laws]
    joint[0][-1][-1] = value
    contexts[-1] = replace(c, joint_laws=joint)
    calls = []
    monkeypatch.setattr(
        rf, "execute_registered_branch_financial", lambda *_: calls.append(1)
    )
    with pytest.raises(ValueError):
        ru.register_branch_utility(**{**args, "contingencies": contexts})
    assert not calls


def test_new_actual_financial_disagreement_refused_after_single_parent_route(
    monkeypatch,
):
    reg = register()
    before = ru.registered_branch_utility_record(reg)
    original = rf.execute_registered_branch_financial
    calls = []

    def altered(value):
        calls.append(1)
        record = rf.registered_branch_financial_record(original(value))
        p = next(
            p
            for row in record["financial"]["payment_reports"]
            for p in row["all_original_people"]
        )
        p["final_net"] = {"numerator": "999", "denominator": "1"}
        return rf.RegisteredBranchFinancialResult(record)

    monkeypatch.setattr(rf, "execute_registered_branch_financial", altered)
    with pytest.raises(ValueError, match="actual complete financial"):
        ru.execute_registered_branch_utility(reg)
    assert calls == [1] and bf._same(before, ru.registered_branch_utility_record(reg))
