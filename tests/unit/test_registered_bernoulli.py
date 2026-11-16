"""A.10/A.5/B0/V9/B4: NEW120974 declarations, no old fixture/seed/tape."""

import itertools
from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest

from airline_market.pipeline.screen_generators import screen_spec
from airline_market.sim import registered_bernoulli as rb
from airline_market.sim import source_future_transport as st
from airline_market.sim.bernoulli_boundary import DeclaredBaseLedger
from airline_market.sim.bernoulli_future import (
    BernoulliFuturePath,
    BernoulliFutureResult,
    BernoulliOpportunity,
)
from airline_market.sim.payoff_logit import ContingentPayoff
from airline_market.sim.shared_mark_attempt import _encoded
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

SOURCES = tuple((k, "explicit declared component120974:" + k) for k in st.COMPONENTS)


def q(v):
    return None if v is None else F(int(v["numerator"]), int(v["denominator"]))


def model(T=2, *, multi=False, arrival=0.5):
    s = screen_spec(
        {"T": T, "actions": 2, "n_spokes": 1, "capacity": 2, "capacity_support": [2]}
    )
    if multi:
        tastes = (
            replace(s.segments[0].classes[0], weight=1.0),
            replace(s.segments[0].classes[0], weight=0.0),
        )
        return compile_market(
            replace(
                s,
                segments=(replace(s.segments[0], classes=tastes, mixed=None),),
                demand=replace(
                    s.demand,
                    mode="single_request_bernoulli",
                    arrival_probability=PeriodSchedule((0,), (arrival,)),
                    segment_weights=PeriodSchedule((0,), ((1.0,),)),
                    intensity=None,
                ),
            )
        )
    p = next(p for p in s.products if p.id == "A_OH0")
    m = next(m for m in s.markets if m.id == p.market)
    taste = replace(
        s.segments[0].classes[0],
        weight=1.0,
        alpha=0.01,
        carrier_pref={"A": 1.0, "B": 0.0},
        market_value={"*": 0.0},
        connection_penalty=0.0,
        product_shift={},
        time_penalty_per_hour=0.0,
        schedule_penalty_per_hour=0.0,
        ideal_dep_hour=None,
    )
    return compile_market(
        replace(
            s,
            products=(
                replace(p, fare_floor=5.0, reference_fare=100.0, fare_cap=500.0),
            ),
            markets=(m,),
            segments=(replace(s.segments[0], classes=(taste,), mixed=None),),
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


def declarations(
    c,
    *,
    rare=F(1, 10**2100),
    open_mass=F(),
    single_path=False,
    underflow_weight=None,
    periods=None,
):
    T = c.spec.horizon.n_periods if periods is None else periods
    initial = tuple(2 for _ in c.spec.legs)
    ledger = DeclaredBaseLedger(
        initial, (), tuple((0.0, 0.0, 0.0, 0, 0.0) for _ in c.spec.carriers), 0
    )
    base = tuple(
        BernoulliOpportunity(0, tuple(100.0 for _ in c.spec.products), initial)
        for _ in range(T)
    )
    if single_path:
        paths = (BernoulliFuturePath("never", F(1), base),)
    else:
        late = list(base)
        early = list(base)
        target = c.leg_index["A_OH0"]
        lost = list(initial)
        lost[target] -= 1
        late[-1] = replace(late[-1], installed_after=tuple(lost))
        early = [replace(o, installed_after=tuple(lost)) for o in early]
        paths = (
            BernoulliFuturePath("never", F(1, 3), base),
            BernoulliFuturePath("late", F(1, 3), late),
            BernoulliFuturePath("early", F(1, 3), early),
        )
    if underflow_weight is not None:
        rare = F()
        paths = (
            BernoulliFuturePath("never", 1 - underflow_weight, base),
            BernoulliFuturePath(
                "underflow",
                underflow_weight,
                tuple(
                    replace(o, prices=tuple(1e6 for _ in c.spec.products)) for o in base
                ),
            ),
        )
    contract = rb.BernoulliProgrammeContract(
        ("rare", "ordinary"),
        ("serve", "fail", "open"),
        "explicit-fare-unit120974",
        "full source before explicit analyst event120974",
        "NEW declared complete joint future120974",
        "constructed declaration120974",
    )
    events = (
        rb.BernoulliDeclaredEvent(
            "choice",
            "both arrivals, first purchase then outside, no source cancellation; no future-cell selector",
            "declared event120974",
        ),
        rb.BernoulliDeclaredEvent(
            "rare", "choice AND explicit rare signal", "declared rare120974"
        ),
        rb.BernoulliDeclaredEvent(
            "all",
            "ALL complete source histories/signals",
            "declared unconditional120974",
        ),
    )
    catalogue = ((None, None, None, None),) + tuple(
        (m, g, k, int(j))
        for m in range(len(c.spec.markets))
        for g, s in enumerate(c.spec.segments)
        for k in range(len(s.classes))
        for j in (*c.market_products[m], len(c.spec.products))
    )
    cases = []
    for path in paths:
        for branches in itertools.product(catalogue, repeat=T):
            purchases = sum(
                b[0] is not None and b[3] != len(c.spec.products) for b in branches
            )
            for active in itertools.product((False, True), repeat=purchases):
                people = []
                rid = 0
                for t, (m, g, k, j) in enumerate(branches):
                    if m is None:
                        continue
                    purchase = j != len(c.spec.products)
                    people.append(
                        (
                            (t, 0, m),
                            rid if purchase else None,
                            active[rid] if purchase else None,
                            j,
                            t,
                        )
                    )
                    rid += int(purchase)
                observations = []
                r = rare if path.name == "never" else F()
                service = {
                    "early": F(1, 4),
                    "late": F(1, 2),
                    "never": F(3, 4),
                    "underflow": F(1, 2),
                }[path.name]
                for signal, prob in (("rare", r), ("ordinary", 1 - r)):
                    cells = []
                    for ci, (name, weight) in enumerate(
                        zip(
                            contract.cells,
                            (
                                service * (1 - open_mass),
                                (1 - service) * (1 - open_mass),
                                open_mass,
                            ),
                        )
                    ):
                        terms = []
                        for key, ticket, alive, j, t in people:
                            fin = name != "open"
                            current = None
                            adjust = []
                            B = F(20)
                            W = F(6)
                            C = F()
                            net = F()
                            closed = True
                            if ticket is not None:
                                if alive:
                                    current = name == "serve"
                                    fare = F.from_float(path.opportunities[t].prices[j])
                                    extra = F() if current else -fare
                                    adjust = [
                                        st.NetTicketAdjustment(
                                            "extra:" + str(key),
                                            ticket,
                                            c.spec.carriers[
                                                int(c.product_carrier[j])
                                            ].code,
                                            extra,
                                            "declared additional NET transfer120974",
                                        )
                                    ]
                                    net = fare + extra
                                    B = F(400) if current else F()
                                    W = F(9)
                                    C = F() if current else F(27)
                                else:
                                    B = W = C = None
                                    closed = False
                            terms.append(
                                st.PersonFutureMaterial(
                                    key,
                                    current,
                                    "declared service120974",
                                    adjust,
                                    ContingentPayoff(B, net if fin else None, W, C),
                                    SOURCES,
                                    fin,
                                    "declared financial tail120974",
                                    closed,
                                    "declared material tail120974",
                                )
                            )
                        cells.append(st.JointFutureCell(name, weight, terms))
                    observations.append(st.HistoryObservation(signal, prob, cells))
                chosen = (
                    T == 2
                    and branches[0][0] is not None
                    and branches[0][3] != len(c.spec.products)
                    and branches[1][0] is not None
                    and branches[1][3] == len(c.spec.products)
                    and all(active)
                )
                cases.append(
                    rb.BernoulliHistoryCase(
                        path.name,
                        branches,
                        active,
                        observations,
                        (
                            ("choice", (chosen, chosen)),
                            ("rare", (chosen, False)),
                            ("all", (True, True)),
                        ),
                    )
                )
    laws = (rb.BernoulliDeclaredLaw("forecast", cases),)
    evaluations = (
        rb.BernoulliEvaluation("choice", "forecast", "choice"),
        rb.BernoulliEvaluation("rare", "forecast", "rare"),
        rb.BernoulliEvaluation("all", "forecast", "all"),
    )
    return ledger, paths, contract, laws, events, evaluations


def register(c, decl, **changes):
    params = {
        "start_period": 0,
        "conditioning": "NEW120974 fixed conditional programme before private marks; no physical/seed/reception law authentication",
        "mark_law": rb.MARK_LAW,
        "independent_future_private_marks": True,
        "unreserved_base_ledger": True,
        "tree_row_limit": 1000000,
        "conditional_call_limit": 1000000,
        "ticket_limit": 100,
        "transport_row_limit": 10**12,
        "support_limit": 10**12,
    }
    params.update(changes)
    return rb.register_bernoulli_source(c, *decl, **params)


@pytest.fixture(scope="module")
def base():
    c = model()
    return register(c, declarations(c))


def results(reg):
    return rb.registered_bernoulli_record(rb.execute_registered_bernoulli_source(reg))


def test_new_closed_form_and_one_source_many_events(base, monkeypatch):
    calls = []
    original = rb.bernoulli_future_expectation

    def track(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(rb, "bernoulli_future_expectation", track)
    r = results(base)
    assert (
        len(calls) == 1
        and r["native_source_compilations"] == 1
        and r["transport_calls"] == 3
    )
    choice = r["evaluations"][0]["result"]["analyst"]
    assert q(choice["event_measure"]) == F(1, 12) and q(
        choice["population_material"]["mean"]
    ) == F(1937, 16)
    rare = r["evaluations"][1]["result"]["analyst"]
    assert q(rare["event_measure"]) == F(1, 48 * 10**2100)
    assert (
        len(r["attachment_links"][0]["matched"]) == 26
        and len(r["attachment_links"][0]["unused"]) == 22
    )
    assert (
        not r["full_primary_default_empirical_production_planner_adopted"]
        and r["lifecycle_utility_CS_total_denial_misallocation"] is None
    )


def test_new_pre_source_registration_no_backend_call(monkeypatch):
    c = model()
    decl = declarations(c)

    def fail(*args, **kwargs):
        raise AssertionError("native source called during registration")

    monkeypatch.setattr(rb, "bernoulli_future_expectation", fail)
    reg = register(c, decl)
    assert reg.digest and reg.bound["registered_contingencies"] == 48
    assert reg.bound["source_branch_rows_upper"] == 36


def test_new_zero_evidence_and_positive_tiny_unknown():
    c = model()
    r = results(register(c, declarations(c, rare=F(), open_mass=F(1, 10**1800))))
    assert r["evaluations"][1]["result"]["analyst"]["conditional_atoms"] is None
    a = r["evaluations"][0]["result"]["analyst"]
    assert a["population_material"]["mean"] is a["population_payment"][
        "mean"
    ] is None and q(a["population_material"]["unknown_mass"]) == F(1, 10**1800)


def test_new_multi_market_products_shared_legs_and_zero_classes():
    c = model(2, multi=True)
    reg = register(c, declarations(c, rare=F(), single_path=True, periods=1))
    r = results(reg)
    source = r["complete_native_source"]["details"]
    assert len(source["nodes"]) == 1
    assert reg.bound["source_branch_rows_upper"] == len(
        source["nodes"][0]["boundary"]["branches"]
    )
    assert any(
        p["taste_class"] == 1
        for ps in r["evaluations"][2]["result"]["analyst"][
            "all_original_person_profiles"
        ]
        for p in ps
    )
    assert any(len(p.legs) > 1 for p in c.spec.products)


def test_new_three_period_zero_arrival_no_person_and_full_zero_support():
    c = model(3, arrival=0.0)
    reg = register(c, declarations(c, rare=F(), single_path=True))
    r = results(reg)
    a = r["evaluations"][2]["result"]["analyst"]
    assert q(a["population_material"]["mean"]) == 0
    assert (
        reg.bound["registered_contingencies"] == 64
        and len(a["all_original_person_profiles"]) > 1
    )
    assert any(p for p in a["all_original_person_profiles"])


@pytest.mark.parametrize("weight", [F(), F(1, 10**2200)])
def test_new_needed_source_underflow_and_exact_zero_path(weight):
    c = model()
    r = results(register(c, declarations(c, underflow_weight=weight)))
    a = r["evaluations"][2]["result"]["analyst"]
    assert a["source_public_joint_known"] is (weight == 0)
    if weight:
        assert (
            a["event_measure"]
            is a["conditional_atoms"]
            is a["population_material"]
            is None
        )
    assert any(
        atom["path"] == "underflow" and q(atom["raw_joint_mass"]) == 0
        for atom in a["all_atoms"]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("support_limit", 1),
        ("conditional_call_limit", 1),
        ("tree_row_limit", 1),
        ("ticket_limit", 1),
    ],
)
def test_new_all_support_limits_before_native(field, value, monkeypatch):
    c = model()
    decl = declarations(c)

    def fail(*args, **kwargs):
        raise AssertionError("native source reached")

    monkeypatch.setattr(rb, "bernoulli_future_expectation", fail)
    with pytest.raises((rb.BernoulliRegistrationLimit, ValueError)):
        register(c, decl, **{field: value})


@pytest.mark.parametrize(
    "field,value", [("digest", "wrong"), ("rule", "old-rule"), ("bound", {"total": 1})]
)
def test_new_recomputed_registration_not_caller_seal(base, field, value, monkeypatch):
    bad = replace(base, **{field: value})
    before = rb.registered_bernoulli_record(bad)

    def fail(*args, **kwargs):
        raise AssertionError("native source reached")

    monkeypatch.setattr(rb, "bernoulli_future_expectation", fail)
    with pytest.raises(ValueError):
        rb.execute_registered_bernoulli_source(bad)
    assert before == rb.registered_bernoulli_record(bad)


@pytest.mark.parametrize(
    "change",
    [
        lambda d: (
            d[0],
            d[1],
            d[2],
            (replace(d[3][0], cases=d[3][0].cases[:-1]),),
            d[4],
            d[5],
        ),
        lambda d: (
            d[0],
            d[1],
            replace(d[2], signals=("ordinary", "rare")),
            d[3],
            d[4],
            d[5],
        ),
        lambda d: (d[0], d[1], d[2], d[3], d[4][:-1], d[5]),
        lambda d: (d[0], d[1], d[2], d[3], d[4], (replace(d[5][0], law="missing"),)),
        lambda d: (d[0], (replace(d[1][0], weight=F(1)), *d[1][1:]), *d[2:]),
        lambda d: (replace(d[0], next_record_id=1), *d[1:]),
    ],
)
def test_new_complete_declared_support_refusals(change, monkeypatch):
    c = model()
    decl = change(declarations(c))

    def fail(*args, **kwargs):
        raise AssertionError("native source reached")

    monkeypatch.setattr(rb, "bernoulli_future_expectation", fail)
    with pytest.raises(ValueError):
        register(c, decl)


def test_new_unused_zero_contingency_requires_four_net_components(monkeypatch):
    c = model()
    d = list(declarations(c))
    law = d[3][0]
    cases = list(law.cases)
    i = next(i for i, x in enumerate(cases) if x.observations[0].cells[0].persons)
    case = cases[i]
    obs = case.observations[0]
    cell = obs.cells[0]
    p = cell.persons[0]
    p = replace(p, payoff=replace(p.payoff, net_ticket_payment=F(123)))
    cell = replace(cell, persons=(p, *cell.persons[1:]))
    obs = replace(obs, cells=(cell, *obs.cells[1:]))
    cases[i] = replace(case, observations=(obs, *case.observations[1:]))
    d[3] = (replace(law, cases=cases),)
    with pytest.raises(ValueError, match="NET"):
        register(c, tuple(d))


def test_new_owned_inputs_exports_and_global_rng(base):
    before = rb.registered_bernoulli_record(base)
    state = np.random.get_state()
    r = results(base)
    after = np.random.get_state()
    assert (
        state[0] == after[0]
        and np.array_equal(state[1], after[1])
        and state[2:] == after[2:]
    )
    r["registration"]["recipe"]["model_json"] = "changed exported context"
    assert rb.registered_bernoulli_record(base) == before


def test_new_exact_recipe_match_and_failure_no_partial(base, monkeypatch):
    original = rb.bernoulli_future_expectation
    calls = []

    def bad(*args, **kwargs):
        calls.append(1)
        source = original(*args, **kwargs)
        d = _encoded(source.details)
        d["conditioning"] = "changed after registration"
        return BernoulliFutureResult(source.joint, source.expected_counts, d)

    monkeypatch.setattr(rb, "bernoulli_future_expectation", bad)
    with pytest.raises(ValueError, match="conditional"):
        rb.execute_registered_bernoulli_source(base)
    assert len(calls) == 1


def test_new_second_transport_failure_no_result_or_retry(base, monkeypatch):
    before = rb.registered_bernoulli_record(base)
    original = rb.transport.condition_source_future
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("injected pure second evaluation failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(rb.transport, "condition_source_future", fail)
    with pytest.raises(RuntimeError, match="injected"):
        rb.execute_registered_bernoulli_source(base)
    assert len(calls) == 2 and before == rb.registered_bernoulli_record(base)


def test_new_nonzero_start_has_no_inferred_earlier_population():
    c = model(3)
    d = list(declarations(c, periods=2))
    cases = []
    for case in d[3][0].cases:
        observations = []
        for obs in case.observations:
            cells = []
            for cell in obs.cells:
                people = tuple(
                    replace(p, person_key=(p.person_key[0] + 1, *p.person_key[1:]))
                    for p in cell.persons
                )
                cells.append(replace(cell, persons=people))
            observations.append(replace(obs, cells=cells))
        cases.append(replace(case, observations=observations))
    d[3] = (replace(d[3][0], cases=cases),)
    r = results(register(c, tuple(d), start_period=1))
    assert {n["period"] for n in r["complete_native_source"]["details"]["nodes"]} == {
        1,
        2,
    }
    profiles = r["evaluations"][2]["result"]["analyst"]["all_original_person_profiles"]
    assert all(p["person_key"][0] >= 1 for ps in profiles for p in ps)


def test_new_nonfinite_carrier_context_is_retained_without_passenger_incidence():
    c = model()
    d = list(declarations(c, single_path=True, periods=1, rare=F()))
    accounts = list(d[0].accounts)
    accounts[0] = (float("nan"), *accounts[0][1:])
    d[0] = replace(d[0], accounts=accounts)
    r = results(register(c, tuple(d)))
    raw = r["complete_native_source"]["details"]["initial_declared_ledger"]["accounts"][
        0
    ][0]
    assert raw["exact"] is None and raw["repr"] == "nan"
    assert (
        r["evaluations"][2]["result"]["analyst"]["population_material"]["mean"]
        is not None
    )


def test_new_mutable_caller_lists_cannot_change_owned_programme():
    c = model()
    d = list(declarations(c))
    for index in (1, 3, 4, 5):
        d[index] = list(d[index])
    reg = register(c, tuple(d))
    before = rb.registered_bernoulli_record(reg)
    for index in (1, 3, 4, 5):
        d[index].clear()
    assert rb.registered_bernoulli_record(reg) == before
    assert q(
        results(reg)["evaluations"][0]["result"]["analyst"]["population_material"][
            "mean"
        ]
    ) == F(1937, 16)


def test_new_unused_event_axis_refused_before_source(monkeypatch):
    c = model()
    d = list(declarations(c))
    law = d[3][0]
    cases = list(law.cases)
    index = next(
        i
        for i, row in enumerate(cases)
        if row.path == "early" and row.active == (False,)
    )
    row = cases[index]
    cases[index] = replace(
        row,
        memberships=(("choice", ((False,) * 3, (False,) * 3)), *row.memberships[1:]),
    )
    d[3] = (replace(law, cases=cases),)

    def fail(*args, **kwargs):
        raise AssertionError("native source reached")

    monkeypatch.setattr(rb, "bernoulli_future_expectation", fail)
    with pytest.raises(ValueError, match="memberships"):
        register(c, tuple(d))
