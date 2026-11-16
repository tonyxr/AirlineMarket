"""A.10/A.5/B0/V9/B4: NEW August24/114974 conditional population units."""

import pickle
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.dated_demand import compile_dated_demand
from airline_market.policy.dated_reserve import UtcBookingClock
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.sim import conditional_population as subject
from airline_market.sim import current_itinerary_service as service
from airline_market.sim import person_material_payoff as material
from airline_market.sim import service_ticket_payments as payments
from airline_market.sim.ordinary_cohort import (
    OrdinaryCohortMarketEngine,
    ordinary_person_panel,
)
from airline_market.sim.passenger_accounts import original_passenger_accounts
from airline_market.sim.payoff_logit import ContingentPayoff
from airline_market.sim.service_observations import service_observation_journal
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

LIMIT = 2_000_000_000
UNIT = "declared-native-fare-unit114974"
CALLS = {}


def q(v):
    return None if v is None else F(int(v["numerator"]), int(v["denominator"]))


def fresh_source(mode="base"):
    spec = screen_spec(
        {
            "T": 3,
            "actions": 2,
            "n_spokes": 1,
            "capacity": 20,
            "capacity_support": [3, 4],
        }
    )
    c = compile_market(
        replace(
            spec,
            demand=replace(
                spec.demand,
                mode="poisson",
                arrival_probability=None,
                market_weights=None,
                intensity={
                    r.id: PeriodSchedule.parse({"constant": 4.0}) for r in spec.markets
                },
            ),
        )
    )
    start = datetime(2027, 8, 24, tzinfo=timezone.utc)
    clock = UtcBookingClock(tuple(start + timedelta(hours=12 * t) for t in range(4)))
    dep = tuple(
        (leg.id, start + timedelta(hours=84 + 4 * i))
        for i, leg in enumerate(c.spec.legs)
    )
    if mode == "unreserved":
        dep = tuple(
            (leg.id, start + timedelta(hours=84 if leg.id == "B_OH0" else 120 + 4 * i))
            for i, leg in enumerate(c.spec.legs)
        )
    scenario = compile_dated_demand(c, clock, dep)
    if mode == "unreserved":
        scenario = replace(
            scenario,
            physical_shocks=(
                B1DatedShock(
                    "NEW-population-exit114974",
                    "rival_exit",
                    "B_OH0",
                    3,
                    rival_carrier="B",
                ),
            ),
        )
    target = c.product_index["B_OH0" if mode == "unreserved" else "A_OH0"]
    market = int(c.product_market[target])
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(114974, "unit/V9_conditional_population/" + mode)
    )
    assert tape.gumbel.shape[1] >= 3
    g = tape.gumbel.copy()
    g[:] = -1e9
    g[:, :, -1] = 0
    buyers = 2 if mode == "base" else 1 if mode == "unreserved" else 0
    g[0, :buyers, target] = 1e9
    g[0, buyers:3, -1] = 2e9
    g.setflags(write=False)
    marks = (
        ()
        if mode == "empty"
        else tuple(ArrivalMark((i + 1) / 4, market, 0, 0, 0.55) for i in range(3))
    )
    tape = replace(tape, arrival_marks=(marks, (), ()), gumbel=g)
    e = OrdinaryCohortMarketEngine(scenario)
    e.reset(tape=tape)
    actions = {
        carrier.code: PriceAction({j: 260 if i == 0 else 390 for j in c.products_of[i]})
        for i, carrier in enumerate(c.spec.carriers)
    }
    for _ in range(3):
        e.step(actions)
    accounts = original_passenger_accounts(e)
    panel = ordinary_person_panel(e.ordinary_periods(), accounts)
    journal = service_observation_journal(
        c,
        panel,
        clock,
        dep,
        definition="NEW fixed original population114974/" + mode,
        observation_scope="declared-individual-leg-observations",
        context_row_limit=200000,
        observation_limit=1000,
    )
    return e, c, journal


def make_programme(source):
    _e, _c, journal = source
    ctx = journal.context
    people = ctx["persons"]
    current = tuple(p for p in people if p["current_ticket"] is not None)
    dep = tuple((leg, payments.utc(at)) for leg, at in ctx["leg_departures"])
    as_of = max(at for _, at in dep) + timedelta(hours=3)
    physical = []
    for index, name in enumerate(("one-first", "one-last", "cancelled", "two-seats")):
        ops = tuple(
            service.LegOperation(
                leg,
                "cancelled" if index == 2 and leg == "A_OH0" else "operates",
                0
                if index == 2 and leg == "A_OH0"
                else (1 if index < 2 else 2)
                if leg == "A_OH0"
                else 20,
                None if index == 2 and leg == "A_OH0" else at,
                None if index == 2 and leg == "A_OH0" else at + timedelta(hours=2),
                at - timedelta(hours=1) if index == 2 and leg == "A_OH0" else None,
            )
            for leg, at in dep
        )
        intent = tuple(
            (p["current_ticket"], leg, True)
            for p in current
            for leg in p["required_legs"]
        )
        order = tuple(
            (
                leg,
                tuple(
                    p["current_ticket"]
                    for p in (tuple(reversed(current)) if index == 1 else current)
                    if leg in p["required_legs"]
                ),
            )
            for leg, _ in dep
        )
        physical.append(
            subject.PhysicalServiceProgramme(
                name,
                ops,
                intent,
                order,
                (),
                "NEW full physical programme114974/" + name,
                "declared114974",
            )
        )
    info = tuple(
        subject.OriginalPersonInformation(
            p["person_key"],
            (
                material.DeclaredDecisionInformation(
                    p["person_key"],
                    "original-opportunity",
                    payments.utc(ctx["booking_clock"][p["person_key"][0]]),
                    (),
                    (),
                    "NEW explicit shared earlier empty service-transfer information114974",
                    "declared114974",
                ),
            ),
        )
        for p in people
    )
    panel = {
        tuple(p["potential_coordinate"]): p
        for p in ctx["complete_ordinary_panel"]["persons"]
    }
    cells = []
    for physical_plan in physical:
        for financial in ("closed", "open"):
            closed = financial == "closed"
            persons = []
            transfer_ids = []
            for p in people:
                account = panel[p["person_key"]]["booked_passenger_account"]
                baseline = (
                    F() if account is None else q(account["net_ticket_fare_paid"])
                )
                outcomes = []
                for outcome in (
                    (False, True) if p["current_ticket"] is not None else (None,)
                ):
                    transfers = ()
                    if outcome is False:
                        ticket = next(
                            t
                            for t in account["tickets"]
                            if t["record_id"] == p["current_ticket"]
                        )
                        fare = q(ticket["fare"]["exact"])
                        when = as_of - timedelta(minutes=30)
                        transfers = (
                            payments.AdditionalTicketTransfer(
                                f"{physical_plan.name}-{financial}-{p['person_key'][1]}-refund",
                                p["person_key"],
                                p["current_ticket"],
                                ticket["carrier"],
                                "additional_refund",
                                float(fare),
                                when,
                                when,
                                "NEW explicit conditional refund114974",
                            ),
                        )
                    payment = (
                        (baseline - (fare if outcome is False else 0))
                        if closed
                        else None
                    )
                    originally_booked = p["original_record"] is not None
                    benefit = (
                        F(540 if outcome else 0)
                        if outcome is not None
                        else None
                        if originally_booked
                        else F(4)
                    )
                    cancellation = (
                        F(17 if outcome is False else 0)
                        if outcome is not None
                        else None
                        if originally_booked
                        else F()
                    )
                    payoff = ContingentPayoff(
                        benefit, payment, F(6 if originally_booked else 1), cancellation
                    )
                    outcomes.append(
                        subject.ServiceOutcomeTerms(
                            outcome,
                            transfers,
                            closed,
                            "NEW financial-tail declaration114974/" + financial,
                            payoff,
                            tuple(
                                (k, "NEW explicit component114974:" + k)
                                for k in material.COMPONENTS
                            ),
                            True,
                            "NEW material-tail declaration114974",
                        )
                    )
                    transfer_ids.extend(t.event_id for t in transfers)
                persons.append(subject.PersonServiceTerms(p["person_key"], outcomes))
            cells.append(
                subject.PopulationPayoffCell(
                    physical_plan.name,
                    financial,
                    persons,
                    transfer_ids,
                    "NEW full conditional cell114974",
                    "declared114974",
                )
            )
    return subject.ConditionalPopulationProgramme(
        physical,
        ("closed", "open"),
        cells,
        info,
        as_of,
        UNIT,
        subject.SCOPE,
        "Conditional on THIS supplied complete realized native prefix; not native arrival/probability truth",
        "NEW owned full joint population programme114974",
        "declared114974",
    )


def laws():
    forecast = tuple(w for p in (F(1, 5), F(1, 10), F(1, 5), F(1, 2)) for w in (p, F()))
    reference = tuple(
        w for p in (F(3, 10), F(3, 10), F(3, 10), F(1, 10)) for w in (p, F())
    )
    eps = F(1, 10**1300)
    tiny = tuple(eps if i == 5 else (1 - eps) * w for i, w in enumerate(forecast))
    return tuple(
        subject.ConditionalPopulationLaw(
            name,
            w,
            "NEW full joint conditional physical/financial declaration114974",
            "declared114974",
        )
        for name, w in (
            ("forecast", forecast),
            ("reference", reference),
            ("tiny-open", tiny),
        )
    )


@pytest.fixture(scope="module")
def source():
    return fresh_source()


@pytest.fixture(scope="module")
def programme(source):
    return make_programme(source)


@pytest.fixture(scope="module")
def result(source, programme):
    e, c, j = source
    before = pickle.dumps(e.snapshot(), protocol=5)
    rng = pickle.dumps(np.random.get_state(), protocol=5)
    mp = pytest.MonkeyPatch()
    for owner, name in (
        (service, "execute_current_itinerary_service"),
        (payments, "reconcile_service_ticket_payments"),
        (material, "reconcile_person_material_payoff"),
    ):
        original = getattr(owner, name)

        def observed(*args, _original=original, _name=name, **kwargs):
            CALLS[_name] = CALLS.get(_name, 0) + 1
            return _original(*args, **kwargs)

        mp.setattr(owner, name, observed)
    try:
        r = subject.evaluate_conditional_population(
            c, j, programme, laws(), row_limit=LIMIT
        )
    finally:
        mp.undo()
    assert (
        pickle.dumps(e.snapshot(), protocol=5) == before
        and pickle.dumps(np.random.get_state(), protocol=5) == rng
    )
    return r


def change_term(programme, cell_index, person_index, outcome_index, **changes):
    cells = list(programme.cells)
    cell = cells[cell_index]
    persons = list(cell.persons)
    person = persons[person_index]
    outcomes = list(person.outcomes)
    outcomes[outcome_index] = replace(outcomes[outcome_index], **changes)
    persons[person_index] = replace(person, outcomes=outcomes)
    cells[cell_index] = replace(cell, persons=persons)
    return replace(programme, cells=cells)


def test_full_joint_closed_forms_and_all_original_people(result):
    a = result.analyst
    forecast, reference, tiny = a["law_results"]
    assert (
        len(a["all_original_people"]) == 3
        and len(a["cells"]) == 8
        and len(a["physical_results"]) == 4
    )
    assert forecast["population_material"]["mean"] == F(3431, 10)
    assert reference["population_material"]["mean"] == F(973, 5)
    assert forecast["current_service_marginals"] == (F(7, 10), F(3, 5), None)
    assert forecast["pairwise_current_service"][0]["covariance_declared_law"] == F(
        2, 25
    )
    assert reference["pairwise_current_service"][0]["covariance_declared_law"] == F(
        -3, 50
    )
    assert forecast["delivered_current_count_variance_declared_law"] == F(61, 100)
    assert reference["delivered_current_count_variance_declared_law"] == F(9, 25)
    assert (
        tiny["population_material"]["mean"] is None
        and tiny["population_payment"]["mean"] is None
    )
    assert tiny["population_material"]["unknown_mass"] == F(1, 10**1300)
    assert (
        a["native_probability_knowledge_preferences_authenticated"] is False
        and a["lifecycle_utility_CS_total_denial_misallocation"] is None
    )


def test_actual_forward_counts_no_native_or_rng_mutation(result):
    assert CALLS == {
        "execute_current_itinerary_service": 4,
        "reconcile_service_ticket_payments": 8,
        "reconcile_person_material_payoff": 8,
    }
    assert (
        result.analyst["service_state_machine_calls"] == 4
        and result.analyst[
            "source_native_engine_account_panel_actor_RNG_or_choice_executed"
        ]
        is False
    )


def test_zero_unknown_full_support_and_outside_not_false_delivery(result):
    a = result.analyst
    f = a["law_results"][0]
    assert (
        f["population_material"]["contributions"][1] == 0
        and a["cells"][1]["population_material"] is None
    )
    assert all(c["delivery"][2] is None for c in a["cells"])
    assert all(
        p["covariance_declared_law"] is None for p in f["pairwise_current_service"][1:]
    )
    assert all(len(c["person_material"]) == 3 for c in a["cells"])
    assert all(len(c["persons"][0]["outcomes"]) == 2 for c in a["programme"]["cells"])


def test_common_earlier_info_and_owned_complete_exports(result):
    assert (
        subject._units(subject.conditional_population_record(result))
        <= result.analyst["structural_bound"]
    )
    a = result.analyst
    views = [
        c["material"]["all_original_persons"][0]["declared_decision_information"]
        for c in a["cells"]
    ]
    assert all(v == views[0] for v in views)
    assert views[0][0]["ex_post_payoff_components_or_future_program_included"] is False
    exported = subject.conditional_population_record(result)
    exported["analyst"]["programme"]["definition"] = "mutated"
    assert a["programme"]["definition"] != "mutated"


@pytest.mark.parametrize(
    "kind",
    [
        "missing_cell",
        "reorder_cell",
        "missing_person",
        "missing_unreachable_outcome",
        "integer_outcome",
        "net_payment",
        "missing_source",
        "integer_closure",
        "missing_transfer_order",
        "bad_transfer_time",
        "wrong_ticket",
        "wrong_unit_scope",
        "missing_leg",
        "bad_priority",
        "future_info",
        "missing_info",
        "common_date",
        "limit",
        "missing_law_weight",
        "float_weight",
        "negative_weight",
        "nonunit_weight",
    ],
)
def test_refusals_before_any_physical_forward(source, programme, monkeypatch, kind):
    _e, c, j = source
    p = programme
    ls = laws()
    limit = LIMIT
    monkeypatch.setattr(
        service,
        "execute_current_itinerary_service",
        lambda *a, **k: pytest.fail("unexpected service forward"),
    )
    if kind in ("missing_cell", "reorder_cell"):
        p = replace(
            p,
            cells=p.cells[:-1] if kind == "missing_cell" else tuple(reversed(p.cells)),
        )
    elif kind == "missing_person":
        cells = list(p.cells)
        cells[-1] = replace(cells[-1], persons=cells[-1].persons[:-1])
        p = replace(p, cells=cells)
    elif kind == "missing_unreachable_outcome":
        cell = list(p.cells)
        persons = list(cell[-1].persons)
        persons[0] = replace(persons[0], outcomes=persons[0].outcomes[:1])
        cell[-1] = replace(cell[-1], persons=persons)
        p = replace(p, cells=cell)
    elif kind == "integer_outcome":
        p = change_term(p, 0, 0, 0, current_itinerary_complete=0)
    elif kind == "net_payment":
        term = p.cells[-1].persons[0].outcomes[1]
        p = change_term(
            p, 7, 0, 1, payoff=replace(term.payoff, net_ticket_payment=F(260))
        )
    elif kind == "missing_source":
        p = change_term(p, 0, 0, 1, component_sources=())
    elif kind == "integer_closure":
        p = change_term(p, 0, 0, 1, no_further_ticket_transfers=1)
    elif kind == "missing_transfer_order":
        cells = list(p.cells)
        cells[0] = replace(cells[0], transfer_order=())
        p = replace(p, cells=cells)
    elif kind in ("bad_transfer_time", "wrong_ticket"):
        term = p.cells[0].persons[0].outcomes[0]
        t = term.transfers[0]
        t = (
            replace(t, recorded_at=t.occurred_at - timedelta(hours=1))
            if kind == "bad_transfer_time"
            else replace(t, ticket_record=9999)
        )
        p = change_term(p, 0, 0, 0, transfers=(t,))
    elif kind == "wrong_unit_scope":
        p = replace(p, conditioning_scope="native authenticated law")
    elif kind in ("missing_leg", "bad_priority"):
        physical = list(p.physical)
        s = physical[-1]
        physical[-1] = (
            replace(s, operations=s.operations[:-1])
            if kind == "missing_leg"
            else replace(s, boarding_orders=())
        )
        p = replace(p, physical=physical)
    elif kind == "future_info":
        infos = list(p.information)
        infos[0] = replace(
            infos[0],
            decisions=(replace(infos[0].decisions[0], service_event_ids=("future",)),),
        )
        p = replace(p, information=infos)
    elif kind == "missing_info":
        p = replace(p, information=p.information[:-1])
    elif kind == "common_date":
        p = replace(p, as_of=payments.utc(j.context["booking_closed_at"]))
    elif kind == "limit":
        limit = 1
    else:
        w = list(ls[0].weights)
        if kind == "missing_law_weight":
            w.pop()
        elif kind == "float_weight":
            w[0] = float(w[0])
        elif kind == "negative_weight":
            w[0] = F(-1)
        else:
            w = [x / 2 for x in w]
        ls = (replace(ls[0], weights=w), *ls[1:])
    with pytest.raises((TypeError, ValueError, RuntimeError)):
        subject.evaluate_conditional_population(c, j, p, ls, row_limit=limit)


@pytest.mark.parametrize("mode", ["empty", "outside", "unreserved"])
def test_new_empty_outside_and_native_cancelled_original_support(mode):
    f = fresh_source(mode)
    p = make_programme(f)
    r = subject.evaluate_conditional_population(f[1], f[2], p, laws(), row_limit=LIMIT)
    a = r.analyst
    if mode == "empty":
        assert a["all_original_people"] == () and a["empty_population_sum_only"] is True
        assert all(law["population_material"]["mean"] == 0 for law in a["law_results"])
    else:
        assert len(a["all_original_people"]) == 3 and all(
            x is None for x in a["law_results"][0]["current_service_marginals"]
        )
        if mode == "outside":
            assert a["law_results"][0]["population_material"]["mean"] == 9
        else:
            assert a["source_context"]["persons"][0]["original_record"] is not None
            assert a["source_context"]["persons"][0]["current_ticket"] is None
            assert (
                a["law_results"][0]["population_payment"]["mean"] == 0
                and a["law_results"][0]["population_material"]["mean"] is None
            )
    assert a["population_service_rate"] is None


def test_single_person_open_financial_tail_does_not_null_other_people(
    source, programme
):
    p = programme
    for cell in range(0, 8, 2):
        for outcome in (0, 1):
            term = p.cells[cell].persons[0].outcomes[outcome]
            p = change_term(
                p,
                cell,
                0,
                outcome,
                no_further_ticket_transfers=False,
                payoff=replace(term.payoff, net_ticket_payment=None),
            )
    r = subject.evaluate_conditional_population(
        source[1], source[2], p, laws()[:1], row_limit=LIMIT
    )
    result = r.analyst["law_results"][0]
    assert (
        result["population_material"]["mean"] is None
        and result["persons"][0]["material"]["mean"] is None
    )
    assert (
        result["persons"][1]["material"]["mean"] is not None
        and result["persons"][2]["material"]["mean"] == 3
    )


def test_nonfinite_zero_weight_transfer_inputs_remain_owned(source, programme):
    # A zero-weight OPEN outcome still retains unknown/nonfinite transfer bits.
    p = programme
    term = p.cells[1].persons[0].outcomes[0]
    t = replace(term.transfers[0], amount=float("nan"))
    p = change_term(p, 1, 0, 0, transfers=(t,))
    r = subject.evaluate_conditional_population(
        source[1], source[2], p, laws()[:1], row_limit=LIMIT
    )
    export = subject.conditional_population_record(r)
    assert (
        export["analyst"]["programme"]["cells"][1]["persons"][0]["outcomes"][0][
            "transfers"
        ][0]["amount"]["binary64_bits"]
        == subject._encoded(float("nan"))["binary64_bits"]
    )
    assert r.analyst["law_results"][0]["population_material"]["mean"] == F(3431, 10)


def test_failure_after_one_new_service_has_no_positive_result_or_source_mutation(
    source, programme, monkeypatch
):
    e, c, j = source
    before = pickle.dumps((e.snapshot(), j), protocol=5)
    original = service.execute_current_itinerary_service
    calls = []

    def failed(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("NEW injected second programme failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "execute_current_itinerary_service", failed)
    with pytest.raises(RuntimeError, match="NEW injected"):
        subject.evaluate_conditional_population(
            c, j, programme, laws(), row_limit=LIMIT
        )
    assert (
        len(calls) == 2
        and pickle.dumps((e.snapshot(), j), protocol=5) == before
        and j.observations == ()
    )


def test_signed_rebate_and_signed_zero_are_retained(source, programme):
    p = programme
    term = p.cells[6].persons[0].outcomes[1]
    when = p.as_of - timedelta(minutes=30)
    key = p.cells[6].persons[0].person_key
    record = source[2].context["persons"][0]["current_ticket"]
    rebate = payments.AdditionalTicketTransfer(
        "NEW-signed-rebate114974",
        key,
        record,
        "A",
        "additional_payment",
        -7.0,
        when,
        when,
        "Explicit negative additional payment, not carrier penalty",
    )
    p = change_term(
        p,
        6,
        0,
        1,
        transfers=(rebate,),
        payoff=replace(term.payoff, net_ticket_payment=F(253)),
    )
    cells = list(p.cells)
    cells[6] = replace(
        cells[6], transfer_order=(*cells[6].transfer_order, rebate.event_id)
    )
    p = replace(p, cells=cells)
    zero = p.cells[1].persons[0].outcomes[0]
    p = change_term(p, 1, 0, 0, transfers=(replace(zero.transfers[0], amount=-0.0),))
    result = subject.evaluate_conditional_population(
        source[1], source[2], p, laws()[:1], row_limit=LIMIT
    )
    assert result.analyst["law_results"][0]["population_material"]["mean"] == F(1733, 5)
    assert result.analyst["cells"][6]["person_payments"][0] == 253
    amount = result.analyst["programme"]["cells"][1]["persons"][0]["outcomes"][0][
        "transfers"
    ][0]["amount"]
    assert amount["binary64_bits"] == "8000000000000000"
