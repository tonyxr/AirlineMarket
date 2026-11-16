"""A.10/A.5/B0/V9: NEW original-person material payoff/information references."""

import importlib.util
import os
import pickle
import sys
from dataclasses import replace
from datetime import timedelta
from fractions import Fraction

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.policy.dated_demand import compile_dated_demand
from airline_market.policy.dated_rescue import DatedRebookingPolicy
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.sim import service_ticket_payments as settle
from airline_market.sim.current_itinerary_service import (
    LegOperation,
    execute_current_itinerary_service,
)
from airline_market.sim.ordinary_cohort import (
    OrdinaryCohortMarketEngine,
    ordinary_person_panel,
)
from airline_market.sim.passenger_accounts import original_passenger_accounts
from airline_market.sim.payoff_logit import ContingentPayoff
from airline_market.sim.service_observations import service_observation_journal
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from tests.unit import test_dated_demand as dated
from tests.unit import test_rationing_native as native

if os.environ.get("PERSON_MATERIAL_PAYOFF_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_person_material_payoff", os.environ["PERSON_MATERIAL_PAYOFF_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.sim import person_material_payoff as module


def booking_source(*, replacement=False, reentry=True, partial=False, empty=False):
    c, clock, dep = dated.inputs()
    scenario = compile_dated_demand(c, clock, dep)
    kwargs = {}
    if replacement:
        scenario = replace(
            scenario,
            physical_shocks=(
                B1DatedShock(
                    "new-material-payoff-source-exit",
                    "rival_exit",
                    "B_OH0",
                    3,
                    rival_carrier="B",
                ),
            ),
        )
        if reentry:
            kwargs["dated_rebooking"] = DatedRebookingPolicy(
                72, 20, 100, interline=True
            )
    e = OrdinaryCohortMarketEngine(scenario, **kwargs)
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(73974, "unit/V9_person_material_payoff")
    )
    j = next(
        i
        for i, p in enumerate(c.spec.products)
        if p.carrier == ("B" if replacement else "A") and len(p.legs) == 2
    )
    m = int(c.product_market[j])
    g = tape.gumbel.copy()
    g[0, :2, :] = -1e6
    g[0, 0, j] = 1e6
    g[0, 1, -1] = 1e6
    g.setflags(write=False)
    marks = (ArrivalMark(0.1, m, 0, 0, 0), ArrivalMark(0.2, m, 0, 0, 0))
    if empty:
        marks = ()
    e.reset(
        tape=replace(
            tape, arrival_marks=(marks, *(() for _ in tape.arrival_marks[1:])), gumbel=g
        )
    )
    while not e.terminal():
        e.step(native.actions(e))
    panel = ordinary_person_panel(e.ordinary_periods(), original_passenger_accounts(e))
    journal = service_observation_journal(
        c,
        panel,
        clock,
        dep,
        definition="NEW native booking input with explicit independent unit service program",
        observation_scope="declared-individual-leg-observations",
        context_row_limit=10000,
        observation_limit=100,
    )
    current = [p for p in journal.context["persons"] if p["current_ticket"] is not None]
    ops = []
    for leg, scheduled in dep:
        second = current and leg == current[0]["required_legs"][-1]
        departure = scheduled + timedelta(hours=4 if second else 0)
        if partial and second:
            ops.append(LegOperation(leg, "cancelled", 20, None, None, departure))
        else:
            ops.append(
                LegOperation(
                    leg, "operates", 20, departure, departure + timedelta(hours=1), None
                )
            )
    intents = tuple(
        (p["current_ticket"], leg, True) for p in current for leg in p["required_legs"]
    )
    orders = tuple(
        (leg, tuple(p["current_ticket"] for p in current if leg in p["required_legs"]))
        for leg, _ in dep
    )
    connections = tuple(
        (p["current_ticket"], p["required_legs"][0], p["required_legs"][1], 3600_000000)
        for p in current
    )
    service = execute_current_itinerary_service(
        c,
        journal,
        ops,
        intents,
        orders,
        connections,
        definition="NEW deterministic service unit supplied full plans, not physical probability authentication",
        support_limit=10000,
    )
    as_of = max(settle.utc(s["at"]) for s in service.steps) + timedelta(hours=5)
    followup = tuple(
        settle.TicketPaymentFollowup(
            p["person_key"],
            as_of,
            True,
            "NEW explicit unit financial-tail declaration; not service-implied closure",
        )
        for p in service.journal.context["persons"]
    )
    return e, service, as_of, followup


def fixture(*, open_payment=False, transfer_amount=0.1, late_record=False, **kwargs):
    f = booking_source(**kwargs)
    _, service, as_of, followup = f
    panel = service.journal.context["complete_ordinary_panel"]["persons"]
    booked = next((p for p in panel if p["original_record"] is not None), None)
    transfers = ()
    if (
        booked
        and booked["booked_passenger_account"]["current_active_record"] is not None
    ):
        account = booked["booked_passenger_account"]
        t = next(
            t
            for t in account["tickets"]
            if t["record_id"] == account["current_active_record"]
        )
        occurred = settle.utc(service.journal.context["booking_closed_at"]) + timedelta(
            hours=1
        )
        transfers = (
            settle.AdditionalTicketTransfer(
                "new-unit-transfer",
                booked["potential_coordinate"],
                t["record_id"],
                t["carrier"],
                "additional_payment",
                transfer_amount,
                occurred,
                occurred + timedelta(hours=2 if late_record else 0),
                "NEW explicit unit extra transfer",
            ),
        )
    if open_payment:
        followup = tuple(
            replace(row, no_further_ticket_transfers=False) for row in followup
        )
    payments = settle.reconcile_service_ticket_payments(
        service,
        transfers,
        followup,
        as_of=as_of,
        native_ticket_unit="explicit common unit for NEW material-payoff witness",
        definition="NEW source with declared additional transfer and financial closure",
        row_limit=10000,
    )
    declarations = []
    for p in payments.data["all_original_persons"]:
        key = p["person_key"]
        info = module.DeclaredDecisionInformation(
            key,
            "original-opportunity",
            settle.utc(service.journal.context["booking_clock"][key[0]]),
            (),
            (),
            "NEW declared original opportunity information only; no future service/payment facts",
            "NEW information declaration",
        )
        payoff = ContingentPayoff(
            Fraction(200 if p["original_record"] is not None else 30),
            settle._rational(p["declared_closed_total_ticket_payment"]),
            Fraction(5 if p["original_record"] is not None else 2),
            Fraction(0),
        )
        declarations.append(
            module.OriginalPersonMaterialPayoff(
                key,
                payoff,
                tuple(
                    (k, "NEW explicit unit component declaration: " + k)
                    for k in module.COMPONENTS
                ),
                as_of,
                True,
                "NEW explicit unit material-tail closure; not service-implied",
                (info,),
            )
        )
    return f, payments, tuple(declarations)


def run(f, **changes):
    _, payments, declarations = f
    args = dict(
        payments=payments,
        declarations=declarations,
        payoff_unit=payments.data["native_ticket_unit"],
        definition="NEW ex-post declared material components with explicit decision-information references; no utility/CS adoption",
        scope=module.SCOPE,
        shock_rule=module.SHOCK_RULE,
        row_limit=100000,
    )
    args.update(changes)
    return module.reconcile_person_material_payoff(**args)


def q(value):
    return (
        None
        if value is None
        else Fraction(int(value["numerator"]), int(value["denominator"]))
    )


def changed_payoff(d, field, value):
    return replace(d, payoff=replace(d.payoff, **{field: value}))


def test_exact_four_components_all_originals_and_source_identity():
    f = fixture()
    e = f[0][0]
    before = pickle.dumps(
        (e.world, e.trace, e.metrics, f[0][1], f[1], np.random.get_state())
    )
    r = run(f).data
    expected = sum(
        (
            d.payoff.service_benefit
            - d.payoff.net_ticket_payment
            - d.payoff.waiting_loss
            - d.payoff.cancellation_loss
            for d in f[2]
        ),
        Fraction(0),
    )
    assert q(r["declared_closed_population_material_payoff"]) == expected
    assert len(r["all_original_persons"]) == 2
    assert q(r["all_original_persons"][1]["declared_closed_material_payoff"]) == 28
    assert (
        r["population_expectation_consumer_surplus_total_surplus_denial_misallocation"]
        is None
    )
    assert (
        pickle.dumps(
            (e.world, e.trace, e.metrics, f[0][1], f[1], np.random.get_state())
        )
        == before
    )


def test_open_payment_cannot_use_observed_prefix_as_final_component():
    f = fixture(open_payment=True)
    r = run(f).data
    assert r["declared_closed_population_material_payoff"] is None
    assert all(
        p["declared_closed_material_payoff"] is None for p in r["all_original_persons"]
    )
    p = f[1].data["all_original_persons"][0]
    bad = changed_payoff(
        f[2][0], "net_ticket_payment", q(p["observed_net_ticket_payment"])
    )
    with pytest.raises(ValueError):
        run(f, declarations=(bad, f[2][1]))


def test_complete_service_does_not_close_material_tail():
    f = fixture()
    d = replace(f[2][0], no_further_material_components=False)
    r = run(f, declarations=(d, f[2][1])).data
    assert (
        r["all_original_persons"][0]["declared_component_arithmetic_diagnostic"]
        is not None
    )
    assert r["all_original_persons"][0]["declared_closed_material_payoff"] is None
    assert r["declared_closed_population_material_payoff"] is None


@pytest.mark.parametrize(
    "field", ["service_benefit", "waiting_loss", "cancellation_loss"]
)
def test_unknown_component_nulls_whole_person_and_population_without_filtering(field):
    f = fixture()
    d = changed_payoff(f[2][0], field, None)
    r = run(f, declarations=(d, f[2][1])).data
    assert r["declared_closed_population_material_payoff"] is None
    assert (
        r["all_original_persons"][0]["declared_component_arithmetic_diagnostic"] is None
    )
    assert q(r["all_original_persons"][1]["declared_closed_material_payoff"]) == 28


def test_unknown_source_transfer_stays_unknown_in_explicit_payment_component():
    f = fixture(transfer_amount=None)
    r = run(f).data
    assert f[2][0].payoff.net_ticket_payment is None
    assert r["declared_closed_population_material_payoff"] is None
    assert r["complete_payment_input"]["additional_transfers"][0]["amount"] is None


def test_exact_binary64_payment_cannot_be_replaced_by_decimal_rounding():
    f = fixture()
    d = f[2][0]
    rounded = Fraction(str(float(d.payoff.net_ticket_payment)))
    assert rounded != d.payoff.net_ticket_payment
    with pytest.raises(ValueError):
        run(f, declarations=(changed_payoff(d, "net_ticket_payment", rounded), f[2][1]))


def test_signed_losses_and_tiny_positive_values_remain_exact():
    f = fixture()
    a = replace(
        f[2][0],
        payoff=ContingentPayoff(
            Fraction(-1000),
            f[2][0].payoff.net_ticket_payment,
            Fraction(-7),
            Fraction(-2),
        ),
    )
    tiny = Fraction(1, 10**600)
    b = replace(
        f[2][1], payoff=ContingentPayoff(tiny, Fraction(0), Fraction(0), Fraction(0))
    )
    r = run(f, declarations=(a, b)).data
    assert q(r["all_original_persons"][0]["declared_closed_material_payoff"]) < 0
    assert q(r["all_original_persons"][1]["declared_closed_material_payoff"]) == tiny


@pytest.mark.parametrize(
    "what",
    [
        "missing_outside",
        "duplicate",
        "missing_source",
        "wrong_unit",
        "wrong_scope",
        "wrong_shocks",
        "wrong_date",
    ],
)
def test_complete_explicit_support_units_sources_scope_and_dates_required(what):
    f = fixture()
    changes = {}
    if what == "missing_outside":
        changes["declarations"] = f[2][:-1]
    if what == "duplicate":
        changes["declarations"] = (f[2][0], f[2][0])
    if what == "missing_source":
        changes["declarations"] = (
            replace(f[2][0], component_sources=f[2][0].component_sources[:-1]),
            f[2][1],
        )
    if what == "wrong_unit":
        changes["payoff_unit"] = "unmatched dollars"
    if what == "wrong_scope":
        changes["scope"] = "native lifecycle welfare"
    if what == "wrong_shocks":
        changes["shock_rule"] = "sum all search shocks"
    if what == "wrong_date":
        changes["declarations"] = (
            replace(
                f[2][0], components_recorded_at=f[0][2] - timedelta(microseconds=1)
            ),
            f[2][1],
        )
    with pytest.raises(ValueError):
        run(f, **changes)


def test_original_information_view_has_no_future_program_or_ex_post_components():
    f = fixture()
    r = run(f).data
    info = r["all_original_persons"][0]["declared_decision_information"][0]
    assert info["service_events"] == () and info["additional_transfers"] == ()
    assert not info["ex_post_payoff_components_or_future_program_included"]
    assert (
        "complete_payment_input" not in info and "four_explicit_components" not in info
    )
    assert r["complete_payment_input"]["complete_service_input"]["observations"]


@pytest.mark.parametrize("which", ["service", "transfer", "equal_time"])
def test_future_and_equal_time_disclosures_refused(which):
    f = fixture()
    d = f[2][0]
    i = d.decision_information[0]
    if which == "service":
        event = f[1].data["complete_service_input"]["observations"][0]
        i = replace(i, service_event_ids=(event["event_id"],))
    else:
        event = f[1].data["additional_transfers"][0]
        i = replace(i, transfer_event_ids=(event["event_id"],))
        if which == "equal_time":
            i = replace(i, decision_at=settle.utc(event["recorded_at"]))
    with pytest.raises(ValueError):
        run(f, declarations=(replace(d, decision_information=(i,)), f[2][1]))


def test_occurrence_before_decision_is_insufficient_when_recorded_later():
    f = fixture(late_record=True)
    event = f[1].data["additional_transfers"][0]
    decision = settle.utc(event["occurred_at"]) + timedelta(hours=1)
    i = replace(
        f[2][0].decision_information[0],
        decision_at=decision,
        transfer_event_ids=(event["event_id"],),
    )
    with pytest.raises(ValueError):
        run(f, declarations=(replace(f[2][0], decision_information=(i,)), f[2][1]))


def test_explicit_later_own_information_multiple_snapshots_not_extra_people():
    f = fixture()
    d = f[2][0]
    i = d.decision_information[0]
    service_ids = tuple(
        e["event_id"] for e in f[1].data["complete_service_input"]["observations"]
    )
    later = replace(
        i,
        decision_id="declared-later-information",
        decision_at=f[0][2],
        service_event_ids=service_ids,
        transfer_event_ids=("new-unit-transfer",),
    )
    r = run(f, declarations=(replace(d, decision_information=(i, later)), f[2][1])).data
    assert len(r["all_original_persons"]) == 2
    assert len(r["all_original_persons"][0]["declared_decision_information"]) == 2
    assert (
        len(
            r["all_original_persons"][0]["declared_decision_information"][-1][
                "service_events"
            ]
        )
        == 2
    )


def test_other_person_private_record_unknown_and_duplicate_ids_refused():
    f = fixture()
    event = f[1].data["additional_transfers"][0]
    i = replace(
        f[2][1].decision_information[0],
        decision_at=f[0][2],
        transfer_event_ids=(event["event_id"],),
    )
    with pytest.raises(ValueError):
        run(f, declarations=(f[2][0], replace(f[2][1], decision_information=(i,))))
    for ids in [("unknown",), (event["event_id"], event["event_id"])]:
        i = replace(
            f[2][0].decision_information[0], decision_at=f[0][2], transfer_event_ids=ids
        )
        with pytest.raises(ValueError):
            run(f, declarations=(replace(f[2][0], decision_information=(i,)), f[2][1]))


@pytest.mark.parametrize("replacement,reentry", [(True, True), (True, False)])
def test_replacement_or_unreserved_current_state_never_changes_original_payoff_population(
    replacement, reentry
):
    f = fixture(replacement=replacement, reentry=reentry, partial=reentry)
    r = run(f).data
    assert len(r["all_original_persons"]) == 2
    assert (
        r["all_original_persons"][0]["original_record"]
        == f[1].data["all_original_persons"][0]["original_record"]
    )
    assert (
        r["all_original_persons"][0]["current_ticket"]
        == f[1].data["all_original_persons"][0]["current_ticket"]
    )


def test_empty_native_request_population_keeps_empty_sum_without_exposure_inference():
    f = fixture(empty=True)
    r = run(f).data
    assert r["all_original_persons"] == ()
    assert q(r["declared_closed_population_material_payoff"]) == 0
    assert (
        r["population_expectation_consumer_surplus_total_surplus_denial_misallocation"]
        is None
    )


def test_limits_owned_exports_and_no_producer_or_timing_backend_calls(monkeypatch):
    f = fixture()
    monkeypatch.setattr(
        settle,
        "reconcile_service_ticket_payments",
        lambda *a, **k: pytest.fail("producer replay"),
    )
    before = pickle.dumps((f[1], f[2]))
    with pytest.raises(RuntimeError):
        run(f, row_limit=1)
    r = run(f)
    out = module.person_material_payoff_record(r)
    out["definition"] = "changed"
    assert r.data["definition"] != "changed"
    assert pickle.dumps((f[1], f[2])) == before
    with pytest.raises(TypeError):
        r.data["all_original_persons"][0]["current_ticket"] = 99


def test_source_person_population_and_financial_closure_arithmetic_cannot_be_filtered():
    f = fixture()
    source = f[1]
    with pytest.raises(ValueError):
        run(
            f,
            payments=replace(
                source,
                data={
                    **source.data,
                    "all_original_persons": source.data["all_original_persons"][:-1],
                },
            ),
        )
    with pytest.raises(ValueError):
        run(
            f,
            payments=replace(
                source,
                data={
                    **source.data,
                    "declared_closed_population_total_ticket_payment": {
                        "numerator": "999",
                        "denominator": "1",
                    },
                },
            ),
        )
