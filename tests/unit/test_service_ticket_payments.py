"""A.10/A.5/B0/V9: NEW explicit transfer/payment-follow-up references."""

import importlib.util
import os
import pickle
import struct
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
from airline_market.sim.current_itinerary_service import (
    LegOperation,
    execute_current_itinerary_service,
)
from airline_market.sim.ordinary_cohort import (
    OrdinaryCohortMarketEngine,
    ordinary_person_panel,
)
from airline_market.sim.passenger_accounts import original_passenger_accounts
from airline_market.sim.service_observations import service_observation_journal
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from tests.unit import test_dated_demand as dated
from tests.unit import test_rationing_native as native

if os.environ.get("SERVICE_TICKET_PAYMENTS_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_service_ticket_payments", os.environ["SERVICE_TICKET_PAYMENTS_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.sim import service_ticket_payments as module


def fixture(*, replacement=False, reentry=True, partial=False):
    c, clock, dep = dated.inputs()
    scenario = compile_dated_demand(c, clock, dep)
    kwargs = {}
    if replacement:
        scenario = replace(
            scenario,
            physical_shocks=(
                B1DatedShock(
                    "new-payment-source-exit",
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
        c, 31, registry=StreamRegistry(72974, "unit/V9_service_ticket_payments")
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
    as_of = max(module.utc(s["at"]) for s in service.steps) + timedelta(hours=5)
    followup = tuple(
        module.TicketPaymentFollowup(
            p["person_key"],
            as_of,
            True,
            "NEW explicit unit financial-tail declaration; not service-implied closure",
        )
        for p in service.journal.context["persons"]
    )
    return e, service, as_of, followup


def run(f, transfers=(), **changes):
    _, service, as_of, followup = f
    args = dict(
        service=service,
        transfers=transfers,
        followup=followup,
        as_of=as_of,
        native_ticket_unit="explicit native fare units, common to additional transfers",
        definition="NEW supplied additional payment/refund incidence and complete original-person follow-up; no utility or automatic refund",
        row_limit=10000,
    )
    args.update(changes)
    return module.reconcile_service_ticket_payments(**args)


def booked(f):
    return next(
        p
        for p in f[1].journal.context["complete_ordinary_panel"]["persons"]
        if p["original_record"] is not None
    )


def event(
    f,
    *,
    amount=10.0,
    kind="additional_refund",
    name="new-transfer",
    ticket=None,
    offset=1,
):
    p = booked(f)
    account = p["booked_passenger_account"]
    row = next(
        t
        for t in account["tickets"]
        if t["record_id"]
        == (account["current_active_record"] if ticket is None else ticket)
    )
    when = module.utc(f[1].journal.context["booking_closed_at"]) + timedelta(
        hours=offset
    )
    return module.AdditionalTicketTransfer(
        name,
        p["potential_coordinate"],
        row["record_id"],
        row["carrier"],
        kind,
        amount,
        when,
        when,
        "NEW explicit ADDITIONAL post-close transfer, not a replay of native fare/refund",
    )


def q(value):
    return (
        None
        if value is None
        else Fraction(int(value["numerator"]), int(value["denominator"]))
    )


def test_native_plus_additional_exact_fares_without_native_or_service_mutation():
    f = fixture()
    e = f[0]
    before = pickle.dumps((e.world, e.trace, e.metrics, f[1], np.random.get_state()))
    base = q(booked(f)["booked_passenger_account"]["net_ticket_fare_paid"])
    transfers = (
        event(f, amount=0.1),
        event(f, amount=0.2, kind="additional_payment", name="surcharge", offset=2),
    )
    r = run(f, transfers).data
    assert q(r["observed_population_net_ticket_payment"]) == base + Fraction.from_float(
        0.2
    ) - Fraction.from_float(0.1)
    assert q(r["declared_closed_population_total_ticket_payment"]) == q(
        r["observed_population_net_ticket_payment"]
    )
    assert len(r["all_original_persons"]) == 2
    assert (
        pickle.dumps((e.world, e.trace, e.metrics, f[1], np.random.get_state()))
        == before
    )
    assert r["lifecycle_consumer_surplus_or_total_surplus"] is None


def test_completed_service_does_not_close_financial_tail_or_create_a_refund():
    f = fixture()
    followup = tuple(replace(row, no_further_ticket_transfers=False) for row in f[3])
    r = run(f, followup=followup).data
    assert not r["additional_transfers"]
    assert q(r["observed_population_net_ticket_payment"]) == q(
        booked(f)["booked_passenger_account"]["net_ticket_fare_paid"]
    )
    assert r["declared_closed_population_total_ticket_payment"] is None
    assert all(
        p["declared_closed_total_ticket_payment"] is None
        for p in r["all_original_persons"]
    )


def test_partial_service_neither_erases_prior_fares_nor_infers_a_full_refund():
    f = fixture(partial=True)
    assert tuple(o.outcome for o in f[1].journal.observations) == (
        "delivered",
        "not_delivered",
    )
    base = q(booked(f)["booked_passenger_account"]["net_ticket_fare_paid"])
    r = run(f, (event(f, amount=20.0),)).data
    assert q(r["declared_closed_population_total_ticket_payment"]) == base - 20
    assert r["all_original_persons"][0]["service_benefit"] is None
    assert r["carrier_profit_or_passenger_penalty_incidence"] is None


def test_interline_replacement_and_additional_settlement_on_cancelled_old_ticket_same_root():
    f = fixture(replacement=True)
    p = booked(f)
    account = p["booked_passenger_account"]
    assert len(account["tickets"]) == 2
    assert account["current_active_record"] != p["original_record"]
    transfers = (
        event(f, ticket=p["original_record"], amount=7.0),
        event(f, amount=3.0, name="current", offset=2),
    )
    r = run(f, transfers).data
    assert {t["carrier"] for t in r["additional_transfers"]} == {"A", "B"}
    assert len(r["all_original_persons"]) == 2
    assert (
        q(r["observed_population_net_ticket_payment"])
        == q(account["net_ticket_fare_paid"]) - 10
    )


def test_unreserved_original_and_outside_rows_have_explicit_open_followup():
    f = fixture(replacement=True, reentry=False)
    r = run(
        f, followup=tuple(replace(x, no_further_ticket_transfers=False) for x in f[3])
    ).data
    assert len(r["all_original_persons"]) == 2
    assert all(p["current_ticket"] is None for p in r["all_original_persons"])
    assert q(r["observed_population_net_ticket_payment"]) == 0
    assert r["declared_closed_population_total_ticket_payment"] is None


@pytest.mark.parametrize(
    "amount",
    [
        None,
        float("inf"),
        float("-inf"),
        struct.unpack(">d", bytes.fromhex("7ff8000000000099"))[0],
    ],
)
def test_unknown_nonfinite_amount_nulls_totals_without_filtering_known_rows(amount):
    f = fixture()
    r = run(f, (event(f, amount=amount),)).data
    assert r["observed_population_net_ticket_payment"] is None
    assert r["declared_closed_population_total_ticket_payment"] is None
    assert len(r["additional_transfers"]) == 1
    assert q(r["all_original_persons"][1]["observed_net_ticket_payment"]) == 0
    if amount is not None:
        assert (
            r["additional_transfers"][0]["amount"]["binary64_bits"]
            == struct.pack(">d", amount).hex()
        )


def test_signed_rebate_and_negative_refund_keep_sign_and_signed_zero():
    f = fixture()
    base = q(booked(f)["booked_passenger_account"]["net_ticket_fare_paid"])
    transfers = (
        event(f, amount=-1000.0, kind="additional_payment"),
        event(f, amount=-2.0, name="refund-charge", offset=2),
        event(f, amount=-0.0, name="negative-zero", offset=3),
    )
    r = run(f, transfers).data
    assert q(r["observed_population_net_ticket_payment"]) == base - 998
    assert (
        r["additional_transfers"][-1]["amount"]["binary64_bits"] == "8000000000000000"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("event_id", ""),
        ("source_ref", ""),
        ("kind", "refund"),
        ("carrier", "wrong"),
        ("ticket_record", True),
        ("amount", 1),
    ],
)
def test_invalid_or_ambiguous_transfer_refused(field, value):
    f = fixture()
    with pytest.raises((ValueError, TypeError)):
        run(f, (replace(event(f), **{field: value}),))


def test_outside_person_cannot_take_another_root_ticket_and_duplicate_events_refused():
    f = fixture()
    t = event(f)
    with pytest.raises(ValueError):
        run(f, (replace(t, person_key=f[3][1].person_key),))
    with pytest.raises(ValueError):
        run(f, (t, t))


@pytest.mark.parametrize("what", ["missing", "duplicate", "past", "nonboolean"])
def test_complete_explicit_common_date_followup_required(what):
    f = fixture()
    rows = f[3]
    if what == "missing":
        rows = rows[:-1]
    if what == "duplicate":
        rows = (rows[0], rows[0])
    if what == "past":
        rows = (
            replace(rows[0], observed_through=f[2] - timedelta(microseconds=1)),
            rows[1],
        )
    if what == "nonboolean":
        rows = (replace(rows[0], no_further_ticket_transfers=1), rows[1])
    with pytest.raises(ValueError):
        run(f, followup=rows)


def test_transfer_information_order_and_recording_after_coverage_refused():
    f = fixture()
    t = event(f)
    with pytest.raises(ValueError):
        run(f, (replace(t, recorded_at=f[2] + timedelta(microseconds=1)),))
    with pytest.raises(ValueError):
        run(f, (event(f, name="later", offset=2), t))
    with pytest.raises(ValueError):
        run(f, (replace(t, occurred_at=t.recorded_at + timedelta(microseconds=1)),))
    with pytest.raises(ValueError):
        run(
            f,
            (
                replace(
                    t,
                    occurred_at=module.utc(f[1].journal.context["booking_closed_at"])
                    - timedelta(microseconds=1),
                ),
            ),
        )


def test_row_limit_all_support_ownership_and_exports():
    f = fixture()
    before = pickle.dumps(f[1])
    with pytest.raises(RuntimeError):
        run(f, row_limit=1)
    assert pickle.dumps(f[1]) == before
    r = run(f)
    assert r.data["all_support_rows"] > 2
    out = module.service_ticket_payment_record(r)
    out["definition"] = "changed"
    assert r.data["definition"] != "changed"
    with pytest.raises(TypeError):
        r.data["all_original_persons"][0]["waiting_loss"] = 0


def test_service_receipt_binding_and_all_original_support_cannot_be_filtered():
    f = fixture()
    service = f[1]
    with pytest.raises(ValueError):
        run(f, service=replace(service, steps=service.steps[:-1]))
    bad = replace(service, context={**service.context, "source_context": {}})
    with pytest.raises(ValueError):
        run(f, service=bad)


def test_open_outside_payment_tail_keeps_population_total_unknown():
    f = fixture()
    rows = (f[3][0], replace(f[3][1], no_further_ticket_transfers=False))
    r = run(f, followup=rows).data
    assert (
        r["all_original_persons"][0]["declared_closed_total_ticket_payment"] is not None
    )
    assert r["all_original_persons"][1]["declared_closed_total_ticket_payment"] is None
    assert r["declared_closed_population_total_ticket_payment"] is None


def test_constructed_nonfinite_native_envelope_retained_without_zero_fill():
    # NEW serialized-input witness, not a claim the native unit produced infinity.
    from airline_market.core.hashing import jsonable

    f = fixture()
    ctx = jsonable(f[1].journal.context)
    account = next(
        p["booked_passenger_account"]
        for p in ctx["complete_ordinary_panel"]["persons"]
        if p["original_record"] is not None
    )
    account["tickets"][0]["fare"] = module._binary(float("inf"))
    account["gross_ticket_fares"] = None
    account["net_ticket_fare_paid"] = None
    journal = replace(f[1].journal, context=ctx)
    service = replace(
        f[1],
        journal=journal,
        context={**f[1].context, "source_context": module._encode(ctx)},
    )
    r = run(f, service=service).data
    assert r["observed_population_net_ticket_payment"] is None
    assert r["all_original_persons"][0]["native_recorded_net_ticket_payment"] is None
    raw = r["complete_service_input"]["journal_context"]["complete_ordinary_panel"][
        "persons"
    ][0]["booked_passenger_account"]["tickets"][0]["fare"]
    assert raw["binary64_bits"] == "7ff0000000000000"
