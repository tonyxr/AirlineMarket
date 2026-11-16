"""A.10/A.5/A.14/B0/V9/B4: NEW realized utility/source references only."""

import json
import pickle
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.dated_demand import compile_dated_demand
from airline_market.policy.dated_rescue import DatedRebookingPolicy
from airline_market.policy.dated_reserve import UtcBookingClock
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.sim import person_material_payoff as material
from airline_market.sim import person_utility_incidence as module
from airline_market.sim import service_ticket_payments as payment
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
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule


def new_source(
    *, replacement=False, reentry=True, partial=False, empty=False, open_payment=False
):
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
                    m.id: PeriodSchedule.parse({"constant": 2.0}) for m in spec.markets
                },
            ),
        )
    )
    start = datetime(2027, 7, 13, tzinfo=timezone.utc)
    clock = UtcBookingClock(tuple(start + timedelta(hours=12 * t) for t in range(4)))
    departures = tuple((leg.id, start + timedelta(hours=84)) for leg in c.spec.legs)
    scenario = compile_dated_demand(c, clock, departures)
    kwargs = {}
    if replacement:
        scenario = replace(
            scenario,
            physical_shocks=(
                B1DatedShock(
                    "new_utility_exit", "rival_exit", "B_OH0", 3, rival_carrier="B"
                ),
            ),
        )
        if reentry:
            kwargs["dated_rebooking"] = DatedRebookingPolicy(
                72, 20, 100, interline=True
            )
    engine = OrdinaryCohortMarketEngine(scenario, **kwargs)
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(108974, "unit/V9_person_utility_incidence")
    )
    j = next(
        i
        for i, p in enumerate(c.spec.products)
        if p.carrier == ("B" if replacement else "A") and len(p.legs) == 2
    )
    market_id = int(c.product_market[j])
    marks = (
        ()
        if empty
        else (
            ArrivalMark(0.1, market_id, 0, 0, 0),
            ArrivalMark(0.2, market_id, 0, 0, 0),
        )
    )
    g = tape.gumbel.copy()
    g[:] = -1e9
    g[:, :, -1] = 0
    g[0, 0, j] = 1e9
    g[0, 1, -1] = 2e9
    g.setflags(write=False)
    engine.reset(tape=replace(tape, arrival_marks=(marks, (), ()), gumbel=g))
    actions = {
        carrier.code: PriceAction({j: 200 if i == 0 else 300 for j in c.products_of[i]})
        for i, carrier in enumerate(c.spec.carriers)
    }
    while not engine.terminal():
        engine.step(actions)
    panel = ordinary_person_panel(
        engine.ordinary_periods(), original_passenger_accounts(engine)
    )
    journal = service_observation_journal(
        c,
        panel,
        clock,
        departures,
        definition="NEW full booking prefix, separately declared service program",
        observation_scope="declared-individual-leg-observations",
        context_row_limit=20000,
        observation_limit=100,
    )
    current = tuple(
        p for p in journal.context["persons"] if p["current_ticket"] is not None
    )
    second_legs = {p["required_legs"][-1] for p in current}
    ops = []
    for leg, scheduled in departures:
        departure = scheduled + timedelta(hours=4 if leg in second_legs else 0)
        ops.append(
            LegOperation(leg, "cancelled", 20, None, None, departure)
            if partial and leg in second_legs
            else LegOperation(
                leg, "operates", 20, departure, departure + timedelta(hours=1), None
            )
        )
    intents = tuple(
        (p["current_ticket"], leg, True) for p in current for leg in p["required_legs"]
    )
    orders = tuple(
        (leg, tuple(p["current_ticket"] for p in current if leg in p["required_legs"]))
        for leg, _ in departures
    )
    minima = tuple(
        (p["current_ticket"], *p["required_legs"], 3600_000000) for p in current
    )
    service = execute_current_itinerary_service(
        c,
        journal,
        ops,
        intents,
        orders,
        minima,
        definition="NEW explicit deterministic service witness, not empirical physical law",
        support_limit=20000,
    )
    as_of = max(payment.utc(s["at"]) for s in service.steps) + timedelta(hours=5)
    transfers = ()
    if current:
        p = current[0]
        account = next(
            row
            for row in panel["persons"]
            if row["potential_coordinate"] == p["person_key"]
        )["booked_passenger_account"]
        ticket = next(
            row for row in account["tickets"] if row["record_id"] == p["current_ticket"]
        )
        at = payment.utc(journal.context["booking_closed_at"]) + timedelta(hours=1)
        transfers = (
            payment.AdditionalTicketTransfer(
                "new_utility_transfer",
                p["person_key"],
                ticket["record_id"],
                ticket["carrier"],
                "additional_payment",
                0.1,
                at,
                at,
                "NEW exact binary64 transfer witness",
            ),
        )
    followup = tuple(
        payment.TicketPaymentFollowup(
            p["person_key"],
            as_of,
            not open_payment,
            "NEW independent financial follow-up declaration",
        )
        for p in journal.context["persons"]
    )
    payments = payment.reconcile_service_ticket_payments(
        service,
        transfers,
        followup,
        as_of=as_of,
        native_ticket_unit="NEW material unit 108974",
        definition="NEW full service/payment inputs",
        row_limit=100000,
    )
    declarations = []
    for p in payments.data["all_original_persons"]:
        key = p["person_key"]
        info = material.DeclaredDecisionInformation(
            key,
            "new_declared_opportunity",
            clock.at(key[0]),
            (),
            (),
            "NEW opportunity declaration, no future disclosures",
            "new unit information source",
        )
        payoff = ContingentPayoff(
            Fraction(200 if p["original_record"] is not None else 30),
            payment._rational(p["declared_closed_total_ticket_payment"]),
            Fraction(5 if p["original_record"] is not None else 2),
            Fraction(-1, 3),
        )
        declarations.append(
            material.OriginalPersonMaterialPayoff(
                key,
                payoff,
                tuple((k, "NEW explicit material " + k) for k in material.COMPONENTS),
                as_of,
                True,
                "NEW material closure",
                (info,),
            )
        )
    report = material.reconcile_person_material_payoff(
        payments,
        declarations,
        payoff_unit=payments.data["native_ticket_unit"],
        definition="NEW complete realized material source",
        scope=material.SCOPE,
        shock_rule=material.SHOCK_RULE,
        row_limit=100000,
    )
    return engine, service, payments, report


@pytest.fixture(scope="module")
def sources():
    return {
        name: new_source(**kwargs)
        for name, kwargs in {
            "base": {},
            "partial": {"partial": True},
            "replacement": {"replacement": True},
            "unreserved": {"replacement": True, "reentry": False},
            "empty": {"empty": True},
            "open": {"open_payment": True},
        }.items()
    }


def declared(source):
    report = source[3]
    at = payment.utc(report.data["complete_payment_input"]["as_of"])
    outside = module.DeclaredOutsideReference(
        "new matched-path outside reference",
        ContingentPayoff(Fraction(17), Fraction(2), Fraction(1, 2), Fraction(-3, 2)),
        tuple((k, "NEW explicit outside " + k) for k in material.COMPONENTS),
        Fraction(-2, 7),
        "NEW outside additional utility source",
        True,
        True,
        True,
        "NEW outside tail closure",
        "same realized path reference; not an intervention law",
    )
    return tuple(
        module.PersonUtilityIncidence(
            p["person_key"],
            Fraction(3, 7),
            "NEW declared realized additional utility",
            True,
            "NEW utility tail closure",
            outside,
            at,
            "B explicitly excludes all additional E; substantive declaration not authentication",
            "NEW candidate E incidence/origin, no native mark decoding",
        )
        for p in report.data["all_original_persons"]
    )


def run(source, declarations=None, **changes):
    args = {
        "material": source[3],
        "declarations": declared(source) if declarations is None else declarations,
        "alpha": Fraction(1, 40),
        "material_unit": "NEW material unit 108974",
        "utility_unit": "NEW utility unit 108974",
        "reference_scope": module.REFERENCE_SCOPE,
        "definition": "NEW realized candidate utility reconciliation, not native CS",
        "row_limit": 10_000_000,
    }
    args.update(changes)
    return module.reconcile_person_utility(**args)


def q(v):
    return payment._rational(v)


def test_full_native_source_exact_accounts_and_purity(sources, monkeypatch):
    source = sources["base"]
    before = pickle.dumps((source, np.random.get_state()))

    def forbidden(*args, **kwargs):
        raise AssertionError("upstream producer must not execute")

    monkeypatch.setattr(material, "reconcile_person_material_payoff", forbidden)
    monkeypatch.setattr(payment, "reconcile_service_ticket_payments", forbidden)
    out = run(source).data
    assert len(out["all_original_persons"]) == 2
    for p, original in zip(
        out["all_original_persons"], source[3].data["all_original_persons"]
    ):
        z = q(original["declared_closed_material_payoff"])
        assert q(p["decision_noise_excluded"]) == z / 40
        assert q(p["outside_decision_noise_excluded"]) == Fraction(16, 40)
        assert q(p["normalized_experienced_candidate"]) == (z - 16) / 40 + Fraction(
            5, 7
        )
    assert pickle.dumps((source, np.random.get_state())) == before
    assert out["complete_material_input"] == material.person_material_payoff_record(
        source[3]
    )
    assert out["native_lifecycle_CS_total_denial_misallocation"] is None
    assert not out["decision_information_or_native_actor_observation_extended"]


@pytest.mark.parametrize("name", ["partial", "replacement", "unreserved", "empty"])
def test_new_source_support_and_no_service_implied_utility(sources, name):
    source = sources[name]
    out = run(source).data
    assert len(out["all_original_persons"]) == (0 if name == "empty" else 2)
    for p in out["all_original_persons"]:
        assert p["declaration"]["additional_utility"] == payment._number(Fraction(3, 7))
    if name == "unreserved":
        assert any(
            p["current_ticket"] is None and p["original_record"] is not None
            for p in out["all_original_persons"]
        )
    if name == "replacement":
        assert any(
            p["current_ticket"] != p["original_record"]
            and p["original_record"] is not None
            for p in out["all_original_persons"]
        )
    if name == "empty":
        assert all(q(v) == 0 for v in out["declared_population_sums"].values())


@pytest.mark.parametrize(
    "field",
    [
        "additional_utility",
        "outside_unknown",
        "utility_tail",
        "outside_financial_tail",
        "outside_material_tail",
        "outside_utility_tail",
    ],
)
def test_unknowns_and_independent_open_tails(sources, field):
    source = sources["base"]
    ds = list(declared(source))
    if field == "additional_utility":
        ds[1] = replace(ds[1], additional_utility=None)
    elif field == "utility_tail":
        ds[1] = replace(ds[1], no_further_utility_components=False)
    else:
        attr = {
            "outside_unknown": "additional_utility",
            "outside_financial_tail": "no_further_financial_components",
            "outside_material_tail": "no_further_material_components",
            "outside_utility_tail": "no_further_utility_components",
        }[field]
        ds[1] = replace(
            ds[1],
            outside=replace(
                ds[1].outside, **{attr: None if field == "outside_unknown" else False}
            ),
        )
    out = run(source, ds).data
    assert out["all_original_persons"][1]["normalized_experienced_candidate"] is None
    assert out["declared_population_sums"]["normalized_experienced_candidate"] is None
    assert (
        out["all_original_persons"][0]["normalized_experienced_candidate"] is not None
    )
    assert out["declared_population_sums"]["decision_noise_excluded"] is not None


def test_open_financial_tail_does_not_use_known_prefix(sources):
    out = run(sources["open"]).data
    assert out["declared_population_sums"]["decision_noise_excluded"] is None
    assert all(p["experienced_candidate"] is None for p in out["all_original_persons"])


def test_common_terminal_translation_exact_and_signed_tiny(sources):
    source = sources["base"]
    ds = declared(source)
    c = Fraction(-(10**600), 7) + Fraction(1, 10**1300)
    shifted = tuple(
        replace(
            d,
            additional_utility=d.additional_utility + c,
            outside=replace(
                d.outside, additional_utility=d.outside.additional_utility + c
            ),
        )
        for d in ds
    )
    base, changed = run(source, ds).data, run(source, shifted).data
    for p, s in zip(base["all_original_persons"], changed["all_original_persons"]):
        assert q(s["experienced_candidate"]) - q(p["experienced_candidate"]) == c
        assert (
            s["normalized_experienced_candidate"]
            == p["normalized_experienced_candidate"]
        )
    assert (
        changed["declared_population_sums"]["normalized_experienced_candidate"]
        == base["declared_population_sums"]["normalized_experienced_candidate"]
    )


def test_explicit_nonzero_outside_matching_and_unknown_not_cancelled(sources):
    source = sources["base"]
    ds = list(declared(source))
    p = source[3].data["all_original_persons"][1]
    payoff = ContingentPayoff(
        *(q(p["four_explicit_components"][k]) for k in material.COMPONENTS)
    )
    ds[1] = replace(
        ds[1],
        outside=replace(
            ds[1].outside, payoff=payoff, additional_utility=ds[1].additional_utility
        ),
    )
    out = run(source, ds).data["all_original_persons"][1]
    assert q(out["experienced_candidate"]) != 0
    assert q(out["normalized_experienced_candidate"]) == 0
    ds[1] = replace(
        ds[1],
        additional_utility=None,
        outside=replace(ds[1].outside, additional_utility=None),
    )
    assert (
        run(source, ds).data["all_original_persons"][1][
            "normalized_experienced_candidate"
        ]
        is None
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing_outside",
        "duplicate",
        "wrong_key",
        "wrong_time",
        "missing_nonoverlap",
        "missing_incidence",
        "missing_extra_source",
        "missing_outside_source",
        "bool_tail",
        "float_extra",
        "missing_component_source",
    ],
)
def test_complete_declaration_refusals(sources, change):
    source = sources["base"]
    ds = list(declared(source))
    if change == "missing_outside":
        ds = ds[:1]
    elif change == "duplicate":
        ds = [ds[0], ds[0]]
    elif change == "wrong_key":
        ds[1] = replace(ds[1], person_key=(0, 99, 0))
    elif change == "wrong_time":
        ds[1] = replace(
            ds[1], recorded_at=ds[1].recorded_at + timedelta(microseconds=1)
        )
    elif change == "missing_nonoverlap":
        ds[1] = replace(ds[1], material_additional_nonoverlap="")
    elif change == "missing_incidence":
        ds[1] = replace(ds[1], utility_origin_and_incidence="")
    elif change == "missing_extra_source":
        ds[1] = replace(ds[1], additional_utility_source="")
    elif change == "missing_outside_source":
        ds[1] = replace(
            ds[1], outside=replace(ds[1].outside, additional_utility_source="")
        )
    elif change == "bool_tail":
        ds[1] = replace(ds[1], no_further_utility_components=1)
    elif change == "float_extra":
        ds[1] = replace(ds[1], additional_utility=0.1)
    elif change == "missing_component_source":
        ds[1] = replace(
            ds[1],
            outside=replace(
                ds[1].outside, component_sources=ds[1].outside.component_sources[:3]
            ),
        )
    with pytest.raises((TypeError, ValueError)):
        run(source, ds)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"alpha": Fraction(0)},
        {"alpha": 0.1},
        {"material_unit": "dollars"},
        {"utility_unit": "NEW material unit 108974"},
        {"reference_scope": "causal-outside-intervention"},
        {"row_limit": True},
    ],
)
def test_conversion_scope_refusals(sources, kwargs):
    with pytest.raises((TypeError, ValueError)):
        run(sources["base"], **kwargs)


def test_structural_limit_before_utility_forwards_and_owned_output(
    sources, monkeypatch
):
    source = sources["base"]
    result = run(source)
    bound = result.data["all_support_structural_bound"]
    assert module._count(result.data) <= bound
    exported = module.person_utility_record(result)
    exported["complete_material_input"].clear()
    assert result.data["complete_material_input"]
    json.dumps(module.person_utility_record(result), allow_nan=False)
    before = pickle.dumps(source)

    def forbidden(*args):
        raise AssertionError("utility forward after bound failure")

    monkeypatch.setattr(module, "_scale", forbidden)
    with pytest.raises(RuntimeError, match="structural limit"):
        run(source, row_limit=1)
    assert pickle.dumps(source) == before


@pytest.mark.parametrize("change", ["sum", "payment", "material", "root", "omitted"])
def test_saved_material_consistency_refusals(sources, change):
    source = sources["base"]
    value = material.person_material_payoff_record(source[3])
    if change == "sum":
        value["declared_closed_population_material_payoff"] = payment._number(
            Fraction(999)
        )
    elif change == "payment":
        value["all_original_persons"][0]["four_explicit_components"][
            "net_ticket_payment"
        ] = payment._number(Fraction(0))
    elif change == "material":
        value["all_original_persons"][0]["declared_closed_material_payoff"] = (
            payment._number(Fraction(0))
        )
    elif change == "root":
        value["all_original_persons"][0]["original_record"] = 999
    elif change == "omitted":
        value["all_original_persons"] = value["all_original_persons"][:1]
    with pytest.raises(ValueError):
        run(source, material=material.PersonMaterialPayoffReport(value))


def test_unexpected_arithmetic_failure_leaves_complete_inputs(sources, monkeypatch):
    source = sources["base"]
    ds = declared(source)
    before = pickle.dumps((source, ds))

    def fail(*args):
        raise RuntimeError("NEW injected utility arithmetic failure")

    monkeypatch.setattr(module, "_plus", fail)
    with pytest.raises(RuntimeError, match="injected"):
        run(source, ds)
    assert pickle.dumps((source, ds)) == before
