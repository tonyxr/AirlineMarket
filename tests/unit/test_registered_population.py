"""A.10/A.5/B0/V9/B4: NEW September7 / 116974 registration units."""

import json
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
from airline_market.policy.regulator import CapPolicy
from airline_market.sim import conditional_population as population
from airline_market.sim import current_itinerary_service as service
from airline_market.sim import person_material_payoff as material
from airline_market.sim import registered_population as subject
from airline_market.sim import service_ticket_payments as payments
from airline_market.sim.ordinary_cohort import (
    OrdinaryCohortMarketEngine,
    ordinary_person_panel,
)
from airline_market.sim.passenger_accounts import original_passenger_accounts
from airline_market.sim.payoff_logit import ContingentPayoff
from airline_market.sim.service_observations import service_observation_journal
from airline_market.sim.shared_mark_attempt import _encoded
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.io import dump_spec, load_spec
from airline_market.spec.schedule import PeriodSchedule

LIMIT = 2_000_000_000
UNIT = "declared-native-fare-unit116974"


def token(key):
    return -1 - key[1]


def q(value):
    return (
        None if value is None else F(int(value["numerator"]), int(value["denominator"]))
    )


def declare():
    base = screen_spec(
        {
            "T": 3,
            "actions": 2,
            "n_spokes": 1,
            "capacity": 20,
            "capacity_support": [3, 4],
        }
    )
    spec = replace(
        base,
        demand=replace(
            base.demand,
            mode="poisson",
            arrival_probability=None,
            market_weights=None,
            intensity={
                m.id: PeriodSchedule.parse({"constant": 6.0}) for m in base.markets
            },
        ),
    )
    c = compile_market(load_spec(json.loads(dump_spec(spec))))
    start = datetime(2027, 9, 7, tzinfo=timezone.utc)
    clock = UtcBookingClock(tuple(start + timedelta(hours=12 * t) for t in range(4)))
    departures = tuple(
        (leg.id, start + timedelta(hours=96 + 4 * i))
        for i, leg in enumerate(c.spec.legs)
    )
    scenario = compile_dated_demand(c, clock, departures)
    product = c.product_index["A_OH0"]
    market = int(c.product_market[product])
    keys = tuple((0, i, market) for i in range(3))
    current = keys[:2]
    controls = {
        "ordinary_priority": None,
        "trigger_rule": None,
        "mechanical_benchmark": None,
        "dated_rebooking": None,
        "dated_menu": None,
        "dated_reserve": None,
        "cap_policy": {"windows": [], "declared_emergencies": []},
        "check_invariants": True,
    }
    actions = {
        carrier.code: {
            str(int(p)): 280.0 if i == 0 else 385.0 for p in c.products_of[i]
        }
        for i, carrier in enumerate(c.spec.carriers)
    }
    recipe = {
        "rule": subject.RULE,
        "complete_native_spec": json.loads(dump_spec(c.spec)),
        "spec_hash": c.spec_hash,
        "complete_native_spec_binary64": _encoded(c.spec),
        "full_dated_scenario": _encoded(scenario),
        "compiled_clock": [t.isoformat() for t in clock.boundaries],
        "all_leg_departures": [(leg, at.isoformat()) for leg, at in departures],
        "engine_class": "airline_market.sim.ordinary_cohort.OrdinaryCohortMarketEngine",
        "controllers": None,
        "explicit_constructor_controls": controls,
        "all_period_postings": [actions for _ in range(3)],
        "native_steps": 3,
        "tape": {
            "tape_seed_field": 31,
            "registry_root": 116974,
            "namespace": "unit/V9_registered_population116974",
            "generate_calls": 1,
            "minimum_generated_gumbel_slots": 3,
            "maximum_generated_structural_rows": 100000,
            "gumbel_fill": -1e8,
            "outside_default": 0.0,
            "target_product": "A_OH0",
            "purchase_slots": [0, 1],
            "purchase_mark": 1e8,
            "outside_slots": [2],
            "outside_mark": 2e8,
            "period_arrival_marks": [
                [
                    {
                        "time": (i + 1) / 5,
                        "market": market,
                        "segment": 0,
                        "thinning": 0.0,
                        "class_uniform": 0.45,
                    }
                    for i in range(3)
                ],
                [],
                [],
            ],
            "all_other_generated_fields": "Retain unchanged, INCLUDING arrival_counts; this artificial unit tape does not authenticate a coherent native Poisson probability law",
        },
        "expected_people": [
            {
                "key": key,
                "expected_current_product": "A_OH0" if i < 2 else None,
                "expected_original_booked": i < 2,
            }
            for i, key in enumerate(keys)
        ],
        "account_panel_joins": 1,
        "context_limit": 200000,
        "observation_limit": 1000,
        "source_definition": "NEW complete fixed original population116974",
        "source_observation_scope": "declared-individual-leg-observations",
    }
    as_of = max(at for _, at in departures) + timedelta(hours=4)
    when = as_of - timedelta(hours=1)
    physical = []
    for i, name in enumerate(
        ("first-priority", "second-priority", "cancelled", "two-seats")
    ):
        ops = tuple(
            service.LegOperation(
                leg,
                "cancelled" if i == 2 and leg == "A_OH0" else "operates",
                0
                if i == 2 and leg == "A_OH0"
                else (1 if i < 2 else 2)
                if leg == "A_OH0"
                else 20,
                None if i == 2 and leg == "A_OH0" else at,
                None if i == 2 and leg == "A_OH0" else at + timedelta(hours=3),
                at - timedelta(hours=2) if i == 2 and leg == "A_OH0" else None,
            )
            for leg, at in departures
        )
        intents = tuple((token(key), "A_OH0", True) for key in current)
        orders = tuple(
            (
                leg,
                tuple(
                    token(key)
                    for key in (tuple(reversed(current)) if i == 1 else current)
                )
                if leg == "A_OH0"
                else (),
            )
            for leg, _ in departures
        )
        physical.append(
            population.PhysicalServiceProgramme(
                name,
                ops,
                intents,
                orders,
                (),
                "Complete registered leg programme116974/" + name,
                "Explicit hypothetical physical law116974",
            )
        )
    cells = []
    for p in physical:
        for financial in ("closed", "open", "served-rebate"):
            people, all_transfers = [], []
            for i, key in enumerate(keys):
                terms = []
                for outcome in (False, True) if i < 2 else (None,):
                    transfers = []
                    label = p.name + "/" + financial + "/" + str(i) + "/" + str(outcome)
                    if outcome is False:
                        transfers.append(
                            payments.AdditionalTicketTransfer(
                                "refund116974/" + label,
                                key,
                                token(key),
                                "A",
                                "additional_refund",
                                280.0,
                                when,
                                when,
                                "Registered additional full-current-fare refund if not complete",
                            )
                        )
                    if outcome is True and financial == "served-rebate":
                        transfers.append(
                            payments.AdditionalTicketTransfer(
                                "rebate116974/" + label,
                                key,
                                token(key),
                                "A",
                                "additional_payment",
                                -13.0,
                                when,
                                when,
                                "Explicit negative payment ONLY if service completes; not carrier-penalty incidence",
                            )
                        )
                    closed = financial != "open"
                    payment = (
                        None
                        if not closed
                        else F(267 if financial == "served-rebate" else 280)
                        if outcome is True
                        else F()
                    )
                    payoff = ContingentPayoff(
                        F(620)
                        if outcome is True
                        else F()
                        if outcome is False
                        else F(7),
                        payment,
                        F(8) if i < 2 else F(3),
                        F(23) if outcome is False else F(),
                    )
                    terms.append(
                        population.ServiceOutcomeTerms(
                            outcome,
                            tuple(transfers),
                            closed,
                            "Registered financial follow-up116974/" + financial,
                            payoff,
                            tuple(
                                (n, "Explicit registered component116974/" + n)
                                for n in material.COMPONENTS
                            ),
                            True,
                            "Explicit independent material closure116974",
                        )
                    )
                    all_transfers.extend(t.event_id for t in transfers)
                people.append(population.PersonServiceTerms(key, tuple(terms)))
            cells.append(
                population.PopulationPayoffCell(
                    p.name,
                    financial,
                    tuple(people),
                    tuple(all_transfers),
                    "Complete registered financial-material cell116974/"
                    + p.name
                    + "/"
                    + financial,
                    "Declared joint follow-through116974",
                )
            )
    info = tuple(
        population.OriginalPersonInformation(
            key,
            (
                material.DeclaredDecisionInformation(
                    key,
                    "earlier116974/" + str(i),
                    clock.at(0),
                    (),
                    (),
                    "Explicit empty earlier service-transfer information, not complete knowledge",
                    "Pre-source declaration116974",
                ),
            ),
        )
        for i, key in enumerate(keys)
    )
    programme = population.ConditionalPopulationProgramme(
        tuple(physical),
        ("closed", "open", "served-rebate"),
        tuple(cells),
        info,
        as_of,
        UNIT,
        population.SCOPE,
        "Conditional on the complete actual source associated with registered-native-prefix116974",
        "Complete prospectively registered programme116974",
        "Declared conditional witness, not native probability truth",
    )
    forecast = (
        F(1, 4),
        F(),
        F(),
        F(1, 4),
        F(),
        F(),
        F(1, 8),
        F(),
        F(),
        F(1, 8),
        F(),
        F(1, 4),
    )
    reference = (
        F(1, 8),
        F(),
        F(),
        F(3, 8),
        F(),
        F(),
        F(1, 4),
        F(),
        F(),
        F(1, 4),
        F(),
        F(),
    )
    eps = F(1, 10**1500)
    tiny = tuple(eps if i == 7 else (1 - eps) * w for i, w in enumerate(forecast))
    laws = tuple(
        population.ConditionalPopulationLaw(
            name,
            weights,
            "Declared law conditional on registered-native-prefix116974",
            "Explicit pre-source law116974",
        )
        for name, weights in (
            ("forecast", forecast),
            ("reference", reference),
            ("tiny-open", tiny),
        )
    )
    plan = subject.NativePopulationSourcePlan(
        dump_spec(c.spec),
        clock.boundaries,
        departures,
        tuple(
            tuple(
                280.0 if int(c.product_carrier[j]) == 0 else 385.0
                for j in range(len(c.spec.products))
            )
            for _ in range(3)
        ),
        subject.PrimitiveTapePlan(
            31,
            116974,
            "unit/V9_registered_population",
            tuple(
                tuple(
                    (
                        float(m["time"]),
                        m["market"],
                        m["segment"],
                        float(m["thinning"]),
                        float(m["class_uniform"]),
                    )
                    for m in period
                )
                for period in recipe["tape"]["period_arrival_marks"]
            ),
            -1e8,
            0.0,
            tuple((0, i, product, 1e8) for i in range(2))
            + ((0, 2, len(c.spec.products), 2e8),),
            3,
            100000,
        ),
        tuple(
            subject.OriginalPersonPlan(
                key, i < 2, "A_OH0" if i < 2 else None, token(key) if i < 2 else None
            )
            for i, key in enumerate(keys)
        ),
        recipe["source_definition"],
        recipe["source_observation_scope"],
        200000,
        1000,
    )
    attachments = []
    owners = {token(key): key for key in current}
    for i, p in enumerate(programme.physical):
        for j, row in enumerate(p.boarding_intents):
            attachments.append(
                subject.TicketAttachment(
                    ("physical", i, "boarding_intents", j, 0), owners[row[0]]
                )
            )
        for j, (_, order) in enumerate(p.boarding_orders):
            for k, rid in enumerate(order):
                attachments.append(
                    subject.TicketAttachment(
                        ("physical", i, "boarding_orders", j, 1, k), owners[rid]
                    )
                )
    for i, cell in enumerate(programme.cells):
        for j, person in enumerate(cell.persons):
            for k, term in enumerate(person.outcomes):
                for z, t in enumerate(term.transfers):
                    attachments.append(
                        subject.TicketAttachment(
                            (
                                "cells",
                                i,
                                "persons",
                                j,
                                "outcomes",
                                k,
                                "transfers",
                                z,
                                "ticket_record",
                            ),
                            person.person_key,
                        )
                    )
    registration = subject.register_population(
        plan, programme, laws, attachments, row_limit=LIMIT
    )
    return c, scenario, registration


def create_source(c, scenario, r):
    # Registration exists BEFORE this new source/tape.
    t = r.source.tape
    generated = ExogenousTape.generate(
        c, t.seed, registry=StreamRegistry(t.registry_root, t.namespace)
    )
    assert generated.gumbel.shape[1] >= t.minimum_slots
    g = generated.gumbel.copy()
    g[:] = t.gumbel_fill
    g[:, :, -1] = t.outside_default
    for period, slot, product, value in t.gumbel_overrides:
        g[period, slot, product] = value
    g.setflags(write=False)
    tape = replace(
        generated,
        arrival_marks=tuple(
            tuple(ArrivalMark(*m) for m in period) for period in t.arrival_marks
        ),
        gumbel=g,
    )
    e = OrdinaryCohortMarketEngine(
        scenario,
        controllers=None,
        ordinary_priority=None,
        trigger_rule=None,
        mechanical_benchmark=None,
        dated_rebooking=None,
        dated_menu=None,
        dated_reserve=None,
        cap_policy=CapPolicy(windows=(), declared_emergencies=()),
        check_invariants=True,
    )
    e.reset(tape=tape)
    for posting in r.source.postings:
        e.step(
            {
                carrier.code: PriceAction(
                    {int(j): posting[j] for j in c.products_of[i]}
                )
                for i, carrier in enumerate(c.spec.carriers)
            }
        )
    accounts = original_passenger_accounts(e)
    panel = ordinary_person_panel(e.ordinary_periods(), accounts)
    journal = service_observation_journal(
        c,
        panel,
        UtcBookingClock(r.source.clock),
        r.source.departures,
        definition=r.source.source_definition,
        observation_scope=r.source.observation_scope,
        context_row_limit=r.source.context_row_limit,
        observation_limit=r.source.observation_limit,
    )
    ev = {
        "engine_class": type(e).__module__ + "." + type(e).__name__,
        "controls": _encoded(
            {
                "ordinary_priority": e.ordinary_priority,
                "trigger_rule": e.trigger.rule if e.trigger else None,
                "mechanical_benchmark": e.trigger.benchmark if e.trigger else None,
                "dated_rebooking": e.dated_rebooking,
                "dated_menu": e.dated_menu,
                "dated_reserve": e.dated_reserve,
                "cap_policy": e.cap_policy,
                "check_invariants": e.check_invariants,
            }
        ),
        "scenario": _encoded(e.demand_scenario),
        "controllers": _encoded(e.assignments),
        "native_period_after": e.world.t,
        "ordinary_periods": _encoded(e.ordinary_periods()),
        "trace": _encoded(e.trace()),
        "metrics": _encoded(e.metrics()),
    }
    supplied = subject.SuppliedPopulationSource(
        journal, ev, _encoded(generated), _encoded(tape)
    )
    return r, supplied, e


@pytest.fixture(scope="module")
def source():
    return create_source(*declare())


@pytest.fixture(scope="module")
def evaluated(source):
    r, s, e = source
    before = pickle.dumps(e.snapshot(), protocol=5)
    random_before = pickle.dumps(np.random.get_state(), protocol=5)
    out = subject.evaluate_registered_population(r, s, row_limit=LIMIT)
    assert pickle.dumps(e.snapshot(), protocol=5) == before
    assert pickle.dumps(np.random.get_state(), protocol=5) == random_before
    return subject.registered_population_record(out)


def test_complete_registered_joint_and_exact_means(evaluated):
    d = evaluated
    assert len(d["registration"]["attachments"]) == 48
    assert len(d["conditional_population"]["analyst"]["cells"]) == 12
    laws = d["conditional_population"]["analyst"]["law_results"]
    assert q(laws[0]["population_material"]["mean"]) == F(1609, 4)
    assert q(laws[0]["population_payment"]["mean"]) == F(687, 2)
    assert q(laws[1]["population_material"]["mean"]) == 305
    assert laws[2]["population_material"]["mean"] is None
    assert laws[2]["population_payment"]["mean"] is None
    assert d["conditional_population_evaluations"] == 1
    assert not d["native_source_or_rng_or_actor_executed"]
    assert d["all_support_bound"] > d["actual_parent_bound"]


def test_only_registered_ticket_fields_attach(evaluated):
    r = evaluated["registration"]["programme"]
    a = evaluated["attached_programme"]
    paths = {tuple(v["path"]) for v in evaluated["registration"]["attachments"]}

    def leaves(value, path=()):
        if isinstance(value, dict):
            return [(p, v) for k, x in value.items() for p, v in leaves(x, (*path, k))]
        if isinstance(value, (tuple, list)):
            return [
                (p, v) for i, x in enumerate(value) for p, v in leaves(x, (*path, i))
            ]
        return [(path, value)]

    old = dict(leaves(r))
    new = dict(leaves(a))
    assert old.keys() == new.keys()
    assert {p for p in old if old[p] != new[p]} == paths
    assert all(type(old[p]) is int and old[p] < 0 and new[p] >= 0 for p in paths)


@pytest.mark.parametrize(
    "fault",
    [
        "rule",
        "digest",
        "missing_attachment",
        "wrong_owner",
        "extra_attachment",
        "token_collision",
        "outside_token",
        "missing_spec_default",
        "missing_posting",
        "nonunit_law",
        "missing_zero_cell",
        "opaque",
    ],
)
def test_registration_refusals_before_future_forward(source, monkeypatch, fault):
    r, s, _e = source

    def refused(*args, **kwargs):
        raise AssertionError("future forward called")

    monkeypatch.setattr(population, "evaluate_conditional_population", refused)
    if fault == "rule":
        bad = replace(r, rule="unrecognised")
    elif fault == "digest":
        bad = replace(r, digest="bad")
    elif fault == "missing_attachment":
        bad = replace(r, attachments=r.attachments[:-1])
    elif fault == "wrong_owner":
        bad = replace(
            r,
            attachments=(
                replace(r.attachments[0], person_key=r.source.people[1].key),
                *r.attachments[1:],
            ),
        )
    elif fault == "extra_attachment":
        bad = replace(r, attachments=(*r.attachments, r.attachments[0]))
    elif fault == "token_collision":
        bad = replace(
            r,
            source=replace(
                r.source,
                people=(
                    r.source.people[0],
                    replace(r.source.people[1], ticket_token=-1),
                    r.source.people[2],
                ),
            ),
        )
    elif fault == "outside_token":
        bad = replace(
            r,
            source=replace(
                r.source,
                people=(
                    *r.source.people[:2],
                    replace(r.source.people[2], ticket_token=-3),
                ),
            ),
        )
    elif fault == "missing_spec_default":
        raw = json.loads(r.source.complete_spec_json)
        raw.pop("metadata", None)
        if raw == json.loads(r.source.complete_spec_json):
            raw.pop("controllers")
        bad = replace(r, source=replace(r.source, complete_spec_json=json.dumps(raw)))
    elif fault == "missing_posting":
        bad = replace(r, source=replace(r.source, postings=r.source.postings[:-1]))
    elif fault == "nonunit_law":
        bad = replace(r, laws=(replace(r.laws[0], weights=(F(),) * 12),))
    elif fault == "missing_zero_cell":
        bad = replace(r, programme=replace(r.programme, cells=r.programme.cells[:-1]))
    else:
        bad = replace(r, programme=replace(r.programme, physical=(object(),)))
    with pytest.raises((ValueError, TypeError, AttributeError, KeyError)):
        subject.evaluate_registered_population(bad, s, row_limit=LIMIT)


@pytest.mark.parametrize(
    "fault",
    [
        "controls",
        "future_scenario",
        "controllers",
        "period",
        "empty_period_omitted",
        "posting",
        "ordinary_person",
        "ordinary_class",
        "ordinary_rank",
        "auxiliary_tape",
        "primitive_tape",
        "missing_tape_field",
        "generated_limit",
        "current_owner",
        "source_model",
        "source_definition",
    ],
)
def test_source_refusals_before_future_forward(source, monkeypatch, fault):
    r, s, _e = source
    ev = json.loads(json.dumps(_encoded(s.evidence)))
    generated = json.loads(json.dumps(_encoded(s.generated_tape)))
    native = json.loads(json.dumps(_encoded(s.native_tape)))
    ctx = json.loads(json.dumps(_encoded(s.journal.context)))
    if fault == "controls":
        ev["controls"]["cap_policy"] = None
    elif fault == "future_scenario":
        ev["scenario"]["physical_shocks"] = [{"unregistered": True}]
    elif fault == "controllers":
        ev["controllers"] = {}
    elif fault == "period":
        ev["native_period_after"] = 2
    elif fault == "empty_period_omitted":
        ev["ordinary_periods"].pop()
    elif fault == "posting":
        ev["trace"][0]["prices"][0] = _encoded(281.0)
    elif fault == "ordinary_person":
        ev["ordinary_periods"][0]["rows"].pop()
    elif fault == "ordinary_class":
        ev["ordinary_periods"][0]["rows"][0]["taste_class"] = 999
    elif fault == "ordinary_rank":
        ev["ordinary_periods"][0]["rows"][0]["native_rank"] = 999
    elif fault == "auxiliary_tape":
        native["initial_exo_uniform"] = _encoded(0.0)
    elif fault == "primitive_tape":
        native["gumbel"][0][0][0] = _encoded(0.0)
    elif fault == "missing_tape_field":
        generated.pop("arrival_counts")
    elif fault == "generated_limit":
        r = replace(
            r,
            source=replace(
                r.source, tape=replace(r.source.tape, generated_row_limit=1)
            ),
        )
        r = replace(r, digest=subject._digest(r))
    elif fault == "current_owner":
        ctx["persons"][0]["current_product"] = "B_OH0"
    elif fault == "source_model":
        ctx["spec_hash"] = "different"
    else:
        ctx["definition"] = "different"
    supplied = subject.SuppliedPopulationSource(
        subject.ServiceObservationJournal(ctx, ()), ev, generated, native
    )

    def refused(*args, **kwargs):
        raise AssertionError("future forward called")

    monkeypatch.setattr(population, "evaluate_conditional_population", refused)
    with pytest.raises((ValueError, KeyError, IndexError)):
        subject.evaluate_registered_population(r, supplied, row_limit=LIMIT)


def test_actual_bound_recomputed_before_execution(source, evaluated, monkeypatch):
    r, s, _e = source

    def refused(*args, **kwargs):
        raise AssertionError("future forward called")

    monkeypatch.setattr(population, "evaluate_conditional_population", refused)
    with pytest.raises(ValueError, match="infrastructure limit"):
        subject.evaluate_registered_population(
            r, s, row_limit=evaluated["all_support_bound"] - 1
        )


def test_owned_inputs_and_export(evaluated, source):
    r, s, _e = source
    ev = json.loads(json.dumps(_encoded(s.evidence)))
    owned = subject.SuppliedPopulationSource(
        s.journal, ev, s.generated_tape, s.native_tape
    )
    ev["trace"].clear()
    assert len(owned.evidence["trace"]) == 3
    out = subject.RegisteredPopulationResult(evaluated)
    exported = json.loads(json.dumps(subject.registered_population_record(out)))
    exported["registration"]["programme"]["cells"].clear()
    assert len(out.analyst["registration"]["programme"]["cells"]) == 12
    assert r.digest == subject._digest(r)


def test_future_failure_has_no_result_or_input_mutation(source, monkeypatch):
    r, s, e = source
    before = subject._canonical({"registration": r, "source": s})

    def failure(*args, **kwargs):
        raise RuntimeError("injected new future failure")

    monkeypatch.setattr(population, "evaluate_conditional_population", failure)
    with pytest.raises(RuntimeError, match="injected new future failure"):
        subject.evaluate_registered_population(r, s, row_limit=LIMIT)
    assert before == subject._canonical({"registration": r, "source": s})
    assert e.world.t == 3


def test_new_empty_source_keeps_all_empty_periods_and_no_ticket_attachments():
    c, scenario, r = declare()
    plan = replace(
        r.source,
        people=(),
        tape=replace(
            r.source.tape,
            namespace="unit/V9_registered_population/empty",
            arrival_marks=((), (), ()),
            gumbel_overrides=(),
        ),
    )
    programme = replace(
        r.programme,
        physical=tuple(
            replace(
                p,
                boarding_intents=(),
                boarding_orders=tuple((leg, ()) for leg, _ in plan.departures),
            )
            for p in r.programme.physical
        ),
        cells=tuple(
            replace(cell, persons=(), transfer_order=()) for cell in r.programme.cells
        ),
        information=(),
    )
    registered = subject.register_population(
        plan, programme, r.laws, (), row_limit=LIMIT
    )
    registered, s, e = create_source(c, scenario, registered)
    result = subject.registered_population_record(
        subject.evaluate_registered_population(registered, s, row_limit=LIMIT)
    )
    assert result["registration"]["attachments"] == ()
    assert len(result["source"]["evidence"]["ordinary_periods"]) == 3
    assert result["conditional_population"]["analyst"]["empty_population_sum_only"]
    assert all(
        q(row["population_material"]["mean"]) == 0
        for row in result["conditional_population"]["analyst"]["law_results"]
    )
    assert e.world.t == 3
