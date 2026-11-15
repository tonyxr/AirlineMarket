"""A.10/A.5/B0/V9: NEW exact controlled laws and genuine native-person bridge."""

import importlib.util
import json
import os
import pickle
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F
from itertools import product

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.dated_demand import compile_dated_demand
from airline_market.policy.dated_reserve import UtcBookingClock
from airline_market.sim.ordinary_information import OrdinaryInformationMarketEngine
from airline_market.sim.received_information import (
    DisclosureField,
    ReceivedDisclosure,
    append_received_disclosures,
    received_decision_receipt,
    received_information_journal,
)
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

if os.environ.get("CONTROLLED_INFORMATION_SOURCE"):
    sp = importlib.util.spec_from_file_location(
        "staged_controlled_information", os.environ["CONTROLLED_INFORMATION_SOURCE"]
    )
    module = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = module
    sp.loader.exec_module(module)
else:
    from airline_market.sim import controlled_information as module


def model(*, query=F(2, 5), periods=2, key=(0, 0, 0), change=False):
    # NEW declarations, different from the CLOSED two-decision analytic example.
    states, actions, signals = (
        ("robust", "fragile"),
        ("inspect", "defer"),
        ("red", "blue"),
    )
    histories, stages = ((),), []
    for t in range(periods):
        kernel = []
        for state, action, following, signal in product(
            states, actions, states, signals
        ):
            # Signal and transition dependent when change=True; joint supplied explicitly.
            if change:
                row = {
                    (states[0], signals[0]): F(1, 8),
                    (states[0], signals[1]): F(3, 8),
                    (states[1], signals[0]): F(1, 4),
                    (states[1], signals[1]): F(1, 4),
                }
                p = row[following, signal]
            else:
                red = F(1, 4) if state == states[0] else F(3, 4)
                if action == "defer":
                    red = F(0)
                p = (
                    (red if signal == "red" else 1 - red)
                    if state == following
                    else F(0)
                )
            kernel.append((state, action, following, signal, p))
        program = tuple(
            (
                h,
                (query, 1 - query)
                if t == 0
                else ((F(3, 5), F(2, 5)) if h[-1][1] == "red" else (F(1, 5), F(4, 5))),
            )
            for h in histories
        )
        stages.append(module.ControlledStage(actions, signals, kernel, program))
        histories = tuple(
            h + ((a, o),) for h in histories for a in actions for o in signals
        )
    return module.ControlledInformationModel(
        key,
        "declared-informed-type",
        "known fixed type, NOT native latent class",
        states,
        (F(2, 3), F(1, 3)),
        stages,
        "complete declared initial information",
        "NEW synthetic conditional law",
        module.ASSUMPTION,
    )


def compile_new(**kwargs):
    return module.compile_controlled_information(model(**kwargs), row_limit=1000000)


def cell(result, history):
    return next(c for c in result.cells if c["history"] == history)


def test_exact_new_sequential_posterior_and_no_prior_reset():
    r = compile_new()
    first = cell(r, (("inspect", "red"),))
    last = cell(r, (("inspect", "red"), ("inspect", "blue")))
    assert first["posterior"] == (("robust", F(2, 5)), ("fragile", F(3, 5)))
    assert last["posterior"] == (("robust", F(2, 3)), ("fragile", F(1, 3)))
    reset_robust = (F(2, 3) * F(3, 4)) / (F(2, 3) * F(3, 4) + F(1, 3) * F(1, 4))
    assert reset_robust == F(6, 7) and last["posterior"][0][1] != reset_robust
    assert len(r.cells) == 1 + 4 + 16 and len(r.transitions) == 16 + 64
    for t in range(3):
        rows = [c for c in r.cells if c["stage"] == t]
        assert sum(c["evidence"] for c in rows) == 1
        for state in r.model.states:
            joint = sum(dict(c["state_joint"])[state] for c in rows)
            assert joint == r.model.initial[r.model.states.index(state)]
            assert (
                sum(
                    c["evidence"] * dict(c["posterior"])[state]
                    for c in rows
                    if c["posterior"] is not None
                )
                == joint
            )


def test_independent_full_hidden_path_reference_three_stages():
    r = compile_new(periods=3)
    expected = {}
    # Independent complete path expansion, without compiler state aggregation.
    for hidden in product(r.model.states, repeat=4):
        for observable in product(
            tuple(product(("inspect", "defer"), ("red", "blue"))), repeat=3
        ):
            mass = r.model.initial[r.model.states.index(hidden[0])]
            for t, (action, signal) in enumerate(observable):
                stage = r.model.stages[t]
                pi = dict(stage.program)[observable[:t]][stage.actions.index(action)]
                k = next(
                    row[4]
                    for row in stage.kernel
                    if row[:4] == (hidden[t], action, hidden[t + 1], signal)
                )
                mass *= pi * k
            key = observable, hidden[-1]
            expected[key] = expected.get(key, F(0)) + mass
    terminal = [c for c in r.cells if c["stage"] == 3]
    assert all(
        dict(c["state_joint"])[state] == expected[c["history"], state]
        for c in terminal
        for state in r.model.states
    )
    assert len(terminal) == 64 and sum(c["evidence"] == 0 for c in terminal) > 0


def test_joint_transition_observation_dependency_not_product_of_marginals():
    r = compile_new(periods=1, change=True)
    red = cell(r, (("inspect", "red"),))
    assert red["posterior"] == (("robust", F(1, 3)), ("fragile", F(2, 3)))
    assert red["posterior"] != (("robust", F(1, 2)), ("fragile", F(1, 2)))


@pytest.mark.parametrize("query", [F(0), F(1, 10**550)])
def test_zero_off_path_null_versus_tiny_exact_positive(query):
    r = compile_new(query=query)
    c = cell(r, (("inspect", "red"),))
    assert c["evidence"] == query * F(5, 12)
    assert (c["posterior"] is None) == (query == 0)
    if query:
        assert c["evidence"] > 0 and float(c["evidence"]) == 0
        assert c["posterior"][0][1] == F(2, 5)
    assert len(r.cells) == 21 and len(r.transitions) == 80


@pytest.mark.parametrize(
    "change",
    [
        "initial",
        "kernel",
        "program",
        "history",
        "missing",
        "assumption",
        "float",
        "duplicate",
    ],
)
def test_complete_exact_law_and_history_only_support_refusals(change):
    m = model()
    s = m.stages[0]
    if change == "initial":
        m = replace(m, initial=(F(1, 2), F(1, 3)))
    elif change == "kernel":
        s = replace(s, kernel=(*s.kernel[:-1], (*s.kernel[-1][:-1], F(2))))
    elif change == "program":
        s = replace(s, program=(((), (F(1), F(1))),))
    elif change == "history":
        s = replace(s, program=(((("hidden-regime", "red"),), (F(1), F(0))),))
    elif change == "missing":
        s = replace(s, kernel=s.kernel[:-1])
    elif change == "assumption":
        m = replace(m, action_assumption="not declared")
    elif change == "float":
        m = replace(m, initial=(0.5, 0.5))
    else:
        s = replace(s, actions=("inspect", "inspect"))
    m = replace(m, stages=(s, *m.stages[1:]))
    with pytest.raises((ValueError, TypeError)):
        module.compile_controlled_information(m, row_limit=1000000)


def test_all_zero_support_limit_before_any_compiler_forwards(monkeypatch):
    m = model(query=F(0), periods=3)
    units = module._validate(m, 1000000)
    with pytest.raises(module.ControlledInformationLimit):
        module.compile_controlled_information(m, row_limit=units - 1)
    r = module.compile_controlled_information(m, row_limit=units)
    assert r.support_units == units and len(r.cells) == 85
    with pytest.raises(ValueError):
        module.compile_controlled_information(m, row_limit=True)


def native_versions(*, outside=False, count=2):
    # NEW genuine native source, no imported previous fixture helper.
    spec = screen_spec(
        {
            "T": 2,
            "actions": 2,
            "n_spokes": 1,
            "capacity": 20,
            "capacity_support": [3, 4],
        }
    )
    demand = replace(
        spec.demand,
        mode="poisson",
        arrival_probability=None,
        market_weights=None,
        intensity={m.id: PeriodSchedule.parse({"constant": 2.0}) for m in spec.markets},
    )
    c = compile_market(replace(spec, demand=demand))
    start = datetime(2026, 12, 4, tzinfo=timezone.utc)
    clock = UtcBookingClock(tuple(start + timedelta(hours=h) for h in (0, 12, 24)))
    departures = tuple((leg.id, start + timedelta(hours=96)) for leg in c.spec.legs)
    engine = OrdinaryInformationMarketEngine(compile_dated_demand(c, clock, departures))
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(80974, "unit/V9_controlled_information")
    )
    arrivals = tuple(
        ArrivalMark((i + 1) / (count + 1), 0, 0, 0, 0) for i in range(count)
    )
    g = tape.gumbel.copy()
    g[:] = -1e9
    g[:, :, -1] = 0
    if count:
        g[:, :, c.product_index["B_OH0"]] = 1e9
        if count > 1:
            g[:, 1, -1] = 2e9
    g.setflags(write=False)
    engine.reset(
        tape=replace(
            tape, arrival_marks=tuple(arrivals for _ in tape.arrival_marks), gumbel=g
        )
    )
    for _ in range(2):
        engine.step(
            {
                carrier.code: PriceAction(
                    {j: 200 if i == 0 else 300 for j in c.products_of[i]}
                )
                for i, carrier in enumerate(c.spec.carriers)
            }
        )
    j = received_information_journal(
        c,
        engine.information_periods(),
        clock,
        [
            DisclosureField(
                "controlled_signal",
                "text",
                "declared controlled observation",
                "category",
            )
        ],
        definition="all own text signals in this declared experiment",
        unobserved_definition="native marks/outcomes/other people/future law excluded",
        row_limit=200000,
    )
    if not count:
        return engine, j, (), (), None
    key = j.person_keys[1 if outside else 0]
    m = model(key=key)
    t0 = j.as_of + timedelta(minutes=1)
    r0 = received_decision_receipt(j, key, decision_id="first", decision_at=t0)
    d1 = ReceivedDisclosure(
        "new-red",
        key,
        "controlled_signal",
        "red",
        t0 + timedelta(minutes=1),
        t0 + timedelta(minutes=2),
        t0 + timedelta(minutes=3),
        "NEW declared signal",
    )
    j1 = append_received_disclosures(j, [d1], as_of=d1.received_at)
    t1 = t0 + timedelta(minutes=4)
    r1 = received_decision_receipt(j1, key, decision_id="second", decision_at=t1)
    d2 = ReceivedDisclosure(
        "new-blue",
        key,
        "controlled_signal",
        "blue",
        t1 + timedelta(minutes=1),
        t1 + timedelta(minutes=2),
        t1 + timedelta(minutes=3),
        "NEW declared second signal",
    )
    j2 = append_received_disclosures(j1, [d2], as_of=d2.received_at)
    r2 = received_decision_receipt(
        j2, key, decision_id="third", decision_at=t1 + timedelta(minutes=4)
    )
    actions = (
        module.DeclaredOwnAction(
            key, "first", "inspect", t0, t0 + timedelta(seconds=1), "NEW own action"
        ),
        module.DeclaredOwnAction(
            key, "second", "inspect", t1, t1 + timedelta(seconds=1), "NEW own action"
        ),
    )
    known = module.KnownPersonType(
        key, m.known_type, m.type_definition, j.as_of, "NEW known-type declaration"
    )
    return engine, m, ((j, r0), (j1, r1), (j2, r2)), actions, known


def bind_new(m, versions, actions, known, **kwargs):
    return module.bind_controlled_receipts(
        module.compile_controlled_information(m, row_limit=1000000),
        versions,
        actions,
        known,
        signal_field="controlled_signal",
        row_limit=kwargs.get("row_limit", 2000000),
    )


@pytest.mark.parametrize("outside", [False, True])
def test_new_native_original_person_all_receipts_actions_type_and_no_native_mutation(
    outside, monkeypatch
):
    e, m, versions, actions, known = native_versions(outside=outside)
    saved = pickle.dumps(e.snapshot())
    rng = pickle.dumps(np.random.get_state())

    def fail(*args, **kwargs):
        raise AssertionError("pure compiler invoked native source")

    monkeypatch.setattr(e, "step", fail)
    monkeypatch.setattr(e, "information_periods", fail)
    b = bind_new(m, versions, actions, known)
    assert b.permitted_history == (("inspect", "red"), ("inspect", "blue"))
    assert b.posterior == (("robust", F(2, 3)), ("fragile", F(1, 3)))
    assert len(b.analyst_context["sources"][-1]["original_person_keys"]) == 4
    assert (
        pickle.dumps(e.snapshot()) == saved
        and pickle.dumps(np.random.get_state()) == rng
    )
    text = json.dumps(module.controlled_information_record(b), allow_nan=False)
    assert '"scientific_budget": null' in text
    narrow = module._encode(module.jsonable(b.decision_view))
    assert "analyst_context" not in narrow and "posterior" not in narrow
    assert narrow["known_type"] == m.known_type and "state_joint" not in narrow


def test_binding_zero_evidence_no_belief_fill_and_all_population_support():
    _e, m, versions, actions, known = native_versions()
    m = replace(m, stages=model(query=F(0)).stages)
    b = bind_new(m, versions, actions, known)
    assert b.evidence == 0 and b.posterior is None
    assert len(b.analyst_context["sources"][0]["original_person_keys"]) == 4
    e, j, _, _, _ = native_versions(count=0)
    assert j.person_keys == () and len(e.information_periods()) == 2


@pytest.mark.parametrize(
    "failure",
    [
        "omitted_action",
        "wrong_person",
        "wrong_type",
        "late_type",
        "late_action",
        "changed_action",
        "later_version",
        "omit_signal",
        "prefix",
        "extra_signal",
        "forged_receipt",
    ],
)
def test_binding_complete_history_and_causal_source_refusals(failure):
    _e, m, versions, actions, known = native_versions()
    versions = list(versions)
    actions = list(actions)
    if failure == "omitted_action":
        actions.pop()
    elif failure == "wrong_person":
        actions[0] = replace(actions[0], person_key=(0, 1, 0))
    elif failure == "wrong_type":
        known = replace(known, value="other type")
    elif failure == "late_type":
        known = replace(known, known_at=actions[0].taken_at)
    elif failure == "late_action":
        actions[0] = replace(actions[0], recorded_at=actions[1].taken_at)
    elif failure == "changed_action":
        actions[0] = replace(actions[0], action="hidden-dependent action")
    elif failure == "later_version":
        versions[1] = (versions[2][0], versions[1][1])
    elif failure == "omit_signal":
        versions[1] = versions[0]
    elif failure == "prefix":
        j, r = versions[2]
        changed = replace(j.disclosures[0], source_ref="changed historical record")
        versions[2] = (replace(j, disclosures=(changed, j.disclosures[1])), r)
    elif failure == "extra_signal":
        j, r = versions[2]
        d = replace(j.disclosures[-1], delivery_id="extra", value="blue")
        # SAME atomic batch time retained in this NEW manual journal, no backfill API.
        j = replace(j, disclosures=(*j.disclosures, d))
        r = received_decision_receipt(
            j,
            m.person_key,
            decision_id="third",
            decision_at=module.utc(r.view["decision_at"]),
        )
        versions[2] = j, r
    else:
        j, r = versions[1]
        v = module.received_decision_view(r)
        v["received"] = []
        versions[1] = j, replace(r, view=v)
    saved = pickle.dumps((m, versions, actions, known))
    with pytest.raises((ValueError, TypeError)):
        bind_new(m, versions, actions, known)
    assert pickle.dumps((m, versions, actions, known)) == saved


def test_owned_full_exports_and_complete_binding_infrastructure_limit():
    _e, m, versions, actions, known = native_versions()
    b = bind_new(m, versions, actions, known)
    out = module.controlled_information_record(b)
    out["result"]["analyst_context"]["model"]["known_type"] = "changed export"
    assert (
        module.controlled_information_record(b)["result"]["analyst_context"]["model"][
            "known_type"
        ]
        == m.known_type
    )
    with pytest.raises(TypeError):
        b.decision_view["known_type"] = "mutation"
    with pytest.raises(module.ControlledInformationLimit):
        bind_new(m, versions, actions, known, row_limit=1)
    with pytest.raises(ValueError):
        replace(known, known_at=datetime(2026, 12, 4))
    with pytest.raises(ValueError):
        replace(actions[0], recorded_at=actions[0].taken_at - timedelta(seconds=1))


def test_new_v2_full_journal_schema_definition_contract_bound():
    _e, m, versions, actions, known = native_versions()
    for field in ("definition", "unobserved_definition", "fields"):
        changed = list(versions)
        j, r = changed[1]
        value = (
            (replace(j.fields[0], definition="changed schema meaning"),)
            if field == "fields"
            else "changed journal meaning"
        )
        j = replace(j, **{field: value})
        r = received_decision_receipt(
            j, m.person_key, decision_id="second", decision_at=actions[1].taken_at
        )
        changed[1] = j, r
        with pytest.raises(ValueError, match="version contract"):
            bind_new(m, changed, actions, known)


def test_new_v2_all_person_global_version_backfill_refused():
    _e, m, versions, actions, known = native_versions()
    changed = list(versions)
    j0, r0 = changed[0]
    # Old version was already as-of 30 seconds after source close; new other-person
    # record predates that version while own controlled signal is correctly later.
    j0 = replace(j0, as_of=j0.as_of + timedelta(seconds=30))
    r0 = received_decision_receipt(
        j0, m.person_key, decision_id="first", decision_at=actions[0].taken_at
    )
    changed[0] = j0, r0
    for i in (1, 2):
        j, r = changed[i]
        other = ReceivedDisclosure(
            "backfill-other",
            j.person_keys[1],
            "controlled_signal",
            "blue",
            j.source_closed_at,
            j.source_closed_at,
            j.source_closed_at + timedelta(seconds=10),
            "NEW explicit invalid backfill",
        )
        j = replace(j, disclosures=(other, *j.disclosures))
        changed[i] = j, r
    with pytest.raises(ValueError, match="backfill"):
        bind_new(m, changed, actions, known)


def test_new_v2_budget_bounds_every_compiled_structural_output():
    for q in (F(0), F(2, 5), F(1, 10**550)):
        r = compile_new(query=q, periods=3)
        assert r.support_units >= module._units(module.jsonable(r))
        json.dumps(module.controlled_information_record(r), allow_nan=False)


def test_new_v2_pure_binding_failure_preserves_inputs(monkeypatch):
    _e, m, versions, actions, known = native_versions()
    saved = pickle.dumps((m, versions, actions, known))

    def fail(value):
        raise RuntimeError("NEW injected full source export failure")

    monkeypatch.setattr(module, "received_information_record", fail)
    with pytest.raises(RuntimeError, match="source export failure"):
        bind_new(m, versions, actions, known)
    assert pickle.dumps((m, versions, actions, known)) == saved
