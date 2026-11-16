"""A.10/A.5/A.14/B0/V9/B4: NEW140974 / NEW141974 local synthetic units."""

import hashlib
import importlib.util
import os
import pickle
import sys
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F

import numpy as np
import pytest

from airline_market.core.rng import StreamRegistry
from airline_market.core.types import PriceAction
from airline_market.pipeline.screen_generators import screen_spec
from airline_market.policy.dated_demand import compile_dated_demand
from airline_market.policy.dated_rescue import DatedRebookingPolicy
from airline_market.policy.dated_reserve import UtcBookingClock
from airline_market.policy.dated_shocks import B1DatedShock
from airline_market.policy.regulator import CapPolicy, CapWindow
from airline_market.sim import action_kernel_utility as utility
from airline_market.sim import context_compatibility as context
from airline_market.sim import period_posting as posting
from airline_market.sim import role_law_context as role
from airline_market.sim.decision_consistency import (
    REFERENCE_SCOPE,
    DecisionAccount,
    DecisionAction,
    DecisionAtom,
    DeclaredDecisionPolicy,
    ObservedDecisionVector,
)
from airline_market.sim.native_causal_information import RULE as CAUSAL_RULE
from airline_market.sim.native_causal_information import (
    TIMING,
    DeclaredInputDelivery,
    DeclaredKnownType,
)
from airline_market.sim.received_information import DisclosureField
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

if "POSTING_ROLE_BINDING_SOURCE" in os.environ:
    _spec = importlib.util.spec_from_file_location(
        "new_posting_role_binding", os.environ["POSTING_ROLE_BINDING_SOURCE"]
    )
    m = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = m
    _spec.loader.exec_module(m)
else:
    from airline_market.sim import posting_role_binding as m

LIMIT = 10**9


def account(a, y, g, unknown=False):
    b, p, w, c = (
        ((13 if y == "served" else -1), 3, 1, (0 if y == "served" else 2))
        if a == "A"
        else ((9 if y == "served" else 0), 2, 0, (0 if y == "served" else 1))
        if a == "B"
        else ((0 if y == "served" else -2), -1, 0, 0)
    )
    e = (
        None
        if unknown or g == "unused"
        else F(1 if a == "A" else -1 if a == "B" else 2)
    )
    return DecisionAccount(
        tuple(map(F, (b, p, w, c))),
        ("service token", "NET payment token", "wait token", "cancellation token"),
        e,
        "declared extra utility, not native Gumbel",
        True,
        True,
        True,
        ("financial closed", "material closed", "utility closed"),
        "NEW140974 complete toy account",
        "explicit declaration",
        "E excludes B/P/W/C",
    )


def laws(case):
    p = utility.ActionKernelProblem(
        "NEW140974-person",
        "analytic known type; native awareness unknown",
        ("high", "low"),
        ("earlier", "other"),
        tuple(
            ObservedDecisionVector(
                g,
                (F(0), F(2 if g == "fast" else 0), F(0)),
                "finite declared vector, not native G realization",
            )
            for g in ("slow", "fast", "unused")
        ),
        ("A", "B", "outside"),
        "outside",
        ("served", "lost"),
        F(1, 2),
        "toy material token",
        "toy utility token",
        "one ordinary commitment law",
        "H event once before G conditioning",
        "H/G only, hidden X/future excluded",
        "native G/type awareness/law unprovided",
        "FOUR material components and extra separate",
        REFERENCE_SCOPE,
    )
    result = []
    delta = F(1, 10**3101)
    for who in ("q", "r"):
        selected = F(1, 3) if who == "q" else F(2, 5)
        if (case == "zero_q" and who == "q") or (case == "zero_r" and who == "r"):
            selected = F(0)
        prior = []
        rows = []
        for x in p.states:
            for h in p.histories:
                hp = selected if h == "earlier" else 1 - selected
                high = (
                    (F(2, 3) if h == "earlier" else F(1, 3))
                    if who == "q"
                    else (F(1, 4) if h == "earlier" else F(3, 4))
                )
                xp = high if x == "high" else 1 - high
                for g in p.vectors:
                    fast = F(1, 4) if x == "high" else F(3, 4)
                    gp = (
                        fast
                        if g.name == "fast"
                        else 1 - fast
                        if g.name == "slow"
                        else F(0)
                    )
                    prior.append(DecisionAtom(x, h, g.name, hp * xp * gp))
                    for a in p.actions:
                        if who == "q":
                            good = (
                                (F(3, 4) if x == "high" else F(1, 4))
                                if a == "A"
                                else (F(1, 2) if x == "high" else F(2, 5))
                                if a == "B"
                                else (F(1, 5) if x == "high" else F(3, 5))
                            )
                        else:
                            good = (
                                (F(1, 2) if x == "high" else F(1, 8))
                                if a == "A"
                                else (F(2, 5) if x == "high" else F(1, 3))
                                if a == "B"
                                else (F(1, 4) if x == "high" else F(1, 2))
                            )
                        rare = case == "tiny" and (x, h, g.name, a) == (
                            "low",
                            "earlier",
                            "slow",
                            "A",
                        )
                        if rare:
                            good = 1 - delta
                        for y in p.outcomes:
                            actual = account(a, y, g.name, unknown=rare and y == "lost")
                            outside = DecisionAccount(
                                (F(2 if y == "served" else -1), F(0), F(0), F(0)),
                                ("B", "P", "W", "C"),
                                None if g.name == "unused" else F(-1),
                                "full same-selected-path reference",
                                True,
                                True,
                                True,
                                (
                                    "financial closed",
                                    "material closed",
                                    "utility closed",
                                ),
                                "explicit full outside account",
                                "declared, not intervention",
                                "FOUR components and E separate",
                            )
                            rows.append(
                                utility.ActionOutcome(
                                    x,
                                    h,
                                    g.name,
                                    a,
                                    y,
                                    good if y == "served" else 1 - good,
                                    actual,
                                    outside,
                                )
                            )
        result.append(
            utility.ActionChanceLaw(
                who, "complete independent role declaration", tuple(prior), tuple(rows)
            )
        )
    desc = DeclaredDecisionPolicy(
        "descriptive",
        "B for every G, no native optimality inference",
        tuple(DecisionAction(h, g.name, "B") for h in p.histories for g in p.vectors),
    )
    return p, *result, desc


@pytest.fixture(scope="module")
def new_source():
    # One NEW named tape/source per fixture invocation, shared by pure tests.
    # No prior fixture helper, source, bank or saved-case loader is called.
    s = screen_spec(
        {
            "T": 4,
            "actions": 2,
            "n_spokes": 1,
            "capacity": 20,
            "capacity_support": [3, 4],
        }
    )
    c = compile_market(
        replace(
            s,
            demand=replace(
                s.demand,
                mode="poisson",
                arrival_probability=None,
                market_weights=None,
                intensity={
                    x.id: PeriodSchedule.parse({"constant": 2.0}) for x in s.markets
                },
            ),
        )
    )
    t0 = datetime(2027, 7, 19, tzinfo=timezone.utc)
    clock = UtcBookingClock(tuple(t0 + timedelta(hours=12 * t) for t in range(5)))
    dep = tuple(
        (l.id, t0 + timedelta(hours=84 if l.id == "B_OH0" else 108))
        for l in c.spec.legs
    )
    scenario = replace(
        compile_dated_demand(c, clock, dep),
        physical_shocks=(
            B1DatedShock("NEW140974-exit", "rival_exit", "B_OH0", 3, rival_carrier="B"),
        ),
    )
    cap = CapPolicy(
        (
            CapWindow(
                0,
                2,
                tuple((p.id, "100") for p in c.spec.products),
                requires_emergency=True,
            ),
        ),
        ((0, 2),),
    )
    rebook = DatedRebookingPolicy(12, 20, 100, interline=True)
    market = int(c.product_market[c.product_index["B_OH0"]])
    keys = tuple((0, i, market) for i in range(2))
    fields = (DisclosureField("notice", "text", "declared earlier notice", "text"),)
    types = tuple(
        DeclaredKnownType(k, None, "no supplied label, not ignorance", "NEW140974")
        for k in keys
    )
    deliveries = tuple(
        DeclaredInputDelivery(
            k,
            t,
            stage,
            f"new140974-{i}-{t}-{stage}",
            "notice",
            "earlier notice",
            "NEW140974-source",
            TIMING,
        )
        for i, k in enumerate(keys)
        for t in range(4)
        for stage in ("ordinary", "own", "interline")
        if t == 0 or stage != "ordinary"
    )
    kwargs = {
        "causal_fields": fields,
        "causal_known_types": types,
        "causal_deliveries": deliveries,
        "causal_definition": "NEW140974 source order not external reception",
        "causal_unobserved_definition": "G/law/awareness/preferences unknown",
        "causal_row_limit": 2000000,
        "dated_rebooking": rebook,
        "cap_policy": cap,
        "posting_call_limit": 8,
    }
    fares = ((226.0, 326.0), (239.0, 339.0), (252.0, 352.0), (268.0, 368.0))
    acts = tuple(
        {
            carrier.code: PriceAction({j: fares[t][i] for j in c.products_of[i]})
            for i, carrier in enumerate(c.spec.carriers)
        }
        for t in range(4)
    )
    fixed = tuple(
        tuple(
            (p.id, 130.0 if t < 2 else fares[t][c.carrier_index[p.carrier]])
            for p in c.spec.products
        )
        for t in range(4)
    )
    static = context._encoded(
        {
            "native_spec": c.spec,
            "spec_hash": c.spec_hash,
            "scenario": scenario.record(),
            "dated_rebooking": rebook,
            "dated_menu": None,
            "dated_reserve": None,
            "cap_policy": cap,
            "trigger_rule": None,
            "mechanical_benchmark": None,
        }
    )
    causal = context._encoded(
        {
            "rule": CAUSAL_RULE,
            "spec_hash": c.spec_hash,
            "fields": fields,
            "known_types": types,
            "deliveries": deliveries,
            "definition": kwargs["causal_definition"],
            "unobserved_definition": kwargs["causal_unobserved_definition"],
            "row_limit": 2000000,
        }
    )
    manifests = tuple(
        role.SourceManifest(
            k,
            "declared synthetic recipe marker, NOT actual execution/source seal",
            hashlib.sha256(("NEW140974-" + k).encode()).hexdigest(),
        )
        for k in role.MANIFEST_ROLES
    )
    prog = role.SourceProgramme(
        static,
        causal,
        fixed,
        manifests,
        "complete fixed synthetic source programme",
        "NEW140974-programme",
    )
    regs = {}
    for case in ("known", "tiny", "zero_q", "zero_r"):
        regs[case] = role.register_role_context(
            *laws(case),
            selected_history="earlier",
            person_key=keys[0],
            actions=(
                context.ContextAction("A", "A_OH0", False),
                context.ContextAction("B", "B_OH0", False),
                context.ContextAction("outside", None, True),
            ),
            programme=prog,
            observation_definition="declared finite G/type observation; native realization unknown",
            criterion="material_plus_decision_noise",
            support_limit=LIMIT,
        )
    bad = list(fixed)
    bad[3] = tuple((pid, v + 1 if pid == "A_OH0" else v) for pid, v in bad[3])
    regs["bad_posting"] = role.register_role_context(
        *laws("known"),
        selected_history="earlier",
        person_key=keys[0],
        actions=regs["known"].actions,
        programme=replace(prog, postings=tuple(bad)),
        observation_definition=regs["known"].observation_definition,
        criterion="material_plus_decision_noise",
        support_limit=LIMIT,
    )
    # ALL numerical/source/manifest programmes fixed above before tape/engine.
    tape = ExogenousTape.generate(
        c, 31, registry=StreamRegistry(141974, "unit/V9_posting_role_binding/July19")
    )
    g = tape.gumbel.copy()
    g[:] = -1e9
    g[:, :, -1] = 0
    g[:, :, c.product_index["B_OH0"]] = 1e9
    g[:, 1, -1] = 2e9
    g.setflags(write=False)
    tape = replace(
        tape,
        arrival_marks=(
            (ArrivalMark(0.2, market, 0, 0, 0), ArrivalMark(0.7, market, 0, 0, 0)),
            (),
            (),
            (),
        ),
        gumbel=g,
    )
    e = posting.PeriodPostingMarketEngine(scenario, **kwargs)
    e.reset(tape=tape)
    for action in acts:
        e.step(action)
    return e, e.posting_source(), e.causal_periods(), regs, manifests


def bind(fixture, case="known", **kw):
    _e, source, periods, regs, manifests = fixture
    return m.bind_posting_role_context(
        kw.pop("registration", regs[case]),
        kw.pop("posting_source", source),
        kw.pop("periods", periods),
        kw.pop("selected_input", periods[0].inputs[0]),
        actual_manifests=kw.pop("actual_manifests", manifests),
        support_limit=kw.pop("support_limit", LIMIT),
        **kw,
    )


def test_full_source_and_independent_closed_forms(new_source):
    v = bind(new_source)
    a = v.attached
    assert a.subjective_evidence == F(1, 3) and a.reference_evidence == F(2, 5)
    ev = a.declared_evaluation.evaluations
    assert ev[0].totals[0] == F(17, 6) and ev[3].totals[0] == F(-1, 2)
    assert ev[1].totals[0] == F(5, 3) and ev[4].totals[0] == F(1, 2)
    assert ev[0].totals[2] == F(19, 12) and ev[3].totals[2] == F(-1, 2)
    assert (
        v.native_G is None
        and v.native_probability_knowledge_preferences_or_CS_authenticated is False
    )
    assert len(v.posting_source.body["periods"]) == 4
    assert (
        not v.posting_source.body["periods"][2]["input_links"]
        and not v.posting_source.body["periods"][3]["input_links"]
    )
    assert context._encoded(a.context.periods) == context._encoded(new_source[2])


def test_tiny_positive_unknown_and_zero_rows_retained(new_source):
    v = bind(new_source, "tiny")
    ev = v.attached.declared_evaluation.evaluations
    d = F(1, 10**3101)
    assert (
        ev[0].totals[0] == F(23, 6) - F(4, 3) * d
        and ev[3].totals[0] == F(17, 8) - 3 * d
    )
    assert ev[0].totals[2] is None and ev[3].totals[2] is None
    assert (
        ev[0].undefined_weights[2] == d / 12
        and ev[3].undefined_weights[2] == F(3, 16) * d
    )
    assert any(
        r.weight == 0 and r.values[2] is None and r.contributions[2] == 0
        for r in ev[0].rows
    )


@pytest.mark.parametrize("case", ["zero_q", "zero_r"])
def test_upstream_zero_keeps_NULL_without_utility_backend(
    new_source, monkeypatch, case
):
    def refuse(*a, **k):
        raise AssertionError("no utility execution for undefined upstream")

    monkeypatch.setattr(utility, "execute_action_utility", refuse)
    v = bind(new_source, case)
    assert v.attached.declared_evaluation is None
    assert (v.attached.subjective_conditional is None) == (case == "zero_q")
    assert (v.attached.reference_conditional is None) == (case == "zero_r")


def test_one_attachment_and_no_reader_classifier_or_solver_replay(
    new_source, monkeypatch
):
    calls = []
    real = role.attach_role_context

    def counted(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(role, "attach_role_context", counted)
    v = bind(new_source)
    assert len(calls) == 1

    def refuse(*a, **k):
        raise AssertionError("reader must not forward classifier/solver/source")

    monkeypatch.setattr(role, "attach_role_context", refuse)
    monkeypatch.setattr(context, "_classify", refuse)
    monkeypatch.setattr(utility, "execute_action_utility", refuse)
    monkeypatch.setattr(posting.PeriodPostingMarketEngine, "step", refuse)
    monkeypatch.setattr(posting.PeriodPostingMarketEngine, "posting_source", refuse)
    m.posting_role_record(v)
    m.earlier_posting_role_view(v)


def test_exact_earlier_view_privacy_and_no_current_result_G_selection(new_source):
    v = bind(new_source)
    sel = new_source[2][0].inputs[0]
    view = m.earlier_posting_role_view(v)
    assert view == context._encoded(sel.receipt.view)
    assert set(view) == {
        "scope",
        "compiled_opportunity_utc",
        "current_stage",
        "known_type",
        "definition",
        "unobserved_definition",
        "schema",
        "earlier_person_history",
    }
    assert (
        v.attached.context.realized_G is None
        and v.attached.context.probability_law is None
    )
    assert v.attached.context.common_action == "B"
    chosen = {
        (c.history, c.vector): c.action
        for c in v.attached.declared_evaluation.solution.cells
    }
    assert chosen["earlier", "slow"] == "A" and chosen["earlier", "fast"] == "B"
    with pytest.raises(TypeError):
        bind(new_source, realized_G="fast")


def test_owned_exports_input_native_state_and_rng_unchanged(new_source):
    e, source, _ps, regs, _ = new_source
    before = pickle.dumps(e.snapshot())
    rng = pickle.dumps(np.random.get_state())
    v = bind(new_source)
    out = v.export()
    out["attached"]["context"]["earlier_view"]["scope"] = "changed"
    assert m.earlier_posting_role_view(v)["scope"] != "changed"
    assert (
        pickle.dumps(e.snapshot()) == before
        and pickle.dumps(np.random.get_state()) == rng
    )
    with pytest.raises(FrozenInstanceError):
        v.native_G = "guessed"
    with pytest.raises(TypeError):
        v.posting_source.body["rule"] = "changed"
    assert (
        source.digest == v.posting_source.digest
        and regs["known"].digest == v.attached.registration.digest
    )


@pytest.mark.parametrize(
    "problem", ["periods", "person", "stage", "posting", "manifest", "rule", "opaque"]
)
def test_complete_source_registration_and_schema_refusals(new_source, problem):
    _e, s, ps, regs, ms = new_source
    kwargs = {}
    if problem == "periods":
        kwargs["periods"] = ps[:3]
    if problem == "person":
        kwargs["selected_input"] = ps[0].inputs[1]
    if problem == "stage":
        kwargs["selected_input"] = ps[1].inputs[0]
    if problem == "posting":
        kwargs["registration"] = regs["bad_posting"]
    if problem == "manifest":
        kwargs["actual_manifests"] = (replace(ms[0], sha256="0" * 64),) + ms[1:]
    if problem == "rule":
        kwargs["posting_source"] = replace(s, rule="unsupported")
    if problem == "opaque":
        body = s.export()
        body["extra"] = np.zeros(1)
        kwargs["posting_source"] = posting.PeriodPostingSource(body, "not-a-seal")
    with pytest.raises((ValueError, TypeError)):
        bind(new_source, **kwargs)


@pytest.mark.parametrize(
    "problem",
    [
        "missing_empty",
        "price",
        "bool_revision",
        "extra_schema",
        "clock",
        "bool_root",
        "bool_key",
    ],
)
def test_posting_internal_consistency_refusals_before_attachment(
    new_source, monkeypatch, problem
):
    s = new_source[1]
    body = s.export()
    if problem == "missing_empty":
        body["periods"] = body["periods"][:2] + body["periods"][3:]
    if problem == "price":
        body["periods"][3]["postings"][0]["prices"][0].update(context._encoded(999.0))
    if problem == "bool_revision":
        body["periods"][0]["postings"][0]["native_cap_receipt"]["revision"] = True
    if problem == "bool_root":
        body["periods"][1]["input_links"][0]["root"] = False
    if problem == "bool_key":
        key = body["periods"][0]["input_links"][0]["original_key"]
        body["periods"][0]["input_links"][0]["original_key"] = (False,) + tuple(key[1:])
    if problem == "extra_schema":
        body["periods"][0]["unused"] = 0
    if problem == "clock":
        body["contract"]["clock"] = tuple(
            v if i != 2 else "2027-07-20T00:00:01+00:00"
            for i, v in enumerate(body["contract"]["clock"])
        )
        body["periods"][2]["compiled_opportunity_utc"] = body["contract"]["clock"][2]
    bad = posting.PeriodPostingSource(body, posting._digest(body))

    def refuse(*a, **k):
        raise AssertionError("no attachment on inconsistent source")

    monkeypatch.setattr(role, "attach_role_context", refuse)
    with pytest.raises((ValueError, TypeError)):
        bind(new_source, posting_source=bad)


def test_complete_support_before_attachment_and_frozen_classifier(
    new_source, monkeypatch
):
    def refuse(*a, **k):
        raise AssertionError("no forward before outer limit")

    monkeypatch.setattr(role, "attach_role_context", refuse)
    monkeypatch.setattr(context, "_classify", refuse)
    with pytest.raises(ValueError, match="support|exceeds"):
        bind(new_source, support_limit=1)
    with pytest.raises(TypeError):
        bind(new_source, support_limit=True)


def test_complete_output_support_accepts_exact_bound_and_refuses_one_less(
    new_source, monkeypatch
):
    value = bind(new_source)
    exact = bind(new_source, support_limit=value.required_nodes)
    assert exact.required_nodes == value.required_nodes
    assert exact.export()["required_nodes"] == value.required_nodes

    def refuse(*args, **kwargs):
        raise AssertionError("no forward below complete support")

    monkeypatch.setattr(role, "attach_role_context", refuse)
    with pytest.raises(ValueError, match="support|exceeds"):
        bind(new_source, support_limit=value.required_nodes - 1)


def test_outer_inner_full_source_must_agree_even_with_repaired_hashes(new_source):
    value = bind(new_source)
    body = value.posting_source.export()
    body["parent_source"]["native_decision_input_rows_captured"] = False
    source = posting.PeriodPostingSource(body, posting._digest(body))
    bad = replace(value, posting_source=source)
    bad = replace(
        bad,
        input_digest=m._hash(
            m._input_payload(
                source,
                value.attached.registration,
                value.attached.context.periods,
                value.attached.context.selected_input,
                value.attached.actual_manifests,
                value.support_limit,
            )
        ),
    )
    bad = replace(bad, output_digest=m._hash(m._result_payload(bad)))
    with pytest.raises(ValueError, match="SAME full source"):
        bad.export()


@pytest.mark.parametrize(
    "field",
    [
        "input_digest",
        "output_digest",
        "required_nodes",
        "native_G",
        "flag",
        "limit",
        "rule",
    ],
)
def test_stored_integrity_refusals_without_resolve(new_source, monkeypatch, field):
    v = bind(new_source)
    if field in ("input_digest", "output_digest"):
        bad = replace(v, **{field: "0" * 64})
    elif field == "required_nodes":
        bad = replace(v, required_nodes=v.required_nodes - 1)
    elif field == "native_G":
        bad = replace(v, native_G="fast")
    elif field == "flag":
        bad = replace(
            v, native_probability_knowledge_preferences_or_CS_authenticated=True
        )
    elif field == "limit":
        bad = replace(v, support_limit=v.required_nodes - 1)
    else:
        bad = replace(v, rule="unsupported")

    def refuse(*a, **k):
        raise AssertionError("no solver replay")

    monkeypatch.setattr(utility, "execute_action_utility", refuse)
    with pytest.raises((ValueError, TypeError)):
        bad.export()


def test_unexpected_pure_failure_returns_no_partial_or_native_mutation(
    new_source, monkeypatch
):
    before = pickle.dumps(new_source[0].snapshot())

    def fail(*a, **k):
        raise RuntimeError("NEW140974 injected pure failure")

    monkeypatch.setattr(role, "attach_role_context", fail)
    with pytest.raises(RuntimeError, match="injected"):
        bind(new_source)
    assert pickle.dumps(new_source[0].snapshot()) == before
