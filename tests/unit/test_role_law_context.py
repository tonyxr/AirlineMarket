"""A.10/A.5/A.14/B0/V9/B4: NEW136974 declarations/NEW137974 source only."""

import copy
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
from airline_market.policy.regulator import NO_CAP
from airline_market.sim import action_kernel_utility as utility
from airline_market.sim import context_compatibility as cc
from airline_market.sim.decision_consistency import (
    REFERENCE_SCOPE,
    DecisionAccount,
    DecisionAction,
    DecisionAtom,
    DeclaredDecisionPolicy,
    ObservedDecisionVector,
)
from airline_market.sim.native_causal_information import RULE as SOURCE_RULE
from airline_market.sim.native_causal_information import (
    TIMING,
    DeclaredInputDelivery,
    DeclaredKnownType,
    NativeCausalInformationMarketEngine,
    native_causal_information_record,
)
from airline_market.sim.received_information import DisclosureField
from airline_market.sim.tape import ArrivalMark, ExogenousTape
from airline_market.spec import compile_market
from airline_market.spec.schedule import PeriodSchedule

if "ROLE_LAW_CONTEXT_SOURCE" in os.environ:
    _spec = importlib.util.spec_from_file_location(
        "new_role_law_context", os.environ["ROLE_LAW_CONTEXT_SOURCE"]
    )
    m = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = m
    _spec.loader.exec_module(m)
else:
    from airline_market.sim import role_law_context as m

LIMIT = 10**8


def account(b, p=0, e=0):
    return DecisionAccount(
        (F(b), F(p), F(0), F(0)),
        ("service", "NET payment", "waiting", "cancellation"),
        F(e),
        "additional utility",
        True,
        True,
        True,
        ("financial follow-up", "material follow-up", "utility follow-up"),
        "NEW136974 full account",
        "declared future",
        "E excludes B/P/W/C",
    )


def laws():
    p = utility.ActionKernelProblem(
        "NEW136974-person",
        "fixed analytic type, not native known label",
        ("robust", "fragile"),
        ("earlier", "other"),
        tuple(
            ObservedDecisionVector(
                g,
                (F(0), F(2 if g == "tilt" else 0), F(0)),
                "declared model vector; native awareness unknown",
            )
            for g in ("calm", "tilt", "zero")
        ),
        ("A", "B", "outside"),
        "outside",
        ("service", "loss"),
        F(1, 2),
        "material token",
        "utility token",
        "ordinary commitment law",
        "earlier event once; no current-result conditioning",
        "H/G only, X/future hidden",
        "native G/type knowledge unprovided",
        "FOUR components separate from E",
        REFERENCE_SCOPE,
    )
    result = []
    for role in ("q", "r"):
        prior, rows = [], []
        for x in p.states:
            for h in p.histories:
                high = (x == "robust") == (h == "earlier")
                base = F((3 if high else 1) if role == "q" else (1 if high else 3), 8)
                for g in p.vectors:
                    prior.append(
                        DecisionAtom(
                            x,
                            h,
                            g.name,
                            base
                            * (
                                F(2, 3)
                                if g.name == "calm"
                                else F(1, 3)
                                if g.name == "tilt"
                                else F(0)
                            ),
                        )
                    )
                    for a in p.actions:
                        good = (
                            (F(3, 4) if x == "robust" else F(1, 4))
                            if a == "A"
                            else F(1, 2)
                            if a == "B"
                            else F(1, 4)
                        )
                        for y in p.outcomes:
                            ac = (
                                account(12 if y == "service" else -4, 2)
                                if a == "A"
                                else account(
                                    8 if y == "service" else 1,
                                    1,
                                    3 if g.name == "tilt" else 0,
                                )
                                if a == "B"
                                else account(2 if y == "service" else -1, -1)
                            )
                            rows.append(
                                utility.ActionOutcome(
                                    x,
                                    h,
                                    g.name,
                                    a,
                                    y,
                                    good if y == "service" else 1 - good,
                                    ac,
                                    account(1, -1, -1),
                                )
                            )
        result.append(
            utility.ActionChanceLaw(
                role, "complete separately declared role law", tuple(prior), tuple(rows)
            )
        )
    desc = DeclaredDecisionPolicy(
        "descriptive",
        "invariant supplied B action, no optimality",
        tuple(DecisionAction(h, g.name, "B") for h in p.histories for g in p.vectors),
    )
    return p, *result, desc


def prepared():
    # NEW local setup; no old fixture helper or saved source is invoked.
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
    t0 = datetime(2027, 6, 21, tzinfo=timezone.utc)
    clock = UtcBookingClock(tuple(t0 + timedelta(hours=12 * t) for t in range(4)))
    departures = tuple(
        (l.id, t0 + timedelta(hours=84 if l.id == "B_OH0" else 108))
        for l in c.spec.legs
    )
    scenario = replace(
        compile_dated_demand(c, clock, departures),
        physical_shocks=(
            B1DatedShock("NEW136974-exit", "rival_exit", "B_OH0", 3, rival_carrier="B"),
        ),
    )
    market = int(c.product_market[c.product_index["B_OH0"]])
    keys = tuple((0, i, market) for i in range(2))
    fields = (DisclosureField("notice", "text", "declared notice", "text"),)
    types = tuple(
        DeclaredKnownType(
            k, None, "no known-label declaration, not ignorance", "NEW136974-type"
        )
        for k in keys
    )
    deliveries = tuple(
        DeclaredInputDelivery(
            k,
            t,
            s,
            f"NEW136974-{i}-{t}-{s}",
            "notice",
            "source-order notice",
            "NEW136974-source",
            TIMING,
        )
        for i, k in enumerate(keys)
        for t in range(3)
        for s in ("ordinary", "own", "interline")
        if t == 0 or s != "ordinary"
    )
    kwargs = {
        "causal_fields": fields,
        "causal_known_types": types,
        "causal_deliveries": deliveries,
        "causal_definition": "NEW136974 source order only",
        "causal_unobserved_definition": "no native awareness/law/preferences authentication",
        "causal_row_limit": 2000000,
    }
    policy = DatedRebookingPolicy(72, 20, 100, interline=True)
    programme = m.SourceProgramme(
        cc._encoded(
            {
                "native_spec": c.spec,
                "spec_hash": c.spec_hash,
                "scenario": scenario.record(),
                "dated_rebooking": policy,
                "dated_menu": None,
                "dated_reserve": None,
                "cap_policy": NO_CAP,
                "trigger_rule": None,
                "mechanical_benchmark": None,
            }
        ),
        cc._encoded(
            {
                "rule": SOURCE_RULE,
                "spec_hash": c.spec_hash,
                "fields": fields,
                "known_types": types,
                "deliveries": deliveries,
                "definition": kwargs["causal_definition"],
                "unobserved_definition": kwargs["causal_unobserved_definition"],
                "row_limit": 2000000,
            }
        ),
        tuple(
            tuple((p.id, 205.0 if p.carrier == "A" else 305.0) for p in c.spec.products)
            for _ in range(3)
        ),
        tuple(
            m.SourceManifest(
                role,
                "declared unit recipe fingerprint, not producer authentication",
                hashlib.sha256((role + c.spec_hash).encode()).hexdigest(),
            )
            for role in m.MANIFEST_ROLES
        ),
        "complete declared static source recipe and fixed regular postings",
        "NEW136974-source-programme",
    )
    return c, scenario, kwargs, policy, programme, keys


def registration(prep, *, parts=None, **kwargs):
    _c, _scenario, _source_kwargs, _policy, programme, keys = prep
    return m.register_role_context(
        *(parts or laws()),
        selected_history=kwargs.pop("selected_history", "earlier"),
        person_key=kwargs.pop("person_key", keys[0]),
        actions=(
            cc.ContextAction("A", "A_OH0", False),
            cc.ContextAction("B", "B_OH0", False),
            cc.ContextAction("outside", None, True),
        ),
        programme=kwargs.pop("programme", programme),
        observation_definition="compatible G symbols only; actual G and awareness unknown",
        criterion=kwargs.pop("criterion", "terminal_utility"),
        support_limit=kwargs.pop("support_limit", LIMIT),
        **kwargs,
    )


@pytest.fixture(scope="module")
def fresh():
    prep = prepared()
    reg = registration(prep)  # Entire typed registration BEFORE tape/engine.
    c, scenario, kwargs, policy, programme, keys = prep
    tape = ExogenousTape.generate(
        c,
        31,
        registry=StreamRegistry(137974, "unit/V9_role_law_context/June21/fixture-v2"),
    )
    market = keys[0][2]
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
        ),
        gumbel=g,
    )
    e = NativeCausalInformationMarketEngine(scenario, **kwargs, dated_rebooking=policy)
    e.reset(tape=tape)
    for posting in programme.postings:
        prices = dict(posting)
        e.step(
            {
                carrier.code: PriceAction(
                    {j: prices[c.spec.products[j].id] for j in c.products_of[i]}
                )
                for i, carrier in enumerate(c.spec.carriers)
            }
        )
    periods = e.causal_periods()
    selected = next(row for p in periods for row in p.inputs if row.kind == "ordinary")
    source = native_causal_information_record(e)
    return prep, reg, e, source, periods, selected


def attach(fresh, reg=None, **kwargs):
    prep, original, _engine, source, periods, selected = fresh
    return m.attach_role_context(
        reg or original,
        kwargs.pop("source", source),
        kwargs.pop("periods", periods),
        kwargs.pop("selected", selected),
        actual_postings=kwargs.pop("actual_postings", prep[4].postings),
        actual_manifests=kwargs.pop("actual_manifests", prep[4].manifests),
        support_limit=kwargs.pop("support_limit", LIMIT),
        **kwargs,
    )


def test_new_complete_registration_before_genuine_source(fresh):
    out = attach(fresh)
    assert out.subjective_evidence == out.reference_evidence == F(1, 2)
    assert len(out.context.periods) == 3 and not out.context.periods[-1].inputs
    assert out.context.common_action == "B"
    assert (
        out.native_G is None
        and not out.native_law_knowledge_preferences_or_CS_authenticated
    )
    assert all(
        a.probability == 0 for a in out.subjective_conditional if a.history == "other"
    )
    assert sum(a.probability for a in out.subjective_conditional) == 1
    assert out.declared_evaluation.native_history_association is None


def test_independent_closed_form_candidate_and_fixed_reference(fresh):
    out = attach(fresh).declared_evaluation
    assert [c.action for c in out.solution.cells] == ["A", "B", None, None, None, None]
    assert out.evaluations[0].totals[0] == F(23, 6)
    assert out.evaluations[0].totals[2] == F(35, 12)
    assert out.evaluations[3].totals[0] == F(7, 6)
    assert out.evaluations[3].totals[2] == F(19, 12)
    for projections in (out.subjective_projections, out.reference_projections):
        assert sum(r.weight for r in projections) == 3


def test_earlier_payload_excludes_model_and_current_result(fresh):
    out = attach(fresh)
    view = m.earlier_role_context_view(out)
    assert view == cc.context_compatibility_view(out.context)
    assert view["known_type"]["value"] is None
    for word in (
        "registration",
        "actual_postings",
        "subjective",
        "scores",
        "probability",
        "native-input-",
        "common_action",
    ):
        assert word not in repr(view)
    view["scope"] = "mutated detached export"
    assert m.earlier_role_context_view(out)["scope"] != view["scope"]


@pytest.mark.parametrize("role", ["q", "r"])
def test_zero_upstream_event_keeps_null_no_backend(fresh, monkeypatch, role):
    p, q, r, desc = laws()
    law = q if role == "q" else r
    prior = tuple(
        replace(a, probability=F(0) if a.history == "earlier" else a.probability * 2)
        for a in law.prior
    )
    law = replace(law, prior=prior)
    reg = registration(
        fresh[0], parts=(p, law if role == "q" else q, law if role == "r" else r, desc)
    )

    def forbidden(*args):
        raise AssertionError("backend may not fabricate a unit prior")

    monkeypatch.setattr(utility, "execute_action_utility", forbidden)
    out = attach(fresh, reg)
    assert out.declared_evaluation is None
    assert (
        out.subjective_conditional if role == "q" else out.reference_conditional
    ) is None
    assert (
        out.reference_conditional if role == "q" else out.subjective_conditional
    ) is not None


def test_tiny_positive_unknown_and_zero_unknown_rows(fresh):
    p, q, r, desc = laws()
    eps = F(1, 10**2800)
    priors = tuple(
        replace(
            a,
            probability=(
                1 - eps
                if (a.state, a.history, a.vector) == ("robust", "earlier", "calm")
                else eps
                if (a.state, a.history, a.vector) == ("robust", "earlier", "tilt")
                else F(0)
            ),
        )
        for a in q.prior
    )
    rows = tuple(
        replace(a, account=replace(a.account, extra_utility=None))
        if a.vector in ("tilt", "zero") and a.action == "B"
        else a
        for a in q.outcomes
    )
    reg = registration(
        fresh[0], parts=(p, replace(q, prior=priors, outcomes=rows), r, desc)
    )
    out = attach(fresh, reg).declared_evaluation
    assert (
        out.solution.cells[1].evidence == eps and out.solution.cells[1].action is None
    )
    assert out.evaluations[0].totals[2] is None
    assert out.evaluations[3].missing_behavior_mass == F(1, 3)
    assert any(
        r.weight == 0 and r.values[2] is None and r.contributions[2] == 0
        for r in out.subjective_projections
    )


def test_source_current_result_does_not_choose_G_or_prior(fresh):
    p = fresh[0]
    first = attach(fresh)
    reg = registration(p, person_key=p[5][1])
    selected = next(
        r
        for period in fresh[4]
        for r in period.inputs
        if r.kind == "ordinary" and r.analyst_original_key == p[5][1]
    )
    second = attach(fresh, reg, selected=selected)
    assert first.subjective_conditional == second.subjective_conditional
    assert first.context.earlier_view == second.context.earlier_view
    assert first.declared_evaluation.solution == second.declared_evaluation.solution


@pytest.mark.parametrize(
    "change",
    [
        "drop-prior",
        "drop-outcome",
        "nonunit-unused",
        "bool-weight",
        "missing-outside",
        "missing-tail",
    ],
)
def test_complete_law_account_refusals(change):
    prep = prepared()
    p, q, r, desc = laws()
    if change == "drop-prior":
        q = replace(q, prior=q.prior[:-1])
    if change == "drop-outcome":
        q = replace(q, outcomes=q.outcomes[:-1])
    if change == "nonunit-unused":
        q = replace(
            q, outcomes=q.outcomes[:-1] + (replace(q.outcomes[-1], probability=F(0)),)
        )
    if change == "bool-weight":
        q = replace(q, prior=(replace(q.prior[0], probability=True),) + q.prior[1:])
    if change == "missing-outside":
        q = replace(
            q,
            outcomes=(
                replace(
                    q.outcomes[0],
                    same_path_outside=replace(
                        q.outcomes[0].same_path_outside, extra_source=""
                    ),
                ),
            )
            + q.outcomes[1:],
        )
    if change == "missing-tail":
        q = replace(
            q,
            outcomes=(
                replace(
                    q.outcomes[0],
                    account=replace(q.outcomes[0].account, closure_sources=("one",)),
                ),
            )
            + q.outcomes[1:],
        )
    with pytest.raises((TypeError, ValueError)):
        registration(prep, parts=(p, q, r, desc))


@pytest.mark.parametrize(
    "change",
    [
        "missing-context",
        "missing-causal",
        "missing-posting",
        "integer-fare",
        "missing-manifest",
        "bad-sha",
        "wrong-history",
        "bool-key",
    ],
)
def test_complete_programme_and_selector_refusals(change):
    prep = prepared()
    pr = prep[4]
    kw = {}
    if change == "missing-context":
        pr = replace(
            pr, context={k: v for k, v in pr.context.items() if k != "dated_reserve"}
        )
    if change == "missing-causal":
        pr = replace(
            pr, causal={k: v for k, v in pr.causal.items() if k != "known_types"}
        )
    if change == "missing-posting":
        pr = replace(pr, postings=pr.postings[:-1])
    if change == "integer-fare":
        pr = replace(
            pr,
            postings=(((pr.postings[0][0][0], 205),) + pr.postings[0][1:],)
            + pr.postings[1:],
        )
    if change == "missing-manifest":
        pr = replace(pr, manifests=pr.manifests[:-1])
    if change == "bad-sha":
        pr = replace(
            pr,
            manifests=(replace(pr.manifests[0], sha256="x" * 64),) + pr.manifests[1:],
        )
    if change == "wrong-history":
        kw["selected_history"] = "later-result"
    if change == "bool-key":
        kw["person_key"] = (False, 0, 0)
    with pytest.raises((TypeError, ValueError, KeyError)):
        registration(prep, programme=pr, **kw)


@pytest.mark.parametrize(
    "change",
    [
        "postings",
        "manifests",
        "source-programme",
        "empty-period",
        "person",
        "sequential",
    ],
)
def test_actual_attachment_cannot_change_owned_contract(fresh, change):
    kw = {}
    if change == "postings":
        kw["actual_postings"] = fresh[0][4].postings[:-1]
    if change == "manifests":
        kw["actual_manifests"] = (
            replace(fresh[0][4].manifests[0], sha256="0" * 64),
        ) + fresh[0][4].manifests[1:]
    if change == "source-programme":
        source = copy.deepcopy(fresh[3])
        source["programme"]["definition"] = "changed after source"
        kw["source"] = source
    if change == "empty-period":
        kw["periods"] = fresh[4][:-1]
    if change == "person":
        kw["selected"] = next(
            r
            for r in fresh[4][0].inputs
            if r.analyst_original_key != fresh[5].analyst_original_key
        )
    if change == "sequential":
        kw["selected"] = next(
            r for p in fresh[4] for r in p.inputs if r.kind != "ordinary"
        )
    with pytest.raises((TypeError, ValueError)):
        attach(fresh, **kw)


def test_full_bound_before_classifier_or_utility_forward(fresh, monkeypatch):
    good = attach(fresh)

    def forbidden(*args):
        raise AssertionError("not reached below complete support limit")

    monkeypatch.setattr(cc, "_classify", forbidden)
    monkeypatch.setattr(utility, "execute_action_utility", forbidden)
    with pytest.raises(ValueError):
        attach(fresh, support_limit=good.required_nodes - 1)


def test_reader_does_not_reclassify_or_resolve(fresh, monkeypatch):
    out = attach(fresh)

    def forbidden(*args):
        raise AssertionError("reader replayed a forward")

    monkeypatch.setattr(cc, "_classify", forbidden)
    monkeypatch.setattr(utility, "execute_action_utility", forbidden)
    assert m.earlier_role_context_view(out)["current_stage"] == "ordinary"


@pytest.mark.parametrize(
    "field,value",
    [
        ("digest", "forged"),
        ("required_nodes", True),
        ("native_G", "guessed"),
        ("native_law_knowledge_preferences_or_CS_authenticated", True),
    ],
)
def test_stored_integrity_and_unprovided_authentication(fresh, field, value):
    out = attach(fresh)
    with pytest.raises((TypeError, ValueError)):
        m.earlier_role_context_view(replace(out, **{field: value}))


def test_owned_exports_and_native_rng_identity(fresh):
    before = pickle.dumps(fresh[2].snapshot())
    rng = pickle.dumps(np.random.get_state())
    out = attach(fresh)
    export = out.export()
    export["registration"]["selected_history"] = "changed"
    export["context"]["earlier_view"]["scope"] = "changed"
    assert out.registration.selected_history == "earlier"
    assert m.earlier_role_context_view(out)["scope"] != "changed"
    assert (
        pickle.dumps(fresh[2].snapshot()) == before
        and pickle.dumps(np.random.get_state()) == rng
    )
    with pytest.raises(FrozenInstanceError):
        out.native_G = "invented"


def test_injected_pure_failure_leaves_inputs_no_partial(fresh, monkeypatch):
    before = pickle.dumps((fresh[1], fresh[2].snapshot()))

    def fail(*args):
        raise RuntimeError("NEW pure injected failure")

    monkeypatch.setattr(utility, "execute_action_utility", fail)
    with pytest.raises(RuntimeError, match="NEW pure"):
        attach(fresh)
    assert pickle.dumps((fresh[1], fresh[2].snapshot())) == before


def test_new_registration_limit_before_any_forward(monkeypatch):
    prep = prepared()
    good = registration(prep)

    def forbidden(*args):
        raise AssertionError("no registration probability/classification forward")

    monkeypatch.setattr(cc, "_classify", forbidden)
    monkeypatch.setattr(utility, "execute_action_utility", forbidden)
    with pytest.raises(ValueError):
        registration(prep, support_limit=good.required_nodes - 1)


def test_new_reference_never_reaches_subjective_solver(fresh, monkeypatch):
    calls = []
    original = utility._solve_subjective

    def observed(problem, subjective, projections, criterion):
        calls.append(subjective.name)
        assert subjective.name == "q"
        return original(problem, subjective, projections, criterion)

    monkeypatch.setattr(utility, "_solve_subjective", observed)
    out = attach(fresh)
    assert calls == ["q"]
    assert out.declared_evaluation.evaluations[3].law == "r"


def test_new_empty_period_postings_remain_owned(fresh):
    programme = fresh[0][4]
    last = list(programme.postings[-1])
    product, fare = last[0]
    last[0] = (product, fare + 1.0)
    with pytest.raises(ValueError):
        attach(fresh, actual_postings=programme.postings[:-1] + (tuple(last),))
