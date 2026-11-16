"""A.10/A.5/B0/V9/B4: NEW132974 finite declarations; no native/RNG fixtures."""

from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F
from inspect import signature

import pytest

import airline_market.sim.action_kernel_utility as m
from airline_market.sim.decision_consistency import (
    REFERENCE_SCOPE,
    DecisionAccount,
    DecisionAction,
    DecisionAtom,
    DeclaredDecisionPolicy,
    ObservedDecisionVector,
)

LIMIT = 10**8


def account(b, p=0, w=0, c=0, e=0):
    return DecisionAccount(
        tuple(None if v is None else F(v) for v in (b, p, w, c)),
        (
            "service declaration",
            "net payment declaration",
            "waiting declaration",
            "cancellation declaration",
        ),
        None if e is None else F(e),
        "explicit additional utility",
        True,
        True,
        True,
        ("financial tail", "material tail", "utility tail"),
        "NEW132974 complete account",
        "declared counterfactual model",
        "E excludes FOUR material components",
    )


def declarations():
    p = m.ActionKernelProblem(
        "NEW132974-person",
        "explicit known label",
        ("bright", "dark"),
        ("clear", "alert"),
        (
            ObservedDecisionVector(
                "steady", (F(0), F(3), F(0)), "declared observed vector"
            ),
            ObservedDecisionVector(
                "unused", (F(0), F(0), F(0)), "retained zero vector"
            ),
        ),
        ("travel", "protect", "outside"),
        "outside",
        ("good", "bad"),
        F(1, 2),
        "material token",
        "utility token",
        "one commitment",
        "earlier event conditioned ONCE",
        "H and G only; fixed known type",
        "X and future Y not observed",
        "material and E have distinct incidence",
        REFERENCE_SCOPE,
    )
    q, r = [], []
    for x in p.states:
        for h in p.histories:
            for g in p.vectors:
                high = (x == "bright") == (h == "clear")
                q.append(
                    DecisionAtom(
                        x,
                        h,
                        g.name,
                        F(3 if high else 1, 8) if g.name == "steady" else F(0),
                    )
                )
                r.append(
                    DecisionAtom(
                        x,
                        h,
                        g.name,
                        F(1 if high else 3, 8) if g.name == "steady" else F(0),
                    )
                )
    rows = []
    for atom in q:
        for a in p.actions:
            good = (
                (F(3, 4) if atom.state == "bright" else F(1, 4))
                if a == "travel"
                else F(1, 2)
                if a == "protect"
                else F(1, 8)
            )
            for y in p.outcomes:
                acc = (
                    account(12 if y == "good" else -4, p=2)
                    if a == "travel"
                    else account(8 if y == "good" else 1, p=1, e=-1)
                    if a == "protect"
                    else account(3 if y == "good" else 1)
                )
                rows.append(
                    m.ActionOutcome(
                        atom.state,
                        atom.history,
                        atom.vector,
                        a,
                        y,
                        good if y == "good" else 1 - good,
                        acc,
                        account(1),
                    )
                )
    desc = DeclaredDecisionPolicy(
        "descriptive",
        "supplied behavior without optimality claim",
        tuple(
            DecisionAction(h, g.name, "travel") for h in p.histories for g in p.vectors
        ),
    )
    return (
        p,
        m.ActionChanceLaw("q", "subjective law", tuple(q), tuple(rows)),
        m.ActionChanceLaw("r", "reference law", tuple(r), tuple(rows)),
        desc,
    )


def register(parts=None, **kwargs):
    return m.register_action_utility(
        *(parts or declarations()),
        criterion=kwargs.pop("criterion", "terminal_utility"),
        support_limit=kwargs.pop("support_limit", LIMIT),
        **kwargs,
    )


def test_closed_form_candidate_and_fixed_reference():
    out = m.execute_action_utility(register())
    assert [c.action for c in out.solution.cells] == ["travel", None, "protect", None]
    assert out.solution.cells[0].posterior == (F(3, 4), F(1, 4))
    assert out.solution.cells[0].terminal_scores == (F(2), F(3, 4), F(5, 8))
    assert out.solution.cells[2].terminal_scores == (F(0), F(3, 4), F(5, 8))
    assert out.evaluations[3].totals[:3] == (F(7, 4), F(7, 8), F(3, 8))
    assert out.native_history_association is None
    assert out.native_preferences_or_lifecycle_CS_adopted is False


def test_independent_full_path_reference_sum():
    reg = register()
    out = m.execute_action_utility(reg)
    pi = {(a.history, a.vector): a.action for a in out.solution.policy.actions}
    expected = F(0)
    for prior in reg.reference.prior:
        for row in reg.reference.outcomes:
            if (row.state, row.history, row.vector, row.action) != (
                prior.state,
                prior.history,
                prior.vector,
                pi[prior.history, prior.vector],
            ):
                continue
            b, p, w, c = row.account.components
            expected += (
                prior.probability
                * row.probability
                * (reg.problem.alpha * (b - p - w - c) + row.account.extra_utility)
            )
    assert out.evaluations[3].totals[2] == expected == F(3, 8)


def test_separate_criterion_and_descriptive_behavior():
    out = m.execute_action_utility(register(criterion="material_plus_decision_noise"))
    assert [c.action for c in out.solution.cells] == ["protect", None, "protect", None]
    assert all(a.action == "travel" for a in out.registration.descriptive.actions)
    assert out.evaluations[1].policy == out.registration.descriptive


def test_solver_sees_no_reference_and_reference_does_not_resolve(monkeypatch):
    assert "reference" not in signature(m._solve_subjective).parameters
    calls = []
    original = m._solve_subjective

    def spy(problem, subjective, projections, criterion):
        calls.append(subjective.name)
        return original(problem, subjective, projections, criterion)

    monkeypatch.setattr(m, "_solve_subjective", spy)
    out = m.execute_action_utility(register())
    assert calls == ["q"]
    assert out.evaluations[3].policy == out.solution.policy


def test_action_blocks_are_not_global_joint_or_normalized():
    out = m.execute_action_utility(register())
    assert sum(r.weight for r in out.subjective_projections) == 3
    for a in out.registration.problem.actions:
        assert sum(r.weight for r in out.subjective_projections if r.action == a) == 1
    assert all(sum(r.weight for r in e.rows) == 1 for e in out.evaluations)
    assert len(out.reference_projections) == 48


def test_same_path_and_separate_outside_policy_differ():
    out = m.execute_action_utility(register())
    compare = out.comparisons[2]
    assert compare.same_selected_path_normalized == F(-1, 8)
    assert compare.experienced_difference == F(-1, 4)
    assert out.evaluations[5].totals[2] == F(5, 8)


def test_subjective_gap_positive_reference_mass_is_once_not_times_actions():
    p, q, r, d = declarations()
    q = replace(
        q,
        prior=tuple(
            replace(
                a,
                probability=F(1, 2)
                if a.history == "clear" and a.vector == "steady"
                else F(0),
            )
            for a in q.prior
        ),
    )
    out = m.execute_action_utility(register((p, q, r, d)))
    e = out.evaluations[3]
    assert e.missing_behavior_mass == F(1, 2)
    assert e.undefined_weights == (F(1, 2),) * 4
    assert e.totals == (None,) * 4
    assert out.solution.cells[2].posterior is None
    assert sum(r.weight for r in e.rows if r.action is None) == F(1, 2)


def change_rows(law, predicate, transform):
    return replace(
        law, outcomes=tuple(transform(r) if predicate(r) else r for r in law.outcomes)
    )


def test_tiny_positive_unknown_extra_and_zero_unknown():
    p, q, r, d = declarations()
    tiny = F(1, 10**2400)

    def change(row):
        prob = tiny if row.outcome == "bad" else 1 - tiny
        return replace(
            row,
            probability=prob,
            account=replace(row.account, extra_utility=None)
            if row.outcome == "bad"
            else row.account,
        )

    q = change_rows(q, lambda row: row.action == "travel", change)
    out = m.execute_action_utility(register((p, q, r, d)))
    assert out.solution.cells[0].terminal_scores[0] is None
    assert out.solution.cells[0].action is None
    q = change_rows(
        q,
        lambda row: row.action == "travel",
        lambda row: replace(row, probability=F(row.outcome == "good")),
    )
    zero = m.execute_action_utility(register((p, q, r, d)))
    assert zero.solution.cells[0].terminal_scores[0] == 5
    assert any(
        row.values[2] is None and row.contributions[2] == 0
        for row in zero.subjective_projections
    )


@pytest.mark.parametrize(
    "closure,metric",
    [("financial_closed", 0), ("material_closed", 0), ("utility_closed", 2)],
)
def test_independent_open_tails(closure, metric):
    p, q, r, d = declarations()
    r = change_rows(
        r,
        lambda row: row.action == "travel",
        lambda row: replace(row, account=replace(row.account, **{closure: False})),
    )
    out = m.execute_action_utility(register((p, q, r, d)))
    assert out.evaluations[4].totals[metric] is None


def test_unknown_outside_is_not_zero_and_unknown_minus_unknown():
    p, q, r, d = declarations()
    r = change_rows(
        r, lambda row: True, lambda row: replace(row, same_path_outside=account(None))
    )
    r = change_rows(
        r,
        lambda row: row.action == "outside",
        lambda row: replace(row, account=account(None)),
    )
    out = m.execute_action_utility(register((p, q, r, d)))
    assert out.evaluations[3].totals[2] == F(3, 8)
    assert out.evaluations[3].totals[3] is None
    assert out.comparisons[2].experienced_difference is None
    assert out.evaluations[5].totals[3] is None


def test_tiny_positive_evidence_no_float_conversion():
    p, q, r, d = declarations()
    tiny = F(1, 10**2450)
    q = replace(
        q,
        prior=tuple(
            replace(
                a,
                probability=(tiny / 2 if a.history == "alert" else (1 - tiny) / 2)
                if a.vector == "steady"
                else F(0),
            )
            for a in q.prior
        ),
    )
    out = m.execute_action_utility(register((p, q, r, d)))
    assert out.solution.cells[2].evidence == tiny
    assert out.solution.cells[2].posterior == (F(1, 2), F(1, 2))


def test_reference_changes_do_not_change_subjective_policy():
    p, q, r, d = declarations()
    different = change_rows(
        r, lambda row: True, lambda row: replace(row, account=account(-(10**80)))
    )
    a = m.execute_action_utility(register((p, q, r, d)))
    b = m.execute_action_utility(register((p, q, different, d)))
    assert a.solution == b.solution
    assert a.evaluations[3].totals != b.evaluations[3].totals


def test_same_direct_projection_different_alternative_law_changes_candidate():
    p, q, r, d = declarations()
    other = change_rows(
        q,
        lambda row: row.action == "protect",
        lambda row: replace(row, account=account(100)),
    )
    a = m.execute_action_utility(register((p, q, r, d)))
    b = m.execute_action_utility(register((p, other, r, d)))
    assert tuple(
        row for row in a.subjective_projections if row.action == "travel"
    ) == tuple(row for row in b.subjective_projections if row.action == "travel")
    assert a.solution.policy != b.solution.policy


def test_first_declared_tie_and_signed_huge_tiny_values():
    p, q, r, d = declarations()
    q = change_rows(
        q,
        lambda row: True,
        lambda row: replace(row, account=account(-(10**60), e=F(1, 10**300))),
    )
    out = m.execute_action_utility(register((p, q, r, d)))
    assert out.solution.cells[0].action == "travel"
    assert (
        out.solution.cells[0].terminal_scores == (F(-(10**60), 2) + F(1, 10**300),) * 3
    )


def test_owned_inputs_exports_and_failure(monkeypatch):
    p, q, r, d = declarations()
    source = list(q.outcomes)
    reg = register((p, replace(q, outcomes=source), r, d))
    source.clear()
    before = reg.export()
    out = m.execute_action_utility(reg)
    exported = out.export()
    exported["registration"]["subjective"]["outcomes"].clear()
    assert len(out.registration.subjective.outcomes) == 48
    with pytest.raises(FrozenInstanceError):
        out.rule = "changed"

    def crash(*args):
        raise RuntimeError("NEW132974 injected pure forward failure")

    monkeypatch.setattr(m, "_project", crash)
    with pytest.raises(RuntimeError, match="injected"):
        m.execute_action_utility(reg)
    assert reg.export() == before


def test_full_input_output_bound_and_limit_before_forward(monkeypatch):
    reg = register()
    out = m.execute_action_utility(reg)
    assert m._nodes({"input": reg, "output": out}) <= reg.required_nodes
    assert (
        register(support_limit=reg.required_nodes).required_nodes == reg.required_nodes
    )

    def no_forward(*args):
        pytest.fail("limit must be checked BEFORE forward")

    monkeypatch.setattr(m, "_project", no_forward)
    with pytest.raises(ValueError, match="exceeds"):
        register(support_limit=reg.required_nodes - 1)
    with pytest.raises(ValueError, match="exceeds"):
        m.execute_action_utility(replace(reg, support_limit=reg.required_nodes - 1))


@pytest.mark.parametrize(
    "field,value",
    [
        ("digest", "wrong"),
        ("required_nodes", 1),
        ("required_nodes", True),
        ("rule", "old"),
    ],
)
def test_stored_registration_integrity(field, value):
    with pytest.raises(ValueError):
        m.execute_action_utility(replace(register(), **{field: value}))


@pytest.mark.parametrize(
    "bad",
    [
        "missing_prior",
        "duplicate_prior",
        "negative_prior",
        "float_prior",
        "bad_mass",
        "missing_kernel",
        "duplicate_kernel",
        "negative_kernel",
        "global_kernel_mass",
        "missing_account",
        "bool_closure",
        "empty_source",
        "missing_policy",
        "hidden_policy",
        "bad_vector",
        "units",
        "alpha",
        "reference_scope",
        "same_role_names",
        "criterion",
    ],
)
def test_complete_declaration_refusals(bad):
    p, q, r, d = declarations()
    criterion = "terminal_utility"
    if bad == "missing_prior":
        q = replace(q, prior=q.prior[:-1])
    elif bad == "duplicate_prior":
        q = replace(q, prior=q.prior + (q.prior[0],))
    elif bad == "negative_prior":
        q = replace(q, prior=(replace(q.prior[0], probability=F(-1)),) + q.prior[1:])
    elif bad == "float_prior":
        q = replace(q, prior=(replace(q.prior[0], probability=0.1),) + q.prior[1:])
    elif bad == "bad_mass":
        q = replace(q, prior=(replace(q.prior[0], probability=F(1)),) + q.prior[1:])
    elif bad == "missing_kernel":
        q = replace(q, outcomes=q.outcomes[:-1])
    elif bad == "duplicate_kernel":
        q = replace(q, outcomes=q.outcomes + (q.outcomes[0],))
    elif bad == "negative_kernel":
        q = replace(
            q, outcomes=(replace(q.outcomes[0], probability=F(-1)),) + q.outcomes[1:]
        )
    elif bad == "global_kernel_mass":
        q = replace(
            q,
            outcomes=tuple(
                replace(v, probability=v.probability / 3) for v in q.outcomes
            ),
        )
    elif bad == "missing_account":
        q = replace(
            q,
            outcomes=(replace(q.outcomes[0], account=account(None)),) + q.outcomes[1:],
        )
        q = replace(
            q,
            outcomes=(
                replace(
                    q.outcomes[0],
                    account=replace(q.outcomes[0].account, components=(F(1),)),
                ),
            )
            + q.outcomes[1:],
        )
    elif bad == "bool_closure":
        q = replace(
            q,
            outcomes=(
                replace(
                    q.outcomes[0],
                    account=replace(q.outcomes[0].account, financial_closed=1),
                ),
            )
            + q.outcomes[1:],
        )
    elif bad == "empty_source":
        q = replace(
            q,
            outcomes=(
                replace(
                    q.outcomes[0],
                    account=replace(q.outcomes[0].account, extra_source=""),
                ),
            )
            + q.outcomes[1:],
        )
    elif bad == "missing_policy":
        d = replace(d, actions=d.actions[:-1])
    elif bad == "hidden_policy":
        d = replace(
            d, actions=d.actions + (DecisionAction("bright", "steady", "travel"),)
        )
    elif bad == "bad_vector":
        p = replace(p, vectors=(replace(p.vectors[0], action_noise=(F(0),)),))
    elif bad == "units":
        p = replace(p, utility_unit=p.material_unit)
    elif bad == "alpha":
        p = replace(p, alpha=F(0))
    elif bad == "reference_scope":
        p = replace(p, reference_scope="outside intervention")
    elif bad == "same_role_names":
        r = replace(r, name=q.name)
    elif bad == "criterion":
        criterion = "implicit native preference"
    with pytest.raises((ValueError, TypeError)):
        register((p, q, r, d), criterion=criterion)


def test_opposite_sign_same_path_and_separate_outside_comparisons():
    p, q, r, d = declarations()
    r = change_rows(
        r, lambda row: True, lambda row: replace(row, same_path_outside=account(-5))
    )
    out = m.execute_action_utility(register((p, q, r, d)))
    assert out.comparisons[2].same_selected_path_normalized == F(23, 8)
    assert out.comparisons[2].experienced_difference == F(-1, 4)


def test_common_terminal_translation_preserves_both_comparisons():
    p, q, r, d = declarations()

    def shift(row):
        return replace(
            row,
            account=replace(row.account, extra_utility=row.account.extra_utility + 7),
            same_path_outside=replace(
                row.same_path_outside,
                extra_utility=row.same_path_outside.extra_utility + 7,
            ),
        )

    original = m.execute_action_utility(register((p, q, r, d)))
    translated = m.execute_action_utility(
        register(
            (
                p,
                change_rows(q, lambda row: True, shift),
                change_rows(r, lambda row: True, shift),
                d,
            )
        )
    )
    assert original.solution.policy == translated.solution.policy
    assert original.comparisons == translated.comparisons
    for a, b in zip(original.evaluations, translated.evaluations):
        assert b.totals[2] == a.totals[2] + 7
        assert b.totals[0] == a.totals[0]


def test_tiny_open_financial_tail_reference_null_without_changing_subjective_policy():
    p, q, r, d = declarations()
    tiny = F(1, 10**2500)

    def change(row):
        return replace(
            row,
            probability=tiny if row.outcome == "bad" else 1 - tiny,
            account=replace(row.account, financial_closed=False)
            if row.outcome == "bad"
            else row.account,
        )

    r = change_rows(r, lambda row: row.action == "travel", change)
    out = m.execute_action_utility(register((p, q, r, d)))
    assert out.solution.cells[0].action == "travel"
    assert out.evaluations[3].totals[0] is None
    assert out.evaluations[3].undefined_weights[0] == tiny / 2
    assert out.evaluations[3].known_numerators[0] > 0
