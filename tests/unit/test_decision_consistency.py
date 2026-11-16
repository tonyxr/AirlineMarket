"""A.10/A.5/B0/V9/B4: NEW130974 finite declarations, no native seed/case replay."""

import inspect
import json
from dataclasses import FrozenInstanceError, replace
from fractions import Fraction as F

import pytest

import airline_market.sim.decision_consistency as dc
from airline_market.sim.decision_consistency import (
    REFERENCE_SCOPE,
    DecisionAccount,
    DecisionAction,
    DecisionAtom,
    DecisionContingency,
    DecisionReferenceLaw,
    DeclaredDecisionPolicy,
    DeclaredDecisionProblem,
    ObservedDecisionVector,
    evaluate_fixed_decision,
    fix_declared_decision_policy,
    solve_declared_decision,
)

LIMIT = 1_000_000


def problem():
    states = ("ready", "disrupted")
    histories = ("clear", "alert", "off_path")
    vectors = (
        ObservedDecisionVector(
            "steady", (F(1), F(0), F(1, 2)), "NEW130974 finite observed vector"
        ),
        ObservedDecisionVector(
            "skeptical",
            (F(-1, 2), F(3, 4), F(0)),
            "Correlated physical/private declaration",
        ),
        ObservedDecisionVector(
            "retained_zero", (F(9), F(-4), F(2)), "Zero marginal, full retained support"
        ),
    )
    actions = ("standard", "insured", "outside")
    atoms, contexts = [], []
    for xi, x in enumerate(states):
        for hi, h in enumerate(histories):
            for vi, v in enumerate(vectors):
                vector_mass = (F(1, 2), F(1, 2), F(0))[vi]
                signal_mass = (
                    (F(3, 4), F(1, 4), F(0)),
                    (F(1, 2), F(1, 2), F(0)),
                    (F(0), F(1), F(0)),
                )[vi][hi]
                ready = (
                    (F(4, 5), F(1, 5), F(1, 7)),
                    (F(1, 3), F(1, 2), F(1, 11)),
                    (F(1), F(0), F(1, 13)),
                )[vi][hi]
                atoms.append(
                    DecisionAtom(
                        x,
                        h,
                        v.name,
                        vector_mass * signal_mass * (ready if xi == 0 else 1 - ready),
                    )
                )
                accounts = []
                for ai in range(3):
                    components = (
                        ((91, 31, 4, 2), (0, 0, 6, 9))[xi]
                        if ai == 0
                        else ((79, 39, 2, 1), (0, -3, 3, 5))[xi]
                        if ai == 1
                        else (9, 1, 1, 1)
                    )
                    extra = v.action_noise[ai] * (
                        -2 if ai == 0 and xi else F(1, 2) if ai == 1 and xi else 1
                    )
                    accounts.append(
                        DecisionAccount(
                            tuple(map(F, components)),
                            (
                                "NEW130974:B",
                                "NEW130974:P-once",
                                "NEW130974:W",
                                "NEW130974:C",
                            ),
                            extra,
                            "NEW130974:E",
                            True,
                            True,
                            True,
                            (
                                "financial-followup",
                                "material-followup",
                                "utility-followup",
                            ),
                            "Explicit full original-person account",
                            "Declared terminal utility origin",
                            "B excludes E; no carrier-penalty incidence",
                        )
                    )
                contexts.append(DecisionContingency(x, h, v.name, tuple(accounts)))
    return DeclaredDecisionProblem(
        "NEW130974-not-native-key",
        "fixed-known-type-not-native-class",
        states,
        histories,
        vectors,
        actions,
        "outside",
        tuple(atoms),
        tuple(contexts),
        F(1, 9),
        "declared-material",
        "declared-utility",
        "One NEW finite commitment",
        "Before decision under full joint signal/private-vector law",
        "Full permitted signal and observed vector; same hidden-state cell action",
        "Physical state and reference law remain analyst-only",
        "B excludes E; material excludes all decision noise",
        REFERENCE_SCOPE,
    )


def reference(p, *, off_path=False):
    atoms = []
    for x in p.states:
        for h in p.histories:
            for vi, v in enumerate(p.vectors):
                vg = (F(2, 5), F(3, 5), F(0))[vi]
                hs = (
                    (F(1, 3), F(1, 3), F(1, 3))
                    if off_path
                    else (F(1, 2), F(1, 2), F(0))
                )
                atoms.append(
                    DecisionAtom(
                        x,
                        h,
                        v.name,
                        vg
                        * hs[p.histories.index(h)]
                        * (F(1, 4) if x == "ready" else F(3, 4)),
                    )
                )
    return DecisionReferenceLaw(
        "NEW130974 reference", "Separate analyst law; not external truth", tuple(atoms)
    )


def subjective_reference(p):
    return DecisionReferenceLaw(
        "subjective-as-explicit-reference",
        "Same saved joint for arithmetic check",
        p.subjective_joint,
    )


def descriptive(p, action="standard"):
    return DeclaredDecisionPolicy(
        "NEW130974 supplied descriptive",
        "Complete supplied policy, no optimality claim",
        tuple(
            DecisionAction(h, v.name, action) for h in p.histories for v in p.vectors
        ),
    )


def solve(p=None, criterion="terminal_utility"):
    return solve_declared_decision(
        problem() if p is None else p, criterion=criterion, support_limit=LIMIT
    )


def change_account(p, coord, ai, **kwargs):
    contexts = list(p.contingencies)
    for ci, c in enumerate(contexts):
        if (c.state, c.history, c.vector) == coord:
            accounts = list(c.accounts)
            accounts[ai] = replace(accounts[ai], **kwargs)
            contexts[ci] = replace(c, accounts=tuple(accounts))
    return replace(p, contingencies=tuple(contexts))


def rare_problem(exponent, *, unknown="E"):
    p = problem()
    eps = F(0) if exponent is None else F(1, 10**exponent)
    law = tuple(
        replace(a, probability=F(3, 8) * (eps if a.state == "disrupted" else 1 - eps))
        if (a.history, a.vector) == ("clear", "steady")
        else a
        for a in p.subjective_joint
    )
    p = replace(p, subjective_joint=law)
    coord = "disrupted", "clear", "steady"
    if unknown == "E":
        return change_account(p, coord, 0, extra_utility=None, utility_closed=False)
    return change_account(
        p, coord, 0, components=(None, F(0), F(6), F(9)), material_closed=False
    )


def test_new_joint_conditions_on_observed_vector_and_criterion_differs():
    p = problem()
    s = solve(p)
    cells = {(c.history, c.vector): c for c in s.cells}
    assert cells["clear", "steady"].posterior == (F(4, 5), F(1, 5))
    assert cells["clear", "skeptical"].posterior == (F(1, 3), F(2, 3))
    assert cells["clear", "steady"].experienced_scores[0] == F(73, 15)
    assert cells["alert", "skeptical"].action == "standard"
    assert (
        solve(p, "material_plus_decision_noise").fixed_policy.policy.actions[4].action
        == "insured"
    )
    assert cells["off_path", "steady"].posterior is None
    assert len(s.cells) == 9 and len(s.fixed_policy.problem.contingencies) == 18


def test_new_full_path_reference_independent_component_arithmetic():
    p = problem()
    fixed = solve(p).fixed_policy
    r = reference(p)
    result = evaluate_fixed_decision(fixed, r, support_limit=LIMIT)
    selected = {(a.history, a.vector): a.action for a in fixed.policy.actions}
    accounts = {(c.state, c.history, c.vector): c.accounts for c in p.contingencies}
    expected = [F(0)] * 4
    for atom in r.joint:
        action = selected[atom.history, atom.vector]
        if atom.probability == 0:
            continue
        current = accounts[atom.state, atom.history, atom.vector][
            p.actions.index(action)
        ]
        outside = accounts[atom.state, atom.history, atom.vector][2]
        b, paid, wait, cancel = current.components
        z = b - paid - wait - cancel
        ob, op, ow, oc = outside.components
        z0 = ob - op - ow - oc
        values = (
            z,
            p.alpha * z,
            p.alpha * z + current.extra_utility,
            p.alpha * (z - z0) + current.extra_utility - outside.extra_utility,
        )
        for i, v in enumerate(values):
            expected[i] += atom.probability * v
    assert result.totals == tuple(expected)
    assert (
        result.known_numerators == result.totals
        and result.undefined_weights == (F(0),) * 4
    )
    assert len(result.rows) == 18 and sum(r.weight for r in result.rows) == 1


def test_reference_evaluation_never_calls_subjective_solver_or_cell(monkeypatch):
    p = problem()
    fixed = solve(p).fixed_policy
    before = fixed.export()

    def forbidden(*args, **kwargs):
        raise AssertionError("subjective solving replayed")

    monkeypatch.setattr(dc, "solve_declared_decision", forbidden)
    monkeypatch.setattr(dc, "_cell", forbidden)
    one = evaluate_fixed_decision(fixed, reference(p), support_limit=LIMIT)
    two = evaluate_fixed_decision(fixed, subjective_reference(p), support_limit=LIMIT)
    assert one.fixed_policy.export() == two.fixed_policy.export() == before
    assert one.totals != two.totals
    assert "reference" not in inspect.signature(solve_declared_decision).parameters


def test_zero_subjective_positive_reference_reach_has_no_behavior_fallback():
    p = problem()
    fixed = solve(p).fixed_policy
    evaluation = evaluate_fixed_decision(
        fixed, reference(p, off_path=True), support_limit=LIMIT
    )
    assert evaluation.totals == (None,) * 4
    assert all(v > 0 for v in evaluation.undefined_weights)
    supplied = fix_declared_decision_policy(p, descriptive(p), support_limit=LIMIT)
    known = evaluate_fixed_decision(
        supplied, reference(p, off_path=True), support_limit=LIMIT
    )
    assert all(v is not None for v in known.totals)
    assert all(
        a.action is None for a in fixed.policy.actions if a.history == "off_path"
    )


def test_tiny_unknown_extra_keeps_noise_material_but_not_experience():
    p = rare_problem(2000)
    terminal = solve(p)
    assert terminal.cells[0].action is None
    noisy = solve(p, "material_plus_decision_noise")
    assert noisy.cells[0].action == "standard"
    evaluation = evaluate_fixed_decision(
        noisy.fixed_policy, subjective_reference(p), support_limit=LIMIT
    )
    assert evaluation.totals[0] is not None and evaluation.totals[1] is not None
    assert evaluation.totals[2:] == (None, None)
    assert evaluation.undefined_weights[2] == F(3, 8 * 10**2000)
    rare = next(
        r
        for r in evaluation.rows
        if (r.state, r.history, r.vector) == ("disrupted", "clear", "steady")
    )
    assert rare.weight > 0 and rare.contributions[2] is None
    assert evaluation.known_numerators[2] is not None


def test_tiny_unknown_material_prevents_complete_ranking_and_integral():
    p = rare_problem(2050, unknown="B")
    assert solve(p).cells[0].action is None
    assert solve(p, "material_plus_decision_noise").cells[0].action is None
    supplied = fix_declared_decision_policy(p, descriptive(p), support_limit=LIMIT)
    assert (
        evaluate_fixed_decision(
            supplied, subjective_reference(p), support_limit=LIMIT
        ).totals
        == (None,) * 4
    )


def test_exact_zero_unknown_retained_without_filling_or_pruning():
    p = rare_problem(None)
    fixed = solve(p).fixed_policy
    evaluation = evaluate_fixed_decision(
        fixed, subjective_reference(p), support_limit=LIMIT
    )
    assert all(v is not None for v in evaluation.totals)
    rare = next(
        r
        for r in evaluation.rows
        if (r.state, r.history, r.vector) == ("disrupted", "clear", "steady")
    )
    assert rare.values[2] is None and rare.contributions[2] == 0 and rare.weight == 0
    assert len(evaluation.rows) == len(p.subjective_joint)
    assert any(
        v.probability == 0 and v.vector == "retained_zero"
        for v in evaluation.reference.joint
    )


@pytest.mark.parametrize(
    "closure,expected",
    [
        ("financial_closed", (None,) * 4),
        ("material_closed", (None,) * 4),
        ("utility_closed", (F(54), F(6), None, None)),
    ],
)
def test_independent_closures(closure, expected):
    p = change_account(problem(), ("ready", "clear", "steady"), 0, **{closure: False})
    fixed = fix_declared_decision_policy(p, descriptive(p), support_limit=LIMIT)
    evaluation = evaluate_fixed_decision(
        fixed, subjective_reference(p), support_limit=LIMIT
    )
    row = next(
        r
        for r in evaluation.rows
        if (r.state, r.history, r.vector) == ("ready", "clear", "steady")
    )
    assert row.values == expected


def test_unknown_outside_minus_itself_is_unknown():
    p = change_account(problem(), ("ready", "clear", "steady"), 2, extra_utility=None)
    fixed = fix_declared_decision_policy(
        p, descriptive(p, "outside"), support_limit=LIMIT
    )
    evaluation = evaluate_fixed_decision(
        fixed, subjective_reference(p), support_limit=LIMIT
    )
    assert evaluation.totals[0] == 6 and evaluation.totals[2:] == (None, None)


def test_common_terminal_translation_preserves_choices_and_reference_differences():
    p = problem()
    shift = F(111, 17)
    translated = replace(
        p,
        contingencies=tuple(
            replace(
                c,
                accounts=tuple(
                    replace(a, extra_utility=a.extra_utility + shift)
                    for a in c.accounts
                ),
            )
            for c in p.contingencies
        ),
    )
    a, b = solve(p), solve(translated)
    assert a.fixed_policy.policy == b.fixed_policy.policy
    x = evaluate_fixed_decision(a.fixed_policy, reference(p), support_limit=LIMIT)
    y = evaluate_fixed_decision(
        b.fixed_policy, reference(translated), support_limit=LIMIT
    )
    assert x.totals[0:2] == y.totals[0:2] and x.totals[3] == y.totals[3]
    assert y.totals[2] - x.totals[2] == shift


def test_signed_huge_tiny_values_remain_exact_and_owned():
    p = change_account(
        problem(),
        ("ready", "clear", "steady"),
        0,
        components=(F(10**700), F(-(10**650)), F(1, 10**2200), F(-3)),
        extra_utility=F(-(10**450)),
    )
    fixed = fix_declared_decision_policy(p, descriptive(p), support_limit=LIMIT)
    evaluation = evaluate_fixed_decision(
        fixed, subjective_reference(p), support_limit=LIMIT
    )
    row = next(
        r
        for r in evaluation.rows
        if (r.state, r.history, r.vector) == ("ready", "clear", "steady")
    )
    assert row.values[0] == F(10**700 + 10**650 + 3) - F(1, 10**2200)
    assert json.loads(json.dumps(evaluation.export()))["rows"][0]["values"][0] == str(
        row.values[0]
    )


def test_tie_break_is_first_declared_action():
    p = problem()
    equal = p.contingencies[0].accounts[2]
    p = replace(
        p,
        vectors=tuple(replace(v, action_noise=(F(0),) * 3) for v in p.vectors),
        contingencies=tuple(replace(c, accounts=(equal,) * 3) for c in p.contingencies),
    )
    assert solve(p).cells[0].action == "standard"
    assert solve(p, "material_plus_decision_noise").cells[0].action == "standard"


def test_ownership_mutable_inputs_exports_and_immutable_results():
    p = problem()
    components = list(p.contingencies[0].accounts[0].components)
    p = change_account(p, ("ready", "clear", "steady"), 0, components=components)
    source_joint = list(p.subjective_joint)
    p = replace(p, subjective_joint=source_joint)
    fixed = solve(p).fixed_policy
    saved = fixed.export()
    components[0] = F(-999)
    source_joint.clear()
    exported = fixed.export()
    exported["problem"]["states"].clear()
    assert fixed.export() == saved
    with pytest.raises(FrozenInstanceError):
        fixed.origin = "changed"


def test_canonical_full_support_order_and_owned_reference():
    p = problem()
    reversed_p = replace(
        p,
        subjective_joint=p.subjective_joint[::-1],
        contingencies=p.contingencies[::-1],
    )
    assert solve(p).export() == solve(reversed_p).export()
    fixed = solve(p).fixed_policy
    atoms = list(reference(p).joint)
    r = replace(reference(p), joint=atoms)
    result = evaluate_fixed_decision(fixed, r, support_limit=LIMIT)
    before = result.export()
    atoms.clear()
    assert result.export() == before


@pytest.mark.parametrize(
    "change",
    [
        lambda p: replace(p, subjective_joint=p.subjective_joint[:-1]),
        lambda p: replace(
            p,
            subjective_joint=(replace(p.subjective_joint[0], probability=F(-1)),)
            + p.subjective_joint[1:],
        ),
        lambda p: replace(
            p,
            subjective_joint=(
                replace(
                    p.subjective_joint[0],
                    probability=p.subjective_joint[0].probability + F(1, 10),
                ),
            )
            + p.subjective_joint[1:],
        ),
        lambda p: replace(p, contingencies=p.contingencies[:-1]),
        lambda p: replace(p, alpha=F(0)),
        lambda p: replace(p, alpha=True),
        lambda p: replace(p, utility_unit=p.material_unit),
        lambda p: replace(p, outside_action="missing"),
        lambda p: replace(p, reference_scope="outside-intervention"),
        lambda p: change_account(
            p, ("ready", "clear", "steady"), 0, financial_closed=1
        ),
        lambda p: change_account(
            p, ("ready", "clear", "steady"), 0, component_sources=("B",)
        ),
        lambda p: change_account(p, ("ready", "clear", "steady"), 0, extra_utility=0.0),
        lambda p: replace(
            p, vectors=tuple(replace(v, action_noise=(F(0),)) for v in p.vectors)
        ),
        lambda p: replace(p, subjective_joint=iter(p.subjective_joint)),
    ],
)
def test_complete_invalid_inputs_fail_without_partial_forward(change, monkeypatch):
    p = change(problem())

    def forbidden(*args, **kwargs):
        raise AssertionError("forward before validation")

    monkeypatch.setattr(dc, "_cell", forbidden)
    with pytest.raises((ValueError, TypeError)):
        solve(p)


def test_full_policy_and_reference_support_are_required():
    p = problem()
    with pytest.raises(ValueError):
        fix_declared_decision_policy(
            p,
            replace(descriptive(p), actions=descriptive(p).actions[:-1]),
            support_limit=LIMIT,
        )
    with pytest.raises(ValueError):
        evaluate_fixed_decision(
            solve(p).fixed_policy,
            replace(reference(p), joint=reference(p).joint[:-1]),
            support_limit=LIMIT,
        )
    with pytest.raises(ValueError):
        solve(p, "undeclared")


def test_complete_structural_bounds_checked_before_forwards(monkeypatch):
    p = problem()
    solved = solve(p)
    evaluated = evaluate_fixed_decision(
        solved.fixed_policy, reference(p), support_limit=LIMIT
    )
    assert (
        dc._nodes({"input": solved.fixed_policy.problem, "output": solved})
        <= solved.required_nodes
    )
    assert (
        dc._nodes(
            {
                "input_fixed": solved.fixed_policy,
                "input_reference": evaluated.reference,
                "output": evaluated,
            }
        )
        == evaluated.required_nodes
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("forward after limit")

    monkeypatch.setattr(dc, "_cell", forbidden)
    with pytest.raises(ValueError):
        solve_declared_decision(
            p, criterion="terminal_utility", support_limit=solved.required_nodes - 1
        )
    monkeypatch.setattr(dc, "_values", forbidden)
    with pytest.raises(ValueError):
        evaluate_fixed_decision(
            solved.fixed_policy,
            reference(p),
            support_limit=evaluated.required_nodes - 1,
        )
    with pytest.raises(TypeError):
        fix_declared_decision_policy(p, descriptive(p), support_limit=True)


@pytest.mark.parametrize(
    "change",
    [
        lambda f: replace(f, rule="old"),
        lambda f: replace(f, source_digest="wrong"),
        lambda f: replace(f, policy_digest="wrong"),
        lambda f: replace(f, required_nodes=f.required_nodes - 1),
    ],
)
def test_stored_context_rule_and_bound_revalidated_without_resolve(change):
    p = problem()
    with pytest.raises(ValueError):
        evaluate_fixed_decision(
            change(solve(p).fixed_policy), reference(p), support_limit=LIMIT
        )


def test_failure_propagates_without_mutation_retry_or_positive_result(monkeypatch):
    p = problem()
    fixed = solve(p).fixed_policy
    prior = evaluate_fixed_decision(fixed, reference(p), support_limit=LIMIT)
    saved = prior.export()
    calls = []
    original = dc._values

    def fail(*args):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("NEW injected pure failure")
        return original(*args)

    monkeypatch.setattr(dc, "_values", fail)
    with pytest.raises(RuntimeError, match="NEW injected"):
        evaluate_fixed_decision(fixed, reference(p), support_limit=LIMIT)
    assert len(calls) == 2 and prior.export() == saved


def test_new_tiny_information_evidence_is_not_rounded_to_zero():
    p = problem()
    epsilon = F(1, 10**2100)
    atoms = []
    for a in p.subjective_joint:
        probability = a.probability
        if a.vector == "steady":
            signal = (
                epsilon
                if a.history == "clear"
                else 1 - epsilon
                if a.history == "alert"
                else F(0)
            )
            ready = F(4, 5) if a.history == "clear" else F(1, 5)
            probability = (
                F(1, 2) * signal * (ready if a.state == "ready" else 1 - ready)
            )
        atoms.append(replace(a, probability=probability))
    solved = solve(replace(p, subjective_joint=tuple(atoms)))
    cell = solved.cells[0]
    assert cell.evidence == epsilon / 2 and cell.evidence > 0
    assert cell.posterior == (F(4, 5), F(1, 5)) and cell.action == "standard"


def test_new_hidden_state_specific_policy_cannot_split_a_permitted_cell():
    p = problem()
    supplied = descriptive(p)
    hidden_split = replace(
        supplied,
        actions=supplied.actions + (DecisionAction("clear", "steady", "outside"),),
    )
    with pytest.raises(ValueError, match="one policy row"):
        fix_declared_decision_policy(p, hidden_split, support_limit=LIMIT)


def test_new_explicit_missing_descriptive_behavior_remains_null():
    p = problem()
    supplied = descriptive(p)
    rows = list(supplied.actions)
    rows[0] = replace(rows[0], action=None)
    fixed = fix_declared_decision_policy(
        p, replace(supplied, actions=tuple(rows)), support_limit=LIMIT
    )
    evaluation = evaluate_fixed_decision(fixed, reference(p), support_limit=LIMIT)
    assert evaluation.totals == (None,) * 4
    assert all(v > 0 for v in evaluation.undefined_weights)
