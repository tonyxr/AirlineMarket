"""A.10/A.5/B0/V9/B4: declared action laws, criteria and fixed policy values.

One commitment on complete declared information cells. No native probability,
awareness, preference, history binding or lifecycle consumer surplus is inferred.
"""

from dataclasses import dataclass, replace
from fractions import Fraction

from airline_market.sim.decision_consistency import (
    REFERENCE_SCOPE,
    DecisionAccount,
    DecisionAction,
    DecisionAtom,
    DeclaredDecisionPolicy,
    ObservedDecisionVector,
    _account,
    _digest,
    _extra,
    _integral,
    _joint,
    _limit,
    _material,
    _names,
    _nodes,
    _policy,
    _primitive,
    _rational,
    _sequence,
    _text,
)

RULE = "complete-declared-action-kernel-utility-v1"
METRICS = ("material", "noise_excluded", "experienced", "same_path_normalized")
CRITERIA = ("terminal_utility", "material_plus_decision_noise")


@dataclass(frozen=True)
class ActionKernelProblem:
    person: str
    known_type: str
    states: tuple[str, ...]
    histories: tuple[str, ...]
    vectors: tuple[ObservedDecisionVector, ...]
    actions: tuple[str, ...]
    outside_action: str
    outcomes: tuple[str, ...]
    alpha: Fraction
    material_unit: str
    utility_unit: str
    definition: str
    conditioning: str
    information_definition: str
    unobserved_definition: str
    non_overlap_definition: str
    reference_scope: str


@dataclass(frozen=True)
class ActionOutcome:
    state: str
    history: str
    vector: str
    action: str
    outcome: str
    probability: Fraction
    account: DecisionAccount
    same_path_outside: DecisionAccount


@dataclass(frozen=True)
class ActionChanceLaw:
    name: str
    definition: str
    prior: tuple[DecisionAtom, ...]
    outcomes: tuple[ActionOutcome, ...]


@dataclass(frozen=True)
class RegisteredActionUtility:
    problem: ActionKernelProblem
    subjective: ActionChanceLaw
    reference: ActionChanceLaw
    descriptive: DeclaredDecisionPolicy
    criterion: str
    digest: str
    required_nodes: int
    support_limit: int
    rule: str

    def export(self):
        return _primitive(self)


@dataclass(frozen=True)
class ActionScoreCell:
    history: str
    vector: str
    evidence: Fraction
    posterior: tuple[Fraction, ...] | None
    material_scores: tuple[Fraction | None, ...]
    terminal_scores: tuple[Fraction | None, ...]
    decision_noise_scores: tuple[Fraction | None, ...]
    action: str | None


@dataclass(frozen=True)
class SubjectiveActionSolution:
    policy: DeclaredDecisionPolicy
    criterion: str
    cells: tuple[ActionScoreCell, ...]


@dataclass(frozen=True)
class ActionProjection:
    state: str
    history: str
    vector: str
    action: str
    outcome: str
    prior: Fraction
    conditional: Fraction
    weight: Fraction
    values: tuple[Fraction | None, ...]
    contributions: tuple[Fraction | None, ...]


@dataclass(frozen=True)
class SelectedActionRow:
    state: str
    history: str
    vector: str
    action: str | None
    outcome: str | None
    weight: Fraction
    values: tuple[Fraction | None, ...]
    contributions: tuple[Fraction | None, ...]


@dataclass(frozen=True)
class ActionPolicyEvaluation:
    law: str
    policy: DeclaredDecisionPolicy
    rows: tuple[SelectedActionRow, ...]
    totals: tuple[Fraction | None, ...]
    known_numerators: tuple[Fraction, ...]
    undefined_weights: tuple[Fraction, ...]
    missing_behavior_mass: Fraction


@dataclass(frozen=True)
class OutsidePolicyComparison:
    law: str
    policy: str
    experienced_difference: Fraction | None
    same_selected_path_normalized: Fraction | None
    definition: str


@dataclass(frozen=True)
class ActionUtilityResult:
    registration: RegisteredActionUtility
    solution: SubjectiveActionSolution
    subjective_projections: tuple[ActionProjection, ...]
    reference_projections: tuple[ActionProjection, ...]
    evaluations: tuple[ActionPolicyEvaluation, ...]
    comparisons: tuple[OutsidePolicyComparison, ...]
    native_history_association: None
    native_preferences_or_lifecycle_CS_adopted: bool
    rule: str

    def export(self):
        return _primitive(self)


def _problem(value):
    if type(value) is not ActionKernelProblem:
        raise TypeError("ActionKernelProblem required")
    actions = _names(value.actions)
    vectors = []
    for v in _sequence(value.vectors):
        if type(v) is not ObservedDecisionVector:
            raise TypeError("ObservedDecisionVector required")
        noise = tuple(_rational(n) for n in _sequence(v.action_noise))
        if len(noise) != len(actions):
            raise ValueError("complete observed action vector required")
        vectors.append(
            ObservedDecisionVector(_text(v.name), noise, _text(v.definition))
        )
    _names([v.name for v in vectors])
    alpha = _rational(value.alpha)
    if alpha <= 0 or _text(value.material_unit) == _text(value.utility_unit):
        raise ValueError("positive conversion and distinct units required")
    if value.outside_action not in actions or value.reference_scope != REFERENCE_SCOPE:
        raise ValueError(
            "explicit outside action and same-path reference scope required"
        )
    return ActionKernelProblem(
        _text(value.person),
        _text(value.known_type),
        _names(value.states),
        _names(value.histories),
        tuple(vectors),
        actions,
        _text(value.outside_action),
        _names(value.outcomes),
        alpha,
        value.material_unit,
        value.utility_unit,
        _text(value.definition),
        _text(value.conditioning),
        _text(value.information_definition),
        _text(value.unobserved_definition),
        _text(value.non_overlap_definition),
        REFERENCE_SCOPE,
    )


def _law(problem, value):
    if type(value) is not ActionChanceLaw:
        raise TypeError("complete ActionChanceLaw required")
    prior = _joint(problem, value.prior)
    expected = tuple(
        (x, h, g.name, a, y)
        for x in problem.states
        for h in problem.histories
        for g in problem.vectors
        for a in problem.actions
        for y in problem.outcomes
    )
    rows = []
    for row in _sequence(value.outcomes):
        if type(row) is not ActionOutcome:
            raise TypeError("ActionOutcome required")
        p = _rational(row.probability)
        if p < 0:
            raise ValueError("negative conditional probability")
        rows.append(
            ActionOutcome(
                _text(row.state),
                _text(row.history),
                _text(row.vector),
                _text(row.action),
                _text(row.outcome),
                p,
                _account(row.account),
                _account(row.same_path_outside),
            )
        )
    keys = [(r.state, r.history, r.vector, r.action, r.outcome) for r in rows]
    if len(keys) != len(expected) or set(keys) != set(expected):
        raise ValueError(
            "complete action/outcome support including unused zero rows required"
        )
    by_key = dict(zip(keys, rows))
    for x, h, g in ((r.state, r.history, r.vector) for r in prior):
        for a in problem.actions:
            if sum(by_key[x, h, g, a, y].probability for y in problem.outcomes) != 1:
                raise ValueError("EACH conditional action block must sum exactly one")
    return ActionChanceLaw(
        _text(value.name),
        _text(value.definition),
        prior,
        tuple(by_key[k] for k in expected),
    )


def _outside(problem):
    return DeclaredDecisionPolicy(
        "all_outside",
        "Separate declared outside policy uses its OWN action kernel",
        tuple(
            DecisionAction(h, g.name, problem.outside_action)
            for h in problem.histories
            for g in problem.vectors
        ),
    )


def _empty_policy(problem):
    return DeclaredDecisionPolicy(
        "subjective_candidate",
        "First declared action breaks ties; undefined score stays NULL",
        tuple(
            DecisionAction(h, g.name, None)
            for h in problem.histories
            for g in problem.vectors
        ),
    )


def _shape(reg):
    """Conservative full exported shape; scalar bit/byte lengths are not bounded."""
    p = reg.problem
    nulls = (None,) * 4
    cells = tuple(
        ActionScoreCell(
            h,
            g.name,
            Fraction(0),
            (None,) * len(p.states),
            (None,) * len(p.actions),
            (None,) * len(p.actions),
            (None,) * len(p.actions),
            None,
        )
        for h in p.histories
        for g in p.vectors
    )
    projections = tuple(
        ActionProjection(
            r.state,
            r.history,
            r.vector,
            r.action,
            r.outcome,
            Fraction(0),
            Fraction(0),
            Fraction(0),
            nulls,
            nulls,
        )
        for r in reg.subjective.outcomes
    )
    evaluations = tuple(
        ActionPolicyEvaluation(
            law.name,
            policy,
            tuple(
                SelectedActionRow(
                    r.state, r.history, r.vector, None, y, Fraction(0), nulls, nulls
                )
                for r in law.prior
                for y in p.outcomes
            ),
            nulls,
            nulls,
            nulls,
            Fraction(0),
        )
        for law in (reg.subjective, reg.reference)
        for policy in (_empty_policy(p), reg.descriptive, _outside(p))
    )
    comparisons = tuple(
        OutsidePolicyComparison(
            law.name,
            policy.name,
            None,
            None,
            "declaration",
        )
        for law in (reg.subjective, reg.reference)
        for policy in (_empty_policy(p), reg.descriptive)
    )
    return ActionUtilityResult(
        reg,
        SubjectiveActionSolution(_empty_policy(p), reg.criterion, cells),
        projections,
        projections,
        evaluations,
        comparisons,
        None,
        False,
        RULE,
    )


def _payload(reg):
    return (
        reg.problem,
        reg.subjective,
        reg.reference,
        reg.descriptive,
        reg.criterion,
        RULE,
    )


def _required(reg):
    return _nodes({"input": reg, "output": _shape(reg)})


def register_action_utility(
    problem, subjective, reference, descriptive, *, criterion, support_limit
):
    """Own BOTH complete role laws/accounts and output support before any forward."""
    problem = _problem(problem)
    subjective, reference = _law(problem, subjective), _law(problem, reference)
    descriptive = _policy(problem, descriptive)
    if criterion not in CRITERIA:
        raise ValueError("explicit supported subjective criterion required")
    if subjective.name == reference.name:
        raise ValueError("distinct subjective/reference role law names required")
    reg = RegisteredActionUtility(
        problem,
        subjective,
        reference,
        descriptive,
        criterion,
        "digest",
        0,
        support_limit,
        RULE,
    )
    needed = _required(reg)
    _limit(support_limit, needed)
    return replace(reg, digest=_digest(_payload(reg)), required_nodes=needed)


def _validated(reg):
    if type(reg) is not RegisteredActionUtility or reg.rule != RULE:
        raise ValueError("current complete registration rule required")
    owned = register_action_utility(
        reg.problem,
        reg.subjective,
        reg.reference,
        reg.descriptive,
        criterion=reg.criterion,
        support_limit=reg.support_limit,
    )
    if (
        type(reg.required_nodes) is not int
        or reg.required_nodes != owned.required_nodes
    ):
        raise ValueError("recomputed complete support mismatch")
    if reg.digest != owned.digest:
        raise ValueError("full declaration digest mismatch")
    return owned


def _values(problem, row):
    z, e = _material(row.account), _extra(row.account)
    z0, e0 = _material(row.same_path_outside), _extra(row.same_path_outside)
    return (
        z,
        None if z is None else problem.alpha * z,
        None if z is None or e is None else problem.alpha * z + e,
        None
        if any(v is None for v in (z, e, z0, e0))
        else problem.alpha * (z - z0) + e - e0,
    )


def _contributions(weight, values):
    return tuple(
        Fraction(0) if weight == 0 else None if v is None else weight * v
        for v in values
    )


def _project(problem, law):
    prior = {(r.state, r.history, r.vector): r.probability for r in law.prior}
    rows = []
    for row in law.outcomes:
        base = prior[row.state, row.history, row.vector]
        weight = base * row.probability
        values = _values(problem, row)
        rows.append(
            ActionProjection(
                row.state,
                row.history,
                row.vector,
                row.action,
                row.outcome,
                base,
                row.probability,
                weight,
                values,
                _contributions(weight, values),
            )
        )
    return tuple(rows)


def _solve_subjective(problem, subjective, projections, criterion):
    """Candidate receives ONLY q; no reference-law parameter or truth access."""
    cells, actions = [], []
    for h in problem.histories:
        for g in problem.vectors:
            raw = tuple(
                r.probability
                for r in subjective.prior
                if (r.history, r.vector) == (h, g.name)
            )
            evidence = sum(raw, Fraction(0))
            posterior = tuple(w / evidence for w in raw) if evidence else None
            material, terminal, noise = [], [], []
            for ai, a in enumerate(problem.actions):
                selected = tuple(
                    r
                    for r in projections
                    if (r.history, r.vector, r.action) == (h, g.name, a)
                )
                scores = tuple(
                    None
                    if not evidence
                    else _integral((r.weight / evidence, r.values[i]) for r in selected)
                    for i in (1, 2)
                )
                material.append(scores[0])
                terminal.append(scores[1])
                noise.append(
                    None if scores[0] is None else scores[0] + g.action_noise[ai]
                )
            target = terminal if criterion == "terminal_utility" else noise
            action = (
                None
                if any(v is None for v in target)
                else problem.actions[target.index(max(target))]
            )
            cells.append(
                ActionScoreCell(
                    h,
                    g.name,
                    evidence,
                    posterior,
                    tuple(material),
                    tuple(terminal),
                    tuple(noise),
                    action,
                )
            )
            actions.append(DecisionAction(h, g.name, action))
    return SubjectiveActionSolution(
        replace(_empty_policy(problem), actions=tuple(actions)),
        criterion,
        tuple(cells),
    )


def _evaluate_fixed(problem, law, policy, projections):
    """Internal validated fixed-policy evaluator: never computes scores or resolves."""
    actions = {(r.history, r.vector): r.action for r in policy.actions}
    rows, missing = [], Fraction(0)
    grouped = {}
    for row in projections:
        grouped.setdefault((row.state, row.history, row.vector, row.action), []).append(
            row
        )
    for atom in law.prior:
        action = actions[atom.history, atom.vector]
        if action is None:
            missing += atom.probability
            values = (None,) * 4
            rows.append(
                SelectedActionRow(
                    atom.state,
                    atom.history,
                    atom.vector,
                    None,
                    None,
                    atom.probability,
                    values,
                    _contributions(atom.probability, values),
                )
            )
        else:
            rows.extend(
                SelectedActionRow(
                    r.state,
                    r.history,
                    r.vector,
                    action,
                    r.outcome,
                    r.weight,
                    r.values,
                    r.contributions,
                )
                for r in grouped[atom.state, atom.history, atom.vector, action]
            )
    totals = tuple(_integral((r.weight, r.values[i]) for r in rows) for i in range(4))
    known = tuple(
        sum(
            (r.contributions[i] for r in rows if r.contributions[i] is not None),
            Fraction(0),
        )
        for i in range(4)
    )
    undefined = tuple(
        sum((r.weight for r in rows if r.values[i] is None), Fraction(0))
        for i in range(4)
    )
    return ActionPolicyEvaluation(
        law.name, policy, tuple(rows), totals, known, undefined, missing
    )


def execute_action_utility(registration):
    """Validate complete owned support, solve q once, evaluate fixed policies under q/r."""
    reg = _validated(registration)
    q_rows, r_rows = (
        _project(reg.problem, reg.subjective),
        _project(reg.problem, reg.reference),
    )
    solution = _solve_subjective(reg.problem, reg.subjective, q_rows, reg.criterion)
    policies = (solution.policy, reg.descriptive, _outside(reg.problem))
    evaluations, comparisons = [], []
    for law, projections in ((reg.subjective, q_rows), (reg.reference, r_rows)):
        values = tuple(
            _evaluate_fixed(reg.problem, law, policy, projections)
            for policy in policies
        )
        evaluations.extend(values)
        for result in values[:2]:
            exp, outside = result.totals[2], values[2].totals[2]
            comparisons.append(
                OutsidePolicyComparison(
                    law.name,
                    result.policy.name,
                    None if exp is None or outside is None else exp - outside,
                    result.totals[3],
                    "Separate all-outside policy uses its own kernel; same-path reference uses selected kernel; neither identifies causal welfare/CS",
                )
            )
    return ActionUtilityResult(
        reg,
        solution,
        q_rows,
        r_rows,
        tuple(evaluations),
        tuple(comparisons),
        None,
        False,
        RULE,
    )
