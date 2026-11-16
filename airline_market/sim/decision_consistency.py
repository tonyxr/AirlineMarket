"""A.10/A.5/B0/V9/B4: finite declared criteria and fixed-policy evaluation.

This opt-in single-commitment model identifies no native probability,
received knowledge, preference, lifecycle utility or consumer surplus.
"""

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from fractions import Fraction

RULE = "declared-decision-consistency-v1"
REFERENCE_SCOPE = "same-selected-path-outside-account-not-outside-intervention"
METRICS = ("material", "noise_excluded", "experienced", "normalized_experience")


@dataclass(frozen=True)
class DecisionAccount:
    components: tuple[Fraction | None, ...]
    component_sources: tuple[str, ...]
    extra_utility: Fraction | None
    extra_source: str
    financial_closed: bool
    material_closed: bool
    utility_closed: bool
    closure_sources: tuple[str, ...]
    definition: str
    origin: str
    incidence: str


@dataclass(frozen=True)
class ObservedDecisionVector:
    name: str
    action_noise: tuple[Fraction, ...]
    definition: str


@dataclass(frozen=True)
class DecisionAtom:
    state: str
    history: str
    vector: str
    probability: Fraction


@dataclass(frozen=True)
class DecisionContingency:
    state: str
    history: str
    vector: str
    accounts: tuple[DecisionAccount, ...]


@dataclass(frozen=True)
class DeclaredDecisionProblem:
    person: str
    known_type: str
    states: tuple[str, ...]
    histories: tuple[str, ...]
    vectors: tuple[ObservedDecisionVector, ...]
    actions: tuple[str, ...]
    outside_action: str
    subjective_joint: tuple[DecisionAtom, ...]
    contingencies: tuple[DecisionContingency, ...]
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
class DecisionAction:
    history: str
    vector: str
    action: str | None


@dataclass(frozen=True)
class DeclaredDecisionPolicy:
    name: str
    definition: str
    actions: tuple[DecisionAction, ...]


@dataclass(frozen=True)
class DecisionReferenceLaw:
    name: str
    definition: str
    joint: tuple[DecisionAtom, ...]


@dataclass(frozen=True)
class FixedDecisionPolicy:
    problem: DeclaredDecisionProblem
    policy: DeclaredDecisionPolicy
    origin: str
    source_digest: str
    policy_digest: str
    required_nodes: int
    support_limit: int
    rule: str

    def export(self):
        return _primitive(self)


@dataclass(frozen=True)
class DecisionCell:
    history: str
    vector: str
    evidence: Fraction
    posterior: tuple[Fraction, ...] | None
    material_scores: tuple[Fraction | None, ...]
    experienced_scores: tuple[Fraction | None, ...]
    noise_scores: tuple[Fraction | None, ...]
    action: str | None


@dataclass(frozen=True)
class DecisionSolution:
    fixed_policy: FixedDecisionPolicy
    criterion: str
    cells: tuple[DecisionCell, ...]
    required_nodes: int
    support_limit: int
    rule: str

    def export(self):
        return _primitive(self)


@dataclass(frozen=True)
class DecisionEvaluationRow:
    state: str
    history: str
    vector: str
    weight: Fraction
    action: str | None
    values: tuple[Fraction | None, ...]
    contributions: tuple[Fraction | None, ...]


@dataclass(frozen=True)
class FixedDecisionEvaluation:
    fixed_policy: FixedDecisionPolicy
    reference: DecisionReferenceLaw
    rows: tuple[DecisionEvaluationRow, ...]
    totals: tuple[Fraction | None, ...]
    known_numerators: tuple[Fraction, ...]
    undefined_weights: tuple[Fraction, ...]
    required_nodes: int
    support_limit: int
    rule: str

    def export(self):
        return _primitive(self)


def _primitive(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is Fraction:
        return str(value)
    if is_dataclass(value):
        return {f.name: _primitive(getattr(value, f.name)) for f in fields(value)}
    if type(value) in (tuple, list):
        return [_primitive(v) for v in value]
    if type(value) is dict:
        return {k: _primitive(v) for k, v in value.items()}
    raise TypeError("unsupported declared value")


def _digest(value):
    return hashlib.sha256(
        json.dumps(_primitive(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _nodes(value):
    value = _primitive(value)
    if type(value) is dict:
        return 1 + sum(1 + _nodes(v) for v in value.values())
    if type(value) is list:
        return 1 + sum(_nodes(v) for v in value)
    return 1


def _sequence(value):
    if type(value) not in (tuple, list):
        raise TypeError("explicit finite sequence required")
    return tuple(value)


def _text(value):
    if type(value) is not str or not value:
        raise ValueError("explicit nonempty text required")
    return value


def _rational(value, unknown=False):
    if unknown and value is None:
        return None
    if type(value) is not Fraction:
        raise TypeError("explicit Fraction required")
    return value


def _names(value):
    values = tuple(_text(v) for v in _sequence(value))
    if not values or len(set(values)) != len(values):
        raise ValueError("complete distinct nonempty support required")
    return values


def _coordinates(problem):
    return tuple(
        (x, h, v.name)
        for x in problem.states
        for h in problem.histories
        for v in problem.vectors
    )


def _joint(problem, atoms):
    owned = []
    for atom in _sequence(atoms):
        if type(atom) is not DecisionAtom:
            raise TypeError("DecisionAtom required")
        probability = _rational(atom.probability)
        if probability < 0:
            raise ValueError("negative probability")
        owned.append(
            DecisionAtom(
                _text(atom.state), _text(atom.history), _text(atom.vector), probability
            )
        )
    coords = [(v.state, v.history, v.vector) for v in owned]
    expected = _coordinates(problem)
    if len(coords) != len(expected) or set(coords) != set(expected):
        raise ValueError("complete joint support required, including zero atoms")
    if sum(v.probability for v in owned) != 1:
        raise ValueError("joint probabilities must sum exactly one")
    by_key = dict(zip(coords, owned))
    return tuple(by_key[k] for k in expected)


def _account(value):
    if type(value) is not DecisionAccount:
        raise TypeError("DecisionAccount required")
    components = tuple(_rational(v, unknown=True) for v in _sequence(value.components))
    sources = tuple(_text(v) for v in _sequence(value.component_sources))
    closures = tuple(_text(v) for v in _sequence(value.closure_sources))
    if len(components) != 4 or len(sources) != 4 or len(closures) != 3:
        raise ValueError(
            "four explicit components/sources and three closure sources required"
        )
    if any(
        type(v) is not bool
        for v in (value.financial_closed, value.material_closed, value.utility_closed)
    ):
        raise TypeError("independent Boolean closures required")
    return DecisionAccount(
        components,
        sources,
        _rational(value.extra_utility, unknown=True),
        _text(value.extra_source),
        value.financial_closed,
        value.material_closed,
        value.utility_closed,
        closures,
        _text(value.definition),
        _text(value.origin),
        _text(value.incidence),
    )


def _problem(value):
    if type(value) is not DeclaredDecisionProblem:
        raise TypeError("DeclaredDecisionProblem required")
    states, histories, actions = (
        _names(value.states),
        _names(value.histories),
        _names(value.actions),
    )
    vectors = []
    for v in _sequence(value.vectors):
        if type(v) is not ObservedDecisionVector:
            raise TypeError("ObservedDecisionVector required")
        noise = tuple(_rational(n) for n in _sequence(v.action_noise))
        if len(noise) != len(actions):
            raise ValueError("complete private action vector required")
        vectors.append(
            ObservedDecisionVector(_text(v.name), noise, _text(v.definition))
        )
    _names([v.name for v in vectors])
    if value.outside_action not in actions:
        raise ValueError("explicit outside action must be in support")
    alpha = _rational(value.alpha)
    if alpha <= 0 or _text(value.material_unit) == _text(value.utility_unit):
        raise ValueError("positive conversion and distinct units required")
    if value.reference_scope != REFERENCE_SCOPE:
        raise ValueError("same-selected-path reference scope required")
    result = DeclaredDecisionProblem(
        _text(value.person),
        _text(value.known_type),
        states,
        histories,
        tuple(vectors),
        actions,
        _text(value.outside_action),
        (),
        (),
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
    joint = _joint(result, value.subjective_joint)
    contingencies = []
    for c in _sequence(value.contingencies):
        if type(c) is not DecisionContingency:
            raise TypeError("DecisionContingency required")
        accounts = tuple(_account(v) for v in _sequence(c.accounts))
        if len(accounts) != len(actions):
            raise ValueError("complete action accounts required")
        contingencies.append(
            DecisionContingency(
                _text(c.state), _text(c.history), _text(c.vector), accounts
            )
        )
    keys = [(c.state, c.history, c.vector) for c in contingencies]
    expected = _coordinates(result)
    if len(keys) != len(expected) or set(keys) != set(expected):
        raise ValueError(
            "complete contingent accounts required, including unused support"
        )
    by_key = dict(zip(keys, contingencies))
    return DeclaredDecisionProblem(
        result.person,
        result.known_type,
        states,
        histories,
        tuple(vectors),
        actions,
        result.outside_action,
        joint,
        tuple(by_key[k] for k in expected),
        alpha,
        result.material_unit,
        result.utility_unit,
        result.definition,
        result.conditioning,
        result.information_definition,
        result.unobserved_definition,
        result.non_overlap_definition,
        result.reference_scope,
    )


def _policy(problem, value):
    if type(value) is not DeclaredDecisionPolicy:
        raise TypeError("DeclaredDecisionPolicy required")
    owned = []
    for row in _sequence(value.actions):
        if type(row) is not DecisionAction:
            raise TypeError("DecisionAction required")
        if row.action is not None:
            _text(row.action)
        if row.action is not None and row.action not in problem.actions:
            raise ValueError("unknown selected action")
        owned.append(DecisionAction(_text(row.history), _text(row.vector), row.action))
    keys = [(v.history, v.vector) for v in owned]
    expected = [(h, v.name) for h in problem.histories for v in problem.vectors]
    if len(keys) != len(expected) or set(keys) != set(expected):
        raise ValueError(
            "one policy row per full permitted history/private vector required"
        )
    by_key = dict(zip(keys, owned))
    return DeclaredDecisionPolicy(
        _text(value.name), _text(value.definition), tuple(by_key[k] for k in expected)
    )


def _limit(limit, needed):
    if type(limit) is not int or limit <= 0:
        raise TypeError("positive integer structural limit required")
    if needed > limit:
        raise ValueError("complete structural support exceeds limit")


def _empty_policy(problem):
    return DeclaredDecisionPolicy(
        "placeholder",
        "complete programme",
        tuple(
            DecisionAction(h, v.name, None)
            for h in problem.histories
            for v in problem.vectors
        ),
    )


def _fixed_shape(problem, policy, origin):
    return FixedDecisionPolicy(problem, policy, origin, "digest", "digest", 0, 0, RULE)


def _solve_required(problem):
    fixed = _fixed_shape(problem, _empty_policy(problem), "subjective_criterion")
    cells = tuple(
        DecisionCell(
            h,
            v.name,
            Fraction(0),
            (None,) * len(problem.states),
            (None,) * len(problem.actions),
            (None,) * len(problem.actions),
            (None,) * len(problem.actions),
            None,
        )
        for h in problem.histories
        for v in problem.vectors
    )
    return _nodes(
        {
            "input": problem,
            "output": DecisionSolution(fixed, "criterion", cells, 0, 0, RULE),
        }
    )


def _fixed_required(problem, policy):
    return _nodes(
        {
            "input_problem": problem,
            "input_policy": policy,
            "output": _fixed_shape(problem, policy, "supplied_descriptive"),
        }
    )


def _material(account):
    if (
        not account.financial_closed
        or not account.material_closed
        or any(v is None for v in account.components)
    ):
        return None
    b, p, w, c = account.components
    return b - p - w - c


def _extra(account):
    return account.extra_utility if account.utility_closed else None


def _integral(terms):
    terms = tuple(terms)
    if any(w > 0 and v is None for w, v in terms):
        return None
    return sum((w * v for w, v in terms if w != 0), Fraction(0))


def _cell(problem, history, vector):
    weights = {
        (v.state, v.history, v.vector): v.probability for v in problem.subjective_joint
    }
    accounts = {
        (v.state, v.history, v.vector): v.accounts for v in problem.contingencies
    }
    raw = tuple(weights[x, history, vector.name] for x in problem.states)
    evidence = sum(raw)
    posterior = tuple(w / evidence for w in raw) if evidence > 0 else None
    material, experienced, noise = [], [], []
    for ai in range(len(problem.actions)):
        mz = (
            _integral(
                (posterior[i], _material(accounts[x, history, vector.name][ai]))
                for i, x in enumerate(problem.states)
            )
            if posterior is not None
            else None
        )
        extra = (
            _integral(
                (posterior[i], _extra(accounts[x, history, vector.name][ai]))
                for i, x in enumerate(problem.states)
            )
            if posterior is not None
            else None
        )
        material.append(None if mz is None else problem.alpha * mz)
        experienced.append(
            None if mz is None or extra is None else problem.alpha * mz + extra
        )
        noise.append(
            None if mz is None else problem.alpha * mz + vector.action_noise[ai]
        )
    return evidence, posterior, tuple(material), tuple(experienced), tuple(noise)


def solve_declared_decision(problem, *, criterion, support_limit):
    """Solve ONE subjective candidate; no reference-law parameter is accepted."""
    problem = _problem(problem)
    if criterion not in ("terminal_utility", "material_plus_decision_noise"):
        raise ValueError("explicit supported criterion required")
    needed = _solve_required(problem)
    _limit(support_limit, needed)
    cells = []
    actions = []
    for history in problem.histories:
        for vector in problem.vectors:
            evidence, posterior, material, experienced, noise = _cell(
                problem, history, vector
            )
            scores = experienced if criterion == "terminal_utility" else noise
            action = (
                None
                if any(v is None for v in scores)
                else problem.actions[scores.index(max(scores))]
            )
            cells.append(
                DecisionCell(
                    history,
                    vector.name,
                    evidence,
                    posterior,
                    material,
                    experienced,
                    noise,
                    action,
                )
            )
            actions.append(DecisionAction(history, vector.name, action))
    policy = DeclaredDecisionPolicy(
        criterion,
        "Subjective exact finite argmax; first declared action breaks ties; undefined criterion stays NULL",
        tuple(actions),
    )
    fixed = FixedDecisionPolicy(
        problem,
        policy,
        "subjective_criterion",
        _digest(problem),
        _digest(policy),
        needed,
        support_limit,
        RULE,
    )
    return DecisionSolution(fixed, criterion, tuple(cells), needed, support_limit, RULE)


def fix_declared_decision_policy(problem, policy, *, support_limit):
    """Own an explicit descriptive policy without claiming optimality."""
    problem = _problem(problem)
    policy = _policy(problem, policy)
    needed = _fixed_required(problem, policy)
    _limit(support_limit, needed)
    return FixedDecisionPolicy(
        problem,
        policy,
        "supplied_descriptive",
        _digest(problem),
        _digest(policy),
        needed,
        support_limit,
        RULE,
    )


def _validated_fixed(fixed):
    if type(fixed) is not FixedDecisionPolicy or fixed.rule != RULE:
        raise ValueError("current FixedDecisionPolicy rule required")
    problem = _problem(fixed.problem)
    policy = _policy(problem, fixed.policy)
    if fixed.source_digest != _digest(problem) or fixed.policy_digest != _digest(
        policy
    ):
        raise ValueError("complete fixed source/policy digest mismatch")
    if fixed.origin == "subjective_criterion":
        needed = _solve_required(problem)
    elif fixed.origin == "supplied_descriptive":
        needed = _fixed_required(problem, policy)
    else:
        raise ValueError("unknown policy origin")
    if type(fixed.required_nodes) is not int or fixed.required_nodes != needed:
        raise ValueError("stored complete structural support mismatch")
    _limit(fixed.support_limit, needed)
    return FixedDecisionPolicy(
        problem,
        policy,
        fixed.origin,
        fixed.source_digest,
        fixed.policy_digest,
        needed,
        fixed.support_limit,
        RULE,
    )


def _values(problem, accounts, action):
    if action is None:
        return (None,) * len(METRICS)
    current = accounts[problem.actions.index(action)]
    outside = accounts[problem.actions.index(problem.outside_action)]
    z, e, z0, e0 = (
        _material(current),
        _extra(current),
        _material(outside),
        _extra(outside),
    )
    return (
        z,
        None if z is None else problem.alpha * z,
        None if z is None or e is None else problem.alpha * z + e,
        None
        if any(v is None for v in (z, e, z0, e0))
        else problem.alpha * (z - z0) + e - e0,
    )


def evaluate_fixed_decision(fixed, reference, *, support_limit):
    """Evaluate unchanged actions under a separate full law; never solve again."""
    fixed = _validated_fixed(fixed)
    if type(reference) is not DecisionReferenceLaw:
        raise TypeError("complete DecisionReferenceLaw required")
    reference = DecisionReferenceLaw(
        _text(reference.name),
        _text(reference.definition),
        _joint(fixed.problem, reference.joint),
    )
    placeholder = tuple(
        DecisionEvaluationRow(
            v.state, v.history, v.vector, Fraction(0), None, (None,) * 4, (None,) * 4
        )
        for v in reference.joint
    )
    shape = FixedDecisionEvaluation(
        fixed, reference, placeholder, (None,) * 4, (None,) * 4, (None,) * 4, 0, 0, RULE
    )
    needed = _nodes(
        {"input_fixed": fixed, "input_reference": reference, "output": shape}
    )
    _limit(support_limit, needed)
    actions = {(v.history, v.vector): v.action for v in fixed.policy.actions}
    contexts = {
        (v.state, v.history, v.vector): v.accounts for v in fixed.problem.contingencies
    }
    rows = []
    for atom in reference.joint:
        key = atom.state, atom.history, atom.vector
        action = actions[atom.history, atom.vector]
        values = _values(fixed.problem, contexts[key], action)
        contributions = tuple(
            Fraction(0)
            if atom.probability == 0
            else None
            if v is None
            else atom.probability * v
            for v in values
        )
        rows.append(
            DecisionEvaluationRow(*key, atom.probability, action, values, contributions)
        )
    totals = tuple(
        _integral((row.weight, row.values[i]) for row in rows) for i in range(4)
    )
    known = tuple(
        sum(
            (row.contributions[i] for row in rows if row.contributions[i] is not None),
            Fraction(0),
        )
        for i in range(4)
    )
    undefined = tuple(
        sum((row.weight for row in rows if row.values[i] is None), Fraction(0))
        for i in range(4)
    )
    return FixedDecisionEvaluation(
        fixed,
        reference,
        tuple(rows),
        totals,
        known,
        undefined,
        needed,
        support_limit,
        RULE,
    )
