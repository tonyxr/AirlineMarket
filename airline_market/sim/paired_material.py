"""A.10/A.5/A.14/B0/V9: fixed-policy material under two DECLARED laws.

No native source, belief authentication, policy optimization or lifecycle CS.
"""

import hashlib
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

from airline_market.core.hashing import canonical_json
from airline_market.sim.received_information import _text, _units
from airline_market.sim.shared_mark_attempt import _array, _encoded, _names, _owned
from airline_market.sim.terminal_material import TerminalMaterialAccount, _accounts

RULE = "paired-declared-fixed-policy-material-v1"
SCOPE = "declared-precategory-after-original-history-not-native-derived"
SHOCK_SCOPE = "ALL-private-choice-shock-utility-excluded"


class PairedMaterialLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class MaterialSupport:
    original_person_label: str
    market: str
    original_carrier: str
    physical_states: tuple
    latent_types: tuple
    private_marks: tuple
    observations: tuple
    type_observation: tuple
    signals: tuple
    products: tuple
    product_carriers: tuple
    own_menus: tuple
    interline_menus: tuple
    interline_enabled: bool
    outcomes: tuple
    payoff_unit: str
    conditioning_scope: str
    conditioning_definition: str
    observation_definition: str
    private_mark_interpretation: str
    choice_shock_scope: str

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _owned(getattr(self, name)))


@dataclass(frozen=True)
class FixedMaterialPolicy:
    own_program: tuple
    interline_program: tuple
    definition: str
    source_ref: str

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _owned(getattr(self, name)))


@dataclass(frozen=True)
class DeclaredMaterialLaw:
    name: str
    initial_joint: tuple
    transition_signal: tuple
    terminal_probabilities: tuple
    terminal_accounts: tuple
    definition: str
    source_ref: str

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            object.__setattr__(
                self,
                name,
                _accounts(value) if name == "terminal_accounts" else _owned(value),
            )


@dataclass(frozen=True)
class PairedMaterialModel:
    support: MaterialSupport
    policy: FixedMaterialPolicy
    forecast: DeclaredMaterialLaw
    reference: DeclaredMaterialLaw


@dataclass(frozen=True)
class PairedMaterialResult:
    analyst: object
    row_limit: int
    rule: str = RULE

    def __post_init__(self):
        object.__setattr__(self, "analyst", _owned(self.analyst))


def _sequence(value, dimensions, leaf):
    if not dimensions:
        leaf(value)
    else:
        if type(value) is not tuple or len(value) != dimensions[0]:
            raise ValueError("complete ordered support required")
        for row in value:
            _sequence(row, dimensions[1:], leaf)


def _validate(model, row_limit):
    if (
        type(model) is not PairedMaterialModel
        or type(model.support) is not MaterialSupport
        or type(model.policy) is not FixedMaterialPolicy
    ):
        raise TypeError(
            "complete typed support, ONE fixed policy and TWO laws required"
        )
    if type(row_limit) is not int or row_limit <= 0:
        raise ValueError("positive integer structural limit required")
    s = model.support
    for name in (
        "original_person_label",
        "market",
        "original_carrier",
        "payoff_unit",
        "conditioning_definition",
        "observation_definition",
        "private_mark_interpretation",
    ):
        _text(getattr(s, name))
    if s.conditioning_scope != SCOPE or s.choice_shock_scope != SHOCK_SCOPE:
        raise ValueError("explicit declared conditioning/material/shock scope required")
    for name in (
        "physical_states",
        "latent_types",
        "private_marks",
        "observations",
        "signals",
        "products",
        "outcomes",
    ):
        _names(getattr(s, name), name)
    n, k, g, o, h, p, d = (
        len(getattr(s, name))
        for name in (
            "physical_states",
            "latent_types",
            "private_marks",
            "observations",
            "signals",
            "products",
            "outcomes",
        )
    )
    a = p + 1
    if len(s.type_observation) != k or any(
        x not in s.observations for x in s.type_observation
    ):
        raise ValueError("complete deterministic O=m(K) required")
    if len(s.product_carriers) != p or type(s.interline_enabled) is not bool:
        raise ValueError("complete carriers and explicit interline enable required")
    for carrier in s.product_carriers:
        _text(carrier)

    def boolean(x):
        if type(x) is not bool:
            raise TypeError("explicit boolean fixed menu required")

    _sequence(s.own_menus, (o, p), boolean)
    _sequence(s.interline_menus, (o, h, p), boolean)
    for obs, j in product(range(o), range(p)):
        if s.own_menus[obs][j] and s.product_carriers[j] != s.original_carrier:
            raise ValueError("own menu must use declared original carrier")
        for signal in range(h):
            if s.interline_menus[obs][signal][j] and (
                not s.interline_enabled or s.product_carriers[j] == s.original_carrier
            ):
                raise ValueError(
                    "interline menu must match enabled other-carrier declaration"
                )
    policy = model.policy
    _text(policy.definition)
    _text(policy.source_ref)
    complete = True

    def policy_row(row, mask):
        nonlocal complete
        if type(row) is not tuple or len(row) != a:
            raise ValueError("complete policy row including outside required")
        if all(x is None for x in row):
            complete = False
            return
        if (
            any(type(x) is not Fraction or x < 0 for x in row)
            or sum(row, Fraction()) != 1
        ):
            raise ValueError(
                "unit exact policy OR WHOLE explicitly missing row required"
            )
        if any(row[j] and not mask[j] for j in range(p)):
            raise ValueError("positive action on unavailable product refused")

    _sequence(policy.own_program, (o, g, a), lambda x: None)
    _sequence(policy.interline_program, (o, h, g, a), lambda x: None)
    for obs, mark in product(range(o), range(g)):
        policy_row(policy.own_program[obs][mark], s.own_menus[obs])
        for signal in range(h):
            policy_row(
                policy.interline_program[obs][signal][mark],
                s.interline_menus[obs][signal],
            )
    if (
        type(model.forecast) is not DeclaredMaterialLaw
        or type(model.reference) is not DeclaredMaterialLaw
    ):
        raise TypeError("TWO complete typed laws required")
    if model.forecast.name == model.reference.name:
        raise ValueError(
            "distinct law names required; role does not authenticate truth"
        )
    for law in (model.forecast, model.reference):
        for text in (law.name, law.definition, law.source_ref):
            _text(text)
        _array(law.initial_joint, (n, k, g))
        if (
            sum(
                (
                    law.initial_joint[y][kind][mark]
                    for y, kind, mark in product(range(n), range(k), range(g))
                ),
                Fraction(),
            )
            != 1
        ):
            raise ValueError("complete initial joint must sum EXACTLY one")
        _array(law.transition_signal, (n, k, g, n, h))
        for y, kind, mark in product(range(n), range(k), range(g)):
            if (
                sum(
                    (
                        law.transition_signal[y][kind][mark][yn][signal]
                        for yn, signal in product(range(n), range(h))
                    ),
                    Fraction(),
                )
                != 1
            ):
                raise ValueError(
                    "EVERY joint transition/signal row must sum EXACTLY one"
                )
        _array(law.terminal_probabilities, (2, n, k, g, a, d))

        def account(value):
            if (
                type(value) is not TerminalMaterialAccount
                or value.unit != s.payoff_unit
            ):
                raise ValueError(
                    "EVERY outcome needs explicit FOUR-component account/common unit"
                )

        _sequence(law.terminal_accounts, (2, n, k, g, a, d), account)
        for stage, y, kind, mark, action in product(
            range(2), range(n), range(k), range(g), range(a)
        ):
            if (
                sum(
                    law.terminal_probabilities[stage][y][kind][mark][action], Fraction()
                )
                != 1
            ):
                raise ValueError(
                    "EVERY terminal row including unused support must sum EXACTLY one"
                )
    rows = n * k * g * a * d * (1 + n * h)
    cells = o * (1 + a + h + h * a)
    required = (
        16 * _units(_encoded(model))
        + 2 * (512 * rows + 128 * cells * n * k * g)
        + 20000
    )
    if required > row_limit:
        raise PairedMaterialLimit(
            "ALL input/zero/unknown/proposed output support exceeds structural limit"
        )
    return (n, k, g, o, h, a, d), required, complete


def _multiply(*values):
    if any(x == 0 for x in values):
        return Fraction()
    if any(x is None for x in values):
        return None
    result = Fraction(1)
    for x in values:
        result *= x
    return result


def _sum(values):
    values = tuple(values)
    return None if any(x is None for x in values) else sum(values, Fraction())


def _evaluate(s, policy, law, shape, selected):
    n, k, g, o, h, a, d = shape
    latent = tuple(product(range(n), range(k), range(g)))
    cells = {}

    def add(kind, coord, y, kind_index, mark, mass):
        cell = cells.setdefault((kind, *coord), {x: Fraction() for x in latent})
        key = (y, kind_index, mark)
        cell[key] = _sum((cell[key], mass))

    rows = []
    routing = Fraction()
    for y, kind, mark in latent:
        obs = s.observations.index(s.type_observation[kind])
        initial = law.initial_joint[y][kind][mark]
        own = policy.own_program[obs][mark]
        add("category", (obs,), y, kind, mark, initial)
        if own[0] is None:
            routing += initial
        for action in range(a):
            add(
                "own-action",
                (obs, action),
                y,
                kind,
                mark,
                _multiply(initial, own[action]),
            )
        for action, outcome in product(range(a), range(d)):
            q = law.terminal_probabilities[0][y][kind][mark][action][outcome]
            executable = action != a - 1 or not s.interline_enabled
            factors = (initial, own[action], Fraction(1), Fraction(1), q)
            rows.append(
                {
                    "stage": "own",
                    "coordinate": (y, kind, mark, obs, y, -1, action, outcome),
                    "factors": factors,
                    "terminal_executable": executable,
                    "mass": _multiply(*factors) if executable else Fraction(),
                    "account": _encoded(
                        law.terminal_accounts[0][y][kind][mark][action][outcome]
                    ),
                }
            )
        for yn, signal in product(range(n), range(h)):
            transition = law.transition_signal[y][kind][mark][yn][signal]
            reach = _multiply(
                initial, own[-1], transition, Fraction(int(s.interline_enabled))
            )
            add("outside-signal", (obs, signal), yn, kind, mark, reach)
            inter = policy.interline_program[obs][signal][mark]
            if own[0] is not None and inter[0] is None and s.interline_enabled:
                routing += reach
            for action in range(a):
                add(
                    "interline-action",
                    (obs, signal, action),
                    yn,
                    kind,
                    mark,
                    _multiply(reach, inter[action]),
                )
            for action, outcome in product(range(a), range(d)):
                q = law.terminal_probabilities[1][yn][kind][mark][action][outcome]
                factors = (initial, own[-1], transition, inter[action], q)
                rows.append(
                    {
                        "stage": "interline",
                        "coordinate": (y, kind, mark, obs, yn, signal, action, outcome),
                        "factors": factors,
                        "terminal_executable": s.interline_enabled,
                        "mass": _multiply(*factors)
                        if s.interline_enabled
                        else Fraction(),
                        "account": _encoded(
                            law.terminal_accounts[1][yn][kind][mark][action][outcome]
                        ),
                    }
                )
    known, unknown, known_mass = Fraction(), Fraction(), Fraction()
    selected_rows = []
    for row in rows:
        stage = int(row["stage"] == "interline")
        _, kind, mark, obs, yn, signal, action, outcome = row["coordinate"]
        account = law.terminal_accounts[stage][yn][kind][mark][action][outcome]
        material = account.payoff.net_value()
        effective = (
            material
            if account.financial_tail_closed and account.material_tail_closed
            else None
        )
        contribution = _multiply(row["mass"], effective)
        row.update(
            material_components_diagnostic=material,
            material_if_closed=effective,
            contribution=contribution,
        )
        if row["mass"] is not None:
            known_mass += row["mass"]
            if row["mass"] and effective is None:
                unknown += row["mass"]
            if contribution is not None:
                known += contribution
        if stage and (obs, signal) == selected:
            selected_rows.append(row)
    if known_mass + routing != 1:
        raise ArithmeticError(
            "full known-terminal plus unknown-routing mass differs; no repair"
        )
    histories = []
    for label, support in (
        ("category", product(range(o))),
        ("own-action", product(range(o), range(a))),
        ("outside-signal", product(range(o), range(h))),
        ("interline-action", product(range(o), range(h), range(a))),
    ):
        for coord in support:
            component = cells.get((label, *coord), {x: Fraction() for x in latent})
            evidence = _sum(component.values())
            histories.append(
                {
                    "kind": label,
                    "coordinate": coord,
                    "evidence": evidence,
                    "components": tuple(
                        {
                            "latent": x,
                            "joint": mass,
                            "posterior": mass / evidence if evidence else None,
                        }
                        for x, mass in component.items()
                    ),
                }
            )
    cell = next(
        x
        for x in histories
        if x["kind"] == "outside-signal" and x["coordinate"] == selected
    )
    numerator = _sum(row["contribution"] for row in selected_rows)
    terminal_mass = _sum(row["mass"] for row in selected_rows)
    if terminal_mass is not None and terminal_mass != cell["evidence"]:
        raise ArithmeticError("complete selected mass differs; no normalization repair")
    evidence = cell["evidence"]
    return {
        "rows": rows,
        "histories": histories,
        "routing_unknown_mass": routing,
        "terminal_unknown_mass": unknown,
        "known_terminal_mass": known_mass,
        "known_subtotal_diagnostic_only": known,
        "root_material": None if routing or unknown else known,
        "selected_evidence": evidence,
        "selected_material": numerator / evidence
        if numerator is not None and evidence
        else None,
        "lifecycle_utility_CS": None,
    }


def evaluate_paired_material(
    model, *, selected_observation, selected_signal, row_limit
):
    """Evaluate ONE fixed programme under TWO full declared laws, no optimizer."""
    shape, bound, complete = _validate(model, row_limit)
    s = model.support
    if selected_observation not in s.observations or selected_signal not in s.signals:
        raise ValueError("explicit selected history must belong to complete support")
    selected = (
        s.observations.index(selected_observation),
        s.signals.index(selected_signal),
    )
    policy_sha = hashlib.sha256(
        canonical_json(_encoded(model.policy)).encode()
    ).hexdigest()
    forecast = _evaluate(s, model.policy, model.forecast, shape, selected)
    reference = _evaluate(s, model.policy, model.reference, shape, selected)
    q, r = forecast["root_material"], reference["root_material"]
    return PairedMaterialResult(
        {
            "plan_id": "A.10/A.5/A.14/B0/V9",
            "model": _encoded(model),
            "selected_history": (selected_observation, selected_signal),
            "policy_sha256": policy_sha,
            "full_policy_explicitly_complete": complete,
            "policy_reoptimized_or_reference_truth_inserted": False,
            "forecast": forecast,
            "reference": reference,
            "reference_minus_forecast_material": r - q
            if r is not None and q is not None
            else None,
            "structural_bound": bound,
            "native_law_knowledge_service_payment_preference_or_source_time_authenticated": False,
            "lifecycle_utility_CS": None,
            "scientific_budget": None,
        },
        row_limit,
    )


def paired_material_record(result):
    """Owned lossless ANALYST export; no public/passenger-view permission."""
    if type(result) is not PairedMaterialResult or result.rule != RULE:
        raise TypeError("complete typed paired material result required")
    return _encoded(result)
