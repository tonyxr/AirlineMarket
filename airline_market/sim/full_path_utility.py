"""A.10/A.5/A.14/B0/V9/B4: declared utility on NEW complete paired paths.

Pure saved-row consumer. No native probability, knowledge, preference or CS
authentication; same-path outside accounts are not outside-intervention laws.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

from airline_market.sim.paired_history_binding import (
    RULE as PARENT_RULE,
)
from airline_market.sim.paired_history_binding import (
    PairedHistoryBinding,
    paired_history_record,
)
from airline_market.sim.received_information import _text, _units
from airline_market.sim.shared_mark_attempt import _encoded, _owned
from airline_market.sim.terminal_material import TerminalMaterialAccount

RULE = "declared-full-conditioned-path-utility-v1"
REFERENCE_SCOPE = "same-realized-path-declared-reference-not-outside-intervention"
FIELDS = (
    "material",
    "outside_material",
    "normalized_material",
    "noise_excluded",
    "outside_noise_excluded",
    "normalized_noise_excluded",
    "experienced_candidate",
    "outside_experienced_candidate",
    "normalized_candidate",
)


class FullPathUtilityLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class TerminalUtilityAccount:
    additional_utility: Fraction | None
    additional_utility_source: str
    utility_tail_closed: bool
    outside_material: TerminalMaterialAccount
    outside_additional_utility: Fraction | None
    outside_additional_utility_source: str
    outside_utility_tail_closed: bool
    followup_source: str

    def __post_init__(self):
        for value in (self.additional_utility, self.outside_additional_utility):
            if value is not None and type(value) is not Fraction:
                raise TypeError(
                    "explicit Fraction-or-unknown additional utility required"
                )
        for value in (self.utility_tail_closed, self.outside_utility_tail_closed):
            if type(value) is not bool:
                raise TypeError("explicit independent utility closures required")
        if type(self.outside_material) is not TerminalMaterialAccount:
            raise TypeError("full typed FOUR-component outside account required")
        for value in (
            self.additional_utility_source,
            self.outside_additional_utility_source,
            self.followup_source,
        ):
            _text(value)


def _accounts(value):
    if type(value) is TerminalUtilityAccount:
        return value
    if type(value) not in (tuple, list):
        raise TypeError("complete typed BOTH-stage utility accounts required")
    return tuple(_accounts(x) for x in value)


@dataclass(frozen=True)
class FullPathUtilityLaw:
    name: str
    physical_states: tuple
    latent_types: tuple
    private_marks: tuple
    action_products: tuple
    outcomes: tuple
    accounts: tuple
    definition: str
    source_ref: str

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            object.__setattr__(
                self, name, _accounts(value) if name == "accounts" else _owned(value)
            )


@dataclass(frozen=True)
class FullPathUtilityResult:
    analyst: object
    view: object
    row_limit: int
    rule: str = RULE

    def __post_init__(self):
        object.__setattr__(self, "analyst", _owned(self.analyst))
        object.__setattr__(self, "view", _owned(self.view))


def _decode(value):
    if isinstance(value, Mapping):
        if set(value) == {"numerator", "denominator"}:
            return Fraction(int(value["numerator"]), int(value["denominator"]))
        return {k: _decode(v) for k, v in value.items()}
    if type(value) in (tuple, list):
        return tuple(_decode(v) for v in value)
    return value


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _at(table, coordinate):
    for index in coordinate:
        table = table[index]
    return table


def _shape(value, dimensions, leaf):
    if not dimensions:
        leaf(value)
        return
    _check(
        type(value) is tuple and len(value) == dimensions[0],
        "complete ordered support including zero/unused rows required",
    )
    for row in value:
        _shape(row, dimensions[1:], leaf)


def _material(account, unit):
    _check(account["unit"] == unit, "same explicit material unit required")
    for text in (*account["component_sources"], account["definition"]):
        _text(text)
    _check(len(account["component_sources"]) == 4, "FOUR component sources required")
    for flag in (account["financial_tail_closed"], account["material_tail_closed"]):
        _check(type(flag) is bool, "explicit financial/material closures required")
    values = tuple(
        account["payoff"][n]
        for n in (
            "service_benefit",
            "net_ticket_payment",
            "waiting_loss",
            "cancellation_loss",
        )
    )
    _check(
        all(x is None or type(x) is Fraction for x in values),
        "FOUR explicit rational-or-unknown material components required",
    )
    _check(
        account["financial_tail_closed"] or values[1] is None,
        "open financial tail cannot contain known final payment",
    )
    diagnostic = None if None in values else values[0] - sum(values[1:], Fraction())
    return diagnostic, (
        diagnostic
        if account["financial_tail_closed"] and account["material_tail_closed"]
        else None
    )


def _product(values):
    answer = Fraction(1)
    for value in values:
        _check(
            type(value) is Fraction and value >= 0,
            "complete nonnegative exact saved factors required",
        )
        answer *= value
    return answer


def _contribution(mass, value):
    return Fraction() if mass == 0 else None if value is None else mass * value


def _difference(a, b):
    return None if a is None or b is None else a - b


def _complete(values):
    values = tuple(values)
    return None if any(v is None for v in values) else sum(values, Fraction())


def _saved_rows(parent, shape, unit):
    """Check supplied row arithmetic/links without invoking any parent forward."""
    n, k, g, a, d, h = shape
    inputs = parent["inputs"]
    programme = inputs["programme"]
    support, policy = programme["support"], programme["policy"]
    paired = parent["paired"]
    for i, role in enumerate(("forecast", "reference")):
        transport = parent["transports"][i]["analyst"]
        evidence, joint = (
            parent["original_history_evidence"][i],
            parent["precategory_joints"][i],
        )
        _check(
            type(evidence) is Fraction
            and evidence >= 0
            and transport["evidence"] == evidence
            and transport["transported_selected_mass"] == evidence
            and transport["conditional_joint"] == joint,
            "same saved original evidence and full transported joint required",
        )
        _check((evidence == 0) == (joint is None), "zero original evidence stays NULL")
        if joint is not None:
            _shape(joint, (n, k, g), lambda x: _product((x,)))
            _check(
                sum(
                    (_at(joint, c) for c in product(range(n), range(k), range(g))),
                    Fraction(),
                )
                == 1,
                "complete transported unit joint required; no normalization",
            )
    _check(
        (paired is None) == any(j is None for j in parent["precategory_joints"])
        and parent["paired_evaluator_called"] is (paired is not None)
        and parent["original_conditioning_calls"] == 2,
        "undefined upstream pair stays NULL without fabricated prior",
    )
    if paired is None:
        return
    p = paired["analyst"]
    model = p["model"]
    _check(
        model["support"] == support
        and model["policy"] == policy
        and p["full_policy_explicitly_complete"] is True
        and p["policy_reoptimized_or_reference_truth_inserted"] is False,
        "ONE unchanged full fixed policy and support required",
    )
    selected = (
        support["observations"].index(inputs["observation"]["category"]),
        support["signals"].index(inputs["observation"]["signal"]),
    )
    _check(
        p["selected_history"]
        == (inputs["observation"]["category"], inputs["observation"]["signal"]),
        "same received history required",
    )
    for i, role in enumerate(("forecast", "reference")):
        law, saved = inputs[role], p[role]
        future = model[role]
        _check(
            future["initial_joint"] == parent["precategory_joints"][i],
            "same pre-category joint, not a copied posterior",
        )
        _check(
            all(
                future[name] == law[name]
                for name in (
                    "name",
                    "transition_signal",
                    "terminal_probabilities",
                    "terminal_accounts",
                    "definition",
                    "source_ref",
                )
            ),
            "complete same terminal law required",
        )
        expected = []
        for y, kind, mark in product(range(n), range(k), range(g)):
            obs = support["observations"].index(support["type_observation"][kind])
            for action, outcome in product(range(a), range(d)):
                expected.append(("own", (y, kind, mark, obs, y, -1, action, outcome)))
            for yn, signal, action, outcome in product(
                range(n), range(h), range(a), range(d)
            ):
                expected.append(
                    ("interline", (y, kind, mark, obs, yn, signal, action, outcome))
                )
        _check(
            tuple((r["stage"], r["coordinate"]) for r in saved["rows"])
            == tuple(expected),
            "ALL ordered saved terminal rows required",
        )
        selected_rows = []
        for row in saved["rows"]:
            y, kind, mark, obs, yn, signal, action, outcome = row["coordinate"]
            stage = int(row["stage"] == "interline")
            own = policy["own_program"][obs][mark]
            factors = (
                future["initial_joint"][y][kind][mark],
                own[-1] if stage else own[action],
                future["transition_signal"][y][kind][mark][yn][signal]
                if stage
                else Fraction(1),
                policy["interline_program"][obs][signal][mark][action]
                if stage
                else Fraction(1),
                future["terminal_probabilities"][stage][yn][kind][mark][action][
                    outcome
                ],
            )
            executable = (
                support["interline_enabled"]
                if stage
                else action != a - 1 or not support["interline_enabled"]
            )
            mass = _product(factors) if executable else Fraction()
            _check(
                row["factors"] == factors
                and row["mass"] == mass
                and row["terminal_executable"] is executable,
                "same complete factors, execution and mass required",
            )
            account = future["terminal_accounts"][stage][yn][kind][mark][action][
                outcome
            ]
            diagnostic, value = _material(account, unit)
            _check(
                row["account"] == account
                and row["material_components_diagnostic"] == diagnostic
                and row["material_if_closed"] == value
                and row["contribution"] == _contribution(mass, value),
                "same saved FOUR-component material arithmetic required",
            )
            if stage and (obs, signal) == selected:
                selected_rows.append(row)
        evidence = sum((r["mass"] for r in selected_rows), Fraction())
        numerator = _complete(r["contribution"] for r in selected_rows)
        _check(
            sum((r["mass"] for r in saved["rows"]), Fraction()) == 1
            and saved["routing_unknown_mass"] == 0
            and saved["root_material"]
            == _complete(r["contribution"] for r in saved["rows"])
            and saved["selected_evidence"] == evidence
            and saved["selected_material"]
            == (numerator / evidence if numerator is not None and evidence else None),
            "complete root and selected sums/evidence required; no repair",
        )


def evaluate_full_path_utility(
    binding,
    forecast,
    reference,
    *,
    alpha,
    material_unit,
    utility_unit,
    material_additional_nonoverlap,
    utility_origin_and_incidence,
    definition,
    reference_scope,
    row_limit,
):
    """Project NEW complete saved paired rows; never replay any parent producer."""
    if type(binding) is not PairedHistoryBinding or binding.rule != PARENT_RULE:
        raise TypeError("complete typed paired-history binding required")
    if type(alpha) is not Fraction or alpha <= 0:
        raise ValueError("explicit fixed positive rational alpha required")
    if type(row_limit) is not int or row_limit <= 0:
        raise ValueError("positive structural limit required")
    for text in (
        material_unit,
        utility_unit,
        material_additional_nonoverlap,
        utility_origin_and_incidence,
        definition,
    ):
        _text(text)
    _check(
        material_unit != utility_unit and reference_scope == REFERENCE_SCOPE,
        "distinct explicit units and same-path reference scope required",
    )
    source = paired_history_record(binding)
    parent = _decode(source)["analyst"]
    s = parent["inputs"]["programme"]["support"]
    _check(s["payoff_unit"] == material_unit, "parent material unit must match exactly")
    n, k, g, a, d, h = (
        len(s["physical_states"]),
        len(s["latent_types"]),
        len(s["private_marks"]),
        len(s["products"]) + 1,
        len(s["outcomes"]),
        len(s["signals"]),
    )
    count = n * k * g * a * d * (1 + n * h)
    declarations = (forecast, reference)
    for i, law in enumerate(declarations):
        if type(law) is not FullPathUtilityLaw:
            raise TypeError("TWO complete typed utility declarations required")
        for text in (law.name, law.definition, law.source_ref):
            _text(text)
        _check(
            law.name == parent["inputs"][("forecast", "reference")[i]]["name"],
            "ordered utility and original law names must match",
        )
        _check(
            (
                law.physical_states,
                law.latent_types,
                law.private_marks,
                law.action_products,
                law.outcomes,
            )
            == (
                s["physical_states"],
                s["latent_types"],
                s["private_marks"],
                (*s["products"], None),
                s["outcomes"],
            ),
            "complete named axes must match",
        )

        def leaf(account):
            if type(account) is not TerminalUtilityAccount:
                raise TypeError("EVERY terminal utility account required")
            _check(
                account.outside_material.unit == material_unit,
                "outside material unit must match exactly",
            )

        _shape(law.accounts, (2, n, k, g, a, d), leaf)
    encoded_declarations = _encoded(declarations)
    bound = 3 * _units(source) + 4 * _units(encoded_declarations) + 2048 * count + 20000
    if bound > row_limit:
        raise FullPathUtilityLimit(
            "ALL source, utility declarations and proposed output exceed structural limit"
        )
    _saved_rows(parent, (n, k, g, a, d, h), material_unit)
    paired = parent["paired"]
    outputs = None
    if paired is not None:
        outputs = {}
        p = paired["analyst"]
        selected = (
            s["observations"].index(p["selected_history"][0]),
            s["signals"].index(p["selected_history"][1]),
        )
        for role, law in zip(("forecast", "reference"), declarations):
            rows = []
            for index, row in enumerate(p[role]["rows"]):
                _, kind, mark, obs, yn, signal, action, outcome = row["coordinate"]
                stage = int(row["stage"] == "interline")
                account = _at(law.accounts, (stage, yn, kind, mark, action, outcome))
                outside = _decode(_encoded(account.outside_material))
                z0d, z0 = _material(outside, material_unit)
                z = row["material_if_closed"]
                noise = None if z is None else alpha * z
                outside_noise = None if z0 is None else alpha * z0
                e = account.additional_utility if account.utility_tail_closed else None
                e0 = (
                    account.outside_additional_utility
                    if account.outside_utility_tail_closed
                    else None
                )
                experience = None if noise is None or e is None else noise + e
                outside_experience = (
                    None if outside_noise is None or e0 is None else outside_noise + e0
                )
                values = dict(
                    zip(
                        FIELDS,
                        (
                            z,
                            z0,
                            _difference(z, z0),
                            noise,
                            outside_noise,
                            _difference(noise, outside_noise),
                            experience,
                            outside_experience,
                            _difference(experience, outside_experience),
                        ),
                    )
                )
                rows.append(
                    {
                        "parent_row_index": index,
                        "stage": row["stage"],
                        "coordinate": row["coordinate"],
                        "mass": row["mass"],
                        "selected": bool(stage and (obs, signal) == selected),
                        "utility_account_coordinate": (
                            stage,
                            yn,
                            kind,
                            mark,
                            action,
                            outcome,
                        ),
                        "outside_material_diagnostic": z0d,
                        "values": values,
                        "contributions": {
                            key: _contribution(row["mass"], value)
                            for key, value in values.items()
                        },
                    }
                )
            summaries = {}
            for scope in ("root", "selected"):
                subset = rows if scope == "root" else [r for r in rows if r["selected"]]
                evidence = (
                    Fraction(1) if scope == "root" else p[role]["selected_evidence"]
                )
                _check(
                    sum((r["mass"] for r in subset), Fraction()) == evidence,
                    "same scope evidence required",
                )
                fields = {}
                for key in FIELDS:
                    contributions = tuple(r["contributions"][key] for r in subset)
                    total = _complete(contributions)
                    fields[key] = {
                        "mean": total / evidence
                        if evidence and total is not None
                        else None,
                        "known_subtotal_diagnostic_only": sum(
                            (x for x in contributions if x is not None), Fraction()
                        ),
                        "unknown_mass": sum(
                            (r["mass"] for r in subset if r["values"][key] is None),
                            Fraction(),
                        ),
                    }
                summaries[scope] = {"evidence": evidence, "fields": fields}
            outputs[role] = {"rows": rows, "summaries": summaries}
    return FullPathUtilityResult(
        {
            "plan_id": "A.10/A.5/A.14/B0/V9/B4",
            "source": source,
            "utility_declarations": encoded_declarations,
            "alpha": alpha,
            "material_unit": material_unit,
            "utility_unit": utility_unit,
            "material_additional_nonoverlap": material_additional_nonoverlap,
            "utility_origin_and_incidence": utility_origin_and_incidence,
            "definition": definition,
            "reference_scope": reference_scope,
            "expectations": outputs,
            "structural_bound": bound,
            "parent_producer_or_probability_forward_called": False,
            "native_law_knowledge_preferences_service_or_payment_authenticated": False,
            "lifecycle_CS": None,
            "scientific_budget": None,
        },
        binding.view,
        row_limit,
    )


def full_path_utility_record(result):
    if type(result) is not FullPathUtilityResult or result.rule != RULE:
        raise TypeError("complete typed full-path utility result required")
    return _encoded(result)


def full_path_utility_view(result):
    if type(result) is not FullPathUtilityResult or result.rule != RULE:
        raise TypeError("complete typed full-path utility result required")
    return _encoded(result.view)
