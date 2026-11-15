"""A.10/A.5/A.14/B0/V9: one DECLARED finite latent-type attempt.

Observation-category policies carry the SAME type/mark through own outside.
This is not a native probability, knowledge, preference or welfare model.
"""

import hashlib
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

from airline_market.core.hashing import canonical_json
from airline_market.core.snapshot import deep_freeze
from airline_market.sim.received_information import _text, _units
from airline_market.sim.shared_mark_attempt import (
    AttemptOffer,
    _array,
    _encoded,
    _names,
    _owned,
)

RULE = "finite-declared-latent-type-observation-attempt-v1"


class TypeObservationLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class TypeObservationAttempt:
    physical_states: tuple
    latent_types: tuple
    private_marks: tuple
    observations: tuple
    type_observation: tuple
    signals: tuple
    own_menus: tuple
    interline_menus: tuple
    product_catalogue_disclosed: bool
    interline_enabled: bool
    initial_joint: tuple
    own_program: tuple
    next_state_signal: tuple
    interline_program: tuple
    conditioning_view: object
    conditioning_definition: str
    observation_definition: str
    private_mark_interpretation: str
    definition: str
    source_ref: str

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _owned(getattr(self, name)))


@dataclass(frozen=True)
class TypeObservationResult:
    model: TypeObservationAttempt
    report: object
    row_limit: int
    rule: str = RULE

    def __post_init__(self):
        object.__setattr__(self, "report", deep_freeze(self.report))


def _validate(model):
    if type(model) is not TypeObservationAttempt:
        raise TypeError("complete typed latent-type attempt required")
    for name in ("physical_states", "latent_types", "private_marks", "observations"):
        _names(getattr(model, name), name)
    for name in (
        "conditioning_definition",
        "observation_definition",
        "private_mark_interpretation",
        "definition",
        "source_ref",
    ):
        _text(getattr(model, name))
    n, k, g, o = (
        len(getattr(model, name))
        for name in (
            "physical_states",
            "latent_types",
            "private_marks",
            "observations",
        )
    )
    if len(model.type_observation) != k or any(
        x not in model.observations for x in model.type_observation
    ):
        raise ValueError("complete deterministic observation map for ALL latent types")
    if (
        not model.signals
        or len(set(model.signals)) != len(model.signals)
        or None not in model.signals
    ):
        raise ValueError("complete distinct signals including explicit absence None")
    for signal in model.signals:
        if signal is not None:
            _text(signal)
    for b in (model.product_catalogue_disclosed, model.interline_enabled):
        if type(b) is not bool:
            raise TypeError("explicit catalogue/interline booleans required")
    s = len(model.signals)
    if (
        len(model.own_menus) != o
        or len(model.interline_menus) != o
        or any(len(x) != s for x in model.interline_menus)
    ):
        raise ValueError("complete observation and signal menus required")
    menus = list(model.own_menus) + [m for row in model.interline_menus for m in row]
    catalogue = None
    for menu in menus:
        names = []
        for offer in menu:
            if type(offer) is not AttemptOffer or type(offer.fare) is not float:
                raise TypeError(
                    "typed full offer catalogue with binary64 fare metadata required"
                )
            _text(offer.product_id)
            if any(
                type(b) is not bool
                for b in (offer.offered, offer.disclose_fare, offer.disclose_offer)
            ):
                raise TypeError("explicit offered and visibility booleans required")
            if not model.product_catalogue_disclosed and (
                offer.disclose_fare or offer.disclose_offer
            ):
                raise ValueError("visibility requires declared catalogue disclosure")
            names.append(offer.product_id)
        if len(set(names)) != len(names):
            raise ValueError("distinct full products required")
        if catalogue is None:
            catalogue = tuple(names)
        elif tuple(names) != catalogue:
            raise ValueError(
                "same full ordered catalogue for ALL observation/signal menus"
            )
    p = len(catalogue) + 1
    _array(model.initial_joint, (n, k, g))
    _array(model.own_program, (o, g, p))
    _array(model.next_state_signal, (n, k, g, n, s))
    _array(model.interline_program, (o, s, g, p))
    if sum(q for state in model.initial_joint for kind in state for q in kind) != 1:
        raise ValueError("full initial X/K/G joint must sum exactly one")
    for physical in model.next_state_signal:
        for kind in physical:
            for mark in kind:
                if sum(q for state in mark for q in state) != 1:
                    raise ValueError(
                        "every complete joint next-state/signal row must sum exactly one"
                    )
    for obs in range(o):
        programs = [(model.own_program[obs], model.own_menus[obs])]
        programs.extend(
            (model.interline_program[obs][sig], model.interline_menus[obs][sig])
            for sig in range(s)
        )
        for program, menu in programs:
            for row in program:
                if sum(row) != 1:
                    raise ValueError(
                        "every action row including unreachable support must sum exactly one"
                    )
                if any(q and not offer.offered for q, offer in zip(row, menu)):
                    raise ValueError(
                        "positive action on unoffered product, including off-path rows"
                    )
    return n, k, g, o, s, p


def _limit(model, shape, row_limit):
    if type(row_limit) is not int or row_limit < 1:
        raise ValueError("positive ALL-support structural limit required")
    n, k, g, o, s, p = shape
    latent = n * k * g
    cells = o * (1 + p + s + s * p)
    # Includes full repeated model/rule/limit context and all zero support.
    # 256 is above each encoded row/component's structural field count.
    upper = (
        2 * _units(_encoded(model))
        + 256
        * (latent * p + latent * n * s + o * s * latent * p + cells * (1 + latent))
        + 2000
    )
    if upper > row_limit:
        raise TypeObservationLimit(
            "ALL input/output/zero support exceeds structural limit; no partial result"
        )


def _forward(model, shape, row_limit):
    n, k, g, o, s, p = shape
    coords = tuple(product(range(n), range(k), range(g)))
    obs_index = {name: i for i, name in enumerate(model.observations)}
    observed = tuple(obs_index[name] for name in model.type_observation)
    cells = {}
    for obs in range(o):
        keys = [("initial", obs, None, None)]
        keys += [("own", obs, None, a) for a in range(p)]
        keys += [("reach", obs, sig, None) for sig in range(s)]
        keys += [("interline", obs, sig, a) for sig in range(s) for a in range(p)]
        for key in keys:
            cells[key] = dict.fromkeys(coords, Fraction())
    own, transitions, later = [], [], []
    for x, kind, mark in coords:
        obs = observed[kind]
        initial = model.initial_joint[x][kind][mark]
        cells["initial", obs, None, None][x, kind, mark] += initial
        for action in range(p):
            q = model.own_program[obs][mark][action]
            mass = initial * q
            own.append(
                {
                    "physical": x,
                    "type": kind,
                    "mark": mark,
                    "observation": obs,
                    "action": action,
                    "initial": initial,
                    "action_probability": q,
                    "mass": mass,
                    "interline_reach": model.interline_enabled and action == p - 1,
                }
            )
            cells["own", obs, None, action][x, kind, mark] += mass
        outside = initial * model.own_program[obs][mark][p - 1]
        for nxt, sig in product(range(n), range(s)):
            q = model.next_state_signal[x][kind][mark][nxt][sig]
            mass = outside * q if model.interline_enabled else Fraction()
            transitions.append(
                {
                    "physical": x,
                    "type": kind,
                    "mark": mark,
                    "observation": obs,
                    "next_physical": nxt,
                    "signal": sig,
                    "own_outside_joint": outside,
                    "joint_kernel": q,
                    "enabled": model.interline_enabled,
                    "mass": mass,
                }
            )
            cells["reach", obs, sig, None][nxt, kind, mark] += mass
    for obs, sig, x, kind, mark, action in product(
        range(o), range(s), range(n), range(k), range(g), range(p)
    ):
        joint = cells["reach", obs, sig, None][x, kind, mark]
        q = model.interline_program[obs][sig][mark][action]
        mass = joint * q
        later.append(
            {
                "observation": obs,
                "signal": sig,
                "physical": x,
                "type": kind,
                "mark": mark,
                "action": action,
                "reaching_joint_mass": joint,
                "action_probability": q,
                "mass": mass,
            }
        )
        cells["interline", obs, sig, action][x, kind, mark] += mass
    beliefs = []
    for key, weights in cells.items():
        evidence = sum(weights.values(), Fraction())
        beliefs.append(
            {
                "history": key,
                "evidence": evidence,
                "components": [
                    {
                        "physical": x,
                        "type": kind,
                        "mark": mark,
                        "joint_mass": weights[x, kind, mark],
                        "posterior": weights[x, kind, mark] / evidence
                        if evidence
                        else None,
                    }
                    for x, kind, mark in coords
                ],
            }
        )
    terminal = sum(
        (r["mass"] for r in own if not r["interline_reach"]), Fraction()
    ) + sum((r["mass"] for r in later), Fraction())
    if terminal != 1:
        raise ArithmeticError("complete terminal mass differs; no normalization")
    report = {
        "plan_id": "A.10/A.5/A.14/B0/V9",
        "rule": RULE,
        "input_digest": hashlib.sha256(
            canonical_json(_encoded(model)).encode()
        ).hexdigest(),
        "own_rows": own,
        "transition_signal_rows": transitions,
        "interline_rows": later,
        "history_cells": beliefs,
        "terminal_mass": terminal,
        "same_type_and_private_mark_carried": True,
        "policy_indexing": "declared-observation/history/private-mark-only",
        "source_receipt_bridge_executed": False,
        "native_probability_received_knowledge_preferences_or_utility_authenticated": False,
        "lifecycle_utility_CS_service_payoff": None,
        "scientific_budget": None,
    }
    return TypeObservationResult(model, report, row_limit)


def compile_type_observation_attempt(model, *, row_limit):
    shape = _validate(model)
    _limit(model, shape, row_limit)
    return _forward(model, shape, row_limit)


def type_observation_record(result):
    if type(result) is not TypeObservationResult or result.rule != RULE:
        raise TypeError("typed declared type-observation result required")
    return _encoded(result)
