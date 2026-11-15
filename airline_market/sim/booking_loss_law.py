"""A.10/A.5/A.14/B0/V9: complete DECLARED original choice/boundary law.

J_h(x,k,g) = sum_(z,u) prior(z,k,u,g) choice(a|k,u) boundary(x,c|z,k,u,g,a).
Bayes division is conditional on exactly positive evidence; no native adoption.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

from airline_market.core.hashing import canonical_json
from airline_market.sim.received_information import _text, _units
from airline_market.sim.shared_mark_attempt import _array, _encoded, _names, _owned

RULE = "finite-declared-original-choice-boundary-law-v1"
SCOPE = "before-original-choice-and-boundary"


class BookingLossLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class OriginalOffer:
    product_id: str
    fare: float
    offered: bool

    def __post_init__(self):
        _text(self.product_id)
        if type(self.fare) is not float or type(self.offered) is not bool:
            raise TypeError(
                "binary64 fare metadata and explicit offered boolean required"
            )


def _context(value):
    if isinstance(value, Mapping):
        if any(type(k) is not str for k in value):
            raise TypeError("string context keys required for lossless export")
        return _owned({k: _context(v) for k, v in value.items()})
    if type(value) in (tuple, list):
        return tuple(_context(v) for v in value)
    return _owned(value)


@dataclass(frozen=True)
class BookingLossLaw:
    original_person_label: str
    initial_states: tuple
    boundary_states: tuple
    latent_types: tuple
    ordinary_marks: tuple
    attempt_marks: tuple
    products: tuple
    initial_joint: tuple
    ordinary_likelihood: tuple
    boundary_kernel: tuple
    conditioning_context: object
    conditioning_scope: str
    conditioning_definition: str
    ordinary_mark_definition: str
    attempt_mark_definition: str
    boundary_definition: str
    definition: str
    source_ref: str

    def __post_init__(self):
        offers = tuple(self.products)
        if any(type(p) is not OriginalOffer for p in offers):
            raise TypeError("full typed original product catalogue required")
        object.__setattr__(self, "products", offers)
        for name in self.__dataclass_fields__:
            if name != "products":
                own = _context if name == "conditioning_context" else _owned
                object.__setattr__(self, name, own(getattr(self, name)))


@dataclass(frozen=True)
class BookingLossResult:
    law: BookingLossLaw
    report: object
    row_limit: int
    rule: str = RULE

    def __post_init__(self):
        object.__setattr__(self, "report", _owned(self.report))


def _validate(law):
    if type(law) is not BookingLossLaw:
        raise TypeError("typed complete original-choice/boundary law required")
    for name in (
        "initial_states",
        "boundary_states",
        "latent_types",
        "ordinary_marks",
        "attempt_marks",
    ):
        _names(getattr(law, name), name)
    for name in (
        "original_person_label",
        "conditioning_definition",
        "ordinary_mark_definition",
        "attempt_mark_definition",
        "boundary_definition",
        "definition",
        "source_ref",
    ):
        _text(getattr(law, name))
    if law.conditioning_scope != SCOPE:
        raise ValueError(
            "explicit BEFORE original choice and boundary conditioning required"
        )
    if len({p.product_id for p in law.products}) != len(law.products):
        raise ValueError("distinct full declared products required")
    z, x, k, u, g = map(
        len,
        (
            law.initial_states,
            law.boundary_states,
            law.latent_types,
            law.ordinary_marks,
            law.attempt_marks,
        ),
    )
    a = len(law.products) + 1
    _array(law.initial_joint, (z, k, u, g))
    _array(law.ordinary_likelihood, (k, u, a))
    _array(law.boundary_kernel, (z, k, u, g, a, x, 2))
    if (
        sum(
            (
                q
                for by_k in law.initial_joint
                for by_u in by_k
                for by_g in by_u
                for q in by_g
            ),
            Fraction(),
        )
        != 1
    ):
        raise ValueError("complete initial joint must sum EXACTLY one")
    for by_u in law.ordinary_likelihood:
        for row in by_u:
            if sum(row) != 1:
                raise ValueError(
                    "complete ordinary likelihood row must sum EXACTLY one"
                )
            if any(q and not offer.offered for q, offer in zip(row, law.products)):
                raise ValueError(
                    "positive unoffered action even on zero-prior type/mark"
                )
    for iz, ik, iu, ig, ia in product(range(z), range(k), range(u), range(g), range(a)):
        row = law.boundary_kernel[iz][ik][iu][ig][ia]
        if sum((q for state in row for q in state), Fraction()) != 1:
            raise ValueError("complete JOINT boundary kernel must sum EXACTLY one")
        if ia == a - 1 and any(state[1] for state in row):
            raise ValueError("outside has no ticket to cancel; full zero rows required")
    return z, x, k, u, g, a


def _limit(law, shape, row_limit):
    if type(row_limit) is not int or row_limit < 1:
        raise ValueError("positive ALL-support structural limit required")
    z, x, k, u, g, a = shape
    paths = z * k * u * g * a * x * 2
    components = a * z * k * g + a * 2 * x * k * g
    # Repeated full law, all paths, both cell families and conditional exports.
    required = 4 * _units(_encoded(law)) + 256 * (paths + components + a * 3) + 4000
    if required > row_limit:
        raise BookingLossLimit(
            "ALL input/output/zero support exceeds structural limit; no partial result"
        )
    return required


def _cell(state_names, types, marks, joint, action_index, product_id, cancelled):
    evidence = sum(joint.values(), Fraction())
    coords = tuple(
        product(range(len(state_names)), range(len(types)), range(len(marks)))
    )
    return {
        "action_index": action_index,
        "product_id": product_id,
        "outside": product_id is None,
        "cancelled": cancelled,
        "evidence": evidence,
        "components": [
            {
                "state": s,
                "type": k,
                "attempt_mark": g,
                "joint_mass": joint[s, k, g],
                "posterior": joint[s, k, g] / evidence if evidence else None,
            }
            for s, k, g in coords
        ],
        "conditional_joint": None
        if not evidence
        else tuple(
            tuple(
                tuple(joint[s, k, g] / evidence for g in range(len(marks)))
                for k in range(len(types))
            )
            for s in range(len(state_names))
        ),
    }


def _forward(law, shape, row_limit, required):
    z, x, k, u, g, a = shape
    ordinary = {
        i: dict.fromkeys(product(range(z), range(k), range(g)), Fraction())
        for i in range(a)
    }
    histories = {
        (i, c): dict.fromkeys(product(range(x), range(k), range(g)), Fraction())
        for i, c in product(range(a), range(2))
    }
    rows = []
    for iz, ik, iu, ig, ia in product(range(z), range(k), range(u), range(g), range(a)):
        prior = law.initial_joint[iz][ik][iu][ig]
        choice = law.ordinary_likelihood[ik][iu][ia]
        ordinary[ia][iz, ik, ig] += prior * choice
        for ix, ic in product(range(x), range(2)):
            boundary = law.boundary_kernel[iz][ik][iu][ig][ia][ix][ic]
            mass = prior * choice * boundary
            rows.append(
                {
                    "initial_state": iz,
                    "type": ik,
                    "ordinary_mark": iu,
                    "attempt_mark": ig,
                    "action_index": ia,
                    "boundary_state": ix,
                    "cancelled": bool(ic),
                    "prior": prior,
                    "ordinary_likelihood": choice,
                    "joint_boundary": boundary,
                    "joint_mass": mass,
                }
            )
            histories[ia, ic][ix, ik, ig] += mass
    product_ids = tuple(p.product_id for p in law.products) + (None,)
    ordinary_cells = tuple(
        _cell(
            law.initial_states,
            law.latent_types,
            law.attempt_marks,
            ordinary[i],
            i,
            product_ids[i],
            None,
        )
        for i in range(a)
    )
    boundary_cells = tuple(
        _cell(
            law.boundary_states,
            law.latent_types,
            law.attempt_marks,
            histories[i, c],
            i,
            product_ids[i],
            bool(c),
        )
        for i, c in product(range(a), range(2))
    )
    total = sum((row["joint_mass"] for row in rows), Fraction())
    if total != 1 or sum(c["evidence"] for c in boundary_cells) != 1:
        raise ArithmeticError("complete joint mass differs; no normalization")
    report = {
        "plan_id": "A.10/A.5/A.14/B0/V9",
        "rule": RULE,
        "input_digest": hashlib.sha256(
            canonical_json(_encoded(law)).encode()
        ).hexdigest(),
        "scope": "ONE declared original request conditional on full supplied I0; analyst only",
        "paths": rows,
        "ordinary_choice_cells": ordinary_cells,
        "boundary_history_cells": boundary_cells,
        "joint_mass": total,
        "structural_bound": required,
        "ordinary_and_attempt_marks_are_distinct_axes": True,
        "same_type_and_attempt_mark_carried_to_boundary": True,
        "native_source_bridge_or_native_probability_authenticated": False,
        "known_type_awareness_received_knowledge_or_preferences_adopted": False,
        "lifecycle_utility_CS_service_payoff": None,
        "scientific_budget": None,
    }
    return BookingLossResult(law, report, row_limit)


def compile_booking_loss_law(law, *, row_limit):
    shape = _validate(law)
    required = _limit(law, shape, row_limit)
    return _forward(law, shape, row_limit, required)


def booking_loss_history(result, product_id, *, cancelled):
    """Pure NEW result reader; zero evidence retains full rows and NULL joint."""
    if type(result) is not BookingLossResult or result.rule != RULE:
        raise TypeError("typed complete booking/loss result required")
    if type(cancelled) is not bool:
        raise TypeError("explicit cancellation boolean required")
    ids = tuple(p.product_id for p in result.law.products) + (None,)
    if product_id is not None and type(product_id) is not str:
        raise TypeError("declared physical product ID or outside None required")
    if product_id not in ids:
        raise ValueError("product outside complete declared support")
    return result.report["boundary_history_cells"][
        ids.index(product_id) * 2 + int(cancelled)
    ]


def booking_loss_record(result):
    if type(result) is not BookingLossResult or result.rule != RULE:
        raise TypeError("typed complete booking/loss result required")
    return _encoded(result)
