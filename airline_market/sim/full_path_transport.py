"""A.10/A.5/A.14/B0/V9: declared full conditioned-path transport.

Use Z/U before marginalizing them; carry K/G without reset or redraw. This
operator supplies neither a native future law nor passenger knowledge or CS.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import product

from airline_market.sim.booking_loss_binding import (
    bind_choice_boundary,
    choice_boundary_record,
)
from airline_market.sim.booking_loss_law import (
    _limit,
    _validate,
    booking_loss_history,
    booking_loss_record,
    compile_booking_loss_law,
)
from airline_market.sim.received_information import _text, _units
from airline_market.sim.shared_mark_attempt import _array, _encoded, _names, _owned

RULE = "declared-full-conditioned-path-transport-v1"
SCOPE = "after-original-choice-boundary-before-later-observation"


class FullPathTransportLimit(RuntimeError):
    pass


@dataclass(frozen=True)
class FullPathKernel:
    initial_states: tuple
    boundary_states: tuple
    latent_types: tuple
    ordinary_marks: tuple
    attempt_marks: tuple
    action_products: tuple
    future_states: tuple
    probabilities: tuple
    conditioning_scope: str
    definition: str
    source_ref: str

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _owned(getattr(self, name)))


@dataclass(frozen=True)
class FullPathTransport:
    analyst: object
    row_limit: int
    rule: str = RULE

    def __post_init__(self):
        object.__setattr__(self, "analyst", _owned(self.analyst))


def _prepare(law, kernel, row_limit, source_inputs=None):
    shape = _validate(law)
    if type(kernel) is not FullPathKernel:
        raise TypeError("complete typed full-path conditional kernel required")
    for field in (
        "initial_states",
        "boundary_states",
        "latent_types",
        "ordinary_marks",
        "attempt_marks",
    ):
        if getattr(kernel, field) != getattr(law, field):
            raise ValueError("same full named parent axes required")
    if kernel.action_products != tuple(p.product_id for p in law.products) + (None,):
        raise ValueError("complete ordered product/outside action support required")
    _names(kernel.future_states, "future_states")
    for field in ("definition", "source_ref"):
        _text(getattr(kernel, field))
    if kernel.conditioning_scope != SCOPE:
        raise ValueError("explicit BEFORE later observation scope required")
    z, x, k, u, g, a = shape
    y = len(kernel.future_states)
    _array(kernel.probabilities, (z, k, u, g, a, x, 2, y))
    for iz, ik, iu, ig, ia, ix, ic in product(
        range(z), range(k), range(u), range(g), range(a), range(x), range(2)
    ):
        if sum(kernel.probabilities[iz][ik][iu][ig][ia][ix][ic]) != 1:
            raise ValueError("EVERY full-path kernel row must sum EXACTLY one")
    backend = _limit(law, shape, row_limit)
    paths = z * k * u * g * a * x * 2
    # Full parent/binding exports, ALL proposed terms/components and metadata.
    required = (
        8 * backend
        + 8 * _units(_encoded(kernel))
        + 256 * (paths * y + y * k * g + k * g)
        + 16 * _units(_encoded(source_inputs))
        + 10000
    )
    if required > row_limit:
        raise FullPathTransportLimit(
            "ALL source/law/kernel/zero/output support exceeds structural limit"
        )
    return required


def _fraction(value):
    # Exact lossless parent compiler export, never Float64/decimal coercion.
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def _transport(law, kernel, rows, history, parent, row_limit, required, bound):
    evidence = history["evidence"]
    action = history["action_index"]
    cancelled = history["cancelled"]
    y, k, g = map(len, (kernel.future_states, law.latent_types, law.attempt_marks))
    coords = tuple(product(range(y), range(k), range(g)))
    joint = dict.fromkeys(coords, Fraction())
    terms = []
    for row in rows:
        iz, ik, iu, ig, ia, ix = (
            row[n]
            for n in (
                "initial_state",
                "type",
                "ordinary_mark",
                "attempt_mark",
                "action_index",
                "boundary_state",
            )
        )
        ic = int(row["cancelled"])
        selected = ia == action and row["cancelled"] == cancelled
        mass = _fraction(row["joint_mass"])
        for iy, probability in enumerate(
            kernel.probabilities[iz][ik][iu][ig][ia][ix][ic]
        ):
            weighted = mass * probability if selected else Fraction()
            joint[iy, ik, ig] += weighted
            terms.append(
                {
                    "parent_coordinate": (iz, ik, iu, ig, ia, ix, bool(ic)),
                    "future_state": iy,
                    "selected_history": selected,
                    "parent_joint_mass": mass,
                    "conditional_probability": probability,
                    "selected_joint_mass": weighted,
                    "conditional_mass": weighted / evidence if evidence else None,
                }
            )
    total = sum(joint.values(), Fraction())
    if total != evidence:
        raise ArithmeticError("transport mass differs; no normalization repair")
    carried = tuple(
        tuple(
            sum((joint[iy, ik, ig] for iy in range(y)), Fraction()) for ig in range(g)
        )
        for ik in range(k)
    )
    parent_kg = tuple(
        tuple(
            sum(
                (
                    c["joint_mass"]
                    for c in history["components"]
                    if c["type"] == ik and c["attempt_mark"] == ig
                ),
                Fraction(),
            )
            for ig in range(g)
        )
        for ik in range(k)
    )
    if carried != parent_kg:
        raise ArithmeticError("SAME K/G joint must be carried, not redrawn")
    conditional = (
        None
        if not evidence
        else tuple(
            tuple(
                tuple(joint[iy, ik, ig] / evidence for ig in range(g))
                for ik in range(k)
            )
            for iy in range(y)
        )
    )
    analyst = {
        "plan_id": "A.10/A.5/A.14/B0/V9",
        "rule": RULE,
        "parent": parent,
        "kernel": _encoded(kernel),
        "selected_history": history,
        "terms": terms,
        "components": tuple(
            {
                "future_state": iy,
                "type": ik,
                "attempt_mark": ig,
                "joint_mass": joint[iy, ik, ig],
                "conditional_mass": joint[iy, ik, ig] / evidence if evidence else None,
            }
            for iy, ik, ig in coords
        ),
        "evidence": evidence,
        "transported_selected_mass": total,
        "conditional_joint": conditional,
        "carried_KG_joint": carried,
        "structural_bound": required,
        "source_prefix_binding_executed": bound,
        "all_parent_paths_and_kernel_rows_retained": True,
        "same_type_and_attempt_mark_carried": True,
        "downstream_choice_compiler_called": False,
        "native_probability_received_knowledge_preferences_authenticated": False,
        "lifecycle_utility_CS_service_payoff": None,
        "scientific_budget": None,
    }
    return FullPathTransport(analyst, row_limit)


def compile_full_path_transport(law, kernel, product_id, *, cancelled, row_limit):
    """ONE new parent compilation, then full-path transport on NEW declarations."""
    required = _prepare(law, kernel, row_limit)
    # Validate selection before any parent forward, including bool vs int.
    if type(cancelled) is not bool:
        raise TypeError("explicit cancellation boolean required")
    if product_id is not None and type(product_id) is not str:
        raise TypeError("physical product ID or outside None required")
    if product_id not in kernel.action_products:
        raise ValueError("selected product outside declared support")
    result = compile_booking_loss_law(law, row_limit=row_limit)
    history = booking_loss_history(result, product_id, cancelled=cancelled)
    parent = booking_loss_record(result)
    return _transport(
        law,
        kernel,
        parent["report"]["paths"],
        history,
        parent,
        row_limit,
        required,
        False,
    )


def bind_full_path_transport(
    law,
    kernel,
    ordinary_input,
    native_periods,
    source_record,
    association,
    *,
    row_limit,
):
    """ONE fresh native-prefix/parent association; no old result/producer replay.

    Retain all earlier people/periods and the exact closing prefix. The inherited
    association observes original action/cancellation only; all X remains latent.
    """
    periods = tuple(native_periods)
    required = _prepare(
        law,
        kernel,
        row_limit,
        {
            "ordinary": ordinary_input,
            "periods": periods,
            "source": source_record,
            "association": association,
        },
    )
    binding = bind_choice_boundary(
        law, ordinary_input, periods, source_record, association, row_limit=row_limit
    )
    parent = choice_boundary_record(binding)
    return _transport(
        law,
        kernel,
        binding.analyst["law_result"]["report"]["paths"],
        binding.analyst["selected_history"],
        parent,
        row_limit,
        required,
        True,
    )


def full_path_transport_record(result):
    """Owned lossless reader; invokes no compiler or native source."""
    if type(result) is not FullPathTransport or result.rule != RULE:
        raise TypeError("complete typed full-path transport required")
    return _encoded(result)


def transported_joint(result):
    """Y/K/G joint BEFORE later observation, or NULL for zero upstream evidence."""
    if type(result) is not FullPathTransport or result.rule != RULE:
        raise TypeError("complete typed full-path transport required")
    return result.analyst["conditional_joint"]
