"""A.10/A.5/A.14/B0/V9: NEW complete declared original-person law units."""

import importlib.util
import json
import math
import os
import pickle
import sys
from dataclasses import replace
from fractions import Fraction as F
from itertools import product
from pathlib import Path

import pytest

from airline_market.sim.shared_mark_attempt import _encoded

if os.environ.get("BOOKING_LOSS_SOURCE_DIR"):
    spec = importlib.util.spec_from_file_location(
        "airline_market.sim.booking_loss_law",
        Path(os.environ["BOOKING_LOSS_SOURCE_DIR"]) / "booking_loss_law.py",
    )
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
from airline_market.sim import booking_loss_law as m

LIMIT = 10_000_000


def law(scale=F(1)):
    # NEW 94974 logical label only: no seed/native key/RNG/source or old fixture.
    prior_k = (F(2, 5), F(3, 5), F())
    z1 = (F(1, 3), F(4, 5), F(1, 2))
    ug = (
        ((F(1, 5), F(3, 10)), (F(2, 5), F(1, 10))),
        ((F(1, 4), F(1, 4)), (F(), F(1, 2))),
        ((F(1, 4), F(1, 4)), (F(1, 4), F(1, 4))),
    )
    initial = tuple(
        tuple(
            tuple(
                tuple(
                    prior_k[k] * (z1[k] if z else 1 - z1[k]) * ug[k][u][g]
                    for g in range(2)
                )
                for u in range(2)
            )
            for k in range(3)
        )
        for z in range(2)
    )
    choices = (
        ((F(1, 2), F(1, 4), F(1, 4)), (F(1, 6), F(1, 3), F(1, 2))),
        ((F(1, 5), F(2, 5), F(2, 5)), (F(3, 5), F(1, 5), F(1, 5))),
        ((F(1, 3),) * 3, (F(1, 3),) * 3),
    )

    def kernel(z, a, x, c):
        cancel = (
            scale * ((F(1, 4), F(2, 3))[z] if a == 0 else (F(1, 3), F(3, 4))[z])
            if a < 2
            else F()
        )
        next1 = (F(2, 5), F(4, 5))[z] if c else (F(1, 5), F(3, 5))[z]
        return (cancel if c else 1 - cancel) * (next1 if x else 1 - next1)

    boundary = tuple(
        tuple(
            tuple(
                tuple(
                    tuple(
                        tuple(
                            tuple(kernel(z, a, x, c) for c in range(2))
                            for x in range(2)
                        )
                        for a in range(3)
                    )
                    for _ in range(2)
                )
                for _ in range(2)
            )
            for _ in range(3)
        )
        for z in range(2)
    )
    return m.BookingLossLaw(
        original_person_label="logical-person-94974",
        initial_states=("z0", "z1"),
        boundary_states=("x0", "x1"),
        latent_types=("k0", "k1", "zero-k"),
        ordinary_marks=("u0", "u1"),
        attempt_marks=("g0", "g1"),
        products=(
            m.OriginalOffer("p0", -0.0, True),
            m.OriginalOffer("p1", -12.5, True),
        ),
        initial_joint=initial,
        ordinary_likelihood=choices,
        boundary_kernel=boundary,
        conditioning_context={
            "context": "NEW fixed pre-choice context, no other varying records",
            "known_label": None,
            "unknown": None,
            "signed": -0.0,
            "tiny": F(1, 10**900),
        },
        conditioning_scope="before-original-choice-and-boundary",
        conditioning_definition="Complete supplied I0 excludes this choice and cancellation; declaration not source authentication",
        ordinary_mark_definition="U is a distinct descriptive finite mark",
        attempt_mark_definition="G joint with U, not an inferred native Gumbel law",
        boundary_definition="Complete declared JOINT X/cancellation, not a native resolution operator",
        definition="NEW finite conditional one-original-request law",
        source_ref="synthetic-94974",
    )


def compile_(value=None, **kwargs):
    return m.compile_booking_loss_law(
        law() if value is None else value, row_limit=kwargs.get("row_limit", LIMIT)
    )


def change(value, indices, replacement):
    if not indices:
        return replacement
    rows = list(value)
    i, *rest = indices
    rows[i] = change(rows[i], rest, replacement)
    return tuple(rows)


def test_purchase_and_cancellation_joint_selection():
    result = compile_()
    purchase = result.report["ordinary_choice_cells"][0]
    selected = m.booking_loss_history(result, "p0", cancelled=True)
    assert purchase["evidence"] == F(28, 75)
    assert sum(r["posterior"] for r in purchase["components"] if r["type"] == 0) == F(
        5, 14
    )
    assert selected["evidence"] == F(259, 1350)
    assert sum(r["posterior"] for r in selected["components"] if r["type"] == 0) == F(
        10, 37
    )
    assert sum(
        r["posterior"] for r in selected["components"] if r["attempt_mark"] == 0
    ) == F(67, 296)
    assert len(selected["components"]) == 12
    assert all(r["posterior"] == 0 for r in selected["components"] if r["type"] == 2)
    assert (
        result.report["native_source_bridge_or_native_probability_authenticated"]
        is False
    )


def test_independent_complete_path_reference_and_tower():
    value = law()
    result = compile_(value)
    expected = {
        (a, c): {(x, k, g): F() for x, k, g in product(range(2), range(3), range(2))}
        for a, c in product(range(3), range(2))
    }
    paths = []
    for z, k, u, g, a, x, c in product(
        range(2), range(3), range(2), range(2), range(3), range(2), range(2)
    ):
        q = (
            value.initial_joint[z][k][u][g]
            * value.ordinary_likelihood[k][u][a]
            * value.boundary_kernel[z][k][u][g][a][x][c]
        )
        expected[a, c][x, k, g] += q
        paths.append((z, k, u, g, a, x, bool(c), q))
    actual = [
        tuple(
            r[key]
            for key in (
                "initial_state",
                "type",
                "ordinary_mark",
                "attempt_mark",
                "action_index",
                "boundary_state",
                "cancelled",
                "joint_mass",
            )
        )
        for r in result.report["paths"]
    ]
    assert actual == paths and len(actual) == 288
    assert sum(row[-1] for row in paths) == result.report["joint_mass"] == 1
    cells = result.report["boundary_history_cells"]
    assert len(cells) == 6 and sum(c["evidence"] for c in cells) == 1
    for cell in cells:
        a, c = cell["action_index"], int(cell["cancelled"])
        for row in cell["components"]:
            key = row["state"], row["type"], row["attempt_mark"]
            assert row["joint_mass"] == expected[a, c][key]
            if cell["evidence"]:
                assert row["posterior"] * cell["evidence"] == row["joint_mass"]
            else:
                assert row["posterior"] is None
    for a, k, g in product(range(3), range(3), range(2)):
        before = sum(
            row["joint_mass"]
            for row in result.report["ordinary_choice_cells"][a]["components"]
            if row["type"] == k and row["attempt_mark"] == g
        )
        after = sum(
            row["joint_mass"]
            for c in cells
            if c["action_index"] == a
            for row in c["components"]
            if row["type"] == k and row["attempt_mark"] == g
        )
        assert before == after


@pytest.mark.parametrize("scale", [F(1, 10**800), F()])
def test_tiny_positive_and_exact_zero_evidence(scale):
    result = compile_(law(scale))
    selected = m.booking_loss_history(result, "p0", cancelled=True)
    assert selected["evidence"] == F(259, 1350) * scale
    if scale:
        assert selected["evidence"] > 0 and sum(
            r["posterior"] for r in selected["components"] if r["type"] == 0
        ) == F(10, 37)
        assert selected["conditional_joint"] is not None
    else:
        assert selected["conditional_joint"] is None and all(
            r["posterior"] is None for r in selected["components"]
        )
    impossible = m.booking_loss_history(result, None, cancelled=True)
    assert impossible["evidence"] == 0 and impossible["conditional_joint"] is None
    assert len(impossible["components"]) == 12


def test_joint_is_not_product_of_type_mark_and_physical_marginals():
    cell = m.booking_loss_history(compile_(), "p0", cancelled=True)

    def marginal(field, index):
        return sum(r["posterior"] for r in cell["components"] if r[field] == index)

    actual = next(
        r["posterior"]
        for r in cell["components"]
        if (r["state"], r["type"], r["attempt_mark"]) == (0, 0, 0)
    )
    assert actual != marginal("state", 0) * marginal("type", 0) * marginal(
        "attempt_mark", 0
    )


def test_owned_context_full_lossless_metadata_and_no_float_probability_coercion():
    context = {
        "nested": [{"fare": -0.0}],
        "nonfinite": float("inf"),
        "signed": F(-2, 3),
    }
    value = replace(law(), conditioning_context=context)
    context["nested"][0]["fare"] = 10.0
    result = compile_(value)
    record = m.booking_loss_record(result)
    assert (
        value.conditioning_context["nested"][0]["fare"] == 0.0
        and math.copysign(1, value.conditioning_context["nested"][0]["fare"]) == -1
    )
    assert "8000000000000000" in json.dumps(record)
    assert (
        "inf" in json.dumps(record)
        and record["report"]["lifecycle_utility_CS_service_payoff"] is None
    )
    record["report"] = {}
    assert len(result.report["paths"]) == 288
    with pytest.raises(TypeError):
        replace(law(), conditioning_context={1: "non-string key"})
    with pytest.raises(TypeError):
        replace(law(), conditioning_context={"opaque": object()})


@pytest.mark.parametrize(
    "name",
    [
        "initial_states",
        "boundary_states",
        "latent_types",
        "ordinary_marks",
        "attempt_marks",
    ],
)
def test_complete_distinct_named_axes(name):
    with pytest.raises(ValueError):
        compile_(replace(law(), **{name: ("same", "same")}))


@pytest.mark.parametrize(
    "case",
    [
        "initial_sum",
        "negative",
        "float",
        "bool",
        "missing",
        "ordinary_sum",
        "hidden_axis",
        "boundary_sum",
        "outside_cancel",
        "scope",
        "duplicate_product",
        "unoffered_zero_type",
    ],
)
def test_full_invalid_declarations_before_forward(case, monkeypatch):
    value = law()
    if case == "initial_sum":
        value = replace(
            value, initial_joint=change(value.initial_joint, (0, 0, 0, 0), F(1))
        )
    elif case == "negative":
        value = replace(
            value, initial_joint=change(value.initial_joint, (0, 0, 0, 0), F(-1))
        )
    elif case == "float":
        value = replace(
            value, initial_joint=change(value.initial_joint, (0, 0, 0, 0), 0.1)
        )
    elif case == "bool":
        value = replace(
            value, initial_joint=change(value.initial_joint, (0, 0, 0, 0), False)
        )
    elif case == "missing":
        value = replace(value, boundary_kernel=value.boundary_kernel[:-1])
    elif case == "ordinary_sum":
        value = replace(
            value,
            ordinary_likelihood=change(value.ordinary_likelihood, (2, 0, 0), F(1)),
        )
    elif case == "hidden_axis":
        value = replace(value, ordinary_likelihood=(value.ordinary_likelihood,) * 2)
    elif case == "boundary_sum":
        value = replace(
            value,
            boundary_kernel=change(value.boundary_kernel, (0, 2, 0, 0, 0, 0, 0), F(1)),
        )
    elif case == "outside_cancel":
        value = replace(
            value,
            boundary_kernel=change(
                value.boundary_kernel,
                (0, 2, 0, 0, 2),
                ((F(7, 10), F(1, 10)), (F(1, 5), F())),
            ),
        )
    elif case == "scope":
        value = replace(value, conditioning_scope="already-after-purchase-and-boundary")
    elif case == "duplicate_product":
        value = replace(value, products=(value.products[0], value.products[0]))
    else:
        likelihood = tuple(
            group if k == 2 else tuple((row[0], F(), row[1] + row[2]) for row in group)
            for k, group in enumerate(value.ordinary_likelihood)
        )
        value = replace(
            value,
            products=(value.products[0], replace(value.products[1], offered=False)),
            ordinary_likelihood=likelihood,
        )

    def fail(*args, **kwargs):
        raise AssertionError("invalid input must fail before forward")

    monkeypatch.setattr(m, "_forward", fail)
    with pytest.raises((TypeError, ValueError)):
        compile_(value)


def test_unoffered_product_zero_rows_are_retained():
    value = law()
    likelihood = tuple(
        tuple((row[0], F(), row[1] + row[2]) for row in group)
        for group in value.ordinary_likelihood
    )
    value = replace(
        value,
        products=(value.products[0], replace(value.products[1], offered=False)),
        ordinary_likelihood=likelihood,
    )
    result = compile_(value)
    assert m.booking_loss_history(result, "p1", cancelled=False)["evidence"] == 0
    assert any(r["action_index"] == 1 for r in result.report["paths"])


def test_outside_only_request_has_no_ticket_or_invented_cancellation():
    boundary = (F(1), F())
    for _ in range(6):
        boundary = (boundary,)
    value = m.BookingLossLaw(
        "outside-person",
        ("z",),
        ("x",),
        ("k",),
        ("u",),
        ("g",),
        (),
        ((((F(1),),),),),
        (((F(1),),),),
        boundary,
        {"request": "declared one ordinary request"},
        m.SCOPE,
        "complete fixed context",
        "ordinary U",
        "later G",
        "no ticket/no cancellation",
        "NEW outside-only law",
        "synthetic-outside",
    )
    result = compile_(value)
    assert len(result.report["paths"]) == 2
    assert m.booking_loss_history(result, None, cancelled=False)["evidence"] == 1
    assert (
        m.booking_loss_history(result, None, cancelled=True)["conditional_joint"]
        is None
    )


def test_reader_no_backend_call_and_exact_types(monkeypatch):
    result = compile_()

    def fail(*args, **kwargs):
        raise AssertionError("reader cannot forward")

    monkeypatch.setattr(m, "_forward", fail)
    assert m.booking_loss_history(result, "p0", cancelled=True)["evidence"] == F(
        259, 1350
    )
    with pytest.raises(TypeError):
        m.booking_loss_history(result, "p0", cancelled=1)
    with pytest.raises(TypeError):
        m.booking_loss_history(result, 0, cancelled=False)
    with pytest.raises(ValueError):
        m.booking_loss_history(result, "missing", cancelled=False)
    with pytest.raises(TypeError):
        m.booking_loss_record({})


def test_pre_forward_all_support_limit_and_failure_ownership(monkeypatch):
    value = law()
    saved = pickle.dumps(value)
    result = compile_(value)
    assert m._units(m._encoded(result)) <= result.report["structural_bound"]

    def fail(*args, **kwargs):
        raise RuntimeError("NEW injected pure failure")

    monkeypatch.setattr(m, "_forward", fail)
    with pytest.raises(m.BookingLossLimit):
        compile_(value, row_limit=result.report["structural_bound"] - 1)
    with pytest.raises(RuntimeError, match="injected"):
        compile_(value)
    assert pickle.dumps(value) == saved


@pytest.mark.parametrize("limit", [True, 0, -1, 1.0])
def test_typed_positive_limit(limit):
    with pytest.raises(ValueError):
        compile_(row_limit=limit)


@pytest.mark.parametrize("fare,offered", [(1, True), (F(1), True), (1.0, 1)])
def test_no_fare_or_boolean_coercion(fare, offered):
    with pytest.raises(TypeError):
        m.OriginalOffer("p", fare, offered)


def test_actual_probability_rationals_and_owned_export():
    result = compile_()
    exported = m.booking_loss_record(result)
    assert exported["report"]["joint_mass"] == _encoded(F(1))
    assert exported["law"]["original_person_label"] == "logical-person-94974"
    assert result.report["ordinary_and_attempt_marks_are_distinct_axes"] is True
