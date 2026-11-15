"""A.10/A.5/A.14/B0/V9: NEW pure latent-type observation units."""

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

from airline_market.core.hashing import jsonable
from airline_market.sim.received_information import _units

if os.environ.get("LATENT_TYPE_SOURCE_DIR"):
    p = Path(os.environ["LATENT_TYPE_SOURCE_DIR"]) / "latent_type_attempt.py"
    spec = importlib.util.spec_from_file_location(
        "airline_market.sim.latent_type_attempt", p
    )
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
from airline_market.sim import latent_type_attempt as m

LIMIT = 10_000_000


def model(full=False):
    observations = (
        ("revealed-a", "revealed-b", "unreached") if full else ("merged", "unreached")
    )
    mapping = observations[:2] if full else ("merged", "merged")
    own = (
        m.AttemptOffer("own", -0.0, True, False, False),
        m.AttemptOffer("other", -7.25, False, False, False),
    )
    other = (replace(own[0], offered=False), replace(own[1], offered=True))
    initial = (
        ((F(1, 10), F(1, 5), F()), (F(1, 20), F(3, 20), F())),
        ((F(1, 4), F(), F()), (F(1, 20), F(1, 5), F())),
    )
    programs = []
    for obs in range(len(observations)):
        probs = (
            (F(2, 5), F(1, 4), F(1, 7))
            if full and obs == 0
            else (F(4, 5), F(2, 3), F(1, 9))
            if full and obs == 1
            else (F(1, 3), F(3, 5), F(1, 7))
        )
        programs.append(tuple((q, F(), 1 - q) for q in probs))
    first_kernel = ((F(1, 4), F(1, 4), F()), (F(1, 3), F(1, 6), F()))
    second_kernel = ((F(1, 8), F(3, 8), F()), (F(1, 8), F(3, 8), F()))
    kernel = tuple(
        tuple(
            tuple(
                first_kernel if (x + k + g) % 2 == 0 else second_kernel
                for g in range(3)
            )
            for k in range(2)
        )
        for x in range(2)
    )
    later = tuple(
        tuple(
            tuple(
                (F(), q, 1 - q)
                for q in (
                    (F(3, 7), F(1, 5), F(1, 2))
                    if sig == 0
                    else (F(1, 6), F(5, 6), F(2, 7))
                )
            )
            for sig in range(3)
        )
        for _ in observations
    )
    return m.TypeObservationAttempt(
        physical_states=("calm", "storm"),
        latent_types=("a", "b"),
        private_marks=("left", "right", "zero"),
        observations=observations,
        type_observation=mapping,
        signals=(None, "notice", "never"),
        own_menus=tuple(own for _ in observations),
        interline_menus=tuple(tuple(other for _ in range(3)) for _ in observations),
        product_catalogue_disclosed=False,
        interline_enabled=True,
        initial_joint=initial,
        own_program=tuple(programs),
        next_state_signal=kernel,
        interline_program=later,
        conditioning_view={"declared": ["earlier own allowed history"]},
        conditioning_definition="NEW conditional attempt before current own choice",
        observation_definition="NEW deterministic type category; not actual native awareness",
        private_mark_interpretation="explicit finite shared private-vector index, not an adopted noise meaning",
        definition="NEW finite constructed transition/selection law, not native law",
        source_ref="unit/V9_latent_type_attempt/91974-local-coordinate-not-native-key-or-seed",
    )


def run(x=None, **kw):
    return m.compile_type_observation_attempt(
        model() if x is None else x, row_limit=kw.get("row_limit", LIMIT)
    )


def cell(result, history):
    return next(
        c for c in result.report["history_cells"] if tuple(c["history"]) == history
    )


@pytest.mark.parametrize("full", [False, True])
def test_new_independent_full_path_reference(full):
    x = model(full)
    result = run(x)
    expected = {}
    stopped = F()
    for state, kind, mark, action in product(range(2), range(2), range(3), range(3)):
        obs = x.observations.index(x.type_observation[kind])
        mass = x.initial_joint[state][kind][mark] * x.own_program[obs][mark][action]
        if action != 2:
            stopped += mass
            continue
        for nxt, signal, later in product(range(2), range(3), range(3)):
            q = (
                mass
                * x.next_state_signal[state][kind][mark][nxt][signal]
                * x.interline_program[obs][signal][mark][later]
            )
            key = (obs, signal, nxt, kind, mark, later)
            expected[key] = expected.get(key, F()) + q
    for row in result.report["interline_rows"]:
        key = tuple(
            row[k]
            for k in ("observation", "signal", "physical", "type", "mark", "action")
        )
        assert row["mass"] == expected.get(key, F())
    assert stopped + sum(expected.values()) == result.report["terminal_mass"] == 1
    assert result.report["same_type_and_private_mark_carried"]


def test_new_coarse_observation_keeps_both_types_and_selected_posterior():
    result = run()
    prior = cell(result, ("initial", 0, None, None))
    after = cell(result, ("own", 0, None, 2))
    a0 = sum(r["posterior"] for r in prior["components"] if r["type"] == 0)
    a1 = sum(r["posterior"] for r in after["components"] if r["type"] == 0)
    assert a0 == F(11, 20)
    assert a1 != a0 and 0 < a1 < 1
    assert {r["type"] for r in after["components"] if r["joint_mass"]} == {0, 1}
    for mark in range(3):
        rows = [
            r
            for r in result.report["own_rows"]
            if r["mark"] == mark and r["action"] == 0
        ]
        assert len({r["action_probability"] for r in rows}) == 1


def test_new_joint_physical_transition_carries_same_type_and_mark():
    x = model()
    result = run(x)
    assert any(
        r["physical"] != r["next_physical"] and r["mass"] > 0
        for r in result.report["transition_signal_rows"]
    )
    for obs, signal, nxt, kind, mark in product(
        range(2), range(3), range(2), range(2), range(3)
    ):
        c = cell(result, ("reach", obs, signal, None))
        actual = next(
            r["joint_mass"]
            for r in c["components"]
            if (r["physical"], r["type"], r["mark"]) == (nxt, kind, mark)
        )
        expected = sum(
            (
                r["mass"]
                for r in result.report["transition_signal_rows"]
                if (
                    r["observation"],
                    r["signal"],
                    r["next_physical"],
                    r["type"],
                    r["mark"],
                )
                == (obs, signal, nxt, kind, mark)
            ),
            F(),
        )
        assert actual == expected


def test_new_zero_support_complete_null_and_tower_identities():
    result = run()
    assert len(result.report["own_rows"]) == 36
    assert len(result.report["transition_signal_rows"]) == 72
    assert len(result.report["interline_rows"]) == 216
    assert len(result.report["history_cells"]) == 32
    for c in result.report["history_cells"]:
        assert len(c["components"]) == 12
        assert sum(r["joint_mass"] for r in c["components"]) == c["evidence"]
        if c["evidence"]:
            assert sum(r["posterior"] for r in c["components"]) == 1
        else:
            assert all(r["posterior"] is None for r in c["components"])
        if c["history"][1] == 1 or c["history"][2] == 2:
            assert c["evidence"] == 0
    for obs, signal in product(range(2), range(3)):
        assert (
            sum(
                cell(result, ("interline", obs, signal, a))["evidence"]
                for a in range(3)
            )
            == cell(result, ("reach", obs, signal, None))["evidence"]
        )


@pytest.mark.parametrize("tiny", [F(1, 10**650), F()])
def test_new_exact_tiny_type_mass_or_zero_not_float_substituted(tiny):
    x = model(full=True)
    initial = tuple(
        tuple(
            tuple(
                (tiny if k == 0 else 1 - tiny) if s == 0 and g == 0 else F()
                for g in range(3)
            )
            for k in range(2)
        )
        for s in range(2)
    )
    result = run(replace(x, initial_joint=initial))
    c = cell(result, ("initial", 0, None, None))
    assert c["evidence"] == tiny
    assert (
        (c["components"][0]["posterior"] == 1)
        if tiny
        else all(r["posterior"] is None for r in c["components"])
    )
    if tiny:
        assert float(tiny) == 0 and c["evidence"] > 0


def test_new_disabled_interline_retains_all_kernel_and_null_stage_rows():
    x = replace(model(), interline_enabled=False)
    result = run(x)
    assert result.report["terminal_mass"] == 1
    assert all(r["mass"] == 0 for r in result.report["transition_signal_rows"])
    assert all(r["mass"] == 0 for r in result.report["interline_rows"])
    assert len(result.report["transition_signal_rows"]) == 72


def test_new_outside_only_catalogue_and_empty_declared_menu():
    x = model()
    x = replace(
        x,
        own_menus=((), ()),
        interline_menus=(((), (), ()), ((), (), ())),
        own_program=tuple(tuple((F(1),) for _ in range(3)) for _ in range(2)),
        interline_program=tuple(
            tuple(tuple((F(1),) for _ in range(3)) for _ in range(3)) for _ in range(2)
        ),
    )
    assert run(x).report["terminal_mass"] == 1


def test_new_owned_input_output_and_metadata_binary64_exports():
    x = model()
    original = {"declared": ["before"]}
    x = replace(x, conditioning_view=original)
    original["declared"].append("later")
    result = run(x)
    record = m.type_observation_record(result)
    assert tuple(x.conditioning_view["declared"]) == ("before",)
    record["model"]["conditioning_view"]["declared"] = ("external",)
    assert tuple(x.conditioning_view["declared"]) == ("before",)
    assert math.copysign(1, x.own_menus[0][0].fare) == -1
    assert json.dumps(record)
    with pytest.raises(TypeError):
        result.report["new"] = 1
    assert _units(m.type_observation_record(result)) < LIMIT


@pytest.mark.parametrize(
    "change",
    [
        "initial_shape",
        "float",
        "negative",
        "initial_sum",
        "own_hidden_axis",
        "own_sum",
        "kernel_shape",
        "kernel_sum",
        "interline_hidden_axis",
        "signal_none",
        "observation_missing",
        "boolean",
    ],
)
def test_new_complete_declared_input_refusals(change):
    x = model()
    values = {
        "initial_shape": {"initial_joint": x.initial_joint[:-1]},
        "float": {
            "initial_joint": (
                ((0.1, F(1, 5), F()), x.initial_joint[0][1]),
                x.initial_joint[1],
            )
        },
        "negative": {
            "initial_joint": (
                ((F(-1, 10), F(1, 5), F()), x.initial_joint[0][1]),
                x.initial_joint[1],
            )
        },
        "initial_sum": {
            "initial_joint": tuple(
                tuple(tuple(q / 2 for q in row) for row in state)
                for state in x.initial_joint
            )
        },
        "own_hidden_axis": {"own_program": (x.own_program,)},
        "own_sum": {
            "own_program": tuple(
                tuple(tuple(q / 2 for q in row) for row in obs) for obs in x.own_program
            )
        },
        "kernel_shape": {"next_state_signal": x.next_state_signal[:-1]},
        "kernel_sum": {
            "next_state_signal": tuple(
                tuple(
                    tuple(
                        tuple(tuple(q / 2 for q in row) for row in mark)
                        for mark in kind
                    )
                    for kind in state
                )
                for state in x.next_state_signal
            )
        },
        "interline_hidden_axis": {"interline_program": (x.interline_program,)},
        "signal_none": {"signals": ("absent", "notice", "never")},
        "observation_missing": {"type_observation": ("merged",)},
        "boolean": {"interline_enabled": 1},
    }[change]
    with pytest.raises((ValueError, TypeError)):
        run(replace(x, **values))


def test_new_unreachable_unoffered_action_still_refused():
    x = model()
    rows = list(x.interline_program)
    hidden = list(rows[1])
    marks = list(hidden[2])
    marks[2] = (F(1), F(), F())
    hidden[2] = tuple(marks)
    rows[1] = tuple(hidden)
    with pytest.raises(ValueError, match="unoffered"):
        run(replace(x, interline_program=tuple(rows)))


def test_new_all_menus_and_explicit_visibility_checked():
    x = model()
    wrong = (replace(x.own_menus[1][0], product_id="different"), x.own_menus[1][1])
    with pytest.raises(ValueError, match="catalogue"):
        run(replace(x, own_menus=(x.own_menus[0], wrong)))
    hidden = (replace(x.own_menus[0][0], disclose_fare=True), x.own_menus[0][1])
    with pytest.raises(ValueError, match="visibility"):
        run(replace(x, own_menus=(hidden, x.own_menus[1])))


def test_new_opaque_metadata_refused_and_failure_inputs_owned():
    x = model()
    with pytest.raises(TypeError, match="opaque"):
        replace(x, conditioning_view={"opaque": bytearray(b"a")})
    before = pickle.dumps(x)
    with pytest.raises(m.TypeObservationLimit):
        run(x, row_limit=1)
    assert pickle.dumps(x) == before


def test_new_full_output_limit_before_probability_forward(monkeypatch):
    def forbidden(*args):
        raise AssertionError("forward must not run")

    monkeypatch.setattr(m, "_forward", forbidden)
    with pytest.raises(m.TypeObservationLimit):
        run(row_limit=1)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_new_positive_structural_limit_required(limit):
    with pytest.raises(ValueError):
        run(row_limit=limit)


def test_new_no_automatic_knowledge_or_native_welfare_claim():
    result = run()
    assert result.report["source_receipt_bridge_executed"] is False
    assert (
        result.report[
            "native_probability_received_knowledge_preferences_or_utility_authenticated"
        ]
        is False
    )
    assert result.report["lifecycle_utility_CS_service_payoff"] is None
    assert result.report["scientific_budget"] is None
    assert result.model.observation_definition != result.model.latent_types[0]
    with pytest.raises(TypeError):
        m.type_observation_record(replace(result, rule="wrong"))
    assert jsonable(result.model)["type_observation"] == ["merged", "merged"]
