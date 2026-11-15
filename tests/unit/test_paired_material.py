"""A.10/A.5/A.14/B0/V9: NEW pure finite inputs; no old helper/case replay."""

import pickle
from dataclasses import replace
from fractions import Fraction as F
from itertools import product

import pytest

from airline_market.sim import paired_material as subject
from airline_market.sim.payoff_logit import ContingentPayoff
from airline_market.sim.terminal_material import TerminalMaterialAccount

LIMIT = 8_000_000


def nested(shape, fn, prefix=()):
    return (
        fn(*prefix)
        if not shape
        else tuple(nested(shape[1:], fn, prefix + (i,)) for i in range(shape[0]))
    )


def example(*, rho=F(), disabled=False):
    support = subject.MaterialSupport(
        "constructed104974-not-native-or-seed",
        "declared-OD",
        "A",
        ("y0", "y1"),
        ("k0", "k1", "zero-k"),
        ("g0", "g1"),
        ("o0", "o1", "unused"),
        ("o0", "o1", "unused"),
        ("notice", "quiet", "unused"),
        ("own", "other"),
        ("A", "B"),
        ((True, False),) * 3,
        (((False, not disabled),) * 3,) * 3,
        not disabled,
        ("record-a", "record-b", "unreported"),
        "unit104974",
        subject.SCOPE,
        "Complete DECLARED pre-category conditioning, not native-derived",
        "O=m(K), received labels are declarations",
        "SAME persistent declared G; no native Gumbel/awareness inference",
        subject.SHOCK_SCOPE,
    )
    policy = subject.FixedMaterialPolicy(
        nested(
            (3, 2, 3),
            lambda o, g, a: (
                F(1 + o + g, 7) if a == 0 else F() if a == 1 else 1 - F(1 + o + g, 7)
            ),
        ),
        nested(
            (3, 3, 2, 3),
            lambda o, s, g, a: (
                F()
                if a == 0
                else F()
                if disabled and a == 1
                else F(1)
                if disabled
                else F(2 + o + s + g, 9)
                if a == 1
                else 1 - F(2 + o + s + g, 9)
            ),
        ),
        "Fixed policy104974, not optimized from either law",
        "explicit-policy-source",
    )

    def law(reference, missing):
        def initial(y, k, g):
            kp = (F(4, 7), F(3, 7), F()) if reference else (F(3, 8), F(5, 8), F())
            gm = F(2 + k, 6) if reference else F(1 + k, 5)
            yp = F(3 + k + g, 7) if reference else F(2 + k + g, 8)
            return kp[k] * (gm if g else 1 - gm) * (yp if y else 1 - yp)

        def transition(y, k, g, yn, s):
            yp = F(1 + y + k + g, 6) if reference else F(2 + y + k + g, 8)
            high = F(2 + yn + g, 6) if reference else F(1 + yn + g, 5)
            return (yp if yn else 1 - yp) * (
                high if s == 0 else 1 - high if s == 1 else F()
            )

        def probability(stage, y, k, g, a, d):
            if d == 2:
                return missing
            if a == 2:
                return 1 - missing if d == 0 else F()
            delivered = (
                F(1 + stage + y + k, 7) if reference else F(2 + stage + y + k, 8)
            )
            return (1 - missing) * (delivered if d == 0 else 1 - delivered)

        def account(stage, y, k, g, a, d):
            if d == 2:
                values, closed = (None,) * 4, False
            elif a == 2:
                values, closed = (F(20), F(), F(4), F(-1)), True
            else:
                values = (
                    F(800 + 70 * k + 10 * stage) if d == 0 else F(),
                    F(290 if a == 0 else 455) if d == 0 else F(-5),
                    F(3) + F(y, 2) if d == 0 else F(9 + k),
                    F() if d == 0 else F(20 + stage),
                )
                closed = True
            return TerminalMaterialAccount(
                ContingentPayoff(*values),
                ("B-source", "P-source", "W-source", "C-source"),
                "unit104974",
                closed,
                closed,
                "Explicit conditional material; no actual ledger/service inference",
            )

        return subject.DeclaredMaterialLaw(
            "reference104974" if reference else "forecast104974",
            nested((2, 3, 2), initial),
            nested((2, 3, 2, 2, 3), transition),
            nested((2, 2, 3, 2, 3, 3), probability),
            nested((2, 2, 3, 2, 3, 3), account),
            "Complete declared law104974",
            "reference-source" if reference else "forecast-source",
        )

    return subject.PairedMaterialModel(support, policy, law(False, F()), law(True, rho))


def run(model, **kw):
    return subject.evaluate_paired_material(
        model,
        selected_observation="o1",
        selected_signal="notice",
        row_limit=LIMIT,
        **kw,
    )


def independent(model, law):
    known = F()
    unknown = F()
    for y, k, g in product(range(2), range(3), range(2)):
        o = k
        pi = model.policy.own_program[o][g]
        for a, d in product(range(2), range(3)):
            mass = (
                law.initial_joint[y][k][g]
                * pi[a]
                * law.terminal_probabilities[0][y][k][g][a][d]
            )
            values = law.terminal_accounts[0][y][k][g][a][d].payoff.components()
            if None in values:
                unknown += mass
            else:
                known += mass * (values[0] - values[1] - values[2] - values[3])
        for yn, s, a, d in product(range(2), range(3), range(3), range(3)):
            mass = (
                law.initial_joint[y][k][g]
                * pi[-1]
                * law.transition_signal[y][k][g][yn][s]
                * model.policy.interline_program[o][s][g][a]
                * law.terminal_probabilities[1][yn][k][g][a][d]
            )
            values = law.terminal_accounts[1][yn][k][g][a][d].payoff.components()
            if None in values:
                unknown += mass
            else:
                known += mass * (values[0] - values[1] - values[2] - values[3])
    return None if unknown else known


def test_full_path_independent_reference_both_laws_and_all_support():
    model = example()
    before = pickle.dumps(model)
    result = run(model)
    for name, law in (("forecast", model.forecast), ("reference", model.reference)):
        r = result.analyst[name]
        assert r["root_material"] == independent(model, law)
        assert len(r["rows"]) == 756 and len(r["histories"]) == 48
        assert any(
            row["material_components_diagnostic"] is not None
            and row["material_components_diagnostic"] < 0
            for row in r["rows"]
        )
        assert any(
            row["mass"] == 0 and row["material_if_closed"] is None for row in r["rows"]
        )
        for cell in r["histories"]:
            assert len(cell["components"]) == 12
            if cell["evidence"]:
                assert sum(x["posterior"] for x in cell["components"]) == 1
                assert all(
                    x["posterior"] * cell["evidence"] == x["joint"]
                    for x in cell["components"]
                )
            else:
                assert all(x["posterior"] is None for x in cell["components"])
    assert pickle.dumps(model) == before
    assert (
        result.analyst["forecast"]["root_material"]
        != result.analyst["reference"]["root_material"]
    )
    assert result.analyst["full_policy_explicitly_complete"]
    assert not result.analyst["policy_reoptimized_or_reference_truth_inserted"]
    assert (
        subject._units(subject.paired_material_record(result))
        < result.analyst["structural_bound"]
    )


@pytest.mark.parametrize("rho", [F(1, 19), F(1, 10**1001)])
def test_positive_unknown_reference_never_float_pruned(rho):
    model = example(rho=rho)
    result = run(model)
    assert result.analyst["forecast"]["root_material"] is not None
    assert result.analyst["reference"]["root_material"] is None
    assert result.analyst["reference"]["selected_material"] is None
    assert result.analyst["reference"]["terminal_unknown_mass"] == rho > 0
    assert result.analyst["reference_minus_forecast_material"] is None


def zero_forecast(model):
    return replace(
        model.forecast,
        initial_joint=nested((2, 3, 2), lambda y, k, g: F(1, 4) if k == 0 else F()),
    )


def test_zero_evidence_no_belief_fill_but_explicit_offpath_policy():
    model = example()
    result = run(replace(model, forecast=zero_forecast(model)))
    assert result.analyst["forecast"]["selected_evidence"] == 0
    assert result.analyst["forecast"]["selected_material"] is None
    assert result.analyst["reference"]["selected_material"] is not None
    assert result.analyst["forecast"]["root_material"] is not None


def test_missing_behavior_positive_reference_null_no_uniform_fallback():
    model = example()
    policy = replace(
        model.policy,
        own_program=nested(
            (3, 2, 3),
            lambda o, g, a: None if o == 1 else model.policy.own_program[o][g][a],
        ),
        interline_program=nested(
            (3, 3, 2, 3),
            lambda o, s, g, a: (
                None if o == 1 else model.policy.interline_program[o][s][g][a]
            ),
        ),
    )
    result = run(replace(model, policy=policy, forecast=zero_forecast(model)))
    assert not result.analyst["full_policy_explicitly_complete"]
    assert result.analyst["forecast"]["root_material"] is not None
    assert result.analyst["reference"]["root_material"] is None
    assert result.analyst["reference"]["routing_unknown_mass"] == F(3, 7)
    assert result.analyst["reference"]["selected_evidence"] is None
    assert any(row["mass"] is None for row in result.analyst["reference"]["rows"])


def test_reference_zero_cell_retains_root_and_null_conditional():
    model = example()
    law = replace(zero_forecast(model), name="reference-zero")
    result = run(replace(model, reference=law))
    assert result.analyst["reference"]["root_material"] is not None
    assert result.analyst["reference"]["selected_evidence"] == 0
    assert result.analyst["reference"]["selected_material"] is None


def test_disabled_interline_outside_terminates_explicit_own_account():
    model = example(disabled=True)
    result = run(model)
    for name, law in (("forecast", model.forecast), ("reference", model.reference)):
        r = result.analyst[name]
        expected = F()
        for y, k, g, a, d in product(range(2), range(3), range(2), range(3), range(3)):
            weight = (
                law.initial_joint[y][k][g]
                * model.policy.own_program[k][g][a]
                * law.terminal_probabilities[0][y][k][g][a][d]
            )
            value = law.terminal_accounts[0][y][k][g][a][d].payoff.net_value()
            expected += F() if not weight else weight * value
        assert r["root_material"] == expected
        assert r["selected_evidence"] == 0 and r["selected_material"] is None
        assert all(
            row["mass"] == 0 and not row["terminal_executable"]
            for row in r["rows"]
            if row["stage"] == "interline"
        )
        assert any(
            row["coordinate"][-2] == 2 and row["mass"] > 0
            for row in r["rows"]
            if row["stage"] == "own"
        )


@pytest.mark.parametrize(
    "case",
    [
        "joint",
        "transition",
        "terminal",
        "account",
        "unit",
        "partial-policy",
        "unavailable",
        "menu-bool",
        "carrier",
        "scope",
        "role",
        "hidden-axis",
    ],
)
def test_complete_invalid_support_refused_before_evaluation(monkeypatch, case):
    m = example()
    if case == "joint":
        m = replace(
            m,
            forecast=replace(
                m.forecast, initial_joint=nested((2, 3, 2), lambda *x: F())
            ),
        )
    elif case == "transition":
        m = replace(m, reference=replace(m.reference, transition_signal=()))
    elif case == "terminal":
        m = replace(
            m,
            reference=replace(
                m.reference,
                terminal_probabilities=nested(
                    (2, 2, 3, 2, 3, 3),
                    lambda st, y, k, g, a, d: (
                        F()
                        if k == 2
                        else m.reference.terminal_probabilities[st][y][k][g][a][d]
                    ),
                ),
            ),
        )
    elif case == "account":
        m = replace(m, reference=replace(m.reference, terminal_accounts=()))
    elif case == "unit":
        m = replace(m, support=replace(m.support, payoff_unit="wrong"))
    elif case == "partial-policy":
        m = replace(
            m,
            policy=replace(
                m.policy,
                own_program=nested(
                    (3, 2, 3),
                    lambda o, g, a: (
                        None
                        if (o, g, a) == (0, 0, 0)
                        else m.policy.own_program[o][g][a]
                    ),
                ),
            ),
        )
    elif case == "unavailable":
        m = replace(
            m, policy=replace(m.policy, own_program=(((F(), F(1), F()),) * 2,) * 3)
        )
    elif case == "menu-bool":
        m = replace(m, support=replace(m.support, own_menus=((1, False),) * 3))
    elif case == "carrier":
        m = replace(m, support=replace(m.support, product_carriers=("B", "A")))
    elif case == "scope":
        m = replace(
            m, support=replace(m.support, conditioning_scope="native-authenticated")
        )
    elif case == "role":
        m = replace(m, reference=replace(m.reference, name=m.forecast.name))
    else:
        m = replace(
            m, policy=replace(m.policy, own_program=(m.policy.own_program,) * 2)
        )

    def forbidden(*a, **kw):
        raise AssertionError("invalid support reached arithmetic forward")

    monkeypatch.setattr(subject, "_evaluate", forbidden)
    with pytest.raises((ValueError, TypeError)):
        run(m)


def test_all_support_limit_before_forward_inputs_owned(monkeypatch):
    model = example()
    before = pickle.dumps(model)

    def forbidden(*a, **kw):
        raise AssertionError("limit reached arithmetic")

    monkeypatch.setattr(subject, "_evaluate", forbidden)
    with pytest.raises(subject.PairedMaterialLimit):
        subject.evaluate_paired_material(
            model, selected_observation="o1", selected_signal="notice", row_limit=1
        )
    assert pickle.dumps(model) == before


def test_huge_and_tiny_signed_accounts_without_float_conversion():
    model = example()
    account = TerminalMaterialAccount(
        ContingentPayoff(F(10**500), F(-1, 10**1200), F(-2), F(3)),
        ("B", "P", "W", "C"),
        "unit104974",
        True,
        True,
        "Explicit huge/tiny signed account",
    )
    laws = [
        replace(law, terminal_accounts=nested((2, 2, 3, 2, 3, 3), lambda *x: account))
        for law in (model.forecast, model.reference)
    ]
    result = run(replace(model, forecast=laws[0], reference=laws[1]))
    assert result.analyst["forecast"]["root_material"] == account.payoff.net_value()
    assert result.analyst["reference"]["root_material"] == account.payoff.net_value()


def test_open_material_tail_null_retains_known_components():
    model = example()
    law = replace(
        model.reference,
        terminal_accounts=nested(
            (2, 2, 3, 2, 3, 3),
            lambda st, y, k, g, a, d: replace(
                model.reference.terminal_accounts[st][y][k][g][a][d],
                material_tail_closed=False,
            ),
        ),
    )
    result = run(replace(model, reference=law))
    assert result.analyst["reference"]["root_material"] is None
    assert result.analyst["reference"]["terminal_unknown_mass"] == 1
    assert any(
        r["material_components_diagnostic"] is not None
        for r in result.analyst["reference"]["rows"]
    )


def test_injected_second_law_failure_propagates_without_mutation(monkeypatch):
    model = example()
    before = pickle.dumps(model)
    forward = subject._evaluate
    calls = []

    def fail(*args):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("NEW reference forward failure")
        return forward(*args)

    monkeypatch.setattr(subject, "_evaluate", fail)
    with pytest.raises(RuntimeError, match="reference forward"):
        run(model)
    assert calls == [1, 1] and pickle.dumps(model) == before


def test_result_is_owned_analyst_only_policy_fixed_across_law_changes():
    model = example()
    first = run(model)
    second = run(example(rho=F(1, 19)))
    assert first.analyst["policy_sha256"] == second.analyst["policy_sha256"]
    assert first.analyst["model"]["forecast"] == second.analyst["model"]["forecast"]
    assert first.analyst["forecast"] == second.analyst["forecast"]
    out = subject.paired_material_record(first)
    out["analyst"] = {}
    assert first.analyst["model"]
    with pytest.raises(TypeError):
        first.analyst["model"] = None
    with pytest.raises(TypeError):
        subject.paired_material_record("not typed")
    assert first.analyst["lifecycle_utility_CS"] is None
    assert not first.analyst[
        "native_law_knowledge_service_payment_preference_or_source_time_authenticated"
    ]


def test_single_state_single_signal_shape_and_own_purchase_stop():
    m = example()
    support = replace(
        m.support,
        physical_states=("only-y",),
        latent_types=("only-k",),
        private_marks=("only-g",),
        observations=("only-o",),
        type_observation=("only-o",),
        signals=("only-s",),
        own_menus=((True, False),),
        interline_menus=(((False, True),),),
        outcomes=("only-d",),
    )
    policy = subject.FixedMaterialPolicy(
        nested((1, 1, 3), lambda o, g, a: F(int(a == 0))),
        nested((1, 1, 1, 3), lambda o, s, g, a: F(int(a == 1))),
        "Explicit forced own purchase",
        "single-source",
    )
    account = TerminalMaterialAccount(
        ContingentPayoff(F(7), F(), F(), F()),
        ("B", "P", "W", "C"),
        "unit104974",
        True,
        True,
        "Explicit seven, no default zero tail",
    )

    def law(name):
        return subject.DeclaredMaterialLaw(
            name,
            nested((1, 1, 1), lambda *x: F(1)),
            nested((1, 1, 1, 1, 1), lambda *x: F(1)),
            nested((2, 1, 1, 1, 3, 1), lambda *x: F(1)),
            nested((2, 1, 1, 1, 3, 1), lambda *x: account),
            "Complete one-state law",
            "single-source",
        )

    result = subject.evaluate_paired_material(
        subject.PairedMaterialModel(support, policy, law("q"), law("r")),
        selected_observation="only-o",
        selected_signal="only-s",
        row_limit=LIMIT,
    )
    for name in ("forecast", "reference"):
        r = result.analyst[name]
        assert r["root_material"] == 7 and r["selected_evidence"] == 0
        assert r["selected_material"] is None
        assert all(row["mass"] == 0 for row in r["rows"] if row["stage"] == "interline")
