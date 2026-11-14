import numpy as np
import pytest
from airline_market.game import solve_bimatrix, psro_matrix, PolicyRef, estimate_payoffs
from airline_market.spec.library_builders import cx_example


def test_equilibria_and_exact_finite_psro():
    for A, B in [
        (np.array([[1, -1], [-1, 1]]), np.array([[-1, 1], [1, -1]])),
        (np.array([[2, 0], [0, 1]]), np.array([[1, 0], [0, 2]])),
        (
            np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]]),
            np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]]),
        ),
    ]:
        result = solve_bimatrix(A, B)
        assert max(result["deviation_gains"]) <= 1e-9
        restricted = psro_matrix(A, B)
        assert restricted["certified"] and restricted["exploitability"] <= 1e-9
        if A.shape == (3, 3):
            assert np.allclose(result["strategies"][0], np.full(3, 1 / 3))


def test_payoff_cache_and_mixed_determinism():
    s = cx_example("C")
    profile = {
        "F": PolicyRef(
            "mixed",
            {
                "policies": [PolicyRef("fixed", (0.0,)), PolicyRef("fixed", (9850.0,))],
                "probabilities": [0.5, 0.5],
            },
        ),
        "R": PolicyRef(
            "controller",
            {
                "controller": "threshold_linear_rule",
                "params": dict(s.controllers[1].params),
            },
        ),
    }
    cache = {}
    a = estimate_payoffs(s, profile, [1, 2], cache=cache)
    n = len(cache)
    b = estimate_payoffs(s, profile, [1, 2], cache=cache)
    assert a.per_seed == b.per_seed and len(cache) == n == 2


def test_ibr_detects_response_cycle(monkeypatch):
    import importlib

    module = importlib.import_module("airline_market.game.ibr")
    from airline_market.game.payoffs import PayoffEstimate

    def payoff(spec, profile, seeds):
        same = profile["A"].value == profile["B"].value
        v = 1.0 if same else -1.0
        return PayoffEstimate(
            {"A": v, "B": -v},
            {"A": [v] * len(seeds), "B": [-v] * len(seeds)},
            tuple(seeds),
            {},
        )

    monkeypatch.setattr(module, "estimate_payoffs", payoff)

    def br(i, profile):
        return PolicyRef(
            "fixed", profile["B"].value if i == "A" else 1 - profile["A"].value
        )

    result = module.ibr(
        None,
        ["A", "B"],
        {"A": PolicyRef("fixed", 0), "B": PolicyRef("fixed", 0)},
        br,
        max_iters=10,
        seeds=[1, 2],
    )
    assert not result["converged"] and result["cycle"]["period"] == 2
