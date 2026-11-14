"""Finite-policy PSRO with exact finite-population response diagnostics."""

import numpy as np
from .meta_solvers import solve_bimatrix, deviation_gains, replicator


def psro_matrix(A, B, initial=((0,), (0,)), max_iters=30, tol=1e-9):
    """
    Expand finite response populations until the declared matrix deviation bound holds
    (Eqs. 20–23).
    """
    A = np.asarray(A)
    B = np.asarray(B)
    I = list(initial[0])
    J = list(initial[1])
    history = []
    for iteration in range(max_iters):
        meta = solve_bimatrix(A[np.ix_(I, J)], B[np.ix_(I, J)])
        x = np.zeros(A.shape[0])
        y = np.zeros(A.shape[1])
        x[I] = meta["strategies"][0]
        y[J] = meta["strategies"][1]
        gains = deviation_gains(A, B, x, y)
        history.append(
            {
                "iteration": iteration,
                "population_sizes": [len(I), len(J)],
                "gains": gains,
            }
        )
        if max(gains) <= tol:
            return {
                "strategies": (x, y),
                "history": history,
                "exploitability": max(gains),
                "certified": True,
                "policy_class": "full_declared_finite_matrix",
            }
        i = int(np.argmax(A @ y))
        j = int(np.argmax(x @ B))
        if i not in I:
            I.append(i)
        if j not in J:
            J.append(j)
    return {
        "strategies": (x, y),
        "history": history,
        "exploitability": max(gains),
        "certified": False,
        "policy_class": "full_declared_finite_matrix",
    }


def psro(spec, populations, seeds, br_solver=None, max_iters=10):
    """
    Estimate a restricted game and expand its response-policy populations (Eqs. 20–23).
    """
    from .payoffs import estimate_payoffs, PolicyRef
    import copy

    carriers = list(populations)
    if len(carriers) != 2:
        raise ValueError(
            "simulated PSRO currently uses two carriers; use replicator for supplied multiplayer tensors"
        )
    populations = copy.deepcopy(populations)
    history = []
    for iteration in range(max_iters):
        shape = tuple(len(populations[i]) for i in carriers)
        A = np.zeros(shape)
        B = np.zeros(shape)
        for a in range(shape[0]):
            for b in range(shape[1]):
                p = estimate_payoffs(
                    spec,
                    {
                        carriers[0]: populations[carriers[0]][a],
                        carriers[1]: populations[carriers[1]][b],
                    },
                    seeds,
                )
                A[a, b] = p.means[carriers[0]]
                B[a, b] = p.means[carriers[1]]
        if max(shape) <= 12:
            meta = solve_bimatrix(A, B)
        else:
            meta = replicator([A, B])
            x, y = meta["strategies"]
            meta["payoffs"] = (float(x @ A @ y), float(x @ B @ y))
        history.append(
            {"iteration": iteration, "shape": shape, "payoffs": meta["payoffs"]}
        )
        if br_solver is None:
            break
        profile = {
            i: PolicyRef(
                "mixed",
                {
                    "policies": populations[i],
                    "probabilities": meta["strategies"][k].tolist(),
                },
            )
            for k, i in enumerate(carriers)
        }
        added = False
        for i in carriers:
            ref = br_solver(i, profile)
            if ref.hash() not in {r.hash() for r in populations[i]}:
                populations[i].append(ref)
                added = True
        if not added:
            break
    return {
        "populations": populations,
        "meta": meta,
        "history": history,
        "certification": "finite restricted population only; held-out best-response audit required for game-wide exploitability",
    }
