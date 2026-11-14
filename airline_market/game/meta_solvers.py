"""Finite bimatrix Nash by support feasibility; general replicator reference."""

from itertools import combinations, product
import numpy as np
from scipy.optimize import linprog


def deviation_gains(A, B, x, y):
    """Return each player's best unilateral gain in a bimatrix game (Eqs. 20–23)."""
    return float(np.max(A @ y) - x @ A @ y), float(np.max(x @ B) - x @ B @ y)


def solve_bimatrix(A, B, tol=1e-10):
    """Find a finite-population Nash equilibrium by support feasibility (Eqs. 20–23)."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != B.shape or A.ndim != 2:
        raise ValueError("two matching payoff matrices required")
    m, n = A.shape
    for i, j in product(range(m), range(n)):
        if A[i, j] >= np.max(A[:, j]) - tol and B[i, j] >= np.max(B[i, :]) - tol:
            x = np.eye(m)[i]
            y = np.eye(n)[j]
            return {
                "strategies": (x, y),
                "payoffs": (float(A[i, j]), float(B[i, j])),
                "deviation_gains": deviation_gains(A, B, x, y),
                "certified": True,
                "policy_class": "finite_population",
            }
    if max(m, n) > 12:
        raise ValueError("support enumeration is limited to 12 strategies per carrier")

    def feasible(payoff, own, opponent):
        """
        Find an opponent mixture making the proposed support a best response (Eqs.
        20–23).
        """
        # Opponent mix makes every supported own action a best response.
        matrix = payoff[np.ix_(own, opponent)]
        eq = np.c_[matrix, -np.ones(len(own))]
        eq = np.vstack([eq, np.r_[np.ones(len(opponent)), 0.0]])
        outside = [i for i in range(payoff.shape[0]) if i not in own]
        ub = (
            np.c_[payoff[np.ix_(outside, opponent)], -np.ones(len(outside))]
            if outside
            else None
        )
        result = linprog(
            np.zeros(len(opponent) + 1),
            A_ub=ub,
            b_ub=np.zeros(len(outside)) if outside else None,
            A_eq=eq,
            b_eq=np.r_[np.zeros(len(own)), 1.0],
            bounds=[(0.0, 1.0)] * len(opponent) + [(None, None)],
            method="highs",
        )
        return result.x[:-1] if result.success else None

    sizes = sorted(
        product(range(1, m + 1), range(1, n + 1)),
        key=lambda ij: (max(ij), abs(ij[0] - ij[1]), ij),
    )
    for I, J in (
        (I, J)
        for a, b in sizes
        for I in combinations(range(m), a)
        for J in combinations(range(n), b)
    ):
        y0 = feasible(A, I, J)
        if y0 is None:
            continue
        x0 = feasible(B.T, J, I)
        if x0 is None:
            continue
        x = np.zeros(m)
        y = np.zeros(n)
        x[list(I)] = x0
        y[list(J)] = y0
        gains = deviation_gains(A, B, x, y)
        if max(gains) <= max(tol, 1e-9):
            return {
                "strategies": (x, y),
                "payoffs": (float(x @ A @ y), float(x @ B @ y)),
                "deviation_gains": gains,
                "certified": True,
                "policy_class": "finite_population",
            }
    raise RuntimeError(
        "support enumeration did not find a numerically valid equilibrium"
    )


def replicator(payoffs, iterations=10000, step=0.1, tol=1e-8):
    """Run the declared approximate multiplayer meta-strategy dynamics (Eqs. 20–23)."""
    tensors = [np.asarray(p, dtype=float) for p in payoffs]
    n = len(tensors)
    shape = tensors[0].shape
    mixes = [np.full(k, 1 / k) for k in shape]

    def pure_values(i):
        """Integrate opponents' mixtures to score every pure action (Eqs. 20–23)."""
        values = np.zeros(shape[i])
        for profile in product(*(range(k) for k in shape)):
            weight = np.prod([mixes[j][profile[j]] for j in range(n) if j != i])
            values[profile[i]] += tensors[i][profile] * weight
        return values

    residual = float("inf")
    for iteration in range(iterations):
        new = []
        gains = []
        for i in range(n):
            v = pure_values(i)
            gains.append(float(v.max() - mixes[i] @ v))
            z = mixes[i] * np.exp(np.clip(step * (v - mixes[i] @ v), -100, 100))
            new.append(z / z.sum())
        residual = max(gains)
        mixes = new
        if residual < tol:
            break
    return {
        "strategies": mixes,
        "deviation_gains": gains,
        "iterations": iteration + 1,
        "certified": False,
        "converged": residual < tol,
        "policy_class": "finite_population",
    }
