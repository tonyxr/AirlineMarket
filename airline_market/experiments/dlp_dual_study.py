"""Native DLP values and bid prices versus independent exact rational vertices."""

import math
from fractions import Fraction
from itertools import combinations
from pathlib import Path

from airline_market.agents.bidprice import solve_dlp
from airline_market.campaign.artifacts import atomic_json, file_hash


def dot(a, b):
    return sum((x * y for x, y in zip(a, b)), Fraction(0))


def solve_square(matrix, rhs):
    """Exact Gauss-Jordan elimination; singular active sets are not vertices."""
    n = len(rhs)
    rows = [list(a) + [b] for a, b in zip(matrix, rhs)]
    for j in range(n):
        pivot = next((k for k in range(j, n) if rows[k][j]), None)
        if pivot is None:
            return None
        rows[j], rows[pivot] = rows[pivot], rows[j]
        scale = rows[j][j]
        rows[j] = [x / scale for x in rows[j]]
        for k in range(n):
            if k != j:
                scale = rows[k][j]
                rows[k] = [a - scale * b for a, b in zip(rows[k], rows[j])]
    return tuple(row[-1] for row in rows)


def vertices(matrix, rhs, dimension):
    """Enumerate every feasible full-rank active basis with exact inequalities."""
    points = set()
    for active in combinations(range(len(rhs)), dimension):
        point = solve_square([matrix[i] for i in active], [rhs[i] for i in active])
        if point is not None and all(dot(a, point) <= b for a, b in zip(matrix, rhs)):
            points.add(point)
    if not points:
        raise ValueError("no feasible vertex")
    return sorted(points)


def rational_inputs(A, capacity, revenues, demand):
    if not 1 <= len(capacity) <= 3 or not 1 <= len(demand) <= 4:
        raise ValueError("declared small LP domain is 1..3 resources, 1..4 products")
    m, n = len(capacity), len(demand)
    if len(A) != m or any(len(row) != n for row in A) or len(revenues) != n:
        raise ValueError("LP dimensions differ")
    flat = [x for row in A for x in row] + capacity + revenues + demand
    if any(isinstance(x, bool) or not math.isfinite(float(x)) for x in flat):
        raise ValueError("finite numeric LP inputs required")
    convert = lambda row: tuple(Fraction(str(x)) for x in row)
    a, c, r, d = (
        tuple(map(convert, A)),
        convert(capacity),
        convert(revenues),
        convert(demand),
    )
    if any(x < 0 for row in a for x in row) or any(x <= 0 for x in c + d + r):
        raise ValueError(
            "nonnegative incidence and positive capacity/demand/revenue required"
        )
    if any(not any(a[i][j] for i in range(m)) for j in range(n)):
        raise ValueError("every product must use a resource")
    return a, c, r, d


def exact_reference(A, capacity, revenues, demand):
    """Independent primal and dual; no floating-point optimizer or native call."""
    a, c, r, d = rational_inputs(A, capacity, revenues, demand)
    m, n = len(c), len(d)
    eye = [tuple(Fraction(int(i == j)) for j in range(n)) for i in range(n)]
    primal = vertices(
        list(a) + eye + [tuple(-x for x in row) for row in eye],
        list(c) + list(d) + [Fraction(0)] * n,
        n,
    )
    value = max(dot(r, x) for x in primal)
    optimal_primal = [x for x in primal if dot(r, x) == value]
    q = m + n
    constraints = [
        tuple(-a[i][j] for i in range(m))
        + tuple(-Fraction(int(j == k)) for k in range(n))
        for j in range(n)
    ]
    constraints += [tuple(-Fraction(int(i == j)) for j in range(q)) for i in range(q)]
    dual = vertices(constraints, [-x for x in r] + [Fraction(0)] * q, q)
    minimum = min(dot(c + d, z) for z in dual)
    if minimum != value:
        raise AssertionError("exact primal/dual strong duality failed")
    optimal_dual = [z for z in dual if dot(c + d, z) == minimum]
    ranges = [
        (min(z[i] for z in optimal_dual), max(z[i] for z in optimal_dual))
        for i in range(m)
    ]
    return {
        "objective_USD": str(value),
        "optimal_primal_vertices": [[str(v) for v in x] for x in optimal_primal],
        "optimal_dual_vertices_pi_then_mu": [[str(v) for v in z] for z in optimal_dual],
        "resource_dual_ranges": [[str(v) for v in pair] for pair in ranges],
        "resource_duals_unique": all(lo == hi for lo, hi in ranges),
        "all_primal_vertices": [[str(v) for v in x] for x in primal],
        "all_dual_vertices": [[str(v) for v in z] for z in dual],
        "exact_strong_duality": True,
    }


def native_comparison(A, capacity, revenues, demand, reference, tolerance):
    """Certify the raw native solution; never modify or round returned duals."""
    x, pi, objective = solve_dlp(A, capacity, revenues, demand)
    x, pi, objective = list(map(float, x)), list(map(float, pi)), float(objective)
    a_pi = [math.fsum(A[i][j] * pi[i] for i in range(len(pi))) for j in range(len(x))]
    mu = [max(0.0, r - v) for r, v in zip(revenues, a_pi)]
    used = [math.fsum(v * y for v, y in zip(row, x)) for row in A]
    dual_objective = math.fsum(c * p for c, p in zip(capacity, pi)) + math.fsum(
        d * u for d, u in zip(demand, mu)
    )
    residuals = [
        abs(objective - math.fsum(r * y for r, y in zip(revenues, x))),
        abs(objective - float(Fraction(reference["objective_USD"]))),
        abs(dual_objective - objective),
    ]
    residuals += [max(0.0, -y, y - d) for y, d in zip(x, demand)]
    residuals += [max(0.0, u - c) for u, c in zip(used, capacity)]
    residuals += [max(0.0, -p) for p in pi]
    residuals += [abs(p * (c - u)) for p, c, u in zip(pi, capacity, used)]
    residuals += [abs(u * (d - y)) for u, d, y in zip(mu, demand, x)]
    residuals += [abs(y * (v + u - r)) for y, v, u, r in zip(x, a_pi, mu, revenues)]
    ranges = [
        [float(Fraction(v)) for v in pair] for pair in reference["resource_dual_ranges"]
    ]
    residuals += [max(0.0, lo - p, p - hi) for p, (lo, hi) in zip(pi, ranges)]
    finite = all(
        math.isfinite(v) for v in x + pi + mu + [objective, dual_objective] + residuals
    )
    unique = reference["resource_duals_unique"]
    error = max(abs(p - lo) for p, (lo, _) in zip(pi, ranges)) if unique else None
    return {
        "native_quantities": x,
        "native_raw_resource_duals_USD_per_seat": pi,
        "algebraic_upper_bound_multipliers_USD_per_seat": mu,
        "native_objective_USD": objective,
        "native_dual_objective_USD": dual_objective,
        "max_certificate_abs_residual": max(residuals),
        "unique_dual_max_abs_error": error,
        "unique_vector_agreement": error <= tolerance if unique else None,
        "certificate_pass": finite and max(residuals) <= tolerance,
        "case_pass": finite
        and max(residuals) <= tolerance
        and (not unique or error <= tolerance),
    }


def run(cell, context):
    p = cell["parameters"]
    guards = [
        (p["native_source"], p["native_source_sha256"]),
        (p["design_registration"], p["design_registration_sha256"]),
    ]
    for name, expected in guards:
        if file_hash(Path(name)) != expected:
            raise ValueError("frozen DLP input changed: " + name)
    reference = exact_reference(p["A"], p["capacity"], p["revenues"], p["demand"])
    comparison = native_comparison(
        p["A"], p["capacity"], p["revenues"], p["demand"], reference, p["tolerance"]
    )
    for name, expected in guards:
        if file_hash(Path(name)) != expected:
            raise ValueError("frozen DLP input changed during operation: " + name)
    atomic_json(context.directory / "exact_reference.json", reference)
    atomic_json(context.directory / "native_comparison.json", comparison)
    return {
        "status": "completed",
        "case_name": p["name"],
        "DLP_primal_dual_case_pass": comparison["case_pass"],
        "resource_duals_unique": reference["resource_duals_unique"],
        "unique_vector_agreement": comparison["unique_vector_agreement"],
        "simulation_cohorts": 0,
        "actor_forwards": 0,
        "optimizer_updates": 0,
        "full_V4_pass": None,
        "native_fixed_fare_DP_bound_pass": None,
        "scientific_budget": None,
        "artifact_hashes": {
            name: file_hash(context.directory / name)
            for name in ["exact_reference.json", "native_comparison.json"]
        },
        "scope": "Independent exact-rational small-LP values and raw resource-dual certificates; no native fixed-fare admission DP bound, demand forecast, controller pricing, R1 learning or full V4 claim.",
    }
