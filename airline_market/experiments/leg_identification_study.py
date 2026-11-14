"""Read-only native-control identification and DLP-unit audit, not R1 acceptance.

A many-to-one executed price map is not a symmetry of Gaussian policy densities
or neural weights. Saved sampled controls are not deterministic actor means.
"""

import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from airline_market.agents.bidprice import solve_dlp
from airline_market.agents.forecast import remaining_demand
from airline_market.agents.price_rule import AdditivePriceRule
from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.hashing import jsonable
from airline_market.learn.structured_runtime import StructuredAdapter
from airline_market.pomdp.observation import Observation, QuoteObs
from airline_market.sim.state import Booking
from airline_market.spec import compile_market, load_spec


def exact_solve(matrix, target):
    """Solve a full-column-rank rational system; reject inconsistent features."""
    if (
        len(matrix) != len(target)
        or not len(matrix)
        or not len(matrix[0])
        or any(len(row) != len(matrix[0]) for row in matrix)
        or any(x != int(x) for row in matrix for x in row)
        or any(y != int(y) for y in target)
    ):
        raise ValueError("exact static incidence system requires matching integer data")
    rows = [
        [Fraction(int(x)) for x in row] + [Fraction(int(y))]
        for row, y in zip(matrix, target)
    ]
    n = len(rows[0]) - 1
    pivot, positions = 0, []
    for col in range(n):
        use = next((k for k in range(pivot, len(rows)) if rows[k][col]), None)
        if use is None:
            continue
        rows[pivot], rows[use] = rows[use], rows[pivot]
        scale = rows[pivot][col]
        rows[pivot] = [x / scale for x in rows[pivot]]
        for k in range(len(rows)):
            if k != pivot:
                scale = rows[k][col]
                rows[k] = [a - scale * b for a, b in zip(rows[k], rows[pivot])]
        positions.append(col)
        pivot += 1
    if any(all(x == 0 for x in r[:n]) and r[n] for r in rows) or len(positions) != n:
        raise ValueError("feature not uniquely represented by full-rank incidence")
    answer = [Fraction(0) for _ in range(n)]
    for k, col in enumerate(positions):
        answer[col] = rows[k][n]
    return answer


def price_view(c, focal="A", arm="direct"):
    """Use original pricing on public fields; no engine, world, tape or reset."""
    i = c.carrier_index[focal]
    info = next(a for a in c.spec.information if a.observer == focal and a.name == arm)
    p = next(p for p in c.spec.pricing if p.carrier == focal)
    context = SimpleNamespace(
        compiled=c,
        i=i,
        focal=focal,
        arm=info,
        pricing=p,
        price_rule=AdditivePriceRule(c, p, info),
    )
    return StructuredAdapter(context)


def decode_observation(data):
    """Restore only already-saved permitted observations, not a simulator state."""
    if data.get("privileged") is not None:
        raise ValueError(
            "identification audit requires nonprivileged saved observations"
        )
    x = dict(data)
    for name in ["own_capacity", "own_previous_prices", "own_inventory"]:
        x[name] = tuple(x[name])
    x["own_posted_prices"] = (
        tuple(x["own_posted_prices"]) if x["own_posted_prices"] is not None else None
    )
    x["own_purchases"] = tuple(tuple(v) for v in x.get("own_purchases", ()))
    x["own_bookings"] = tuple(Booking(**b) for b in x["own_bookings"])
    x["quotes"] = tuple(QuoteObs(**q) for q in x["quotes"])
    return Observation(**x)


def feature_leg_basis(adapter, obs):
    """Exact incidence representation of native intercept/connection/scarcity."""
    c, env = adapter.c, adapter.env
    p = env.pricing
    names = [f.name for f in p.controllable_features]
    if p.sharing != "carrier" or names != [
        "intercept",
        "connection_indicator",
        "scarcity_sum",
    ]:
        raise ValueError("audit requires the original shared three-feature class")
    A = np.asarray(c.A[env.i], dtype=int)
    static = [
        np.ones(len(adapter.products), dtype=int),
        np.array(
            [len(c.spec.products[j].legs) > 1 for j in adapter.products], dtype=int
        ),
    ]
    columns = [exact_solve(A.T, f) for f in static]
    columns.append([Fraction(1, 1 + int(x)) for x in obs.own_inventory])
    U = np.asarray(
        [[float(columns[k][l]) for k in range(3)] for l in range(A.shape[0])]
    )
    features = np.array(
        [
            [env.price_rule.evaluator(f, j, obs) for f in p.controllable_features]
            for j in adapter.products
        ]
    )
    for j in range(A.shape[1]):
        expected = [
            Fraction(int(static[0][j])),
            Fraction(int(static[1][j])),
            sum(Fraction(int(A[l, j])) * columns[2][l] for l in range(A.shape[0])),
        ]
        for k in range(3):
            assert (
                sum(Fraction(int(A[l, j])) * columns[k][l] for l in range(A.shape[0]))
                == expected[k]
            )
    if not np.allclose(A.T @ U, features, atol=1e-12, rtol=0):
        raise AssertionError(
            "native feature matrix differs from exact incidence representation"
        )
    return A, features, U


def shifted_controls(adapter, obs, physical, cap=1.0, minimum=1e-8):
    """Symmetric legal shifts with recorded amplitudes, no clipping or nudging."""
    A, features, U = feature_leg_basis(adapter, obs)
    n = adapter.layout.n_legs
    physical = np.asarray(physical, dtype=float)
    pi, theta = np.logaddexp(0.0, physical[:n]), physical[n:]
    bounds = np.asarray(adapter.env.pricing.theta_bounds)
    original = adapter.price_action(physical, obs).prices
    results = []
    for k in range(3):
        u = U[:, k]
        positive = np.abs(u) > 0
        leg_bound = (
            float(np.min(pi[positive] / np.abs(u[positive])))
            if positive.any()
            else float("inf")
        )
        amplitude = min(
            cap,
            0.5 * (theta[k] - bounds[k, 0]),
            0.5 * (bounds[k, 1] - theta[k]),
            0.5 * leg_bound,
        )
        if amplitude < minimum:
            results.append(
                {
                    "feature": k,
                    "status": "amplitude_limited_not_checked",
                    "amplitude": amplitude,
                    "requested_cap": cap,
                }
            )
            continue
        for direction in [-1, 1]:
            delta = direction * amplitude
            new_pi, new_theta = pi - delta * u, theta.copy()
            new_theta[k] += delta
            if (
                np.any(new_pi <= 0)
                or np.any(new_theta < bounds[:, 0])
                or np.any(new_theta > bounds[:, 1])
            ):
                raise AssertionError(
                    "declared symmetric shift left the original control bounds"
                )
            # Stable inverse softplus. Do not perturb a half-cent boundary to force parity.
            preactivation = new_pi + np.log(-np.expm1(-new_pi))
            transformed = np.r_[preactivation, new_theta]
            changed = adapter.price_action(transformed, obs).prices
            executed_pi = np.logaddexp(0.0, preactivation)
            residual = A.T @ (executed_pi - pi) + features @ (new_theta - theta)
            results.append(
                {
                    "feature": k,
                    "direction": direction,
                    "status": "checked",
                    "amplitude": amplitude,
                    "requested_cap": cap,
                    "raw_price_difference_max": float(np.max(np.abs(residual))),
                    "cent_prices_equal": changed == original,
                    "max_leg_value_change": float(np.max(np.abs(executed_pi - pi))),
                    "original_leg_values": pi.tolist(),
                    "changed_leg_values": executed_pi.tolist(),
                    "inverse_softplus_roundtrip_error": float(
                        np.max(np.abs(executed_pi - new_pi))
                    ),
                    "changed_physical": transformed.tolist(),
                    "changed_prices": changed,
                }
            )
    return results


def lp_comparison(adapter, obs, physical):
    """Original reference-offer DLP, unrounded net-dollar duals, not deterministic demand."""
    c, i = adapter.c, adapter.env.i
    capacity = np.asarray(obs.own_inventory, dtype=float)
    revenues = np.array(
        [
            c.spec.products[j].reference_fare - c.product_cost[j]
            for j in adapter.products
        ]
    )
    demand = remaining_demand(c, i, obs)
    x, dual, value = solve_dlp(c.A[i], capacity, revenues, demand)
    mu = np.maximum(0.0, revenues - c.A[i].T @ dual)
    gap = float(capacity @ dual + demand @ mu - value)
    if (
        abs(gap) > 1e-8
        or np.min(dual) < -1e-9
        or np.min(x) < -1e-9
        or np.any(c.A[i] @ x > capacity + 1e-9)
    ):
        raise AssertionError("original DLP primal/dual certificate failed")
    pi = np.logaddexp(0.0, np.asarray(physical[: adapter.layout.n_legs]))
    return {
        "sampled_leg_values": pi.tolist(),
        "DLP_duals": dual.tolist(),
        "DLP_value": value,
        "DLP_dual_gap": gap,
        "reference_net_revenues": revenues.tolist(),
        "forecast_demand": demand.tolist(),
        "capacity": capacity.tolist(),
    }


def descriptive_metrics(pi, target):
    """Never turn zero/constant DLP targets into manufactured relative errors."""
    pi, target = np.asarray(pi), np.asarray(target)
    if pi.shape != target.shape or pi.ndim != 1 or not np.isfinite([pi, target]).all():
        raise ValueError("invalid descriptive control/dual vectors")
    zero = target == 0
    positive = target > 0
    pearson = (
        float(np.corrcoef(pi, target)[0, 1])
        if len(pi) > 1 and np.std(pi) > 0 and np.std(target) > 0
        else None
    )
    relative = np.abs((pi[positive] - target[positive]) / target[positive])
    return {
        "components": len(pi),
        "Pearson": pearson,
        "median_relative_error_all_components": float(np.median(relative))
        if positive.all()
        else None,
        "positive_target_components": int(positive.sum()),
        "positive_target_median_relative_error": float(np.median(relative))
        if positive.any()
        else None,
        "zero_target_components": int(zero.sum()),
        "negative_target_components": int((target < 0).sum()),
        "zero_target_max_absolute_error": float(np.max(np.abs(pi[zero])))
        if zero.any()
        else None,
        "R1_acceptance": None,
        "scope": "Saved sampled controls on stochastic models, not deterministic actor means or the deterministic-demand V7R1 benchmark.",
    }


def run(cell, context):
    """New sealed read-only audit of every saved native control; no source rollout."""
    p = cell["parameters"]
    for path, h in p["input_hashes"].items():
        if file_hash(path) != h:
            raise ValueError("registered saved source changed")
    receipt = json.loads(Path(p["source_receipt"]).read_text())
    if receipt["results_sha256"] != file_hash(p["source_results"]):
        raise ValueError("source result receipt mismatch")
    result = json.loads(Path(p["source_results"]).read_text())
    source = Path(p["source_rows"])
    expected = result.get("artifact_hashes", {}).get(
        source.name, result.get("comparisons_sha256")
    )
    if expected != file_hash(source):
        raise ValueError(
            "saved rows are not transitively hashed by sealed source results"
        )
    c = compile_market(load_spec(p["source_spec"]))
    if c.spec_hash != result["spec_hash"] or c.spec_hash != p["compiled_spec_hash"]:
        raise ValueError("saved compiled physical model mismatch")
    adapter = price_view(c)
    data = [json.loads(line) for line in source.read_text().splitlines()]
    if len(data) != p["rows"]:
        raise ValueError("registered saved row count mismatch")
    current = {"stage": "read_only_controls", "rows_total": len(data)}
    context.counters = lambda: current.copy()
    all_pi, all_dual, checks, limited = [], [], [], 0
    for k, row in enumerate(data):
        current["rows_completed"] = k
        obs = decode_observation(row[p["observation_field"]])
        physical = (
            row["physical_action"]
            if p["kind"] == "event"
            else row["physical_alternatives"][0]
        )
        prices = row["prices"] if p["kind"] == "event" else row["price_alternatives"][0]
        if adapter.price_action(physical, obs).prices != {
            int(j): v for j, v in prices.items()
        }:
            raise AssertionError(
                "saved native prices do not replay from saved permitted controls"
            )
        transforms = shifted_controls(adapter, obs, physical)
        limited += sum(x["status"] != "checked" for x in transforms)
        checks.extend(x for x in transforms if x["status"] == "checked")
        lp = lp_comparison(adapter, obs, physical)
        all_pi.extend(lp["sampled_leg_values"])
        all_dual.extend(lp["DLP_duals"])
        record = {
            "source_index": k,
            "seed": row["seed"],
            "period": obs.t,
            "transforms": transforms,
            "original_prices": prices,
            "DLP": lp,
        }
        with (context.directory / "control_checks.jsonl").open("a") as f:
            f.write(json.dumps(jsonable(record), allow_nan=False) + "\n")
    metrics = descriptive_metrics(all_pi, all_dual)
    report = {
        "rows": len(data),
        "physical_seed_IDs": len({r["seed"] for r in data}),
        "checked_transforms": len(checks),
        "amplitude_limited_feature_rows": limited,
        "cent_price_mismatches": sum(not r["cent_prices_equal"] for r in checks),
        "max_raw_price_difference": max(
            (r["raw_price_difference_max"] for r in checks), default=None
        ),
        "max_leg_value_change": max(
            (r["max_leg_value_change"] for r in checks), default=None
        ),
        "price_control_nullity_at_least": 3,
        "descriptive_DLP_metrics": metrics,
        "R1_acceptance": None,
        "neural_policy_density_equivalence": False,
        "scope": "Constructive bounded action-price-map ambiguity only. No neural parameter/density/entropy equivalence, new training/physical simulation, deterministic demand benchmark, exact DP, profits or R1pass.",
    }
    atomic_json(context.directory / "identification.json", report)
    return {
        "status": "completed",
        "spec_hash": c.spec_hash,
        **report,
        "checks_sha256": file_hash(context.directory / "control_checks.jsonl"),
        "report_sha256": file_hash(context.directory / "identification.json"),
        "scientific_RL_budget": None,
    }
