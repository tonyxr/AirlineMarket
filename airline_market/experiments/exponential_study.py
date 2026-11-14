"""Five prospective exponential-demand scales: exact values, HJB and native tapes."""

import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import norm, skew

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.rng import StreamRegistry
from airline_market.sim.accounting import carrier_metrics
from airline_market.sim.exponential_pricing import (
    draw_tape,
    fixed_price,
    optimal_value,
    run_profile,
)
from airline_market.spec import compile_market, load_spec


def fixed_expectation(capacity, horizon, a, beta):
    """Finite Poisson expectation including its exact remaining tail mass."""
    price = fixed_price(capacity, horizon, a, beta)
    mu = a * horizon * math.exp(-beta * price)
    mass = math.exp(-mu)
    masses = [mass]
    for j in range(1, capacity):
        mass *= mu / j
        masses.append(mass)
    mean_sold = math.fsum(j * q for j, q in enumerate(masses)) + capacity * (
        1 - math.fsum(masses)
    )
    return price * mean_sold


def reference(model, tolerance):
    """Independent Bellman ODE integration against the exponential polynomial."""
    K, a, b, T = model["capacity"], model["a"], model["beta"], model["horizon"]

    def rhs(t, v):
        previous = np.r_[0.0, v[:-1]]
        return a / (b * math.e) * np.exp(-b * (v - previous))

    times = np.linspace(0, T, 101)
    sol = solve_ivp(
        rhs, (0, T), np.zeros(K), method="DOP853", t_eval=times, rtol=2e-13, atol=1e-12
    )
    assert sol.success
    formula = np.asarray(
        [[optimal_value(n, t, a, b) for n in range(1, K + 1)] for t in times]
    )
    error = float(np.max(np.abs(sol.y.T - formula)))
    value = optimal_value(K, T, a, b)
    fp = fixed_expectation(K, T, a, b)
    return {
        "scale": model["scale"],
        "capacity": K,
        "a": a,
        "beta": b,
        "horizon": T,
        "exact_DP_USD": value,
        "exact_fluid_fixed_USD": fp,
        "exact_ratio": fp / value,
        "fixed_price_USD": fixed_price(K, T, a, b),
        "max_HJB_error_USD": error,
        "HJB_tolerance_USD": tolerance,
        "HJB_agreement": error <= tolerance,
        "HJB_nfev": sol.nfev,
        "times": times.tolist(),
        "HJB_values_USD": sol.y.T.tolist(),
        "formula_values_USD": formula.tolist(),
        "rows_processed": 0,
    }


def paired_stats(values, boot):
    """Paired physical-cohort normal/percentile or skew-triggered BCa intervals."""
    n = len(values)
    avg = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / math.sqrt(n)
    sk = float(skew(values, bias=False)) if sd else None
    bca = None
    if sk is not None and abs(sk) > 1:
        fraction = (np.sum(boot < avg) + 0.5 * np.sum(boot == avg)) / len(boot)
        assert 0 < fraction < 1
        center = (values - avg) / (n - 1)
        accel = np.sum(center**3) / (6 * np.sum(center**2) ** 1.5)
        z0 = norm.ppf(fraction)
        z = norm.ppf([0.025, 0.975])
        bca = np.quantile(
            boot, norm.cdf(z0 + (z0 + z) / (1 - accel * (z0 + z)))
        ).tolist()
    return {
        "mean_USD": avg,
        "SE_USD": se,
        "normal_95_USD": [avg - 1.959963984540054 * se, avg + 1.959963984540054 * se],
        "skewness": sk,
        "percentile_95_USD": np.quantile(boot, [0.025, 0.975]).tolist(),
        "BCa_95_USD": bca,
        "reported_bootstrap": "BCa" if bca is not None else "percentile",
    }


def run(cell, context):
    p = cell["parameters"]
    guards = [(p["design_registration"], p["design_registration_sha256"])]
    models = []
    for entry in p["models"]:
        guards.append((entry["path"], entry["sha256"]))
        s = json.loads(Path(entry["path"]).read_text())
        c = compile_market(load_spec(s["physical_spec"]))
        assert c.A[0].tolist() == [[1]] and all(v == 0 for v in c.product_cost)
        assert c.spec.horizon.n_periods == 2 and len(c.spec.products) == 1
        assert c.spec.legs[0].capacity.support[0] == s["capacity"]
        models.append((s, c))
    assert all(file_hash(n) == h for n, h in guards)
    refs = [reference(s, p["HJB_tolerance_USD"]) for s, c in models]
    atomic_json(
        context.directory / "exact_references.json",
        {"scales": refs, "rows_processed": 0},
    )
    refhash = file_hash(context.directory / "exact_references.json")
    N, start = p["cohorts"], p["validation_seed_start"]
    assert 200000 <= start and start + N <= 300000
    seeds = np.arange(start, start + N, dtype=np.int64)
    amax = max(s["a"] for s, c in models)
    T = models[0][0]["horizon"]
    revenues = np.empty((N, len(models), 2))
    sales = np.empty((N, len(models), 2), dtype=np.int64)
    tape_rows = []
    tape_offsets = [0]
    event_rows = []
    profile_offsets = [0]
    bookings = 0
    current = {
        "stage": "native_exponential_profiles",
        "physical_cohorts_completed": 0,
        "profiles_completed": 0,
        "native_bookings": 0,
    }
    context.counters = lambda: dict(current)
    for i, seed in enumerate(seeds):
        tape = draw_tape(int(seed), amax, T)
        tape_rows.append(tape.rows)
        tape_offsets.append(tape_offsets[-1] + len(tape.rows))
        for j, (s, c) in enumerate(models):
            for q, policy in enumerate(["fixed", "dynamic"]):
                world, events = run_profile(
                    c, tape, s["a"] / amax, s["a"], s["beta"], T, policy
                )
                m = carrier_metrics(world, c, 0)
                bs = world.carriers[0].bookings
                assert m["profit"] == m["revenue"] == world.carriers[0].cash and m[
                    "inventory"
                ] == [s["capacity"] - len(bs)]
                assert abs(math.fsum(b.fare for b in bs) - m["revenue"]) <= 1e-8
                revenues[i, j, q] = m["revenue"]
                sales[i, j, q] = len(bs)
                bookings += len(bs)
                event_rows.append(events)
                profile_offsets.append(profile_offsets[-1] + len(events))
        current.update(
            physical_cohorts_completed=i + 1,
            profiles_completed=(i + 1) * len(models) * 2,
            native_bookings=bookings,
        )
    np.savez_compressed(
        context.directory / "native_profiles.npz",
        seeds=seeds,
        tapes=np.concatenate(tape_rows),
        tape_offsets=np.asarray(tape_offsets),
        events=np.concatenate(event_rows),
        profile_offsets=np.asarray(profile_offsets),
        revenues=revenues,
        sales=sales,
    )
    differences = revenues[:, :, 1] - revenues[:, :, 0]
    registry = StreamRegistry(773, "boot")
    boot = np.empty((2000, len(models)))
    indices = np.empty((2000, N), dtype=np.int64)
    for k in range(2000):
        ix = registry.generator(36, k).integers(0, N, size=N)
        indices[k] = ix
        boot[k] = differences[ix].mean(axis=0)
    np.savez_compressed(
        context.directory / "bootstrap.npz", physical_indices=indices, paired_means=boot
    )
    rows = []
    for j, (s, c) in enumerate(models):
        ref = refs[j]
        mc = []
        for q, key in enumerate(["exact_fluid_fixed_USD", "exact_DP_USD"]):
            x = revenues[:, j, q]
            se = float(x.std(ddof=1)) / math.sqrt(N)
            mc.append(
                {
                    "policy": ["fixed", "dynamic"][q],
                    "mean_USD": float(x.mean()),
                    "SE_USD": se,
                    "exact_reference_USD": ref[key],
                    "within_4SE_diagnostic": abs(float(x.mean()) - ref[key]) <= 4 * se,
                }
            )
        stats = paired_stats(differences[:, j], boot[:, j])
        stats.update(
            percent_fixed_revenue=100 * stats["mean_USD"] / mc[0]["mean_USD"],
            oracle_gap_closure=None,
        )
        rows.append(
            {
                "scale": s["scale"],
                "capacity": s["capacity"],
                "a": s["a"],
                "exact_DP_USD": ref["exact_DP_USD"],
                "exact_fixed_USD": ref["exact_fluid_fixed_USD"],
                "ratio": ref["exact_ratio"],
                "HJB_agreement": ref["HJB_agreement"],
                "HJB_error_USD": ref["max_HJB_error_USD"],
                "native_MC": mc,
                "paired_dynamic_minus_fixed": stats,
            }
        )
    ratios = np.asarray([r["ratio"] for r in rows])
    scales = np.asarray([r["scale"] for r in rows])
    slope, intercept = np.polyfit(np.log(scales), np.log1p(-ratios), 1)
    summary = {
        "rows": rows,
        "ratios_strictly_increasing": bool(np.all(np.diff(ratios) > 0)),
        "all_ratios_below_one": bool(np.all(ratios < 1)),
        "fitted_gap_loglog_slope": float(slope),
        "fitted_gap_intercept": float(intercept),
        "fitted_gap_rate": float(-slope),
        "fit_scope": "descriptive fit to five exact expectations; no sampling CI or infinite-limit inference",
        "all_HJB_agree": all(r["HJB_agreement"] for r in rows),
        "physical_cohorts": N,
        "native_profiles": N * len(models) * 2,
        "native_bookings": bookings,
        "physical_tapes_generated": N,
        "potential_events": int(tape_offsets[-1]),
        "profile_event_decisions": int(profile_offsets[-1]),
        "exact_reference_sha256": refhash,
        "bootstrap": {
            "B": 2000,
            "namespace": "boot",
            "root": 773,
            "key": 36,
            "unit": "physical cohort",
            "common_indices_across_scales_and_policies": True,
        },
        "full_V4_pass": None,
        "scientific_budget": None,
    }
    assert refhash == file_hash(context.directory / "exact_references.json")
    assert all(file_hash(n) == h for n, h in guards)
    atomic_json(context.directory / "scaling_summary.json", summary)
    return {
        "status": "completed",
        **summary,
        "artifact_hashes": {
            n: file_hash(context.directory / n)
            for n in [
                "exact_references.json",
                "native_profiles.npz",
                "bootstrap.npz",
                "scaling_summary.json",
            ]
        },
        "scope": "New continuous-price exponential demand/native-ledger analytical benchmark, no original cent/MNL edit, training or final seeds. Exact ratio criterion separate from finite MC4SE diagnostics.",
    }
