"""Registered100k-cohort full protection grid versus discrete Littlewood's rule."""

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from airline_market.sim.batch_admission import BatchAdmissionEngine, draw_tape, run_tape
from scipy.stats import norm, skew

from airline_market.campaign.artifacts import atomic_json, file_hash
from airline_market.core.rng import StreamRegistry
from airline_market.sim.accounting import carrier_metrics
from airline_market.spec import compile_market, load_spec


def binomial_pmf(n, p):
    """Exact rational finite demand law, independent of sampler/native ledger."""
    p = Fraction(str(p))
    return [Fraction(math.comb(n, k)) * p**k * (1 - p) ** (n - k) for k in range(n + 1)]


def exact_design(source):
    """Independent source-formula means and discrete marginal-seat criterion."""
    low, high = source["demand_counts"]
    pl, ph = binomial_pmf(**low), binomial_pmf(**high)
    c = source["physical_spec"]["legs"][0]["capacity"]["support"][0]
    fl, fh = [
        Fraction(str(p["reference_fare"])) for p in source["physical_spec"]["products"]
    ]
    y = next((k for k in range(c + 1) if fh * sum(ph[k + 1 :]) <= fl), c)
    means = []
    for k in range(c + 1):
        v = Fraction(0)
        for l, p in enumerate(pl):
            a = min(l, c - k)
            for h, q in enumerate(ph):
                v += p * q * (fl * a + fh * min(h, c - a))
        means.append(v)
    best = max(means)
    return {
        "low_pmf": list(map(str, pl)),
        "high_pmf": list(map(str, ph)),
        "Littlewood_protection": y,
        "discrete_rule": "smallest y in0..C with fH*P(DH>y)<=fL; truncate at C",
        "exact_grid_means_USD": list(map(str, means)),
        "exact_optimal_set": [i for i, v in enumerate(means) if v == best],
        "rows_processed": 0,
    }


def paired_grid_statistics(counts, payoff, B=2000):
    """Whole-cohort multinomial histogram bootstrap; same resample for all y."""
    n = int(counts.sum())
    means = counts @ payoff / n
    registry = StreamRegistry(773, "boot")
    draws = np.empty((B, payoff.shape[1]))
    for k in range(B):
        weights = registry.generator(35, k).multinomial(n, counts / n)
        draws[k] = weights @ payoff / n
    return means, draws


def run(cell, context):
    p = cell["parameters"]
    guards = [
        (p["problem"], p["problem_sha256"]),
        (p["design_registration"], p["design_registration_sha256"]),
    ]
    assert all(file_hash(n) == h for n, h in guards)
    source = json.loads(Path(p["problem"]).read_text())
    c = compile_market(load_spec(source["physical_spec"]))
    assert len(c.spec.legs) == 1 and all(x == 0 for x in c.product_cost)
    assert c.A[0].tolist() == [[1, 1]]
    engine = BatchAdmissionEngine(c)
    K = c.spec.legs[0].capacity.support[0]
    grid = list(range(K + 1))
    exact = exact_design(source)
    atomic_json(context.directory / "exact_reference.json", exact)
    reference_hash = file_hash(context.directory / "exact_reference.json")
    N = p["cohorts"]
    start = p["validation_seed_start"]
    assert 200000 <= start and start + N <= 300000
    seeds = np.arange(start, start + N, dtype=np.int64)
    tape = np.empty((N, 2), dtype=np.int16)
    # Lossless native Booking fields; carrier0, fare-by-product, segment/taste0
    # are schema constants. Pad unused booking slots with -1.
    ledger = np.full((N, len(grid), K, 4), -1, dtype=np.int16)
    outcomes = np.empty((N, len(grid), 5), dtype=np.float64)
    bookings = requests = 0
    for i, seed in enumerate(seeds):
        d = draw_tape(
            int(seed),
            source["demand_counts"][0]["n"],
            float(source["demand_counts"][0]["p"]),
            source["demand_counts"][1]["n"],
            float(source["demand_counts"][1]["p"]),
        )
        tape[i] = [d.low, d.high]
        for y in grid:
            world = run_tape(engine, d, y)
            m = carrier_metrics(world, c, 0)
            bs = world.carriers[0].bookings
            nl = sum(b.product == 0 for b in bs)
            nh = len(bs) - nl
            assert m["revenue"] == m["profit"] == world.carriers[0].cash
            assert (
                m["inventory"] == [K - len(bs)]
                and len(world.request_log) == d.low + d.high
            )
            outcomes[i, y] = [nl, nh, m["revenue"], m["profit"], m["inventory"][0]]
            for k, b in enumerate(bs):
                assert b.carrier == b.segment == b.taste_class == 0
                assert b.fare == c.spec.products[b.product].reference_fare
                ledger[i, y, k] = [b.record_id, b.product, b.period, b.arrival_rank]
            bookings += len(bs)
            requests += d.low + d.high
        if (i + 1) % 1000 == 0:
            context.counter.update(
                stage="native_protection_grid",
                cohorts_completed=i + 1,
                cohorts_total=N,
                protection_profiles=(i + 1) * len(grid),
                request_decisions=requests,
                native_bookings=bookings,
            )
    np.savez_compressed(
        context.directory / "native_cohorts.npz",
        seeds=seeds,
        demand_tape=tape,
        native_bookings=ledger,
        outcomes=outcomes,
    )
    # Independent source-incidence formula is a check, never a simulation shortcut.
    low_n, high_n = [d["n"] for d in source["demand_counts"]]
    hist = np.zeros((low_n + 1, high_n + 1), dtype=np.int64)
    np.add.at(hist, (tape[:, 0], tape[:, 1]), 1)
    payoff = []
    fl, fh = [x.reference_fare for x in c.spec.products]
    for l in range(low_n + 1):
        for h in range(high_n + 1):
            payoff.append(
                [fl * min(l, K - y) + fh * min(h, K - min(l, K - y)) for y in grid]
            )
    payoff = np.asarray(payoff)
    flat = hist.ravel()
    means, draws = paired_grid_statistics(flat, payoff)
    assert np.array_equal(means, outcomes[:, :, 2].mean(axis=0))
    np.savez_compressed(
        context.directory / "bootstrap_means.npz",
        means=draws,
        empirical_demand_histogram=hist,
    )
    best = np.flatnonzero(means == means.max()).tolist()
    target = exact["Littlewood_protection"]
    distance = max(abs(y - target) for y in best)
    diff = outcomes[:, target, 2] - outcomes[:, 0, 2]
    avg = float(diff.mean())
    sd = float(diff.std(ddof=1))
    se = sd / math.sqrt(N)
    sample_skew = float(skew(diff, bias=False)) if sd else None
    bd = draws[:, target] - draws[:, 0]
    bca = None
    if sample_skew is not None and abs(sample_skew) > 1:
        bias = (np.sum(bd < avg) + 0.5 * np.sum(bd == avg)) / len(bd)
        assert 0 < bias < 1
        centered = (diff - avg) / (N - 1)
        accel = np.sum(centered**3) / (6 * np.sum(centered**2) ** 1.5)
        z0 = norm.ppf(bias)
        z = norm.ppf([0.025, 0.975])
        levels = norm.cdf(z0 + (z0 + z) / (1 - accel * (z0 + z)))
        bca = np.quantile(bd, levels).tolist()
    summary = {
        "n_physical_cohorts": N,
        "n_native_profiles": N * len(grid),
        "grid": grid,
        "native_grid_means_USD": means.tolist(),
        "sample_optimal_set": best,
        "Littlewood_protection": target,
        "max_optimal_set_distance_seats": distance,
        "Littlewood_within_one_seat": distance <= 1,
        "paired_prespecified_Littlewood_minus_y0": {
            "mean_USD": avg,
            "SE_USD": se,
            "normal_95_USD": [
                avg - 1.959963984540054 * se,
                avg + 1.959963984540054 * se,
            ],
            "percentile_95_USD": np.quantile(bd, [0.025, 0.975]).tolist(),
            "BCa_95_USD": bca,
            "sample_skewness": sample_skew,
            "reported_bootstrap": "BCa" if bca is not None else "percentile",
            "percent_y0_revenue": 100 * avg / means[0],
            "oracle_gap_closure": None,
        },
        "native_bookings": bookings,
        "request_decisions": requests,
        "bootstrap_namespace": "boot",
        "bootstrap_root": 773,
        "bootstrap_key": 35,
        "bootstrap_replicates": 2000,
        "bootstrap_resampling": "multinomial resampling of whole-cohort demand-pair histogram; common counts across all protection levels",
        "exact_reference_sha256": reference_hash,
        "physical_tapes_generated": N,
        "training_cohorts": 0,
        "full_V4_pass": None,
        "scientific_budget": None,
    }
    assert reference_hash == file_hash(context.directory / "exact_reference.json")
    assert all(file_hash(n) == h for n, h in guards)
    atomic_json(context.directory / "grid_summary.json", summary)
    return {
        "status": "completed",
        **summary,
        "artifact_hashes": {
            n: file_hash(context.directory / n)
            for n in [
                "exact_reference.json",
                "native_cohorts.npz",
                "bootstrap_means.npz",
                "grid_summary.json",
            ]
        },
        "scope": "Discrete independent low-before-high demand, native-ledger batch admission at fixed fares; original MNL protocol unchanged. Grid selection is validation only, no learning or full V4 claim.",
    }
