"""Initial-only capacity-law diagnostic; never substitutes for a full filter.

This reproduces the original prior RNG/grouping/bootstrap order on an explicitly
restricted, audited model. No existing filter, metric or full-horizon result changes.
"""

import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from airline_market.belief import ExactBelief
from airline_market.belief.template_particle import (
    TemplateExactKernel,
    TemplateParticleBelief,
)
from airline_market.campaign.artifacts import atomic_json, file_hash, utc_now
from airline_market.core.hashing import canonical_json
from airline_market.core.rng import StreamRegistry
from airline_market.experiments.template_particle_study import fingerprint
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.spec import compile_market, load_spec


def audit(c):
    """Certify the restricted initial law, not later observations or proposals."""
    s = c.spec
    arm = next(a for a in s.information if a.name == "direct" and a.observer == "A")
    assert not arm.privileged and not s.capacity_priors
    assert len(s.exogenous.states) == 1 and s.exogenous.initial == (1.0,)
    assert s.exogenous.states[0].public
    assert all(
        not leg.capacity.public and min(leg.capacity.support) > 0 for leg in s.legs
    )
    assert all(leg.capacity.support == (3,) for leg in s.legs if leg.carrier == "A")
    relevant = [
        q for q in s.quotes.requests if q.id in arm.requests or q.observer != "A"
    ]
    for q in relevant:
        assert q.lag_periods == 1 and q.availability == "flag"
        assert q.missing_probability == q.fare_noise_sd == 0
        assert all(target in c.product_index for target in q.target_products)
    # Initial quote fares are posted reference fares; positive capacities make
    # every flag true. Neither the flag nor fare exposes the underlying seat count.
    private = [
        leg
        for i in range(len(s.carriers))
        if s.carriers[i].code != "A"
        for index in c.legs_of[i]
        for leg in [s.legs[index]]
        if len(leg.capacity.support) > 1
    ]
    assert len(private) == 32
    assert all(
        leg.capacity.support == (1, 3) and leg.capacity.probabilities == (0.5, 0.5)
        for leg in private
    )
    assert all(len(leg.capacity.support) == 1 or leg in private for leg in s.legs)
    return [leg.id for leg in private]


def cloud(c, n, index):
    """Keep all original deterministic-leg and exogenous RNG draws and resampling."""
    private = audit(c)
    position = {leg: k for k, leg in enumerate(private)}
    rng = StreamRegistry(
        StreamRegistry(773, "plan/F4_hub/filter").child_seed(index), "plan/belief"
    ).generator()
    codes = np.zeros(n, dtype=np.uint64)
    for leg in c.spec.legs:
        p = np.asarray(leg.capacity.probabilities)
        p = p / sum(p)
        draw = rng.choice(leg.capacity.support, size=n, p=p)
        if leg.id in position:
            codes |= (draw == 3).astype(np.uint64) << np.uint64(31 - position[leg.id])
    # Untracked background priors make no draws; the single valid exogenous
    # state still consumes the reference categorical draw for every particle.
    rng.choice([0], size=n, p=np.array([1.0]))
    grouped = defaultdict(int)
    for value in codes.tolist():
        grouped[value] += 1
    total = sum(count / n for count in grouped.values())
    ordered = sorted(grouped)
    cdf = np.cumsum([grouped[value] / n / total for value in ordered])
    idx = np.minimum(
        np.searchsorted(cdf, rng.random(n), side="right"), len(ordered) - 1
    )
    sampled = np.asarray(ordered, dtype=np.uint64)[idx]
    mass = defaultdict(float)
    for value in sampled.tolist():
        mass[value] += float(1 / n)
    total = sum(mass.values())
    mass = {value: p / total for value, p in mass.items()}
    return sampled, np.full(n, 1 / n), rng.bit_generator.state, mass


def codes_from_capacities(c, capacity):
    """Project the archived ordered rival capacity vectors without losing variables."""
    private = audit(c)
    rival_legs = [
        c.spec.legs[leg].id
        for i in range(len(c.spec.carriers))
        if i != 0
        for leg in c.legs_of[i]
    ]
    result = np.zeros(len(capacity), dtype=np.uint64)
    for k, leg in enumerate(private):
        result |= (capacity[:, rival_legs.index(leg)] == 3).astype(
            np.uint64
        ) << np.uint64(31 - k)
    return result


def reconstructed_fingerprint(c, sampled, weights, rng_state):
    """Recreate the declared HiddenState keys from model-generated static state.

    The noncomparing kernel pointer is excluded from repr by its existing type.
    This checks the complete initial hidden-key order, not future kernel paths.
    """
    kernel = TemplateExactKernel(c, validate=False)
    base = kernel.encode(kernel.initial_engine().world, 0)
    belief = ExactBelief(c, "A", "direct", kernel=kernel)
    private = audit(c)
    cache = {}
    digest = hashlib.sha256()
    for code in sampled.tolist():
        if code not in cache:
            caps = {
                leg: 1 + 2 * ((code >> (31 - k)) & 1) for k, leg in enumerate(private)
            }
            carriers = tuple(
                replace(
                    state,
                    capacity=tuple(
                        caps.get(c.spec.legs[leg].id, state.capacity[k])
                        for k, leg in enumerate(c.legs_of[i])
                    ),
                )
                for i, state in enumerate(base.carriers)
            )
            cache[code] = belief._hidden(replace(base, carriers=carriers))
        digest.update(repr(cache[code]).encode())
        digest.update(b"\n")
    digest.update(weights.tobytes())
    digest.update(canonical_json(rng_state).encode())
    return digest.hexdigest()


def run(config):
    """Verify against real filters and the sealed profiled500 cloud before analysis."""
    out = Path(config["out"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETE.json").exists():
        raise ValueError("completed matching diagnostic exists")
    path = Path(config["source_spec"])
    assert file_hash(path) == config["source_spec_sha256"]
    c = compile_market(load_spec(path))
    private = audit(c)
    profile = Path(config["profile_root"])
    saved = json.load(open(profile / "prior.json"))
    actual = np.load(profile / "initial_cloud.npz")
    sampled, weights, rng_state, _ = cloud(c, 500, 0)
    np.testing.assert_array_equal(
        sampled, codes_from_capacities(c, actual["capacities"])
    )
    np.testing.assert_array_equal(weights, actual["weights"])
    assert canonical_json(rng_state) == canonical_json(
        json.load(open(profile / "initial_rng.json"))
    )
    reconstructed = reconstructed_fingerprint(c, sampled, weights, rng_state)
    assert reconstructed == saved["fingerprint"] == config["sealed_fingerprint"]
    # Additional genuinely small native resets check more than oneN/draw order.
    checks = []
    for n, index in [(5, 1), (13, 2)]:
        env = FocalCarrierEnv(c, "A", "direct")
        obs, _ = env.reset(200256 + index)
        b = TemplateParticleBelief(
            c,
            "A",
            "direct",
            n_particles=n,
            seed=StreamRegistry(773, "plan/F4_hub/filter").child_seed(index),
            strict=True,
        )
        assert not b.track_background
        b.reset(obs)
        small, w, state, _ = cloud(c, n, index)
        assert reconstructed_fingerprint(c, small, w, state) == fingerprint(b)
        checks.append(
            {
                "N": n,
                "stream_index": index,
                "native_ordered_hidden_weights_rng_match": True,
            }
        )
    atomic_json(
        out / "verification.json",
        {
            "profiled500_exact_order_weights_rng_match": True,
            "sealed500_fingerprint_match": True,
            "small_native_checks": checks,
            "private_variable_count": len(private),
            "initial_support": 2 ** len(private),
            "physical_initial_observation_scope": "All capacities positive, initial lag1 flags constant; single public exogenous state, deterministic own capacities. Background unqueried/untracked. Only initial hidden keys certified, not transition/proposal laws.",
        },
    )
    rows = []
    for index in range(config["cohorts"]):
        _, _, _, reference = cloud(c, 100000, index)
        for n in config["particles"]:
            _, _, _, candidate = cloud(c, n, index)
            tv = 0.5 * sum(
                abs(candidate.get(k, 0.0) - reference.get(k, 0.0))
                for k in candidate.keys() | reference.keys()
            )
            rows.append(
                {
                    "stream_index": index,
                    "N": n,
                    "initial_empirical_full_hidden_tv": tv,
                    "candidate_support": len(candidate),
                    "reference_support": len(reference),
                    "overlap_states": len(candidate.keys() & reference.keys()),
                    "necessary_all_frame_mean_lower_bound": tv
                    / (c.spec.horizon.n_periods + 1),
                }
            )
        atomic_json(out / "initial_rows.json", rows)
    per_n = []
    for n in config["particles"]:
        cases = [r for r in rows if r["N"] == n]
        lower = float(
            np.mean([r["necessary_all_frame_mean_lower_bound"] for r in cases])
        )
        per_n.append(
            {
                "N": n,
                "initial_tv_mean": float(
                    np.mean([r["initial_empirical_full_hidden_tv"] for r in cases])
                ),
                "minimum_initial_tv": min(
                    r["initial_empirical_full_hidden_tv"] for r in cases
                ),
                "all_frame_mean_tv_lower_bound": lower,
                "exceeds_registered_mean_TV_point02": lower > 0.02,
            }
        )
    result = {
        "completed_at": utc_now(),
        "initial_only": True,
        "cohorts": config["cohorts"],
        "per_N": per_n,
        "full100k_trajectory_completed": False,
        "global_N_selected": None,
        "scope": "Necessary-condition diagnostic under the explicitly registered indexed initial streams and independentN/count-dependent prior draws. All later frame TVs are nonnegative. This is not a full100k reference trajectory, not a successful rerun of the memory-limited implementation, and does not modify the registered full-hidden-TV metric. The analytic true-prior sparse-support bound and the empirical-reference comparison are distinct.",
        "true_initial_prior_TV_lower_bound_at_N50000": 1 - 50000 / (2**32),
    }
    atomic_json(out / "summary.json", result)
    atomic_json(
        out / "COMPLETE.json",
        {
            "summary_sha256": file_hash(out / "summary.json"),
            "rows_sha256": file_hash(out / "initial_rows.json"),
        },
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run(json.loads(Path(sys.argv[1]).read_text()))
