"""Independent initial-law checks and strict scope guards for resource diagnostics."""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from airline_market.belief.particle import ParticleBelief
from airline_market.core.rng import StreamRegistry
from airline_market.experiments.template_particle_study import fingerprint
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.spec import compile_market, load_spec


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(file))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


diag = module("initial_tv_check", "analysis/feasibility/initial_capacity_tv_v2.py")
profile = module("profile_check", "analysis/feasibility/profile_hub_filter.py")


@pytest.fixture(scope="module")
def compiled():
    return compile_market(load_spec("config/instances/I6.json"))


def test_initial_codes_match_original_unmodified_particle_filter(compiled):
    env = FocalCarrierEnv(compiled, "A", "direct")
    obs, _ = env.reset(200259)
    original = ParticleBelief(
        compiled,
        "A",
        "direct",
        n_particles=7,
        seed=StreamRegistry(773, "plan/F4_hub/filter").child_seed(3),
    )
    original.reset(obs)
    codes, weights, state, mass = diag.cloud(compiled, 7, 3)
    assert diag.reconstructed_fingerprint(
        compiled, codes, weights, state
    ) == fingerprint(original)
    capacities = np.array(
        [[x for r in h.rivals for x in r.capacity] for h in original.particles]
    )
    np.testing.assert_array_equal(
        codes, diag.codes_from_capacities(compiled, capacities)
    )
    np.testing.assert_array_equal(weights, original.weights)
    assert sum(mass.values()) == pytest.approx(1)


def test_audit_retains_irrelevant_background_channels_without_conditioning_on_them(
    compiled,
):
    assert any(
        q.target_products[0].startswith("XY_") for q in compiled.spec.quotes.requests
    )
    private = diag.audit(compiled)
    assert len(private) == 32
    assert 2 ** len(private) == 4294967296


@pytest.mark.parametrize(
    "field,value",
    [
        ("availability", "seats_capped"),
        ("lag_periods", 0),
        ("fare_noise_sd", 1.0),
        ("missing_probability", 0.1),
    ],
)
def test_future_observation_law_changes_refuse_the_initial_uniform_shortcut(
    compiled, field, value
):
    requests = compiled.spec.quotes.requests
    changed = replace(requests[0], **{field: value})
    spec = replace(
        compiled.spec,
        quotes=replace(compiled.spec.quotes, requests=(changed, *requests[1:])),
    )
    c = SimpleNamespace(
        spec=spec, legs_of=compiled.legs_of, product_index=compiled.product_index
    )
    with pytest.raises(AssertionError):
        diag.audit(c)


def test_graph_measure_counts_shared_container_once_and_includes_array_base():
    shared = [17]
    root = [shared, shared]
    assert profile.graph_size(root) == sys.getsizeof(root) + sys.getsizeof(
        shared
    ) + sys.getsizeof(17)
    base = np.arange(10)
    view = base[2:]
    assert profile.graph_size([base, view]) == sys.getsizeof(
        [base, view]
    ) + sys.getsizeof(base) + sys.getsizeof(view)
