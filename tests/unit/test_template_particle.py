"""Template construction is a bitwise-preserving optimization, not a filter change."""

import copy

import numpy as np
import pytest

import airline_market.belief.template_particle as m
from airline_market.belief import particle as particle_module
from airline_market.belief.cached_prior import CachedPriorParticleBelief
from airline_market.belief.particle import ParticleBelief
from airline_market.campaign.instances import build_instance
from airline_market.core.errors import ZeroLikelihoodError
from airline_market.core.hashing import canonical_json
from airline_market.core.types import CoefficientAction
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.sim.kernel import ExactKernel
from airline_market.spec import compile_market, load_spec


def assert_same(left, right):
    assert left.particles == right.particles
    assert left.mass == right.mass
    np.testing.assert_array_equal(left.weights, right.weights)
    assert canonical_json(left.rng.bit_generator.state) == canonical_json(
        right.rng.bit_generator.state
    )
    assert left.history == right.history
    assert left.diagnostics == right.diagnostics


@pytest.mark.parametrize("arm", ["direct", "linked"])
@pytest.mark.parametrize("physical_seed", [200256, 200257])
def test_prior_and_all_updates_match_original_particle_bitwise(arm, physical_seed):
    c = compile_market(build_instance("hub_feeder_small", {"T": 3, "actions": 2}))
    env = FocalCarrierEnv(c, "A", arm)
    obs, _ = env.reset(physical_seed)
    old = ParticleBelief(c, "A", arm, n_particles=64, seed=773)
    new = m.TemplateParticleBelief(c, "A", arm, n_particles=64, seed=773)
    old.reset(obs)
    new.reset(obs)
    assert_same(old, new)
    for _ in range(c.spec.horizon.n_periods):
        action = CoefficientAction(c.spec.pricing[0].theta_grid[1])
        after, _, _, _ = env.step(action)
        old.update(obs, action, after)
        new.update(obs, action, after)
        assert_same(old, new)
        obs = after
    assert new.kernel.template_builds == 1
    assert new.prior_world_copies > 0


def test_simulated_proposals_match_draws_or_the_same_strict_failure():
    c = compile_market(build_instance("hub_feeder_small", {"T": 2, "actions": 2}))
    env = FocalCarrierEnv(c, "A", "direct")
    obs, _ = env.reset(200256)
    params = {"n_particles": 16, "seed": 773, "inner_draws": 8}
    old = ParticleBelief(c, "A", "direct", **params)
    new = m.TemplateParticleBelief(c, "A", "direct", **params)
    for b in [old, new]:
        b.reset(obs)
        b.exact_proposals = False
    assert_same(old, new)
    action = CoefficientAction(c.spec.pricing[0].theta_grid[0])
    after, _, _, _ = env.step(action)
    errors = []
    for b in [old, new]:
        try:
            b.update(obs, action, after)
            errors.append(None)
        except ZeroLikelihoodError as error:
            errors.append(str(error))
    assert errors[0] == errors[1]
    assert_same(old, new)


def test_hub_prior_uses_same_particles_as_cached_reference():
    c = compile_market(load_spec("config/instances/I5.json"))
    env = FocalCarrierEnv(c, "A", "direct")
    obs, _ = env.reset(200256)
    params = {"n_particles": 12, "seed": 773}
    old = CachedPriorParticleBelief(c, "A", "direct", **params)
    new = m.TemplateParticleBelief(c, "A", "direct", **params)
    old.reset(obs)
    new.reset(obs)
    assert_same(old, new)


def test_two_hypothetical_engines_have_independent_world_controller_and_rule_caches():
    c = compile_market(build_instance("hub_feeder_small", {"T": 2, "actions": 2}))
    old, new = ExactKernel(c), m.TemplateExactKernel(c)
    state = old.initial_distribution(0)[0][1]
    expected, first, second = (
        old.engine(state, 0),
        new.engine(state, 0),
        new.engine(state, 0),
    )
    assert old.encode(expected.world, 0) == new.encode(first.world, 0)
    assert first.compiled is second.compiled is c
    assert first.tape is second.tape
    assert first.controllers[1] is not second.controllers[1]
    assert first.rules[0] is not second.rules[0]
    assert first.rules[0].evaluator is not second.rules[0].evaluator
    before = new.encode(second.world, 0)
    first.world.carriers[0].capacity[0] = 999
    first.rules[0].evaluator._observation = "changed"
    first._trace.append("changed")
    assert new.encode(second.world, 0) == before
    assert not hasattr(second.rules[0].evaluator, "_observation")
    assert second._trace == []
    # The cached seed-zero template also remains untouched.
    third = new.engine(state, 0)
    assert new.encode(third.world, 0) == before


def test_impossible_prior_restores_constructor_alias_and_does_not_read_truth():
    from dataclasses import replace

    c = compile_market(build_instance("hub_feeder_small", {"T": 2, "actions": 2}))
    obs, _ = FocalCarrierEnv(c, "A", "linked").reset(200256)
    q = replace(obs.quotes[0], fare=999999)
    impossible = replace(obs, quotes=(q, *obs.quotes[1:]))
    old, new = (
        ParticleBelief(c, "A", "linked", n_particles=16, seed=773),
        m.TemplateParticleBelief(c, "A", "linked", n_particles=16, seed=773),
    )
    original = particle_module.MarketEngine
    for b in [old, new]:
        with pytest.raises(ZeroLikelihoodError):
            b.reset(impossible)
        assert particle_module.MarketEngine is original


def test_branch_restore_and_pre_resampling_ESS_are_preserved():
    c = compile_market(build_instance("hub_feeder_small", {"T": 3, "actions": 2}))
    env = FocalCarrierEnv(c, "A", "direct")
    obs, _ = env.reset(200256)
    belief = m.TemplateParticleBelief(
        c, "A", "direct", n_particles=64, seed=773, ess_fraction=1.1
    )
    belief.reset(obs)
    snapshot = copy.deepcopy(belief.snapshot())
    action = CoefficientAction(c.spec.pricing[0].theta_grid[0])
    after, _, _, _ = env.step(action)
    belief.update(obs, action, after)
    first = copy.deepcopy(belief.snapshot())
    diagnostic = belief.diagnostics[-1]
    assert diagnostic["resampled"]
    assert 0 < diagnostic["ess"] <= 64 + 1e-10
    assert 1 / (belief.weights @ belief.weights) == pytest.approx(64)
    belief.restore(snapshot)
    belief.update(obs, action, after)
    assert canonical_json(belief.snapshot()) == canonical_json(first)


def test_resource_driver_preserves_measured_initial_null_preESS_and_paired_hashes(
    tmp_path,
):
    from types import SimpleNamespace

    from airline_market.campaign.artifacts import file_hash
    from airline_market.experiments import template_particle_study as driver
    from airline_market.spec import dump_spec

    c = build_instance("hub_feeder_small", {"T": 2, "actions": 2})
    source = tmp_path / "spec.json"
    dump_spec(c, source)
    leg = next(x.id for x in c.legs if x.carrier == "B")
    params = {
        "source_spec": str(source),
        "source_spec_sha256": file_hash(source),
        "particles": 16,
        "validation_seed": 200256,
        "operation": "one_update",
        "inner_draws": 8,
        "rival": "B",
        "joint_legs": [leg],
    }
    outputs = []
    for implementation in ["cached_reference", "template"]:
        directory = tmp_path / implementation
        directory.mkdir()
        outputs.append(
            driver.probe(
                {"parameters": {**params, "implementation": implementation}},
                SimpleNamespace(directory=directory, update=lambda fields: None),
            )
        )
    assert outputs[0]["status"] == outputs[1]["status"]
    assert outputs[0]["initial_fingerprint"] == outputs[1]["initial_fingerprint"]
    assert outputs[0]["after_fingerprint"] == outputs[1]["after_fingerprint"]
    import json

    result = json.loads((tmp_path / "template" / "measurements.json").read_text())
    assert result["initial"]["pre_resampling_ESS"] is None
    assert result["after_update"]["pre_resampling_ESS"] > 0
