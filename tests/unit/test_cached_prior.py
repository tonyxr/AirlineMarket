"""Immutable prior-tape reuse preserves the full reference filter and failures."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from airline_market.belief import ParticleBelief
from airline_market.belief.cached_prior import CachedPriorParticleBelief
from airline_market.campaign.instances import build_instance
from airline_market.core.errors import ZeroLikelihoodError
from airline_market.core.types import CoefficientAction
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.sim.engine import MarketEngine
from airline_market.spec import compile_market


@pytest.mark.parametrize("generator", ["toy_hub", "hub_feeder_small"])
def test_cached_prior_matches_reference_particles_and_updates(generator):
    compiled = compile_market(build_instance(generator, {"T": 3, "actions": 2}))
    env = FocalCarrierEnv(compiled, "A", "linked")
    for seed in [200000, 200001]:
        obs, _ = env.reset(seed)
        original = ParticleBelief(compiled, "A", "linked", n_particles=500, seed=773)
        cached = CachedPriorParticleBelief(
            compiled, "A", "linked", n_particles=500, seed=773
        )
        reset = MarketEngine.reset
        original.reset(obs)
        cached.reset(obs)
        assert MarketEngine.reset is reset
        while True:
            assert original.mass == cached.mass
            assert original.particles == cached.particles
            assert np.array_equal(original.weights, cached.weights)
            assert original.rng.bit_generator.state == cached.rng.bit_generator.state
            if obs.t == compiled.spec.horizon.n_periods:
                break
            action = CoefficientAction(tuple(compiled.spec.pricing[0].theta_grid[0]))
            after, _, _, _ = env.step(action)
            original.update(obs, action, after)
            cached.update(obs, action, after)
            obs = after
        assert not cached._construction_tape.gumbel.flags.writeable


def test_impossible_observation_remains_failure_and_reset_binding_restored():
    compiled = compile_market(build_instance("toy_hub", {"T": 3, "actions": 2}))
    env = FocalCarrierEnv(compiled, "A", "direct")
    obs, _ = env.reset(200002)
    impossible = replace(
        obs, quotes=tuple(replace(q, fare=123456.0) for q in obs.quotes)
    )
    reset = MarketEngine.reset
    for cls in [ParticleBelief, CachedPriorParticleBelief]:
        with pytest.raises(ZeroLikelihoodError):
            cls(compiled, "A", "direct", n_particles=100, seed=774).reset(impossible)
        assert MarketEngine.reset is reset


def test_small_f4_runner_records_both_posterior_metrics_and_real_decisions(tmp_path):
    from airline_market.campaign.artifacts import file_hash
    from airline_market.experiments.f4_study import compare

    compiled = compile_market(build_instance("toy_hub", {"T": 3, "actions": 2}))
    tails = tmp_path / "tails.json"
    tails.write_text(
        json.dumps(
            {
                "spec_hash": compiled.spec_hash,
                "focal": "A",
                "fixed": list(compiled.spec.pricing[0].theta_grid[0]),
            }
        )
    )

    class Context:
        directory = tmp_path

        def update(self, values):
            self.latest = values

    result = compare(
        {
            "generator": "toy_hub",
            "parameters": {
                "T": 3,
                "actions": 2,
                "seed_start": 200000,
                "n_cohorts": 1,
                "arm": "direct",
                "particles": 100,
                "planning_scenarios": 4,
                "tails_file": str(tails),
                "tails_sha256": file_hash(tails),
                "rival": "B",
                "joint_legs": ["B_OH", "B_HD"],
            },
        },
        Context(),
    )
    assert result["status"] == "completed"
    assert result["observations"] == 4 and result["decisions"] == 3
    assert result["provisional_criteria_pass"] is None
    assert result["particle_selection_status"].startswith("not_selected")
    rows = json.loads((tmp_path / "posterior_rows.json").read_text())
    assert all(0 <= r["full_hidden_tv"] <= 1 for r in rows)
    assert rows[-1]["posted_price_agreement"] is None
