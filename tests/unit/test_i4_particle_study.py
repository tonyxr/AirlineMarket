"""Frozen physical identity and independently reconstructed I4 filter diagnostics."""

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from airline_market.belief import ExactBelief
from airline_market.belief.particle import ParticleBelief
from airline_market.campaign.artifacts import file_hash
from airline_market.campaign.instances import build_instance
from airline_market.core.errors import ZeroLikelihoodError
from airline_market.core.rng import StreamRegistry
from airline_market.core.types import CoefficientAction
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.spec import compile_market, dump_spec

if os.environ.get("I4_STUDY_STAGE"):
    spec = importlib.util.spec_from_file_location(
        "i4_study_stage", os.environ["I4_STUDY_STAGE"]
    )
    study = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(study)
else:
    from airline_market.experiments import i4_particle_study as study


@pytest.fixture
def setup_study(tmp_path):
    compiled = compile_market(
        build_instance(
            "hub_feeder_family",
            {
                "T": 2,
                "actions": 2,
                "n_spokes": 1,
                "lag_periods": 1,
                "capacity_support": [1, 2],
                "connecting_share": 0.7,
            },
        )
    )
    path = tmp_path / "physical.json"
    dump_spec(compiled.spec, path)
    params = {
        "source_spec": str(path),
        "source_spec_sha256": file_hash(path),
        "train_seed_start": 100000,
        "n_train": 2,
        "seed_start": 200320,
        "n_cohorts": 1,
        "particles": 64,
        "focal": "A",
        "arm": "direct",
        "rival": "B",
        "joint_legs": ["B_OH0", "B_HD0"],
        "planning_scenarios": 2,
    }
    ctx = SimpleNamespace(directory=tmp_path, update=lambda counters: None)
    return compiled, params, ctx


def fitted(setup_study):
    c, p, ctx = setup_study
    study.fit_fixed({"parameters": p}, ctx)
    p["tails_file"] = str(ctx.directory / "tails.json")
    p["tails_sha256"] = file_hash(p["tails_file"])
    return c, p, ctx


def test_frozen_source_round_trip_and_changed_bytes_refused(setup_study):
    c, p, _ = setup_study
    assert study.frozen_model(p).spec_hash == c.spec_hash
    Path(p["source_spec"]).write_text(Path(p["source_spec"]).read_text() + "\n")
    with pytest.raises(ValueError, match="physical specification changed"):
        study.frozen_model(p)


@pytest.mark.parametrize("mode", ["training", "validation"])
def test_stream_boundaries_enforced_before_execution(setup_study, mode):
    _, p, ctx = setup_study
    if mode == "training":
        p["train_seed_start"] = 200000
        function = study.fit_fixed
    else:
        p["seed_start"] = 300000
        function = study.compare
    with pytest.raises(ValueError, match="training|validation"):
        function({"parameters": p}, ctx)


def test_full_fixed_grid_fit_matches_independent_physical_rollouts(setup_study):
    c, p, ctx = fitted(setup_study)
    tails = json.loads(Path(p["tails_file"]).read_text())
    expected = []
    for theta in c.spec.pricing[0].theta_grid:
        values = []
        for seed in range(100000, 100002):
            env = FocalCarrierEnv(c, "A", "direct")
            obs, _ = env.reset(seed)
            total = 0
            while obs.t < 2:
                obs, reward, _, _ = env.step(CoefficientAction(theta))
                total += reward
            values.append(total)
        expected.append(float(np.mean(values)))
    assert [r["training_objective"] for r in tails["fixed_training_table"]] == expected
    assert (
        tuple(tails["fixed"]) == c.spec.pricing[0].theta_grid[int(np.argmax(expected))]
    )
    assert tails["spec_hash"] == c.spec_hash


def test_lagged_actual_study_matches_original_reference_filters_and_pre_ess(
    setup_study,
):
    c, p, ctx = fitted(setup_study)
    result = study.compare({"parameters": p}, ctx)
    rows = json.loads((ctx.directory / "posterior_rows.json").read_text())
    assert [r["period"] for r in rows] == [0, 1, 2]
    env = FocalCarrierEnv(c, "A", "direct")
    obs, _ = env.reset(200320)
    exact = ExactBelief(c, "A", "direct")
    particle = ParticleBelief(
        c,
        "A",
        "direct",
        n_particles=64,
        seed=StreamRegistry(773, "plan/F4_I4/filter").child_seed(0),
        conditional_sampling="systematic",
        strict=True,
    )
    exact.reset(obs)
    particle.reset(obs)
    fixed = json.loads(Path(p["tails_file"]).read_text())["fixed"]
    for i, row in enumerate(rows):
        assert row["full_hidden_tv"] == pytest.approx(
            study.total_variation(exact.mass, particle.mass), abs=1e-15
        )
        assert row["post_resampling_ess"] == float(
            1 / (particle.weights @ particle.weights)
        )
        assert row["pre_resampling_ess"] == (
            None if i == 0 else particle.diagnostics[-1]["ess"]
        )
        if i < 2:
            action = CoefficientAction(tuple(fixed))
            after, _, _, _ = env.step(action)
            exact.update(obs, action, after)
            particle.update(obs, action, after)
            obs = after
    assert result["provisional_criteria_pass"] is None
    assert result["observations"] == 3
    assert result["decisions"] == 2
    assert result["rows_sha256"] == file_hash(ctx.directory / "posterior_rows.json")


def test_changed_fit_refused(setup_study):
    _, p, ctx = fitted(setup_study)
    Path(p["tails_file"]).write_text("{}")
    with pytest.raises(ValueError, match="fitted-tail input changed"):
        study.compare({"parameters": p}, ctx)


def test_filter_failure_propagates_and_keeps_completed_posterior_rows(
    setup_study, monkeypatch
):
    _, p, ctx = fitted(setup_study)

    def impossible(*args):
        raise ZeroLikelihoodError("declared impossible observation")

    monkeypatch.setattr(study.TemplateParticleBelief, "update", impossible)
    with pytest.raises(ZeroLikelihoodError, match="declared impossible"):
        study.compare({"parameters": p}, ctx)
    rows = json.loads((ctx.directory / "posterior_rows.json").read_text())
    assert len(rows) == 1 and rows[0]["period"] == 0
