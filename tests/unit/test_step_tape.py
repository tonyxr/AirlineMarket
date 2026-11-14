"""Active tape-value, physical-path and strict proposal/RNG equivalence."""

import ast
import importlib.util
import inspect
import os
import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from airline_market.belief.particle import ParticleBelief
from airline_market.belief.template_particle import TemplateParticleBelief
from airline_market.campaign.instances import build_instance
from airline_market.core.errors import ZeroLikelihoodError
from airline_market.core.hashing import canonical_json
from airline_market.core.types import CoefficientAction
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.sim.tape import ExogenousTape
from airline_market.spec import compile_market, load_spec

if os.environ.get("STEP_TAPE_STAGE"):
    stage = Path(os.environ["STEP_TAPE_STAGE"])
    for name in [
        "airline_market.sim.step_tape",
        "airline_market.belief.step_particle",
        "airline_market.experiments.step_particle_study",
    ]:
        spec = importlib.util.spec_from_file_location(
            name, stage / (name.split(".")[-1] + ".py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
from airline_market.belief.step_particle import StepTapeParticleBelief
from airline_market.sim.step_tape import OneStepTape


def compiled(instance):
    if instance in ["I5", "I6"]:
        return compile_market(load_spec("config/instances/" + instance + ".json"))
    return compile_market(build_instance("hub_feeder_small", {"T": 3, "actions": 2}))


@pytest.mark.parametrize(
    "instance,period",
    [
        ("small", 0),
        ("small", 1),
        ("small", 2),
        ("I5", 0),
        ("I5", 10),
        ("I5", 19),
        ("I6", 0),
        ("I6", 29),
    ],
)
def test_every_active_value_and_global_shape_equals_original(instance, period):
    c = compiled(instance)
    old = ExogenousTape.generate(c, 773)
    new = OneStepTape.generate(c, 773, period)
    assert old.gumbel.shape == new.gumbel.shape
    np.testing.assert_array_equal(old.gumbel[period], new.gumbel[period])
    assert old.arrival_marks[period] == new.arrival_marks[period]
    for field, value in vars(old).items():
        if field in ["gumbel", "arrival_marks"]:
            continue
        other = getattr(new, field)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, other)
            assert not other.flags.writeable
        else:
            assert value == other
    assert not new.gumbel[period].flags.writeable
    with pytest.raises(IndexError):
        new.gumbel[(period + 1) % c.spec.horizon.n_periods]
    with pytest.raises(IndexError):
        new.arrival_marks[(period + 1) % c.spec.horizon.n_periods]


@pytest.mark.parametrize("instance", ["small", "I6"])
def test_same_hypothesis_step_observation_and_snapshot_replay(instance):
    c = compiled(instance)
    env = FocalCarrierEnv(c, "A", "direct")
    obs, _ = env.reset(200384)
    b = TemplateParticleBelief(c, "A", "direct", n_particles=2, seed=773)
    b.reset(obs)
    h = b.particles[0]
    a = CoefficientAction(c.spec.pricing[0].theta_grid[0])
    worlds = []
    for tape in [ExogenousTape.generate(c, 123), OneStepTape.generate(c, 123, 0)]:
        e = b.kernel.engine(h.kernel, 0)
        e.tape = tape
        snap = e.snapshot()
        e.step({0: a})
        first = (b.kernel.encode(e.world, 0), canonical_json(e.observe(0, b.arm)))
        e.restore(snap)
        e.step({0: a})
        assert first == (
            b.kernel.encode(e.world, 0),
            canonical_json(e.observe(0, b.arm)),
        )
        worlds.append(first)
    assert worlds[0] == worlds[1]


@pytest.mark.parametrize("simulation", [False, True])
def test_actual_update_preserves_particle_order_weights_diagnostics_rng_or_failure(
    simulation,
):
    c = compiled("small")
    env = FocalCarrierEnv(c, "A", "linked")
    before, _ = env.reset(200384)
    old = TemplateParticleBelief(
        c, "A", "linked", n_particles=16, seed=773, inner_draws=4
    )
    new = StepTapeParticleBelief(
        c, "A", "linked", n_particles=16, seed=773, inner_draws=4
    )
    for b in [old, new]:
        b.reset(before)
        if simulation:
            b.exact_proposals = False
    for t in range(3):
        action = CoefficientAction(c.spec.pricing[0].theta_grid[t % 2])
        after, _, _, _ = env.step(action)
        errors = []
        for b in [old, new]:
            try:
                b.update(before, action, after)
                errors.append(None)
            except ZeroLikelihoodError as error:
                errors.append(str(error))
        assert errors[0] == errors[1]
        assert old.particles == new.particles and old.mass == new.mass
        np.testing.assert_array_equal(old.weights, new.weights)
        assert old.diagnostics == new.diagnostics
        assert canonical_json(old.rng.bit_generator.state) == canonical_json(
            new.rng.bit_generator.state
        )
        if errors[0]:
            break
        before = after


def test_impossible_update_is_not_given_positive_mass():
    c = compiled("small")
    env = FocalCarrierEnv(c, "A", "direct")
    before, _ = env.reset(200384)
    action = CoefficientAction(c.spec.pricing[0].theta_grid[0])
    after, _, _, _ = env.step(action)
    impossible = replace(after, own_cash=after.own_cash + 1e9)
    errors = []
    for cls in [TemplateParticleBelief, StepTapeParticleBelief]:
        b = cls(c, "A", "direct", n_particles=8, seed=773, inner_draws=2)
        b.reset(before)
        b.exact_proposals = False
        with pytest.raises(ZeroLikelihoodError):
            b.update(before, action, impossible)
        errors.append(canonical_json(b.rng.bit_generator.state))
    assert errors[0] == errors[1]


def test_copied_conditional_method_has_only_declared_tape_and_same_exact_target_changes():
    original = ast.parse(
        textwrap.dedent(inspect.getsource(ParticleBelief.conditional_successors))
    )
    revised = ast.parse(
        textwrap.dedent(
            inspect.getsource(StepTapeParticleBelief.conditional_successors)
        )
    )

    class Canonical(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            # Import order is formatting, not the copied numerical body.
            return None

        def visit_Name(self, node):
            if node.id == "OneStepTape":
                node.id = "ExogenousTape"
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "ExogenousTape"
            ):
                node.keywords = [k for k in node.keywords if k.arg != "period"]
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "ExactBelief"
                and node.func.attr == "conditional_successors"
            ):
                node.func.value = ast.Call(
                    func=ast.Name(id="super", ctx=ast.Load()), args=[], keywords=[]
                )
                node.args = node.args[1:]
            return node

    assert ast.dump(Canonical().visit(original), include_attributes=False) == ast.dump(
        Canonical().visit(revised), include_attributes=False
    )


def test_nonzero_simulated_successor_masses_and_complete_kernel_states_match():
    c = compile_market(
        build_instance(
            "hub_feeder_family",
            {
                "T": 2,
                "actions": 2,
                "n_spokes": 1,
                "arrival_probability": 0.0,
            },
        )
    )
    env = FocalCarrierEnv(c, "A", "linked")
    before, _ = env.reset(200385)
    action = CoefficientAction(c.spec.pricing[0].theta_grid[0])
    after, _, _, _ = env.step(action)
    beliefs = [
        cls(c, "A", "linked", n_particles=4, seed=773, inner_draws=4)
        for cls in [TemplateParticleBelief, StepTapeParticleBelief]
    ]
    results = []
    for belief in beliefs:
        belief.reset(before)
        belief.exact_proposals = False
        results.append(
            belief.conditional_successors(belief.particles[0], before, action, after)
        )
    assert results[0] and results[0] == results[1]
    by_key = [{repr(h): canonical_json(h.kernel) for h in result} for result in results]
    assert by_key[0] == by_key[1]
    assert canonical_json(beliefs[0].rng.bit_generator.state) == canonical_json(
        beliefs[1].rng.bit_generator.state
    )


def test_actual_paired_study_runs_and_retains_both_cases(tmp_path):
    from types import SimpleNamespace

    from airline_market.campaign.artifacts import file_hash
    from airline_market.experiments import step_particle_study
    from airline_market.spec import dump_spec

    c = compiled("small")
    path = tmp_path / "spec.json"
    dump_spec(c.spec, path)
    cell = {
        "parameters": {
            "source_spec": str(path),
            "source_spec_sha256": file_hash(path),
            "validation_seed": 200386,
            "particles": 4,
            "inner_draws": 2,
            "execution_order": ["original", "step_tape"],
        }
    }
    context = SimpleNamespace(directory=tmp_path, update=lambda _: None)
    result = step_particle_study.compare(cell, context)
    assert result["prior_identity"] and result["proposal_full_state_mass_rng_identity"]
    assert set(result["cases"]) == {"original", "step_tape"}
    assert result["update_throughput"] is None and result["N_selected"] is None
    assert result["original_stored_array_bytes"] > result["step_stored_array_bytes"]
    assert (tmp_path / "measurements.json").exists()
