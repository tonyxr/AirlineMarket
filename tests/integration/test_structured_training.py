"""Real structured PPO updates, information scope and checkpoint continuation."""

import numpy as np
import pytest
from types import SimpleNamespace

torch = pytest.importorskip("torch")

from airline_market.campaign.instances import build_instance
from airline_market.core.rng import StreamRegistry
from airline_market.core.hashing import canonical_json
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.spec import compile_market
from airline_market.vendor.ride_response.recurrent.ppo import RecurrentPPOConfig
from airline_market.learn.structured_runtime import StructuredAdapter, StructuredNetwork, StructuredValue, StructuredPPOAgent
from airline_market.learn.structured_train import collect_structured, branch_credit, train_structured, load_structured


def adapter():
    compiled = compile_market(build_instance("toy_hub", {"T": 3, "actions": 4}))
    return StructuredAdapter(FocalCarrierEnv(compiled, "A", "direct"))


def agent_for(a, auxiliary=False):
    torch.set_num_threads(1)
    config = RecurrentPPOConfig(hidden_dim=8, update_epochs=2, minibatch_episodes=2,
                                gamma=1.0, seed=11,
                                observation_identity_indices=tuple(range(a.layout.observation_dim)))
    with torch.random.fork_rng(devices=[]):
        return StructuredPPOAgent(a, config, auxiliary=auxiliary)


def scientific_state(env):
    """Compare actual snapshot contents rather than cloned object addresses."""
    world = env.engine.world
    fields = dict(vars(world))
    fields["events_fired"] = sorted(world.events_fired)
    fields["posted_leaders"] = sorted(world.posted_leaders)
    fields["quote_history"] = {"initial": world.quote_history.initial,
                                "records": world.quote_history.records}
    return canonical_json({"world": fields, "trace": env.engine.trace()})


def test_structured_price_map_adds_shared_leg_costs_before_clipping():
    a = adapter()
    a.reset(100000)
    pi = np.arange(1, a.layout.n_legs + 1, dtype=float)
    theta = np.zeros(a.layout.n_coefficients)
    action = np.r_[np.log(np.expm1(pi)), theta]
    prices = a.price_action(action, a.obs).prices
    for local, j in enumerate(a.products):
        p = a.c.spec.products[j]
        base = a.env.pricing.base.get(p.id, p.reference_fare)
        base += sum(w * a.env.price_rule.evaluator(f, j, a.obs)
                    for f, w in a.env.pricing.fixed_features)
        expected = base + pi @ a.c.A[a.env.i][:, local]
        assert prices[j] == pytest.approx(round(min(p.fare_cap, max(p.fare_floor, expected)), 2))


def test_leg_reordering_permutes_costs_and_preserves_global_actor_and_value():
    a = adapter()
    a.reset(100000)
    agent = agent_for(a)
    x = torch.tensor(a.encode(a.obs))[None, None]
    context = agent._context(x, torch.zeros(1, 1, agent.action_dim),
                             torch.zeros(1, 1, 1), torch.zeros(1, 1, 1))
    layout = a.layout
    permutation = torch.arange(layout.n_legs - 1, -1, -1)
    swapped = context.clone()
    rows = swapped[..., layout.global_dim:layout.observation_dim].reshape(1, 1, layout.n_legs, -1)
    rows.copy_(rows[..., permutation, :].clone())
    original = agent.actor.distribution(context)[0].mean
    changed = agent.actor.distribution(swapped)[0].mean
    torch.testing.assert_close(changed[..., :layout.n_legs], original[..., permutation])
    torch.testing.assert_close(changed[..., layout.n_legs:], original[..., layout.n_legs:])
    torch.testing.assert_close(agent.value(context)[0], agent.value(swapped)[0])


@pytest.mark.parametrize("variant", ["srl", "srl_c", "srl_ca"])
def test_each_variant_executes_real_updates_with_correct_auxiliary_and_credit(variant):
    a = adapter()
    agent = agent_for(a, auxiliary=variant == "srl_ca")
    before = {k: v.clone() for k, v in agent.actor.state_dict().items()}
    for seed in range(100000, 100004):
        episode, logp, _ = collect_structured(
            a, agent, seed, training=True,
            credit={"fraction": 1, "K": 2, "H": 1, "mix": 0.5} if variant != "srl" else None,
            event_legs=("B_OH", "B_HD") if variant == "srl_ca" else ())
        assert ("branch_returns" in episode.metadata) == (variant != "srl")
        assert ("joint_event_labels" in episode.metadata) == (variant == "srl_ca")
        agent.add_episode(episode, actions_are_physical=False, behavior_log_probabilities=logp)
    stats = agent.update()
    assert stats["episodes"] == 4
    assert agent.behavior_log_probability_checks == 12
    assert agent.max_behavior_log_probability_error < 1e-4
    assert any(not torch.equal(before[k], v) for k, v in agent.actor.state_dict().items())
    assert (stats["aux_loss"] > 0) == (variant == "srl_ca")
    assert not torch.equal(before["structure.dual_head.weight"],
                           agent.actor.state_dict()["structure.dual_head.weight"])


def test_credit_branches_preserve_parent_and_torch_stream_and_identical_hold_returns():
    a = adapter()
    agent = agent_for(a)
    x = a.reset(100000)
    action, hidden, info = agent.act(x, deterministic=True)
    before = scientific_state(a.env)
    rng = torch.get_rng_state().clone()
    result = branch_credit(a, agent, action, hidden, action, seed=773, K=2,
                           horizon=3, coordinates=info["policy_coordinates"],
                           previous_coordinates=info["policy_coordinates"])
    assert result["returns"][0] == result["returns"][1]
    assert result["centered"] == result["hold_relative"] == 0
    assert scientific_state(a.env) == before
    assert torch.equal(rng, torch.get_rng_state())


def test_public_actor_is_invariant_to_hidden_world_changes_and_rejects_evaluation_labels():
    a = adapter()
    agent = agent_for(a)
    x = a.reset(200000)
    expected = agent.act(x, deterministic=True)[0]
    a.env.engine.world.carriers[1].capacity[:] = 99
    np.testing.assert_array_equal(agent.act(a.encode(a.obs), deterministic=True)[0], expected)
    with pytest.raises(ValueError, match="training labels"):
        collect_structured(a, agent, 200000, training=False, event_legs=("B_HD",))
    with pytest.raises(ValueError, match="registered split"):
        collect_structured(a, agent, 300000, training=False)


def test_checkpoint_continuation_matches_uninterrupted_training(tmp_path):
    a = adapter()
    config = {"learner": "srl_ca", "n_train": 4, "n_validation": 2,
              "network_seed": 0, "cohorts_per_update": 2,
              "event_legs": ["B_OH", "B_HD"],
              "credit": {"fraction": 1, "K": 2, "H": 1, "mix": 0.5},
              "ppo": {"hidden_dim": 8, "update_epochs": 1, "minibatch_episodes": 2}}
    def interrupt(row):
        if row["cohorts_completed"] == 2:
            raise RuntimeError("controlled interruption")
    with pytest.raises(RuntimeError, match="controlled interruption"):
        train_structured(a.c, config, tmp_path / "resumed", interrupt)
    result = train_structured(a.c, dict(config, resume_checkpoint=str(tmp_path / "resumed/last.pt")),
                              tmp_path / "resumed")
    train_structured(a.c, config, tmp_path / "whole")
    resumed, _ = load_structured(tmp_path / "resumed/last.pt", a)
    whole, _ = load_structured(tmp_path / "whole/last.pt", a)
    for name in ("actor", "value"):
        for key, value in getattr(whole, name).state_dict().items():
            assert torch.equal(value, getattr(resumed, name).state_dict()[key])
    assert result["test_seeds_opened"] is False
    assert result["cohorts"] == 4


@pytest.mark.parametrize("learner", ["ppo", "srl_ca"])
def test_f6_study_executes_distinct_training_paths_and_excludes_warmup(tmp_path, learner):
    from airline_market.experiments.f6_study import run

    cell = {"parameters": {"learner": learner, "T": 3, "stage": "throughput",
                            "n_train": 4, "cohorts_per_update": 2, "n_validation": 2,
                            "event_legs": ["B_OH", "B_HD"],
                            "credit": {"fraction": 1, "K": 2, "H": 1, "mix": 0.5},
                            "ppo": {"hidden_dim": 8, "update_epochs": 1,
                                    "minibatch_episodes": 2}}}
    result = run(cell, SimpleNamespace(directory=tmp_path))
    assert result["status"] == "completed"
    assert result["warmup_updates"] == result["measured_updates"] == 1
    assert result["test_seeds_opened"] is False
    assert result["median_seconds_per_cohort"] > 0
    assert result["policy_class"] == ("coefficient_recurrent" if learner == "ppo"
                                        else "network_opportunity_cost_recurrent")
