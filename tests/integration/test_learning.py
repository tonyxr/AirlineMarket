from copy import deepcopy
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from airline_market.core.hashing import jsonable
from airline_market.spec import compile_market
from airline_market.spec.library_builders import cx_example
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.learn.adapters import RecurrentEnvAdapter
from airline_market.learn.ppo import train_ppo, load_checkpoint
from airline_market.learn.structured import StructuredActorCritic, pretrain_duals


def test_ppo_determinism_hash_and_critic_routing(tmp_path):
    cfg = {
        "spec": jsonable(cx_example("X")),
        "focal": "F",
        "arm": "empty",
        "seeds": {"train": [0, 2], "validation": [10, 2], "test": [20, 2]},
        "updates": 2,
        "episodes_per_update": 2,
        "critic_privileged": True,
        "ppo": {"hidden_dim": 8, "update_epochs": 1},
    }
    results = []
    for run in ["a", "b"]:
        cfg["out"] = str(tmp_path / run)
        results.append(train_ppo(cfg))
    a = torch.load(results[0].checkpoint, weights_only=False)
    b = torch.load(results[1].checkpoint, weights_only=False)
    assert all(
        torch.equal(v, b[network][k])
        for network in ("actor", "value")
        for k, v in a[network].items()
    )
    c = compile_market(cx_example("X"))
    adapter = RecurrentEnvAdapter(
        FocalCarrierEnv(c, "F", "empty", critic_privileged=True)
    )
    agent = load_checkpoint(results[0].checkpoint, adapter)
    x = adapter.reset(3)
    y = x.copy()
    y[adapter.public_dim :] += 100
    out1 = agent.act(x, deterministic=True)
    out2 = agent.act(y, deterministic=True)
    assert np.array_equal(out1[0], out2[0])
    assert out1[2]["value"] != out2[2]["value"]
    wrong = RecurrentEnvAdapter(
        FocalCarrierEnv(
            compile_market(cx_example("C")), "F", "empty", critic_privileged=True
        )
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_checkpoint(results[0].checkpoint, wrong)
    transform = adapter.action_transform
    z = np.array([1.2])
    assert transform.physical_to_coordinates(
        transform.coordinates_to_physical(z)
    ) == pytest.approx(z)


def test_structured_incidence_gradients_duals_and_permutation():
    torch.manual_seed(9)
    torch.set_num_threads(1)
    m = StructuredActorCritic(2, 1, 1, hidden=16)
    legs = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    g = torch.zeros(1, 1)
    target = torch.tensor([[100.0, 0.0]])
    loss = pretrain_duals(m, legs, g, target, steps=1200, lr=0.02)
    assert loss < 1.0
    pi, theta, v = m(legs, g)
    permuted = m(legs.flip(1), g)
    assert torch.allclose(pi.flip(1), permuted[0], atol=1e-6)
    pi.retain_grad()
    prices = m.price_tensor(
        torch.tensor([10.0, 20.0]),
        torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
        pi,
        torch.ones(1, 2, 1),
        theta,
        torch.tensor([0.0, 0.0]),
        torch.tensor([300.0, 300.0]),
    )
    prices.sum().backward()
    assert pi.grad.abs().sum() > 0 and (prices <= 300).all()


def test_counterfactual_training_tape_branching():
    from airline_market.learn.credit import counterfactual_credit
    from airline_market.core.types import CoefficientAction

    c = compile_market(cx_example("X"))
    env = FocalCarrierEnv(c, "F", "empty")
    env.reset(3)
    before = env.snapshot()
    sell = CoefficientAction((0.0,))
    close = CoefficientAction((9850.0,))
    result = counterfactual_credit(
        env, [sell, close], lambda: lambda obs, previous, reward, rng: sell
    )
    assert result["returns"] == pytest.approx([150.0, 200.0], abs=1e-9)
    assert (
        result["hold_relative"][1] == 0
        and result["returns"][1] - result["returns"][0] > 0
    )
    assert (
        env.engine.world.t == before.world.t
        and env.engine.world.carriers[0].cash == before.world.carriers[0].cash
    )


def test_counterfactual_credit_is_training_only(tmp_path, monkeypatch):
    import airline_market.learn.credit as credit

    original = credit.counterfactual_credit
    seen = []

    def record(env, *args, **kwargs):
        seen.append(env.engine.tape.seed)
        return original(env, *args, **kwargs)

    monkeypatch.setattr(credit, "counterfactual_credit", record)
    result = train_ppo(
        {
            "spec": jsonable(cx_example("X")),
            "focal": "F",
            "arm": "empty",
            "seeds": {"train": [0, 2], "validation": [10, 2], "test": [20, 2]},
            "updates": 1,
            "episodes_per_update": 2,
            "credit": "mix",
            "credit_params": {"fraction": 1.0, "K": 2, "H": 2},
            "ppo": {"hidden_dim": 8, "update_epochs": 1},
            "out": str(tmp_path / "cf"),
        }
    )
    assert seen and set(seen) <= set([0, 1])


def test_collect_refuses_zero_filled_belief_actor_credit():
    from types import SimpleNamespace
    from airline_market.learn.ppo import collect

    with pytest.raises(ValueError, match="belief-feature actors"):
        collect(
            SimpleNamespace(belief_features=True), None, 0, credit={"fraction": 1.0}
        )
