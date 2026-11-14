from collections import defaultdict
from dataclasses import replace
import numpy as np
import pytest
from airline_market.spec import compile_market
from airline_market.spec.library_builders import toy_hub, cx_example
from airline_market.core.types import CoefficientAction
from airline_market.core.errors import ZeroLikelihoodError
from airline_market.sim.engine import MarketEngine
from airline_market.sim.tape import LegacyToyTapeAdapter
from airline_market.belief import ExactBelief, ParticleBelief, factorize, PointBelief
from airline_market.belief.factorized import inventory_key
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.pomdp.game import MarketGame
from airline_market.legacy_toy.model import (
    Config,
    Model,
    AirlineEnv,
    NoiseTape,
    Information,
    COEFFICIENT_GRID,
    RivalState,
)
from airline_market.legacy_toy.belief import Belief


@pytest.mark.parametrize("shock", [0.0, 0.5])
def test_exact_filter_legacy_identity(shock):
    cfg = Config(capacity_shock_probability=shock)
    c = compile_market(toy_hub(cfg))
    random = np.random.default_rng(591)
    for seed in range(12):
        actions = random.integers(18, size=10)
        for arm in Information:
            old = AirlineEnv(Model(cfg))
            old.reset(seed)
            new = MarketEngine(c)
            new.reset(
                tape=LegacyToyTapeAdapter.from_legacy(NoiseTape.from_seed(seed, 10), c)
            )
            before = new.observe("A", arm.value)
            prior = ExactBelief(c, "A", arm.value)
            prior.reset(before)
            ref = Belief(old.model)
            ob = old.observe(arm)
            ref.condition(ob)
            for action in actions:
                theta = COEFFICIENT_GRID[int(action)]
                sale = old.step(theta)
                after_old = old.observe(arm)
                ref.advance(ob, theta, sale, after_old)
                ob = after_old
                new.step({"A": CoefficientAction(theta)})
                after = new.observe("A", arm.value)
                prior.update(before, CoefficientAction(theta), after)
                before = after
                mass = defaultdict(float)
                for h, p in prior.mass.items():
                    mass[RivalState(h.rivals[0].capacity, h.rivals[0].counts)] += p
                assert set(mass) == set(ref.mass)
                assert max(abs(mass[h] - p) for h, p in ref.mass.items()) < 1e-12


def test_env_rewards_game_and_impossible_quote():
    c = compile_market(toy_hub())
    e = FocalCarrierEnv(c, "A", "direct")
    obs, _ = e.reset(7)
    total = 0.0
    b = ExactBelief(c, "A", "direct")
    b.reset(obs)
    with pytest.raises(ZeroLikelihoodError):
        b.reset(
            replace(obs, quotes=tuple(replace(q, fare=123456.0) for q in obs.quotes))
        )
    for _ in range(10):
        obs, r, done, _ = e.step(0)
        total += r
    assert done and total == pytest.approx(e.engine.metrics()["objective"], abs=1e-9)
    g = MarketGame(c)
    assert set(g.reset(7)) == {"B"}
    leader = next(p for p in c.spec.pricing if p.carrier == "B")
    seen = g.post_leaders(
        {"B": CoefficientAction(tuple(0.0 for _ in leader.theta_bounds))}
    )
    assert set(seen) == {"A", "B"}
    with pytest.raises(ValueError):
        g.post_leaders({"B": CoefficientAction(())})
    assert not g.step({"A": CoefficientAction(COEFFICIENT_GRID[0])})[2]


def test_cx_compression_and_particle_point():
    compressed = []
    for label in ("C", "X"):
        c = compile_market(cx_example(label))
        e = MarketEngine(c)
        e.reset(4)
        obs = e.observe("F", "empty")
        b = ExactBelief(c, "F", "empty")
        b.reset(obs)
        f = factorize(b)
        compressed.append({inventory_key(h, c, 0): p for h, p in f.mass.items()})
        assert f.truncated_mass == 0
        for leg, p in b.leg_marginals().items():
            assert np.allclose(p, f.leg_marginals()[leg])
    assert compressed[0] == compressed[1]
    c = compile_market(toy_hub())
    e = FocalCarrierEnv(c, "A", "linked")
    obs, _ = e.reset(3)
    exact = ExactBelief(c, "A", "linked")
    p = ParticleBelief(c, "A", "linked", n_particles=20000, seed=43)
    point = PointBelief(base=exact)
    point.reset(obs)
    p.reset(obs)
    for _ in range(10):
        after, _, _, _ = e.step(0)
        point.update(obs, CoefficientAction(COEFFICIENT_GRID[0]), after)
        p.update(obs, CoefficientAction(COEFFICIENT_GRID[0]), after)
        for leg, m in exact.leg_marginals().items():
            assert np.max(abs(m - p.leg_marginals()[leg])) < 0.02
        assert len(point.mass) == 1
        obs = after


def test_oracle_filter_includes_leader_controller_memory_updates():
    from airline_market.pipeline.generators import hub_feeder_family

    c = compile_market(
        hub_feeder_family(
            {"n_spokes": 1, "periods": 3, "rival_controller": "bid_price_dlp"}
        )
    )
    env = FocalCarrierEnv(c, "A", "oracle")
    obs, _ = env.reset(4)
    b = ExactBelief(c, "A", "oracle")
    b.reset(obs)
    while obs.t < c.spec.horizon.n_periods:
        action = CoefficientAction((0.0, 0.0, 0.0))
        after, _, _, _ = env.step(action)
        b.update(obs, action, after)
        obs = after
        assert sum(b.mass.values()) == pytest.approx(1.0)


def test_unseen_factorized_inventories_keep_latent_nuisance_support():
    from airline_market.spec.model import ExogenousSpec, ExoState

    spec = cx_example("C")
    spec = replace(
        spec,
        exogenous=ExogenousSpec(
            states=(
                ExoState(name="low", public=False),
                ExoState(name="high", public=False),
            ),
            initial=(0.5, 0.5),
            transitions={},
        ),
    )
    c = compile_market(spec)
    e = MarketEngine(c)
    e.reset(8)
    b = ExactBelief(c, "F", "empty")
    b.reset(e.observe("F", "empty"))
    f = factorize(b)
    for state in (0, 1):
        assert sum(
            p for h, p in f.mass.items() if h.exo_latent == state
        ) == pytest.approx(0.5)


def test_infeasible_factorization_reports_mass_and_marginal_distortion():
    from airline_market.core.hashing import jsonable
    from airline_market.spec import load_spec

    data = jsonable(cx_example("C"))
    data["products"] = [p for p in data["products"] if p["id"] not in ("R_L1", "R_L2")]
    data["capacity_priors"] = []
    for leg in data["legs"]:
        if leg["carrier"] == "R":
            leg["capacity"] = {"support": [2], "probabilities": [1.0]}
    c = compile_market(load_spec(data))
    b = ExactBelief(c, "F", "empty")
    _, ks = b.kernel.initial_distribution()[0]
    ks = replace(ks, t=1)
    sold = replace(
        ks.carriers[1], counts=(1,), cash=190.0, booking_order=(0,), periods=((-1,),)
    )
    b.mass = {
        b._hidden(ks): 0.8,
        b._hidden(replace(ks, carriers=(ks.carriers[0], sold))): 0.2,
    }
    f = factorize(b)
    assert f.truncated_mass == pytest.approx(0.32)
    assert f.max_marginal_deviation == pytest.approx(0.2 - 0.04 / 0.68)


def test_reconstructed_rival_counts_use_only_remaining_request_budget():
    from airline_market.belief.factorized import _reconstruct_template

    c = compile_market(toy_hub())
    e = FocalCarrierEnv(c, "A", "linked")
    obs, _ = e.reset(8)
    b = ExactBelief(c, "A", "linked")
    b.reset(obs)
    before = obs
    after, _, _, _ = e.step(0)
    # Install a known focal purchase prefix to isolate reconstruction's budget.
    h = next(iter(b.mass))
    own = replace(
        h.kernel.carriers[0],
        counts=(1, 0),
        fares=((100.0,), ()),
        booking_order=(0,),
        periods=((0,), ()),
    )
    ks = replace(h.kernel, t=1, carriers=(own, h.kernel.carriers[1]))
    template = b._hidden(ks)
    target = tuple(k - 1 for k in template.rivals[0].capacity)
    assert _reconstruct_template(b, target, template) == []
