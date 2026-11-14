from dataclasses import replace
import math
import numpy as np
import pytest
from scipy.stats import chisquare
from airline_market.core.errors import KernelLimitError
from airline_market.core.types import CoefficientAction
from airline_market.core.hashing import jsonable
from airline_market.legacy_toy.model import Model, Config, COEFFICIENT_GRID
from airline_market.spec import compile_market, load_spec
from airline_market.spec.library_builders import toy_hub, cx_example
from airline_market.sim.kernel import ExactKernel
from airline_market.sim.engine import MarketEngine
from airline_market.sim.ledger import inventory


def test_kernel_legacy_choice_vector_30_states_all_actions():
    c = compile_market(toy_hub())
    kernel = ExactKernel(c)
    old = Model()
    engine = MarketEngine(c)
    cases = []
    for seed in range(6):
        engine.reset(seed)
        for t in range(5):
            cases.append(kernel.encode(engine.world, 0))
            engine.step({"A": CoefficientAction(COEFFICIENT_GRID[(seed + t) % 18])})
    for ks in cases:
        e = kernel.engine(ks)
        a_stock = tuple(inventory(e.world.carriers[0], c, 0))
        b_stock = tuple(inventory(e.world.carriers[1], c, 1))
        previous = ks.carriers[0].previous_prices
        for theta in COEFFICIENT_GRID:
            action = CoefficientAction(theta)
            posted = kernel.posted_prices(ks, 0, action)
            observed = np.zeros(6)
            for p, outcome in kernel.period_outcomes(ks, posted):
                j = outcome.choices[0] if outcome.choices else None
                observed[5 if j is None else j] += p
            expected = old.outcomes(
                ks.t, a_stock, b_stock, tuple(posted.values[:2]), previous, bool(ks.exo)
            )
            np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=0)


def test_kernel_gumbel_sampler_agreement_30_pairs():
    # Vectorize only the iid utility draws; the probabilities come from the
    # actual kernel and the sampler's market-local utility/choice primitives.
    from airline_market.sim.choice import systematic_utilities, sampled_choice
    from airline_market.sim.arrivals import bernoulli_probability
    from airline_market.sim.tape import segment_weights
    from airline_market.sim.exogenous import categorical

    c = compile_market(toy_hub())
    k = ExactKernel(c)
    engine = MarketEngine(c)
    rng = np.random.default_rng(8872)
    pairs = []
    for seed in range(6):
        engine.reset(seed)
        for t in range(5):
            pairs.append((k.encode(engine.world), COEFFICIENT_GRID[(seed + t) % 18]))
            engine.step({"A": CoefficientAction(COEFFICIENT_GRID[(seed + t) % 18])})
    for ks, theta in pairs:
        post = k.posted_prices(ks, 0, CoefficientAction(theta))
        expected = np.zeros(6)
        for p, o in k.period_outcomes(ks, post):
            expected[5 if not o.choices or o.choices[0] is None else o.choices[0]] += p
        e = k.engine(ks)
        active = e._sellable_global()
        N = 20000
        draws = rng.random((N, 3))
        gumbels = rng.gumbel(size=(N, 6))
        chosen = np.full(N, 5)
        for m, wm in enumerate(c.spec.demand.market_weights.value_at(ks.t)):
            market = np.array(
                [
                    categorical(u, c.spec.demand.market_weights.value_at(ks.t))
                    for u in draws[:, 1]
                ]
            )
            for g, weight in enumerate(segment_weights(c, ks.t, m)):
                taste = np.array(
                    [categorical(u, segment_weights(c, ks.t, m)) for u in draws[:, 2]]
                )
                indexes = np.flatnonzero(
                    (draws[:, 0] < bernoulli_probability(c, ks.t, ks.exo))
                    & (market == m)
                    & (taste == g)
                )
                if not len(indexes):
                    continue
                columns = (*c.market_products[m], 5)
                utilities = systematic_utilities(
                    c, m, c.spec.segments[g].classes[0], post.values, active
                )
                best = np.argmax(utilities + gumbels[np.ix_(indexes, columns)], axis=1)
                chosen[indexes] = np.array(columns)[best]
        observed = np.bincount(chosen, minlength=6)
        assert np.max(np.abs(observed / N - expected)) <= 0.015
        use = expected > 0
        assert chisquare(observed[use], N * expected[use])[1] > 1e-4


def test_transition_probability_and_reward():
    c = compile_market(cx_example("C"))
    k = ExactKernel(c)
    prior = k.initial_distribution()
    assert sum(p for p, _ in prior) == 1
    for p, ks in prior:
        rows = k.transition(ks, 0, CoefficientAction((0.0,)))
        assert abs(sum(p for p, _, _ in rows) - 1) < 1e-12
        assert all(signal.cash_delta == 150 for _, _, signal in rows)
        assert all(state.carriers[0].counts == (1, 0) for _, state, _ in rows)


def test_poisson_tail_accounting_and_limits():
    data = jsonable(toy_hub())
    data["demand"]["mode"] = "poisson"
    data["demand"]["intensity"] = {m: {"constant": 0.01} for m in ("OH", "HD", "OD")}
    c = compile_market(load_spec(data))
    k = ExactKernel(c, max_requests=2, prob_tol_tail=1e-4)
    ks = k.initial_distribution()[0][1]
    posted = k.posted_prices(ks, 0, CoefficientAction((0.0, 0.0, 0.0)))
    outcomes = k.period_outcomes(ks, posted)
    assert 0 < k.tail_mass < 1e-4
    assert abs(sum(p for p, _ in outcomes) - 1) < 1e-12
    strict = ExactKernel(c, max_requests=1, prob_tol_tail=1e-9)
    with pytest.raises(KernelLimitError):
        strict.period_outcomes(ks, posted)
    small = ExactKernel(compile_market(toy_hub()), max_outcomes=1)
    with pytest.raises(KernelLimitError):
        small.initial_distribution()


def test_binned_noisy_observation_distribution():
    data = jsonable(toy_hub())
    data["quotes"]["requests"][0].update(
        fare_bin=10.0, fare_noise_sd=2.0, missing_probability=0.2
    )
    c = compile_market(load_spec(data))
    kernel = ExactKernel(c)
    ks = kernel.initial_distribution()[0][1]
    arm = next(a for a in c.spec.information if a.name == "direct")
    rows = kernel.observation_distribution(ks, 0, arm)
    assert abs(sum(p for p, _ in rows) - 1) < 1e-12
    assert any(o.quotes[0].missing for _, o in rows)
    for p, obs in rows:
        conditioned = kernel.observation_distribution(ks, 0, arm, obs.quotes)
        assert abs(conditioned[0][0] - p) < 1e-12
