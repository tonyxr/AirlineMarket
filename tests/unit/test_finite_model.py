"""Independent sampler-kernel checks for the sufficient-state exact model."""

from dataclasses import replace
from collections import defaultdict
import numpy as np
import pytest
from airline_market.spec import compile_market
from airline_market.spec.library_builders import toy_hub, cx_example
from airline_market.sim.engine import MarketEngine
from airline_market.sim.kernel import ExactKernel
from airline_market.sim.finite import FinitePricingModel
from airline_market.core.types import CoefficientAction
from airline_market.solvers import ExactBeliefDP
from airline_market.solvers.exact_dp import DPNode
from airline_market.pomdp.env import FocalCarrierEnv
from airline_market.belief import ExactBelief


def test_compact_probabilities_and_transitions_equal_reference():
    c = compile_market(toy_hub())
    ref = ExactKernel(c)
    model = FinitePricingModel(c, "A", "linked")
    engine = MarketEngine(c)
    for seed in range(6):
        engine.reset(seed)
        for t in range(4):
            obs = engine.observe("A", "linked")
            ks = ref.encode(engine.world)
            hidden = model.hidden(ks)
            theta = c.spec.pricing[0].theta_grid[(seed * 3 + t) % 18]
            prices = model.prices(
                theta,
                t,
                obs.own_capacity,
                obs.own_inventory,
                model.observed_signal(obs),
            )
            expected = np.zeros(len(c.spec.products) + 1)
            for p, outcome in ref.period_outcomes(
                ks, ref.posted_prices(ks, 0, CoefficientAction(theta), obs)
            ):
                j = outcome.choices[0] if outcome.choices else None
                expected[-1 if j is None else j] += p
            np.testing.assert_allclose(
                model.outcomes(
                    t, obs.own_inventory, hidden[0], prices, obs.own_previous_prices
                ),
                expected,
                atol=1e-12,
                rtol=0,
            )
            fast = defaultdict(float)
            slow = defaultdict(float)
            for p, own, journeys, h, reward, sale in model.step(
                t, obs.own_inventory, 0, hidden, prices, obs.own_previous_prices
            ):
                fast[(own, h[0], sale)] += p
            for p, next_state, signal in ref.transition(
                ks, 0, CoefficientAction(theta), arm=model.arm, observation=obs
            ):
                sale = (
                    c.products_of[0].index(signal.purchases[0][0])
                    if signal.purchases
                    else -1
                )
                slow[
                    (
                        signal.observation.own_inventory,
                        model.hidden(next_state)[0],
                        sale,
                    )
                ] += p
            assert fast.keys() == slow.keys()
            assert max(abs(fast[k] - slow[k]) for k in fast) < 1e-12
            engine.step({"A": CoefficientAction(theta)})


@pytest.mark.parametrize("label", ["C", "X"])
def test_compact_bellman_equals_reference(label):
    c = compile_market(cx_example(label))
    a = ExactBeliefDP(c, "F", "empty", backend="compact").solve()
    b = ExactBeliefDP(c, "F", "empty", backend="reference").solve()
    assert a.value == pytest.approx(b.value, abs=1e-12)


def test_decision_key_ignores_sunk_cash_and_booking_dates():
    c = compile_market(cx_example("C"))
    env = FocalCarrierEnv(c, "F", "empty")
    obs, _ = env.reset(3)
    belief = ExactBelief(c, "F", "empty")
    belief.reset(obs)
    after, _, _, _ = env.step(0)
    belief.update(obs, CoefficientAction((0.0,)), after)
    altered = replace(
        after,
        own_cash=after.own_cash + 123.0,
        own_bookings=tuple(replace(b, period=99) for b in after.own_bookings),
    )
    changed = ExactBelief(c, "F", "empty")
    changed.mass = {}
    for h, p in belief.mass.items():
        carriers = list(h.kernel.carriers)
        carriers[0] = replace(
            carriers[0],
            cash=carriers[0].cash + 123.0,
            periods=tuple((99,) * n for n in carriers[0].counts),
        )
        changed.mass[changed._hidden(replace(h.kernel, carriers=tuple(carriers)))] = p
    for backend in ["reference", "compact"]:
        dp = ExactBeliefDP(c, "F", "empty", backend=backend)
        if backend == "reference":
            assert dp.key(DPNode(after, belief)) == dp.key(DPNode(altered, changed))
        np.testing.assert_allclose(
            dp.q_values(DPNode(after, belief)),
            dp.q_values(DPNode(altered, changed)),
            atol=1e-12,
            rtol=0,
        )


def test_exact_tail_matches_general_policy_evaluation():
    from airline_market.legacy_toy.model import Config
    from airline_market.solvers.finite_dp import ExactTailEvaluator

    c = compile_market(toy_hub(Config(horizon=3, shock_step=1)))
    grid = [c.spec.pricing[0].theta_grid[i] for i in (0, 8, 17)]
    tail = grid[-1]
    dp = ExactBeliefDP(c, "A", "direct", theta_grid=grid, backend="reference")
    root = dp.roots()[0][1]
    evaluator = ExactTailEvaluator(FinitePricingModel(c, "A", "direct"), tail)
    q = evaluator.q_values(root.observation, root.belief, grid)
    for a in range(len(grid)):
        value = dp.evaluate_policy(lambda obs, b: a if obs.t == 0 else 2)
        assert q[a] == pytest.approx(value, abs=1e-12)


def test_monopoly_and_rounded_price_keys_are_value_equivalent():
    c = compile_market(cx_example("C"))
    m = FinitePricingModel(c, "F", "empty")
    from airline_market.solvers.finite_dp import FiniteBeliefSolver

    solver = FiniteBeliefSolver(m, c.spec.pricing[0].theta_grid)
    # Equivalent prefixes have the same observed current offers and no refunds.
    # Exercise sufficient-statistic equivalence against two separate caches.
    e = FocalCarrierEnv(c, "F", "empty")
    obs, _ = e.reset(1)
    b = ExactBelief(c, "F", "empty")
    b.reset(obs)
    n = m.project(obs, b)
    other = (*n[:4], tuple(p + 0.01 for p in n[4]), *n[5:])
    assert solver.key(n) == solver.key(other)
    a = FiniteBeliefSolver(m, c.spec.pricing[0].theta_grid).q_values(n)
    z = FiniteBeliefSolver(m, c.spec.pricing[0].theta_grid).q_values(other)
    np.testing.assert_allclose(a, z, atol=1e-12, rtol=0)


def test_fast_full_kernel_matches_reference_states_and_signals():
    from airline_market.sim.fast_kernel import FastExactKernel

    c = compile_market(toy_hub())
    ref = ExactKernel(c)
    fast = FastExactKernel(c, "A", "unrelated")
    e = MarketEngine(c)
    for seed in range(3):
        e.reset(seed)
        for t in range(4):
            obs = e.observe("A", "unrelated")
            state = ref.encode(e.world)
            action = CoefficientAction(c.spec.pricing[0].theta_grid[(seed + t) % 18])
            rows = []
            for kernel in [ref, fast]:
                mass = defaultdict(float)
                for p, ks, signal in kernel.transition(
                    state, 0, action, arm=fast.model.arm, observation=obs
                ):
                    mass[(ks, signal.key())] += p
                rows.append(mass)
            assert rows[0].keys() == rows[1].keys()
            assert max(abs(rows[0][key] - rows[1][key]) for key in rows[0]) < 1e-12
            e.step({"A": action})


def test_cross_route_quote_target_survives_last_period_key():
    from airline_market.spec.model import FeatureSpec
    from airline_market.solvers.finite_dp import FiniteBeliefSolver

    spec = toy_hub()
    feature = FeatureSpec(
        name="direct_rival_gap",
        params={"targets": {"A_OH": "B_OD", "A_OD": "B_OD"}, "missing_value": -30.0},
    )
    pricing = replace(spec.pricing[0], fixed_features=((feature, 1.0),))
    c = compile_market(replace(spec, pricing=(pricing, *spec.pricing[1:])))
    model = FinitePricingModel(c, "A", "linked")
    assert model.cross_market_features
    t = model.T - 1
    capacity = (2, 2)
    own = (2, 0)
    previous = (110.0, 185.0)
    nodes = []
    for stock in [(2, 1), (2, 0)]:
        hidden = (stock, 0)
        nodes.append(
            (
                t,
                capacity,
                own,
                0,
                previous,
                model.signal(t, hidden, previous),
                ((hidden, 1.0),),
            )
        )
    solver = FiniteBeliefSolver(model, ((0.0, 0.0, 0.0),))
    assert solver.key(nodes[0]) != solver.key(nodes[1])
    first = solver.q_values(nodes[0])
    second = solver.q_values(nodes[1])
    assert not np.allclose(first, second, atol=1e-12, rtol=0)
