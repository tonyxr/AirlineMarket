import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from airline_market.legacy_toy.belief import Belief
from airline_market.legacy_toy.experiment import (
    paired_statistics,
    run_experiment,
    save_results,
)
from airline_market.legacy_toy.model import (
    AirlineEnv,
    Booking,
    COEFFICIENT_GRID,
    Config,
    Information,
    Model,
    NoiseTape,
    Observation,
    RivalState,
    WorldState,
    available,
    inventory,
)
from airline_market.legacy_toy.policies import RolloutPlanner


class AccountingTests(unittest.TestCase):
    def test_connection_uses_both_legs_and_shared_local_stock(self):
        stock = inventory(0, (2, 2), [Booking(0, 100), Booking(1, 180)])
        self.assertEqual(stock, (0, 1))
        self.assertFalse(available(0, 0, stock))
        self.assertFalse(available(0, 1, stock))
        self.assertEqual(inventory(0, (2, 2), [Booking(1, 180)]), (1, 1))

    def test_capacity_loss_refunds_original_fare_and_charges_extra_cost(self):
        model = Model()
        state = WorldState(
            1, [(1, 2), (2, 3)], [[Booking(0, 100), Booking(1, 180)], []], [280, 0]
        )
        model.resolve(state, 0)
        self.assertEqual(state.bookings[0], [Booking(0, 100)])
        self.assertEqual(state.refunds[0], 180)
        self.assertEqual(
            state.cash[0], 75
        )  # 100 retained revenue minus 25 disruption cost.
        self.assertEqual(model.metrics(state)["profit"], 60)
        self.assertEqual(inventory(0, state.capacity[0], state.bookings[0]), (0, 2))

    def test_journey_share_does_not_count_connection_twice_or_include_feeder(self):
        state = WorldState(
            0,
            [(2, 2), (2, 3)],
            [[Booking(1, 180)], [Booking(1, 85), Booking(2, 170)]],
            [180, 255],
        )
        metrics = Model().metrics(state)
        self.assertEqual(metrics["journeys"], 1)
        self.assertEqual(metrics["rival_segment_journeys"], 1)
        self.assertEqual(metrics["share"], 0.5)

    def test_hidden_count_transition_matches_ledger_resolution(self):
        model = Model()
        for hd in (1, 3):
            for oh_sold in range(3):
                for hd_sold in range(hd + 1):
                    for od_sold in range(3):
                        rival = RivalState((2, hd), (oh_sold, hd_sold, od_sold))
                        if min(rival.inventory) < 0:
                            continue
                        ledger = [
                            Booking(j, 100 + j)
                            for j, n in enumerate(rival.sold)
                            for _ in range(n)
                        ]
                        state = WorldState(
                            0,
                            [(2, 2), (2, max(0, hd - 1))],
                            [[], ledger],
                            [0, sum(b.fare for b in ledger)],
                        )
                        model.resolve(state, 1)
                        self.assertEqual(state.rival(), rival.capacity_loss())

    def test_conservation_and_cash_across_stochastic_disruptions(self):
        model = Model(Config(capacity_shock_probability=1.0))
        for seed in range(40):
            env = AirlineEnv(model)
            env.reset(seed)
            while env.state.time < model.config.horizon:
                env.step(
                    COEFFICIENT_GRID[(seed + env.state.time) % len(COEFFICIENT_GRID)]
                )
                for i in (0, 1):
                    self.assertGreaterEqual(
                        min(inventory(i, env.state.capacity[i], env.state.bookings[i])),
                        0,
                    )
                    self.assertAlmostEqual(
                        env.state.cash[i],
                        sum(b.fare for b in env.state.bookings[i])
                        - model.config.disruption_penalty * env.state.canceled[i],
                    )


class ChoiceAndReplayTests(unittest.TestCase):
    def test_heterogeneous_choice_normalization_and_own_price_response(self):
        model = Model()
        p0 = model.probabilities(
            model.utilities("OH", 0, (2, 2), (2, 3), (110, 185), (125, 85, 170))
        )
        p1 = model.probabilities(
            model.utilities("OH", 1, (2, 2), (2, 3), (110, 185), (125, 85, 170))
        )
        dear = model.probabilities(
            model.utilities("OH", 0, (2, 2), (2, 3), (140, 185), (125, 85, 170))
        )
        self.assertAlmostEqual(p0.sum(), 1)
        self.assertAlmostEqual(p1.sum(), 1)
        self.assertLess(dear[0], p0[0])
        self.assertFalse(np.allclose(p0, p1))

    def test_generative_draws_match_filter_transition_probabilities(self):
        model = Model()
        rival = RivalState((2, 1))
        start = WorldState(2, [(2, 2), rival.capacity])
        quotes = model.rival_quotes(rival, start.previous_own_prices)
        theta = (0, 0, 15)
        prices = model.prices(theta, 2, (2, 2), quotes)
        expected = model.outcomes(
            2, (2, 2), (2, 1), prices, start.previous_own_prices, False
        )
        # Many independent requests on fresh states, using indexed shared-randomness tapes.
        rng = np.random.default_rng(887)
        counts = np.zeros(6)
        samples = 10000
        for _ in range(samples):
            requests = rng.random((model.config.horizon, 3))
            tape = NoiseTape(
                requests,
                rng.gumbel(size=(model.config.horizon, 6)),
                rng.random(model.config.horizon),
                1.0,
                0.0,
                3,
            )
            state = start.clone()
            own_sale = model.step(state, theta, tape)
            selected = 5
            if own_sale is not None:
                selected = own_sale
            elif state.bookings[1]:
                selected = 2 + state.bookings[1][0].product
            counts[selected] += 1
        np.testing.assert_allclose(counts / samples, expected, atol=0.015, rtol=0)

    def test_saved_state_replays_identically_and_is_isolated(self):
        env = AirlineEnv()
        env.reset(91)
        saved = env.snapshot()
        for _ in range(3):
            env.step((0, 30, 15))
        result = env.snapshot()
        self.assertEqual(saved.time, 0)
        env.restore(saved)
        for _ in range(3):
            env.step((0, 30, 15))
        self.assertEqual(result, env.snapshot())
        with self.assertRaises(ValueError):
            env.tape.requests[0, 0] = 1.0

    def test_rival_prices_recompute_after_counterfactual_action(self):
        model = Model(Config(arrival_probability=0.0))
        env = AirlineEnv(model)
        env.reset(9)
        saved = env.snapshot()
        env.step((-30, -30, 0))
        low = env.observe(Information.DIRECT).quotes
        env.restore(saved)
        env.step((30, 30, 0))
        high = env.observe(Information.DIRECT).quotes
        self.assertNotEqual(low, high)
        self.assertEqual(env.state.rival().sold, (0, 0, 0))


class InformationTests(unittest.TestCase):
    def test_direct_pools_initial_states_linked_quote_separates_them(self):
        model = Model()
        low = WorldState(0, [(2, 2), (2, 1)])
        high = WorldState(0, [(2, 2), (2, 3)])
        self.assertEqual(
            model.observation(low, Information.DIRECT),
            model.observation(high, Information.DIRECT),
        )
        obs = model.observation(low, Information.LINKED)
        belief = Belief(model)
        belief.condition(obs)
        self.assertEqual(belief.mass, {RivalState((2, 1)): 1.0})
        self.assertIsNone(obs.privileged_rival)
        self.assertNotIn("seed", Observation.__dataclass_fields__)
        self.assertNotIn("tape", Observation.__dataclass_fields__)

    def test_filter_retains_true_state_for_all_information_arms(self):
        model = Model(Config(capacity_shock_probability=0.5))
        for seed in range(12):
            for mode in Information:
                env = AirlineEnv(model)
                env.reset(seed)
                before = env.observe(mode)
                belief = Belief(model)
                belief.condition(before)
                while before.time < model.config.horizon:
                    action = COEFFICIENT_GRID[
                        (seed + before.time) % len(COEFFICIENT_GRID)
                    ]
                    sale = env.step(action)
                    after = env.observe(mode)
                    belief.advance(before, action, sale, after)
                    self.assertAlmostEqual(sum(belief.mass.values()), 1)
                    self.assertGreater(belief.mass.get(env.state.rival(), 0), 0)
                    before = after

    def test_independent_quote_cancels_at_every_bayesian_update(self):
        model = Model(Config(capacity_shock_probability=1.0))
        modes = (Information.DIRECT, Information("unrelated"))
        for seed in range(5):
            env = AirlineEnv(model)
            env.reset(seed)
            observations = [env.observe(mode) for mode in modes]
            filters = [Belief(model), Belief(model)]
            for b, o in zip(filters, observations):
                b.condition(o)
            while env.state.time < model.config.horizon:
                sale = env.step((0, 30, 15))
                after = [env.observe(mode) for mode in modes]
                for b, old, new in zip(filters, observations, after):
                    b.advance(old, (0, 30, 15), sale, new)
                self.assertEqual(filters[0].key(), filters[1].key())
                observations = after

    def test_impossible_quote_raises_instead_of_reading_truth(self):
        from airline_market.legacy_toy.model import Quote

        env = AirlineEnv()
        env.reset(0)
        bad = replace(
            env.observe(Information.DIRECT), quotes=(Quote("OH", 9999, True),)
        )
        with self.assertRaisesRegex(ValueError, "zero model likelihood"):
            Belief(env.model).condition(bad)

    def test_one_step_profit_control_has_no_initial_feeder_information_value(self):
        model = Model()
        env = AirlineEnv(model)
        env.reset(0)
        planner = RolloutPlanner(model, (0, 0, 15), scenarios=8, horizon=1)
        decisions = []
        for mode in (Information.DIRECT, Information.LINKED):
            observation = env.observe(mode)
            belief = Belief(model)
            belief.condition(observation)
            decisions.append(planner.choose(observation, belief, seed=77))
        self.assertEqual(decisions[0], decisions[1])

    def test_public_action_cannot_depend_on_actual_unobserved_state(self):
        model = Model()
        planner = RolloutPlanner(model, (0, 0, 15), scenarios=2)
        decisions = []
        for hd in (1, 3):
            obs = model.observation(
                WorldState(0, [(2, 2), (2, hd)]), Information.DIRECT
            )
            b = Belief(model)
            b.condition(obs)
            decisions.append(planner.choose(obs, b, 77))
        self.assertEqual(decisions[0], decisions[1])


class EvaluationTests(unittest.TestCase):
    def test_paired_statistics_rejects_misaligned_seeds(self):
        with self.assertRaises(ValueError):
            paired_statistics(
                [{"seed": 1, "profit": 1}], [{"seed": 2, "profit": 1}], "profit"
            )

    def test_training_and_test_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            run_experiment(Config(), [1, 2], [2, 3], scenarios=1)

    def test_end_to_end_reproducible_outputs_and_placebo(self):
        arguments = dict(scenarios=1, trace=True, progress_every=0)
        first = run_experiment(Config(), [11, 12], [31, 32], **arguments)
        second = run_experiment(Config(), [11, 12], [31, 32], **arguments)
        self.assertEqual(first["episodes"], second["episodes"])
        self.assertEqual(first["training_grid"], second["training_grid"])
        self.assertTrue(first["placebo_identical"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_results(first, output, ["test"])
            parsed = json.loads((output / "results.json").read_text())
            self.assertEqual(
                parsed["contrasts"]["unrelated_minus_direct"]["profit"]["mean"], 0
            )
            self.assertIn("not an upper bound", (output / "report.md").read_text())
            self.assertEqual(
                len((output / "episodes.csv").read_text().splitlines()), 11
            )


if __name__ == "__main__":
    unittest.main()
