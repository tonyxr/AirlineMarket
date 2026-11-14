"""Sampled population expansion with explicit MPC-generated history policies.

The finite matrix is only a meta-game. New common-action joint-type MPC
responses and independent paired deviations are required each iteration.
No sampled or fixed-tail search result is an exact/Nash certificate.
"""

from copy import deepcopy

import numpy as np

from airline_market.core.errors import KernelLimitError
from airline_market.core.rng import StreamRegistry
from airline_market.game.meta_solvers import solve_bimatrix
from airline_market.game.type_belief import HistoryController, HistoryPolicy
from airline_market.game.type_mpc import EpisodeTypeMPC, JointTypeFilter
from airline_market.sim.accounting import carrier_metrics
from airline_market.sim.engine import MarketEngine


class EpisodeHistoryMixture:
    """Native sampler only: one private deterministic history-policy type/episode."""

    exact_compatible = False
    policy_class = "episode_finite_history_mixture"

    def __init__(self, policies, probabilities):
        self.policies = tuple(policies)
        self.probabilities = np.asarray(probabilities, dtype=float)
        if (
            not policies
            or len(policies) != len(probabilities)
            or len({p.arm for p in policies}) != 1
            or not np.isfinite(self.probabilities).all()
            or np.any(self.probabilities < 0)
            or abs(self.probabilities.sum() - 1) > 1e-12
        ):
            raise ValueError("invalid episode history-policy mixture")
        self.arm_name = policies[0].arm

    def reset(self, ctx, rng):
        self.rng = rng
        self.choice = int(rng.choice(len(self.policies), p=self.probabilities))
        self.child = HistoryController(self.policies[self.choice])
        self.child.reset(ctx, rng)

    def act(self, obs):
        return self.child.act(obs)

    def observe_outcome(self, *args):
        pass

    def memory(self):
        return self.choice, self.child.memory()

    def snapshot(self):
        return (
            self.choice,
            self.child.snapshot(),
            deepcopy(self.rng.bit_generator.state),
        )

    def restore(self, state):
        choice, prefix, random = deepcopy(state)
        if choice != self.choice:
            raise ValueError("cannot change a persistent episode type during restore")
        self.child.restore(prefix)
        self.rng.bit_generator.state = random


def materialize_response(
    compiled,
    focal,
    arm,
    opponents,
    probabilities,
    config,
    *,
    episode_index=0,
    max_policy_nodes=10000,
    max_support=20000,
    progress=None,
):
    """Forward-expand all positive on-policy information histories; no Bellman max.

    Off-support histories use the declared tail action. That total extension is
    part of the exported/hashable strategy, not a hidden-state recovery rule.
    """
    if max_policy_nodes < 1:
        raise ValueError("positive materialized-policy node limit required")
    (rival,) = set(compiled.carrier_index) - {focal}
    belief = JointTypeFilter(
        compiled,
        focal,
        arm,
        [{rival: HistoryController(p)} for p in opponents],
        probabilities,
        max_support,
    )
    planner = EpisodeTypeMPC(belief, config)
    table = {}
    frontier = [node for p, node in belief.model.roots() if p > 0]
    while frontier:
        node = frontier.pop()
        if node.observation.t >= compiled.spec.horizon.n_periods or node.path in table:
            continue
        if len(table) >= max_policy_nodes:
            raise KernelLimitError("MPC materialized policy exceeds max_policy_nodes")
        if len(node.mass) > max_support:
            raise KernelLimitError("MPC materialization exceeds joint max_support")
        if progress:
            progress(
                {
                    "stage": "response_planning",
                    "completed_nodes": len(table),
                    "period": node.observation.t,
                    "posterior_support": len(node.mass),
                }
            )
        belief.restore(node)
        decision = planner.choose(episode_index)
        index = decision.diagnostics["selected_index"]
        table[node.path] = index
        if progress:
            progress(
                {
                    "stage": "response_policy_node",
                    "completed_nodes": len(table),
                    "period": node.observation.t,
                    "posterior_support": len(node.mass),
                    "policy_row": [node.path, index],
                }
            )
        frontier.extend(
            child for p, _, child in belief.model.successors(node, index) if p > 0
        )
    return HistoryPolicy(belief.model.grid, table, default=config.tail_index, arm=arm)


def _native(compiled, controllers, seed):
    engine = MarketEngine(compiled, controllers)
    engine.reset(seed)
    while not engine.terminal():
        engine.step()
    return {
        c.code: carrier_metrics(engine.world, compiled, i)
        for i, c in enumerate(compiled.spec.carriers)
    }


def _validation_seeds(start, count):
    seeds = tuple(range(start, start + count))
    if count < 2 or any(not 200000 <= s < 300000 for s in seeds):
        raise ValueError("population estimates require validation seeds only")
    return seeds


def profile_values(compiled, policies, seeds, cache, progress=None):
    """Native pure-profile payoffs cached by full policy hashes and physical seed."""
    carriers = tuple(c.code for c in compiled.spec.carriers)
    values = []
    for seed in seeds:
        if not 200000 <= seed < 300000:
            raise ValueError("payoff uses validation cohorts only")
        key = compiled.spec_hash, tuple(policies[c].hash() for c in carriers), seed
        if key not in cache:
            cache[key] = _native(
                compiled, {c: HistoryController(policies[c]) for c in carriers}, seed
            )
            if progress:
                progress(
                    {
                        "stage": "payoff_cohort",
                        "seed": seed,
                        "profile_hashes": list(key[1]),
                        "values": cache[key],
                    }
                )
        values.append(deepcopy(cache[key]))
    return values


def paired_deviation(
    compiled,
    focal,
    response,
    populations,
    mixtures,
    seeds,
    *,
    iteration=0,
    bootstrap=2000,
    progress=None,
):
    """Fresh candidate vs current meta-profile, same physical tape/opponent type."""
    carriers = tuple(c.code for c in compiled.spec.carriers)
    (rival,) = set(carriers) - {focal}
    rows = []
    for seed in seeds:
        if not 200000 <= seed < 300000:
            raise ValueError("deviation uses validation cohorts only")
        sampled = []
        types = []
        for candidate in [False, True]:
            rival_controller = EpisodeHistoryMixture(
                populations[rival], mixtures[rival]
            )
            own = (
                HistoryController(response)
                if candidate
                else EpisodeHistoryMixture(populations[focal], mixtures[focal])
            )
            sampled.append(
                _native(compiled, {focal: own, rival: rival_controller}, seed)[focal]
            )
            types.append(rival_controller.choice)
        if types[0] != types[1]:
            raise ValueError(
                "paired deviation did not preserve the opponent episode type"
            )
        rows.append(
            {
                "seed": seed,
                "opponent_type": types[0],
                "baseline": sampled[0],
                "response": sampled[1],
                "objective_difference": sampled[1]["objective"]
                - sampled[0]["objective"],
                "profit_difference": sampled[1]["profit"] - sampled[0]["profit"],
            }
        )
        if progress:
            progress(
                {
                    "stage": "deviation_cohort",
                    "completed_cohorts": len(rows),
                    "row": rows[-1],
                }
            )
    difference = np.array([r["objective_difference"] for r in rows])
    if len(rows) < 2 or bootstrap < 2:
        raise ValueError("paired deviation needs multiple cohorts/bootstrap draws")
    rng = StreamRegistry(773, "analysis/F7/materialized-MPC-deviation-v1").generator(
        iteration, carriers.index(focal)
    )
    means = difference[rng.integers(len(rows), size=(bootstrap, len(rows)))].mean(
        axis=1
    )
    return {
        "mean": float(difference.mean()),
        "se": float(difference.std(ddof=1) / np.sqrt(len(rows))),
        "bootstrap95": np.quantile(means, [0.025, 0.975]).tolist(),
        "n": len(rows),
        "sample_variance_zero": bool(np.all(difference == difference[0])),
        "rows": rows,
        "scope": "Conditional paired sampling interval for one fixed candidate and iteration, not simultaneous/sequential coverage or a certified exploitability bound.",
    }


def sampled_mpc_psro(
    compiled,
    populations,
    config,
    *,
    max_iters=4,
    tolerance=1e-6,
    payoff_seed_start=200512,
    n_payoff=256,
    deviation_seed_start=200768,
    n_deviation=256,
    max_policy_nodes=10000,
    max_support=20000,
    progress=None,
):
    """Actual population expansion; matrix residuals never substitute for responses."""
    carriers = tuple(c.code for c in compiled.spec.carriers)
    if (
        len(carriers) != 2
        or set(populations) != set(carriers)
        or max_iters < 1
        or n_deviation < 2
        or tolerance < 0
    ):
        raise ValueError("two complete populations/positive iterations required")
    for carrier in carriers:
        grid = next(p for p in compiled.spec.pricing if p.carrier == carrier)
        if (
            not grid.controllable_features
            or len(grid.theta_grid or ()) < 2
            or not populations[carrier]
            or any(p.grid != grid.theta_grid for p in populations[carrier])
            or len({p.arm for p in populations[carrier]}) != 1
        ):
            raise ValueError(
                "populations require nondegenerate matching grids and common arms"
            )
    payoff_seeds = _validation_seeds(payoff_seed_start, n_payoff)
    all_deviation = _validation_seeds(deviation_seed_start, n_deviation * max_iters)
    if set(payoff_seeds) & set(all_deviation):
        raise ValueError("payoff selection and fresh deviations must be disjoint")
    populations = {c: list(populations[c]) for c in carriers}
    cache, history = {}, []
    status = "iteration_limit"
    for iteration in range(max_iters):
        a, b = carriers
        matrices = [
            np.empty((len(populations[a]), len(populations[b]))) for _ in carriers
        ]
        for i, left in enumerate(populations[a]):
            for j, right in enumerate(populations[b]):
                values = profile_values(
                    compiled, {a: left, b: right}, payoff_seeds, cache, progress
                )
                for k, carrier in enumerate(carriers):
                    matrices[k][i, j] = np.mean(
                        [r[carrier]["objective"] for r in values]
                    )
                if progress:
                    progress(
                        {
                            "stage": "payoff_entry",
                            "iteration": iteration,
                            "row": i,
                            "column": j,
                            "cached_profile_cohorts": len(cache),
                        }
                    )
        meta = solve_bimatrix(*matrices)
        mixtures = {c: meta["strategies"][k] for k, c in enumerate(carriers)}
        row = {
            "iteration": iteration,
            "population_sizes": [len(populations[c]) for c in carriers],
            "mixtures": {c: w.tolist() for c, w in mixtures.items()},
            "payoff_matrices": [m.tolist() for m in matrices],
            "meta_payoffs": list(meta["payoffs"]),
            "meta_scope": "Finite-population sampled payoff matrix only; not the stopping certificate.",
            "responses": {},
        }
        responses = {}
        for k, focal in enumerate(carriers):
            rival = carriers[1 - k]
            active = [
                (p, float(w))
                for p, w in zip(populations[rival], mixtures[rival])
                if w > 0
            ]

            def event(update, iteration=iteration, focal=focal):
                if progress:
                    progress({"iteration": iteration, "focal": focal, **update})

            response = materialize_response(
                compiled,
                focal,
                populations[focal][0].arm,
                [p for p, _ in active],
                [w for _, w in active],
                config,
                episode_index=iteration * 2 + k,
                max_policy_nodes=max_policy_nodes,
                max_support=max_support,
                progress=event,
            )
            responses[focal] = response
            if progress:
                progress(
                    {
                        "stage": "response_complete",
                        "iteration": iteration,
                        "focal": focal,
                        "policy": response.to_dict(),
                        "policy_hash": response.hash(),
                    }
                )
            seeds = all_deviation[
                iteration * n_deviation : (iteration + 1) * n_deviation
            ]
            evaluation = paired_deviation(
                compiled,
                focal,
                response,
                populations,
                mixtures,
                seeds,
                iteration=iteration,
                progress=event,
            )
            row["responses"][focal] = {
                "policy_hash": response.hash(),
                "materialized_nodes": len(response.table),
                "opponent_policy_hashes": [p.hash() for p, _ in active],
                "evaluation": evaluation,
            }
        history.append(row)
        if progress:
            progress({"stage": "iteration_complete", **row})
        # Conditional bootstrap intervals remain approximate; even target-met is NOT certified.
        if all(
            row["responses"][c]["evaluation"]["bootstrap95"][1] <= tolerance
            for c in carriers
        ):
            status = "search_target_met_with_sampling_caveat"
            break
        added = False
        for c in carriers:
            if responses[c].hash() not in {p.hash() for p in populations[c]}:
                populations[c].append(responses[c])
                added = True
        if not added:
            status = "stalled_existing_responses_with_uncertainty"
            break
    return {
        "status": status,
        "history": history,
        "populations": populations,
        "cached_profile_cohorts": len(cache),
        "certified": False,
        "exact_exploitability_lower_bound": None,
        "final_game_budget": None,
        "scope": "MPC-generated total finite-history policies with declared tail off-support; fresh best-found paired deviations. Approximate search and sampling uncertainty, never exact/game-wide/Nash certification or unrestricted online-MPC equivalence.",
    }
