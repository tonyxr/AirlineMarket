"""Common-tape coefficient and departure-schedule selection."""

from dataclasses import dataclass
import numpy as np
from ..core.types import CoefficientAction
from ..vendor.ride_response.cem import CEMConfig, CEMOptimizer


@dataclass(frozen=True)
class FixedSearchResult:
    """
    Selected train-only coefficients and their common-tape comparison table (Eqs.
    26–32).
    """

    best_theta: tuple
    table: list
    diagnostics: dict | None = None


def episode_value(env, seed, policy):
    """
    Execute a supplied policy on one training tape and sum objective rewards (Eqs.
    26–32).
    """
    obs, _ = env.reset(seed)
    total = 0.0
    while obs.t < env.compiled.spec.horizon.n_periods:
        obs, r, _, _ = env.step(policy(obs))
        total += r
    return total


def search_fixed_grid(env_factory, grid, train_seeds, arm="direct"):
    """
    Evaluate the common coefficient grid on identical training cohorts (Eqs. 26–32).
    """
    seeds = tuple(train_seeds)
    if not seeds:
        raise ValueError("training seeds must be nonempty")
    env = env_factory()
    table = []
    for theta in grid:
        values = [
            episode_value(env, s, lambda o: CoefficientAction(tuple(theta)))
            for s in seeds
        ]
        table.append(
            {"coefficients": tuple(theta), "training_objective": float(np.mean(values))}
        )
    return FixedSearchResult(
        max(table, key=lambda r: r["training_objective"])["coefficients"], table
    )


def search_fixed_cem(env_factory, bounds, train_seeds, cem_config=None):
    """
    Optimize bounded fixed coefficients with common training random inputs (Eqs. 26–32).
    """
    bounds = np.asarray(bounds)
    env = env_factory()
    seeds = tuple(train_seeds)
    table = []

    def objective(x):
        """
        Score one candidate on the same training/calibration random inputs (Eqs. 26–32).
        """
        theta = np.clip(x, *bounds.T)
        value = float(np.mean([episode_value(env, s, lambda o: theta) for s in seeds]))
        table.append({"coefficients": tuple(theta), "training_objective": value})
        return value

    result = CEMOptimizer(len(bounds), cem_config).optimize(
        objective,
        initial_mean=bounds.mean(1),
        initial_std=np.maximum((bounds[:, 1] - bounds[:, 0]) / 3, 0.01),
    )
    return FixedSearchResult(
        tuple(np.clip(result["coordinates"], *bounds.T)), table, result
    )


def search_schedule(env_factory, bounds, train_seeds, blocks, cem_config=None):
    """
    Fit a piecewise-constant departure schedule on training cohorts only (Eqs. 26–32).
    """
    bounds = np.asarray(bounds)
    env = env_factory()
    T = env.compiled.spec.horizon.n_periods
    if [t for a, b in blocks for t in range(a, b)] != list(range(T)):
        raise ValueError("blocks must partition the horizon")
    table = []

    def schedule(x):
        """Reshape and bound the per-block coefficient candidate (Eqs. 26–32)."""
        return np.clip(np.asarray(x).reshape(len(blocks), len(bounds)), *bounds.T)

    def objective(x):
        """
        Score one candidate on the same training/calibration random inputs (Eqs. 26–32).
        """
        theta = schedule(x)
        value = float(
            np.mean(
                [
                    episode_value(
                        env,
                        s,
                        lambda o: theta[
                            next(k for k, (a, b) in enumerate(blocks) if a <= o.t < b)
                        ],
                    )
                    for s in train_seeds
                ]
            )
        )
        table.append({"schedule": theta.tolist(), "training_objective": value})
        return value

    result = CEMOptimizer(len(blocks) * len(bounds), cem_config).optimize(
        objective,
        initial_mean=np.tile(bounds.mean(1), len(blocks)),
        initial_std=np.tile(np.maximum(np.ptp(bounds, axis=1) / 3, 0.01), len(blocks)),
    )
    return {
        "blocks": blocks,
        "thetas": schedule(result["coordinates"]).tolist(),
        "table": table,
        "diagnostics": result,
    }
